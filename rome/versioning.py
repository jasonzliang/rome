import datetime
import hashlib
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from contextlib import contextmanager

import psutil
import portalocker

from .config import META_DIR_EXT
from .logger import get_logger


class VersionManager:
    """Manages file versioning and analysis for the agent"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = get_logger()
        self.active_files: Set[str] = set()  # Track active files for this manager's agent

    def _infer_main_file_from_test(self, test_file_path: str) -> Optional[str]:
        """Infer the main file path from a test file path using _test.py naming convention."""
        test_dir = os.path.dirname(test_file_path)
        test_filename = os.path.basename(test_file_path)

        if test_filename.endswith('_test.py'):
            main_filename = test_filename[:-8] + '.py'
            main_file_path = os.path.join(test_dir, main_filename)

            if os.path.exists(main_file_path):
                self.logger.debug(f"Inferred main file {main_file_path} from test file {test_file_path}")
                return main_file_path

        self.logger.warning(f"Could not infer main file from test file: {test_file_path}")
        return None

    def _get_meta_dir(self, file_path: str) -> str:
        """Get the meta directory path for a file."""
        meta_dir = f"{file_path}.{META_DIR_EXT}"
        os.makedirs(meta_dir, exist_ok=True)
        return meta_dir

    def _get_test_meta_dir(self, test_file_path: str, main_file_path: str) -> str:
        """Get the meta directory path for a test file (nested within main file's meta directory)."""
        main_meta_dir = self._get_meta_dir(main_file_path)
        test_filename = os.path.basename(test_file_path)
        test_meta_dir = os.path.join(main_meta_dir, f"{test_filename}.{META_DIR_EXT}")
        os.makedirs(test_meta_dir, exist_ok=True)
        return test_meta_dir

    @contextmanager
    def _file_lock(self, lock_file_path: str):
        """Context manager for file locking."""
        with open(lock_file_path, 'w') as lock_file:
            portalocker.lock(lock_file, portalocker.LOCK_EX)
            yield

    def _load_json_safely(self, file_path: str, default: Dict = None) -> Dict:
        """Load JSON file safely, returning default if file doesn't exist or is corrupted."""
        if not os.path.exists(file_path):
            return default or {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            self.logger.warning(f"Corrupted JSON file {file_path}: {e}")
            return default or {}

    def _save_json(self, file_path: str, data: Dict) -> None:
        """Save data to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def _get_content_hash(self, content: str) -> str:
        """Generate SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.datetime.now().isoformat()

    def _find_existing_version_by_hash(self, index: Dict, content_hash: str) -> Optional[int]:
        """Find existing version with the same content hash."""
        for version in index.get('versions', []):
            if version.get('hash') == content_hash:
                return version.get('version')
        return None

    def _get_next_version_number(self, index: Dict) -> int:
        """Get the next version number from index."""
        versions = index.get('versions', [])
        return max((v.get('version', 0) for v in versions), default=0) + 1

    def _create_version_metadata(self, file_path: str, content_hash: str, version_number: int,
                               changes: Optional[List[Dict[str, str]]], explanation: Optional[str],
                               main_file_path: Optional[str] = None) -> Dict:
        """Create metadata entry for a version."""
        metadata = {
            'version': version_number,
            'file_path': file_path,
            'timestamp': self._get_timestamp(),
            'hash': content_hash,
            'changes': changes or [],
            'explanation': explanation or "No explanation provided"
        }
        if main_file_path:
            metadata['main_file_path'] = main_file_path
        return metadata

    def _save_version_internal(self, file_path: str, content: str, versions_dir: str,
                              changes: Optional[List[Dict[str, str]]] = None,
                              explanation: Optional[str] = None,
                              main_file_path: Optional[str] = None) -> int:
        """Internal method to save a versioned snapshot with incremental version numbers."""
        assert os.path.exists(versions_dir), f"File meta dir {versions_dir} does not exist"

        content_hash = self._get_content_hash(content)
        self.logger.debug(f"File content hash: {content_hash}")

        index_file_path = os.path.join(versions_dir, "index.json")
        lock_file_path = os.path.join(versions_dir, "index.lock")

        with self._file_lock(lock_file_path):
            index = self._load_json_safely(index_file_path, {'versions': []})
            if main_file_path and 'main_file_path' not in index:
                index['main_file_path'] = main_file_path

            # Check for existing version with same hash
            existing_version = self._find_existing_version_by_hash(index, content_hash)
            if existing_version:
                self.logger.debug(f"Content already exists in version {existing_version}. Skipping save.")
                return existing_version

            # Create new version
            version_number = self._get_next_version_number(index)

            # Save version file
            file_name = os.path.basename(file_path)
            file_base, file_ext = os.path.splitext(file_name)
            version_file_name = f"{file_base}_v{version_number}{file_ext}"
            version_file_path = os.path.join(versions_dir, version_file_name)

            with open(version_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Update index
            metadata = self._create_version_metadata(
                file_path, content_hash, version_number, changes, explanation, main_file_path
            )
            index['versions'].append(metadata)
            index['versions'].sort(key=lambda x: x.get('version', 0))

            self._save_json(index_file_path, index)

            num_changes = len(changes) if changes else 0
            self.logger.debug(f"Successfully saved version {version_number} with {num_changes} documented changes")

            return version_number

    def save_original(self, file_path: str, content: str) -> int:
        """Save the original unedited file into the meta directory."""
        self.logger.info(f"Saving original version for file: {file_path}")

        if os.path.exists(self._get_meta_dir(file_path)):
            return 1

        return self.save_version(
            file_path=file_path,
            content=content,
            changes=[{"type": "initial", "description": "Original file content"}],
            explanation="Initial version: Original unedited file"
        )

    def save_version(self, file_path: str, content: str,
                     changes: Optional[List[Dict[str, str]]] = None,
                     explanation: Optional[str] = None) -> int:
        """Save a versioned snapshot of a main file with incremental version numbers."""
        self.logger.info(f"Saving version for file: {file_path}")
        versions_dir = self._get_meta_dir(file_path)
        return self._save_version_internal(file_path, content, versions_dir, changes, explanation)

    def save_test_version(self, test_file_path: str, content: str,
                         changes: Optional[List[Dict[str, str]]] = None,
                         explanation: Optional[str] = None,
                         main_file_path: Optional[str] = None) -> int:
        """Save a versioned snapshot of a test file in the main file's meta directory."""
        self.logger.info(f"Saving test version for file: {test_file_path}")

        if main_file_path is None:
            main_file_path = self._infer_main_file_from_test(test_file_path)
            if main_file_path is None:
                raise ValueError(f"Could not infer main file path for test file {test_file_path}. Please provide main_file_path explicitly.")

        if not os.path.exists(main_file_path):
            raise ValueError(f"Main file does not exist: {main_file_path}")

        test_meta_dir = self._get_test_meta_dir(test_file_path, main_file_path)
        return self._save_version_internal(
            test_file_path, content, test_meta_dir, changes, explanation, main_file_path
        )

    def save_analysis(self, file_path: str, analysis: str,
                      test_path: Optional[str] = None,
                      exit_code: Optional[int] = None,
                      output: Optional[str] = None) -> str:
        """Save the latest analysis results in the same versioning directory structure."""
        self.logger.info(f"Saving analysis for file: {file_path}")

        versions_dir = self._get_meta_dir(file_path)
        analysis_file_path = os.path.join(versions_dir, "code_analysis.json")

        analysis_data = {
            'timestamp': self._get_timestamp(),
            'file_path': file_path,
            'test_path': test_path,
            'exit_code': exit_code,
            'output': output,
            'analysis': analysis
        }

        self._save_json(analysis_file_path, analysis_data)
        self.logger.info(f"Saved latest analysis to {analysis_file_path}")
        return analysis_file_path

    def _load_analysis(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load the latest analysis results for a given file."""
        versions_dir = self._get_meta_dir(file_path)
        if not os.path.exists(versions_dir):
            self.logger.debug(f"No metadata directory found for {file_path}")
            return None

        analysis_file_path = os.path.join(versions_dir, "code_analysis.json")
        analysis_data = self._load_json_safely(analysis_file_path)

        if analysis_data:
            self.logger.info(f"Loaded analysis for {file_path}")
            return analysis_data
        else:
            self.logger.debug(f"No analysis file found for {file_path}")
            return None

    def _format_analysis_context(self, analysis_data: Dict[str, Any]) -> str:
        """Format analysis data into a context string for prompts."""
        if not analysis_data:
            return ""

        return f"""Code test output:
```
{analysis_data.get('output', 'No output available')}
```

Code analysis:
{analysis_data.get('analysis', 'No analysis available')}

IMPORTANT: Please take this analysis into account when improving the code or tests.
"""

    def get_analysis_prompt(self, file_path: str) -> Optional[str]:
        """Get formatted analysis context specifically for code editing prompts."""
        analysis_data = self._load_analysis(file_path)
        return self._format_analysis_context(analysis_data) if analysis_data else None

    def create_analysis(self, agent, original_file_content: str, test_file_content: str,
                       output: str, exit_code: int) -> str:
        """Generate an analysis of test execution results using LLM."""
        prompt = f"""Analyze the following test execution results and provide comprehensive feedback:

Code file content:
```python
{original_file_content}
```

Test file content:
```python
{test_file_content}
```

Execution output:
```
{output}
```

Exit code: {exit_code}

Please provide an analysis covering:
1. Overall test execution status (passed/failed)
2. Specific test failures and their root causes
3. Any errors, exceptions, or warnings found
4. Code quality issues revealed by the tests
5. Suggestions for fixing identified problems
6. Recommendations for improving test coverage

Your analysis:
"""

        try:
            return agent.chat_completion(prompt=prompt, system_message=agent.role)
        except Exception as e:
            self.logger.error(f"Failed to generate LLM analysis: {str(e)}")
            return f"Failed to generate analysis: {str(e)}"

    def _get_pid_from_agent_id(self, agent_id: str) -> Optional[int]:
        """Extract PID from agent ID string."""
        try:
            parts = agent_id.split('_')
            if len(parts) >= 3:
                return int(parts[-1])
        except (ValueError, IndexError):
            pass
        return None

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is still running, is a Python process, and belongs to the same user."""
        try:
            process = psutil.Process(pid)
            current_user = os.getlogin() if hasattr(os, 'getlogin') else os.environ.get('USER', os.environ.get('USERNAME'))

            return (process.username() == current_user and
                   ('python' in process.name().lower() or
                    any('python' in arg.lower() for arg in process.cmdline()[:2])))

        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, AttributeError):
            return False
        except Exception as e:
            self.logger.warning(f"Error checking process {pid}: {e}")
            return False

    def _handle_stale_file(self, file_path: str, agent_id: str) -> bool:
        """Check if agent process is running, remove stale file if not."""
        if not agent_id:
            os.remove(file_path)
            return False

        pid = self._get_pid_from_agent_id(agent_id)
        if pid is None or not self._is_process_running(pid):
            os.remove(file_path)
            return False

        return True

    def get_active_files(self) -> Set[str]:
        """Get all files currently active for this agent"""
        return self.active_files.copy()

    def cleanup_active_files(self, agent) -> None:
        """Clear all active files (useful for shutdown)"""
        for file_path in self.active_files.copy():
            try:
                self.unflag_active(agent, file_path)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup active file {file_path}: {e}")
        self.active_files.clear()

    def _add_active_file(self, file_path: str) -> None:
        """Add a file to the active files set"""
        self.active_files.add(os.path.abspath(file_path))

    def _remove_active_file(self, file_path: str) -> bool:
        """Remove a file from the active files set"""
        abs_path = os.path.abspath(file_path)
        if abs_path in self.active_files:
            self.active_files.discard(abs_path)
            return True
        return False

    def _has_active_files(self) -> bool:
        """Check if agent has any active files"""
        return len(self.active_files) > 0

    def get_active_files(self) -> Set[str]:
        """Get all files currently active for this agent"""
        return self.active_files.copy()

    def cleanup_active_files(self, agent) -> None:
        """Clear all active files (useful for shutdown)"""
        for file_path in self.active_files.copy():
            try:
                self.unflag_active(agent, file_path)
            except Exception as e:
                self.logger.debug(f"Failed to cleanup active file {file_path}: {e}")
        self.active_files.clear()

    def check_active(self, file_path: str, ignore_self: bool = True) -> bool:
        """Check if there is an active agent working on the given file."""
        meta_dir = self._get_meta_dir(file_path)
        active_file_path = os.path.join(meta_dir, "active.json")
        lock_file_path = os.path.join(meta_dir, "active.lock")

        if not os.path.exists(active_file_path):
            return False

        with self._file_lock(lock_file_path):
            if not os.path.exists(active_file_path):
                return False

            active_data = self._load_json_safely(active_file_path)
            if not active_data:
                os.remove(active_file_path)
                return False

            agent_id = active_data.get('agent_id')

            # Check if process is still alive
            if not self._handle_stale_file(active_file_path, agent_id):
                # Process is dead, file was cleaned up
                return False

            # If ignore_self is True and this is our own process, return False
            if ignore_self:
                pid = self._get_pid_from_agent_id(agent_id)
                current_pid = os.getpid()
                if pid == current_pid:
                    return False

            return True

    def flag_active(self, agent, file_path: str, allow_multiple: bool = False) -> bool:
        """Flag a file as being actively worked on by the given agent."""

        # Check if agent already has active files (unless multiple allowed)
        if not allow_multiple and self._has_active_files():
            existing_files = list(self.get_active_files())
            raise RuntimeError(
                f"Agent {agent.get_id()} already has active file(s): {existing_files}. "
                f"Only one file can be active per agent."
            )

        meta_dir = self._get_meta_dir(file_path)
        active_file_path = os.path.join(meta_dir, "active.json")
        lock_file_path = os.path.join(meta_dir, "active.lock")

        agent_id = agent.get_id()

        with self._file_lock(lock_file_path):
            if os.path.exists(active_file_path):
                active_data = self._load_json_safely(active_file_path)
                existing_agent_id = active_data.get('agent_id')

                if existing_agent_id:
                    pid = self._get_pid_from_agent_id(existing_agent_id)
                    current_pid = os.getpid()

                    if pid and self._is_process_running(pid):
                        if pid != current_pid:
                            raise RuntimeError(
                                f"File {file_path} is already being worked on by agent {existing_agent_id} "
                                f"(PID {pid}). Current process PID is {current_pid}."
                            )
                        elif existing_agent_id != agent_id:
                            raise RuntimeError(
                                f"File {file_path} is already flagged by a different agent {existing_agent_id} "
                                f"in the same process. Current agent is {agent_id}."
                            )

            # Create/update the active file
            active_data = {
                'agent_id': agent_id,
                'timestamp': self._get_timestamp(),
                'file_path': file_path
            }

            self._save_json(active_file_path, active_data)

            # Add to version manager's active files set
            self._add_active_file(file_path)

            self.logger.debug(f"Flagged {file_path} as active by agent {agent_id}")
            return True

    def unflag_active(self, agent, file_path: str) -> bool:
        """Remove the active flag for a file if it was set by the given agent."""
        meta_dir = self._get_meta_dir(file_path)
        active_file_path = os.path.join(meta_dir, "active.json")
        lock_file_path = os.path.join(meta_dir, "active.lock")

        agent_id = agent.get_id()

        if not os.path.exists(active_file_path):
            # Clean up version manager tracking just in case
            self._remove_active_file(file_path)
            self.logger.debug(f"No active file to remove for {file_path}")
            return False

        with self._file_lock(lock_file_path):
            if not os.path.exists(active_file_path):
                self._remove_active_file(file_path)
                self.logger.debug(f"No active file to remove for {file_path}")
                return False

            active_data = self._load_json_safely(active_file_path)
            if not active_data or active_data.get('agent_id') != agent_id:
                self.logger.debug(f"Cannot unflag {file_path} - not flagged by agent {agent_id}")
                return False

            # Remove file and update version manager tracking
            os.remove(active_file_path)
            self._remove_active_file(file_path)
            self.logger.debug(f"Unflagged {file_path} from agent {agent_id}")
            return True

    def check_finished(self, agent, file_path: str) -> bool:
        """Check if a file has been marked as finished."""
        meta_dir = self._get_meta_dir(file_path)
        finished_file_path = os.path.join(meta_dir, "finished.json")

        finished_data = self._load_json_safely(finished_file_path)
        if not finished_data:
            return False

        agents = finished_data.get('agents', [])
        return any(agent_entry.get('agent_id') == agent.get_id() for agent_entry in agents)

    def flag_finished(self, agent, file_path: str) -> None:
        """Flag a file as finished being worked on by the given agent."""
        meta_dir = self._get_meta_dir(file_path)
        finished_file_path = os.path.join(meta_dir, "finished.json")
        lock_file_path = os.path.join(meta_dir, "finished.lock")

        with self._file_lock(lock_file_path):
            finished_data = self._load_json_safely(finished_file_path, {'agents': []})

            # Check if agent already marked as finished (prevent duplicates)
            for agent_entry in finished_data['agents']:
                if agent_entry.get('agent_id') == agent.get_id():
                    self.logger.debug(f"Agent {agent.get_id()} already marked {file_path} as finished")
                    return

            finished_data['agents'].append({
                'agent_id': agent.get_id(),
                'timestamp': self._get_timestamp(),
                'file_path': file_path
            })

            self._save_json(finished_file_path, finished_data)
            self.logger.debug(f"Flagged {file_path} as finished by agent {agent.get_id()}")

    def shutdown(self):
        """Clean up resources before termination"""
        self.logger.info("Cleaning up agent resources")
        self.version_manager.cleanup_active_files(self)  # Clean up active files
        if self.agent_api:
            self.agent_api.shutdown()

