import datetime
import hashlib
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from contextlib import contextmanager

import psutil
import portalocker

from .config import META_DIR_EXT, set_attributes_from_config
from .logger import get_logger
from .parsing import hash_string


class VersionManager:
    """Manages code/test file activity, ownership, and versioning for the agent"""

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
        return hash_string(content)

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

            # FIX: Ensure we use the original file path for naming, not a meta directory path
            # Strip any meta directory extension from the file_path to get the original file name
            original_file_path = file_path
            meta_ext_suffix = f".{META_DIR_EXT}"
            if original_file_path.endswith(meta_ext_suffix):
                original_file_path = original_file_path[:-len(meta_ext_suffix)]
                self.logger.warning(f"Stripped meta extension from file_path: {file_path} -> {original_file_path}")

            # Save version file using the clean original file name
            file_name = os.path.basename(original_file_path)
            file_base, file_ext = os.path.splitext(file_name)
            version_file_name = f"{file_base}_v{version_number}{file_ext}"
            version_file_path = os.path.join(versions_dir, version_file_name)

            with open(version_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Update index - use the original file_path parameter for metadata consistency
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

    def _get_active_files(self) -> Set[str]:
        """Get all files currently active for this agent"""
        return self.active_files.copy()

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

    def flag_active(self, agent, file_path: str) -> bool:
        """Flag a file as being actively worked on by the given agent."""

        # Check if agent already has active files (unless multiple allowed)
        if self._has_active_files():
            existing_files = list(self._get_active_files())
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

    def _validate_index_json(self, index_path: str, main_file_path: str) -> None:
        """Validate the structure and content of index.json file."""
        index_data = self._load_json_safely(index_path)
        if not index_data:
            raise ValueError(f"Index JSON file is empty or corrupted: {index_path}")

        # Check required structure
        if 'versions' not in index_data:
            raise ValueError(f"Index JSON missing 'versions' field: {index_path}")

        if not isinstance(index_data['versions'], list):
            raise ValueError(f"Index JSON 'versions' field must be a list: {index_path}")

        # Validate each version entry
        for i, version in enumerate(index_data['versions']):
            if not isinstance(version, dict):
                raise ValueError(f"Index JSON version {i} must be a dict: {index_path}")

            required_version_fields = ['version', 'file_path', 'timestamp', 'hash', 'changes', 'explanation']
            for field in required_version_fields:
                if field not in version:
                    raise ValueError(f"Index JSON version {i} missing field '{field}': {index_path}")

            # Validate version number
            if not isinstance(version['version'], int) or version['version'] <= 0:
                raise ValueError(f"Index JSON version {i} has invalid version number: {version['version']}")

            # Validate file path
            if os.path.abspath(version['file_path']) != os.path.abspath(main_file_path):
                raise ValueError(
                    f"Index JSON version {i} file_path mismatch. Expected: {main_file_path}, Found: {version['file_path']}"
                )

            # Validate timestamp
            try:
                datetime.datetime.fromisoformat(version['timestamp'])
            except ValueError as e:
                raise ValueError(f"Index JSON version {i} invalid timestamp: {version['timestamp']} - {e}")

            # Validate hash format (should be SHA256)
            if not isinstance(version['hash'], str) or len(version['hash']) != 64:
                raise ValueError(f"Index JSON version {i} invalid hash format: {version['hash']}")

            # Validate changes is a list
            if not isinstance(version['changes'], list):
                raise ValueError(f"Index JSON version {i} 'changes' must be a list: {index_path}")

            # Validate explanation is a string
            if not isinstance(version['explanation'], str):
                raise ValueError(f"Index JSON version {i} 'explanation' must be a string: {index_path}")

    def _validate_analysis_json(self, analysis_path: str, main_file_path: str) -> None:
        """Validate the structure and content of code_analysis.json file."""
        analysis_data = self._load_json_safely(analysis_path)
        if not analysis_data:
            raise ValueError(f"Analysis JSON file is empty or corrupted: {analysis_path}")

        required_analysis_fields = ['timestamp', 'file_path', 'analysis']
        for field in required_analysis_fields:
            if field not in analysis_data:
                raise ValueError(f"Analysis JSON missing required field '{field}': {analysis_path}")

        # Validate timestamp
        try:
            datetime.datetime.fromisoformat(analysis_data['timestamp'])
        except ValueError as e:
            raise ValueError(f"Analysis JSON invalid timestamp: {analysis_data['timestamp']} - {e}")

        # Validate file path
        if os.path.abspath(analysis_data['file_path']) != os.path.abspath(main_file_path):
            raise ValueError(
                f"Analysis JSON file_path mismatch. Expected: {main_file_path}, Found: {analysis_data['file_path']}"
            )

        # Validate optional fields if present
        if 'exit_code' in analysis_data and analysis_data['exit_code'] is not None:
            if not isinstance(analysis_data['exit_code'], int):
                raise ValueError(f"Analysis JSON 'exit_code' must be an integer: {analysis_data['exit_code']}")

        if 'output' in analysis_data and analysis_data['output'] is not None:
            if not isinstance(analysis_data['output'], str):
                raise ValueError(f"Analysis JSON 'output' must be a string")

        if 'test_path' in analysis_data and analysis_data['test_path'] is not None:
            if not isinstance(analysis_data['test_path'], str):
                raise ValueError(f"Analysis JSON 'test_path' must be a string")

    def _validate_finished_json(self, finished_path: str) -> None:
        """Validate the structure and content of finished.json file."""
        finished_data = self._load_json_safely(finished_path)
        if not finished_data:
            raise ValueError(f"Finished JSON file is empty or corrupted: {finished_path}")

        if 'agents' not in finished_data:
            raise ValueError(f"Finished JSON missing 'agents' field: {finished_path}")

        if not isinstance(finished_data['agents'], list):
            raise ValueError(f"Finished JSON 'agents' field must be a list: {finished_path}")

        # Validate each agent entry
        for i, agent_entry in enumerate(finished_data['agents']):
            if not isinstance(agent_entry, dict):
                raise ValueError(f"Finished JSON agent {i} must be a dict: {finished_path}")

            required_agent_fields = ['agent_id', 'timestamp', 'file_path']
            for field in required_agent_fields:
                if field not in agent_entry:
                    raise ValueError(f"Finished JSON agent {i} missing field '{field}': {finished_path}")

            # Validate timestamp
            try:
                datetime.datetime.fromisoformat(agent_entry['timestamp'])
            except ValueError as e:
                raise ValueError(f"Finished JSON agent {i} invalid timestamp: {agent_entry['timestamp']} - {e}")

    def _validate_version_files(self, meta_dir: str, index_path: str) -> None:
        """Validate that all version files referenced in index.json actually exist and have correct content."""
        index_data = self._load_json_safely(index_path)
        if not index_data or 'versions' not in index_data:
            return

        for version in index_data['versions']:
            version_num = version['version']
            file_path = version['file_path']
            expected_hash = version['hash']

            # Construct expected version file name
            file_name = os.path.basename(file_path)
            file_base, file_ext = os.path.splitext(file_name)
            version_file_name = f"{file_base}_v{version_num}{file_ext}"
            version_file_path = os.path.join(meta_dir, version_file_name)

            # Check file exists
            if not os.path.exists(version_file_path):
                raise FileNotFoundError(f"Version file referenced in index does not exist: {version_file_path}")

            # Validate content hash
            try:
                with open(version_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                actual_hash = self._get_content_hash(content)

                if actual_hash != expected_hash:
                    raise ValueError(
                        f"Version file {version_file_path} content hash mismatch. "
                        f"Expected: {expected_hash}, Actual: {actual_hash}"
                    )
            except Exception as e:
                raise ValueError(f"Failed to validate version file {version_file_path}: {e}")

    def _validate_test_meta_dirs(self, main_meta_dir: str, main_file_path: str) -> None:
        """Validate test meta directories within the main file's meta directory."""
        if not os.path.exists(main_meta_dir):
            return

        # Look for test meta directories (directories ending with .meta_ext)
        for item in os.listdir(main_meta_dir):
            item_path = os.path.join(main_meta_dir, item)
            if os.path.isdir(item_path) and item.endswith(f".{META_DIR_EXT}"):
                # This is a test meta directory
                test_index_path = os.path.join(item_path, "index.json")
                if os.path.exists(test_index_path):
                    # Extract test file name from directory name
                    test_file_name = item[:-len(f".{META_DIR_EXT}")]
                    test_file_dir = os.path.dirname(main_file_path)
                    test_file_path = os.path.join(test_file_dir, test_file_name)

                    # Validate test index
                    self._validate_test_index_json(test_index_path, test_file_path, main_file_path)

                    # Validate test version files
                    self._validate_version_files(item_path, test_index_path)

    def _validate_test_index_json(self, test_index_path: str, test_file_path: str, main_file_path: str) -> None:
        """Validate the structure and content of a test file's index.json."""
        index_data = self._load_json_safely(test_index_path)
        if not index_data:
            raise ValueError(f"Test index JSON file is empty or corrupted: {test_index_path}")

        # Check that main_file_path is recorded
        if 'main_file_path' not in index_data:
            raise ValueError(f"Test index JSON missing 'main_file_path' field: {test_index_path}")

        if os.path.abspath(index_data['main_file_path']) != os.path.abspath(main_file_path):
            raise ValueError(
                f"Test index JSON main_file_path mismatch. Expected: {main_file_path}, "
                f"Found: {index_data['main_file_path']}"
            )

        # Validate versions structure (same as main index validation)
        if 'versions' not in index_data:
            raise ValueError(f"Test index JSON missing 'versions' field: {test_index_path}")

        if not isinstance(index_data['versions'], list):
            raise ValueError(f"Test index JSON 'versions' field must be a list: {test_index_path}")

        # Validate each version entry
        for i, version in enumerate(index_data['versions']):
            if not isinstance(version, dict):
                raise ValueError(f"Test index JSON version {i} must be a dict: {test_index_path}")

            required_fields = ['version', 'file_path', 'timestamp', 'hash', 'changes', 'explanation', 'main_file_path']
            for field in required_fields:
                if field not in version:
                    raise ValueError(f"Test index JSON version {i} missing field '{field}': {test_index_path}")

            # Validate file path matches expected test file
            if os.path.abspath(version['file_path']) != os.path.abspath(test_file_path):
                raise ValueError(
                    f"Test index JSON version {i} file_path mismatch. Expected: {test_file_path}, "
                    f"Found: {version['file_path']}"
                )

            # Validate main file path
            if os.path.abspath(version['main_file_path']) != os.path.abspath(main_file_path):
                raise ValueError(
                    f"Test index JSON version {i} main_file_path mismatch. Expected: {main_file_path}, "
                    f"Found: {version['main_file_path']}"
                )

            # Validate other fields (same as main validation)
            if not isinstance(version['version'], int) or version['version'] <= 0:
                raise ValueError(f"Test index JSON version {i} has invalid version number: {version['version']}")

            try:
                datetime.datetime.fromisoformat(version['timestamp'])
            except ValueError as e:
                raise ValueError(f"Test index JSON version {i} invalid timestamp: {version['timestamp']} - {e}")

            if not isinstance(version['hash'], str) or len(version['hash']) != 64:
                raise ValueError(f"Test index JSON version {i} invalid hash format: {version['hash']}")

            if not isinstance(version['changes'], list):
                raise ValueError(f"Test index JSON version {i} 'changes' must be a list: {test_index_path}")

            if not isinstance(version['explanation'], str):
                raise ValueError(f"Test index JSON version {i} 'explanation' must be a string: {test_index_path}")

    def validate_active_files(self, agent) -> None:
        """
        Validate that there is at most one active file for the agent and that
        the meta directory structure and files are properly formatted.

        Args:
            agent: The agent instance to validate

        Raises:
            RuntimeError: If validation fails
            ValueError: If file formats are invalid
            FileNotFoundError: If expected files are missing
        """
        agent_id = agent.get_id()

        # Check that agent has at most 1 active file
        active_files = self._get_active_files()
        if len(active_files) > 1:
            raise RuntimeError(
                f"Agent {agent_id} has {len(active_files)} active files, but only 1 is allowed: {list(active_files)}"
            )

        # If no active files, validation passes
        if len(active_files) == 0:
            self.logger.debug(f"Agent {agent_id} has no active files - validation passed")
            return

        # Validate the single active file
        active_file_path = list(active_files)[0]
        self.logger.debug(f"Validating active file: {active_file_path}")

        # Validate file exists
        if not os.path.exists(active_file_path):
            raise FileNotFoundError(f"Active file does not exist: {active_file_path}")

        # Validate meta directory structure
        meta_dir = self._get_meta_dir(active_file_path)
        if not os.path.exists(meta_dir):
            raise FileNotFoundError(f"Meta directory does not exist for active file: {meta_dir}")

        # Validate active.json file
        active_json_path = os.path.join(meta_dir, "active.json")
        if not os.path.exists(active_json_path):
            raise FileNotFoundError(f"Active JSON file missing: {active_json_path}")

        # Validate active.json format and content
        active_data = self._load_json_safely(active_json_path)
        if not active_data:
            raise ValueError(f"Active JSON file is empty or corrupted: {active_json_path}")

        required_active_fields = ['agent_id', 'timestamp', 'file_path']
        for field in required_active_fields:
            if field not in active_data:
                raise ValueError(f"Active JSON missing required field '{field}': {active_json_path}")

        # Validate that the agent_id matches
        if active_data['agent_id'] != agent_id:
            raise ValueError(
                f"Active file agent_id mismatch. Expected: {agent_id}, Found: {active_data['agent_id']}"
            )

        # Validate that the file_path matches
        if os.path.abspath(active_data['file_path']) != os.path.abspath(active_file_path):
            raise ValueError(
                f"Active file path mismatch. Expected: {active_file_path}, Found: {active_data['file_path']}"
            )

        # Validate timestamp format
        try:
            datetime.datetime.fromisoformat(active_data['timestamp'])
        except ValueError as e:
            raise ValueError(f"Invalid timestamp format in active.json: {active_data['timestamp']} - {e}")

        # Validate index.json if it exists
        index_json_path = os.path.join(meta_dir, "index.json")
        if os.path.exists(index_json_path):
            self._validate_index_json(index_json_path, active_file_path)

        # Validate code_analysis.json if it exists
        analysis_json_path = os.path.join(meta_dir, "code_analysis.json")
        if os.path.exists(analysis_json_path):
            self._validate_analysis_json(analysis_json_path, active_file_path)

        # Validate finished.json if it exists
        finished_json_path = os.path.join(meta_dir, "finished.json")
        if os.path.exists(finished_json_path):
            self._validate_finished_json(finished_json_path)

        # Validate version files referenced in index
        if os.path.exists(index_json_path):
            self._validate_version_files(meta_dir, index_json_path)

        # Validate test meta directories if they exist
        self._validate_test_meta_dirs(meta_dir, active_file_path)

        self.logger.debug(f"Validation passed for active file: {active_file_path}")

    def shutdown(self, agent):
        """Clear all active files (useful for shutdown)"""
        for file_path in self.active_files.copy():
            try:
                self.unflag_active(agent, file_path)
            except Exception as e:
                self.logger.debug(f"Failed to cleanup active file {file_path}: {e}")
        self.active_files.clear()
