import datetime
import hashlib
import json
import os
import sys
from typing import List, Dict, Any, Optional

import psutil
import portalocker

from .config import META_DIR_EXT
from .logger import get_logger

class VersionManager:
    """Manages file versioning and analysis for the agent"""

    def __init__(self, config: Dict = None):
        """
        Initialize the VersionManager

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger()

    def _infer_main_file_from_test(self, test_file_path: str) -> Optional[str]:
        """
        Infer the main file path from a test file path using _test.py naming convention.

        Args:
            test_file_path: Path to the test file

        Returns:
            Path to the inferred main file, or None if cannot infer
        """
        test_dir = os.path.dirname(test_file_path)
        test_filename = os.path.basename(test_file_path)

        # Only check for _test.py pattern: module_test.py -> module.py
        if test_filename.endswith('_test.py'):
            main_filename = test_filename[:-8] + '.py'  # Replace '_test.py' with '.py'
            main_file_path = os.path.join(test_dir, main_filename)

            if os.path.exists(main_file_path):
                self.logger.debug(f"Inferred main file {main_file_path} from test file {test_file_path}")
                return main_file_path

        self.logger.warning(f"Could not infer main file from test file: {test_file_path}")
        return None

    def _get_meta_dir(self, file_path: str) -> str:
        """Get the meta directory path for a test file"""
        meta_dir = f"{file_path}.{META_DIR_EXT}"
        if not os.path.exists(meta_dir):
            os.makedirs(meta_dir)
        return meta_dir

    def _get_test_meta_dir(self, test_file_path: str, main_file_path: str) -> str:
        """Get the meta directory path for a test file (nested within main file's meta directory)."""
        main_meta_dir = self._get_meta_dir(main_file_path)
        test_filename = os.path.basename(test_file_path)
        test_meta_dir = os.path.join(main_meta_dir, f"{test_filename}.{META_DIR_EXT}")
        if not os.path.exists(test_meta_dir):
            os.makedirs(test_meta_dir)
        return test_meta_dir

    def save_original(self, file_path: str, content: str) -> int:
        """
        Save the original unedited file into the meta directory.

        Args:
            file_path: Path to the original file being versioned
            content: Content of the original file to save

        Returns:
            The version number assigned to the original file (typically 1)
        """
        self.logger.info(f"Saving original version for file: {file_path}")

        # If meta dir already exists you can assume original has already been saved
        if os.path.exists(self._get_meta_dir(file_path)):
            return 1

        return self.save_version(
            file_path=file_path,
            content=content,
            changes=[{"type": "initial", "description": "Original file content"}],
            explanation="Initial version: Original unedited file"
        )

    def _save_version_internal(self, file_path: str, content: str, versions_dir: str,
                              changes: Optional[List[Dict[str, str]]] = None,
                              explanation: Optional[str] = None,
                              main_file_path: Optional[str] = None) -> int:
        """
        Internal method to save a versioned snapshot with incremental version numbers.

        Args:
            file_path: Path to the file being versioned
            content: Current content of the file to save
            versions_dir: Directory to store versions
            changes: Optional list of changes made in this version
            explanation: Optional explanation of changes made
            main_file_path: Optional main file path for test files

        Returns:
            The version number assigned to this save, or existing version if content hasn't changed
        """
        # Create file hash to check for duplicates
        assert os.path.exists(versions_dir), f"File meta dir {versions_dir} does not exist"
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        self.logger.debug(f"File content hash: {content_hash}")

        # Path to the index file that contains all metadata
        index_file_path = os.path.join(versions_dir, "index.json")
        lock_file_path = os.path.join(versions_dir, "index.lock")

        with open(lock_file_path, 'w') as lock_file:
            portalocker.lock(lock_file, portalocker.LOCK_EX)

            # Check if the index file exists and load it, otherwise create a new one
            if os.path.exists(index_file_path):
                with open(index_file_path, 'r', encoding='utf-8') as f:
                    index = json.load(f)
            else:
                self.logger.debug("No existing index file. Creating new index.")
                index = {'versions': []}
                if main_file_path:
                    index['main_file_path'] = main_file_path

            # Check if we already have a version with the same hash
            for version in index.get('versions', []):
                if version.get('hash') == content_hash:
                    existing_version = version.get('version')
                    self.logger.debug(f"Content already exists in version {existing_version}. Skipping save.")
                    return existing_version

            # Get the next version number
            if index.get('versions'):
                version_number = max(v.get('version', 0) for v in index['versions']) + 1
            else:
                version_number = 1

            # Create version file path
            file_name = os.path.basename(file_path)
            file_base, file_ext = os.path.splitext(file_name)
            version_file_name = f"{file_base}_v{version_number}{file_ext}"
            version_file_path = os.path.join(versions_dir, version_file_name)

            self.logger.debug(f"Version file will be saved as: {version_file_path}")

            # Create metadata entry
            timestamp = datetime.datetime.now().isoformat()
            metadata = {
                'version': version_number,
                'file_path': file_path,
                'timestamp': timestamp,
                'hash': content_hash,
                'changes': changes or [],
                'explanation': explanation or "No explanation provided"
            }
            if main_file_path:
                metadata['main_file_path'] = main_file_path

            # Save the versioned file
            with open(version_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Add the metadata to the index and save it
            index['versions'].append(metadata)
            index['versions'].sort(key=lambda x: x.get('version', 0))
            if main_file_path:
                index['main_file_path'] = main_file_path

            with open(index_file_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=4)

            num_changes = len(changes) if changes else 0
            self.logger.debug(f"Successfully saved version {version_number} with {num_changes} documented changes")

            return version_number

    def save_version(self, file_path: str, content: str,
                     changes: Optional[List[Dict[str, str]]] = None,
                     explanation: Optional[str] = None) -> int:
        """
        Save a versioned snapshot of a main file with incremental version numbers.

        Args:
            file_path: Path to the original file being versioned
            content: Current content of the file to save
            changes: Optional list of changes made in this version
            explanation: Optional explanation of changes made

        Returns:
            The version number assigned to this save, or existing version if content hasn't changed
        """
        self.logger.info(f"Saving version for file: {file_path}")
        versions_dir = self._get_meta_dir(file_path)
        return self._save_version_internal(file_path, content, versions_dir, changes, explanation)

    def save_test_version(self, test_file_path: str, content: str,
                         changes: Optional[List[Dict[str, str]]] = None,
                         explanation: Optional[str] = None,
                         main_file_path: Optional[str] = None) -> int:
        """
        Save a versioned snapshot of a test file in the main file's meta directory.

        Args:
            test_file_path: Path to the test file being versioned
            content: Current content of the test file to save
            changes: Optional list of changes made in this version
            explanation: Optional explanation of changes made
            main_file_path: Optional path to main file; if not provided, will try to infer

        Returns:
            The version number assigned to this save, or existing version if content hasn't changed
        """
        self.logger.info(f"Saving test version for file: {test_file_path}")

        # Determine main file path
        if main_file_path is None:
            main_file_path = self._infer_main_file_from_test(test_file_path)
            if main_file_path is None:
                raise ValueError(f"Could not infer main file path for test file {test_file_path}. Please provide main_file_path explicitly.")

        # Validate main file exists
        if not os.path.exists(main_file_path):
            raise ValueError(f"Main file does not exist: {main_file_path}")

        # Get the nested test meta directory
        test_meta_dir = self._get_test_meta_dir(test_file_path, main_file_path)

        return self._save_version_internal(
            test_file_path, content, test_meta_dir, changes, explanation, main_file_path
        )

    def save_analysis(self, file_path: str, analysis: str,
                      test_path: Optional[str] = None,
                      exit_code: Optional[int] = None,
                      output: Optional[str] = None) -> str:
        """
        Save the latest analysis results in the same versioning directory structure.
        Overwrites any existing analysis file.

        Args:
            file_path: Path to the original file being analyzed
            analysis: The analysis text/results to save
            test_path: Optional path to test file that was executed
            exit_code: Optional exit code from test execution
            output: Optional output from test execution

        Returns:
            Path to the saved analysis file
        """
        self.logger.info(f"Saving analysis for file: {file_path}")

        # Create the versions directory - same as file_path with meta dir ext appended
        versions_dir = self._get_meta_dir(file_path)

        # Use a fixed filename for the latest analysis (no timestamp)
        analysis_filename = "code_analysis.json"
        analysis_file_path = os.path.join(versions_dir, analysis_filename)

        # Create analysis data structure
        analysis_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'test_path': test_path,
            'exit_code': exit_code,
            'output': output,
            'analysis': analysis
        }

        # Save the analysis (overwriting any existing analysis)
        with open(analysis_file_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=4)
        self.logger.info(f"Saved latest analysis to {analysis_file_path}")

        return analysis_file_path

    def _load_analysis(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load the latest analysis results for a given file.

        Args:
            file_path: Path to the original file

        Returns:
            Analysis data dictionary or None if no analysis found
        """
        # Check if versions directory exists
        versions_dir = self._get_meta_dir(file_path)
        if not os.path.exists(versions_dir):
            self.logger.debug(f"No metadata directory found for {file_path}")
            return None

        # Check for latest analysis file
        analysis_file_path = os.path.join(versions_dir, "code_analysis.json")
        if not os.path.exists(analysis_file_path):
            self.logger.debug(f"No analysis file found for {file_path}")
            return None

        try:
            with open(analysis_file_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            self.logger.info(f"Loaded analysis for {file_path}")
            return analysis_data
        except Exception as e:
            self.logger.error(f"Error loading analysis for {file_path}: {str(e)}")
            return None

    def _format_analysis_context(self, analysis_data: Dict[str, Any]) -> str:
        """
        Format analysis data into a context string for prompts.

        Args:
            analysis_data: Analysis data dictionary

        Returns:
            Formatted analysis context string
        """
        if not analysis_data:
            return ""

        context = f"""Code test output:
```
{analysis_data.get('output', 'No output available')}
```

Code analysis:
{analysis_data.get('analysis', 'No analysis available')}

IMPORTANT: Please take this analysis into account when improving the code or tests.
"""
        return context

    def get_analysis_prompt(self, file_path: str) -> str:
        """
        Get formatted analysis context specifically for code editing prompts.

        Args:
            file_path: Path to the file being edited

        Returns:
            Formatted analysis context string or empty string if no analysis
        """
        analysis_data = self._load_analysis(file_path)
        if analysis_data:
            return self._format_analysis_context(analysis_data)
        return None

    def create_analysis(self,
        agent,
        original_file_content: str,
        test_file_content: str,
        output: str,
        exit_code: int) -> str:
        """
        Generate detailed analysis of test execution results using LLM

        Args:
            original_file_content: Content of the original code file
            test_file_content: Content of the test file
            output: Output from test execution
            exit_code: Exit code from test execution

        Returns:
            Detailed analysis of the execution results
        """
        prompt = f"""Analyze the following test execution results and provide comprehensive feedback:

ORIGINAL CODE FILE:
```python
{original_file_content}
```

TEST FILE:
```python
{test_file_content}
```

EXECUTION OUTPUT:
```
{output}
```

EXIT CODE: {exit_code}

Please provide a detailed analysis covering:
1. Overall test execution status (passed/failed)
2. Specific test failures and their root causes
3. Any errors, exceptions, or warnings found
4. Code quality issues revealed by the tests
5. Suggestions for fixing identified problems
6. Recommendations for improving test coverage

Your analysis:
"""

        try:
            response = agent.chat_completion(
                prompt=prompt,
                system_message=agent.role
            )
            return response
        except Exception as e:
            self.logger.error(f"Failed to generate LLM analysis: {str(e)}")
            return f"Failed to generate analysis: {str(e)}"

    def _get_pid_from_agent_id(self, agent_id: str) -> Optional[int]:
        """
        Extract PID from agent ID string.

        Args:
            agent_id: Agent ID in format 'agent_name_pid'

        Returns:
            PID as integer or None if cannot extract
        """
        try:
            # Agent ID format: agent_{safe_name}_{pid}
            parts = agent_id.split('_')
            if len(parts) >= 3:
                return int(parts[-1])  # Last part should be PID
        except (ValueError, IndexError):
            pass
        return None

    def _is_process_running(self, pid: int) -> bool:
        """
        Check if a process with given PID is still running, is a Python process,
        and belongs to the same user.

        Args:
            pid: Process ID to check

        Returns:
            True if process is running, is Python, and owned by current user; False otherwise
        """
        try:
            process = psutil.Process(pid)

            # Check user ownership and Python process in one go
            current_user = os.getlogin() if hasattr(os, 'getlogin') else os.environ.get('USER', os.environ.get('USERNAME'))

            return (
                process.username() == current_user and
                ('python' in process.name().lower() or
                 any('python' in arg.lower() for arg in process.cmdline()[:2]))
            )

        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, AttributeError):
            return False
        except Exception as e:
            self.logger.warning(f"Error checking process {pid}: {e}")
            return False

    def check_active(self, file_path: str) -> bool:
        """
        Check if there is an active agent working on the given file.
        If active file exists but process is no longer running, removes the stale active file.

        Args:
            file_path: Path to the code file to check

        Returns:
            True if there is an active agent working on the file, False otherwise
        """
        meta_dir = self._get_meta_dir(file_path)
        active_file_path = os.path.join(meta_dir, "active.json")
        lock_file_path = os.path.join(meta_dir, "active.lock")

        if not os.path.exists(active_file_path):
            return False

        with open(lock_file_path, 'w') as lock_file:
            portalocker.lock(lock_file, portalocker.LOCK_EX)

            # Re-check existence after acquiring lock
            if not os.path.exists(active_file_path):
                return False

            try:
                with open(active_file_path, 'r', encoding='utf-8') as f:
                    active_data = json.load(f)
            except (json.JSONDecodeError, Exception):
                # Corrupted file, remove it
                os.remove(active_file_path)
                return False

            agent_id = active_data.get('agent_id')
            if not agent_id:
                # Invalid active file, remove it
                os.remove(active_file_path)
                return False

            pid = self._get_pid_from_agent_id(agent_id)
            if pid is None:
                # Cannot extract PID, remove stale file
                os.remove(active_file_path)
                return False

            if self._is_process_running(pid):
                return True
            else:
                # Process is no longer running, remove stale file
                os.remove(active_file_path)
                return False

    def flag_active(self, agent, file_path: str) -> bool:
        """
        Flag a file as being actively worked on by the given agent.
        Uses file locking to prevent race conditions.

        Args:
            agent: Agent instance wanting to work on the file
            file_path: Path to the code file to flag as active

        Returns:
            True if successfully flagged as active, False if already active by another agent
        """
        meta_dir = self._get_meta_dir(file_path)
        active_file_path = os.path.join(meta_dir, "active.json")
        lock_file_path = os.path.join(meta_dir, "active.lock")

        # Create and acquire exclusive lock
        with open(lock_file_path, 'w') as lock_file:
            portalocker.lock(lock_file, portalocker.LOCK_EX)

            # Now safely check if file is already active
            if os.path.exists(active_file_path):
                with open(active_file_path, 'r', encoding='utf-8') as f:
                    try:
                        active_data = json.load(f)
                        agent_id = active_data.get('agent_id')

                        if agent_id:
                            pid = self._get_pid_from_agent_id(agent_id)
                            if pid and self._is_process_running(pid):
                                self.logger.debug(f"File {file_path} is already being worked on by agent {agent_id}")
                                return False
                            else:
                                # Stale active file, will be overwritten
                                self.logger.debug(f"Removing stale active file for {file_path}")
                    except (json.JSONDecodeError, Exception) as e:
                        # Corrupted active file, will be overwritten
                        self.logger.debug(f"Corrupted active file for {file_path}: {e}")

            # File is not actively being worked on, claim it
            active_data = {
                'agent_id': agent.get_id(),
                'timestamp': datetime.datetime.now().isoformat(),
                'file_path': file_path
            }

            with open(active_file_path, 'w', encoding='utf-8') as f:
                json.dump(active_data, f, indent=4)

            self.logger.debug(f"Flagged {file_path} as active by agent {agent.get_id()}")
            return True

    def unflag_active(self, agent, file_path: str) -> bool:
        """
        Remove the active flag for a file if it was set by the given agent.

        Args:
            agent: Agent instance that wants to remove its active flag
            file_path: Path to the code file to unflag

        Returns:
            True if successfully unflagged, False if not flagged by this agent or file doesn't exist
        """
        meta_dir = self._get_meta_dir(file_path)
        active_file_path = os.path.join(meta_dir, "active.json")
        lock_file_path = os.path.join(meta_dir, "active.lock")

        if not os.path.exists(active_file_path):
            self.logger.debug(f"No active file to remove for {file_path}")
            return False

        with open(lock_file_path, 'w') as lock_file:
            portalocker.lock(lock_file, portalocker.LOCK_EX)

            # Re-check existence after acquiring lock
            if not os.path.exists(active_file_path):
                self.logger.debug(f"No active file to remove for {file_path}")
                return False

            try:
                with open(active_file_path, 'r', encoding='utf-8') as f:
                    active_data = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                self.logger.debug(f"Corrupted active file for {file_path}: {e}")
                return False

            if active_data.get('agent_id') != agent.get_id():
                self.logger.debug(f"Cannot unflag {file_path} - not flagged by agent {agent.get_id()}")
                return False

            os.remove(active_file_path)
            self.logger.debug(f"Unflagged {file_path} from agent {agent.get_id()}")
            return True

    def check_finished(self, agent, file_path: str) -> bool:
        """
        Check if a file has been marked as finished.

        Args:
           file_path: Path to the code file to check
           agent_id: Optional specific agent ID to check; if None, checks if any agent finished

        Returns:
           True if file has been finished (by specified agent or any agent), False otherwise
        """
        meta_dir = self._get_meta_dir(file_path)
        finished_file_path = os.path.join(meta_dir, "finished.json")

        if not os.path.exists(finished_file_path):
           return False

        with open(finished_file_path, 'r', encoding='utf-8') as f:
           finished_data = json.load(f)

        agents = finished_data.get('agents', [])
        if not agents:
           return False

        for agent_entry in agents:
           if agent_entry.get('agent_id') == agent.get_id():
               return True
        return False

    def flag_finished(self, agent, file_path: str) -> None:
        """
        Flag a file as finished being worked on by the given agent.

        Args:
            agent: Agent instance that finished working on the file
            file_path: Path to the code file to flag as finished
        """
        meta_dir = self._get_meta_dir(file_path)
        finished_file_path = os.path.join(meta_dir, "finished.json")
        lock_file_path = os.path.join(meta_dir, "finished.lock")

        with open(lock_file_path, 'w') as lock_file:
            portalocker.lock(lock_file, portalocker.LOCK_EX)

            if os.path.exists(finished_file_path):
                try:
                    with open(finished_file_path, 'r', encoding='utf-8') as f:
                        finished_data = json.load(f)
                except (json.JSONDecodeError, Exception):
                    # Corrupted file, start fresh
                    finished_data = {'agents': []}
            else:
                finished_data = {'agents': []}

            # Check if agent already marked as finished (prevent duplicates)
            for agent_entry in finished_data['agents']:
                if agent_entry.get('agent_id') == agent.get_id():
                    self.logger.debug(f"Agent {agent.get_id()} already marked {file_path} as finished")
                    return

            finished_data['agents'].append({
                'agent_id': agent.get_id(),
                'timestamp': datetime.datetime.now().isoformat(),
                'file_path': file_path
            })

            with open(finished_file_path, 'w', encoding='utf-8') as f:
                json.dump(finished_data, f, indent=4)

            self.logger.debug(f"Flagged {file_path} as finished by agent {agent.get_id()}")
