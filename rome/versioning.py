import os
import json
import datetime
import hashlib
from typing import List, Dict, Any, Optional
from .config import META_DIR_EXT
from .logger import get_logger

class VersionManager:
    """Manages file versioning and analysis for the agent"""

    def __init__(self, config: Dict = None):
        """
        Initialize the VersionManager

        Args:
            agent: Reference to the agent instance
        """
        self.config = config or {}
        self.logger = get_logger()

    def save_original(self, file_path: str, content: str) -> int:
        """
        Save the original unedited file into the meta directory.
        This is a convenience function that calls save_version with
        an appropriate explanation.

        Args:
            file_path: Path to the original file being versioned
            content: Content of the original file to save

        Returns:
            The version number assigned to the original file (typically 1)
        """
        self.logger.info(f"Saving original version for file: {file_path}")

        # If meta dir already exists you can assume original has already been saved
        if os.path.exists(f"{file_path}.{META_DIR_EXT}"):
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
        """
        Save a versioned snapshot of a file with incremental version numbers.
        Version files are stored in a directory with the same name as the file
        but with meta dir ext appended. All metadata is stored in a index.json file.

        Args:
            file_path: Path to the original file being versioned
            content: Current content of the file to save
            changes: Optional list of changes made in this version
            explanation: Optional explanation of changes made

        Returns:
            The version number assigned to this save, or existing version if content hasn't changed
        """
        self.logger.info(f"Saving version for file: {file_path}")

        # Create file hash to check for duplicates
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        self.logger.debug(f"File content hash: {content_hash}")

        # Create versions directory with the meta dir ext
        base_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        file_base, file_ext = os.path.splitext(file_name)

        # Create the versions directory - same as file_path with meta dir ext appended
        versions_dir = f"{file_path}.{META_DIR_EXT}"

        if not os.path.exists(versions_dir):
            os.makedirs(versions_dir)

        # Path to the index file that contains all metadata
        index_file_path = os.path.join(versions_dir, "index.json")

        # Check if the index file exists and load it, otherwise create a new one
        if os.path.exists(index_file_path):
            try:
                with open(index_file_path, 'r', encoding='utf-8') as f:
                    index = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading index file: {str(e)}")
                raise
        else:
            self.logger.debug("No existing index file. Creating new index.")
            index = {'versions': []}

        # Check if we already have a version with the same hash
        for version in index.get('versions', []):
            if version.get('hash') == content_hash:
                existing_version = version.get('version')
                self.logger.error(f"Content already exists in version {existing_version}. Skipping save.")
                return existing_version

        # Get the next version number
        if index.get('versions'):
            version_number = max(v.get('version', 0) for v in index['versions']) + 1
        else:
            version_number = 1

        # Create version file path
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

        # Save the versioned file
        try:
            with open(version_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            self.logger.error(f"Failed to save version file: {str(e)}")
            raise

        try:
            # Add the metadata to the index and save it
            index['versions'].append(metadata)
            # Sort versions by version number
            index['versions'].sort(key=lambda x: x.get('version', 0))

            with open(index_file_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to update index file: {str(e)}")
            raise

        num_changes = len(changes) if changes else 0
        self.logger.info(f"Successfully saved version {version_number} with {num_changes} documented changes")

        return version_number

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
        versions_dir = f"{file_path}.{META_DIR_EXT}"

        if not os.path.exists(versions_dir):
            os.makedirs(versions_dir)
            self.logger.debug(f"Created metadata directory: {versions_dir}")

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
        try:
            with open(analysis_file_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2)
            self.logger.info(f"Saved latest analysis to {analysis_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save analysis file: {str(e)}")
            raise

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
        versions_dir = f"{file_path}.{META_DIR_EXT}"
        if not os.path.exists(versions_dir):
            self.logger.debug(f"No metadata directory found for {file_path}")
            return None

        # Check for latest analysis file
        analysis_file_path = os.path.join(versions_dir, "latest_analysis.json")
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
            raise

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

# - Execution Date: {analysis_data.get('timestamp', 'Unknown')}
# - Test File: {analysis_data.get('test_path', 'Unknown')}
# - Exit Code: {analysis_data.get('exit_code', 'Unknown')}
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
            return self._format_analysis(analysis_data)
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
            raise