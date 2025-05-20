import glob
import os
import pprint
import random  # Added for probability selection
import sys
import time
import traceback
from typing import Dict, Optional, Any, Union, List
from .action import Action
from ..logger import get_logger
from ..config import check_attrs


class SearchAction(Action):
    """Action to search the repository for code files using OpenAI selection"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()

        # Check required config parameters are set properly
        check_attrs(['epilson_oldest', 'max_files', 'file_types', 'exclude_dirs', 'exclude_types', 'selection_criteria', 'batch_size'])

    def _create_global_overview(self, agent, files: List[str]) -> List[Dict]:
        """Create a high-level overview of all files in the repository"""
        self.logger.info(f"Creating global overview of {len(files)} files")

        # Collect file data for overview
        file_overviews = []
        current_time = time.time()

        for file_path in files:
            try:
                # Get file stats
                stats = os.stat(file_path)
                size_kb = stats.st_size / 1024
                modified_age = current_time - stats.st_mtime  # Age in seconds

                # Read functions/classes from file
                definitions = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract function and class definitions
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith('def ') or line.startswith('class '):
                            definitions.append(line)

                file_overviews.append({
                    "path": file_path,
                    "size_kb": round(size_kb, 2),
                    "modified_age": round(modified_age, 1),  # Age in seconds
                    "definitions": definitions[:10]  # Limit to first 10 definitions
                })

            except Exception as e:
                self.logger.info(f"Error reading {file_path}: {e}")

        # self.logger.info(f"Collected overview data for {len(file_overviews)} files")
        return file_overviews

    def _prioritize_files(self, agent, file_overviews: List[Dict]) -> List[Dict]:
        """Use LLM to prioritize all files at once based on size, age, and definitions"""
        self.logger.info(f"Prioritizing all {len(file_overviews)} files at once")

        # Create a clear and direct prompt
        prompt = f"""Return a JSON ARRAY of ALL {len(file_overviews)} files with priority scores (1-5).

Assign a priority score (1-5, 5 being highest) to each file based on:
1. File age (older files are more stable and likely more important)
2. File size (smaller files are more focused and easier to analyze)
3. Function/class definitions that match your role and selection criteria

Files to prioritize:
"""
        # Add file information to the prompt
        for j, file_info in enumerate(file_overviews):
            prompt += f"\n--- File {j+1}: {file_info['path']} ---\n"
            prompt += f"Size: {file_info['size_kb']} KB\n"
            prompt += f"Last modified: {file_info['modified_age']} seconds ago\n"

            if file_info['definitions']:
                prompt += "Definitions:\n"
                for defn in file_info['definitions']:
                    prompt += f"  {defn}\n"
            else:
                prompt += "No function/class definitions found\n"

        prompt += f"""
Return a JSON ARRAY with ALL {len(file_overviews)} files like this:
[
  {{
    "file_path": "path/to/file1.py",
    "priority": 1-5,
    "reason": "Brief reason"
  }},
  {{
    "file_path": "path/to/file2.py",
    "priority": 1-5,
    "reason": "Brief reason"
  }}
  {', ...' if len(file_overviews) > 2 else ''}
]

IMPORTANT: Your response MUST be a valid JSON ARRAY starting with [ and ending with ], not a single object.
"""

        # Get priorities from LLM without specifying response_format
        response = agent.chat_completion(
            prompt=prompt,
            system_message=agent.role
            # No response_format parameter
        )

        # Use parse_json_response to extract the result
        result = agent.parse_json_response(response)

        # Initialize prioritized_files list
        prioritized_files = []

        # Handle the result based on its type
        if isinstance(result, list):
            # Result is a list - use it directly
            self.logger.info(f"Received array with {len(result)} files")
            prioritized_files = result
        elif isinstance(result, dict) and "file_path" in result:
            # Result is a single object - create array with all files
            self.logger.info("Received single object instead of array, creating array manually")
            prioritized_files.append(result)

            # Add all other files with lower priority
            for file_info in file_overviews:
                if file_info["path"] != result["file_path"]:
                    prioritized_files.append({
                        "file_path": file_info["path"],
                        "priority": 1,  # Default priority for files not included
                        "reason": "Added by default (not in LLM response)"
                    })
        else:
            # Invalid response - create default priorities for all files
            self.logger.error("Invalid response format, using default priorities for all files")
            for file_info in file_overviews:
                prioritized_files.append({
                    "file_path": file_info["path"],
                    "priority": 1,
                    "reason": "Default priority (LLM response format error)"
                })

        # Ensure all files from file_overviews are included
        all_paths = {file_info["path"] for file_info in file_overviews}
        prioritized_paths = {file_info.get("file_path") for file_info in prioritized_files}

        # Add any missing files
        for path in all_paths - prioritized_paths:
            self.logger.info(f"File missing from prioritized list: {path}, adding with default priority")
            prioritized_files.append({
                "file_path": path,
                "priority": 1,
                "reason": "Added by default (missing from LLM response)"
            })

        # Sort by priority (higher first)
        prioritized_files.sort(key=lambda x: x.get("priority", 0), reverse=True)
        self.logger.info(f"Prioritized {len(prioritized_files)} files")

        return prioritized_files

    def _filter_excluded_dirs(self, files: List[str]) -> List[str]:
        """Filter out files from excluded directories"""
        filtered_files = []
        for file_path in files:
            # Check if file is in an excluded directory
            exclude = False
            for excluded_dir in self.exclude_dirs:
                # Normalize path separators for cross-platform compatibility
                normalized_path = file_path.replace('\\', '/')
                normalized_excluded = excluded_dir.replace('\\', '/')

                if f'/{normalized_excluded}/' in normalized_path or \
                   normalized_path.startswith(f'{normalized_excluded}/'):
                    exclude = True
                    break
            if not exclude:
                filtered_files.append(file_path)
        return filtered_files

    def _filter_excluded_types(self, files: List[str]) -> List[str]:
        """Filter out files with excluded file types"""
        if not hasattr(self, 'exclude_types') or not self.exclude_types:
            return files

        filtered_files = []
        for file_path in files:
            # Get file extension
            _, ext = os.path.splitext(file_path)
            # Remove leading dot from extension if present
            ext = ext[1:] if ext.startswith('.') else ext

            # Check if file type is excluded
            if ext not in self.exclude_types:
                filtered_files.append(file_path)
        return filtered_files

    def _filter_max_limit(self, files: List[str]) -> List[str]:
        if len(files) > self.max_files:
            random.shuffle(files)
            files = files[:self.max_files]
        return files

    def _process_file_batches(self, agent, file_paths: List[str], prioritized_files: List[Dict]) -> Dict:
        """
        Process batches of files to find the most relevant match

        Args:
            agent: The agent instance
            file_paths: List of file paths to process
            prioritized_files: List of prioritized files with metadata

        Returns:
            Dictionary with selected file or None
        """
        self.logger.info("Processing file batches to find relevant match")
        current_batch = 0
        selected_file = None

        while current_batch * self.batch_size < len(file_paths) and selected_file is None:
            start_idx = current_batch * self.batch_size
            end_idx = min((current_batch + 1) * self.batch_size, len(file_paths))
            batch_paths = file_paths[start_idx:end_idx]

            self.logger.info(f"Processing batch {current_batch+1}: files {start_idx+1}-{end_idx}")

            # Load full content for current batch
            current_batch_data = []
            for file_path in batch_paths:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # Find the priority info for this file
                        priority_info = next((f for f in prioritized_files if f["file_path"] == file_path), None)
                        priority = priority_info.get("priority", 1) if priority_info else 1
                        reason = priority_info.get("reason", "No reason provided") if priority_info else "No reason provided"

                        current_batch_data.append({
                            'path': file_path,
                            'content': content,
                            'priority': priority,
                            'priority_reason': reason
                        })
                except Exception as e:
                    self.logger.info(f"Error reading {file_path}: {e}")

            if not current_batch_data:
                self.logger.info(f"No readable files in batch {current_batch+1}")
                current_batch += 1
                continue

            # Create a more concise selection prompt
            prompt = self._create_selection_prompt(current_batch_data, current_batch,
                                                 (len(file_paths) + self.batch_size - 1) // self.batch_size)

            # Request LLM to select a file
            self.logger.info("Querying OpenAI for file selection")
            response = agent.chat_completion(
                prompt=prompt,
                system_message=agent.role,
                response_format={"type": "json_object"}
            )

            # Parse OpenAI response
            result = agent.parse_json_response(response)

            # Check for parsing errors
            if not result or "error" in result:
                self.logger.error(f"Error parsing OpenAI response: {result.get('error', 'Unknown error')}")
                current_batch += 1
                continue

            # Process OpenAI response
            selected_file_info = result.get('selected_file')

            if selected_file_info:
                file_number = selected_file_info.get('file_number')
                if file_number and 1 <= file_number <= len(current_batch_data):
                    file_data = current_batch_data[file_number - 1]
                    reason = selected_file_info.get('reason', 'Selected by LLM')

                    selected_file = {
                        'path': file_data['path'],
                        'content': file_data['content'],
                        'reason': reason,
                        'priority': file_data['priority']
                    }

                    self.logger.info(f"Selected file: {file_data['path']} - {reason}")
                else:
                    self.logger.error(f"Invalid file_number in response: {file_number}")

            # If perfect match found or explicitly directed to stop, break out of loop
            if selected_file:
                self.logger.info("Perfect match found, stopping search")
                break

            # Move to next batch
            current_batch += 1

        return selected_file

    def _create_selection_prompt(self, batch_data: List[Dict], current_batch: int, total_batches: int) -> str:
        """
        Create a concise prompt for file selection

        Args:
            batch_data: List of file data for current batch
            current_batch: Current batch number (1-indexed)
            total_batches: Total number of batches

        Returns:
            Prompt string for LLM
        """
        # File list summary at the top
        prompt = f"""Select the most relevant file based on your role and selection criteria: {self.selection_criteria}

Current batch: {current_batch+1}/{total_batches}
File list:
"""
        # Add short file summaries
        for i, file_info in enumerate(batch_data):
            filename = os.path.basename(file_info['path'])
            prompt += f"- File {i+1}: {filename} (Priority: {file_info['priority']}/5) - {file_info['priority_reason']}\n"

        # Add full file contents
        prompt += "\nDetailed file contents:"
        for i, file_info in enumerate(batch_data):
            prompt += f"\n\n--- File {i+1}: {file_info['path']} ---\n"
            prompt += file_info['content']

        # Add expected response format
        prompt += """

Respond with a JSON object:
{
    "selected_file": {
        "file_number": <int>,
        "path": "<file_path>",
        "reason": "<rationale>"
    }
}

- Set "selected_file" to null if no file meets the criteria
"""
        return prompt

    def _find_oldest_file(self, file_overviews: List[Dict]) -> Dict:
        """Find the oldest file based on modification age"""
        if not file_overviews:
            return None

        oldest_file = max(file_overviews, key=lambda x: x.get("modified_age", 0))
        self.logger.info(f"Oldest file found: {oldest_file['path']} (age: {oldest_file['modified_age']} seconds)")

        # Get file content
        try:
            with open(oldest_file['path'], 'r', encoding='utf-8') as f:
                content = f.read()

            return {
                'path': oldest_file['path'],
                'content': content,
                'reason': "Selected as oldest file based on epilson_oldest probability",
                'priority': 5  # Give it highest priority since it was selected by age
            }
        except Exception as e:
            self.logger.error(f"Error reading oldest file {oldest_file['path']}: {e}")
            return None

    def execute(self, agent, **kwargs) -> bool:
        self.logger.info("Starting SearchAction execution with file type list and oldest file probability")

        # Ensure agent has an OpenAI handler (either openai_handler or self.openai_handler)
        has_openai_handler = hasattr(agent, 'openai_handler') and agent.openai_handler is not None

        if not has_openai_handler:
            error_msg = "Agent must have an openai_handler attribute initialized"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Get the repo root path from agent
        all_files = []

        # Handle multiple file types
        for file_type in self.file_types:
            # Ensure file_type has a leading dot if needed
            if not file_type.startswith('.'):
                file_type = '.' + file_type

            search_path = os.path.join(agent.repository, f'**/*{file_type}')
            self.logger.info(f"Searching for files in: {search_path}")

            # Find all files matching the pattern
            files = glob.glob(search_path, recursive=True)
            self.logger.info(f"Found {len(files)} {file_type} files")
            all_files.extend(files)

        self.logger.info(f"Found {len(all_files)} total files before filtering")

        # Filter out files from excluded directories
        filtered_files = self._filter_excluded_dirs(all_files)
        self.logger.info(f"Found {len(filtered_files)} files after directory filtering")

        # Filter out files with excluded types
        filtered_files = self._filter_excluded_types(filtered_files)
        self.logger.info(f"Found {len(filtered_files)} files after type filtering")

        # Filter out files if they exceed max limit
        filtered_files = self._filter_max_limit(filtered_files)
        self.logger.info(f"Found {len(filtered_files)} files after max-limit filtering")

        # Step 1: Create global overview of all files
        file_overviews = self._create_global_overview(agent, filtered_files)

        # Check if we should select the oldest file based on probability
        if self.epilson_oldest > 0 and random.random() < self.epilson_oldest:
            self.logger.info(f"Using epilson_oldest probability ({self.epilson_oldest}) to select oldest file")
            oldest_file = self._find_oldest_file(file_overviews)

            if oldest_file:
                agent.context['selected_file'] = oldest_file
                self.logger.info(f"Selected oldest file: {oldest_file['path']}")
                return True
            else:
                # If oldest file wasn't selected, continue with normal process
                self.logger.error("Failed to read oldest file, continuing with normal selection process")

        # Step 2: Prioritize files based on overview data
        prioritized_files = self._prioritize_files(agent, file_overviews)

        # Step 3: Process batches of files in priority order for detailed review
        file_paths = [file_info["file_path"] for file_info in prioritized_files]
        selected_file = self._process_file_batches(agent, file_paths, prioritized_files)

        # Add selected file to agent context
        if selected_file:
            agent.context['selected_file'] = selected_file
            self.logger.info(f"Search completed. Selected file: {selected_file['path']}")
            return True
        else:
            agent.context['selected_file'] = None
            self.logger.info("Search completed. No file was selected.")
            return False
