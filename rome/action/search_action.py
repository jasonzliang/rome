import glob
import os
import sys
import time
import traceback
from typing import Dict, List, Any, Optional
from .action import Action
from .logger import get_logger


class SearchAction(Action):
    """Action to search the repository for code files using OpenAI selection"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()

        # Assert required config parameters using a loop
        required_attrs = ['max_files', 'file_type', 'exclude_dirs', 'selection_criteria', 'batch_size']
        for attr in required_attrs:
            assert hasattr(self, attr), f"{attr} not provided in SearchAction config"

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
                self.logger.warning(f"Error reading {file_path}: {e}")

        self.logger.info(f"Collected overview data for {len(file_overviews)} files")
        return file_overviews

    def _prioritize_files(self, agent, file_overviews: List[Dict]) -> List[Dict]:
        """Use LLM to prioritize files based on size, age, and definitions"""
        self.logger.info("Prioritizing files based on overview data")

        # Create batches for prioritization (to avoid token limits)
        batch_size = 50  # Process 50 files at a time for prioritization
        prioritized_files = []

        for i in range(0, len(file_overviews), batch_size):
            batch = file_overviews[i:i+batch_size]
            self.logger.info(f"Prioritizing batch {i//batch_size + 1} of {(len(file_overviews) + batch_size - 1)//batch_size}")

            # Create prioritization prompt
            prompt = f"""Assign a priority score (1-5, 5 being highest) to each file based on:
1. File age (older files are more stable and likely more important)
2. File size (smaller files are more focused and easier to analyze)
3. Function/class definitions that match the selection criteria and your role

Selection criteria: {self.selection_criteria}

Please review these files and assign a priority score to each:

"""
            for j, file_info in enumerate(batch):
                prompt += f"\n--- File {j+1}: {file_info['path']} ---\n"
                prompt += f"Size: {file_info['size_kb']} KB\n"
                prompt += f"Last modified: {file_info['modified_age']} seconds ago\n"

                if file_info['definitions']:
                    prompt += "Definitions:\n"
                    for defn in file_info['definitions']:
                        prompt += f"  {defn}\n"
                else:
                    prompt += "No function/class definitions found\n"

            prompt += """
Respond with a JSON array of prioritized files in this format:
[
  {
    "file_path": "path/to/file.py",
    "priority": 5,
    "reason": "Brief reason for this priority score"
  },
  ...
]

Assign higher priority (4-5) to files that:
- Are older (more stable)
- Are smaller (more focused)
- Have function/class definitions relevant to the selection criteria/role
"""

            # Get priorities from LLM
            response = agent.chat_completion(
                prompt=prompt,
                system_message=agent.role,
                response_format={"type": "json_object"}
            )

            # Parse response
            result = agent.parse_json_response(response)

            if result and isinstance(result, list):
                prioritized_files.extend(result)
            else:
                self.logger.error(f"Error parsing file priorities: {result}")
                # Assign default priority 1 to all files in this batch if parsing fails
                for file_info in batch:
                    prioritized_files.append({
                        "file_path": file_info["path"],
                        "priority": 1,
                        "reason": "Default priority (parsing error)"
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

    def _create_selection_prompt(self, files_batch: List[Dict]) -> str:
        """Create a prompt for OpenAI to evaluate file selection"""
        prompt = f"""You are analyzing code files to select THE MOST RELEVANT file.

Selection criteria: {self.selection_criteria}

IMPORTANT: You must select EXACTLY ONE file from the batch below.

Please analyze the following files and select the most relevant one:

"""
        for i, file_info in enumerate(files_batch):
            prompt += f"\n--- File {i+1}: {file_info['path']} ---\n"
            # Include full content (no truncation)
            prompt += file_info['content'] + "\n"

        prompt += """
Please respond with a single JSON object in the following format:
{
    "selected_file": {
        "file_number": 1,
        "path": "path/to/file.py",
        "reason": "Detailed reason for selection"
    },
    "should_continue": true/false
}

- You MUST select exactly one file
- Set "should_continue" to false if you found the perfect file, true otherwise
- Provide a detailed reason for your selection
"""
        return prompt

    def execute(self, agent, **kwargs):
        self.logger.info("Starting simplified three-step SearchAction execution")

        # Ensure agent has an OpenAI handler (either openai_handler or self.openai_handler)
        has_openai_handler = hasattr(agent, 'openai_handler') and agent.openai_handler is not None

        if not has_openai_handler:
            error_msg = "Agent must have an openai_handler attribute initialized"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Get the repo root path from agent
        search_path = os.path.join(agent.repository, '**/*' + self.file_type)
        self.logger.info(f"Searching for files in: {search_path}")

        # Find all files matching the pattern
        files = glob.glob(search_path, recursive=True)
        self.logger.info(f"Found {len(files)} files before filtering")

        # Filter out files from excluded directories
        filtered_files = self._filter_excluded_dirs(files)
        self.logger.info(f"Found {len(filtered_files)} files after directory filtering")

        # Step 1: Create global overview of all files
        file_overviews = self._create_global_overview(agent, filtered_files)

        # Step 2: Prioritize files based on overview data
        prioritized_files = self._prioritize_files(agent, file_overviews)

        # Store prioritized files in agent context for reference (not needed)
        # agent.context['prioritized_files'] = prioritized_files

        # Step 3: Process batches of files in priority order for detailed review
        file_paths = [file_info["file_path"] for file_info in prioritized_files]
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
                    self.logger.warning(f"Error reading {file_path}: {e}")

            if not current_batch_data:
                self.logger.info(f"No readable files in batch {current_batch+1}")
                current_batch += 1
                continue

            # Create selection prompt
            prompt = f"""You are selecting the most relevant file based on full content.

Selection criteria: {self.selection_criteria}

IMPORTANT: You must select THE MOST RELEVANT file from the batch below OR explicitly indicate that none are sufficiently relevant.

Current batch: {current_batch+1} of {(len(file_paths) + self.batch_size - 1) // self.batch_size}
Files in this batch (with assigned priorities):
"""

            for i, file_info in enumerate(current_batch_data):
                filename = os.path.basename(file_info['path'])
                prompt += f"\n- File {i+1}: {filename} (Priority: {file_info['priority']}/5) - {file_info['priority_reason']}"

            prompt += "\n\nDetailed file contents:"

            for i, file_info in enumerate(current_batch_data):
                prompt += f"\n\n--- File {i+1}: {file_info['path']} (Priority: {file_info['priority']}/5) ---\n"
                prompt += file_info['content']

            prompt += """

Please respond with a single JSON object in the following format:
{
    "selected_file": {
        "file_number": 1,
        "path": "path/to/file.py",
        "reason": "Detailed reason for selection"
    },
    "should_continue": true/false
}

- Set "selected_file" to null if no file meets the selection criteria
- Set "should_continue" to false if you found the perfect file or true if we should continue searching
"""

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
            should_continue = result.get('should_continue', True)

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
                    self.logger.warning(f"Invalid file_number in response: {file_number}")

            # If perfect match found or explicitly directed to stop, break out of loop
            if (selected_file and not should_continue):
                self.logger.info("Perfect match found, stopping search")
                break

            # Move to next batch
            current_batch += 1

        # Add selected file to agent context
        if selected_file:
            agent.context['selected_file'] = selected_file
            self.logger.info(f"Search completed. Selected file: {selected_file['path']}")
        else:
            agent.context['selected_file'] = None
            self.logger.info("Search completed. No file was selected.")
