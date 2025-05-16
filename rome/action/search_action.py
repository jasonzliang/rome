import glob
import os
import sys
import traceback
from typing import Dict, List, Any, Optional
from .action import Action
from .logger import get_logger


class SearchAction(Action):
    """Action to search the repository for code files using OpenAI selection"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()

        # Set default configuration for search action with proper fallbacks
        self.max_files = self.config.get('max_files', sys.maxsize)
        self.file_type = self.config.get('file_type', '.py')
        self.depth = self.config.get('depth', sys.maxsize)
        self.exclude_dirs = self.config.get('exclude_dirs',
           ['.git', 'node_modules', 'venv', '__pycache__', 'dist', 'build'])
        # Parameter K for batch processing
        self.batch_size = self.config.get('batch_size', 5)
        # Selection criteria for OpenAI
        self.selection_criteria = self.config.get('selection_criteria',
            "Select the most relevant file for the current task")

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
        self.logger.info("Starting SearchAction execution")

        # Ensure agent has OpenAI handler
        if not hasattr(agent, 'openai_handler') or agent.openai_handler is None:
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

        self.logger.info(f"Found {len(filtered_files)} files after filtering")

        # Limit the number of files if specified
        if self.max_files and len(filtered_files) > self.max_files:
            self.logger.info(f"Limited search results to {self.max_files} files")
            filtered_files = filtered_files[:self.max_files]

        if not filtered_files:
            self.logger.info("No files found matching search criteria")
            agent.context['selected_file'] = None
            return

        # Process files in batches of K until a file is selected
        current_index = 0
        selected_file = None

        while current_index < len(filtered_files) and selected_file is None:
            # Get next batch of K files
            batch_end = min(current_index + self.batch_size, len(filtered_files))
            current_batch_paths = filtered_files[current_index:batch_end]

            self.logger.info(f"Processing batch {current_index//self.batch_size + 1}: files {current_index+1}-{batch_end}")

            # Load content for current batch
            current_batch = []
            for file_path in current_batch_paths:
                # Try to read file, skip if there are issues
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    current_batch.append({
                        'path': file_path,
                        'content': content
                    })

            if not current_batch:
                self.logger.info(f"No readable files in batch {current_index//self.batch_size + 1}")
                current_index = batch_end
                continue

            # Create prompt for OpenAI
            prompt = self._create_selection_prompt(current_batch)

            # Use action-specific LLM config if available
            self.logger.info("Querying OpenAI for file selection")
            response = agent.chat_completion(
                prompt=prompt,
                system_message=agent.role,
                action_type='search',
                response_format={"type": "json_object"}
            )

            # Parse OpenAI response
            result = agent.parse_json_response(response)

            # Check for parsing errors
            if "error" in result:
                self.logger.error(f"Error parsing OpenAI response: {result['error']}")
                current_index = batch_end
                continue

            # Process OpenAI response
            selected_file_info = result.get('selected_file', {})
            should_continue = result.get('should_continue', True)

            if selected_file_info:
                file_path = selected_file_info.get('path')
                reason = selected_file_info.get('reason', 'Selected by OpenAI')

                # Find the full file info
                for file_info in current_batch:
                    if file_info['path'] == file_path:
                        selected_file = {
                            'path': file_path,
                            'content': file_info['content'],
                            'reason': reason
                        }
                        self.logger.info(f"Selected file: {file_path} - {reason}")
                        break

            # If we have a file and shouldn't continue, break
            if selected_file and not should_continue:
                self.logger.info(f"Agent {agent.name} indicated to stop searching")
                break

            # Move to next batch
            current_index = batch_end

            self.logger.info(f"Processed batch {current_index//self.batch_size}: "
                            f"{'File selected' if selected_file else 'No file selected'}")

        # Add selected file to agent context
        if selected_file:
            agent.context['selected_file'] = selected_file
            self.logger.info(f"Search completed. Selected file: {selected_file['path']}")
        else:
            agent.context['selected_file'] = None
            self.logger.info("Search completed. No file was selected.")