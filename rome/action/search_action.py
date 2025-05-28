import glob
import os
import random
import sys
import time
from typing import Dict, Optional, Any, Union, List
from .action import Action
from ..logger import get_logger
from ..config import LOG_DIR_NAME, SUMMARY_LENGTH, check_attrs
from ..parsing import extract_all_definitions

class SearchAction(Action):
    """Action to search the repository for code files using OpenAI selection"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()

        check_attrs(self, ['max_files', 'file_types', 'exclude_dirs',
                          'exclude_types', 'selection_criteria', 'batch_size',
                          'batch_sampling'])

        if LOG_DIR_NAME not in self.exclude_dirs:
            self.exclude_dirs.append(LOG_DIR_NAME)

    def summary(self, agent) -> str:
        """Return a detailed summary of the search action"""
        file_types_str = ', '.join(self.file_types)
        excluded_dirs_str = ', '.join(self.exclude_dirs) if self.exclude_dirs else 'none'

        return (f"search repository for {file_types_str} files using multi-stage LLM selection: "
                f"(1) scan all files and create overview with size/age/function definitions, "
                f"(2) LLM prioritizes all files 1-5 based on {self.selection_criteria}, "
                f"(3) process top files in batches of {self.batch_size} with full content for LLM to select best match "
                f"(max files: {self.max_files}, excluding dirs: {excluded_dirs_str})")

    def _truncate_text(self, text: str, max_length: int = SUMMARY_LENGTH) -> str:
        """Truncate text to specified length with ellipsis if needed"""
        return text[:max_length] + "..." if len(text) > max_length else text

    def _get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """Get file statistics and content analysis"""
        try:
            stats = os.stat(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            all_definitions = extract_all_definitions(content)

            return {
                "path": file_path,
                "content": content,
                "size_kb": round(stats.st_size / 1024, 2),
                "modified_age": round(time.time() - stats.st_mtime, 1),
                "all_definitions": all_definitions,
                "definition_count": len(all_definitions),
                "function_count": len([d for d in all_definitions if d['type'] == 'function']),
                "class_count": len([d for d in all_definitions if d['type'] == 'class']),
            }
        except Exception as e:
            self.logger.info(f"Error reading {file_path}: {e}")
            return None

    def _create_definition_summary(self, definition: Dict) -> str:
        """Create a concise summary of a function/class definition"""
        summary = definition['signature']

        if definition['docstring']:
            # Allow multiple lines but limit total characters to SUMMARY_LENGTH
            docstring_lines = [line.strip() for line in definition['docstring'].split('\n') if line.strip()]
            if docstring_lines:
                # Join lines with space and truncate to SUMMARY_LENGTH
                docstring_text = ' '.join(docstring_lines)
                truncated_docstring = self._truncate_text(docstring_text)
                summary += f" # {truncated_docstring}"

        if definition['type'] == 'class' and definition.get('methods'):
            method_count = len(definition['methods'])
            summary += f" ({method_count} methods)"

        return summary

    def _create_global_overview(self, agent, files: List[str]) -> List[Dict]:
        """Create a high-level overview of all files in the repository"""
        self.logger.info(f"Creating global overview of {len(files)} files")

        file_overviews = []
        for file_path in files:
            file_stats = self._get_file_stats(file_path)
            if file_stats:
                # Create definition summaries
                definition_summaries = [
                    self._create_definition_summary(defn)
                    for defn in file_stats['all_definitions'][:10]
                ]

                file_overviews.append({
                    **{k: v for k, v in file_stats.items() if k != 'content'},
                    "definition_summaries": definition_summaries
                })

        return file_overviews

    def _create_prioritization_prompt(self, file_overviews: List[Dict]) -> str:
        """Create prompt for file prioritization"""
        prompt = f"""Return a JSON ARRAY of ALL {len(file_overviews)} files with priority scores.

Assign a priority score (1-10, 10 being highest) using the following criteria in descending importance:
1. File modification age (older files get higher priority)
2. File size in KB (smaller files get higher priority)
3. Function definitions that match your role and selection criteria get higher priority

Files to prioritize:
"""

        for j, file_info in enumerate(file_overviews):
            prompt += f"\n--- File {j+1}: {file_info['path']} ---\n"
            prompt += f"Size: {file_info['size_kb']} KB\n"
            prompt += f"Last modified: {file_info['modified_age']} seconds ago\n"
            prompt += f"Total definitions: {file_info['definition_count']} ({file_info['function_count']} functions, {file_info['class_count']} classes)\n"

            if file_info['definition_summaries']:
                prompt += "Code definitions:\n"
                for def_summary in file_info['definition_summaries']:
                    prompt += f"  {def_summary}\n"
            else:
                prompt += "No function/class definitions found\n"

        prompt += f"""
Return a JSON ARRAY with ALL {len(file_overviews)} files like this:
[
  {{
    "file_path": "path/to/file1.py",
    "priority": 1-10,
    "reason": "Brief reason"
  }},
  {{
    "file_path": "path/to/file2.py",
    "priority": 1-10,
    "reason": "Brief reason"
  }}
  {', ...' if len(file_overviews) > 2 else ''}
]

IMPORTANT: Your response MUST be a valid JSON ARRAY starting with [ and ending with ], not a single object.
"""
        return prompt

    def _ensure_all_files_prioritized(self, prioritized_files: List[Dict], file_overviews: List[Dict]) -> List[Dict]:
        """Ensure all files from overviews are included in prioritized list with fallback priorities"""
        all_paths = {file_info["path"] for file_info in file_overviews}
        prioritized_paths = {file_info.get("file_path") for file_info in prioritized_files}

        # Add missing files with default priority
        for path in all_paths - prioritized_paths:
            self.logger.info(f"File missing from prioritized list: {path}, adding with default priority")
            prioritized_files.append({
                "file_path": path,
                "priority": 1,
                "reason": "Added by default (missing from LLM response)"
            })

        return sorted(prioritized_files, key=lambda x: x.get("priority", 0), reverse=True)

    def _prioritize_files(self, agent, file_overviews: List[Dict]) -> List[Dict]:
        """Use LLM to prioritize all files at once based on size, age, and function definitions"""
        self.logger.info(f"Prioritizing all {len(file_overviews)} files at once")

        prompt = self._create_prioritization_prompt(file_overviews)
        response = agent.chat_completion(prompt=prompt, system_message=agent.role)
        result = agent.parse_json_response(response)

        # Handle different response formats
        if isinstance(result, list):
            prioritized_files = result
        elif isinstance(result, dict) and "file_path" in result:
            # Single object response - create list and add remaining files
            prioritized_files = [result]
            for file_info in file_overviews:
                if file_info["path"] != result["file_path"]:
                    prioritized_files.append({
                        "file_path": file_info["path"],
                        "priority": 1,
                        "reason": "Added by default (not in LLM response)"
                    })
        else:
            # Invalid response - create default priorities
            self.logger.error("Invalid response format, using default priorities for all files")
            prioritized_files = [{
                "file_path": file_info["path"],
                "priority": 1,
                "reason": "Default priority (LLM response format error)"
            } for file_info in file_overviews]

        return self._ensure_all_files_prioritized(prioritized_files, file_overviews)

    def _apply_filters(self, agent, files: List[str]) -> List[str]:
        """Apply all file filters in sequence"""
        filters = [
            ("excluded directories", self._filter_excluded_dirs),
            ("excluded types", self._filter_excluded_types),
            ("flagged files", lambda f: self._filter_flagged_files(agent, f)),
            ("max limit", self._filter_max_limit)
        ]

        filtered_files = files
        for filter_name, filter_func in filters:
            filtered_files = filter_func(filtered_files)
            self.logger.info(f"Found {len(filtered_files)} files after {filter_name} filtering")

        return filtered_files

    def _filter_excluded_dirs(self, files: List[str]) -> List[str]:
        """Filter out files from excluded directories"""
        if not self.exclude_dirs:
            return files

        filtered_files = []
        for file_path in files:
            normalized_path = file_path.replace('\\', '/')
            exclude = any(
                f'/{excluded_dir_norm}/' in normalized_path or
                normalized_path.startswith(f'{excluded_dir_norm}/')
                for excluded_dir in self.exclude_dirs
                for excluded_dir_norm in [excluded_dir.replace('\\', '/')]
            )
            if not exclude:
                filtered_files.append(file_path)

        return filtered_files

    def _filter_excluded_types(self, files: List[str]) -> List[str]:
        """Filter out files with excluded file types"""
        if not self.exclude_types:
            return files

        return [f for f in files if not any(f.endswith(exclude_type) for exclude_type in self.exclude_types)]

    def _filter_flagged_files(self, agent, files: List[str]) -> List[str]:
        """Filter out files that are currently active or already finished"""
        if not files:
            return files

        filtered_files = []
        for file_path in files:
            is_active = agent.version_manager.check_active(file_path)
            is_finished = agent.version_manager.check_finished(agent, file_path)

            if is_active:
                self.logger.debug(f"Filtering out active file: {file_path}")
            elif is_finished:
                self.logger.debug(f"Filtering out finished file: {file_path}")
            else:
                filtered_files.append(file_path)

        return filtered_files

    def _filter_max_limit(self, files: List[str]) -> List[str]:
        """Filter out files if they exceed max file limit"""
        if len(files) > self.max_files:
            random.shuffle(files)
            return files[:self.max_files]
        return files

    def _create_batch_overview(self, batch_data: List[Dict]) -> str:
        """Create overview section for batch selection prompt"""
        overview = "File list:\n"
        for i, file_info in enumerate(batch_data):
            filename = os.path.basename(file_info['path'])
            def_count = len(file_info['all_definitions'])
            func_count = file_info['function_count']
            class_count = file_info['class_count']
            overview += (f"- File {i+1}: {filename} (Priority: {file_info['priority']}/10, "
                        f"{def_count} total: {func_count} functions, {class_count} classes) "
                        f"- {file_info['priority_reason']}\n")
        return overview

    def _create_definitions_overview(self, batch_data: List[Dict]) -> str:
        """Create definitions overview section for batch selection prompt"""
        overview = "\nCode definitions overview:"

        for i, file_info in enumerate(batch_data):
            overview += f"\n\n--- File {i+1}: {file_info['path']} ---"

            if not file_info['all_definitions']:
                overview += "\nNo function or class definitions found"
                continue

            functions = [d for d in file_info['all_definitions'] if d['type'] == 'function']
            classes = [d for d in file_info['all_definitions'] if d['type'] == 'class']

            for def_type, definitions in [("Functions", functions), ("Classes", classes)]:
                if definitions:
                    overview += f"\n{def_type} ({len(definitions)}):"
                    for defn in definitions[:3]:  # Limit to first 3
                        overview += f"\n  {defn['signature']}"
                        if defn['docstring']:
                            # Allow multiple lines but limit total characters to SUMMARY_LENGTH
                            docstring_lines = [line.strip() for line in defn['docstring'].split('\n') if line.strip()]
                            if docstring_lines:
                                docstring_text = ' '.join(docstring_lines)
                                overview += f" # {self._truncate_text(docstring_text)}"

                        if defn['type'] == 'class' and defn.get('methods'):
                            methods = defn['methods'][:5]
                            overview += f" (methods: {', '.join(methods)}{'...' if len(defn['methods']) > 5 else ''})"

                    if len(definitions) > 3:
                        overview += f"\n  ... and {len(definitions) - 3} more {def_type.lower()}"

        return overview

    def _create_selection_prompt(self, batch_data: List[Dict], current_batch: int, total_batches: int) -> str:
        """Create a concise prompt for file selection"""
        prompt = f"""Select the most relevant file based on your role and selection criteria: {self.selection_criteria}

Current batch: {current_batch+1}/{total_batches}
{self._create_batch_overview(batch_data)}
{self._create_definitions_overview(batch_data)}

Detailed file contents:"""

        for i, file_info in enumerate(batch_data):
            prompt += f"\n\n--- File {i+1}: {file_info['path']} ---\n{file_info['content']}"

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

    def _prepare_batch_data(self, file_paths: List[str], prioritized_files: List[Dict]) -> List[Dict]:
        """Prepare batch data with file content and metadata"""
        batch_data = []

        for file_path in file_paths:
            file_stats = self._get_file_stats(file_path)
            if not file_stats:
                continue

            # Find priority info for this file
            priority_info = next((f for f in prioritized_files if f["file_path"] == file_path), None)
            priority = priority_info.get("priority", 1) if priority_info else 1
            reason = priority_info.get("reason", "No reason provided") if priority_info else "No reason provided"

            batch_data.append({
                **file_stats,
                'priority': priority,
                'priority_reason': reason
            })

        return batch_data

    def _weighted_random_sample(self, file_paths: List[str], prioritized_files: List[Dict], batch_size: int) -> List[str]:
        """Sample files using weighted random sampling based on priorities"""
        priority_map = {f["file_path"]: f.get("priority", 1) for f in prioritized_files}
        weights = [priority_map.get(path, 1) for path in file_paths]
        sample_size = min(batch_size, len(file_paths))

        # Use random.choices for sampling without replacement
        sampled_indices = set()
        sampled_paths = []

        for _ in range(sample_size):
            available_indices = [i for i in range(len(file_paths)) if i not in sampled_indices]
            if not available_indices:
                break

            available_weights = [weights[i] for i in available_indices]
            total_weight = sum(available_weights)

            if total_weight == 0:
                chosen_idx = random.choice(available_indices)
            else:
                probabilities = [w/total_weight for w in available_weights]
                chosen_relative_idx = random.choices(range(len(available_indices)),
                    weights=probabilities)[0]
                chosen_idx = available_indices[chosen_relative_idx]

            sampled_indices.add(chosen_idx)
            sampled_paths.append(file_paths[chosen_idx])

        return sampled_paths

    def _process_single_batch(self, agent, batch_paths: List[str], prioritized_files: List[Dict],
                             batch_info: str) -> Optional[Dict]:
        """Process a single batch of files and return selected file if found"""
        self.logger.info(f"Processing {batch_info}: {len(batch_paths)} files")

        current_batch_data = self._prepare_batch_data(batch_paths, prioritized_files)
        if not current_batch_data:
            self.logger.info(f"No readable files in {batch_info}")
            return None

        # Get LLM selection
        prompt = self._create_selection_prompt(current_batch_data, 0, 1)  # Simplified for batch info
        response = agent.chat_completion(prompt=prompt, system_message=agent.role,
                                       response_format={"type": "json_object"})
        result = agent.parse_json_response(response)

        if not result or "error" in result:
            self.logger.error(f"Error parsing response for {batch_info}: {result.get('error', 'Unknown')}")
            return None

        selected_info = result.get('selected_file')
        if not selected_info:
            return None

        file_number = selected_info.get('file_number')
        if not (file_number and 1 <= file_number <= len(current_batch_data)):
            self.logger.error(f"Invalid file_number {file_number} for {batch_info}")
            return None

        file_data = current_batch_data[file_number - 1]
        selected_file = {
            'path': file_data['path'],
            'content': file_data['content'],
            'reason': selected_info.get('reason', 'Selected by LLM'),
            'priority': file_data['priority'],
            'all_definitions': file_data['all_definitions']
        }

        self.logger.info(f"Selected: {file_data['path']} - {selected_file['reason']}")
        return selected_file

    def _process_file_batches(self, agent, file_paths: List[str], prioritized_files: List[Dict]) -> Optional[Dict]:
        """Process batches of files to find the most relevant match"""
        self.logger.info(f"Processing batches using {'weighted sampling' if self.batch_sampling else 'sequential'} mode")

        if self.batch_sampling:
            remaining_files = file_paths.copy()
            batch_count = 0

            while remaining_files:
                batch_count += 1
                batch_paths = self._weighted_random_sample(remaining_files, prioritized_files, self.batch_size)

                # Remove sampled files
                for path in batch_paths:
                    remaining_files.remove(path)

                selected_file = self._process_single_batch(agent, batch_paths, prioritized_files, f"sampled batch {batch_count}")
                if selected_file:
                    return selected_file
        else:
            total_batches = (len(file_paths) + self.batch_size - 1) // self.batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(file_paths))
                batch_paths = file_paths[start_idx:end_idx]

                batch_info = f"batch {batch_idx + 1}/{total_batches}"
                selected_file = self._process_single_batch(agent, batch_paths, prioritized_files, batch_info)
                if selected_file:
                    return selected_file

        return None

    def _collect_all_files(self, agent) -> List[str]:
        """Collect all files matching the specified file types"""
        all_files = []

        for file_type in self.file_types:
            # Ensure file_type has a leading dot if needed
            if not file_type.startswith('.'):
                file_type = '.' + file_type

            search_path = os.path.join(agent.repository, f'**/*{file_type}')
            self.logger.info(f"Searching for files in: {search_path}")

            files = glob.glob(search_path, recursive=True)
            self.logger.info(f"Found {len(files)} {file_type} files")
            all_files.extend(files)

        return all_files

    def execute(self, agent, **kwargs) -> bool:
        """Execute the search action"""
        self.logger.info("Starting SearchAction execution")

        # Validate OpenAI handler
        if not (hasattr(agent, 'openai_handler') and agent.openai_handler is not None):
            error_msg = "Agent must have an openai_handler attribute initialized"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Collect and filter files
        all_files = self._collect_all_files(agent)
        self.logger.info(f"Found {len(all_files)} total files before filtering")

        filtered_files = self._apply_filters(agent, all_files)

        if not filtered_files:
            self.logger.error("No files left after filtering")
            return False

        # Create overview, prioritize, and process batches
        file_overviews = self._create_global_overview(agent, filtered_files)
        prioritized_files = self._prioritize_files(agent, file_overviews)

        file_paths = [file_info["file_path"] for file_info in prioritized_files]
        selected_file = self._process_file_batches(agent, file_paths, prioritized_files)

        if selected_file:
            # Save original file and update agent context
            agent.version_manager.save_original(selected_file['path'], selected_file['content'])
            agent.context['selected_file'] = selected_file
            agent.version_manager.flag_active(agent, selected_file['path'])
            self.logger.info(f"Search completed with selected file: {selected_file['path']}")
            return True
        else:
            self.logger.error("Search completed but no file was selected by agent.")
            return False
