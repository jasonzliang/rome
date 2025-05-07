# agent_actions.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
import os
import glob


class Action(ABC):
    """Abstract base class for all actions"""

    def __init__(self, config: Dict = None):
        """Initialize the action with a configuration dictionary"""
        self.config = config or {}

    @abstractmethod
    def execute(self, agent, **kwargs):
        """Execute the action and return the next state"""
        pass


class SearchAction(Action):
    """Action to search the repository for code files"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        # Set default configuration for search action
        self.max_files = self.config.get('max_files', 100)
        self.include_content = self.config.get('include_content', True)
        self.depth = self.config.get('depth', 3)
        self.exclude_dirs = self.config.get('exclude_dirs',
                                           ['.git', 'node_modules', 'venv', '__pycache__', 'dist', 'build'])

    def execute(self, agent, **kwargs):
        from agent_fsm import AgentState

        query = kwargs.get('query', '')
        file_pattern = kwargs.get('file_pattern', '*.py')

        try:
            # Get the repository root path from agent config
            repo_path = agent.config.get('repository', {}).get('path', '.')
            search_path = os.path.join(repo_path, '**', file_pattern)

            # Find all files matching the pattern
            files = glob.glob(search_path, recursive=True)

            # Filter out files from excluded directories
            filtered_files = []
            for file_path in files:
                # Check if file is in an excluded directory
                exclude = False
                for excluded_dir in self.exclude_dirs:
                    if f'/{excluded_dir}/' in file_path or file_path.startswith(f'{excluded_dir}/'):
                        exclude = True
                        break

                if not exclude:
                    filtered_files.append(file_path)

            # Limit the number of files if specified
            if self.max_files and len(filtered_files) > self.max_files:
                agent.logger.warning(f"Limited search results to {self.max_files} files")
                filtered_files = filtered_files[:self.max_files]

            # If a query is provided, filter files by searching their content
            if query:
                matching_files = []
                for file_path in filtered_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if query.lower() in content.lower():
                                matching_files.append({
                                    'path': file_path,
                                    'content': content if self.include_content else None
                                })
                    except Exception as e:
                        agent.logger.error(f"Error reading file {file_path}: {str(e)}")

                agent.context['search_results'] = matching_files
            else:
                # Just collect file paths if no query provided
                agent.context['search_results'] = [{'path': f, 'content': None} for f in filtered_files]

            return AgentState.ANALYZING

        except Exception as e:
            agent.logger.error(f"Search action failed: {str(e)}")
            agent.context['error'] = str(e)
            return AgentState.ERROR


class AnalyzeAction(Action):
    """Action to analyze code using LLM"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        # Set default configuration for analyze action
        self.default_prompt = self.config.get('default_prompt',
            "Analyze the following code file and provide a summary:")
        self.model = self.config.get('model', 'gpt-4')
        self.temperature = self.config.get('temperature', 0.2)
        self.max_tokens = self.config.get('max_tokens', 2000)
        self.use_context = self.config.get('use_context', True)

    def execute(self, agent, **kwargs):
        from agent_fsm import AgentState

        try:
            search_results = agent.context.get('search_results', [])

            if not search_results:
                agent.logger.warning("No files to analyze")
                return AgentState.IDLE

            analysis_results = []

            # Only analyze files with content loaded
            for file_info in search_results:
                if file_info.get('content'):
                    analysis = self._analyze_with_llm(
                        agent=agent,
                        file_content=file_info['content'],
                        file_path=file_info['path'],
                        **kwargs
                    )
                    analysis_results.append({
                        'path': file_info['path'],
                        'analysis': analysis
                    })

            agent.context['analysis_results'] = analysis_results
            return AgentState.IDLE

        except Exception as e:
            agent.logger.error(f"Analysis action failed: {str(e)}")
            agent.context['error'] = str(e)
            return AgentState.ERROR

    def _analyze_with_llm(self, agent, file_content: str, file_path: str, **kwargs) -> str:
        """Use LLM to analyze the code file"""
        prompt = kwargs.get('prompt', f"{self.default_prompt}\n\n{file_content}")

        # Add context from previous analyses if enabled
        if self.use_context and 'analysis_results' in agent.context:
            context_str = "Based on previous analyses:\n"
            for prev_analysis in agent.context.get('analysis_results', [])[:3]:  # Limit to 3 previous analyses
                context_str += f"- File {prev_analysis['path']}: {prev_analysis['analysis'][:200]}...\n"
            prompt = f"{context_str}\n\n{prompt}"

        response = agent.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a code analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content


class UpdateAction(Action):
    """Action to update a code file using LLM"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        # Set default configuration for update action
        self.create_backup = self.config.get('create_backup', True)
        self.model = self.config.get('model', 'gpt-4')
        self.temperature = self.config.get('temperature', 0.2)
        self.max_tokens = self.config.get('max_tokens', 4000)
        self.preview_changes = self.config.get('preview_changes', True)

        # Get update prompt template
        self.update_prompt_template = self.config.get('update_prompt_template', """
        You are tasked with updating the following {file_type} code according to these instructions:

        INSTRUCTIONS:
        {instructions}

        ORIGINAL CODE:
        ```
        {original_content}
        ```

        Please provide ONLY the updated code without any explanations or markdown formatting.
        """)

    def execute(self, agent, **kwargs):
        from agent_fsm import AgentState

        file_path = kwargs.get('file_path')
        instructions = kwargs.get('instructions', '')

        if not file_path:
            agent.logger.error("No file path provided for update")
            agent.context['error'] = "No file path provided for update"
            return AgentState.ERROR

        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Generate updated content with LLM
            updated_content = self._update_with_llm(
                agent=agent,
                original_content=original_content,
                instructions=instructions,
                file_path=file_path
            )

            # If preview is enabled, store the changes for review
            if self.preview_changes:
                agent.context['update_preview'] = {
                    'file_path': file_path,
                    'original': original_content,
                    'updated': updated_content,
                    'instructions': instructions
                }
                agent.logger.info(f"Update preview generated for {file_path}")

                # If we're just previewing, return without making changes
                if kwargs.get('preview_only', False):
                    agent.context['update_result'] = {
                        'file_path': file_path,
                        'success': True,
                        'preview_only': True
                    }
                    return AgentState.IDLE

            # Create backup if enabled
            if self.create_backup:
                backup_dir = agent.config.get('safety', {}).get('backup_dir', './backups')
                os.makedirs(backup_dir, exist_ok=True)

                file_name = os.path.basename(file_path)
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"{file_name}.{timestamp}.bak")

                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                agent.logger.info(f"Created backup at {backup_path}")

            # Write the updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            agent.context['update_result'] = {
                'file_path': file_path,
                'success': True,
                'backup_path': backup_path if self.create_backup else None
            }

            return AgentState.IDLE

        except Exception as e:
            agent.logger.error(f"Update action failed: {str(e)}")
            agent.context['error'] = str(e)
            agent.context['update_result'] = {
                'file_path': file_path,
                'success': False,
                'error': str(e)
            }
            return AgentState.ERROR

    def _update_with_llm(self, agent, original_content: str, instructions: str, file_path: str) -> str:
        """Use LLM to update the code file"""
        file_extension = os.path.splitext(file_path)[1]
        file_type = "Python" if file_extension == '.py' else f"{file_extension[1:]} file"

        # Format the prompt template
        prompt = self.update_prompt_template.format(
            file_type=file_type,
            instructions=instructions,
            original_content=original_content
        )

        response = agent.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a code modification assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        updated_content = response.choices[0].message.content

        # Remove markdown code blocks if present
        if updated_content.startswith("```") and updated_content.endswith("```"):
            lines = updated_content.split("\n")
            if len(lines) > 2:
                # Remove first and last line if they're markdown fences
                if lines[0].startswith("```") and lines[-1].startswith("```"):
                    updated_content = "\n".join(lines[1:-1])

        return updated_content


# Add additional action classes as needed
class LoadFileAction(Action):
    """Action to load a file's content"""

    def __init__(self, config: Dict = None):
        super().__init__(config)

    def execute(self, agent, **kwargs):
        from agent_fsm import AgentState

        file_path = kwargs.get('file_path')
        if not file_path:
            agent.logger.error("No file path provided for loading")
            agent.context['error'] = "No file path provided for loading"
            return AgentState.ERROR

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            agent.context['loaded_file'] = {
                'path': file_path,
                'content': content
            }

            return AgentState.IDLE

        except Exception as e:
            agent.logger.error(f"Load file action failed: {str(e)}")
            agent.context['error'] = str(e)
            return AgentState.ERROR


class SaveFileAction(Action):
    """Action to save content to a file"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.create_backup = self.config.get('create_backup', True)

    def execute(self, agent, **kwargs):
        from agent_fsm import AgentState

        file_path = kwargs.get('file_path')
        content = kwargs.get('content')

        if not file_path:
            agent.logger.error("No file path provided for saving")
            agent.context['error'] = "No file path provided for saving"
            return AgentState.ERROR

        if content is None:
            agent.logger.error("No content provided for saving")
            agent.context['error'] = "No content provided for saving"
            return AgentState.ERROR

        try:
            # Create backup if enabled and file exists
            if self.create_backup and os.path.exists(file_path):
                backup_dir = agent.config.get('safety', {}).get('backup_dir', './backups')
                os.makedirs(backup_dir, exist_ok=True)

                file_name = os.path.basename(file_path)
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"{file_name}.{timestamp}.bak")

                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()

                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                agent.logger.info(f"Created backup at {backup_path}")

            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Save the content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            agent.context['save_result'] = {
                'file_path': file_path,
                'success': True
            }

            return AgentState.IDLE

        except Exception as e:
            agent.logger.error(f"Save file action failed: {str(e)}")
            agent.context['error'] = str(e)
            agent.context['save_result'] = {
                'file_path': file_path,
                'success': False,
                'error': str(e)
            }
            return AgentState.ERROR


# Registration function to easily get all available actions
def get_available_actions():
    """Return a dictionary of all available actions"""
    return {
        "search": SearchAction,
        "analyze": AnalyzeAction,
        "update": UpdateAction,
        "load_file": LoadFileAction,
        "save_file": SaveFileAction
    }
