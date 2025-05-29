import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..config import check_attrs
from ..logger import get_logger
from ..parsing import hash_string


def create_analysis_prompt(agent, file_path: str) -> Optional[str]:
    """
    Create analysis context for code editing prompts by loading execution and analysis data.

    Args:
        agent: Agent instance with version manager
        file_path: Path to the main file

    Returns:
        Formatted analysis context string or None if no data available
    """
    # Load the latest execution results
    execution_data = agent.version_manager.get_latest_data(file_path, 'exec_result')
    if not execution_data:
        return None
    context_parts = []

    # Add execution output
    if execution_data.get('output'):
        context_parts.append(f"Code test output:\n```\n{execution_data['output']}\n```")

    # Add exit code info
    if execution_data.get('exit_code'):
        context_parts.append(f"Exit code: {execution_data['exit_code']}")

    # Add analysis if available
    if execution_data.get('analysis'):
        context_parts.append(f"Code analysis:\n{execution_data['analysis']}")

    if not context_parts:
        return None

    context = "\n\n".join(context_parts)
    context += "\n\nIMPORTANT: Please take this analysis into account when improving the code or tests.\n"
    return context


class EditCodeAction(Action):
    """Base action class for editing and improving code files"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()
        check_attrs(self, ['custom_prompt'])

    def summary(self, agent) -> str:
        """Return a short summary of the code editing action"""
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        filename = os.path.basename(file_path)
        return f"analyze and improve code in {filename} focusing on bugs, performance, and readability"

    def execute(self, agent, **kwargs) -> bool:
        """Execute code editing action to improve the current selected file"""
        self.logger.info("Starting EditCodeAction execution")

        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        original_content = selected_file['content']

        # Initialize changes list if it doesn't exist
        if 'changes' not in selected_file:
            selected_file['changes'] = []

        # Prepare prompt for code improvement
        prompt = self._create_improvement_prompt(agent, file_path, original_content)

        # Get improved code from LLM
        self.logger.info(f"Requesting code improvements for {file_path}")

        response = agent.chat_completion(
            prompt=prompt,
            system_message=agent.role,
            response_format={"type": "json_object"}
        )

        # Parse the response to get improved code
        result = agent.parse_json_response(response)

        if not result or "improved_code" not in result:
            self.logger.error(f"Invalid response format from LLM: {response}")
            return False

        improved_code = result['improved_code']
        explanation = result.get('explanation', 'No explanation provided')
        changes = result.get('changes', [])

        # Validate the improved code
        if not improved_code or not isinstance(improved_code, str):
            self.logger.error("Improved code is missing or invalid")
            return False

        if improved_code == original_content:
            self.logger.info("No changes made to the code")
            # Even if no changes, we'll continue to the next state

        # Update the selected file in the agent context
        selected_file['content'] = improved_code

        # Record the explanation and changes
        change_record = {
            'explanation': explanation,
            'changes': changes
        }
        selected_file['change_record'] = change_record

        # Store code editing session in TinyDB
        # edit_data = {
        #     'original_content_hash': hash_string(original_content),
        #     'improved_content_hash': hash_string(improved_code),
        #     'changes': changes,
        #     'explanation': explanation,
        #     'agent_id': agent.get_id(),
        #     'content_changed': improved_code != original_content
        # }
        # agent.version_manager.store_data(file_path, 'code_edits', edit_data)

        # Write the improved code back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(improved_code)
        self.logger.info(f"Successfully wrote improved code to {file_path}")

        # Save version using agent's version manager
        version_number = agent.version_manager.save_version(
            file_path=file_path,
            content=improved_code,
            changes=changes,
            explanation=explanation)

        self.logger.info(f"Code editing completed for {file_path}")
        self.logger.info(f"Changes made: {changes}")

        return True

    def _create_improvement_prompt(self, agent, file_path: str, content: str) -> str:
        """Create a prompt for the LLM to improve the code"""

        # Use custom prompt if provided in config
        if self.custom_prompt is not None:
            prompt = self.custom_prompt
        else:
            # Base prompt without relying on a configured improvement_prompt
            prompt = """Analyze the code file and suggest improvements. Focus on:
1. Implementing missing code and filling in empty functions
2. Code quality and readability
3. Bug fixes and edge cases
4. Performance optimizations
5. Documentation improvements
"""

        # Get analysis context using the static function
        analysis_prompt = create_analysis_prompt(agent, file_path)
        if analysis_prompt:
            prompt += f"\n{analysis_prompt}\n"

        # Add file info and original code
        prompt += f"""Code file path: {file_path}
Code file content:
```python
{content}
```

Respond with a JSON object containing:
{{
    "improved_code": "The complete improved code as a string, including all original code with your improvements",
    "explanation": "A clear explanation of the changes you made and why",
    "changes": [
        {{
            "type": "improvement type (bug fix, performance, readability, etc.)",
            "description": "Description of the specific change"
        }},
        ...
    ]
}}

IMPORTANT:
- Return the ENTIRE file content with your improvements, not just the changed parts
- Make sure the improved code is valid Python syntax
- Be conservative with changes - prioritize correctness over style
"""

        if analysis_prompt:
            prompt += "- Pay special attention to addressing any issues identified in the code analysis"

        return prompt
