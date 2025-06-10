import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..config import check_attrs
from ..logger import get_logger
from ..parsing import hash_string


def create_analysis_prompt(agent, file_path: str) -> Optional[str]:
    """Create analysis context for code editing prompts by loading execution and analysis data."""

    # Load the latest execution results
    execution_data = agent.version_manager.get_data(file_path, 'exec_result')

    context_parts = []

    # Add execution output if available
    if execution_data:
        # Get the agent ID from execution data, fallback to current agent ID
        exec_agent_id = execution_data['agent_id']
        current_agent_id = agent.get_id()

        context_parts.append(f"# Execution output:\n```\n{execution_data['exec_output']}\n```")
        context_parts.append(f"# Exit code: {execution_data['exec_exit_code']}")
        context_parts.append(f"# Execution analysis:\n{execution_data['exec_analysis']}")

        # Add note if analysis was done by a different agent
        if exec_agent_id != current_agent_id:
            context_parts.append(f"# Note: The above analysis was performed by a different agent ({exec_agent_id}). Please double-check and verify the analysis results.")

    if not context_parts:
        return None

    # Create a clearly separated analysis section
    analysis_context = "\n---\n"
    analysis_context += "Previous code/test execution results and analysis:\n"
    analysis_context += "---\n\n"
    analysis_context += "\n\n".join(context_parts)
    analysis_context += "\n---\n"
    analysis_context += "Please address any issues identified above\n"
    analysis_context += "---\n"

    return analysis_context


class EditCodeAction(Action):
    """Base action class for editing and improving code files"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()

    def summary(self, agent) -> str:
        """Return a short summary of the code editing action"""
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        filename = os.path.basename(file_path)
        return f"analyze and improve code in {filename}, addressing execution issues, bugs, performance, and readability"

    def execute(self, agent, **kwargs) -> bool:
        """Execute code editing action to improve the current selected file"""

        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        original_content = selected_file['content']

        # Prepare prompt for code improvement
        prompt = self._create_code_prompt(agent, file_path, original_content)

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

        improved_code = result.get('improved_code')
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

        # Write the improved code back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(improved_code)

        # Note: Version saving is now handled in ExecuteCodeAction to include execution results
        self.logger.info(f"Code editing completed for {file_path}")
        self.logger.info(f"Successfully edited and wrote improved code to {file_path}")
        return True

    def _create_code_prompt(self, agent, file_path: str, content: str) -> str:
        """Create a prompt for the LLM to improve the code"""

        # Base prompt without relying on a configured improvement_prompt
        prompt = """Analyze the code file and suggest improvements. Focus on:
    1. Implementing missing code and filling in empty functions
    2. Code quality and readability
    3. Bug fixes and edge cases
    4. Performance optimizations
    5. Documentation improvements
    """

        # Load original version content
        original_version = agent.version_manager.load_original_version(file_path, include_content=True)

        # Add original version in its own section if available
        if original_version and original_version.get('content'):
            prompt += f"""
    # Original file content:
    ```python
    {original_version['content']}
    ```
    """

        # Add current file info and code
        prompt += f"""
    # Current code file path: {file_path}
    # Current code file content:
    ```python
    {content}
    ```
    """
        # Get analysis context using the static function
        analysis_prompt = create_analysis_prompt(agent, file_path)
        if analysis_prompt:
            prompt += f"\n{analysis_prompt}\n"

        prompt += """
    Respond with a JSON object containing:
    {{
        "improved_code": "The new and improved code with changes or edits made to it",
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
    - Make sure the improved code is valid Python syntax and contains no markdown formatting like ```python...```
    - Be conservative with changes - prioritize correctness over style
    - List improvements you made in "changes" and summarize the changes in "explanation"
    - If improved code is unchanged, be sure give an empty list for "changes" and mention it in "explanation"
    """

        if analysis_prompt:
            prompt += "- Pay special attention to addressing any issues identified in the code analysis"

        if original_version and original_version.get('content'):
            prompt += "\n- Compare current code with the original version to understand the evolution and make informed improvements"

        return prompt
