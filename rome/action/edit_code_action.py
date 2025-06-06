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
    if not execution_data:
        return None

    context_parts = []

    # Add execution output
    context_parts.append(f"# Test Execution Output:\n```\n{execution_data['exec_output']}\n```")

    # Add exit code info
    context_parts.append(f"# Exit Code: {execution_data['exec_exit_code']}")

    # Add analysis if available
    context_parts.append(f"# Automated Code Analysis:\n{execution_data['exec_analysis']}")

    if not context_parts:
        return None

    # Create a clearly separated analysis section
    analysis_context = "\n---\n"
    analysis_context += "Previous execution results and analysis:\n"
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
        check_attrs(self, ['custom_prompt'])

    def summary(self, agent) -> str:
        """Return a short summary of the code editing action"""
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        filename = os.path.basename(file_path)
        return f"analyze and improve code in {filename}, addressing execution issues, bugs, performance, and readability"

    def execute(self, agent, **kwargs) -> bool:
        """Execute code editing action to improve the current selected file"""
        self.logger.info("Starting EditCodeAction execution")

        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        original_content = selected_file['content']

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

    def _create_improvement_prompt(self, agent, file_path: str, content: str) -> str:
        """Create a structured prompt for the LLM to improve the code using evidence-based strategies"""

        # Use custom prompt if provided in config
        if self.custom_prompt is not None:
            prompt = self.custom_prompt
        else:
            # Enhanced prompt using structured analysis strategies
            prompt = """Analyze and improve the code using this systematic approach:

1. **Understand the code**: Purpose, core logic, dependencies, current state
2. **Identify issues**: Syntax errors, logic flaws, performance problems, missing implementations
3. **Extract requirements**: Analyze docstrings/comments for constraints and expected behavior
4. **Plan improvements**: Prioritize fixes - correctness, completeness, robustness, performance, readability"""

        # Add file info and original code
        prompt += f"""
# Code File Information
**File Path**: {file_path}
**File Content**:
```python
{content}
```
"""

        # Get analysis context using the static function
        analysis_prompt = create_analysis_prompt(agent, file_path)
        if analysis_prompt:
            prompt += f"\n{analysis_prompt}\n"
            prompt += "\n**Address execution issues**: Fix errors identified in test results above, including any execution errors and test failures.\n"

        prompt += """
Focus improvements on: correctness, completeness, robustness, performance, readability.

Respond with JSON:
{
    "improved_code": "Complete improved file content",
    "explanation": "Summary of changes made and reasoning",
    "changes": [
        {
            "type": "change_category",
            "description": "Specific change made",
            "reasoning": "Why this change was necessary"
        }
    ]
}

## Critical Technical Requirements
1. **Complete File**: Return the ENTIRE file content with improvements, not just changed sections
2. **Valid Python**: Ensure all syntax is correct, imports are proper, and code will execute
3. **No Formatting**: Do not include markdown code blocks (```python) in the improved_code field
4. **Preserve Structure**: Maintain existing class/function organization unless restructuring is necessary
5. **Consistent Style**: Follow existing code style and naming conventions where appropriate
6. **Accountability**: If improved code is unchanged, mention it in "explanation" and "changes"
"""
        return prompt