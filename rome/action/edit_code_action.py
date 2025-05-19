import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..logger import get_logger


class EditCodeAction(Action):
    """Action to edit and improve code in the selected file"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()

        # custom_prompt will be automatically set from config by set_attributes_from_config in parent class

    def execute(self, agent, **kwargs) -> bool:
        """Execute code editing action to improve the current selected file"""
        self.logger.info("Starting EditCodeAction execution")

        # Ensure agent has a valid selected file
        # if not agent.context.get('selected_file'):
        #     self.logger.error("No selected file found in agent context")
        #     return False

        selected_file = agent.context['selected_file']

        # Ensure the selected file has required attributes
        # required_keys = ['path', 'content']
        # for key in required_keys:
        #     if key not in selected_file:
        #         self.logger.error(f"Missing {key} in selected file")
        #         return False

        file_path = selected_file['path']
        original_content = selected_file['content']

        # Initialize changes list if it doesn't exist
        if 'changes' not in selected_file:
            selected_file['changes'] = []

        # Prepare prompt for code improvement
        prompt = self._create_improvement_prompt(file_path, original_content)

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
        selected_file['changes'].append(change_record)

        # Write the improved code back to the file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(improved_code)
            self.logger.info(f"Successfully wrote improved code to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to write improved code to {file_path}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

        self.logger.info(f"Code editing completed for {file_path}")
        self.logger.info(f"Changes made: {changes}")

        return True

    def _create_improvement_prompt(self, file_path: str, content: str) -> str:
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

        # Add file info and original code
        prompt += f"""Code file path: {file_path}
Code file contents:
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

        return prompt
