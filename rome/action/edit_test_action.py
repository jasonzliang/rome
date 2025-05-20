import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..logger import get_logger


class EditTestAction(Action):
    """Action to create or edit tests for the selected file"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()

        # custom_prompt will be automatically set from config by set_attributes_from_config in parent class

    def execute(self, agent, **kwargs) -> bool:
        """Execute test editing action to create or improve tests for the current selected file"""
        self.logger.info("Starting EditTestAction execution")

        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        file_content = selected_file['content']

        # Determine test file path
        base_path, extension = os.path.splitext(file_path)
        test_path = f"{base_path}.test.py"

        # Initialize changes and test-related fields if they don't exist
        if 'changes' not in selected_file:
            selected_file['changes'] = []

        if 'test_changes' not in selected_file:
            selected_file['test_changes'] = []

        # Check if test file already exists
        test_exists = os.path.exists(test_path)
        test_content = ""

        if test_exists:
            try:
                with open(test_path, 'r', encoding='utf-8') as f:
                    test_content = f.read()
                self.logger.info(f"Existing test file found: {test_path}")
            except Exception as e:
                self.logger.error(f"Error reading test file {test_path}: {e}, overwriting")
        else:
            self.logger.info(f"No existing test file found. Will create new test file at: {test_path}")

        # Prepare prompt for test creation/improvement
        prompt = self._create_test_prompt(file_path, file_content, test_path,
            test_content, test_exists)

        # Get improved tests from LLM
        self.logger.info(f"Requesting test {'improvements' if test_exists else 'creation'} for {file_path}")

        response = agent.chat_completion(
            prompt=prompt,
            system_message=agent.role,
            response_format={"type": "json_object"}
        )

        # Parse the response to get test code
        result = agent.parse_json_response(response)

        if not result or "test_code" not in result:
            self.logger.error(f"Invalid response format from LLM: {response}")
            return False

        new_test_code = result.get('test_code')
        explanation = result.get('explanation', 'No explanation provided')
        test_changes = result.get('changes', [])

        # Validate the test code
        if not new_test_code or not isinstance(new_test_code, str):
            self.logger.error("Test code is missing or invalid")
            return False

        if test_exists and new_test_code == test_content:
            self.logger.info("No changes made to the test code")
            # Even if no changes, we'll continue to the next state

        # Update the selected file in the agent context with test information
        selected_file['test_path'] = test_path
        selected_file['test_content'] = new_test_code

        # Record the explanation and changes
        change_record = {
            'explanation': explanation,
            'changes': test_changes
        }
        selected_file['test_changes'].append(change_record)

        # Write the test code to disk
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(test_path), exist_ok=True)

            # Write the test file
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write(new_test_code)
            self.logger.info(f"Successfully wrote test code to {test_path}")
        except Exception as e:
            self.logger.error(f"Failed to write test code to {test_path}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

        self.logger.info(f"Test editing completed for {file_path}")
        self.logger.info(f"Test file: {test_path}")

        return True

    def _create_test_prompt(self, file_path: str, file_content: str, test_path: str, test_content: str, test_exists: bool) -> str:
        """Create a prompt for the LLM to create or improve tests"""

        # Extract file and module information for proper imports
        file_name = os.path.basename(file_path)
        module_name, _ = os.path.splitext(file_name)

        # Get directory structure for proper relative import
        file_dir = os.path.dirname(file_path)
        rel_path = os.path.relpath(file_dir, os.path.dirname(test_path))
        import_path = rel_path.replace(os.sep, '.') if rel_path != '.' else ''
        if import_path:
            import_statement = f"from {import_path} import {module_name}"
        else:
            import_statement = f"from . import {module_name}"

        # Use custom prompt if provided in config
        if self.custom_prompt is not None:
            base_prompt = self.custom_prompt
        else:
            # Base prompt
            base_prompt = """Create comprehensive unit tests for the provided code file. Focus on:
1. Testing all public methods and functions
2. Edge cases and error conditions
3. Integration between components
4. Good test organization and documentation
"""

        action_type = "improve the existing" if test_exists else "create new"

        prompt = f"""{base_prompt}
I need you to {action_type} unit tests for the following Python file:
File path: {file_path}
Code to test:
```python
{file_content}
```
"""
        if test_exists:
            prompt += f"""Current test file ({test_path}):
```python
{test_content}
```
"""
        else:
            prompt += f"""Create a new test file that will be saved at: {test_path}
Include proper test setup, all necessary imports, and comprehensive test cases.
"""

        prompt += f"""Respond with a JSON object containing:
{{
    "test_code": "The complete test code as a string",
    "explanation": "A clear explanation of the test approach and coverage",
    "changes": [
        {{
            "type": "test type (unit test, integration test, etc.)",
            "description": "Description of what this test verifies"
        }},
        ...
    ]
}}

IMPORTANT:
- Make sure the test code is valid Python syntax
- Tests should be compatible with pytest and be sure to do "import pytest" first
- Use RELATIVE IMPORTS to import the module being tested, like: `{import_statement}`
- Avoid absolute imports that would break in different environments
- Include all necessary imports that are required by the tests.
- Avoid external dependencies other than standard test libraries
"""

        return prompt
