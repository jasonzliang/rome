import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from .edit_code_action import create_analysis_prompt
from ..config import check_attrs
from ..logger import get_logger
from ..parsing import hash_string

class EditTestAction2(Action):
    """Action to create or edit tests for the selected file - now inherits from Action"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()
        check_attrs(self, ['custom_prompt'])

    def summary(self, agent) -> str:
        """Return a short summary of the test editing action"""
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        filename = os.path.basename(file_path)
        test_exists = 'test_path' in selected_file and \
            os.path.exists(selected_file.get('test_path', ''))
        action_type = "update existing tests" if test_exists else "create comprehensive unit tests"
        return f"{action_type} for {filename}, covering edge cases, error conditions, and test failures using a different prompt modified by LLM"

    def execute(self, agent, **kwargs) -> bool:
        """Execute test editing action to create or improve tests for the current selected file"""
        self.logger.info("Starting EditTestAction execution")

        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        file_content = selected_file['content']

        # Determine test file path
        base_path, extension = os.path.splitext(file_path)
        test_path = f"{base_path}_test.py"

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
        prompt = self._create_test_prompt(agent, file_path, file_content, test_path,
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

        if not result or "improved_test_code" not in result:
            self.logger.error(f"Invalid response format from LLM: {response}")
            return False

        improved_test_content = result.get('improved_test_code')
        explanation = result.get('explanation', 'No explanation provided')
        test_changes = result.get('changes', [])

        # Validate the test code
        if not improved_test_content or not isinstance(improved_test_content, str):
            self.logger.error("Test code is missing or invalid")
            return False

        if test_exists and improved_test_content == test_content:
            self.logger.error("No changes made to the test code")
            # Even if no changes, we'll continue to the next state

        # Update the selected file in the agent context with test information
        selected_file['test_path'] = test_path
        selected_file['test_content'] = improved_test_content

        # Record the explanation and changes
        change_record = {
            'explanation': explanation,
            'changes': test_changes
        }
        selected_file['test_changes'].append(change_record)

        # Write the test code to disk
        with open(test_path, 'w', encoding='utf-8') as f:
            f.write(improved_test_content)

        # Note: Version saving is now handled in ExecuteCodeAction to include execution results
        self.logger.info(f"Successfully edited and wrote test code to {test_path}")
        self.logger.info(f"Test editing completed for {file_path}")
        return True

    def _create_test_prompt(self, agent, file_path: str, file_content: str, test_path: str,
                           test_content: str, test_exists: bool) -> str:
        """Create a structured prompt for comprehensive test creation using evidence-based strategies"""

        # Extract file and module information for proper imports
        file_name = os.path.basename(file_path)
        module_name, _ = os.path.splitext(file_name)

        # Use custom prompt if provided in config
        if self.custom_prompt is not None:
            base_prompt = self.custom_prompt
        else:
            action_type = "improve existing" if test_exists else "create comprehensive"
            base_prompt = f"""Analyze the code and {action_type} unit tests using this approach:

1. **Identify what to test**: Functions, methods, edge cases, error conditions
2. **Extract requirements**: Analyze docstrings for behavioral specs and constraints
3. **Plan coverage**: Normal cases, boundaries, exceptions, integration points"""

            if test_exists:
                base_prompt += """
4. **Evaluate existing tests**: Find coverage gaps and quality issues"""
            else:
                base_prompt += """
4. **Design test structure**: Organize by function/class with clear naming"""

        prompt = f"""{base_prompt}

# Source Code Information
**File Path**: {file_path}
**Module Name**: {module_name}
**Source Code**:
```python
{file_content}
```
"""

        if test_exists:
            prompt += f"""
# Existing Test Information
**Test File Path**: {test_path}
**Current Test Code**:
```python
{test_content}
```
"""
        else:
            prompt += f"""
# Test File Creation
**Target Test Path**: {test_path}
**Requirements**: Create a new comprehensive test file with proper imports and test structure
"""

        # Get analysis context using the static function
        analysis_prompt = create_analysis_prompt(agent, file_path)
        if analysis_prompt:
            prompt += f"\n{analysis_prompt}\n"
            prompt += "\n**Address execution issues**: Focus on creating tests that prevent the reported errors.\n"

        prompt += """
## Requirements
Write comprehensive tests covering:
- **Functionality**: All public methods with normal inputs
- **Edge Cases**: Boundaries, empty inputs, extreme values
- **Errors**: Invalid inputs and exception handling
- **Integration**: Component interactions where applicable

Respond with JSON:
{
    "improved_test_code": "Complete test file with comprehensive coverage",
    "explanation": "Key testing decisions and coverage strategy",
    "changes": [
        {
            "type": "test_category",
            "description": "What this test verifies",
            "reasoning": "Why this test matters"
        }
    ]
}

## Critical Technical Requirements
1. **Complete Test File**: Return the ENTIRE test file content with all tests included
2. **Pytest Compatibility**: Use pytest conventions and include `import pytest` at the top
3. **Proper Imports**: Import the module under test correctly using `from {module_name} import *`
4. **Valid Python**: Ensure all syntax is correct and tests will execute without errors
5. **No Markdown**: Do not include ```python``` code block formatting in the improved_test_code field
6. **Clear Organization**: Group related tests together with logical structure and helpful comments
7. **File Locations**: Assume the code and test files are in the root of the current working directory
8. **Accountability**: If improved test code is unchanged, mention it in "explanation" and "changes"
"""
        return prompt
