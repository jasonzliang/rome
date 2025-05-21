import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..config import check_attrs
from ..logger import get_logger
from ..executor import CodeExecutor, CodeBlock

class ExecuteCodeAction(Action):
    """Action to execute test code for the selected file"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()
        check_attrs(self, ['executor_config'])

        # Initialize code executor
        self.executor = CodeExecutor(self.executor_config)

    def execute(self, agent, **kwargs) -> bool:
        """Execute tests for the current selected file"""
        self.logger.info("Starting ExecuteCodeAction execution")

        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        test_path = selected_file['test_path']

        # Ensure the test file exists
        # if not os.path.exists(test_path):
        #     self.logger.error(f"Test file does not exist: {test_path}")
        #     return False

        self.logger.info(f"Executing test file: {test_path}")

        try:
            # Determine the language based on file extension
            _, ext = os.path.splitext(test_path)
            # Execute the test file
            result = self.executor.execute_file(test_path)

        except Exception as e:
            error_msg = f"Error executing test file {test_path}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())

            # Add error to context
            selected_file['exec_output'] = error_msg
            selected_file['exec_exit_code'] = 1
            selected_file['exec_analysis'] = f"Test execution failed due to an error: {str(e)}"

            return False

        # Add execution results to selected_file context
        selected_file['exec_output'] = result.output
        selected_file['exec_exit_code'] = result.exit_code

        # Generate execution analysis
        analysis = self._analyze_execution_results(
            original_file_content=selected_file['content'],
            test_file_content=selected_file['test_content'],
            output=result.output,
            exit_code=result.exit_code
        )

        selected_file['exec_analysis'] = analysis

        self.logger.info(f"Test execution completed with exit code: {result.exit_code}")
        self.logger.info(f"Execution analysis: {analysis[:100]}...")  # Log first 100 chars

        # Return True if tests passed (exit code 0), False otherwise
        return result.exit_code == 0


    def _analyze_execution_results(self, agent,
                                  original_file_content: str,
                                  test_file_content: str,
                                  output: str,
                                  exit_code: int) -> str:
        """
        Analyze test execution results and provide feedback

        Args:
            original_file_content: Content of the original code file
            test_file_content: Content of the test file
            output: Output from test execution
            exit_code: Exit code from test execution

        Returns:
            Analysis of the execution results
        """
        # Prepare a prompt for the LLM to analyze the test results
        prompt = f"""Analyze the following test execution results and provide feedback:

Original Code File:
```python
{original_file_content}
```

Test File:
```python
{test_file_content}
```

Execution Output:
```
{output}
```

Exit Code: {exit_code}

Please provide a detailed analysis of:
1. Whether the tests passed or failed
2. If failed, what specific tests failed and why
3. Any errors or exceptions found in the output
4. Suggestions for fixing the issues

Your analysis:
"""
        # Get feedback from LLM
        # try:
        response = self.agent.chat_completion(
            prompt=prompt,
            system_message=agent.role
        )
        return response
#         except Exception as e:
#             # If LLM analysis fails, provide a simple analysis based on exit code
#             if exit_code == 0:
#                 return "Test execution completed successfully. All tests passed."
#             else:
#                 # Extract error messages from output
#                 error_lines = [line for line in output.split('\n')
#                               if 'error' in line.lower()
#                               or 'exception' in line.lower()
#                               or 'fail' in line.lower()]

#                 error_summary = '\n'.join(error_lines[:5])  # First 5 error lines

#                 return f"""Test execution failed with exit code {exit_code}.

# Key errors from output:
# {error_summary}

# Review the full output for more details.
# """
