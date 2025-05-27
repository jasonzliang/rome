import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..config import SUMMARY_LENGTH, check_attrs
from ..logger import get_logger
from ..executor import CodeExecutor, CodeBlock

class ExecuteCodeAction(Action):
    """Action to execute test code for the selected file"""

    def __init__(self, config: Dict = None, executor_config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()
        self.executor_config = executor_config
        # check_attrs(self, ['executor_config'])

        # Initialize code executor
        self.executor = CodeExecutor(self.executor_config)

    def summary(self, agent) -> str:
        """Return a short summary of the code execution action"""
        selected_file = agent.context['selected_file']
        test_path = selected_file['test_path']
        test_filename = os.path.basename(test_path)
        return f"execute {test_filename} and analyze results for failures or issues"

    def execute(self, agent, **kwargs) -> bool:
        """Execute tests for the current selected file"""
        self.logger.info("Starting ExecuteCodeAction execution")

        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        test_path = selected_file['test_path']

        self.logger.info(f"Executing test file: {test_path}")

        try:
            # Execute the test file
            result = self.executor.execute_file(test_path)

            # Store execution results in context
            selected_file['exec_output'] = result.output
            selected_file['exec_exit_code'] = result.exit_code

        except Exception as e:
            error_msg = f"Error executing test file {test_path}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())

            # Store error information in context
            selected_file['exec_output'] = error_msg
            selected_file['exec_exit_code'] = 1
            selected_file['exec_analysis'] = f"Test execution failed due to an error: {str(e)}"

            return False

        # Generate execution analysis using version manager
        analysis = agent.version_manager.create_analysis(
            agent,
            original_file_content=selected_file['content'],
            test_file_content=selected_file['test_content'],
            output=result.output,
            exit_code=result.exit_code
        )

        # Store analysis in context
        selected_file['exec_analysis'] = analysis

        # Save analysis using agent's version manager
        analysis_file_path = agent.version_manager.save_analysis(
            file_path=file_path,
            analysis=analysis,
            test_path=test_path,
            exit_code=result.exit_code,
            output=result.output
        )
        self.logger.info(f"Analysis saved to: {analysis_file_path}")

        self.logger.info(f"Test execution completed with exit code: {result.exit_code}")
        self.logger.info(f"Analysis summary: {analysis[:SUMMARY_LENGTH]}...")

        # Return True if tests passed (exit code 0), False otherwise
        return result.exit_code == 0
