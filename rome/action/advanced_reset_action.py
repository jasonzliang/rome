import glob
import os
import sys
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..logger import get_logger

class AdvancedResetAction(Action):
    """Advanced reset action that analyzes execution results and manages file flags appropriately"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = get_logger()

    def summary(self, agent) -> str:
        """Return a short summary of the advanced reset action"""
        return ("analyze test execution results using LLM to determine if code/tests are correct, "
                "mark code/tests as finished if correct, and return to initial state "
                "to start a new coding task")

    def _analyze_execution_results(self, agent, selected_file: Dict) -> bool:
        """
        Use LLM to analyze execution results and determine if work is complete

        Args:
            agent: Agent instance with LLM capabilities
            selected_file: Selected file context with execution results

        Returns:
            True if LLM determines work is complete and correct, False otherwise
        """
        try:
            execution_data = agent.version_manager.get_data(selected_file['path'], 'exec_result')
            exit_code = execution_data['exec_exit_code']
            output = execution_data['exec_output']
            analysis = execution_data['exec_analysis']
        except:
            self.logger.error("Cannot load latest execution data, looking in selected_file instead")
            self.logger.error(traceback.format_exc())
            exit_code = selected_file['exec_exit_code']
            output = selected_file['exec_output']
            analysis = selected_file['exec_analysis']

        prompt = f"""Examine the test execution results and analysis, use them to determine if the code and tests are now correct and complete.

# Test exit code: {exit_code}

# Test execution output:
```
{output}
```

# Test analysis:
{analysis}
"""

        prompt += """
Based on the execution results and analysis, please determine:
1. Are all tests passing successfully?
2. Is the code functioning correctly?
3. Are there any remaining issues that need to be addressed?
4. Is the work on this file complete and ready to be marked as finished?

Respond with a JSON object:
{{
    "work_complete": true/false,
    "confidence": 1-10,
    "reasoning": "Explanation of your assessment",
    "remaining_issues": "Short summary of remaining issues",
}}

Set work_complete to true ONLY if:
- Exit code is 0 (success)
- All tests are passing (unless the failing test(s) are clearly incorrect or unnecessary)
- No critical errors or failures
- Code appears to be functioning correctly
- You are confident (confidence >= 8) that no further work is needed
"""

        response = agent.chat_completion(
            prompt=prompt,
            system_message=agent.role,
            response_format={"type": "json_object"}
        )

        result = agent.parse_json_response(response)

        if not result or "error" in result:
            self.logger.error(f"Error parsing LLM analysis response: {result}")
            return False

        work_complete = result.get('work_complete', False)
        confidence = result.get('confidence', 1)
        reasoning = result.get('reasoning', 'No reasoning provided')
        remaining_issues = result.get('remaining_issues', "No issues provided")

        progress = {
            'work_complete': work_complete,
            'confidence': confidence,
            'reasoning': reasoning,
            'remaining_issues': remaining_issues,
            'agent_id': agent.get_id()
        }
        agent.version_manager.store_data(selected_file['path'], 'progress', progress)

        self.logger.info(f"LLM analysis - Work complete: {work_complete}, "
                       f"Confidence: {confidence}/10, Reasoning: {reasoning}")
        if remaining_issues:
            self.logger.info(f"Remaining issues identified: {remaining_issues}")

        # Only consider work complete if LLM is confident
        work_complete = work_complete and confidence >= 8
        return work_complete


    def execute(self, agent, **kwargs) -> bool:
        """
        Execute advanced reset with analysis and appropriate flagging

        Args:
            agent: Agent instance
            **kwargs: Additional arguments

        Returns:
            True if reset completed successfully
        """
        self.logger.info("Starting AdvancedResetAction execution")

        # Get selected file from context before clearing
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']

        self.logger.info("Analyzing execution results to determine if work is complete")
        work_complete = self._analyze_execution_results(agent, selected_file)


        # Handle file flagging if we have a file path
        # Always unflag active since we're resetting
        unflag_result = agent.version_manager.unflag_active(agent, file_path)
        if unflag_result:
            self.logger.info(f"Successfully unflagged active status for {file_path}")

        # Clear context first (parent class behavior)
        agent.context.clear()
        self.logger.info("Agent context has been cleared")

        # Flag as finished only if LLM determined work is complete
        if work_complete:
            agent.version_manager.flag_finished(agent, file_path)
            self.logger.info(f"Flagged {file_path} as finished based on LLM analysis")
            return True
        else:
            self.logger.error(f"{file_path} is incomplete based on LLM analysis")
            return False
