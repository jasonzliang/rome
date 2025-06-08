import glob
import os
import sys
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..logger import get_logger
from ..config import check_attrs

class AdvancedResetAction(Action):
    """Advanced reset action that analyzes execution results and manages file flags appropriately"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.logger = get_logger()
        check_attrs(self, ['completion_confidence', 'max_versions'])

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
        execution_data = agent.version_manager.get_data(selected_file['path'], 'exec_result')

        if execution_data:
            exit_code = execution_data.get('exec_exit_code', 'unknown')
            output = execution_data.get('exec_output', 'No output available')
            analysis = execution_data.get('exec_analysis', 'No analysis available')
        else:
            exit_code = selected_file.get('exec_exit_code', 'unknown')
            output = selected_file.get('exec_output', 'No output available')
            analysis = selected_file.get('exec_analysis', 'No analysis available')

        prompt = f"""Examine the test execution results and analysis, use them to determine if the code and tests are now correct and complete.

# Test exit code: {exit_code}

# Test execution output:
```
{output}
```

# Test analysis:
{analysis}

Based on the execution results and analysis, please determine how confident you are that the work is complete.

Consider:
1. Are all tests passing successfully?
2. Is the code functioning correctly?
3. Are there any remaining issues that need to be addressed?
4. Is the work on this file complete and ready to be marked as finished?

Respond with a JSON object:
{{
    "confidence": 0-100,
    "reasoning": "Explanation of your assessment",
    "remaining_issues": "Short summary of remaining issues",
}}

For confidence (0-100):
- Represents how confident you are that the work is COMPLETE
- 100 = absolutely certain work is complete
- {self.completion_confidence}+ = high confidence work is complete, should be flagged as finished
- 50-79 = moderate confidence, some uncertainty remains
- 0-49 = low confidence, significant issues likely remain

Consider work complete if:
- Exit code is 0 (success)
- All tests are passing (unless the failing test(s) are clearly incorrect or unnecessary)
- No critical errors or failures
- Code appears to be functioning correctly
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

        confidence = result.get('confidence', 0)
        reasoning = result.get('reasoning', 'No reasoning provided')
        remaining_issues = result.get('remaining_issues', 'No issues provided')

        # Validate and convert confidence to number
        if isinstance(confidence, str):
            confidence = float(confidence) if confidence.replace('.', '').isdigit() else 0
        elif not isinstance(confidence, (int, float)):
            confidence = 0

        confidence = max(0, min(100, float(confidence)))
        work_complete = confidence >= self.completion_confidence

        progress = {
            'confidence': confidence,
            'reasoning': reasoning,
            'remaining_issues': remaining_issues,
            'agent_id': agent.get_id()
        }

        agent.version_manager.store_data(selected_file['path'], 'progress', progress)

        self.logger.info(f"Confidence: {confidence}/100, Complete: \
            {work_complete}, Reasoning: {reasoning}")

        return work_complete

    def _check_max_versions_reached(self, agent, file_path: str) -> bool:
        """
        Check if file has reached maximum number of versions

        Args:
            agent: Agent instance
            file_path: Path to the file to check

        Returns:
            True if max versions reached, False otherwise
        """
        versions = agent.version_manager.load_version(file_path,
            k=self.max_versions + 1, include_content=False)

        if versions is None:
            version_count = 0
        elif isinstance(versions, list):
            version_count = len(versions)
        else:
            version_count = 1

        if version_count >= self.max_versions:
            self.logger.info(f"Max versions ({self.max_versions}) reached for {file_path}")
            return True

        return False

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

        selected_file = agent.context['selected_file']
        file_path = selected_file['path']

        work_complete = self._analyze_execution_results(agent, selected_file)
        if not work_complete:
            work_complete = self._check_max_versions_reached(agent, file_path)

        agent.version_manager.unflag_active(agent, file_path)
        agent.context.clear()

        if work_complete:
            agent.version_manager.flag_finished(agent, file_path)
            self.logger.info(f"Flagged {file_path} as finished")
            return True
        else:
            self.logger.error(f"Flagged {file_path} as not finished")
            return False
