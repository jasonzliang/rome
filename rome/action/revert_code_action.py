import os
import traceback
from typing import Dict, List, Any, Optional

from .action import Action
from ..config import check_attrs, LONGEST_SUMMARY_LEN
from ..logger import get_logger
from ..state import truncate_text


class RevertCodeAction(Action):
    """Action to analyze version history and potentially revert code/test files to a previous version"""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.logger = get_logger()
        check_attrs(self, ['num_versions'])

    def summary(self, agent) -> str:
        """Return a short summary of the revert action"""
        selected_file = agent.context['selected_file']
        filename = os.path.basename(selected_file['path'])
        return f"analyze last {self.num_versions} versions of {filename} and revert to a previous version with better, working code if needed"

    def _analyze_version_history(self, agent, code_versions: List[Dict], test_versions: List[Dict]) -> Dict:
        """Use LLM to analyze version history and decide if reversion is needed"""

        code_summary = self._create_version_summary(code_versions, "Code")
        test_summary = self._create_version_summary(test_versions, "Test")

        base_prompt = """Analyze the version history of code and test files to determine if reverting to a previous version would be beneficial.

Focus on:
1. Execution success/failure trends over versions
2. Code quality improvements vs regressions
3. Test coverage and effectiveness changes
4. Overall project health trajectory
"""

        prompt = f"""{base_prompt}

{code_summary}

{test_summary}

Based on this version history, determine:
1. Should we revert the code file to a previous version?
2. Should we revert the test file to a previous version?
3. What are the specific reasons for any recommended reversions?

Respond with a JSON object:
{{
    "revert_code": {{
        "should_revert": true/false,
        "target_version": <version_number or null>,
        "reason": "Explanation for reversion decision"
    }},
    "revert_test": {{
        "should_revert": true/false,
        "target_version": <version_number or null>,
        "reason": "Explanation for reversion decision"
    }},
    "overall_assessment": "Brief summary of version history analysis"
}}

Only recommend reversion if there is clear evidence that a previous version was significantly better.
"""

        response = agent.chat_completion(prompt=prompt, system_message=agent.role, response_format={"type": "json_object"})
        return agent.parse_json_response(response)

    def _create_version_summary(self, versions: List[Dict], file_type: str) -> str:
        """Create a formatted summary of version history"""
        if not versions:
            return f"{file_type} versions: No versions found"

        summary = f"{file_type} version history (most recent first):\n"

        for version in versions:
            version_num = version.get('version', 'Unknown')
            timestamp = version.get('timestamp', 'Unknown')
            explanation = version.get('explanation', 'No explanation')
            changes = version.get('changes', [])
            exit_code = version.get('exit_code')
            execution_output = version.get('execution_output', '')

            summary += f"\n--- Version {version_num} ({timestamp}) ---\n"

            summary += f"# Explanation: {explanation}\n"
            summary += "\n"

            summary += f"# Changes: {len(changes)} modifications\n"
            if changes:
                changes_text = ""
                for i, change in enumerate(changes):
                    change_desc = f"  {i+1}. {change.get('type', 'Unknown')}: {change.get('description', 'No description')}\n"
                    if len(changes_text + change_desc) > LONGEST_SUMMARY_LEN:
                        changes_text += f"  ... and {len(changes) - i} more changes (truncated)\n"
                        break
                    changes_text += change_desc
                summary += changes_text
            summary += "\n"

            if exit_code is not None:
                status = "PASSED" if exit_code == 0 else "FAILED"
                summary += f"# Execution: {status} (exit code: {exit_code})\n"

                if execution_output:
                    truncated_output = truncate_text(execution_output, length=LONGEST_SUMMARY_LEN)
                    summary += f"Output:\n{truncated_output}\n"
                summary += "-" * 50 + "\n"

        return summary

    def _revert_file(self, agent, file_path: str, target_version: Dict, file_type: str) -> bool:
        """Revert a file to a specific version"""
        version_number = target_version.get('version')
        content = target_version.get('content')

        if not content:
            self.logger.error(f"No content available for {file_type} version {version_number}")
            return False

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Update agent context
            selected_file = agent.context['selected_file']
            revert_change = {"type": "reversion", "description": f"Reverted to version {version_number}"}
            revert_explanation = f"Reverted to version {version_number}"

            if file_type == "code":
                selected_file['content'] = content
                selected_file['change_record'] = {'explanation': revert_explanation, 'changes': [revert_change]}
            elif file_type == "test":
                selected_file['test_content'] = content
                selected_file['test_changes'] = [{'explanation': revert_explanation, 'changes': [revert_change]}]

            self.logger.info(f"Successfully reverted {file_type} file to version {version_number}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to revert {file_type} file to version {version_number}: {e}")
            return False

    def _handle_reversion(self, agent, revert_info: Dict, versions: List[Dict], file_path: str, file_type: str) -> bool:
        """Handle reversion for a single file type"""
        if not revert_info.get('should_revert'):
            return True

        target_version_num = revert_info.get('target_version')
        reason = revert_info.get('reason', 'No reason provided')

        self.logger.info(f"LLM recommends reverting {file_type} to version {target_version_num}: {reason}")

        target_version = next((v for v in versions if v.get('version') == target_version_num), None)
        if not target_version:
            self.logger.error(f"Target {file_type} version {target_version_num} not found in loaded versions")
            return False

        return self._revert_file(agent, file_path, target_version, file_type)

    def execute(self, agent, **kwargs) -> bool:
        """Execute the revert analysis and potential reversion"""

        # Step 1: Validate inputs
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        test_path = selected_file.get('test_path')

        if not test_path:
            self.logger.error("No test path found in selected file context")
            return False

        # Step 2: Load version histories
        self.logger.info(f"Loading last {self.num_versions} versions for analysis")
        code_versions = agent.version_manager.load_version(file_path, k=self.num_versions, include_content=True)
        test_versions = agent.version_manager.load_test_version(test_path, k=self.num_versions, include_content=True)

        # Normalize to lists
        code_versions = [code_versions] if isinstance(code_versions, dict) else (code_versions or [])
        test_versions = [test_versions] if isinstance(test_versions, dict) else (test_versions or [])

        if not code_versions and not test_versions:
            self.logger.error("No version history found for analysis")
            return True

        # Step 3: Analyze version history with LLM
        self.logger.info("Analyzing version history with LLM")
        analysis_result = self._analyze_version_history(agent, code_versions, test_versions)

        if not analysis_result or "error" in analysis_result:
            self.logger.error(f"Failed to analyze version history: {analysis_result}")
            return False

        overall_assessment = analysis_result.get('overall_assessment', 'No assessment provided')
        self.logger.info(f"LLM Analysis: {overall_assessment}")

        # Step 4: Execute reversions if recommended
        revert_code = analysis_result.get('revert_code', {})
        revert_test = analysis_result.get('revert_test', {})

        code_success = self._handle_reversion(agent, revert_code, code_versions, file_path, "code")
        test_success = self._handle_reversion(agent, revert_test, test_versions, test_path, "test")

        if not (code_success and test_success):
            return False

        # Step 5: Log completion
        if not revert_code.get('should_revert') and not revert_test.get('should_revert'):
            self.logger.info("LLM analysis determined no reversion is necessary")

        self.logger.info("RevertCodeAction execution completed successfully")
        return True
