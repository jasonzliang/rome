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


class EditCodeAction2(Action):
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
        return f"analyze and improve code in {filename}, addressing execution issues, bugs, performance, and readability using a more complex LLM prompt"

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
        """Create a structured prompt for the LLM to improve code using evidence-based strategies"""

        # Build dynamic sections
        analysis_context = self._build_analysis_context(agent, file_path)

        # Use custom prompt if provided, otherwise use optimized default
        if self.custom_prompt is not None:
            base_prompt = self.custom_prompt
        else:
            base_prompt = self._get_optimized_prompt_template()

        # Format the complete prompt
        return base_prompt.format(
            file_path=file_path,
            content=content,
            analysis_context=analysis_context
        )

    def _build_analysis_context(self, agent, file_path: str) -> str:
        """Build analysis context if available"""
        analysis_prompt = create_analysis_prompt(agent, file_path)
        if analysis_prompt:
            return f"""
## Execution Analysis
{analysis_prompt}
**Priority**: Fix execution failures and address identified issues"""
        return ""

    def _get_optimized_prompt_template(self) -> str:
        """Return the optimized prompt template"""
        return """# Code Improvement Analysis

## Code Assessment
**File**: {file_path}
```python
{content}
```

{analysis_context}

## Improvement Strategy
Apply systematic code enhancement across these dimensions:

**Correctness & Stability**:
- Fix syntax errors, logic bugs, and runtime exceptions
- Handle edge cases and invalid inputs properly
- Ensure proper error handling and recovery

**Performance & Efficiency**:
- Optimize algorithms and data structures
- Eliminate redundant operations and memory leaks
- Improve computational complexity where possible

**Code Quality & Maintainability**:
- Enhance readability with clear naming and structure
- Add comprehensive documentation and type hints
- Modularize code for better organization

## Technical Specifications
- **Standards**: Follow PEP 8 and Python best practices
- **Compatibility**: Maintain existing interface contracts
- **Dependencies**: Preserve import structure and requirements
- **Focus**: Address execution issues identified in analysis above

## Response Format
```json
{{
    "improved_code": "Complete enhanced file content (no markdown blocks)",
    "explanation": "Summary of improvements and technical decisions",
    "changes": [
        {{
            "type": "correctness|performance|maintainability|feature",
            "description": "Short description of what code was added/improved",
            "reasoning": "Short reason of   why this coding approach was chosen"
        }}
    ]
}}
```

**Critical Requirements**:
1. Return ENTIRE file content with all improvements integrated
2. Ensure syntactic correctness and proper Python execution
3. Preserve original functionality while enhancing implementation
4. Address specific execution failures if analysis provided
5. Document all changes with clear technical reasoning"""
