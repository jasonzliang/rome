# Updated state.py with enhanced summary methods
from abc import ABC, abstractmethod
import os
import sys
from typing import Dict, List, Optional, Callable

from .config import SUMMARY_LENGTH, set_attributes_from_config
from .logger import get_logger


# Helper functions for common context validation patterns
def validate_context_keys(context: Dict, required_keys: List[str], context_name: str = "context") -> None:
    """Validate that all required keys exist in the given context"""
    for key in required_keys:
        assert key in context, f"Missing {key} in {context_name}"


def validate_selected_file_context(agent, required_keys: List[str]) -> Dict:
    """Validate and return selected_file from agent context with required keys"""
    selected_file = agent.context.get('selected_file')
    assert selected_file is not None, "selected_file is None in agent context"

    validate_context_keys(selected_file, required_keys, 'selected file')
    return selected_file


def validate_file_exists(file_path: str) -> None:
    """Validate that a file path exists"""
    assert os.path.exists(file_path), f"File path does not exist: {file_path}"


def get_file_size_info(file_path: str) -> str:
    """Get formatted file size information"""
    try:
        size_kb = os.path.getsize(file_path) / 1024
        return f" ({size_kb:.1f}KB)"
    except:
        return ""


def truncate_text(text: str, max_length: int = SUMMARY_LENGTH) -> str:
    """Truncate text with ellipsis if it exceeds max_length"""
    return text[:max_length] + '...' if len(text) > max_length else text

class State(ABC):
    """Abstract state"""
    def __init__(self,
        actions: List[str] = None,
        config: Dict = None):

        self.name = self.__class__.__name__
        self.config = config or {}
        # Initialize with empty actions list - will be populated by FSM
        self.actions = actions or []  # Store available actions for this state
        self.logger = get_logger()

        # Set attributes from config if provided
        set_attributes_from_config(self, self.config)

    @abstractmethod
    def check_context(self, agent, **kwargs) -> bool:
        """Check if agent context has required values"""
        pass

    @abstractmethod
    def summary(self, agent) -> str:
        """A short summary description of the current state"""
        pass

    def get_available_actions(self) -> List[str]:
        """Get the list of available actions for this state"""
        return self.actions.copy()

    def add_action(self, action_name: str) -> None:
        """Add an action to this state if it doesn't already exist"""
        if action_name not in self.actions:
            self.actions.append(action_name)


class IdleState(State):
    """Initial state where the agent waits for commands"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """In idle state, clear any previous context and always return True"""
        agent.context.clear()
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for idle state"""
        return f"{self.name}: you are idling in initial state and ready to start a new task"


class CodeLoadedState(State):
    """State where code has been loaded and is ready for analysis"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        selected_file = validate_selected_file_context(agent, ['path', 'content', 'reason'])
        validate_file_exists(selected_file['path'])
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for code loaded state"""
        selected_file = agent.context['selected_file']
        filename = os.path.basename(selected_file['path'])
        size_info = get_file_size_info(selected_file['path'])
        reason = truncate_text(selected_file['reason'])

        return f"{self.name}: you selected file {filename}{size_info} for editing, selection reason: {reason}"


class CodeEditedState(State):
    """State where code has been edited and updated"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        selected_file = validate_selected_file_context(agent, ['path', 'content', 'change_record'])

        # Validate nested change_record structure
        validate_context_keys(selected_file['change_record'], ['changes', 'explanation'], 'change record')
        validate_file_exists(selected_file['path'])
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for code edited state"""
        selected_file = agent.context['selected_file']
        filename = os.path.basename(selected_file['path'])
        change_record = selected_file['change_record']

        num_changes = len(change_record['changes'])
        explanation = truncate_text(change_record['explanation'])

        return f"{self.name}: you edited {filename} with {num_changes} change(s) with explanation: {explanation}"


class TestEditedState(State):
    """State where code tests has been edited and updated"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        required_keys = ['path', 'content', 'changes', 'test_path', 'test_content', 'test_changes']
        selected_file = validate_selected_file_context(agent, required_keys)

        # Validate both file paths exist
        for path in [selected_file['path'], selected_file['test_path']]:
            validate_file_exists(path)
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for test edited state"""
        selected_file = agent.context['selected_file']
        filename = os.path.basename(selected_file['path'])
        test_filename = os.path.basename(selected_file['test_path'])

        num_test_changes = len(selected_file['test_changes'])
        action_type = "created" if not os.path.exists(selected_file['test_path']) else "updated"

        return f"{self.name}: you {action_type} tests in {test_filename} for {filename} with {num_test_changes} change(s)"


class CodeExecutedPassState(State):
    """State where code or test file has been executed successfully"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context with successful execution"""
        required_keys = ['path', 'content', 'test_path', 'test_content', 'exec_output', 'exec_exit_code', 'exec_analysis']
        selected_file = validate_selected_file_context(agent, required_keys)

        # Validate file paths exist
        paths_to_check = [selected_file['path'], selected_file['test_path']]
        for path in paths_to_check:
            validate_file_exists(path)

        # Validate that execution was successful
        exit_code = selected_file['exec_exit_code']
        assert exit_code == 0, f"Expected successful execution (exit code 0), got {exit_code}"
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for code executed pass state"""
        selected_file = agent.context['selected_file']
        test_filename = os.path.basename(selected_file['test_path'])

        # Get brief output summary
        output = selected_file['exec_output'] or 'No output'
        output_summary = truncate_text(output.split('\n')[0] if output else 'No output')

        return f"{self.name}: you executed {test_filename}: ✓ PASSED (exit code: 0), output: {output_summary}"


class CodeExecutedFailState(State):
    """State where code or test file execution failed"""

    def __init__(self, config: Dict = None):
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context with failed execution"""
        required_keys = ['path', 'content', 'test_path', 'test_content', 'exec_output', 'exec_exit_code', 'exec_analysis']
        selected_file = validate_selected_file_context(agent, required_keys)

        # Validate file paths exist
        paths_to_check = [selected_file['path'], selected_file['test_path']]
        for path in paths_to_check:
            validate_file_exists(path)

        # Validate that execution failed
        exit_code = selected_file['exec_exit_code']
        assert exit_code != 0, f"Expected failed execution (exit code != 0), got {exit_code}"
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for code executed fail state"""
        selected_file = agent.context['selected_file']
        test_filename = os.path.basename(selected_file['test_path'])
        exit_code = selected_file['exec_exit_code']

        # Get brief output summary focusing on error info
        output = selected_file['exec_output'] or 'No output'
        output_summary = truncate_text(output.split('\n')[0] if output else 'No output')

        return f"{self.name}: you executed {test_filename}: ✗ FAILED (exit code: {exit_code}), output: {output_summary}"
