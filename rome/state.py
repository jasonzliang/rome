# Updated state.py with enhanced summary methods
from abc import ABC, abstractmethod
import os
import sys
from typing import Dict, List, Optional, Callable
from .logger import get_logger
from .config import set_attributes_from_config

SUMMARY_LENGTH=100

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
        # No more hardcoded available actions - let FSM manage this
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """In idle state, clear any previous context and always return True"""
        # Clear any previous context when entering idle state
        agent.context.clear()
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for idle state"""
        return f"{self.name}: Agent is idle and ready to start a new task"


class CodeLoadedState(State):
    """State where code has been loaded and is ready for analysis"""

    def __init__(self, config: Dict = None):
        # No more hardcoded available actions
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        assert agent.context.get('selected_file') is not None, \
            "selected_file is None in agent context"
        selected_file = agent.context['selected_file']

        # Check required keys in selected_file with a more compact assertion
        required_keys = ['path', 'content', 'reason']
        for key in required_keys:
            assert key in selected_file, f"Missing {key} in selected file"

        assert os.path.exists(selected_file['path']), \
            f"File path does not exist: {selected_file['path']}"
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for code loaded state"""
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        filename = os.path.basename(file_path)
        reason = selected_file['reason']

        # Get file size info
        try:
            size_kb = os.path.getsize(file_path) / 1024
            size_info = f" ({size_kb:.1f}KB)"
        except:
            size_info = ""

        return f"{self.name}: Agent selected file {filename}{size_info} for editing, selection reason: {reason}"


class CodeEditedState(State):
    """State where code has been edited and updated"""

    def __init__(self, config: Dict = None):
        # No more hardcoded available actions
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        assert agent.context.get('selected_file') is not None, \
            "selected_file is None in agent context"
        selected_file = agent.context['selected_file']

        # Check required keys in selected_file with a more compact assertion
        required_keys = ['path', 'content', 'change_record']
        for key in required_keys:
            assert key in selected_file, f"Missing {key} in selected file"

        assert 'changes' in selected_file['change_record'], f"Missing changes in change record"
        assert 'explanation' in selected_file['change_record'], f"Missing explanation in change record"

        assert os.path.exists(selected_file['path']), \
            f"File path does not exist: {selected_file['path']}"
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for code edited state"""
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        filename = os.path.basename(file_path)

        # Get changes info
        change_record = selected_file['change_record']
        num_changes = len(change_record['changes'])
        explanation = change_record['explanation']

        # Get last change explanation if available
        return f"{self.name}: Agent edited {filename} with {num_changes} change(s) with explanation: {explanation[:SUMMARY_LENGTH]}{'...' if len(explanation) > SUMMARY_LENGTH else ''}"


class TestEditedState(State):
    """State where code tests has been edited and updated"""

    def __init__(self, config: Dict = None):
        # No more hardcoded available actions
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        assert agent.context.get('selected_file') is not None, \
            "selected_file is None in agent context"
        selected_file = agent.context['selected_file']

        # Check required keys in selected_file with a more compact assertion
        required_keys = ['path', 'content', 'changes', 'test_path', 'test_content', 'test_changes']
        for key in required_keys:
            assert key in selected_file, f"Missing {key} in selected file"

        for path in [selected_file['path'], selected_file['test_path']]:
            assert os.path.exists(path), f"File path does not exist: {path}"
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for test edited state"""
        selected_file = agent.context['selected_file']
        file_path = selected_file['path']
        test_path = selected_file['test_path']
        filename = os.path.basename(file_path)
        test_filename = os.path.basename(test_path)

        # Get test changes info
        test_changes = selected_file['test_changes']
        num_test_changes = len(test_changes)

        # Check if test file was created or updated
        action_type = "created" if not os.path.exists(test_path) else "updated"

        return f"{self.name}: Agent {action_type} tests in {test_filename} for {filename} with {num_test_changes} change(s)"


class CodeExecutedState(State):
    """State where code or test file has been executed"""

    def __init__(self, config: Dict = None):
        # No more hardcoded available actions
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        assert agent.context.get('selected_file') is not None, \
            "selected_file is None in agent context"
        selected_file = agent.context['selected_file']

        # Check required keys in selected_file with a more compact assertion
        required_keys = ['path', 'content', 'exec_output', 'exec_exit_code']
        if 'test_path' in selected_file:
            required_keys += ['test_path', 'test_content']
        for key in required_keys:
            assert key in selected_file, f"Missing {key} in selected file"

        assert os.path.exists(selected_file['path']), \
            f"File path does not exist: {selected_file['path']}"
        if 'test_path' in selected_file:
            assert os.path.exists(selected_file['test_path']), \
                f"File path does not exist: {selected_file['test_path']}"
        return True

    def summary(self, agent) -> str:
        """Enhanced summary for code executed state"""
        selected_file = agent.context['selected_file']
        test_path = selected_file['test_path']
        test_filename = os.path.basename(test_path)

        # Get execution results
        exit_code = selected_file['exec_exit_code']
        output = selected_file['exec_output']

        # Determine status
        if exit_code == 0:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"

        # Get brief output summary
        output_lines = output.split('\n') if output else []
        output_summary = output_lines[0] if output_lines else 'No output'
        if len(output_summary) > SUMMARY_LENGTH:
            output_summary = output_summary[:SUMMARY_LENGTH] + '...'

        return f"{self.name}: Agent executed {test_filename}: {status} (exit code: {exit_code}), output: {output_summary}"
