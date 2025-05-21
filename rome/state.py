# state.py
from abc import ABC, abstractmethod
import os
import sys
from typing import Dict, List, Optional, Callable
from .logger import get_logger
from .config import set_attributes_from_config


class State(ABC):
    """Abstract state"""
    def __init__(self,
        actions: List[str] = None,
        config: Dict = None):

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
    def get_state_prompt(self, agent) -> str:
        """Returns prompt for state that is passed into llm when action is taken"""
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

    def get_state_prompt(self, agent) -> str:
        """Prompt for idle state"""
        return f"""You are in an idle state, ready to start on a new task."""


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

    def get_state_prompt(self, agent) -> str:
        """Prompt for code loaded state"""
        return f"""You are in code loaded state, having selected a code file to edit."""


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
        required_keys = ['path', 'content', 'changes']
        for key in required_keys:
            assert key in selected_file, f"Missing {key} in selected file"

        assert os.path.exists(selected_file['path']), \
            f"File path does not exist: {selected_file['path']}"
        return True

    def get_state_prompt(self, agent) -> str:
        """Prompt for code loaded state"""
        return f"""You are in code edited state, having successfully edited and updated a code file."""


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

    def get_state_prompt(self, agent) -> str:
        """Prompt for code loaded state"""
        return f"""You are in test edited state, having successfully created and updated tests for a code file."""


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
        required_keys = ['path', 'content', 'output', 'exit_code']
        if 'test_path' in selected_file:
            required_keys += ['test_path', 'test_content']
        for key in required_keys:
            assert key in selected_file, f"Missing {key} in selected file"

        assert os.path.exists(selected_file['path']),
            f"File path does not exist: {selected_file['path']}"
        if 'test_path' in selected_file:
            assert os.path.exists(selected_file['test_path']),
                f"File path does not exist: {selected_file['test_path']}"
        return True

    def get_state_prompt(self, agent) -> str:
        """Prompt for code executed state"""
        return f"""You are in code executed state, having finished running the code or test file."""
