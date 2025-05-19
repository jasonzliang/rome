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
        return f"""You are in an idle state, ready to start coding. You can search for files in the repository to start analyzing code."""


class CodeLoadedState(State):
    """State where code has been loaded and is ready for analysis"""

    def __init__(self, config: Dict = None):
        # No more hardcoded available actions
        super().__init__(actions=[], config=config)

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        assert 'selected_file' in agent.context, "Missing selected_file in agent context"
        selected_file = agent.context['selected_file']
        assert selected_file is not None, "selected_file is None in agent context"

        # Check required keys in selected_file with a more compact assertion
        required_keys = ['path', 'content']
        for key in required_keys:
            assert key in selected_file, f"Missing {key} in selected file"

        assert os.path.exists(selected_file['path']), \
            f"File path does not exist: {selected_file['path']}"
        return True

    def get_state_prompt(self, agent) -> str:
        """Prompt for code loaded state"""
        selected_file = agent.context.get('selected_file', {})
        file_path = selected_file.get('path', 'Not provided')
        file_content = selected_file.get('content', 'Not provided')
        selection_reason = selected_file.get('reason', 'Not provided')

        return f"""Code file has been selected and loaded.

Current code file summary:
- File path: {file_path}
- File content:\n{file_content}
- Selection reason: {selection_reason}"""
