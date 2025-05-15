from abc import ABC, abstractmethod
import os
import sys
from typing import Dict, List, Optional, Callable


class State(ABC):
    """Abstract state"""
    def __init__(self, name: str, actions: List[str] = None, required_context: list = None):
        self.name = name
        self.actions = actions or []  # Store available actions for this state
        self.required_context = required_context or []
        self.logger = get_logger()

    @abstractmethod
    def check_context(self, agent, **kwargs) -> bool:
        """Return true if agent context has required values"""
        pass

    @abstractmethod
    def get_state_prompt(self, agent) -> str:
        """Returns prompt for state that is passed into llm when action is taken"""
        pass

    def get_available_actions(self) -> List[str]:
        """Get the list of available actions for this state"""
        return self.actions.copy()


class IdleState(State):
    """Initial state where the agent waits for commands"""

    def __init__(self):
        # Define available actions for the idle state
        available_actions = ["search"]
        super().__init__("IDLE", actions=available_actions, required_context=[])

    def check_context(self, agent, **kwargs) -> bool:
        """In idle state, clear any previous context and always return True"""
        # Clear any previous context when entering idle state
        agent.context.clear()
        # return True

    def get_state_prompt(self, agent) -> str:
        """Prompt for idle state"""
        return f"""You are in an idle state, ready to begin code analysis.
You can search for files in the repository to start analyzing code."""


class CodeLoadedState(State):
    """State where code has been loaded and is ready for analysis"""

    def __init__(self):
        # Define available actions for the code loaded state
        available_actions = ["retry"]
        super().__init__("CODELOADED", actions=available_actions, required_context=["selected_file"])

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        assert 'selected_file' not in agent.context:
        selected_file = agent.context['selected_file']
        assert 'path' not in selected_file and 'content' not in selected_file:
        assert os.path.exists(selected_file['path'])

    def get_state_prompt(self, agent) -> str:
        """Prompt for code loaded state"""
        selected_file = agent.context.get('selected_file', {})
        file_path = selected_file.get('path', 'Not provided')
        file_content = selected_file.get('content', 'Not provided')
        selection_reason = selected_file.get('reason', 'Not provided')

        return f"""Code file has been selected and loaded.

Current code file summary:
- File path:{file_path}
- File content:\n{file_content}
- Selection reason: {selection_reason}"""

