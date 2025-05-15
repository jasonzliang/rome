from abc import ABC, abstractmethod
import os
import sys
from typing import Dict, List, Optional, Callable
from .action import Action
from .action import SearchAction
from .action import RetryAction
from .logger import get_logger


class State(ABC):
    """Abstract state"""
    def __init__(self, name: str, required_context: list = None):
        self.name = name
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


class IdleState(State):
    """Initial state where the agent waits for commands"""

    def __init__(self):
        super().__init__("IDLE", required_context=[])

    def check_context(self, agent, **kwargs) -> bool:
        """In idle state, clear any previous context and always return True"""
        # Clear any previous context when entering idle state
        agent.context.clear()
        self.logger.info("Agent is now idle and ready for new tasks")
        return True

    def get_state_prompt(self, agent) -> str:
        """Prompt for idle state"""
        return """You are in an idle state, ready to begin code analysis.
You can search for files in the repository to start analyzing code.
Available actions: search"""


class CodeLoadedState(State):
    """State where code has been loaded and is ready for analysis"""

    def __init__(self):
        super().__init__("CODELOADED", required_context=["selected_file"])

    def check_context(self, agent, **kwargs) -> bool:
        """Check if we have a selected file in context"""
        selected_file = agent.context.get('selected_file')
        if not selected_file:
            self.logger.warning("No file selected in context")
            return False

        # Validate that the selected file has required fields
        required_fields = ['path', 'content']
        for field in required_fields:
            if field not in selected_file:
                self.logger.warning(f"Selected file missing required field: {field}")
                return False

        # Log successful entry
        file_path = selected_file.get('path', 'Unknown file')
        self.logger.info(f"Code loaded successfully: {file_path}")

        # Optional: Log file statistics
        content = selected_file.get('content', '')
        if content:
            lines = len(content.split('\n'))
            chars = len(content)
            self.logger.info(f"File stats - Lines: {lines}, Characters: {chars}")

        return True

    def get_state_prompt(self, agent) -> str:
        """Prompt for code loaded state"""
        selected_file = agent.context.get('selected_file', {})
        file_path = selected_file.get('path', 'Unknown')

        return f"""Code has been loaded from: {file_path}

You can now analyze the loaded code or search for different files.
Available actions: retry (to load different code)

Current file summary:
- Path: {file_path}
- Content available for analysis
- Selection reason: {selected_file.get('reason', 'Not provided')}"""


class FSM:
    """Finite State Machine as a directed graph"""

    def __init__(self, initial_state: str = None):
        self.states: Dict[str, State] = {}  # nodes
        self.transitions: Dict[str, Dict[str, str]] = {}  # edges: from_state -> action -> to_state
        self.actions: Dict[str, Action] = {}  # action handlers
        self.current_state = initial_state
        self.logger = get_logger()

    def add_state(self, state_name: str, state: State):
        """Add a state (node) to the FSM"""
        self.states[state_name] = state
        # Initialize empty transitions for this state
        if state_name not in self.transitions:
            self.transitions[state_name] = {}
        self.logger.debug(f"Added state: {state_name}")

    def add_action(self, from_state: str, to_state: str, action_name: str, action: Action):
        """Add an action (edge) between two states"""
        # Ensure states exist
        if from_state not in self.states:
            raise ValueError(f"State '{from_state}' doesn't exist")
        if to_state not in self.states:
            raise ValueError(f"State '{to_state}' doesn't exist")

        # Add the transition edge
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        self.transitions[from_state][action_name] = to_state

        # Add action handler
        self.actions[action_name] = action

        self.logger.debug(f"Added action: {from_state} --[{action_name}]--> {to_state}")

    def set_initial_state(self, state_name: str):
        """Set the starting state"""
        if state_name not in self.states:
            raise ValueError(f"State '{state_name}' doesn't exist")
        self.current_state = state_name
        self.logger.info(f"Set initial state: {state_name}")

    def execute_action(self, action_name: str, agent, **kwargs):
        """Execute an action and transition to new state"""
        # Check if action is valid from current state
        if self.current_state not in self.transitions:
            error_msg = f"No transitions defined from state '{self.current_state}'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if action_name not in self.transitions[self.current_state]:
            available_actions = list(self.transitions[self.current_state].keys())
            error_msg = f"Action '{action_name}' not available from state '{self.current_state}'. Available actions: {available_actions}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Get the target state
        next_state = self.transitions[self.current_state][action_name]

        # Execute action if handler exists
        result = None
        if action_name in self.actions:
            self.logger.info(f"Executing action: {action_name}")
            try:
                result = self.actions[action_name].execute(agent, **kwargs)
            except Exception as e:
                self.logger.error(f"Action execution failed: {str(e)}")
                # Stay in current state if action fails
                return None

        # Check if context is valid for target state using check_context
        target_state_obj = self.states[next_state]
        self.logger.debug(f"Checking context for state transition to '{next_state}'")

        if not target_state_obj.check_context(agent, **kwargs):
            self.logger.warning(f"Context validation failed for state '{next_state}'. Staying in '{self.current_state}'")
            # Stay in current state if context validation fails
            return result

        # Transition to new state
        old_state = self.current_state
        self.current_state = next_state

        self.logger.info(f"Transitioned: {old_state} --[{action_name}]--> {next_state}")
        return result

    def get_current_state(self) -> str:
        """Get current state name"""
        return self.current_state

    def get_current_state_object(self) -> State:
        """Get current state object"""
        return self.states.get(self.current_state)

    def get_available_actions(self) -> List[str]:
        """Get actions available from current state"""
        actions = list(self.transitions.get(self.current_state, {}).keys())
        self.logger.debug(f"Available actions from {self.current_state}: {actions}")
        return actions

    def get_state_prompt(self, agent) -> str:
        """Get the prompt for the current state"""
        if self.current_state and self.current_state in self.states:
            return self.states[self.current_state].get_state_prompt(agent)
        return "FSM not properly initialized"

    def get_graph(self) -> Dict:
        """Get the graph structure for visualization"""
        graph = {
            "states": list(self.states.keys()),
            "current_state": self.current_state,
            "transitions": []
        }

        for from_state, actions in self.transitions.items():
            for action, to_state in actions.items():
                graph["transitions"].append({
                    "from": from_state,
                    "action": action,
                    "to": to_state
                })

        return graph

    def validate_fsm(self) -> bool:
        """Validate the FSM structure"""
        issues = []

        # Check if initial state is set and exists
        if not self.current_state:
            issues.append("No initial state set")
        elif self.current_state not in self.states:
            issues.append(f"Initial state '{self.current_state}' not in states")

        # Check that all transition targets exist
        for from_state, actions in self.transitions.items():
            for action, to_state in actions.items():
                if to_state not in self.states:
                    issues.append(f"Transition {from_state}--[{action}]-->{to_state}: target state doesn't exist")
                if action not in self.actions:
                    issues.append(f"Action '{action}' referenced but not defined")

        # Check for unreachable states
        reachable = {self.current_state} if self.current_state else set()
        changed = True
        while changed:
            changed = False
            for from_state in list(reachable):
                for action, to_state in self.transitions.get(from_state, {}).items():
                    if to_state not in reachable:
                        reachable.add(to_state)
                        changed = True

        unreachable = set(self.states.keys()) - reachable
        if unreachable:
            issues.append(f"Unreachable states: {unreachable}")

        if issues:
            for issue in issues:
                self.logger.warning(f"FSM validation issue: {issue}")
            return False

        self.logger.info("FSM validation passed")
        return True


# Create and configure FSM
def create_simple_fsm(config):
    """Create FSM with configuration support"""
    logger = get_logger()
    logger.info("Creating FSM")

    fsm = FSM()

    # Add states (nodes) with actual state objects
    fsm.add_state("IDLE", IdleState())
    fsm.add_state("CODELOADED", CodeLoadedState())

    # Get action configurations
    search_config = config.get('actions', {}).get('search', {})
    retry_config = config.get('actions', {}).get('retry', {})

    # Add actions (edges between states) with proper configuration
    fsm.add_action("IDLE", "CODELOADED", "search", SearchAction(search_config))
    fsm.add_action("CODELOADED", "IDLE", "retry", RetryAction(retry_config))

    # Set initial state
    fsm.set_initial_state("IDLE")

    # Validate the FSM
    fsm.validate_fsm()

    logger.info("FSM created successfully")
    return fsm


if __name__ == "__main__":
    pass
    # # Example usage for testing
    # from .config import DEFAULT_CONFIG

    # # Create and test FSM
    # fsm = setup_default_fsm(DEFAULT_CONFIG)

    # print("FSM Graph Structure:")
    # print(fsm.get_graph())

    # print(f"\nCurrent State: {fsm.get_current_state()}")
    # print(f"Available Actions: {fsm.get_available_actions()}")