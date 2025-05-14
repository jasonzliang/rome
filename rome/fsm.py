import os
import sys
from typing import Dict, List, Optional, Callable
from .action import Action
from .action import SearchAction
from .action import RetryAction
from .logger import get_logger


class State:
    """Simple state with entry/exit callbacks"""

    def __init__(self, name: str, on_enter: Optional[Callable] = None, on_exit: Optional[Callable] = None):
        self.name = name
        self.on_enter = on_enter or (lambda agent: None)
        self.on_exit = on_exit or (lambda agent: None)
        self.logger = get_logger()


class FSM:
    """Finite State Machine as a directed graph"""

    def __init__(self, initial_state: str = None):
        self.states: Dict[str, State] = {}  # nodes
        self.transitions: Dict[str, Dict[str, str]] = {}  # edges: from_state -> action -> to_state
        self.actions: Dict[str, Action] = {}  # action handlers
        self.current_state = initial_state
        self.logger = get_logger()

    def add_state(self, state_name: str, on_enter: Callable = None, on_exit: Callable = None):
        """Add a state (node) to the FSM"""
        self.states[state_name] = State(state_name, on_enter, on_exit)
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
            error_msg = f"Action '{action_name}' not available from state '{self.current_state}'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Get the target state
        next_state = self.transitions[self.current_state][action_name]

        # Exit current state
        self.logger.debug(f"Exiting state: {self.current_state}")
        self.states[self.current_state].on_exit(agent)

        # Execute action if handler exists
        result = None
        if action_name in self.actions:
            self.logger.info(f"Executing action: {action_name}")
            result = self.actions[action_name].execute(agent, **kwargs)

        # Enter new state
        old_state = self.current_state
        self.current_state = next_state
        self.logger.debug(f"Entering state: {self.current_state}")
        self.states[self.current_state].on_enter(agent)

        self.logger.info(f"Transitioned: {old_state} --[{action_name}]--> {next_state}")
        return result

    def get_current_state(self) -> str:
        """Get current state name"""
        return self.current_state

    def get_available_actions(self) -> List[str]:
        """Get actions available from current state"""
        actions = list(self.transitions.get(self.current_state, {}).keys())
        self.logger.debug(f"Available actions from {self.current_state}: {actions}")
        return actions

    def get_graph(self) -> Dict:
        """Get the graph structure for visualization"""
        graph = {
            "states": list(self.states.keys()),
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


# Create and configure FSM
def create_simple_fsm(config=None):
    """Create FSM with configuration support"""
    logger = get_logger()
    logger.info("Creating FSM")

    fsm = FSM()

    # Add states (nodes)
    fsm.add_state("IDLE", on_enter=None, on_exit=None)
    fsm.add_state("CODELOADED", on_enter=None, on_exit=None)

    # Get action configurations from main config
    search_config = {}
    retry_config = {}

    if config:
        search_config = config.get('actions', {}).get('search', {})
        retry_config = config.get('actions', {}).get('retry', {})

    # Add actions (edges between states) with proper configuration
    fsm.add_action("IDLE", "CODELOADED", "search", SearchAction(search_config))
    fsm.add_action("CODELOADED", "IDLE", "retry", RetryAction(retry_config))

    # Set initial state
    fsm.set_initial_state("IDLE")

    logger.info("FSM created successfully")
    return fsm


def setup_default_fsm(config=None):
    """Setup the default FSM using configuration"""
    return create_simple_fsm(config)


if __name__ == "__main__":
    pass