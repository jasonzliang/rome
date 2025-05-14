import os
import sys
from typing import Dict, List, Optional, Callable
from .action import *

class State:
    """Simple state with entry/exit callbacks"""

    def __init__(self, name: str, on_enter: Optional[Callable] = None, on_exit: Optional[Callable] = None):
        self.name = name
        self.on_enter = on_enter or (lambda agent: None)
        self.on_exit = on_exit or (lambda agent: None)


class FSM:
    """Finite State Machine as a directed graph"""

    def __init__(self, initial_state: str = None):
        self.states: Dict[str, State] = {}  # nodes
        self.transitions: Dict[str, Dict[str, str]] = {}  # edges: from_state -> action -> to_state
        self.actions: Dict[str, Callable] = {}  # action handlers
        self.current_state = initial_state

    def add_state(self, state_name: str, on_enter: Callable = None, on_exit: Callable = None):
        """Add a state (node) to the FSM"""
        self.states[state_name] = State(state_name, on_enter, on_exit)
        # Initialize empty transitions for this state
        if state_name not in self.transitions:
            self.transitions[state_name] = {}

    def add_action(self, from_state: str, to_state: str, action: Action):
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

        # Add action handler if provided
        if handler:
            self.actions[action_name] = handler

    def set_initial_state(self, state_name: str):
        """Set the starting state"""
        if state_name not in self.states:
            raise ValueError(f"State '{state_name}' doesn't exist")
        self.current_state = state_name

    def execute_action(self, action_name: str, agent, **kwargs):
        """Execute an action and transition to new state"""
        # Check if action is valid from current state
        if self.current_state not in self.transitions:
            raise ValueError(f"No transitions defined from state '{self.current_state}'")

        if action_name not in self.transitions[self.current_state]:
            raise ValueError(f"Action '{action_name}' not available from state '{self.current_state}'")

        # Get the target state
        next_state = self.transitions[self.current_state][action_name]

        # Exit current state
        self.states[self.current_state].on_exit(agent)

        # Execute action if handler exists
        result = None
        if action_name in self.actions:
            result = self.actions[action_name](agent, **kwargs)

        # Enter new state
        old_state = self.current_state
        self.current_state = next_state
        self.states[self.current_state].on_enter(agent)

        print(f"Transitioned: {old_state} --[{action_name}]--> {next_state}")
        return result

    def get_current_state(self) -> str:
        """Get current state name"""
        return self.current_state

    def get_available_actions(self) -> List[str]:
        """Get actions available from current state"""
        return list(self.transitions.get(self.current_state, {}).keys())

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
def create_simple_fsm():
    fsm = FSM()

    # Add states (nodes)
    fsm.add_state("IDLE", on_enter=None, on_exit=None)
    fsm.add_state("CODELOADED", on_enter=None, on_exit=None)

    # Add actions (edges between states)
    fsm.add_action("IDLE", "CODELOADED", SearchAction)
    fsm.add_action("CODELOADED", "IDLE", RetryAction)

    # Set initial state
    fsm.set_initial_state("IDLE")

    return fsm


if __name__ == "__main__":
    pass
