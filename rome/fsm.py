from abc import ABC, abstractmethod
import os
import sys
from typing import Dict, List, Optional, Callable

from .action import Action
from .search_action import SearchAction
from .retry_action import RetryAction
from .state import State
from .state import IdleState
from .state import CodeLoadedState
from .logger import get_logger
from .config import set_attributes_from_config

class FSM:
    """Finite State Machine as a directed graph"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        # Set attributes from FSM config
        fsm_config = self.config.get('FSM', {})
        set_attributes_from_config(self, fsm_config)

        self.states: Dict[str, State] = {}  # nodes
        self.transitions: Dict[str, Dict[str, str]] = {}  # edges: from_state -> action -> to_state
        self.actions: Dict[str, Action] = {}  # action handlers
        self.current_state = self.default_state = None
        self.logger = get_logger()

    def reset(self, agent):
        agent.context.clear()
        self.logger("Clearing agent context")
        self.current_state = self.default_state
        self.logger(f"Setting FSM current state to default: {self.current_state}")

    def add_state(self, state_name: str, state: State):
        """Add a state (node) to the FSM"""
        self.states[state_name] = state
        # Initialize empty transitions for this state
        if state_name not in self.transitions:
            self.transitions[state_name] = {}
        self.logger.info(f"Added state: {state_name} with actions: {state.actions}")

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

        # Update the state's actions list if not already present
        if action_name not in self.states[from_state].actions:
            self.states[from_state].actions.append(action_name)
            self.logger.info(f"Added action '{action_name}' to state '{from_state}' actions list")

        self.logger.info(f"Added action: {from_state} --[{action_name}]--> {to_state}")

    def set_initial_state(self, state_name: str):
        """Set the starting state"""
        if state_name not in self.states:
            raise ValueError(f"State '{state_name}' doesn't exist")
        self.current_state = state_name
        self.default_state = state_name
        self.logger.info(f"Set initial state: {state_name}")

    def execute_action(self, action_name: str, agent, **kwargs):
        """Execute an action and transition to new state"""
        # Check if action is valid from current state
        if action_name not in self.actions:
            error_msg = f"Action '{action_name}' does not exist in FSM"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

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
        self.logger.info(f"Executing action: {action_name}")
        result = self.actions[action_name].execute(agent, **kwargs)

        # Check if context is valid for target state using check_context
        target_state_obj = self.states[next_state]
        self.logger.info(f"Checking context for state transition to '{next_state}'")
        target_state_obj.check_context(agent, **kwargs)

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
        # Now we can get actions from either the state object or the transitions
        if self.current_state and self.current_state in self.states:
            # Use the state's actions list for consistency
            actions = self.states[self.current_state].get_available_actions()
        else:
            # Fallback to transitions if state not found
            actions = list(self.transitions.get(self.current_state, {}).keys())

        self.logger.info(f"Available actions from {self.current_state}: {actions}")
        return actions

    def _get_state_prompt(self, agent) -> str:
        """Get the prompt for the current state"""
        if self.current_state and self.current_state in self.states:
            return self.states[self.current_state].get_state_prompt(agent)
        return "FSM not properly initialized"

    def get_action_prompt(self, agent) -> str:
        """
        Construct a prompt that combines state information and available actions
        """
        # Get current state prompt
        state_prompt = self._get_state_prompt(agent)

        # Get available actions
        available_actions = self.get_available_actions()

        # Construct the full prompt
        prompt = f"""{state_prompt}

Available actions: {', '.join(available_actions)}

Please select one of the available actions to execute. Respond with a JSON object containing:
{{
    "action": "chosen_action_name",
    "reasoning": "Brief explanation of why you chose this action"
}}

Choose the most appropriate action based on the current state and context."""

        return prompt

    def get_graph(self) -> Dict:
        """Get the graph structure for visualization"""
        graph = {
            "states": list(self.states.keys()),
            "current_state": self.current_state,
            "transitions": [],
            "state_actions": {}  # Add state actions mapping
        }

        # Add state actions mapping
        for state_name, state_obj in self.states.items():
            graph["state_actions"][state_name] = state_obj.get_available_actions()

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

        # Check default state is set
        if not self.default_state:
            issues.append("No default state set")

        # Check if initial state is set and exists
        if self.current_state not in self.states:
            issues.append(f"Initial state '{self.current_state}' not in states")

        # Check that all transition targets exist
        for from_state, actions in self.transitions.items():
            for action, to_state in actions.items():
                if to_state not in self.states:
                    issues.append(f"Transition {from_state}--[{action}]-->{to_state}: target state doesn't exist")
                if action not in self.actions:
                    issues.append(f"Action '{action}' referenced but not defined")

        # Validate state actions consistency with transitions
        for state_name, state_obj in self.states.items():
            state_actions = set(state_obj.get_available_actions())
            transition_actions = set(self.transitions.get(state_name, {}).keys())

            # Check if state has actions not in transitions
            extra_actions = state_actions - transition_actions
            if extra_actions:
                issues.append(f"State '{state_name}' has actions not in transitions: {extra_actions}")

            # Check if transitions have actions not in state
            missing_actions = transition_actions - state_actions
            if missing_actions:
                issues.append(f"State '{state_name}' missing actions from transitions: {missing_actions}")

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
                self.logger.error(f"FSM validation issue: {issue}")
            return False

        self.logger.info("FSM validation passed")
        return True


# Create and configure FSM
def create_simple_fsm(config):
    """Create FSM with configuration support"""
    logger = get_logger()
    logger.info("Creating FSM")

    fsm = FSM(config)

    # Get state configurations (empty dicts if not present)
    idle_state_config = config.get('IdleState', {})
    code_loaded_state_config = config.get('CodeLoadedState', {})

    # Add states (nodes) with actual state objects and their configs
    fsm.add_state("IDLE", IdleState(idle_state_config))
    fsm.add_state("CODELOADED", CodeLoadedState(code_loaded_state_config))

    # Get action configurations
    search_config = config.get('SearchAction', {})
    retry_config = config.get('RetryAction', {})

    # Add actions (edges between states) with proper configuration
    fsm.add_action("IDLE", "CODELOADED", "SearchAction", SearchAction(search_config))
    fsm.add_action("CODELOADED", "IDLE", "RetryAction", RetryAction(retry_config))

    # Set initial state
    fsm.set_initial_state("IDLE")

    # Validate the FSM
    fsm.validate_fsm()

    logger.info("FSM created successfully")
    return fsm
