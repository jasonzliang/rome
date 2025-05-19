# fsm.py
from abc import ABC, abstractmethod
import os
import sys
from typing import Dict, List, Optional, Callable

from .action import Action
from .action import SearchAction
from .action import RetryAction
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
        """Reset FSM state and clear agent context"""
        agent.context.clear()
        self.logger.info("Clearing agent context")
        self.current_state = self.default_state
        self.logger.info(f"Setting FSM current state to default: {self.current_state}")

    def add_state(self, state: State, state_name: str = None):
        """
        Add a state (node) to the FSM

        Args:
            state: The State object to add
            state_name: Optional name override. If not provided, uses the class name
        """
        # Use provided state_name or get from state object (which is the class name)
        state_name = state_name or state.name

        self.states[state_name] = state
        # Initialize empty transitions for this state
        if state_name not in self.transitions:
            self.transitions[state_name] = {}
        self.logger.info(f"Added state: {state_name} with actions: {state.actions}")
        return state_name

    def add_action(self, from_state: str, to_state: str, action: Action, action_name: str = None):
        """
        Add an action (edge) between two states

        Args:
            from_state: Source state name
            to_state: Target state name
            action: The Action object to add
            action_name: Optional name for the action. If not provided, uses action.__class__.__name__
        """
        # Use provided action_name or derive from action class name
        action_name = action_name or action.__class__.__name__

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

        # Update the state's actions list using state's add_action method
        self.states[from_state].add_action(action_name)

        self.logger.info(f"Added action: {from_state} --[{action_name}]--> {to_state}")
        return action_name

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

    def get_current_state(self) -> str:
        """Get current state name"""
        return self.current_state

    def get_current_state_object(self) -> State:
        """Get current state object"""
        return self.states.get(self.current_state)

    def get_available_actions(self) -> List[str]:
        """Get actions available from current state"""
        # Now we can get actions directly from the state object
        if self.current_state and self.current_state in self.states:
            return self.states[self.current_state].get_available_actions()
        # Fallback to transitions if state not found
        return list(self.transitions.get(self.current_state, {}).keys())

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

    def draw_graph(self, output_path: str = None) -> str:
        """
        Draw the FSM graph to a PNG file

        Args:
            output_path: Path where to save the graph image. If None,
                         a default path will be used.

        Returns:
            The path to the generated PNG file
        """
        try:
            # Try to import graphviz
            import graphviz
        except ImportError:
            error_msg = "Unable to draw FSM graph: graphviz package not installed. Please install with 'pip install graphviz'"
            self.logger.error(error_msg)
            raise ImportError(error_msg)

        # Create a new directed graph
        dot = graphviz.Digraph('FSM', format='png')
        dot.attr(rankdir='LR', size='11,8')  # Left to right layout, 8x5 inch size

        # Set the node styles
        dot.attr('node', shape='circle', style='filled', color='lightblue2')

        # Add states as nodes
        for state_name in self.states.keys():
            # Current state gets a special color
            if state_name == self.current_state:
                dot.node(state_name, style='filled', color='lightgreen')
            else:
                dot.node(state_name)

        # Add transitions as edges
        for from_state, actions in self.transitions.items():
            for action, to_state in actions.items():
                dot.edge(from_state, to_state, label=action)

        # Determine output path if not provided
        if output_path is None:
            # Check if we have a base_dir from the logger that might be a good place to save
            log_dir = get_logger().get_log_dir()
            if log_dir:
                output_path = os.path.join(log_dir, 'fsm_graph.png')
            else:
                # If no specific dir, just use current directory
                output_path = 'fsm_graph.png'

        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Render the graph to a file
        try:
            # Use the directory and filename from output_path
            output_dir = os.path.dirname(output_path)
            output_filename = os.path.basename(output_path)

            # Remove extension from filename for graphviz
            if '.' in output_filename:
                output_filename = output_filename.rsplit('.', 1)[0]

            # Render the file - this will create output_filename.png
            output_file = dot.render(filename=output_filename, directory=output_dir, cleanup=True)

            self.logger.info(f"FSM graph rendered to: {output_file}")
            return output_file

        except Exception as e:
            error_msg = f"Error rendering FSM graph: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

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

    # Create state objects first - names will be derived from class names
    idle_state = IdleState(config.get('IdleState', {}))
    code_loaded_state = CodeLoadedState(config.get('CodeLoadedState', {}))

    # Add states to FSM - using explicit names for backward compatibility
    idle_state = fsm.add_state(idle_state)
    code_loaded_state = fsm.add_state(code_loaded_state)

    # Create action objects
    search_action = SearchAction(config.get('SearchAction', {}))
    retry_action = RetryAction(config.get('RetryAction', {}))

    # Add actions to FSM - can use explicit names or let it use action class name
    fsm.add_action(idle_state, code_loaded_state, search_action)
    fsm.add_action(code_loaded_state, idle_state, retry_action)

    # Set initial state
    fsm.set_initial_state("IDLE")

    # Validate the FSM
    fsm.validate_fsm()

    logger.info("FSM created successfully")
    return fsm