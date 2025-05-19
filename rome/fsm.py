# fsm.py
from abc import ABC, abstractmethod
import os
import sys
from typing import Dict, List, Optional, Callable, Union, Tuple

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
        set_attributes_from_config(self, self.config)

        self.states: Dict[str, State] = {}  # nodes
        self.transitions: Dict[str, Dict[str, Tuple[str, Optional[str]]]] = {}  # from_state -> action -> (target, fallback)
        self.actions: Dict[str, Action] = {}  # action handlers
        self.current_state = self.default_state = None
        self.logger = get_logger()

    def reset(self, agent):
        """Reset FSM state and clear agent context"""
        agent.context.clear()
        self.logger.info("Clearing agent context")
        self.current_state = self.default_state
        self.logger.info(f"Setting FSM current state to default: {self.current_state}")

    def add_state(self, state: State, state_name: str = None) -> str:
        """Add a state (node) to the FSM"""
        state_name = state_name or state.__class__.__name__
        self.states[state_name] = state
        if state_name not in self.transitions:
            self.transitions[state_name] = {}
        self.logger.info(f"Added state: {state_name} with actions: {state.actions}")
        return state_name

    def add_action(self, from_state: str, to_state: str, action: Action,
                  action_name: str = None, fallback_state: str = None) -> str:
        """Add an action (edge) between states with optional fallback state"""
        action_name = action_name or action.__class__.__name__

        # Validate states
        for state in (from_state, to_state, fallback_state):
            if state and state not in self.states:
                raise ValueError(f"State '{state}' doesn't exist")

        # Add the transition
        if from_state not in self.transitions:
            self.transitions[from_state] = {}

        self.transitions[from_state][action_name] = (to_state, fallback_state)
        self.actions[action_name] = action
        self.states[from_state].add_action(action_name)

        log_msg = f"Added action: {from_state} --[{action_name}]--> {to_state}"
        if fallback_state:
            log_msg += f" (fallback: {fallback_state})"
        self.logger.info(log_msg)

        return action_name

    def set_initial_state(self, state_name: str):
        """Set the starting state"""
        if state_name not in self.states:
            raise ValueError(f"State '{state_name}' doesn't exist")
        self.current_state = self.default_state = state_name
        self.logger.info(f"Set initial state: {state_name}")

    def execute_action(self, action_name: str, agent, **kwargs):
        """Execute an action and transition to new state based on execution result"""
        # Validate action
        if action_name not in self.actions:
            raise ValueError(f"Action '{action_name}' does not exist in FSM")

        if self.current_state not in self.transitions:
            raise ValueError(f"No transitions defined from state '{self.current_state}'")

        if action_name not in self.transitions[self.current_state]:
            available = list(self.transitions[self.current_state].keys())
            raise ValueError(f"Action '{action_name}' not available from '{self.current_state}'. Available: {available}")

        # Get transition information
        target_state, fallback_state = self.transitions[self.current_state][action_name]

        # Execute action and determine next state
        self.logger.info(f"Executing action: {action_name}")
        result = self.actions[action_name].execute(agent, **kwargs)

        next_state = target_state
        if result is False and fallback_state:
            next_state = fallback_state
            self.logger.info(f"Action {action_name} failed, using fallback state: {fallback_state}")

        # Validate context for target state
        self.logger.info(f"Checking context for state transition to '{next_state}'")
        self.states[next_state].check_context(agent, **kwargs)

        # Transition to new state
        old_state = self.current_state
        self.current_state = next_state

        transition_type = " (FAILED)" if result is False and fallback_state else ""
        self.logger.info(f"Transitioned: {old_state} --[{action_name}{transition_type}]--> {next_state}")

    def get_current_state(self) -> str:
        """Get current state name"""
        return self.current_state

    def get_current_state_object(self) -> State:
        """Get current state object"""
        return self.states.get(self.current_state)

    def get_available_actions(self) -> List[str]:
        """Get actions available from current state"""
        if self.current_state and self.current_state in self.states:
            return self.states[self.current_state].get_available_actions()
        return list(self.transitions.get(self.current_state, {}).keys())

    def get_action_prompt(self, agent) -> str:
        """
        Construct a prompt that combines state information and available actions

        Raises:
            ValueError: If the current state is not properly initialized
        """
        if self.current_state not in self.states:
            raise ValueError(f"Current state '{self.current_state}' is not a valid state in the FSM")

        state_prompt = self.states[self.current_state].get_state_prompt(agent)
        available_actions = self.get_available_actions()

        prompt = f"""{state_prompt}

Available actions: {', '.join(available_actions)}

Please select one of the available actions to execute. Respond with a JSON object containing:
{{
    "action": "chosen_action_name",
    "reasoning": "Brief explanation of why you chose this action"
}}
Choose the most appropriate action based on the current state and context.
"""
        return prompt

    def get_graph(self) -> Dict:
        """Get the graph structure for visualization including fallback transitions"""
        graph = {
            "states": list(self.states.keys()),
            "current_state": self.current_state,
            "transitions": [],
            "state_actions": {state_name: state_obj.get_available_actions()
                             for state_name, state_obj in self.states.items()}
        }

        for from_state, actions in self.transitions.items():
            for action, (to_state, fallback_state) in actions.items():
                # Add main transition
                graph["transitions"].append({
                    "from": from_state,
                    "action": action,
                    "to": to_state,
                    "type": "success"
                })

                # Add fallback transition if exists
                if fallback_state:
                    graph["transitions"].append({
                        "from": from_state,
                        "action": f"{action} (FAILED)",
                        "to": fallback_state,
                        "type": "fallback"
                    })

        return graph

    def draw_graph(self, output_path: str = None) -> str:
        """Draw the FSM graph to a PNG file including fallback transitions"""
        try:
            import graphviz
        except ImportError:
            error_msg = "Unable to draw FSM graph: graphviz package not installed. Please install with 'pip install graphviz'"
            self.logger.error(error_msg)
            raise ImportError(error_msg)

        # Create and configure graph
        dot = graphviz.Digraph('FSM', format='png')
        dot.attr(rankdir='LR', size='11,8')
        dot.attr('node', shape='circle', style='filled', color='lightblue2')

        # Add states as nodes
        for state_name in self.states.keys():
            dot.node(state_name, style='filled', color='lightgreen' if state_name == self.current_state else 'lightblue2')

        # Add transitions as edges
        for transition in self.get_graph()["transitions"]:
            edge_style = {"style": "dashed", "color": "red"} if transition["type"] == "fallback" else {}
            dot.edge(transition["from"], transition["to"], label=transition["action"], **edge_style)

        # Determine output path
        if output_path is None:
            log_dir = get_logger().get_log_dir()
            output_path = os.path.join(log_dir, 'fsm_graph.png') if log_dir else 'fsm_graph.png'

        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Render the graph
        try:
            output_dir = os.path.dirname(output_path)
            output_filename = os.path.basename(output_path).rsplit('.', 1)[0] if '.' in os.path.basename(output_path) else os.path.basename(output_path)
            output_file = dot.render(filename=output_filename, directory=output_dir, cleanup=True)
            self.logger.info(f"FSM graph rendered to: {output_file}")
            return output_file
        except Exception as e:
            error_msg = f"Error rendering FSM graph: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def validate_fsm(self) -> bool:
        """Validate the FSM structure including fallback transitions"""
        issues = []

        # Check default state is set and exists
        if not self.default_state:
            issues.append("No default state set")
        elif self.current_state not in self.states:
            issues.append(f"Initial state '{self.current_state}' not in states")

        # Check that all transition targets exist and actions are defined
        for from_state, actions in self.transitions.items():
            for action, (to_state, fallback_state) in actions.items():
                if to_state not in self.states:
                    issues.append(f"Transition {from_state}--[{action}]-->{to_state}: target state doesn't exist")

                if fallback_state and fallback_state not in self.states:
                    issues.append(f"Transition {from_state}--[{action}]-->{to_state}: fallback state '{fallback_state}' doesn't exist")

                if action not in self.actions:
                    issues.append(f"Action '{action}' referenced but not defined")

        # Validate state actions consistency with transitions
        for state_name, state_obj in self.states.items():
            state_actions = set(state_obj.get_available_actions())
            transition_actions = set(self.transitions.get(state_name, {}).keys())

            extra_actions = state_actions - transition_actions
            if extra_actions:
                issues.append(f"State '{state_name}' has actions not in transitions: {extra_actions}")

            missing_actions = transition_actions - state_actions
            if missing_actions:
                issues.append(f"State '{state_name}' missing actions from transitions: {missing_actions}")

        # Check for unreachable states
        reachable = {self.current_state} if self.current_state else set()
        changed = True
        while changed:
            changed = False
            for from_state in list(reachable):
                for _, (target_state, fallback_state) in self.transitions.get(from_state, {}).items():
                    if target_state and target_state not in reachable:
                        reachable.add(target_state)
                        changed = True

                    if fallback_state and fallback_state not in reachable:
                        reachable.add(fallback_state)
                        changed = True

        unreachable = set(self.states.keys()) - reachable
        if unreachable:
            issues.append(f"Unreachable states: {unreachable}")

        # Report validation results
        if issues:
            for issue in issues:
                self.logger.error(f"FSM validation issue: {issue}")
            return False

        self.logger.info("FSM validation passed")
        return True


def create_minimal_fsm(config):
    """Create minimal FSM just two states"""
    logger = get_logger()
    logger.info("Creating FSM")

    fsm = FSM(config.get("FSM", {}))

    # Create and add states
    idle_state = fsm.add_state(IdleState(config.get('IdleState', {})))
    code_loaded_state = fsm.add_state(CodeLoadedState(config.get('CodeLoadedState', {})))

    # Create actions and add them to FSM
    fsm.add_action(idle_state, code_loaded_state,
                 SearchAction(config.get('SearchAction', {})),
                 fallback_state=idle_state)

    fsm.add_action(code_loaded_state, idle_state,
                 RetryAction(config.get('RetryAction', {})))

    # Set initial state and validate
    fsm.set_initial_state(idle_state)
    fsm.validate_fsm()

    logger.info("FSM created successfully")
    return fsm

def create_simple_fsm(config):
    """Create simple FSM with code editing and writing tests"""
    logger = get_logger()
    logger.info("Creating FSM")

    fsm = FSM(config.get("FSM", {}))

    # Create and add states
    idle_state = fsm.add_state(IdleState(config.get('IdleState', {})))
    code_loaded_state = fsm.add_state(CodeLoadedState(config.get('CodeLoadedState', {})))

    # Create actions and add them to FSM
    fsm.add_action(idle_state, code_loaded_state,
                 SearchAction(config.get('SearchAction', {})),
                 fallback_state=idle_state)

    fsm.add_action(code_loaded_state, idle_state,
                 RetryAction(config.get('RetryAction', {})))

    # Set initial state and validate
    fsm.set_initial_state(idle_state)
    fsm.validate_fsm()

    logger.info("FSM created successfully")
    return fsm

FSM_FACTORY = {
    'minimal': create_minimal_fsm,
    'simple': create_simple_fsm,
}
