# agent_fsm.py
from enum import Enum, auto
from typing import List, Dict


class AgentState(Enum):
    """States for the Agent FSM"""
    IDLE = auto()
    SEARCHING = auto()
    ANALYZING = auto()
    UPDATING = auto()
    ERROR = auto()
    FILE_OPERATION = auto()  # Added for file load/save operations


class FiniteStateMachine:
    """Implementation of a Finite State Machine for the Agent"""

    def __init__(self, initial_state: AgentState, config: Dict = None):
        """Initialize FSM with an initial state and optional config"""
        self.current_state = initial_state
        self.config = config or {}
        self.transitions = {
            AgentState.IDLE: {},
            AgentState.SEARCHING: {},
            AgentState.ANALYZING: {},
            AgentState.UPDATING: {},
            AgentState.ERROR: {},
            AgentState.FILE_OPERATION: {},
        }
        self.actions = {}

        # Configure transition error handling
        self.strict_transitions = self.config.get('strict_transitions', True)

        # Get the default error state
        default_error_state_name = self.config.get('default_error_state', 'ERROR')
        if isinstance(default_error_state_name, str):
            try:
                self.default_error_state = getattr(AgentState, default_error_state_name)
            except AttributeError:
                self.default_error_state = AgentState.ERROR
        else:
            self.default_error_state = default_error_state_name or AgentState.ERROR

    def add_transition(self, from_state: AgentState, action_name: str, to_state: AgentState) -> None:
        """Add a transition from one state to another via an action"""
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        self.transitions[from_state][action_name] = to_state

    def add_action(self, action_name: str, action) -> None:
        """Register an action with the FSM"""
        self.actions[action_name] = action

    def get_available_actions(self) -> List[str]:
        """Get all available actions from the current state"""
        return list(self.transitions[self.current_state].keys())

    def execute_action(self, action_name: str, agent, **kwargs) -> AgentState:
        """Execute an action and update the current state"""
        # Check if action is available from current state
        if action_name not in self.transitions[self.current_state]:
            if self.strict_transitions:
                raise ValueError(f"Action '{action_name}' is not available from state '{self.current_state}'")
            else:
                agent.logger.warning(
                    f"Action '{action_name}' is not available from state '{self.current_state}'. "
                    f"Continuing anyway due to non-strict transition policy."
                )

        # Check if action is registered
        if action_name not in self.actions:
            raise ValueError(f"Action '{action_name}' is not registered")

        # Execute the action
        action = self.actions[action_name]
        try:
            next_state = action.execute(agent, **kwargs)
            self.current_state = next_state
            return next_state
        except Exception as e:
            agent.logger.error(f"Error executing action '{action_name}': {str(e)}")
            agent.context['error'] = str(e)
            self.current_state = self.default_error_state
            return self.default_error_state


def setup_default_fsm(config: Dict = None):
    """Create and return a default configured FSM"""
    from agent_actions import get_available_actions

    # Default config if none provided
    config = config or {}

    # Extract specific configs for each component
    fsm_config = config.get('fsm', {})
    actions_config = config.get('actions', {})

    # Determine initial state
    initial_state_name = fsm_config.get('initial_state', 'IDLE')
    if isinstance(initial_state_name, str):
        try:
            initial_state = getattr(AgentState, initial_state_name)
        except AttributeError:
            initial_state = AgentState.IDLE
    else:
        initial_state = initial_state_name or AgentState.IDLE

    # Create FSM with config
    fsm = FiniteStateMachine(initial_state=initial_state, config=fsm_config)

    # Get all available action classes
    action_classes = get_available_actions()

    # Register actions with their configs
    for action_name, action_class in action_classes.items():
        action_config = actions_config.get(action_name, {})
        fsm.add_action(action_name, action_class(config=action_config))

    # Set up transitions - from IDLE state
    fsm.add_transition(AgentState.IDLE, "search", AgentState.SEARCHING)
    fsm.add_transition(AgentState.IDLE, "update", AgentState.UPDATING)
    fsm.add_transition(AgentState.IDLE, "load_file", AgentState.FILE_OPERATION)
    fsm.add_transition(AgentState.IDLE, "save_file", AgentState.FILE_OPERATION)

    # From SEARCHING state
    fsm.add_transition(AgentState.SEARCHING, "analyze", AgentState.ANALYZING)
    fsm.add_transition(AgentState.SEARCHING, "search", AgentState.SEARCHING)  # Can search again

    # From ANALYZING state
    fsm.add_transition(AgentState.ANALYZING, "update", AgentState.UPDATING)
    fsm.add_transition(AgentState.ANALYZING, "search", AgentState.SEARCHING)
    fsm.add_transition(AgentState.ANALYZING, "analyze", AgentState.ANALYZING)  # Can analyze again

    # From UPDATING state
    fsm.add_transition(AgentState.UPDATING, "search", AgentState.SEARCHING)
    fsm.add_transition(AgentState.UPDATING, "analyze", AgentState.ANALYZING)
    fsm.add_transition(AgentState.UPDATING, "update", AgentState.UPDATING)  # Can update again

    # From FILE_OPERATION state
    fsm.add_transition(AgentState.FILE_OPERATION, "search", AgentState.SEARCHING)
    fsm.add_transition(AgentState.FILE_OPERATION, "update", AgentState.UPDATING)
    fsm.add_transition(AgentState.FILE_OPERATION, "load_file", AgentState.FILE_OPERATION)
    fsm.add_transition(AgentState.FILE_OPERATION, "save_file", AgentState.FILE_OPERATION)

    # From ERROR state (recovery transitions)
    fsm.add_transition(AgentState.ERROR, "search", AgentState.SEARCHING)
    fsm.add_transition(AgentState.ERROR, "update", AgentState.UPDATING)
    fsm.add_transition(AgentState.ERROR, "load_file", AgentState.FILE_OPERATION)
    fsm.add_transition(AgentState.ERROR, "save_file", AgentState.FILE_OPERATION)

    return fsm
