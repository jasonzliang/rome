# agent_fsm.py
from abc import ABC, abstractmethod
from typing import List, Dict, Type, Optional, Any


class StateBase(ABC):
    """Abstract base class for all states"""

    @abstractmethod
    def check_prerequisites(self, agent) -> bool:
        """Check if the agent has the necessary context to be in this state"""
        pass

    @abstractmethod
    def get_available_actions(self) -> List[str]:
        """Get list of actions available from this state"""
        pass

    @abstractmethod
    def on_enter(self, agent) -> None:
        """Called when entering this state"""
        pass

    @abstractmethod
    def on_exit(self, agent) -> None:
        """Called when exiting this state"""
        pass


class IdleState(StateBase):
    """Idle state - waiting for user input or task assignment"""

    def check_prerequisites(self, agent) -> bool:
        """In Idle state, agent should have basic initialization"""
        # Check that agent has logger
        if not hasattr(agent, 'logger') or agent.logger is None:
            return False

        # Check that agent has context dictionary
        if not hasattr(agent, 'context') or not isinstance(agent.context, dict):
            return False

        # Check that agent has repo_dir set
        if not hasattr(agent, 'repo_dir') or not agent.repo_dir:
            return False

        return True

    def get_available_actions(self) -> List[str]:
        """Actions available from Idle state"""
        return ["search"]

    def on_enter(self, agent) -> None:
        """Called when entering Idle state"""
        agent.logger.info("Entered IDLE state")
        # Clear any previous task-specific context
        if 'selected_file' in agent.context:
            del agent.context['selected_file']
        if 'error' in agent.context:
            del agent.context['error']

    def on_exit(self, agent) -> None:
        """Called when exiting Idle state"""
        agent.logger.info("Exiting IDLE state")


class SearchingState(StateBase):
    """Searching state - actively searching for relevant files"""

    def check_prerequisites(self, agent) -> bool:
        """Check prerequisites for Searching state"""
        # All Idle state prerequisites must be met
        idle_state = IdleState()
        if not idle_state.check_prerequisites(agent):
            return False

        # Check that agent has OpenAI handler for file selection
        if not hasattr(agent, 'openai_handler') or agent.openai_handler is None:
            agent.logger.error("Agent must have openai_handler for searching")
            return False

        # Check that search configuration is available
        if not hasattr(agent, 'search_config'):
            # Set default search config if missing
            agent.search_config = {}

        return True

    def get_available_actions(self) -> List[str]:
        """Actions available from Searching state"""
        # For now, no actions available while actively searching
        # In a real implementation, you might want to allow cancellation
        return []

    def on_enter(self, agent) -> None:
        """Called when entering Searching state"""
        agent.logger.info("Entered SEARCHING state")
        # Initialize search-specific context
        agent.context['search_started'] = True
        agent.context['files_processed'] = 0

    def on_exit(self, agent) -> None:
        """Called when exiting Searching state"""
        agent.logger.info("Exiting SEARCHING state")
        # Clean up search-specific context
        if 'search_started' in agent.context:
            del agent.context['search_started']


class FiniteStateMachine:
    """Implementation of a Finite State Machine for the Agent using class-based states"""

    def __init__(self, initial_state: Type[StateBase], config: Dict = None):
        """Initialize FSM with an initial state class and optional config"""
        self.current_state_class = initial_state
        self.current_state = initial_state()
        self.config = config or {}

        # Define state classes mapping
        self.state_classes = {
            'IDLE': IdleState,
            'SEARCHING': SearchingState,
        }

        # Define transitions between states
        self.transitions = {
            IdleState: {
                'search': SearchingState
            },
            SearchingState: {
                # Searching completes automatically, returns to IDLE
                # This would be handled by the search action itself
            }
        }

        self.actions = {}

        # Configure transition error handling
        self.strict_transitions = self.config.get('strict_transitions', True)

        # Get the default error state class name
        default_error_state_name = self.config.get('default_error_state', 'IDLE')
        self.default_error_state_class = self.state_classes.get(default_error_state_name, IdleState)

    def add_transition(self, from_state_class: Type[StateBase], action_name: str, to_state_class: Type[StateBase]) -> None:
        """Add a transition from one state class to another via an action"""
        if from_state_class not in self.transitions:
            self.transitions[from_state_class] = {}
        self.transitions[from_state_class][action_name] = to_state_class

    def add_action(self, action_name: str, action) -> None:
        """Register an action with the FSM"""
        self.actions[action_name] = action

    def get_available_actions(self) -> List[str]:
        """Get all available actions from the current state"""
        return self.current_state.get_available_actions()

    def transition_to_state(self, new_state_class: Type[StateBase], agent) -> bool:
        """Transition to a new state, checking prerequisites"""
        # Create new state instance
        new_state = new_state_class()

        # Check prerequisites
        if not new_state.check_prerequisites(agent):
            agent.logger.error(f"Prerequisites not met for state {new_state_class.__name__}")
            return False

        # Exit current state
        self.current_state.on_exit(agent)

        # Transition to new state
        self.current_state_class = new_state_class
        self.current_state = new_state

        # Enter new state
        self.current_state.on_enter(agent)

        return True

    def execute_action(self, action_name: str, agent, **kwargs) -> Type[StateBase]:
        """Execute an action and update the current state"""
        # Check if action is available from current state
        if action_name not in self.transitions.get(self.current_state_class, {}):
            if self.strict_transitions:
                raise ValueError(f"Action '{action_name}' is not available from state '{self.current_state_class.__name__}'")
            else:
                agent.logger.warning(
                    f"Action '{action_name}' is not available from state '{self.current_state_class.__name__}'. "
                    f"Continuing anyway due to non-strict transition policy."
                )

        # Check if action is registered
        if action_name not in self.actions:
            raise ValueError(f"Action '{action_name}' is not registered")

        # Get the target state class
        next_state_class = self.transitions.get(self.current_state_class, {}).get(action_name)

        if next_state_class:
            # Transition to the new state first
            if not self.transition_to_state(next_state_class, agent):
                # If transition fails, go to error state
                agent.context['error'] = f"Failed to transition to {next_state_class.__name__}"
                self.transition_to_state(self.default_error_state_class, agent)
                return self.current_state_class

        # Execute the action
        action = self.actions[action_name]
        try:
            # The action execution might return a next state or None
            result = action.execute(agent, **kwargs)

            # If action returns a specific state to transition to
            if result and hasattr(result, '__name__'):
                # Handle enum-style returns from existing actions
                if hasattr(result, 'name'):
                    state_name = result.name
                    if state_name in self.state_classes:
                        next_state_class = self.state_classes[state_name]
                        self.transition_to_state(next_state_class, agent)
                        return self.current_state_class

            # Default behavior: search action completes and returns to IDLE
            if action_name == 'search' and self.current_state_class == SearchingState:
                self.transition_to_state(IdleState, agent)

            return self.current_state_class

        except Exception as e:
            agent.logger.error(f"Error executing action '{action_name}': {str(e)}")
            agent.context['error'] = str(e)
            self.transition_to_state(self.default_error_state_class, agent)
            return self.current_state_class

    def get_current_state_name(self) -> str:
        """Get the name of the current state"""
        return self.current_state_class.__name__


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

    # Map state name to class
    state_classes = {
        'IDLE': IdleState,
        'SEARCHING': SearchingState,
    }

    initial_state_class = state_classes.get(initial_state_name, IdleState)

    # Create FSM with config
    fsm = FiniteStateMachine(initial_state=initial_state_class, config=fsm_config)

    # Get all available action classes
    action_classes = get_available_actions()

    # Register actions with their configs
    for action_name, action_class in action_classes.items():
        action_config = actions_config.get(action_name, {})
        fsm.add_action(action_name, action_class(config=action_config))

    # Set up transitions - already defined in FSM constructor
    # You can add more transitions here if needed

    return fsm
