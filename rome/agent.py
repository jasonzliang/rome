import os
import json
import traceback
from typing import Dict, List

# Import the OpenAIHandler we created
from .openai import OpenAIHandler
# Import default config utilities
from .config import DEFAULT_CONFIG, LOG_DIR_NAME
from .config import set_attributes_from_config, load_config, merge_with_default_config
# Import singleton logger
from .logger import get_logger
from .fsm import FSM, FSM_FACTORY
# Import the new AgentHistory class
from .history import AgentHistory
# Import the VersionManager class
from .versioning import VersionManager

class Agent:
    """Agent class using OpenAI API, YAML config, and FSM architecture"""

    def __init__(self,
        name: str = "CodingExpert",
        role: str = "You are a Python coding expert",
        config_dict: Dict = None):
        """Initialize the Agent with either a config path or a config dictionary"""

        # Setup logging first
        self.logger = get_logger()

        # Store the name for later use with logging
        self.name = name

        # Load configuration next
        if config_dict:
            self.config = merge_with_default_config(config_dict)
        else:
            self.logger.info("Using DEFAULT_CONFIG, no config dict provided")
            self.config = DEFAULT_CONFIG.copy()

        # Validate and properly format the role string
        self.role = self._validate_and_format_role(role)

        # Set up agent configuration
        agent_config = self.config.get('Agent', {})
        # Automatically set attributes from Agent config
        set_attributes_from_config(self, agent_config, ['repository', 'fsm_type', 'agent_api', 'history_context_len'])

        # Validate repository attribute
        self.logger.assert_attribute(self, 'repository', "repository not provided in Agent config")
        self.logger.assert_true(
            self.repository is not None and os.path.exists(self.repository),
            f"Repository path does not exist: {self.repository}"
        )

        # Configure logging after repository is validated
        self._setup_logging()

        # Set up FSM
        if self.fsm_type:
            self._setup_fsm()
        else:
            self.fsm = None

        # Initialize context and history using the new AgentHistory class
        self.context = {}
        self.history = AgentHistory()

        # Initialize VersionManager
        version_config = self.config.get('VersionManager', {})
        self.version_manager = VersionManager(config=version_config)

        # Add initial state to history if FSM is set up
        if self.fsm and self.fsm.current_state:
            self.history.add_initial_state(self.fsm.current_state)

        # Initialize OpenAI handler with OpenAIHandler config
        openai_config = self.config.get('OpenAIHandler', {})
        self.openai_handler = OpenAIHandler(config=openai_config)

        if self.agent_api:
            from .agent_api import AgentApi
            api_config = self.config.get('AgentApi', {})
            self.api = AgentApi(agent=self, config=api_config)
            self.api.run()

        self.logger.info(f"Agent {self.name} initialized with role:\n{self.role}")


    def shutdown(self):
        """Clean up resources before termination"""
        self.logger.info("Cleaning up agent resources")
        if self.api:
            self.api.shutdown()


    def _validate_and_format_role(self, role: str) -> str:
        """Validates and formats the agent's role string. If not, it formats the role with a proper header."""

        # Check if "your role" exists anywhere in the string (case insensitive)
        if "your role" in role.lower():
            return role

        # If not found, add a properly formatted header
        self.logger.info("Role string does not contain 'your role', reformatting")
        return f"Your role:\n{role}"

    def _get_agent_id(self):
        """Unique id for identifying agent in file system"""
        safe_name = ''.join(c if c.isalnum() else '_' for c in self.name).lower()
        return f'agent_{safe_name}_{os.getpid()}'

    def _setup_logging(self):
        """Configure logging based on config"""
        # Get the logging configuration
        log_config = self.config.get('Logger', {}).copy()

        # Set base_dir and filename if not already set
        if not log_config.get('base_dir'):
            log_config['base_dir'] = os.path.join(self.repository, LOG_DIR_NAME)

        if not log_config.get('filename'):
            # Ensure agent name is suitable for a filename
            log_config['filename'] = f"{self._get_agent_id()}.log"

        # Configure the singleton logger with the loaded config
        get_logger().configure(log_config)

        self.logger.info(f"Logging configured. Log directory: {log_config['base_dir']}")

    def _setup_fsm(self):
        """Initialize the Finite State Machine"""
        self.logger.assert_true(
            self.fsm_type in FSM_FACTORY,
            f"{self.fsm_type} FSM is not defined, cannot be loaded"
        )

        self.fsm = FSM_FACTORY[self.fsm_type](config=self.config)
        self.logger.info(f"Initialized {self.fsm_type} FSM with state: {self.fsm.get_current_state()}")

    def set_fsm(self, fsm: FSM):
        self.fsm = fsm
        self.logger.info(f"Initialized user FSM with state: {self.fsm.get_current_state()}")
        # Add initial state to history
        if self.fsm.current_state:
            self.history.add_initial_state(self.fsm.current_state)

    def draw_fsm_graph(self, output_path: str = None) -> str:
        """
        Draw the FSM graph to a PNG file
        """
        # If no output path specified, use the .rome directory
        if output_path is None:
            output_path = os.path.join(self.logger.get_log_dir(),
                f"fsm_graph_{self._get_agent_id()}.png")

        # Draw the graph using the FSM method
        self.logger.info(f"Drawing FSM graph to {output_path}")
        return self.fsm.draw_graph(output_path)

    # Chat completion methods
    def chat_completion(self, prompt: str,
        system_message: str = None,
        override_config: Dict = None,
        response_format: Dict = None,
        extra_body: Dict = None) -> str:
        """
        Direct access to chat completion with configuration options

        Args:
            prompt: The user prompt
            system_message: Optional system message
            action_type: Use action-specific config (not used anymore)
            override_config: Dictionary to override any config parameters
            response_format: Optional response format (e.g., {"type": "json_object"})
            extra_body: Additional parameters to pass to the API

        Returns:
            The response content as string
        """
        # Set up base config
        config = {}

        # Use action-specific system message if not provided
        if system_message is None:
            system_message = self.role

        # Merge with any overrides
        if override_config:
            config.update(override_config)

        return self.openai_handler.chat_completion(
            prompt=prompt,
            system_message=system_message,
            override_config=config,
            response_format=response_format,
            extra_body=extra_body
        )

    # Utility methods
    def parse_json_response(self, response: str) -> Dict:
        """Parse JSON response using the handler"""
        result = self.openai_handler.parse_json_response(response)
        if result is None:
            self.logger.error("Failed to parse JSON object from response")
        return result

    def parse_python_response(self, response: str) -> str:
        """Parse Python code from response using the handler"""
        result = self.openai_handler.parse_python_response(response)
        if result is None:
            self.logger.error("Failed to parse Python code from response")
        return result

    def _extract_action_from_response(self, response: str) -> str:
        """
        Extract the action name from the LLM response
        """
        # Try to parse as JSON first
        parsed_json = self.parse_json_response(response)

        if parsed_json and 'action' in parsed_json:
            action = parsed_json['action']
            reasoning = parsed_json.get('reasoning', 'No reasoning provided')
            self.logger.info(f"Selected action: {action} - {reasoning}")
            return action, reasoning

        # Fallback: look for action names in the response text
        available_actions = self.fsm.get_available_actions()
        for action in available_actions:
            if action.lower() in response.lower():
                self.logger.info(f"Extracted action from text: {action}")
                return action, reasoning

        # If no action found, log the issue and return None
        self.logger.error(f"Could not extract valid action from response: {response}")
        raise

    def run_loop(self, max_iterations: int = 10, stop_on_error: bool = True) -> Dict:
        """
        Main execution loop that continuously executes actions based on FSM state

        Args:
            max_iterations: Maximum number of iterations to prevent infinite loops
            stop_on_error: Whether to stop the loop on errors or continue

        Returns:
            Dict containing loop execution results
        """
        self.logger.assert_true(
            self.fsm is not None,
            "FSM has not been properly initialized"
        )

        self.logger.info(f"Starting agent loop for {max_iterations} iterations")

        for iteration in range(1, max_iterations+1):
            try:
                # Check agent context on first iteration to make sure state is valid
                if iteration == 1:
                    self.fsm.check_context(self)

                # Increment iteration counter in history
                self.history.set_iteration(iteration)

                self.logger.info(f"Loop iteration {iteration}/{max_iterations}")
                self.logger.info(f"Current state: {self.fsm.current_state}")

                # Check if there are available actions
                available_actions = self.fsm.get_available_actions()
                if not available_actions:
                    self.logger.error("No available actions in current state. Stopping loop.")
                    if stop_on_error:
                        break
                    else:
                        self.fsm.reset(self)
                        self.history.reset()
                        continue
                else:
                    self.logger.info(f"Available actions from {self.fsm.get_current_state()}: {available_actions}")

                # If only one action available, select it directly without LLM call
                if len(available_actions) == 1:
                    chosen_action = available_actions[0]
                    reasoning = "only one action available - auto-selected"
                    self.logger.info(f"Auto-selecting single available action: {chosen_action}")
                else:
                    # Construct prompt combining role, state prompt, and available actions
                    prompt = self.fsm.get_action_selection_prompt(self)

                    # Get action choice from LLM
                    self.logger.info("Requesting action selection from LLM")
                    response = self.chat_completion(
                        prompt=prompt,
                        system_message=self.role,
                        response_format={"type": "json_object"}
                    )

                    # Extract action from response
                    chosen_action, reasoning = self._extract_action_from_response(response)

                    if not chosen_action:
                        error_msg = f"Could not determine action from LLM response: {response}"
                        self.logger.error(error_msg)
                        self.history.add_error(iteration, error_msg, self.fsm.current_state)
                        if stop_on_error:
                            break
                        else:
                            self.fsm.reset(self)
                            self.history.reset()
                            continue

                    # Validate action is available
                    if chosen_action not in available_actions:
                        error_msg = f"Action '{chosen_action}' not available in state '{self.fsm.current_state}'. Available: {available_actions}"
                        self.logger.error(error_msg)
                        self.history.add_error(iteration, error_msg, self.fsm.current_state)
                        if stop_on_error:
                            break
                        else:
                            self.fsm.reset(self)
                            self.history.reset()
                            continue

                # Store previous state for history
                prev_state = self.fsm.current_state

                # Execute the action through FSM
                self.logger.info(f"Executing action: {chosen_action}")
                success = self.fsm.execute_action(chosen_action, self)

                # Record the execution in history
                self.history.add_action_execution(
                    iteration=iteration,
                    action=chosen_action,
                    result=success,
                    reasoning=reasoning,
                    prev_state=prev_state,
                    curr_state=self.fsm.current_state
                )

                self.logger.info(f"Action executed successfully. New state: {self.fsm.current_state}")

            except Exception as e:
                error_msg = f"Error in loop iteration {iteration}: {str(e)}"
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
                self.history.add_error(iteration, error_msg, self.fsm.current_state, str(e))
                if stop_on_error:
                    break
                else:
                    self.fsm.reset(self)
                    self.history.reset()
                    continue

        # Record final state and context
        self.history.set_final_state(self.fsm.current_state, self.context)

        self.logger.info(f"\n\nAgent loop completed after {self.history.get_iteration()} iterations")
        self.logger.info(f"Final state: {self.history.final_state}")

        action_sequence = [action['action'] for action in self.history.actions_executed]
        self.logger.info(f"Actions executed: {action_sequence}")

        if self.history.has_errors():
            self.logger.info(f"Loop completed with {len(self.history.errors)} errors")

        # Return dictionary format for backward compatibility
        return self.history.to_dict()