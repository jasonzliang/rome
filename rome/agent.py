import os
import json
import traceback
from typing import Dict, List

# Import the OpenAIHandler we created
from .openai import OpenAIHandler
# Import default config utilities
from .config import DEFAULT_CONFIG, set_attributes_from_config
from .config import load_config, merge_with_default_config
# Import singleton logger
from .logger import get_logger


class Agent:
    """Agent class using OpenAI API, YAML config, and FSM architecture"""

    def __init__(self,
        name: str,
        role: str,
        config_dict: Dict = None):
        """Initialize the Agent with either a config path or a config dictionary"""

        # Setup logging first
        self.logger = get_logger()

        # Load configuration next
        if config_dict:
            self.config = merge_with_default_config(config_dict)
        else:
            self.logger.info("Using DEFAULT_CONFIG, no config dict provided")
            self.config = DEFAULT_CONFIG.copy()

        # Setup agent name and basic info
        self.name = name

        # Validate and properly format the role string
        self.role = self._validate_and_format_role(role)

        # Configure logger immediately after loading config
        self._setup_logging()

        # Get the Agent-specific configuration
        agent_config = self.config.get('Agent', {})
        # Automatically set attributes from Agent config
        set_attributes_from_config(self, agent_config)

        # Validate required attributes
        assert hasattr(self, 'repository'), "repository not provided in Agent config"
        assert self.repository is not None and os.path.exists(self.repository), \
            f"Repository path does not exist: {self.repository}"

        # Initialize context
        self.context = {}

        # Set up FSM
        self._setup_fsm()

        # Initialize OpenAI handler with OpenAIHandler config
        openai_config = self.config.get('OpenAIHandler', {})
        self.openai_handler = OpenAIHandler(config=openai_config)

        self.logger.info(f"Agent initialized with model: {openai_config.get('model')}")

    def _validate_and_format_role(self, role: str) -> str:
        """
        Validates and formats the agent's role string.

        Ensures that the role string contains 'your role' (case-insensitive)
        somewhere in the text. If not, it formats the role with a proper header.

        Args:
            role: The original role string

        Returns:
            A properly formatted role string
        """
        # Check if "your role" exists anywhere in the string (case insensitive)
        if "your role" in role.lower() or "you are" in role.lower():
            return role

        # If not found, add a properly formatted header
        self.logger.info("Role string does not contain 'your role', reformatting")
        return f"## Your role:\n{role}"

    def _setup_logging(self):
        """Configure logging based on config"""
        log_config = self.config.get('Logger', {})
        # Configure the singleton logger with the loaded config
        get_logger().configure(log_config)

    def _setup_fsm(self):
        """Initialize the Finite State Machine"""
        from .fsm import create_simple_fsm
        self.fsm = create_simple_fsm(config=self.config)
        self.logger.info(f"FSM initialized with state: {self.fsm.current_state}")

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
        return self.openai_handler.parse_json_response(response)

    def parse_python_response(self, response: str) -> str:
        """Parse Python code from response using the handler"""
        result = self.openai_handler.parse_python_response(response)
        return result or ""  # Return empty string instead of None to avoid errors

    def parse_python_response(self, response: str) -> str:
        """Parse Python code from response using the handler"""
        result = self.openai_handler.parse_python_response(response)
        if result is None:
            self.logger.info("Failed to parse Python code from response")
        return result or ""  # Return empty string instead of None to avoid errors

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
            return action

        # Fallback: look for action names in the response text
        available_actions = self.fsm.get_available_actions()
        for action in available_actions:
            if action.lower() in response.lower():
                self.logger.info(f"Extracted action from text: {action}")
                return action

        # If no action found, log the issue and return None
        self.logger.error(f"Could not extract valid action from response: {response}")
        return None

    def run_loop(self, max_iterations: int = 10, stop_on_error: bool = True) -> Dict:
        """
        Main execution loop that continuously executes actions based on FSM state

        Args:
            max_iterations: Maximum number of iterations to prevent infinite loops
            stop_on_error: Whether to stop the loop on errors or continue

        Returns:
            Dict containing loop execution results
        """
        self.logger.info(f"Starting agent loop for {max_iterations} iterations")

        results = {
            'iterations': 0,
            'actions_executed': [],
            'states_visited': [self.fsm.current_state],
            'errors': [],
            'final_state': None,
            'final_context': None
        }

        for iteration in range(max_iterations):
            try:
                results['iterations'] = iteration + 1
                self.logger.info(f"Loop iteration {iteration + 1}/{max_iterations}")
                self.logger.info(f"Current state: {self.fsm.current_state}")

                # Check if there are available actions
                available_actions = self.fsm.get_available_actions()
                if not available_actions:
                    self.logger.info("No available actions in current state. Stopping loop.")
                    break

                # Construct prompt combining role, state prompt, and available actions
                prompt = self.fsm.get_action_prompt(self)

                # Get action choice from LLM
                self.logger.info("Requesting action selection from LLM")
                response = self.chat_completion(
                    prompt=prompt,
                    system_message=self.role,
                    response_format={"type": "json_object"}
                )

                # Extract action from response
                chosen_action = self._extract_action_from_response(response)

                if not chosen_action:
                    error_msg = f"Could not determine action from LLM response: {response}"
                    self.logger.error(error_msg)
                    results['errors'].append({
                        'iteration': iteration + 1,
                        'error': error_msg,
                        'state': self.fsm.current_state
                    })
                    if stop_on_error:
                        break
                    continue

                # Validate action is available
                if chosen_action not in available_actions:
                    error_msg = f"Action '{chosen_action}' not available in state '{self.fsm.current_state}'. Available: {available_actions}"
                    self.logger.error(error_msg)
                    results['errors'].append({
                        'iteration': iteration + 1,
                        'error': error_msg,
                        'state': self.fsm.current_state
                    })
                    if stop_on_error:
                        break
                    continue

                # Execute the action through FSM
                self.logger.info(f"Executing action: {chosen_action}")
                self.fsm.execute_action(chosen_action, self)

                # Record the execution
                results['actions_executed'].append({
                    'iteration': iteration + 1,
                    'action': chosen_action,
                    'previous_state': results['states_visited'][-1],
                    'new_state': self.fsm.current_state
                })
                results['states_visited'].append(self.fsm.current_state)

                self.logger.info(f"Action executed successfully. New state: {self.fsm.current_state}")

            except Exception as e:
                error_msg = f"Error in loop iteration {iteration + 1}: {str(e)}"
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
                results['errors'].append({
                    'iteration': iteration + 1,
                    'error': error_msg,
                    'state': self.fsm.current_state,
                    'exception': str(e)
                })
                if stop_on_error:
                    break

        # Record final state and context
        results['final_state'] = self.fsm.current_state
        results['final_context'] = self.context.copy()

        self.logger.info(f"Agent loop completed after {results['iterations']} iterations")
        self.logger.info(f"Final state: {results['final_state']}")
        self.logger.info(f"Actions executed: {[action['action'] for action in results['actions_executed']]}")

        if results['errors']:
            self.logger.info(f"Loop completed with {len(results['errors'])} errors")

        return results
