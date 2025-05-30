import atexit
import json
import io
import os
import pprint
import re
import signal
import sys
import traceback
from typing import Dict, List
import yaml

# Import the OpenAIHandler we created
from .openai import OpenAIHandler
# Import default config utilities
from .config import DEFAULT_CONFIG, LOG_DIR_NAME, SUMMARY_LENGTH, AGENT_NAME_LENGTH
from .config import set_attributes_from_config, load_config, merge_with_default_config
# Import singleton logger
from .logger import get_logger
# Import FSM and factory
from .fsm import FSM
from .fsm_factory import FSM_FACTORY
# Import the new AgentHistory class
from .history import AgentHistory
# Import the VersionManager class
from .metadata import VersionManager
# Import parsing utility functions
from .parsing import parse_python_response, parse_json_response

# Make yaml use compact representation for lists
yaml.add_representer(list, lambda dumper, data: dumper.represent_sequence(
    'tag:yaml.org,2002:seq', data, flow_style=True))


class Agent:
    """Agent class using OpenAI API, YAML config, and FSM architecture"""

    def __init__(self,
        name: str = None,
        role: str = None,
        repository: str = None,
        config: Dict = None):
        """Initialize the Agent with configuration and setup all components"""

        # Core initialization
        self.logger = get_logger()
        self.shutdown_called = False
        self.curr_iteration = 1

        # Configuration setup
        self._setup_config(config)
        self._validate_name_role(name, role)

        # Repository validation and logging setup
        self._setup_repository_and_logging(repository)

        # Initialize core components
        self._setup_components()

        # Register cleanup handlers
        self._register_cleanup()

        self.logger.info(f"Agent {self.name} initialized with role:\n{self.role}")

    def _setup_config(self, config_dict: Dict = None) -> dict:
        """Setup and validate configuration"""
        if config_dict:
            self.config = merge_with_default_config(config_dict)
        else:
            self.logger.info("Using DEFAULT_CONFIG, no config dict provided")
            self.config = DEFAULT_CONFIG.copy()
        self.logger.info(f"Agent full config:\n{pprint.pformat(self.config)}")

        # Set attributes from Agent config
        agent_config = self.config.get('Agent', {})
        set_attributes_from_config(self, agent_config,
            ['name', 'role', 'repository', 'fsm_type', 'agent_api', 'history_context_len', 'patience'])

    def _validate_name_role(self, name: str, role: str) -> None:
        """Validates and formats the agent's role string"""
        if role:
            self.role = role
        if name:
            self.name = name
        self.logger.assert_true(self.role and self.name,
            f"Invalid name or role: {name}, {role}")

        # Make make role description clear what it is
        if "your role" not in self.role.lower():
            self.logger.info("Role string does not contain 'your role', reformatting")
            self.role = f"Your role as an agent:\n{self.role}"

        # Name must be between 8 and 24 char long and alphanum only
        a, b = AGENT_NAME_LENGTH
        # FIXED: Handle name validation properly
        clean_name = ''.join(re.findall(r'[a-zA-Z0-9]+', self.name))
        self.logger.assert_true(clean_name and a <= len(clean_name) <= b,
            f"Agent name must be {a}-{b} alphanumeric characters, got: '{self.name}' -> '{clean_name}'")
        self.name = clean_name

    def _setup_repository_and_logging(self, repository: str = None) -> None:
        """Validate repository and configure logging"""
        # Validate repository
        if repository:
            self.repository = repository
        self.logger.assert_true(
            self.repository and os.path.exists(self.repository),
            f"Repository path does not exist: {self.repository}"
        )

        # Setup logging after repository validation
        log_config = self.config.get('Logger', {}).copy()

        if not log_config.get('base_dir'):
            log_config['base_dir'] = self.get_log_dir()

        if not log_config.get('filename'):
            log_config['filename'] = f"{self.get_id()}.console.log"

        get_logger().configure(log_config)
        self.logger.info(f"Logging configured. Log directory: {log_config['base_dir']}")

    def _setup_components(self) -> None:
        """Initialize all agent components"""
        # FSM setup
        self._setup_fsm()

        # Core components
        self.context = {}
        self.history = AgentHistory()

        # Add initial FSM state to history
        if self.fsm and self.fsm.current_state:
            self.history.add_initial_state(self.fsm.current_state)

        # Version manager
        version_config = self.config.get('VersionManager', {})
        db_config = self.config.get('DatabaseManager', {})
        self.version_manager = VersionManager(config=version_config, db_config=db_config)

        # OpenAI handler
        openai_config = self.config.get('OpenAIHandler', {})
        self.openai_handler = OpenAIHandler(config=openai_config)

        # Optional API setup
        self._setup_agent_api()

    def _setup_agent_api(self) -> None:
        """Setup agent API if enabled"""
        if not self.agent_api:
            return
        from .agent_api import AgentApi
        api_config = self.config.get('AgentApi', {})
        self.agent_api = AgentApi(agent=self, config=api_config)
        self.agent_api.run()

    def _register_cleanup(self) -> None:
        """Register cleanup handlers for graceful shutdown"""
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully with clean stacktrace"""
        self.logger.info(f"Received signal {signum}, shutting down agent {self.name}")

        # Immediate location for quick reference
        if frame:
            filename = frame.f_code.co_filename
            line_number = frame.f_lineno
            function_name = frame.f_code.co_name
            self.logger.info(f"Interrupted at: {filename}:{line_number} in {function_name}()")

        # Capture traceback to string
        string_buffer = io.StringIO()
        traceback.print_stack(frame, file=string_buffer)
        stacktrace = string_buffer.getvalue()
        string_buffer.close()

        self.logger.info("Execution stack when interrupted:")
        self.logger.info(stacktrace)

        self.shutdown()
        sys.exit(0)

    def _setup_fsm(self):
        """Initialize the Finite State Machine"""
        self.logger.assert_true(
            self.fsm_type in FSM_FACTORY,
            f"{self.fsm_type} FSM is not defined in FSM_FACTORY, cannot be loaded"
        )

        self.fsm = FSM_FACTORY[self.fsm_type](self.config)
        self.logger.info(f"Initialized {self.fsm_type} FSM with state: {self.fsm.get_current_state()}")

    def get_id(self):
        """Unique id for identifying agent in file system"""
        # safe_name = ''.join(c if c.isalnum() else '_' for c in self.name).lower()
        return f'agent_{self.name}_{os.getpid()}'

    def get_log_dir(self):
        """Get agent log directory and create if it doesn't exist"""
        log_dir = os.path.join(self.repository, LOG_DIR_NAME)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def export_config(self, filepath: str = None):
        """Export agent configuration to YAML file"""
        if not filepath:
            filepath = os.path.join(self.get_log_dir(), f"{self.get_id()}.config.yaml")
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False, indent=4)
        self.logger.info(f"Agent configuration exported to: {filepath}")

    def shutdown(self):
        """Clean up resources before termination"""
        if self.shutdown_called: return
        self.shutdown_called = True
        try:
            if hasattr(self, 'agent_api') and self.agent_api:
                self.agent_api.shutdown()
            if hasattr(self, 'version_manager'):
                self.version_manager.shutdown(self)
            self.logger.info("Agent shutdown completed successfully")

        except Exception as e:
            self.logger.error(f"Error during agent shutdown: {e}")
            raise

    def draw_fsm_graph(self, output_path: str = None) -> str:
        """
        Draw the FSM graph to a PNG file
        """
        # If no output path specified, use the .rome directory
        if output_path is None:
            output_path = os.path.join(self.get_log_dir(),
                f"{self.get_id()}.fsm.png")

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

        # Use role as system message if not provided
        if system_message is None:
            system_message = self.role

        config = override_config if override_config else None

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
        result = parse_json_response(response)
        if result is None:
            self.logger.error("Failed to parse JSON object from response")
        return result

    def parse_python_response(self, response: str) -> str:
        """Parse Python code from response using the handler"""
        result = parse_python_response(response)
        if result is None:
            self.logger.error("Failed to parse Python code from response")
        return result

    def _extract_action_from_response(self, response: str) -> tuple:
        """
        Extract the action name from the LLM response
        """
        reasoning = "No reasoning provided"

        # Try to parse as JSON first
        parsed_json = self.parse_json_response(response)

        if parsed_json and 'selected_action' in parsed_json:
            action = parsed_json['selected_action']
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
        raise ValueError(f"Could not extract valid action from response: {response}")

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
        end_iteration = self.curr_iteration + max_iterations

        for iteration in range(self.curr_iteration, end_iteration):
            try:
                # Increment iteration counter in history, show iteration
                self.history.set_iteration(iteration)
                self.logger.info(f"Loop iteration {iteration}/{end_iteration-1}")
                self.logger.info(f"Current state: {self.fsm.current_state}")

                # Validate active file state for consistency and check completion
                self.version_manager.validate_active_files(self)
                self.version_manager.check_overall_completion(self)

                # Check agent context on first iteration to make sure state is valid
                if iteration == 1: self.fsm.check_context(self)
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
        self.curr_iteration = end_iteration
        self.history.set_final_state(self.fsm.current_state, self.context)

        print("\n\n")
        self.logger.info(f"Agent loop completed after {self.history.get_iteration()} iterations")
        self.logger.info(f"Final state: {self.history.final_state}")
        self.logger.info(f"Recent history:\n{self.history.get_history_summary(self.history_context_len)}")
        # action_sequence = [action['action'] for action in self.history.actions_executed][-SUMMARY_LENGTH:]
        # self.logger.info(f"Last {SUMMARY_LENGTH} actions executed: {action_sequence}")

        if self.history.has_errors():
            self.logger.info(f"Loop completed with {len(self.history.errors)} errors")

        # Return dictionary format for backward compatibility
        return self.history.to_dict()
