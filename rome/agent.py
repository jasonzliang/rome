import atexit
from datetime import datetime
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
from .fsm_selector import FSMSelector
# Import the new AgentHistory class
from .history import AgentHistory
# Import the Repository manager
from .repository import RepositoryManager
# Import the VersionManager class
from .metadata import VersionManager
# Import parsing utility functions
from .parsing import parse_python_response, parse_json_response
# Import the new ActionSelector
from .action_selector import ActionSelector

# Make yaml use compact representation for lists
yaml.add_representer(list, lambda dumper, data: dumper.represent_sequence(
    'tag:yaml.org,2002:seq', data, flow_style=True))


# TODO: create a repository manager object that keeps track of global statistics like file completion, eligible files for editing, number of attempts on each file, etc, take off some of the load from version manager
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

        # Configuration setup
        self._setup_config(config)
        self._validate_name_role(name, role)

        # Repository validation and logging setup
        self._setup_repository_and_logging(repository)

        # Initialize core components
        self._setup_fsm()
        self._setup_context_history()
        self._setup_repository_manager()
        self._setup_version_manager()
        self._setup_openai_handler()
        self._setup_action_selection()
        self._setup_agent_api()

        # Register cleanup handlers
        self._register_cleanup()

        # Export agent configuration and draw graph
        self.export_config()
        self.draw_fsm_graph()

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
            ['name', 'role', 'repository', 'fsm_type', 'agent_api', 'history_context_len', 'patience', 'action_select_strat'])

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

    def _setup_fsm(self):
        """Initialize the Finite State Machine using FSMSelector"""
        fsm_selector = FSMSelector(fsm_type=self.fsm_type, config=self.config)
        self.fsm = fsm_selector.create_fsm(self.config)
        self.logger.info(f"Initialized {self.fsm_type} FSM using FSMSelector: {self.fsm.get_current_state()}")
        self.logger.info(f"FSM Description: {fsm_selector.get_description()}")

    def _setup_context_history(self) -> None:
        """Initialize core agent components"""
        self.context = {}
        self.history = AgentHistory()
        self.summary_history = []
        self.curr_iteration = 1

        # Add initial FSM state to history if FSM is initialized
        if self.fsm and self.fsm.current_state:
            self.history.add_initial_state(self.fsm.current_state)

    def _setup_repository_manager(self) -> None:
        """Initialize repository manager with configuration"""
        repository_config = self.config.get('RepositoryManager', {})
        self.repository_manager = RepositoryManager(
            repository_path=self.repository,
            config=repository_config
        )
        self.logger.debug("Repository manager initialized")

    def _setup_version_manager(self) -> None:
        """Initialize version manager with database configuration"""
        version_config = self.config.get('VersionManager', {})
        db_config = self.config.get('DatabaseManager', {})
        self.version_manager = VersionManager(
            config=version_config,
            db_config=db_config
        )
        self.logger.debug("Version manager initialized")

    def _setup_openai_handler(self) -> None:
        """Initialize OpenAI handler with configuration"""
        openai_config = self.config.get('OpenAIHandler', {})
        self.openai_handler = OpenAIHandler(config=openai_config)
        self.logger.debug("OpenAI handler initialized")

    def _setup_action_selection(self) -> None:
        """Setup action selection strategy"""
        selection_config = self.config.get('ActionSelector', {})
        strategy_config = selection_config.get(self.action_select_strat, {})

        self.action_selector = ActionSelector(
            strategy=self.action_select_strat,
            config=strategy_config
        )
        self.logger.info(f"Action selector: {self.action_select_strat}")

    def _setup_agent_api(self) -> None:
        """Setup agent API if enabled"""
        if not self.agent_api:
            self.logger.debug("Agent API disabled, skipping initialization")
            return

        from .agent_api import AgentApi
        api_config = self.config.get('AgentApi', {})
        self.agent_api = AgentApi(agent=self, config=api_config)
        self.agent_api.run()
        self.logger.info("Agent API initialized and running")

    def _register_cleanup(self) -> None:
        """Register cleanup handlers for graceful shutdown"""
        self.shutdown_called = False

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
            if hasattr(self, 'repository_manager'):
                self.logger.info("Repository manager shutdown completed")
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

    def _summary(self):
        summary = self.get_summary()
        self.print_summary(summary)
        self.save_summary(summary)

    def run_loop(self, max_iterations: int = 10, stop_on_error: bool = True) -> Dict:
        """
        Main execution loop that continuously executes actions based on FSM state
        Now uses ActionSelector for cleaner action selection logic

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
        start_iteration = self.curr_iteration
        end_iteration = self.curr_iteration + max_iterations

        for iteration in range(start_iteration, end_iteration):
            try:
                # Increment iteration counter in history, show iteration
                self.curr_iteration = iteration
                self.history.set_iteration(iteration)
                self.logger.info(f"Loop iteration {iteration}/{end_iteration-1}")
                self.logger.info(f"Current state: {self.fsm.current_state}")

                # Validate active file state for consistency
                self.version_manager.validate_active_files(self)

                # Print summary and write to file
                self._summary()

                # Check agent context on first iteration to make sure state is valid
                if iteration == 1:
                    self.fsm.check_context(self)

                # NEW: Use ActionSelector for all action selection logic
                chosen_action, reason, should_continue = self.action_selector.select_action(
                    agent=self,
                    iteration=iteration,
                    stop_on_error=stop_on_error
                )

                # Handle action selection results
                if not should_continue:
                    if chosen_action is None:
                        break  # Exit loop due to error or no actions
                    else:
                        continue  # Continue to next iteration (reset case)

                # Store previous state for history
                prev_state = self.fsm.current_state

                # Execute the action through FSM
                success = self.fsm.execute_action(chosen_action, self)

                # Record the execution in history
                self.history.add_action_execution(
                    iteration=iteration,
                    action=chosen_action,
                    action_result=success,
                    action_reason=reason,
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
        if self.history.has_errors():
            self.logger.info(f"Loop completed with {len(self.history.errors)} errors")
        self._summary()

        # Set current iteration for next run_loop call
        self.curr_iteration = end_iteration

        # Return dictionary format for backward compatibility
        return self.history.to_dict()

    def _summary(self):
        summary = self.get_summary()
        self.print_summary(summary)

        # Store summary in history
        self.summary_history.append({
            'iteration': self.curr_iteration,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'summary': summary
        })

        self.save_summary(summary)

    def get_summary(self, recent_history=None) -> str:
        """Get a comprehensive summary of the agent's current state."""
        def _get_context_keys(obj, prefix="", depth=0):
            if depth >= 2 or not isinstance(obj, dict):
                return []

            keys = []
            for k, v in obj.items():
                full_key = f"{prefix}{k}"
                if isinstance(v, dict):
                    keys.extend(_get_context_keys(v, f"{full_key}.", depth + 1))
                else:
                    keys.append(full_key)
            return keys

        if not recent_history:
            recent_history = self.history_context_len

        # Basic agent info
        summary_lines = []
        summary_lines.append(f"Agent: {self.name}")
        summary_lines.append(f"Current State: {self.fsm.current_state}")
        summary_lines.append(f"Iteration: {self.curr_iteration}")
        summary_lines.append(f"Repository: {self.repository}")

        # Context info
        ctx_keys = _get_context_keys(self.context)
        summary_lines.append(f"Context Keys:\n  {'\n  '.join(ctx_keys)}")


        # Enhanced history summary with more details
        actions_count = len(self.history.actions_executed)
        errors_count = len(self.history.errors)
        summary_lines.append(f"Actions Executed: {actions_count}")
        summary_lines.append(f"Errors: {errors_count}")

        # Add recent actions summary
        if self.history.actions_executed:
            recent_actions = self.history.actions_executed[-recent_history:]
            action_summary = {}
            for action_data in recent_actions:
                action_name = action_data.get('action', 'unknown')
                action_summary[action_name] = action_summary.get(action_name, 0) + 1

            summary_lines.append(f"Recent Actions ({len(recent_actions)} of {actions_count}):")
            for action, count in action_summary.items():
                summary_lines.append(f"  {action}: {count}")

        # Add success rate if we have actions
        if actions_count > 0:
            successful_actions = sum(1 for action in self.history.actions_executed
                                   if action['action_result'] == 'success')
            success_rate = (successful_actions / actions_count) * 100
            summary_lines.append(f"Action Success Rate: {success_rate:.1f}% ({successful_actions}/{actions_count})")

        # Repository completion stats - SINGLE LINE FORMAT
        completion = self.repository_manager.get_repository_completion_stats(self)
        summary_lines.append(f"Repository Progress: {completion['finished_files']}/{completion['total_files']} files ({completion['completion_percentage']}% complete)")

        # OpenAI cost summary
        summary_lines.append("OpenAI Cost Summary")
        cost_summary = self.openai_handler.get_cost_summary()

        summary_lines.append(f"  Model: {cost_summary['model']} | Calls: {cost_summary['call_count']} | Cost: ${cost_summary['accumulated_cost']:.2f}")

        if cost_summary['cost_limit']:
            remaining = cost_summary['remaining_budget']
            usage_pct = (cost_summary['accumulated_cost'] / cost_summary['cost_limit']) * 100
            summary_lines.append(f"  Budget: ${remaining:.2f} remaining of ${cost_summary['cost_limit']:.2f} ({usage_pct:.1f}% used)")

        return '\n'.join(summary_lines)

    def print_summary(self, summary):
        """Print agent summary to console."""
        self.logger.info("="*80)
        self.logger.info("AGENT SUMMARY")
        self.logger.info("="*80)
        for line in summary.split('\n'):
            self.logger.info(line)
        self.logger.info("="*80)

    def save_summary(self, current_summary, recent_history=None):
        """Save last 10 summaries to file in log directory."""
        if not recent_history:
            recent_history = self.history_context_len

        log_dir = self.get_log_dir()
        summary_file = os.path.join(log_dir, f"{self.get_id()}.summary.log")

        # Keep only last 10 summaries
        recent_summaries = self.summary_history[-recent_history:]

        # Write all recent summaries to file (overwrite mode)
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"AGENT SUMMARY HISTORY (Last {len(recent_summaries)} iterations)\n")
            f.write("="*80 + "\n\n")

            for summary_entry in recent_summaries:
                f.write(f"[Iteration {summary_entry['iteration']} - {summary_entry['timestamp']}]\n")
                f.write(summary_entry['summary'] + "\n")
                f.write("-"*40 + "\n\n")

        self.logger.debug(f"Summary history ({len(recent_summaries)} entries) saved to: {summary_file}")
