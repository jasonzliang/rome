import atexit
from datetime import datetime
import json
import io
import os
import pprint
import re
import sys
import time
import traceback
from typing import Dict, List, Callable
import yaml

# Import the OpenAIHandler we created
from .openai import OpenAIHandler
# Import default config utilities
from .config import DEFAULT_CONFIG, LOG_DIR_NAME, SUMMARY_LENGTH, AGENT_NAME_LENGTH
from .config import set_attributes_from_config, load_config, merge_with_default_config, format_yaml_like
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
# Import the process management decorator
from .process import process_managed
# Import knowledge base tools
from .kb_client import ChromaClientManager

# Make yaml use compact representation for lists
yaml.add_representer(list, lambda dumper, data: dumper.represent_sequence(
    'tag:yaml.org,2002:seq', data, flow_style=True))


@process_managed
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
        self._setup_openai_handler()
        self._setup_context_history()
        self._setup_repository_manager()
        self._setup_version_manager()
        self._setup_action_selection()
        self._setup_knowledge_base()
        self._setup_callback()
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
            ['name', 'role', 'repository', 'fsm_type', 'agent_api', 'history_context_len', 'patience',
             'action_select_strat', 'log_pid', 'save_hist_interval', 'use_ground_truth',
             'save_insights', 'query_insights'])

        self.logger.assert_true(self.history_context_len > 0,
            f"history_context_len must be greater than 0")
        self.logger.assert_true(self.save_hist_interval > 0,
            f"save_hist_interval must be greater than 0")

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

        # Name must be between 8 and 32 char long and alphanum only
        a, b = AGENT_NAME_LENGTH
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

    def _load_summary_history(self) -> None:
        """Load iteration number, summary history, and OpenAI cost from saved files"""
        try:
            summary_file = os.path.join(self.get_log_dir(), f"{self.get_id()}.summary.json")
            history_file = os.path.join(self.get_log_dir(), f"{self.get_id()}.summary_history.json")

            # Load current iteration from summary file
            iteration = None; accumulated_cost = None; call_count = None

            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                iteration = summary_data.get('iteration')

                # Extract OpenAI cost data from the most recent summary
                openai_cost = summary_data.get('summary', {}).get('openai_cost', {})
                accumulated_cost = openai_cost.get('accumulated_cost')
                call_count = openai_cost.get('call_count')
            else:
                self.logger.info("No summary file found, starting fresh at iteration 1")

            # Load summary history if available
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                self.summary_history = history_data.get('summary_history', [])
                self.logger.info(f"Loaded {len(self.summary_history)} summary history entries")
            else:
                self.summary_history = []
                self.logger.info("No summary history file found, starting with empty history")

            # Set the accumulated cost and call count in OpenAI handler if it exists
            if accumulated_cost is not None and call_count is not None:
                self.openai_handler.accumulated_cost = accumulated_cost
                self.openai_handler.call_count = call_count
                self.logger.info(f"Loaded OpenAI API cost from summary: ${accumulated_cost:.4f}")

            # Set current iteration
            if iteration is not None:
                self.curr_iteration = iteration
                self.logger.info(f"Loaded current iteration from summary: {iteration}")
            else:
                self.curr_iteration = 1

        except Exception as e:
            self.logger.error(f"Error loading summary data: {e}, defaulting to iteration 1")
            self.summary_history = []
            self.curr_iteration = 1

    def _setup_context_history(self) -> None:
        """Simplified setup - loads iteration and summary history"""
        # Set current iteration and OpenAI API cost from summary history json
        self._load_summary_history()

        self.context = {}
        self.history = AgentHistory()
        self.history.set_iteration(self.curr_iteration)

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
        # Get the entire ActionSelector config block
        action_selector_config = self.config.get('ActionSelector', {})

        self.action_selector = ActionSelector(
            strategy=self.action_select_strat,
            config=action_selector_config  # Pass entire config block
        )
        self.logger.info(f"Action selector: {self.action_select_strat}")

    def _setup_knowledge_base(self) -> None:
        """Setup knowledge base if enabled"""
        self.kb_manager = ChromaClientManager(agent=self)
        self.logger.info(f"Knowledge base initialized: {self.kb_manager.info()}")

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

    def _setup_callback(self):
        """Setup a callback variable for running at end of iteration"""
        self.callback = None

    def _register_cleanup(self) -> None:
        """Register cleanup handlers for graceful shutdown"""
        self.shutdown_called = False

    #     atexit.register(self.shutdown)
    #     signal.signal(signal.SIGTERM, self._signal_handler)
    #     signal.signal(signal.SIGINT, self._signal_handler)

    # def _signal_handler(self, signum, frame):
    #     """Handle termination signals gracefully with clean stacktrace"""
    #     self.logger.info(f"Received signal {signum}, shutting down agent {self.name}")

    #     # Immediate location for quick reference
    #     if frame:
    #         filename = frame.f_code.co_filename
    #         line_number = frame.f_lineno
    #         function_name = frame.f_code.co_name
    #         self.logger.info(f"Interrupted at: {filename}:{line_number} in {function_name}()")

    #     # Capture traceback to string
    #     string_buffer = io.StringIO()
    #     traceback.print_stack(frame, file=string_buffer)
    #     stacktrace = string_buffer.getvalue()
    #     string_buffer.close()

    #     self.logger.info("Execution stack when interrupted:")
    #     self.logger.info(stacktrace)

    #     self.shutdown()
    #     sys.exit(0)

    def get_id(self):
        """Unique id for identifying agent in file system"""
        # safe_name = ''.join(c if c.isalnum() else '_' for c in self.name).lower()
        if self.log_pid:
            return f'agent_{self.name}_{os.getpid()}'
        else:
            return f'agent_{self.name}'

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
            if self.agent_api:
                self.agent_api.shutdown()
            self.version_manager.shutdown(self)
            self.kb_manager.shutdown()

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

    def set_callback(self, callback: Callable[[int], None]):
        """Set a callback function to be called at the end of each iteration"""
        self.logger.assert_true(callable(callback), f"Callback must be callable, got {type(callback)}")
        self.callback = callback

    def run_loop(self,
        max_iterations: int = 10,
        stop_on_error: bool = True,
        raise_exception: bool = False,
        auto_resume: bool = True) -> Dict:
        """
        Main execution loop that continuously executes actions based on FSM state
        Now uses ActionSelector for cleaner action selection logic

        Args:
            max_iterations: Maximum number of iterations to prevent infinite loops
            stop_on_error: Whether to stop the loop on errors or continue
            raise_exception: Whether to reraise exception if error encountered
            auto_resume: Whether to automatically resume from last iteration in summary file

        Returns:
            Dict containing loop execution results
        """

        def process_summary():
            summary = self.get_summary()
            self._print_summary(summary)
            self._save_summary(summary)
            return summary

        self.logger.assert_true(
            self.fsm is not None,
            "FSM has not been properly initialized"
        )

        self.logger.info(f"Starting agent loop for {max_iterations} iterations")
        start_iteration = self.curr_iteration
        end_iteration = max_iterations
        summary = None

        for iteration in range(start_iteration, end_iteration + 1):
            try:
                # Increment iteration counter in history, show iteration
                self.curr_iteration = iteration
                self.history.set_iteration(iteration)
                self.logger.info(f"Loop iteration {iteration}/{end_iteration}")
                self.logger.info(f"Current state: {self.fsm.current_state}")
                # Validate active file state for consistency
                self.version_manager.validate_active_files(self)

                # Check agent context on first iteration to make sure state is valid
                if iteration == start_iteration:
                    self.fsm.check_context(self)

                # Use ActionSelector for all action selection logic
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
                self.logger.info(f"Executing {chosen_action} action")
                success = self.fsm.execute_action(chosen_action, self)
                self.logger.info(f"Action executed (success: {success}), new state: {self.fsm.current_state}")

                # Record the execution in history
                self.history.add_action_execution(
                    iteration=iteration,
                    action=chosen_action,
                    action_result=success,
                    action_reason=reason,
                    prev_state=prev_state,
                    curr_state=self.fsm.current_state
                )

                # Print summary and write to file
                summary = process_summary()

                # Handle callbacks at end of iteration
                if self.callback:
                    try:
                        self.callback(self, iteration)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                        self.logger.error(traceback.format_exc())

            except Exception as e:
                if raise_exception:
                    raise
                error_msg = f"Agent run_loop error at iteration {iteration}: {str(e)}"
                stack_trace = traceback.format_exc()
                self.logger.error(error_msg)
                self.logger.error(stack_trace)
                self.history.add_error(iteration, f"{error_msg}\n{stack_trace}",
                    self.fsm.current_state, str(e))
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

        # Set current iteration for next run_loop call
        self.curr_iteration = end_iteration + 1

        # Return agent summary
        return summary if summary else process_summary()

    def get_summary(self, recent_hist_len=None) -> Dict:
        """Get a comprehensive summary of the agent's current state as a dictionary."""
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

        if not recent_hist_len:
            recent_hist_len = self.history_context_len

        # Context info
        ctx_keys = _get_context_keys(self.context)

        # Enhanced history summary with more details
        actions_count = len(self.history.actions_executed)
        errors_count = len(self.history.errors)

        # Add recent actions summary
        recent_actions_summary = {}
        success_rate = 0.0
        if self.history.actions_executed:
            recent_actions = self.history.actions_executed[-recent_hist_len:]
            action_summary = {}
            for action_data in recent_actions:
                action_name = action_data.get('action', 'unknown')
                action_summary[action_name] = action_summary.get(action_name, 0) + 1
            recent_actions_summary = action_summary

            # Calculate success rate
            successful_actions = sum(1 for action in self.history.actions_executed
                                   if action['action_result'] == 'success')
            success_rate = (successful_actions / actions_count) * 100

        # Repository completion stats
        completion = self.repository_manager.get_repository_completion_stats(self)

        # OpenAI cost summary
        cost_summary = self.openai_handler.get_cost_summary()

        return {
            "agent_info": {
                "name": self.name,
                "current_state": self.fsm.current_state,
                "iteration": self.curr_iteration,
                "repository": self.repository
            },
            "context": {
                "keys": ctx_keys,
                "key_count": len(ctx_keys)
            },
            "execution_stats": {
                "actions_executed": actions_count,
                "errors": errors_count,
                "success_rate": f"{success_rate:.1f}%",
                "recent_actions": recent_actions_summary,
                "recent_actions_count": len(self.history.actions_executed[-recent_hist_len:]) if self.history.actions_executed else 0
            },
            "repository_progress": {
                "finished_files": completion['finished_files'],
                "total_files": completion['total_files'],
                "completion_percentage": completion['completion_percentage']
            },
            "openai_cost": {
                "model": cost_summary['model'],
                "call_count": cost_summary['call_count'],
                "accumulated_cost": cost_summary['accumulated_cost'],
                "cost_limit": cost_summary.get('cost_limit'),
                "remaining_budget": cost_summary.get('remaining_budget'),
                "usage_percentage": f"{(cost_summary['accumulated_cost'] / cost_summary['cost_limit']) * 100:.1f}%" if cost_summary.get('cost_limit') else None
            }
        }

    def _print_summary(self, summary):
        """Print agent summary to console in YAML-like format."""
        self.logger.info("="*80)
        self.logger.info("AGENT SUMMARY")
        self.logger.info("="*80)

        # Format and print each line
        yaml_lines = format_yaml_like(summary)
        for line in yaml_lines:
            self.logger.info(line)

        self.logger.info("="*80)

    def _save_summary(self, summary):
        """Save full summary history and most recent summary to separate JSON files."""
        log_dir = self.get_log_dir()

        # Check if we should save full history this iteration
        save_full_history = (self.curr_iteration == 1 or
            self.curr_iteration % self.save_hist_interval == 0)

        # Create recent summary data
        recent_summary_data = {
            'iteration': self.curr_iteration,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'epoch_time': time.time(),
            'summary': summary
        }

        # Only append to summary_history when we're saving full history
        if save_full_history:
            self.summary_history.append(recent_summary_data)

        try:
            # Always save most recent summary
            recent_summary_file = os.path.join(log_dir, f"{self.get_id()}.summary.json")
            with open(recent_summary_file, 'w', encoding='utf-8') as f:
                json.dump(recent_summary_data, f, indent=4, default=str)
            self.logger.debug(f"Latest summary saved to: {recent_summary_file}")

            # Conditionally save full history
            if save_full_history:
                full_history_file = os.path.join(log_dir, f"{self.get_id()}.summary_history.json")
                full_summary_data = {
                    "agent_id": self.get_id(),
                    "summary_count": len(self.summary_history),
                    "summary_history": self.summary_history
                }

                with open(full_history_file, 'w', encoding='utf-8') as f:
                    json.dump(full_summary_data, f, indent=4, default=str)
                self.logger.debug(f"Full summary history ({len(self.summary_history)} entries) saved to: {full_history_file}")
            else:
                self.logger.debug(f"Skipping full history save (iteration {self.curr_iteration}, interval {self.save_hist_interval})")

        except Exception as e:
            self.logger.error(f"Failed to save summary files: {e}")