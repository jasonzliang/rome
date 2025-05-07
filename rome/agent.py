# agent.py
import os
import yaml
import openai
import logging
from typing import Dict, List

# Import default config utilities
from default_config import load_config, merge_with_default_config, DEFAULT_CONFIG


class Agent:
    """Agent class using OpenAI API, YAML config, and FSM architecture"""

    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """Initialize the Agent with either a config path or a config dictionary"""
        # Load configuration
        if config_path:
            # Load from file, creating default if it doesn't exist
            loaded_config = load_config(config_path, create_if_missing=True)
            self.config = loaded_config
        elif config_dict:
            # Merge provided config with defaults
            self.config = merge_with_default_config(config_dict)
        else:
            # Use default config
            self.config = DEFAULT_CONFIG

        # Initialize context (holds runtime data)
        self.context = {}

        # Set up logging
        self._setup_logging()

        # Initialize OpenAI client
        self._setup_llm_client()

        # Initialize the Finite State Machine
        self._setup_fsm()

    def _setup_logging(self):
        """Configure logging based on config"""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file', None)
        console_output = log_config.get('console', True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            filename=log_file,
            filemode='a' if log_file else None
        )

        # Add console handler if enabled
        self.logger = logging.getLogger('Agent')
        if console_output and log_file:
            console = logging.StreamHandler()
            console.setLevel(getattr(logging, log_level))
            console.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(console)

        self.logger.info("Logging initialized")

    def _setup_llm_client(self):
        """Initialize the OpenAI client"""
        openai_config = self.config.get('openai', {})
        api_key = openai_config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        api_base = openai_config.get('api_base')
        timeout = openai_config.get('timeout', 120)

        if not api_key:
            raise ValueError("OpenAI API key is required. Set it in config or OPENAI_API_KEY env var.")

        # Create OpenAI client with configuration
        client_kwargs = {
            'api_key': api_key,
            'timeout': timeout
        }

        if api_base:
            client_kwargs['base_url'] = api_base

        self.llm_client = openai.OpenAI(**client_kwargs)
        self.logger.info("LLM client initialized")

    def _setup_fsm(self):
        """Initialize the Finite State Machine"""
        from agent_fsm import setup_default_fsm
        self.fsm = setup_default_fsm(config=self.config)
        self.logger.info(f"FSM initialized with state: {self.fsm.current_state}")

    def get_current_state(self):
        """Get the current state of the agent"""
        return self.fsm.current_state

    def get_available_actions(self) -> List[str]:
        """Get available actions from the current state"""
        return self.fsm.get_available_actions()

    def execute_action(self, action_name: str, **kwargs):
        """Execute an action and update the agent's state"""
        self.logger.info(f"Executing action: {action_name}")
        return self.fsm.execute_action(action_name, self, **kwargs)

    def search_repository(self, query: str = '', file_pattern: str = '*.py') -> List[Dict]:
        """Search the repository for code files"""
        self.logger.info(f"Searching repository for: {query} with pattern: {file_pattern}")
        self.execute_action("search", query=query, file_pattern=file_pattern)
        return self.context.get('search_results', [])

    def analyze_code(self, prompt: str = None) -> List[Dict]:
        """Analyze code files using LLM"""
        from agent_fsm import AgentState

        # Get prompt from config if not provided
        custom_prompt = prompt or self.config.get('prompts', {}).get('analyze')
        if not custom_prompt:
            # Fall back to default prompt in analyze action config
            custom_prompt = self.config.get('actions', {}).get('analyze', {}).get(
                'default_prompt', "Analyze the following code:"
            )

        self.logger.info("Analyzing code")
        self.execute_action("analyze", prompt=custom_prompt)
        return self.context.get('analysis_results', [])

    def update_file(self, file_path: str, instructions: str, preview_only: bool = False) -> Dict:
        """Update a code file using LLM based on instructions"""
        self.logger.info(f"Updating file: {file_path}")
        self.execute_action("update", file_path=file_path, instructions=instructions, preview_only=preview_only)
        return self.context.get('update_result', {})

    def load_file(self, file_path: str) -> Dict:
        """Load a file's content"""
        self.logger.info(f"Loading file: {file_path}")
        self.execute_action("load_file", file_path=file_path)
        return self.context.get('loaded_file', {})

    def save_file(self, file_path: str, content: str) -> Dict:
        """Save content to a file"""
        self.logger.info(f"Saving file: {file_path}")
        self.execute_action("save_file", file_path=file_path, content=content)
        return self.context.get('save_result', {})

    def save_config(self, config_path: str) -> None:
        """Save the current configuration to a YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        self.logger.info(f"Configuration saved to {config_path}")
