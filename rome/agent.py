# agent.py
import os
import yaml
import logging
from typing import Dict, List, Any, Optional

# Import the OpenAIHandler we created
from .openai_handler import OpenAIHandler
# Import default config utilities
from .config import DEFAULT_CONFIG
from .config import load_config, merge_with_default_config, get_action_llm_config


class Agent:
    """Agent class using OpenAI API, YAML config, and FSM architecture"""

    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """Initialize the Agent with either a config path or a config dictionary"""
        # Load configuration
        if config_path:
            self.config = load_config(config_path, create_if_missing=True)
        elif config_dict:
            self.config = merge_with_default_config(config_dict)
        else:
            self.config = DEFAULT_CONFIG.copy()

        # Initialize context
        self.context = {}

        # Set up components
        self._setup_logging()
        self._setup_fsm()

        # Initialize OpenAI handler with main LLM config
        llm_config = self.config.get('llm', {})
        self.openai_handler = OpenAIHandler(config=llm_config)
        self.llm_client = self.openai_handler  # Backward compatibility

        # Cache for action-specific configurations
        self._action_configs_cache = {}

        self.logger.info(f"Agent initialized with model: {llm_config.get('model', 'gpt-4o')}")

    def _setup_logging(self):
        """Configure logging based on config"""
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            filename=log_config.get('file', None),
            filemode='a' if log_config.get('file') else None
        )

        self.logger = logging.getLogger('Agent')

        # Add console handler if both file and console output are enabled
        if log_config.get('console', True) and log_config.get('file'):
            console = logging.StreamHandler()
            console.setLevel(getattr(logging, log_config.get('level', 'INFO')))
            console.setFormatter(logging.Formatter(log_config.get('format')))
            self.logger.addHandler(console)

    def _setup_fsm(self):
        """Initialize the Finite State Machine"""
        from agent_fsm import setup_default_fsm
        self.fsm = setup_default_fsm(config=self.config)
        self.logger.info(f"FSM initialized with state: {self.fsm.current_state}")

    # Action and state management
    def get_current_state(self):
        """Get the current state of the agent"""
        return self.fsm.current_state if self.fsm else None

    def get_available_actions(self) -> List[str]:
        """Get available actions from the current state"""
        return self.fsm.get_available_actions() if self.fsm else []

    def execute_action(self, action_name: str, **kwargs):
        """Execute an action and update the agent's state"""
        self.logger.info(f"Executing action: {action_name}")
        if self.fsm:
            return self.fsm.execute_action(action_name, self, **kwargs)
        else:
            self.logger.warning("FSM not initialized, cannot execute action")

    # LLM configuration management
    def get_action_llm_config(self, action_name: str) -> Dict:
        """
        Get LLM configuration for a specific action, with action-specific overrides.

        Args:
            action_name: Name of the action

        Returns:
            Dict: OpenAI-compatible configuration with action-specific overrides
        """
        # Check cache first
        if action_name in self._action_configs_cache:
            return self._action_configs_cache[action_name].copy()

        # Get merged LLM config for this action
        action_llm_config = get_action_llm_config(self.config, action_name)

        # Remove None values for cleaner config
        clean_config = {k: v for k, v in action_llm_config.items() if v is not None}

        # Cache the result
        self._action_configs_cache[action_name] = clean_config

        return clean_config.copy()

    # Chat completion methods
    def chat_completion(self, prompt: str, system_message: str = None,
                       action_type: str = None, override_config: Dict = None,
                       response_format: Dict = None, extra_body: Dict = None) -> str:
        """
        Direct access to chat completion with configuration options

        Args:
            prompt: The user prompt
            system_message: Optional system message
            action_type: Use action-specific config
            override_config: Dictionary to override any config parameters
            response_format: Optional response format (e.g., {"type": "json_object"})
            extra_body: Additional parameters to pass to the API

        Returns:
            The response content as string
        """
        # Get action-specific config if requested
        config = {}
        if action_type:
            config = self.get_action_llm_config(action_type)

            # Use action-specific system message if not provided
            if system_message is None:
                action_llm_config = get_action_llm_config(self.config, action_type)
                system_message = action_llm_config.get('system_message')

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

    def execute_action_with_llm(self, action_name: str, prompt: str,
                               system_message: str = None, **kwargs):
        """
        Execute a chat completion using action-specific LLM configuration.

        Args:
            action_name: The action whose LLM config to use
            prompt: The user prompt
            system_message: Optional system message
            **kwargs: Additional arguments passed to chat_completion

        Returns:
            The response content as string
        """
        return self.chat_completion(
            prompt=prompt,
            system_message=system_message,
            action_type=action_name,
            **kwargs
        )

    # Utility methods
    def parse_json_response(self, response: str) -> Dict:
        """Parse JSON response using the handler"""
        return self.openai_handler.parse_json_response(response)

    def update_openai_config(self, config_updates: Dict):
        """Update OpenAI handler configuration"""
        self.openai_handler.update_config(config_updates)
        self.config['llm'].update(config_updates)
        self._action_configs_cache.clear()

    def update_action_llm_config(self, action_name: str, config_updates: Dict):
        """
        Update LLM configuration for a specific action.

        Args:
            action_name: Name of the action
            config_updates: Dictionary of config parameters to update
        """
        # Ensure the action config structure exists
        if 'actions' not in self.config:
            self.config['actions'] = {}
        if action_name not in self.config['actions']:
            self.config['actions'][action_name] = {}
        if 'llm' not in self.config['actions'][action_name]:
            self.config['actions'][action_name]['llm'] = {}

        # Update the action's LLM config
        self.config['actions'][action_name]['llm'].update(config_updates)

        # Clear the cache for this action
        if action_name in self._action_configs_cache:
            del self._action_configs_cache[action_name]

    def get_openai_config(self) -> Dict:
        """Get current OpenAI handler configuration"""
        return self.openai_handler.get_config()

    def print_usage_summary(self):
        """Print usage summary - placeholder for compatibility"""
        self.logger.info("Usage tracking not implemented in OpenAI Handler")

    # Repository management
    @property
    def repo_dir(self) -> str:
        """Get repository directory from config"""
        return self.config.get('repository', {}).get('path', os.getcwd())
