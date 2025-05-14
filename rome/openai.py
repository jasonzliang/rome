import openai
import json
import logging
from typing import Dict, List, Any, Optional

class OpenAIHandler:
    """Handler class for OpenAI API interactions with configuration dictionary"""

    # Default configuration
    DEFAULT_CONFIG = {
        "api_key": None,
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 4000,
        "top_p": 1.0,
        "seed": None,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "timeout": 60,
        "base_url": None,
        "api_version": None,
        "api_type": None
    }

    def __init__(self, config: Dict = None):
        """
        Initialize OpenAI handler with configuration dictionary

        Args:
            config: Configuration dictionary containing OpenAI parameters
        """
        # Merge provided config with defaults
        self.config = self._merge_config(config or {})

        # Setup OpenAI client
        client_kwargs = {
            "api_key": self.config["api_key"],
            "timeout": self.config["timeout"]
        }

        # Add optional parameters if provided
        if self.config["base_url"]:
            client_kwargs["base_url"] = self.config["base_url"]

        self.client = openai.OpenAI(**client_kwargs)
        self.logger = logging.getLogger(__name__)

        # Extract commonly used parameters
        self.model = self.config["model"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        self.top_p = self.config["top_p"]
        self.seed = self.config["seed"]
        self.frequency_penalty = self.config["frequency_penalty"]
        self.presence_penalty = self.config["presence_penalty"]

    def _merge_config(self, config: Dict) -> Dict:
        """Merge provided config with default config"""
        merged = self.DEFAULT_CONFIG.copy()
        merged.update(config)
        return merged

    def chat_completion(self, prompt: str, system_message: str = None,
                       override_config: Dict = None, response_format: Dict = None,
                       extra_body: Dict = None) -> str:
        """
        Chat completion method with configurable parameters

        Args:
            prompt: The user prompt
            system_message: Optional system message
            override_config: Dictionary to override default config parameters
            response_format: Optional response format (e.g., {"type": "json_object"})
            extra_body: Additional parameters to pass to the API

        Returns:
            The response content as string
        """
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            # Merge configs: default -> override
            effective_config = self.config.copy()
            if override_config:
                effective_config.update(override_config)

            # Build API parameters
            kwargs = {
                "model": effective_config["model"],
                "messages": messages,
                "temperature": effective_config["temperature"],
                "max_tokens": effective_config["max_tokens"],
                "top_p": effective_config["top_p"],
                "frequency_penalty": effective_config["frequency_penalty"],
                "presence_penalty": effective_config["presence_penalty"],
            }

            # Add seed if provided
            if effective_config["seed"] is not None:
                kwargs["seed"] = effective_config["seed"]

            # Add response format if provided
            if response_format is not None:
                kwargs["response_format"] = response_format

            # Add any extra parameters
            if extra_body:
                kwargs.update(extra_body)

            response = self.client.chat.completions.create(**kwargs)

            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error in chat completion: {str(e)}")
            raise

    def parse_json_response(self, response: str) -> Dict:
        """
        Parse JSON response, handling code blocks

        Args:
            response: Raw response string

        Returns:
            Parsed JSON dictionary
        """
        try:
            # Clean up response if it contains code blocks
            if response.startswith('```json'):
                response = response[7:-3]
            elif response.startswith('```'):
                response = response[3:-3]

            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON response: {str(e)}")
            return {"error": f"Failed to parse JSON: {str(e)}"}

    def update_config(self, config_updates: Dict):
        """
        Update configuration parameters

        Args:
            config_updates: Dictionary containing parameters to update
        """
        self.config.update(config_updates)

        # Update commonly accessed parameters
        self.model = self.config["model"]
        self.temperature = self.config["temperature"]
        self.max_tokens = self.config["max_tokens"]
        self.top_p = self.config["top_p"]
        self.seed = self.config["seed"]
        self.frequency_penalty = self.config["frequency_penalty"]
        self.presence_penalty = self.config["presence_penalty"]

    def get_config(self) -> Dict:
        """Get current configuration"""
        return self.config.copy()

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.DEFAULT_CONFIG.copy()
        self.update_config({})  # Trigger parameter updates
