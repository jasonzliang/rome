import openai
import json
import logging
import os
from typing import Dict, List, Any, Optional

class OpenAIHandler:
    """Handler class for OpenAI API interactions with configuration dictionary"""

    def __init__(self, config: Dict = None):
        """
        Initialize OpenAI handler with configuration dictionary

        Args:
            config: Configuration dictionary containing OpenAI parameters
        """
        # Use provided config or empty dict (defaults will come from config.py)
        self.config = config or {}

        # Get API key from config or environment
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")

        # Setup OpenAI client
        client_kwargs = {
            "api_key": api_key,
            "timeout": self.config.get("timeout", 60)
        }

        # Add optional parameters if provided
        if self.config.get("base_url"):
            client_kwargs["base_url"] = self.config["base_url"]

        self.client = openai.OpenAI(**client_kwargs)
        self.logger = logging.getLogger(__name__)

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

            # Merge configs: current -> override
            effective_config = self.config.copy()
            if override_config:
                effective_config.update(override_config)

            # Build API parameters with safe defaults
            kwargs = {
                "model": effective_config.get("model", "gpt-4o"),
                "messages": messages,
                "temperature": effective_config.get("temperature", 0.1),
                "max_tokens": effective_config.get("max_tokens", 4000),
                "top_p": effective_config.get("top_p", 1.0),
                "frequency_penalty": effective_config.get("frequency_penalty", 0.0),
                "presence_penalty": effective_config.get("presence_penalty", 0.0),
            }

            # Add seed if provided
            if effective_config.get("seed") is not None:
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

    def get_config(self) -> Dict:
        """Get current configuration"""
        return self.config.copy()

    def reset_config(self, new_config: Dict):
        """Reset configuration to new config"""
        self.config = new_config.copy()
