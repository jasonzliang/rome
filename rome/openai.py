# openai_handler.py
import openai
import json
import os
from typing import Dict, Optional
from .logger import get_logger


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

        # Initialize logger
        self.logger = get_logger()

        # Get API key from config or environment
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = "OpenAI API key not found in config or environment variables"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Setup OpenAI client
        client_kwargs = {
            "api_key": api_key,
            "timeout": self.config.get("timeout", 60)
        }

        # Add optional parameters if provided
        if self.config.get("base_url"):
            client_kwargs["base_url"] = self.config["base_url"]

        self.client = openai.OpenAI(**client_kwargs)

        self.logger.info(f"OpenAI handler initialized with model: {self.config.get('model', 'gpt-4o')}")

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

            self.logger.info(f"API call: {kwargs['model']} (temp={kwargs['temperature']})")

            response = self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content.strip()

            # Log token usage if available
            if hasattr(response, 'usage') and response.usage:
                self.logger.info(f"Tokens: {response.usage.prompt_tokens}→{response.usage.completion_tokens} (total: {response.usage.total_tokens})")

            return content

        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
        except openai.RateLimitError as e:
            self.logger.error(f"Rate limit exceeded: {str(e)}")
            raise
        except openai.Timeout as e:
            self.logger.error(f"Request timeout: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in chat completion: {str(e)}")
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

            parsed = json.loads(response)
            return parsed

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}

    def update_config(self, config_updates: Dict):
        """
        Update configuration parameters

        Args:
            config_updates: Dictionary containing parameters to update
        """
        self.logger.info(f"Updated OpenAI config: {list(config_updates.keys())}")
        self.config.update(config_updates)

    def get_config(self) -> Dict:
        """Get current configuration"""
        return self.config.copy()

    def reset_config(self, new_config: Dict):
        """Reset configuration to new config"""
        self.logger.info("Reset OpenAI handler configuration")
        self.config = new_config.copy()
