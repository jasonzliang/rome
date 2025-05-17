# openai_handler.py
import openai
import json
import os
import re
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
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = "OpenAI API key not found in environment"
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

    # Fix the methods to include self parameter
    def parse_python_response(self, response: str) -> Optional[str]:
        """
        Extract the first Python code block from a response.

        Args:
            response (str): The response text containing code blocks

        Returns:
            Optional[str]: The first Python code block found, or None if no code found
        """
        # Pattern to match Python code blocks
        # Matches both ```python and ``` followed by python keywords
        python_pattern = r'```(?:python)?\n(.*?)```'

        # Find all matches
        matches = re.findall(python_pattern, response, re.DOTALL)

        # Clean up the code blocks by stripping leading/trailing whitespace
        python_blocks = [match.strip() for match in matches]

        # Filter to find the first block that looks like Python code
        for block in python_blocks:
            if block:  # If there's any content, return it
                # Simple validation - check for common Python patterns
                if (any(keyword in block for keyword in ['def ', 'import ', 'from ', 'class ', 'if ', 'for ', 'while '])
                    or '=' in block or 'print(' in block):
                    return block
                # If no specific markers found but has content, still return it
                elif block.strip():
                    return block

        return None

    def parse_json_response(self, response: str) -> Optional[Dict]:
        """
        Extract and parse the first JSON object from a response.

        Args:
            response (str): The response text containing JSON blocks

        Returns:
            Optional[Dict]: The first valid JSON object found, or None if no valid JSON found
        """
        # Pattern to match JSON code blocks
        json_pattern = r'```(?:json)?\n(\{.*?\}|\[.*?\])```'

        # Find all matches in code blocks first
        matches = re.findall(json_pattern, response, re.DOTALL)

        # Try to parse JSON from code blocks first
        for match in matches:
            try:
                cleaned_match = match.strip()
                parsed_json = json.loads(cleaned_match)
                # Return only if it's a dictionary, skip arrays unless needed
                if isinstance(parsed_json, dict):
                    return parsed_json
            except json.JSONDecodeError:
                continue

        # If no valid JSON in code blocks, try inline JSON
        # Look for JSON-like structures outside of code blocks
        inline_json_pattern = r'\{[^{}]*\}'
        inline_matches = re.findall(inline_json_pattern, response)

        for match in inline_matches:
            try:
                parsed_json = json.loads(match)
                if isinstance(parsed_json, dict):
                    return parsed_json
            except json.JSONDecodeError:
                continue

        # If we found array-type JSON in code blocks but no dict, return the first array as dict
        # This handles edge cases where the JSON might be an array
        for match in re.findall(json_pattern, response, re.DOTALL):
            try:
                parsed_json = json.loads(match.strip())
                # If it's an array of objects, return the first object
                if isinstance(parsed_json, list) and parsed_json and isinstance(parsed_json[0], dict):
                    return parsed_json[0]
            except (json.JSONDecodeError, IndexError):
                continue

        return None

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
