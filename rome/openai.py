import openai
import json
import os
import re
from typing import Dict, Optional, Any, Union, List
from .logger import get_logger
from .config import set_attributes_from_config


class OpenAIHandler:
    """Handler class for OpenAI API interactions with configuration dictionary"""

    def __init__(self, config: Dict = None):
        """
        Initialize OpenAI handler with configuration dictionary

        Args:
            config: Configuration dictionary containing OpenAI parameters
        """
        # Use provided config or empty dict
        self.config = config or {}

        # Initialize logger
        self.logger = get_logger()

        # Automatically set attributes from config
        set_attributes_from_config(self, self.config)

        # Validate required attributes with a more compact assertion
        required_attrs = ['model', 'temperature', 'max_tokens', 'timeout', 'top_p', 'base_url', 'system_message']
        for attr in required_attrs:
            assert hasattr(self, attr), f"{attr} not provided in OpenAIHandler config"

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = "OpenAI API key not found in environment"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Setup OpenAI client
        client_kwargs = {
            "api_key": api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }
        self.client = openai.OpenAI(**client_kwargs)

        self.logger.info(f"OpenAI handler initialized with model: {self.model}")

    def _log_messages_with_multiline_support(self, messages):
        """Log messages with proper multiline string formatting"""
        self.logger.debug("Request messages:")

        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            # Log the role and message number
            self.logger.debug(f"[{i}] {role}:")

            # Split content by newlines and log each line separately with proper indentation
            if '\n' in content:
                for line in content.split('\n'):
                    self.logger.debug(f"    {line}")
            else:
                self.logger.debug(f"    {content}")

            # Add a separator between messages
            if i < len(messages) - 1:
                self.logger.debug("----------")

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
        if not system_message:
            system_message = self.system_message
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Build API parameters using object attributes with overrides
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            # "frequency_penalty": getattr(self, 'frequency_penalty', 0.0),
            # "presence_penalty": getattr(self, 'presence_penalty', 0.0),
        }

        # Apply any override config
        if override_config:
            kwargs.update(override_config)

        # Add seed if available
        if hasattr(self, 'seed') and self.seed is not None:
            kwargs["seed"] = self.seed

        # Add response format if provided
        if response_format is not None:
            kwargs["response_format"] = response_format

        # Add any extra parameters
        if extra_body:
            kwargs.update(extra_body)

        # Log request parameters and messages at debug level
        # self.logger.info(f"API call: {kwargs['model']} (temp={kwargs['temperature']})")
        self.logger.debug(f"OpenAI API request parameters: {json.dumps({k: v for k, v in kwargs.items() if k != 'messages'}, indent=4)}")
        self._log_messages_with_multiline_support(messages)

        response = self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content.strip()

        # Log token usage if available
        if hasattr(response, 'usage') and response.usage:
            self.logger.info(f"Tokens: {response.usage.prompt_tokens}→{response.usage.completion_tokens} (total: {response.usage.total_tokens})")

        # Log the full response at debug level
        # self.logger.debug(f"OpenAI API response: {response}")
        self.logger.debug(f"Response content: {content}")

        return content

    def parse_python_response(self, response: str) -> Optional[str]:
        """
        Extract the first Python code block from a response.

        Args:
            response (str): The response text containing code blocks

        Returns:
            Optional[str]: The first Python code block found, or None if no code found
        """
        if not response:
            return None

        # Pattern to match Python code blocks with various formats
        # Handles ```python, ``` python, and plain ``` code blocks
        python_patterns = [
            r'```\s*python\s*\n(.*?)\n```',  # ```python\n...\n```
            r'```\n(.*?)\n```'               # ```\n...\n```
        ]

        for pattern in python_patterns:
            # Find all matches with the current pattern
            matches = re.findall(pattern, response, re.DOTALL)

            # If we found any matches with this pattern
            if matches:
                # Clean up the code blocks by stripping leading/trailing whitespace
                python_blocks = [match.strip() for match in matches]

                # Return first non-empty block (minimal validation)
                for block in python_blocks:
                    if block.strip():
                        return block

        # If no code blocks found, try to find inline code with single backticks
        # This is a fallback for simple code snippets
        inline_pattern = r'`([^`]+)`'
        inline_matches = re.findall(inline_pattern, response)

        # Only consider inline code with Python keywords
        python_keywords = ['def ', 'import ', 'from ', 'class ', 'if ', 'for ', 'while ', 'return ']
        for match in inline_matches:
            # Simple validation that this might be Python code
            if any(keyword in match for keyword in python_keywords) or '=' in match or 'print(' in match:
                return match.strip()

        # No valid Python code found
        return None

    def parse_json_response(self, response: str) -> Union[Dict[str, Any], List[Any]]:
        """
        Parse JSON response with optimized approach for both direct JSON and extraction.

        Handles both responses from JSON mode (response_format={"type": "json_object"})
        and extracts JSON from text responses when needed.

        Args:
            response (str): The response text, either direct JSON or containing JSON fragments

        Returns:
            Union[Dict[str, Any], List[Any]]: The parsed JSON object/array or an empty dict if parsing failed
        """
        if not response or not response.strip():
            self.logger.debug("Empty response received for JSON parsing")
            return {}

        # STEP 1: Try direct JSON parsing first (for response_format="json_object" responses)
        try:
            parsed_json = json.loads(response)
            # Return the parsed JSON as is, whether it's a dict or a list
            return parsed_json
        except json.JSONDecodeError:
            # Direct parsing failed, move to extraction methods
            self.logger.debug("Direct JSON parsing failed, trying extraction methods")

        # STEP 2: Extract JSON from code blocks
        json_block_patterns = [
            r'```\s*json\s*\n([\s\S]*?)\n```',  # ```json\n...\n```
            r'```\n([\s\S]*?)\n```'             # ```\n...\n```
        ]

        for pattern in json_block_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    cleaned_match = match.strip()
                    parsed_json = json.loads(cleaned_match)
                    # Return the parsed JSON as is, whether it's a dict or a list
                    return parsed_json
                except json.JSONDecodeError:
                    continue

        # STEP 3: Extract JSON from free text (balanced bracket approach)
        # Find objects with properly balanced braces
        start_indexes = []
        candidate_objects = []
        depth = 0

        for i, char in enumerate(response):
            if char == '{':
                if depth == 0:
                    start_indexes.append(i)
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start_indexes:
                    candidate_objects.append(response[start_indexes.pop():i+1])
            # Also check for JSON arrays
            elif char == '[':
                if depth == 0:
                    start_indexes.append(i)
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0 and start_indexes:
                    candidate_objects.append(response[start_indexes.pop():i+1])

        # Try to parse each candidate JSON object or array
        for candidate in candidate_objects:
            try:
                parsed_json = json.loads(candidate)
                # Return the parsed JSON regardless of type
                return parsed_json
            except json.JSONDecodeError:
                continue

        # No valid JSON found
        self.logger.info("Failed to parse any valid JSON from the response")
        return {}

    def update_config(self, config_updates: Dict):
        """
        Update configuration parameters

        Args:
            config_updates: Dictionary containing parameters to update
        """
        self.logger.info(f"Updated OpenAI config: {list(config_updates.keys())}")
        self.config.update(config_updates)

        # Update object attributes as well
        for key, value in config_updates.items():
            setattr(self, key, value)

    def get_config(self) -> Dict:
        """Get current configuration"""
        return self.config.copy()

    def reset_config(self, new_config: Dict):
        """Reset configuration to new config"""
        self.logger.info("Reset OpenAI handler configuration")
        self.config = new_config.copy()

        # Update all attributes from the new config
        set_attributes_from_config(self, self.config)
