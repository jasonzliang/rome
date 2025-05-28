import openai
import json
import os
import re
import hashlib
from typing import Dict, Optional, Any, Union, List
from functools import lru_cache
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
        set_attributes_from_config(self, self.config, ['model', 'temperature', 'max_tokens', 'timeout', 'top_p', 'base_url', 'system_message', 'key_name', 'manage_context', 'max_input_tokens',
            'token_count_thres', 'chars_per_token', 'seed'])

        # Get API key from environment
        api_key = os.getenv(self.key_name or 'OPENAI_API_KEY')
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

    @lru_cache(maxsize=1)
    def _get_model_context_length(self) -> int:
        """Get context length for the current model."""
        try:
            # Try to get from OpenAI API
            models = self.client.models.list()
            model_info = next((m for m in models.data if m.id == self.model), None)
            if model_info and hasattr(model_info, 'context_length'):
                return model_info.context_length
        except Exception:
            pass

        return 128000 # Default for most OpenAI models

    def get_max_input_tokens(self) -> int:
        """Get max input tokens, with manual override or auto-calculation."""
        # Check if max_input_tokens is manually set in config
        if self.max_input_tokens:
            _max_input_tokens = self.max_input_tokens
        else:
            # Auto-calculate: context length minus max_tokens
            _max_input_tokens = self._get_model_context_length() - self.max_tokens
        return max(_max_input_tokens, 0)

    @lru_cache(maxsize=1)
    def _get_encoding(self):
        """Get tiktoken encoding for the model."""
        try:
            import tiktoken
            return tiktoken.encoding_for_model(self.model)
        except ImportError:
            self.logger.warning("tiktoken not available, using character estimation")
            return None

    def _estimate_tokens_fast(self, messages: List[Dict]) -> int:
        """Fast token estimation using character count."""
        total_chars = sum(len(str(msg.get('content', ''))) + len(str(msg.get('role', ''))) + 10
                         for msg in messages)
        return total_chars // self.chars_per_token

    def _count_message_tokens_precise(self, message: Dict) -> int:
        """Count tokens for a single message."""
        encoding = self._get_encoding()
        if not encoding:
            # Fallback to character estimation
            content_len = len(str(message.get('content', '')))
            role_len = len(str(message.get('role', '')))
            return (content_len + role_len + 10) // self.chars_per_token

        tokens = 4  # Message overhead
        for key, value in message.items():
            tokens += len(encoding.encode(str(value)))

        return tokens

    def _should_count_tokens_precisely(self, messages: List[Dict]) -> bool:
        """Decide if precise token counting is needed."""
        fast_estimate = self._estimate_tokens_fast(messages)
        threshold = self.get_max_input_tokens() * self.token_count_thres
        return fast_estimate > threshold

    def _count_tokens_precise(self, messages: List[Dict]) -> int:
        """Precise token counting for all messages."""
        total = sum(self._count_message_tokens_precise(msg) for msg in messages)
        return total + 2  # Assistant reply primer

    def _needs_truncation(self, messages: List[Dict]) -> bool:
        """Check if messages need truncation with minimal computation."""
        if not self.manage_context:
            return False

        # Quick check first
        if not self._should_count_tokens_precisely(messages):
            return False

        # Precise count only if potentially over limit
        return self._count_tokens_precise(messages) > self.max_input_tokens

    def _truncate_messages(self, messages: List[Dict]) -> List[Dict]:
        """Truncate messages to fit context, preserving system message."""
        # Preserve system message
        system_msg = None
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            messages = messages[1:]

        result = []
        current_tokens = 0

        if system_msg:
            current_tokens = self._count_message_tokens_precise(system_msg)
            result.append(system_msg)

        # Add messages from most recent
        for msg in reversed(messages):
            msg_tokens = self._count_message_tokens_precise(msg)
            if current_tokens + msg_tokens <= self.max_input_tokens:
                if system_msg:
                    result.insert(-1, msg)
                else:
                    result.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break

        if len(result) < len(messages) + (1 if system_msg else 0):
            truncated_count = len(messages) - len(result) + (1 if system_msg else 0)
            self.logger.info(f"Truncated {truncated_count} messages to fit context")

        return result

    def _prepare_messages(self, messages: List[Dict]) -> List[Dict]:
        """Prepare messages with smart context management."""
        if not self._needs_truncation(messages):
            return messages
        return self._truncate_messages(messages)

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

    def chat_completion(self, prompt: str,
        system_message: str = None,
        override_config: Dict = None,
        response_format: Dict = None,
        extra_body: Dict = None,
        conversation_history: List[Dict] = None) -> str:
        """
        Chat completion method with configurable parameters and smart context management

        Args:
            prompt: The user prompt
            system_message: Optional system message
            override_config: Dictionary to override default config parameters
            response_format: Optional response format (e.g., {"type": "json_object"})
            extra_body: Additional parameters to pass to the API
            conversation_history: Optional conversation history for multi-turn chats

        Returns:
            The response content as string
        """
        messages = []

        # Add system message
        if not system_message:
            system_message = self.system_message
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # Apply smart context management
        messages = self._prepare_messages(messages)

        # Build API parameters using object attributes with overrides
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        # Add seed if available
        if self.seed:
            kwargs["seed"] = self.seed

        # Apply any override config
        if override_config:
            kwargs.update(override_config)

        # Add response format if provided
        if response_format:
            kwargs["response_format"] = response_format

        # Add any extra parameters
        if extra_body:
            kwargs.update(extra_body)

        # Log request parameters and messages at debug level
        self.logger.debug(f"OpenAI API request parameters: {json.dumps({k: v for k, v in kwargs.items() if k != 'messages'}, indent=4)}")
        self._log_messages_with_multiline_support(messages)

        try:
            response = self.client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:
            if "maximum context length" in str(e).lower():
                self.logger.error("Context length exceeded, recheck OpenAI API config")
            raise

        content = response.choices[0].message.content.strip()

        # Log token usage if available
        if hasattr(response, 'usage') and response.usage:
            self.logger.info(f"Tokens: {response.usage.prompt_tokens}→{response.usage.completion_tokens} (total: {response.usage.total_tokens})")

        # Log the full response at debug level
        self.logger.debug(f"Response content: {content}")

        return content
