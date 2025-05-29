from functools import lru_cache
import json
import logging
import openai
import os
from typing import Dict, Optional, Any, Union, List

from .logger import get_logger
from .config import set_attributes_from_config


class CostLimitExceededException(Exception):
    """Exception raised when the cost limit is exceeded."""
    def __init__(self, estimated_cost: float, cost_limit: float, accumulated_cost: float = None):
        self.estimated_cost = estimated_cost
        self.cost_limit = cost_limit
        self.accumulated_cost = accumulated_cost
        if accumulated_cost is not None:
            super().__init__(f"Estimated cost ${estimated_cost:.4f} would bring total to ${accumulated_cost + estimated_cost:.4f}, exceeding limit ${cost_limit:.4f}")
        else:
            super().__init__(f"Estimated cost ${estimated_cost:.4f} exceeds limit ${cost_limit:.4f}")


class OpenAIHandler:
    """Handler class for OpenAI API interactions with configuration dictionary and cost limiting"""

    # OpenAI model pricing (per 1M tokens) - Updated with latest models
    MODEL_PRICING = {
        "gpt-4.1": {"input": 30.0, "output": 120.0},
        "gpt-4.1-nano": {"input": 1.0, "output": 4.0},
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-4o-realtime-preview": {"input": 5.0, "output": 20.0},
        "gpt-4o-audio-preview": {"input": 5.0, "output": 20.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4-turbo-2024-04-09": {"input": 10.0, "output": 30.0},
        "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-0314": {"input": 30.0, "output": 60.0},
        "gpt-4-0613": {"input": 30.0, "output": 60.0},
        "gpt-4-32k": {"input": 60.0, "output": 120.0},
        "gpt-4-32k-0314": {"input": 60.0, "output": 120.0},
        "gpt-4-32k-0613": {"input": 60.0, "output": 120.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "gpt-3.5-turbo-0125": {"input": 0.5, "output": 1.5},
        "gpt-3.5-turbo-1106": {"input": 1.0, "output": 2.0},
        "gpt-3.5-turbo-0613": {"input": 1.5, "output": 2.0},
        "gpt-3.5-turbo-16k": {"input": 3.0, "output": 4.0},
        "gpt-3.5-turbo-16k-0613": {"input": 3.0, "output": 4.0},
        "gpt-3.5-turbo-instruct": {"input": 1.5, "output": 2.0},
        "o1": {"input": 15.0, "output": 60.0},
        "o1-preview": {"input": 15.0, "output": 60.0},
        "o1-mini": {"input": 3.0, "output": 12.0},
        "o3-mini": {"input": 1.10, "output": 4.40},
        "o3": {"input": 10.0, "output": 40.0},
        "o4-mini": {"input": 1.10, "output": 4.40},
    }

    MODEL_CONTEXT_SIZE = {
        "gpt-4.1": 1000000,  # 1M tokens
        "gpt-4.1-nano": 1000000,  # 1M tokens
        "gpt-4o": 128000,  # 128K tokens
        "gpt-4o-mini": 128000,  # 128K tokens
        "gpt-4o-realtime-preview": 128000,  # 128K tokens
        "gpt-4o-audio-preview": 128000,  # 128K tokens
        "gpt-4-turbo": 128000,  # 128K tokens
        "gpt-4-turbo-2024-04-09": 128000,  # 128K tokens
        "gpt-4-turbo-preview": 128000,  # 128K tokens
        "gpt-4": 8192,  # 8K tokens
        "gpt-4-0314": 8192,  # 8K tokens
        "gpt-4-0613": 8192,  # 8K tokens
        "gpt-4-32k": 32768,  # 32K tokens
        "gpt-4-32k-0314": 32768,  # 32K tokens
        "gpt-4-32k-0613": 32768,  # 32K tokens
        "gpt-3.5-turbo": 16385,  # 16K tokens
        "gpt-3.5-turbo-0125": 16385,  # 16K tokens
        "gpt-3.5-turbo-1106": 16385,  # 16K tokens
        "gpt-3.5-turbo-0613": 4096,  # 4K tokens
        "gpt-3.5-turbo-16k": 16384,  # 16K tokens
        "gpt-3.5-turbo-16k-0613": 16384,  # 16K tokens
        "gpt-3.5-turbo-instruct": 4096,  # 4K tokens
        "o1": 200000,  # 200K tokens
        "o1-preview": 128000,  # 128K tokens
        "o1-mini": 128000,  # 128K tokens
        "o3-mini": 200000,  # 200K tokens
        "o3": 200000,  # 200K tokens
        "o4-mini": 200000,  # 200K tokens
    }

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = get_logger()

        # Initialize cost tracking
        self.accumulated_cost = 0.0
        self.call_count = 0
        self.cost_history = []  # List of (timestamp, cost, tokens_used) tuples

        # Set attributes from config
        set_attributes_from_config(self, self.config, [
            'model', 'temperature', 'max_tokens', 'timeout', 'top_p', 'base_url',
            'system_message', 'key_name', 'manage_context', 'max_input_tokens',
            'token_count_thres', 'chars_per_token', 'seed', 'cost_limit'
        ])

        # Initialize client
        api_key = os.getenv(self.key_name or 'OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment")

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

        self.logger.info(f"OpenAI handler initialized with model: {self.model}")
        if self.cost_limit:
            self.logger.info(f"Cost limit enabled: ${self.cost_limit:.4f}")

    def _get_max_input_tokens(self) -> int:
        """Get max input tokens."""
        return max(self.max_input_tokens or (self._get_model_context_length() - self.max_tokens), 0)

    # Private methods (internal use only)
    @lru_cache(maxsize=128)
    def _get_encoding(self, model: str = None):
        """Get tiktoken encoding for the model."""
        try:
            import tiktoken
            return tiktoken.encoding_for_model(self.model)
        except (ImportError, KeyError):
            self.logger.error("Tiktoken not available or model unknown, using character estimation")
            return None

    @lru_cache(maxsize=1)
    def _get_model_context_length(self) -> int:
        """Get context length for the current model."""
        try:
            models = self.client.models.list()
            model_info = next((m for m in models.data if m.id == self.model), None)
            if model_info and hasattr(model_info, 'context_length'):
                return model_info.context_length
            else:
                return self.MODEL_CONTEXT_SIZE[self.model]
        except Exception:
            self.logger.error(f"Unknown model {self.model}, using gpt-4o context size as fallback")
        return self.MODEL_CONTEXT_SIZE['gpt-4o']

    def _count_tokens(self, messages: List[Dict], precise: bool = False) -> int:
        """Count tokens in messages with optional precision."""
        encoding = self._get_encoding(self.model)

        if not encoding or not precise:
            # Fast estimation
            total_chars = sum(len(str(msg.get('content', ''))) + len(str(msg.get('role', ''))) + 10 for msg in messages)
            return total_chars // self.chars_per_token

        # Precise counting
        total = sum(4 + sum(len(encoding.encode(str(v))) for v in msg.values()) for msg in messages)
        return total + 2  # Assistant reply primer

    def _should_use_precise_counting(self, messages: List[Dict]) -> bool:
        """Decide if precise token counting is needed."""
        if not self.manage_context:
            return False
        fast_estimate = self._count_tokens(messages, precise=False)
        threshold = self._get_max_input_tokens() * self.token_count_thres
        return fast_estimate > threshold

    def _prepare_messages(self, messages: List[Dict]) -> List[Dict]:
        """Prepare messages with smart context management."""
        if not self._should_use_precise_counting(messages):
            return messages

        max_input = self._get_max_input_tokens()
        if self._count_tokens(messages, precise=True) <= max_input:
            return messages

        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        other_msgs = messages[1:] if system_msg else messages

        result = [system_msg] if system_msg else []
        current_tokens = self._count_tokens(result, precise=True) if system_msg else 0

        # Build list of messages that fit, then reverse to maintain chronological order
        temp_msgs = []
        for msg in reversed(other_msgs):
            msg_tokens = self._count_tokens([msg], precise=True)
            if current_tokens + msg_tokens <= max_input:
                temp_msgs.append(msg)
                current_tokens += msg_tokens
            else:
                break

        # Add messages in correct order
        result.extend(reversed(temp_msgs))
        truncated = len(messages) - len(result)
        if truncated > 0:
            self.logger.info(f"Truncated {truncated} messages to fit context")

        return result

    def _get_model_pricing(self, model: str = None) -> Dict[str, float]:
        """Get pricing for a model (internal method)."""
        model = model or self.model

        if model in self.MODEL_PRICING:
            return self.MODEL_PRICING[model]

        self.logger.error(f"Unknown model {model}, using gpt-4o pricing as fallback")
        return self.MODEL_PRICING["gpt-4o"]

    def _add_cost(self, actual_cost: float, input_tokens: int = 0, output_tokens: int = 0):
        """Add actual cost to accumulated total and update tracking (internal method)."""
        import time

        self.accumulated_cost += actual_cost
        self.call_count += 1
        self.cost_history.append({
            'timestamp': time.time(),
            'cost': actual_cost,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'accumulated_cost': self.accumulated_cost
        })
        # self.logger.debug(f"Added ${actual_cost:.4f} to total cost. Total: ${self.accumulated_cost:.4f}")

    def _check_and_log_cost(self, messages: List[Dict], max_tokens: int, model: str):
        """Check cost limit including accumulated costs and log estimation."""
        if self.cost_limit is None:
            return

        input_tokens = self._count_tokens(messages, precise=True)
        estimated_cost = self._estimate_cost(input_tokens, max_tokens, model)
        total_projected_cost = self.accumulated_cost + estimated_cost

        if total_projected_cost > self.cost_limit:
            raise CostLimitExceededException(estimated_cost, self.cost_limit, self.accumulated_cost)

        # self.logger.debug(f"Estimated cost: ${estimated_cost:.4f} (input: {input_tokens}, max output: {max_tokens})")
        # self.logger.debug(f"Total cost: ${self.accumulated_cost:.4f}, Projected total: ${total_projected_cost:.4f}")

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

    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: str = None) -> float:
        """Estimate cost based on token usage (does not modify accumulated cost)."""
        pricing = self._get_model_pricing(model)
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    # Public methods (external interface)

    def get_model_pricing(self, model: str = None) -> Dict[str, float]:
        """Get pricing for a model (public interface)."""
        return self._get_model_pricing(model)

    def reset_cost_tracking(self):
        """Reset cost tracking to zero."""
        self.accumulated_cost = 0.0
        self.call_count = 0
        self.cost_history.clear()
        self.logger.info("Cost tracking reset to zero")

    def chat_completion(self, prompt: str, system_message: str = None, override_config: Dict = None,
                       response_format: Dict = None, extra_body: Dict = None,
                       conversation_history: List[Dict] = None) -> str:
        """Chat completion with cost limiting and context management."""

        # Build messages
        messages = []
        if system_message or self.system_message:
            messages.append({"role": "system", "content": system_message or self.system_message})
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})

        # Apply context management
        messages = self._prepare_messages(messages)

        # Build API parameters
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        if self.seed:
            kwargs["seed"] = self.seed
        if response_format:
            kwargs["response_format"] = response_format
        if override_config:
            kwargs.update(override_config)
        if extra_body:
            kwargs.update(extra_body)

        # Check cost limit (including accumulated costs)
        self._check_and_log_cost(messages, kwargs["max_tokens"], kwargs["model"])

        # Log request
        self.logger.debug(f"OpenAI API request parameters: {json.dumps({k: v for k, v in kwargs.items() if k != 'messages'}, indent=4)}")
        self._log_messages_with_multiline_support(messages)

        # Make API call
        try:
            response = self.client.chat.completions.create(**kwargs)
        except openai.APIError as e:
            if "maximum context length" in str(e).lower():
                self.logger.error("Context length exceeded")
            else:
                self.logger.error(f"OpenAI API error: {e}")
            raise

        content = response.choices[0].message.content.strip()
        if content is None: content = ""

        # Log usage and cost, and add to accumulated cost
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            actual_cost = self._estimate_cost(usage.prompt_tokens, usage.completion_tokens, kwargs["model"])

            # Add the actual cost to our tracking
            self._add_cost(actual_cost, usage.prompt_tokens, usage.completion_tokens)

            pricing = self._get_model_pricing(kwargs["model"])
            input_cost = (usage.prompt_tokens * pricing["input"]) / 1_000_000
            output_cost = (usage.completion_tokens * pricing["output"]) / 1_000_000

            self.logger.debug(f"Tokens: {usage.prompt_tokens}→{usage.completion_tokens}, Sum: {usage.total_tokens}")
            self.logger.debug(f"Cost: ${actual_cost:.4f} ({input_cost:.4f}→{output_cost:.4f}), Total: ${self.accumulated_cost:.2f}/{self.cost_limit:.2f}")

        self.logger.debug(f"Response: {content}")
        return content

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary including accumulated costs."""
        return {
            "cost_limit": self.cost_limit,
            "accumulated_cost": self.accumulated_cost,
            "remaining_budget": self.cost_limit - self.accumulated_cost if self.cost_limit else None,
            "call_count": self.call_count,
            "average_cost_per_call": self.accumulated_cost / self.call_count if self.call_count > 0 else 0.0,
            "model": self.model,
            "model_pricing": self._get_model_pricing(),
            "pricing_per_1m_tokens": True,
            "cost_history": self.cost_history[-10:]  # Last 10 calls for brevity
        }
