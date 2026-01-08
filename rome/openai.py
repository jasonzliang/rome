from functools import lru_cache
import json
import logging
import openai
import os
from typing import Dict, Optional, Any, Union, List

from .logger import get_logger
from .config import set_attributes_from_config, DEFAULT_CONFIG


class CostLimitExceededException(Exception):
    """Exception raised when the cost limit is exceeded."""
    def __init__(self, estimated_cost: float, cost_limit: float, accumulated_cost: float = None):
        self.estimated_cost = estimated_cost
        self.cost_limit = cost_limit
        self.accumulated_cost = accumulated_cost
        if accumulated_cost is not None:
            super().__init__(f"Estimated cost ${estimated_cost:.4f} would bring total to ${accumulated_cost + estimated_cost:.2f}, exceeding limit ${cost_limit:.2f}")
        else:
            super().__init__(f"Estimated cost ${estimated_cost:.4f} exceeds limit ${cost_limit:.2f}")


class OpenAIHandler:
    """Handler class for OpenAI API interactions with configuration dictionary and cost limiting"""

    # OpenAI model pricing (per 1M tokens) - Updated with latest models
    MODEL_PRICING = {
        # GPT-5 series (as of November 2025)
        "gpt-5.2": {"input": 1.75, "output": 14.0},
        "gpt-5.1": {"input": 1.25, "output": 10.0},
        "gpt-5": {"input": 1.25, "output": 10.0},
        "gpt-5-mini": {"input": 0.25, "output": 2.0},
        "gpt-5-nano": {"input": 0.05, "output": 0.40},
        "gpt-5-pro": {"input": 15.0, "output": 120.0},

        # GPT-4.1 series (as of April 2025)
        "gpt-4.1": {"input": 2.0, "output": 8.0},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},

        # GPT-4o series (standard text)
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},

        # o-series reasoning models (note: pricing fluctuates)
        "o1": {"input": 15.0, "output": 60.0},
        "o1-mini": {"input": 1.10, "output": 4.40},
        "o1-pro": {"input": 150.0, "output": 600.0},
        "o3": {"input": 2.0, "output": 8.0},
        "o3-mini": {"input": 1.10, "output": 4.40},
        "o4-mini": {"input": 1.10, "output": 4.40},

        # Realtime models (text)
        "gpt-realtime": {"input": 4.0, "output": 16.0},
        "gpt-realtime-mini": {"input": 0.60, "output": 2.40},
    }

    MODEL_CONTEXT_SIZE = {
        # GPT-5 series
        "gpt-5.2": 272000,
        "gpt-5.1": 272000,
        "gpt-5": 272000,
        "gpt-5-mini": 272000,
        "gpt-5-nano": 272000,
        "gpt-5-pro": 272000,

        # GPT-4.1 series
        "gpt-4.1": 1047576,
        "gpt-4.1-mini": 1047576,
        "gpt-4.1-nano": 1047576,

        # GPT-4o series
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,

        # o-series reasoning models
        "o1": 200000,
        "o1-mini": 128000,
        "o1-pro": 200000,
        "o3": 200000,
        "o3-mini": 200000,
        "o4-mini": 200000,

        # Realtime models
        "gpt-realtime": 128000,
        "gpt-realtime-mini": 128000,
    }

    # Newer models with different API
    REASONING_MODELS = {
        "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro",
        "o1", "o1-mini", "o1-pro", "o3", "o3-mini", "o4-mini",
    }

    # Unique identifier for chat completion requests
    USER_ID = "cognizant-ai-labs-jason"

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = get_logger()

        # Initialize cost tracking
        self.accumulated_cost = 0.0
        self.call_count = 0
        self.cost_history = []

        # Set attributes from config
        set_attributes_from_config(self, self.config, DEFAULT_CONFIG['OpenAIHandler'].keys())

        # Initialize client
        api_key = os.getenv(self.key_name or 'OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment")

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=5
        )

        self.logger.info(f"OpenAI handler initialized with model: {self.model}")
        if self.cost_limit:
            self.logger.info(f"Cost limit enabled: ${self.cost_limit:.2f}")

    def _is_reasoning_model(self, model: str = None) -> bool:
        """Check if model is older one."""
        model = model or self.model
        return model in self.REASONING_MODELS

    def _get_max_input_tokens(self) -> int:
        """Get max input tokens."""
        return max(self.max_input_tokens or (self._get_model_context_length() - self.max_completion_tokens), 0)

    @lru_cache(maxsize=128)
    def _get_encoding(self, model: str = None):
        """Get tiktoken encoding for the model."""
        try:
            import tiktoken
            # Hack to ensure Tiktoken support for latest models
            if model.startswith(("gpt-5", "o4")):
                return tiktoken.get_encoding("o200k_base")

            return tiktoken.encoding_for_model(model)
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
            total_chars = sum(len(str(msg.get('content', ''))) + len(str(msg.get('role', ''))) + 10 for msg in messages)
            return total_chars // self.chars_per_token

        total = sum(4 + sum(len(encoding.encode(str(v))) for v in msg.values()) for msg in messages)
        return total + 2

    def _should_use_precise_counting(self, messages: List[Dict]) -> bool:
        """Decide if precise token counting is needed."""
        if not self.manage_context:
            return False
        fast_estimate = self._count_tokens(messages, precise=False)
        threshold = self._get_max_input_tokens() * self.token_count_thres
        return fast_estimate > threshold

    def _prepare_messages(self, messages: List[Dict]) -> List[Dict]:
        """Prepare messages with smart context management and LLMLingua-2 compression."""
        if not self._should_use_precise_counting(messages):
            return messages

        max_input = self._get_max_input_tokens()
        if self._count_tokens(messages, precise=True) <= max_input:
            return messages

        self._init_compressor()

        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        other_msgs = messages[1:] if system_msg else messages

        result = [system_msg] if system_msg else []
        current_tokens = self._count_tokens(result, precise=True) if system_msg else 0

        temp_msgs, compressed_count = self._fit_messages_with_compression(
            reversed(other_msgs), max_input, current_tokens
        )

        result.extend(reversed(temp_msgs))
        self._log_context_changes(len(messages) - len(result), compressed_count)
        return result

    def _init_compressor(self):
        """Initialize LLMLingua-2 compressor (lazy loading)."""
        if hasattr(self, 'compressor'):
            return
        try:
            from llmlingua import PromptCompressor
            self.compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map="cpu"
            )
            self.logger.info("LLMLingua-2 compressor initialized")
        except ImportError:
            self.logger.error("Text compression unavailable, please install: 'pip install llmlingua'")
            self.compressor = None

    def _fit_messages_with_compression(self, messages, max_input, current_tokens):
        """Fit messages using compression when needed."""
        temp_msgs, compressed_count = [], 0

        for msg in messages:
            msg_tokens = self._count_tokens([msg], precise=True)

            if current_tokens + msg_tokens <= max_input:
                temp_msgs.append(msg)
                current_tokens += msg_tokens
            elif compressed_msg := self._try_compress_message(msg, max_input - current_tokens, msg_tokens):
                temp_msgs.append(compressed_msg)
                current_tokens += self._count_tokens([compressed_msg], precise=True)
                compressed_count += 1
            else:
                break

        return temp_msgs, compressed_count

    def _try_compress_message(self, msg, remaining_tokens, msg_tokens):
        """Try to compress a message to fit in remaining token budget."""
        if not (self.compressor and msg.get('content')):
            return None

        compression_rate = min(0.9, remaining_tokens / msg_tokens)
        if compression_rate <= 0.1:
            return None

        result = self.compressor.compress_prompt(
            msg['content'],
            rate=compression_rate,
            force_tokens=['\n', '?', '!', '.', ',']
        )
        compressed_msg = msg.copy()
        compressed_msg['content'] = result['compressed_prompt']

        if self._count_tokens([compressed_msg], precise=True) <= remaining_tokens:
            self.logger.debug(f"Compressed: {msg_tokens}→{self._count_tokens([compressed_msg], precise=True)} ({result.get('ratio', 'N/A')})")
            return compressed_msg

        return None

    def _log_context_changes(self, truncated, compressed_count):
        """Log context management changes."""
        if truncated or compressed_count:
            parts = []
            if compressed_count:
                parts.append(f"compressed {compressed_count}")
            if truncated:
                parts.append(f"truncated {truncated}")
            self.logger.info(f"Context management: {', '.join(parts)} messages")

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

    def _check_and_log_cost(self, messages: List[Dict], max_completion_tokens: int, model: str):
        """Check cost limit including accumulated costs and log estimation."""
        if not self.cost_limit:
            return

        input_tokens = self._count_tokens(messages, precise=True)
        estimated_cost = self._estimate_cost(input_tokens, max_completion_tokens, model)
        total_projected_cost = self.accumulated_cost + estimated_cost

        if total_projected_cost > self.cost_limit:
            raise CostLimitExceededException(estimated_cost, self.cost_limit, self.accumulated_cost)

    def _log_messages_with_multiline_support(self, messages):
        """Log messages with proper multiline string formatting"""
        self.logger.debug("Request messages:")

        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            self.logger.debug(f"[{i}] {role}:")

            if '\n' in content:
                for line in content.split('\n'):
                    self.logger.debug(f"    {line}")
            else:
                self.logger.debug(f"    {content}")

            if i < len(messages) - 1:
                self.logger.debug("----------")

    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: str = None) -> float:
        """Estimate cost based on token usage (does not modify accumulated cost)."""
        pricing = self._get_model_pricing(model)
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    # Public methods

    def get_model_pricing(self, model: str = None) -> Dict[str, float]:
        """Get pricing for a model (public interface)."""
        return self._get_model_pricing(model)

    def reset_cost_tracking(self):
        """Reset cost tracking to zero."""
        self.accumulated_cost = 0.0
        self.call_count = 0
        self.cost_history.clear()
        self.logger.info("Cost tracking reset to zero")

    def chat_completion(self, prompt: str,
        system_message: str = None,
        override_config: Dict = None,
        response_format: Dict = None,
        conversation_history: List[Dict] = None,
        **kwargs) -> str:
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
            "top_p": self.top_p,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "user": self.USER_ID,
        }
        if override_config:
            kwargs.update(override_config)

        if self.seed:
            kwargs["seed"] = self.seed
        if response_format:
            kwargs["response_format"] = response_format
        if self.reasoning_effort and "reasoning_effort" not in kwargs \
            and self._is_reasoning_model(kwargs['model']):
            if self.model.startswith("o") and self.reasoning_effort == "minimal":
                self.reasoning_effort = "low"
            kwargs["reasoning_effort"] = self.reasoning_effort
        if "reasoning_effort" in kwargs and not self._is_reasoning_model(kwargs['model']):
            del kwargs["reasoning_effort"]
        if self._is_reasoning_model(kwargs["model"]):
            if "temperature" in kwargs:
                del kwargs["temperature"]

        # Check cost limit (including accumulated costs)
        self._check_and_log_cost(messages, self.max_completion_tokens, kwargs["model"])

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
            "cost_history": self.cost_history[-10:]
        }
