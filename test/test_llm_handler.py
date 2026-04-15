import pytest
import json
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List
import tempfile
import shutil

# Add the parent directory to the path so we can import from rome
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the real modules
from rome.logger import get_logger
from rome.config import set_attributes_from_config


def _make_mock_response(content="Test response", prompt_tokens=50, completion_tokens=25):
    """Create a mock litellm response object."""
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = content
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    mock_usage = Mock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens
    mock_usage.total_tokens = prompt_tokens + completion_tokens
    mock_response.usage = mock_usage

    return mock_response


def get_test_logger():
    """Get logger for testing"""
    from rome.logger import get_logger
    return get_logger()


# Now import the actual modules
from rome.llm_handler import LLMHandler, CostLimitExceededException, OpenAIHandler


class TestLLMHandler:
    """Test suite for LLMHandler class with cost limiting functionality"""

    @pytest.fixture
    def basic_config(self):
        """Standard configuration for testing"""
        return {
            'provider': 'openai',
            'model': 'gpt-4o',
            'temperature': 0.7,
            'max_completion_tokens': 1000,
            'max_retries': 3,
            'timeout': 30,
            'top_p': 1.0,
            'base_url': 'https://api.openai.com/v1',
            'system_message': 'You are a helpful assistant.',
            'key_name': 'OPENAI_API_KEY',
            'reasoning_effort': None,
            'manage_context': True,
            'max_input_tokens': 100000,
            'token_count_thres': 0.8,
            'chars_per_token': 4,
            'seed': 42,
            'cost_limit': 1.0
        }

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logging tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def handler(self, basic_config, temp_log_dir):
        """Create LLM handler with logging configured"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            logger = get_test_logger()
            log_config = {
                'level': 'DEBUG',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'console': False,
                'base_dir': temp_log_dir,
                'filename': 'test.log'
            }
            logger.configure(log_config)
            with patch('litellm.completion', return_value=_make_mock_response()):
                with patch('litellm.token_counter', return_value=20):
                    with patch('litellm.get_max_tokens', return_value=128000):
                        handler = LLMHandler(basic_config)
            # Patch litellm calls for subsequent test usage
            handler._completion_patch = patch('litellm.completion', return_value=_make_mock_response())
            handler._token_patch = patch('litellm.token_counter', return_value=20)
            handler._cost_patch = patch('litellm.completion_cost', return_value=0.001)
            handler._completion_patch.start()
            handler._token_patch.start()
            handler._cost_patch.start()
            return handler

    @pytest.fixture(autouse=True)
    def cleanup_patches(self, request):
        """Clean up any patches after each test"""
        yield
        # Stop patches if they were started
        if hasattr(request, 'param'):
            return
        # Try to clean up handler patches
        handler = request.node.funcargs.get('handler')
        if handler and hasattr(handler, '_completion_patch'):
            try:
                handler._completion_patch.stop()
                handler._token_patch.stop()
                handler._cost_patch.stop()
            except RuntimeError:
                pass

    def test_backward_compat_alias(self):
        """Test that OpenAIHandler is an alias for LLMHandler"""
        assert OpenAIHandler is LLMHandler

    def test_initialization(self, handler, basic_config):
        """Test proper initialization with configuration"""
        assert handler.model == 'gpt-4o'
        assert handler.temperature == 0.7
        assert handler.provider == 'openai'
        assert handler.cost_limit == 1.0
        assert handler.accumulated_cost == 0.0
        assert handler.call_count == 0
        assert isinstance(handler.cost_history, list)

    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_missing_api_key(self, basic_config, temp_log_dir):
        """Test initialization fails without API key"""
        logger = get_test_logger()
        logger.configure({'level': 'INFO', 'console': False, 'base_dir': temp_log_dir, 'filename': 'test.log'})

        with pytest.raises(ValueError, match="API key not found"):
            LLMHandler(basic_config)

    def test_get_litellm_model_openai(self, handler):
        """Test model name for OpenAI provider (no prefix)"""
        assert handler._get_litellm_model("gpt-4o") == "gpt-4o"

    def test_get_litellm_model_anthropic(self, handler):
        """Test model name for Anthropic provider"""
        handler.provider = "anthropic"
        assert handler._get_litellm_model("claude-sonnet-4-20250514") == "anthropic/claude-sonnet-4-20250514"

    def test_get_litellm_model_already_prefixed(self, handler):
        """Test model name that already has prefix"""
        assert handler._get_litellm_model("anthropic/claude-sonnet-4-20250514") == "anthropic/claude-sonnet-4-20250514"

    def test_get_base_model(self, handler):
        """Test stripping provider prefix"""
        assert handler._get_base_model("anthropic/claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"
        assert handler._get_base_model("gpt-4o") == "gpt-4o"

    def test_model_pricing_known_model(self, handler):
        """Test pricing retrieval for known models"""
        pricing = handler.get_model_pricing('gpt-4o')
        assert 'input' in pricing
        assert 'output' in pricing
        assert pricing['input'] == 2.5
        assert pricing['output'] == 10.0

    def test_model_pricing_unknown_model(self, handler):
        """Test pricing fallback for unknown models"""
        with patch('litellm.cost_per_token', side_effect=Exception("not found")):
            pricing = handler.get_model_pricing('unknown-model')
            assert pricing['input'] == 2.5  # gpt-4o fallback
            assert pricing['output'] == 10.0

    def test_token_counting_fast(self, handler):
        """Test fast token estimation"""
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        tokens = handler._count_tokens(messages, precise=False)
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_token_counting_precise(self, handler):
        """Test precise token counting via litellm"""
        messages = [{"role": "user", "content": "Hello world"}]
        tokens = handler._count_tokens(messages, precise=True)
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_context_management_no_truncation(self, handler):
        """Test context management when no truncation needed"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        prepared = handler._prepare_messages(messages)
        assert len(prepared) == len(messages)
        assert prepared[0]["role"] == "system"

    @patch('rome.llm_handler.LLMHandler._init_compressor')
    def test_context_management_with_truncation(self, mock_init_compressor, handler):
        """Test context management with forced truncation"""
        mock_init_compressor.return_value = None
        handler.compressor = None

        handler.max_input_tokens = 50
        handler.manage_context = True

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "First message with lots of content that takes many tokens"},
            {"role": "assistant", "content": "Response to first message"},
            {"role": "user", "content": "Second message"},
            {"role": "assistant", "content": "Response to second"},
            {"role": "user", "content": "Final message"}
        ]

        prepared = handler._prepare_messages(messages)

        system_msgs = [msg for msg in prepared if msg["role"] == "system"]
        if any(msg["role"] == "system" for msg in messages):
            assert len(system_msgs) == 1

        assert len(prepared) <= len(messages)

    def test_cost_estimation(self, handler):
        """Test cost estimation functionality"""
        cost = handler._estimate_cost(input_tokens=1000, output_tokens=500, model='gpt-4o')
        expected = (1000 * 2.5 + 500 * 10.0) / 1_000_000
        assert abs(cost - expected) < 1e-6

    def test_cost_limit_check_pass(self, handler):
        """Test cost limit check that should pass"""
        messages = [{"role": "user", "content": "Short message"}]
        handler._check_and_log_cost(messages, 100, 'gpt-4o')

    def test_cost_limit_check_fail(self, handler):
        """Test cost limit check that should fail"""
        handler.cost_limit = 0.001
        messages = [{"role": "user", "content": "This is a message that will exceed cost limit"}]

        with pytest.raises(CostLimitExceededException):
            handler._check_and_log_cost(messages, 1000, 'gpt-4o')

    def test_cost_limit_with_accumulated_cost(self, handler):
        """Test cost limit considering accumulated costs"""
        handler.accumulated_cost = 0.8
        handler.cost_limit = 1.0

        with patch.object(handler, '_estimate_cost', return_value=0.3):
            messages = [{"role": "user", "content": "Test message"}]

            with pytest.raises(CostLimitExceededException) as exc_info:
                handler._check_and_log_cost(messages, 1000, 'gpt-4o')

            assert exc_info.value.accumulated_cost == 0.8
            assert exc_info.value.estimated_cost == 0.3
            assert exc_info.value.cost_limit == 1.0

    def test_chat_completion_basic(self, handler):
        """Test basic chat completion functionality"""
        response = handler.chat_completion("Hello, world!")
        assert response == "Test response"
        assert handler.call_count == 1
        assert handler.accumulated_cost > 0

    def test_chat_completion_with_system_message(self, handler):
        """Test chat completion with custom system message"""
        response = handler.chat_completion(
            "Hello",
            system_message="You are a coding assistant"
        )
        assert response == "Test response"

    def test_chat_completion_with_conversation_history(self, handler):
        """Test chat completion with conversation history"""
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        response = handler.chat_completion("Follow-up", conversation_history=history)
        assert response == "Test response"

    def test_chat_completion_with_overrides(self, handler):
        """Test chat completion with parameter overrides"""
        overrides = {'temperature': 0.9, 'max_tokens': 2000}
        response = handler.chat_completion("Hello", override_config=overrides)
        assert response == "Test response"

    def test_chat_completion_with_response_format(self, handler):
        """Test chat completion with response format specification"""
        response_format = {"type": "json_object"}
        response = handler.chat_completion("Return JSON", response_format=response_format)
        assert response == "Test response"

    def test_cost_tracking_accumulation(self, handler):
        """Test that costs accumulate properly across calls"""
        initial_cost = handler.accumulated_cost

        handler.chat_completion("First call")
        first_cost = handler.accumulated_cost

        handler.chat_completion("Second call")
        second_cost = handler.accumulated_cost

        assert first_cost > initial_cost
        assert second_cost > first_cost
        assert handler.call_count == 2
        assert len(handler.cost_history) == 2

    def test_cost_summary(self, handler):
        """Test comprehensive cost summary"""
        handler.chat_completion("Test message")
        summary = handler.get_cost_summary()

        assert 'cost_limit' in summary
        assert 'accumulated_cost' in summary
        assert 'remaining_budget' in summary
        assert 'call_count' in summary
        assert 'average_cost_per_call' in summary
        assert 'provider' in summary
        assert 'model' in summary
        assert 'model_pricing' in summary
        assert 'cost_history' in summary

        assert summary['call_count'] == 1
        assert summary['accumulated_cost'] > 0
        assert summary['model'] == 'gpt-4o'
        assert summary['provider'] == 'openai'

    def test_reset_cost_tracking(self, handler):
        """Test cost tracking reset functionality"""
        handler.chat_completion("Test")
        assert handler.accumulated_cost > 0

        handler.reset_cost_tracking()
        assert handler.accumulated_cost == 0.0
        assert handler.call_count == 0
        assert len(handler.cost_history) == 0

    def test_cost_limit_exception_details(self, handler):
        """Test CostLimitExceededException contains proper details"""
        try:
            raise CostLimitExceededException(0.5, 0.3, 0.1)
        except CostLimitExceededException as e:
            assert e.estimated_cost == 0.5
            assert e.cost_limit == 0.3
            assert e.accumulated_cost == 0.1
            assert "0.6" in str(e)
            assert "exceeding limit" in str(e)

    def test_model_context_sizes(self, handler):
        """Test model context size retrieval"""
        size = handler.MODEL_CONTEXT_SIZE.get('gpt-4o')
        assert size == 128000

    def test_no_cost_limit_scenario(self, basic_config, temp_log_dir):
        """Test behavior when no cost limit is set"""
        basic_config['cost_limit'] = None

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            logger = get_test_logger()
            logger.configure({
                'level': 'DEBUG', 'console': False,
                'base_dir': temp_log_dir, 'filename': 'test.log'
            })

            with patch('litellm.token_counter', return_value=20):
                with patch('litellm.get_max_tokens', return_value=128000):
                    handler = LLMHandler(basic_config)

            assert handler.cost_limit is None
            messages = [{"role": "user", "content": "Very expensive message " * 1000}]
            handler._check_and_log_cost(messages, 10000, 'gpt-4o')

    def test_context_management_disabled(self, handler):
        """Test behavior when context management is disabled"""
        handler.manage_context = False

        large_messages = [{"role": "user", "content": "Very long message " * 1000}]
        assert not handler._should_use_precise_counting(large_messages)

        prepared = handler._prepare_messages(large_messages)
        assert len(prepared) == len(large_messages)

    def test_cost_history_tracking(self, handler):
        """Test detailed cost history tracking"""
        handler.chat_completion("First call")
        handler.chat_completion("Second call")
        handler.chat_completion("Third call")

        assert len(handler.cost_history) == 3

        for entry in handler.cost_history:
            assert 'timestamp' in entry
            assert 'cost' in entry
            assert 'input_tokens' in entry
            assert 'output_tokens' in entry
            assert 'total_tokens' in entry
            assert 'accumulated_cost' in entry

        assert handler.cost_history[1]['accumulated_cost'] > handler.cost_history[0]['accumulated_cost']
        assert handler.cost_history[2]['accumulated_cost'] > handler.cost_history[1]['accumulated_cost']

    def test_map_reasoning_params_openai(self, handler):
        """Test reasoning parameter mapping for OpenAI"""
        kwargs = {"model": "o3", "reasoning_effort": "high", "temperature": 0.5}
        result = handler._map_reasoning_params(kwargs)
        assert result["reasoning_effort"] == "high"
        assert "temperature" not in result

    def test_map_reasoning_params_anthropic(self, handler):
        """Test reasoning parameter mapping for Anthropic"""
        handler.provider = "anthropic"
        kwargs = {"model": "anthropic/claude-sonnet-4-20250514", "reasoning_effort": "medium", "temperature": 0.5}
        result = handler._map_reasoning_params(kwargs)
        assert "thinking" in result
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 20000
        assert "temperature" not in result
        assert "reasoning_effort" not in result


def test_config_module_integration():
    """Test integration with config module"""
    class TestObj:
        pass

    obj = TestObj()
    config = {
        'test_attr': 'test_value',
        'number_attr': 42,
        'bool_attr': True
    }

    set_attributes_from_config(obj, config, required_attrs=['test_attr'])
    assert obj.test_attr == 'test_value'
    assert obj.number_attr == 42
    assert obj.bool_attr is True


def test_logger_module_integration():
    """Test integration with logger module"""
    temp_dir = tempfile.mkdtemp()
    try:
        logger = get_test_logger()
        log_config = {
            'level': 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_dir,
            'filename': 'integration_test.log'
        }
        logger.configure(log_config)

        logger.info("Integration test message")
        log_file = os.path.join(temp_dir, 'integration_test.log')
        assert os.path.exists(log_file)

        with open(log_file, 'r') as f:
            content = f.read()
            assert "Integration test message" in content
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    try:
        import pytest
        exit_code = pytest.main([__file__, "-v", "--tb=short"])
        sys.exit(exit_code)
    except ImportError:
        print("Error: pytest not installed. Please install with: pip install pytest")
        sys.exit(1)
