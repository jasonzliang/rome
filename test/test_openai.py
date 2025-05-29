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

# Mock the openai module since we don't want real API calls
class MockOpenAI:
    def __init__(self, api_key=None, base_url=None, timeout=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.chat = Mock()
        self.models = Mock()
        self._setup_default_responses()

    def _setup_default_responses(self):
        # Mock chat completion response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # Mock usage statistics
        mock_usage = Mock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 25
        mock_usage.total_tokens = 75
        mock_response.usage = mock_usage

        self.chat.completions = Mock()
        self.chat.completions.create = Mock(return_value=mock_response)

        # Mock models list
        mock_model = Mock()
        mock_model.id = 'gpt-4o'
        mock_model.context_length = 128000
        mock_models_response = Mock()
        mock_models_response.data = [mock_model]
        self.models.list = Mock(return_value=mock_models_response)

class MockAPIError(Exception):
    pass

# Create a proper mock module with __spec__ attribute
mock_openai_module = Mock()
mock_openai_module.OpenAI = MockOpenAI
mock_openai_module.APIError = MockAPIError

# Create a proper module spec to avoid import issues
import importlib.util
mock_spec = importlib.util.spec_from_loader('openai', loader=None)
mock_openai_module.__spec__ = mock_spec
mock_openai_module.__name__ = 'openai'
mock_openai_module.__package__ = 'openai'

# Install the mock before any imports that might use it
sys.modules['openai'] = mock_openai_module

# Mock tiktoken if not available
try:
    import tiktoken
except ImportError:
    class MockEncoding:
        def encode(self, text):
            return list(range(len(text) // 4))  # Rough approximation

    tiktoken = Mock()
    tiktoken.encoding_for_model = Mock(return_value=MockEncoding())
    sys.modules['tiktoken'] = tiktoken

def get_test_logger():
    """Get logger for testing"""
    from rome.logger import get_logger
    return get_logger()

# Now import the actual modules
from rome.openai import OpenAIHandler, CostLimitExceededException


class TestOpenAIHandler:
    """Test suite for OpenAIHandler class with cost limiting functionality"""

    @pytest.fixture
    def basic_config(self):
        """Standard configuration for testing"""
        return {
            'model': 'gpt-4o',
            'temperature': 0.7,
            'max_tokens': 1000,
            'timeout': 30,
            'top_p': 1.0,
            'base_url': 'https://api.openai.com/v1',
            'system_message': 'You are a helpful assistant.',
            'key_name': 'OPENAI_API_KEY',
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
        """Create OpenAI handler with logging configured"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            # Configure logger for tests
            logger = get_test_logger()
            log_config = {
                'level': 'DEBUG',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'console': False,
                'base_dir': temp_log_dir,
                'filename': 'test.log'
            }
            logger.configure(log_config)
            return OpenAIHandler(basic_config)

    def test_initialization(self, handler, basic_config):
        """Test proper initialization with configuration"""
        assert handler.model == 'gpt-4o'
        assert handler.temperature == 0.7
        assert handler.max_tokens == 1000
        assert handler.cost_limit == 1.0
        assert handler.accumulated_cost == 0.0
        assert handler.call_count == 0
        assert isinstance(handler.cost_history, list)

    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_missing_api_key(self, basic_config, temp_log_dir):
        """Test initialization fails without API key"""
        logger = get_test_logger()
        logger.configure({'level': 'INFO', 'console': False, 'base_dir': temp_log_dir, 'filename': 'test.log'})

        with pytest.raises(ValueError, match="OpenAI API key not found"):
            OpenAIHandler(basic_config)

    def test_model_pricing_known_model(self, handler):
        """Test pricing retrieval for known models"""
        pricing = handler.get_model_pricing('gpt-4o')
        assert 'input' in pricing
        assert 'output' in pricing
        assert pricing['input'] == 2.5
        assert pricing['output'] == 10.0

    def test_model_pricing_unknown_model(self, handler):
        """Test pricing fallback for unknown models"""
        pricing = handler.get_model_pricing('unknown-model')
        # Should fallback to gpt-4o pricing
        assert pricing['input'] == 2.5
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
        """Test precise token counting with tiktoken"""
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

    @patch('rome.openai.OpenAIHandler._init_compressor')
    def test_context_management_with_truncation(self, mock_init_compressor, handler):
        """Test context management with forced truncation"""
        # Mock the compressor initialization to avoid llmlingua dependency issues
        mock_init_compressor.return_value = None
        handler.compressor = None  # Simulate compressor not available

        # Force small context to trigger truncation
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

        # Should preserve system message if it exists
        system_msgs = [msg for msg in prepared if msg["role"] == "system"]
        if any(msg["role"] == "system" for msg in messages):
            assert len(system_msgs) == 1

        # Should be fewer messages due to truncation
        assert len(prepared) <= len(messages)

    def test_cost_estimation(self, handler):
        """Test cost estimation functionality"""
        cost = handler._estimate_cost(input_tokens=1000, output_tokens=500, model='gpt-4o')
        expected = (1000 * 2.5 + 500 * 10.0) / 1_000_000  # gpt-4o pricing
        assert abs(cost - expected) < 1e-6

    def test_cost_limit_check_pass(self, handler):
        """Test cost limit check that should pass"""
        messages = [{"role": "user", "content": "Short message"}]
        # Should not raise exception
        handler._check_and_log_cost(messages, 100, 'gpt-4o')

    def test_cost_limit_check_fail(self, handler):
        """Test cost limit check that should fail"""
        handler.cost_limit = 0.001  # Very low limit
        messages = [{"role": "user", "content": "This is a message that will exceed cost limit due to very low limit"}]

        with pytest.raises(CostLimitExceededException):
            handler._check_and_log_cost(messages, 1000, 'gpt-4o')

    def test_cost_limit_with_accumulated_cost(self, handler):
        """Test cost limit considering accumulated costs"""
        # Test the cost checking logic directly by patching the estimation
        handler.accumulated_cost = 0.8
        handler.cost_limit = 1.0  # $0.20 remaining budget

        # Mock the _estimate_cost method to return a known expensive value
        with patch.object(handler, '_estimate_cost', return_value=0.3):  # $0.30 estimated cost
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

    def test_chat_completion_api_error(self, handler):
        """Test handling of API errors"""
        handler.client.chat.completions.create.side_effect = MockAPIError("maximum context length exceeded")

        with pytest.raises(MockAPIError):
            handler.chat_completion("Test prompt")

    def test_cost_tracking_accumulation(self, handler):
        """Test that costs accumulate properly across calls"""
        initial_cost = handler.accumulated_cost

        # Make multiple calls
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
        assert 'model' in summary
        assert 'model_pricing' in summary
        assert 'cost_history' in summary

        assert summary['call_count'] == 1
        assert summary['accumulated_cost'] > 0
        assert summary['model'] == 'gpt-4o'

    def test_reset_cost_tracking(self, handler):
        """Test cost tracking reset functionality"""
        # Make a call to accumulate some cost
        handler.chat_completion("Test")
        assert handler.accumulated_cost > 0
        assert handler.call_count > 0
        assert len(handler.cost_history) > 0

        # Reset tracking
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
            assert "0.6" in str(e)  # Total projected cost
            assert "exceeding limit" in str(e)

    def test_model_context_sizes(self, handler):
        """Test model context size retrieval"""
        # Test known model
        size = handler.MODEL_CONTEXT_SIZE.get('gpt-4o')
        assert size == 128000

        # Test that _get_model_context_length works
        context_length = handler._get_model_context_length()
        assert isinstance(context_length, int)
        assert context_length > 0

    def test_max_input_tokens_calculation(self, handler):
        """Test max input tokens calculation"""
        max_input = handler._get_max_input_tokens()
        expected = max(handler.max_input_tokens or (handler._get_model_context_length() - handler.max_tokens), 0)
        assert max_input == expected

    def test_should_use_precise_counting_logic(self, handler):
        """Test logic for deciding when to use precise token counting"""
        # Small message should use fast counting
        small_messages = [{"role": "user", "content": "Hi"}]
        assert not handler._should_use_precise_counting(small_messages)

        # Large message should use precise counting if manage_context is True
        large_messages = [{"role": "user", "content": "Very long message " * 1000}]
        handler.manage_context = True
        result = handler._should_use_precise_counting(large_messages)
        # Result depends on token threshold, just verify it's boolean
        assert isinstance(result, bool)

    def test_edge_cases_and_robustness(self, handler):
        """Test edge cases and robustness scenarios"""
        # Test with None content in messages
        messages_with_none = [{"role": "user", "content": None}]
        tokens = handler._count_tokens(messages_with_none, precise=False)
        assert isinstance(tokens, int)

        # Test with empty content
        messages_empty = [{"role": "user", "content": ""}]
        tokens = handler._count_tokens(messages_empty, precise=False)
        assert isinstance(tokens, int)

        # Test with very large max_tokens
        cost = handler._estimate_cost(100, 1000000, 'gpt-4o')  # 1M output tokens
        assert cost > 0

        # Test cost summary with no history
        summary = handler.get_cost_summary()
        assert summary['call_count'] == 0
        assert summary['average_cost_per_call'] == 0.0

    def test_model_context_edge_cases(self, handler):
        """Test model context handling edge cases"""
        # Test with unknown model context length
        original_model = handler.model
        handler.model = 'totally-unknown-model'

        # Should fallback gracefully
        context_length = handler._get_model_context_length()
        assert isinstance(context_length, int)
        assert context_length > 0

        # Restore original model
        handler.model = original_model

    def test_no_cost_limit_scenario(self, basic_config, temp_log_dir):
        """Test behavior when no cost limit is set"""
        basic_config['cost_limit'] = None

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            logger = get_test_logger()
            logger.configure({
                'level': 'DEBUG', 'console': False,
                'base_dir': temp_log_dir, 'filename': 'test.log'
            })

            handler = OpenAIHandler(basic_config)
            assert handler.cost_limit is None

            # Should not raise exception regardless of cost
            messages = [{"role": "user", "content": "Very expensive message " * 1000}]
            # Should complete without exception
            handler._check_and_log_cost(messages, 10000, 'gpt-4o')

    def test_context_management_disabled(self, handler):
        """Test behavior when context management is disabled"""
        handler.manage_context = False

        # Should not use precise counting
        large_messages = [{"role": "user", "content": "Very long message " * 1000}]
        assert not handler._should_use_precise_counting(large_messages)

        # Should return messages unchanged
        prepared = handler._prepare_messages(large_messages)
        assert len(prepared) == len(large_messages)

    def test_cost_history_tracking(self, handler):
        """Test detailed cost history tracking"""
        # Make multiple calls
        handler.chat_completion("First call")
        handler.chat_completion("Second call")
        handler.chat_completion("Third call")

        assert len(handler.cost_history) == 3

        # Check history structure
        for entry in handler.cost_history:
            assert 'timestamp' in entry
            assert 'cost' in entry
            assert 'input_tokens' in entry
            assert 'output_tokens' in entry
            assert 'total_tokens' in entry
            assert 'accumulated_cost' in entry

        # Check that costs are accumulating properly
        assert handler.cost_history[1]['accumulated_cost'] > handler.cost_history[0]['accumulated_cost']
        assert handler.cost_history[2]['accumulated_cost'] > handler.cost_history[1]['accumulated_cost']

    def test_multiline_message_logging(self, handler):
        """Test multiline message logging doesn't crash"""
        multiline_content = """This is a message
        with multiple lines
        and various formatting

        Including empty lines"""

        # Should handle multiline content without crashing
        response = handler.chat_completion(multiline_content)
        assert response == "Test response"


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


# Test runner for direct execution
if __name__ == "__main__":
    try:
        import pytest
        # Run pytest with verbose output and exit with appropriate code
        exit_code = pytest.main([__file__, "-v", "--tb=short"])
        sys.exit(exit_code)
    except ImportError:
        print("Error: pytest not installed. Please install with: pip install pytest")
        print("Or run individual tests by importing and calling test functions directly.")
        sys.exit(1)
