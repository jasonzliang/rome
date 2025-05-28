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

        # Set up default responses
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
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage

        self.chat.completions = Mock()
        self.chat.completions.create = Mock(return_value=mock_response)

        # Mock models list
        mock_model = Mock()
        mock_model.id = 'gpt-4'
        mock_model.context_length = 8192
        mock_models_response = Mock()
        mock_models_response.data = [mock_model]
        self.models.list = Mock(return_value=mock_models_response)

class MockOpenAIException(Exception):
    pass

class MockBadRequestError(MockOpenAIException):
    def __init__(self, message, response=None, body=None):
        super().__init__(message)
        self.response = response
        self.body = body

# Mock the openai module
mock_openai_module = Mock()
mock_openai_module.OpenAI = MockOpenAI
mock_openai_module.BadRequestError = MockBadRequestError
sys.modules['openai'] = mock_openai_module

# Mock tiktoken if not available
try:
    import tiktoken
except ImportError:
    tiktoken = Mock()
    tiktoken.encoding_for_model = Mock(return_value=None)
    sys.modules['tiktoken'] = tiktoken

# Now import the actual OpenAI handler
from rome.openai import OpenAIHandler


class TestOpenAIHandler:
    """Test suite for OpenAIHandler class"""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing"""
        return {
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 1000,
            'timeout': 30,
            'top_p': 1.0,
            'base_url': 'https://api.openai.com/v1',
            'system_message': 'You are a helpful assistant.',
            'key_name': 'OPENAI_API_KEY',
            'manage_context': True,
            'max_input_tokens': 4000,
            'token_count_thres': 0.8,
            'chars_per_token': 4,
            'seed': 42
        }

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for logging tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_init_with_config(self, basic_config, temp_log_dir):
        """Test initialization with configuration"""
        # Configure logger to use temp directory
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,  # Disable console for tests
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        handler = OpenAIHandler(basic_config)

        # Check that attributes are set correctly using real config function
        assert handler.model == 'gpt-4'
        assert handler.temperature == 0.7
        assert handler.max_tokens == 1000
        assert handler.timeout == 30
        assert handler.manage_context is True

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key(self, basic_config, temp_log_dir):
        """Test initialization fails without API key"""
        # Configure logger
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        basic_config['key_name'] = 'MISSING_KEY'

        with pytest.raises(ValueError, match="OpenAI API key not found"):
            OpenAIHandler(basic_config)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_get_max_input_tokens_manual(self, basic_config, temp_log_dir):
        """Test max input tokens when manually set"""
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        basic_config['max_input_tokens'] = 5000
        handler = OpenAIHandler(basic_config)

        assert handler.get_max_input_tokens() == 5000

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_get_max_input_tokens_auto(self, basic_config, temp_log_dir):
        """Test max input tokens auto-calculation"""
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        basic_config['max_input_tokens'] = None
        handler = OpenAIHandler(basic_config)

        # Should be context_length - max_tokens = 8192 - 1000 = 7192
        # But since we're mocking, we need to ensure the mock returns the right value
        max_tokens = handler.get_max_input_tokens()
        assert isinstance(max_tokens, int)
        assert max_tokens > 0

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_estimate_tokens_fast(self, basic_config, temp_log_dir):
        """Test fast token estimation"""
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        handler = OpenAIHandler(basic_config)

        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well!"}
        ]

        tokens = handler._estimate_tokens_fast(messages)
        assert isinstance(tokens, int)
        assert tokens > 0

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_count_message_tokens_precise(self, basic_config, temp_log_dir):
        """Test precise token counting"""
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        handler = OpenAIHandler(basic_config)

        message = {"role": "user", "content": "Hello world"}
        tokens = handler._count_message_tokens_precise(message)

        assert isinstance(tokens, int)
        assert tokens > 0

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_truncate_messages_with_system(self, basic_config, temp_log_dir):
        """Test message truncation preserving system message"""
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        handler = OpenAIHandler(basic_config)
        # Set a very small token limit to force truncation
        handler.max_input_tokens = 10  # Extremely small to ensure truncation happens

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]

        truncated = handler._truncate_messages(messages)

        # System message should be preserved if it exists
        if any(msg["role"] == "system" for msg in messages):
            # Find system message in result
            system_messages = [msg for msg in truncated if msg["role"] == "system"]
            assert len(system_messages) > 0, "System message should be preserved"
            assert system_messages[0]["content"] == "You are helpful"

        assert len(truncated) <= len(messages)

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_chat_completion_basic(self, basic_config, temp_log_dir):
        """Test basic chat completion"""
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        handler = OpenAIHandler(basic_config)

        response = handler.chat_completion("Hello, world!")

        assert response == "Test response"

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_chat_completion_with_system_message(self, basic_config, temp_log_dir):
        """Test chat completion with custom system message"""
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        handler = OpenAIHandler(basic_config)

        response = handler.chat_completion(
            "Hello, world!",
            system_message="You are a coding assistant"
        )

        assert response == "Test response"

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_chat_completion_with_conversation_history(self, basic_config, temp_log_dir):
        """Test chat completion with conversation history"""
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        handler = OpenAIHandler(basic_config)

        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]

        response = handler.chat_completion(
            "Follow-up question",
            conversation_history=history
        )

        assert response == "Test response"

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_chat_completion_with_overrides(self, basic_config, temp_log_dir):
        """Test chat completion with config overrides"""
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        handler = OpenAIHandler(basic_config)

        overrides = {
            'temperature': 0.9,
            'max_tokens': 2000
        }

        response = handler.chat_completion(
            "Hello",
            override_config=overrides
        )

        assert response == "Test response"

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_chat_completion_with_response_format(self, basic_config, temp_log_dir):
        """Test chat completion with response format"""
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        handler = OpenAIHandler(basic_config)

        response_format = {"type": "json_object"}

        response = handler.chat_completion(
            "Return JSON",
            response_format=response_format
        )

        assert response == "Test response"

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_context_management_flow(self, basic_config, temp_log_dir):
        """Test the complete context management flow"""
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        handler = OpenAIHandler(basic_config)

        # Test that context management works end-to-end
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]

        # Test preparation
        prepared = handler._prepare_messages(messages)
        assert isinstance(prepared, list)
        assert len(prepared) > 0

    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'})
    def test_chat_completion_context_length_error(self, basic_config, temp_log_dir):
        """Test handling of context length exceeded error"""
        logger = get_logger()
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test.log'
        }
        logger.configure(log_config)

        handler = OpenAIHandler(basic_config)

        # Mock the client to raise a BadRequestError
        handler.client.chat.completions.create.side_effect = MockBadRequestError(
            "maximum context length",
            response=Mock(),
            body=None
        )

        with pytest.raises(MockBadRequestError):
            handler.chat_completion("Test prompt")

    def test_config_module_functionality(self):
        """Test that the real config module works correctly"""
        class TestObj:
            pass

        obj = TestObj()
        config = {
            'test_attr': 'test_value',
            'number_attr': 42,
            'bool_attr': True
        }

        # Use the real config function
        set_attributes_from_config(obj, config, required_attrs=['test_attr'])

        assert obj.test_attr == 'test_value'
        assert obj.number_attr == 42
        assert obj.bool_attr is True

    def test_logger_module_functionality(self, temp_log_dir):
        """Test that the real logger module works correctly"""
        logger = get_logger()

        # Configure with test settings
        log_config = {
            'level': 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': False,
            'base_dir': temp_log_dir,
            'filename': 'test_logger.log'
        }
        logger.configure(log_config)

        # Test logging methods
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        logger.error("Test error message")

        # Check that log file was created
        log_file = os.path.join(temp_log_dir, 'test_logger.log')
        assert os.path.exists(log_file)

        # Check that log file contains our messages
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert "Test info message" in log_content
            assert "Test debug message" in log_content


# Test runner that allows running with python command
if __name__ == "__main__":
    import sys

    # Try to import pytest
    try:
        import pytest
        # Run pytest with current file
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        print("pytest not installed. Please install with: pip install pytest")
        sys.exit(1)
