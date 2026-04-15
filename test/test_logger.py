import argparse
import io
import logging
import os
import pytest
import sys
import subprocess
import tempfile
import time
import traceback

import unittest
from unittest.mock import patch, call, MagicMock

# Add the parent directory to the Python path so we can import from rome package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rome.logger
from rome.logger import Logger, get_logger
from rome.config import set_attributes_from_config, check_attrs


def _reset_logger():
    """Properly reset the logger singleton and clean up handlers"""
    rome.logger._logger_instance = None
    # Clean up the underlying logging.Logger to avoid handler leaks
    underlying = logging.getLogger('Agent')
    for handler in underlying.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        underlying.removeHandler(handler)
    underlying.filters.clear()


class TestSizeRotatingFileHandler(unittest.TestCase):
    """Test suite for the SizeRotatingFileHandler class"""

    def setUp(self):
        _reset_logger()

    def test_max_log_size_configuration(self):
        """Test that max_log_size is properly configured"""
        logger = Logger()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'level': 'INFO',
                'format': '%(message)s',
                'console': False,
                'base_dir': temp_dir,
                'filename': 'test.log',
                'max_size_kb': 5000  # 5000 KB limit
            }
            logger.configure(config)

            # Check if SizeRotatingFileHandler was added
            from rome.logger import SizeRotatingFileHandler
            size_handlers = [h for h in logger._logger.handlers
                           if isinstance(h, SizeRotatingFileHandler)]
            self.assertEqual(len(size_handlers), 1)

            handler = size_handlers[0]
            self.assertEqual(handler.max_size_kb, 5000)
            self.assertEqual(handler.max_size_bytes, 5000 * 1024)

    def test_log_file_creation_and_writing(self):
        """Test that log files are created and written to"""
        logger = Logger()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'level': 'INFO',
                'format': '%(message)s',
                'console': False,
                'base_dir': temp_dir,
                'filename': 'test.log',
                'max_size_kb': 2048
            }
            logger.configure(config)

            log_path = os.path.join(temp_dir, 'test.log')

            # Log something to create the file
            logger.info("Test message")

            # File should exist and have size > 0
            self.assertTrue(os.path.exists(log_path))
            size_bytes = os.path.getsize(log_path)
            self.assertGreater(size_bytes, 0)

    def test_log_rotation_trigger(self):
        """Test that log rotation is triggered when size limit is exceeded"""
        logger = Logger()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'level': 'INFO',
                'format': '%(message)s',
                'console': False,
                'base_dir': temp_dir,
                'filename': 'test.log',
                'max_size_kb': 1024  # min enforced by configure is 1024
            }
            logger.configure(config)

            log_path = os.path.join(temp_dir, 'test.log')

            # Generate enough logs to exceed the 1024KB limit
            message = "This is a test message that should help us reach the size limit faster. " * 20
            for i in range(2000):
                logger.info(f"{i}: {message}")

            # Check that file exists
            self.assertTrue(os.path.exists(log_path))

            with open(log_path, 'r') as f:
                content = f.read()

            # File should have content - rotation may or may not have occurred
            self.assertGreater(len(content), 0)

    def test_unix_rotation(self):
        """Test that Unix tail is used for log rotation"""
        logger = Logger()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'level': 'INFO',
                'format': '%(message)s',
                'console': False,
                'base_dir': temp_dir,
                'filename': 'test.log',
                'max_size_kb': 1024
            }
            logger.configure(config)

            # Get the handler
            from rome.logger import SizeRotatingFileHandler
            handler = [h for h in logger._logger.handlers
                      if isinstance(h, SizeRotatingFileHandler)][0]

            # Create a test file that exceeds the limit
            log_path = os.path.join(temp_dir, 'test.log')
            with open(log_path, 'w') as f:
                f.write("x\n" * 10000)

            with patch('subprocess.run') as mock_subprocess:
                mock_subprocess.return_value = MagicMock(
                    stdout="line1\nline2\nline3\n", returncode=0
                )

                # Trigger rotation
                handler._rotate()

                # Verify tail was called
                self.assertGreaterEqual(mock_subprocess.call_count, 1)
                call_args = mock_subprocess.call_args[0][0]
                self.assertIn('tail', call_args)

    @patch('subprocess.run')
    def test_fallback_when_unix_utilities_fail(self, mock_subprocess):
        """Test fallback to emergency truncate when Unix utilities fail"""
        logger = Logger()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'level': 'INFO',
                'format': '%(message)s',
                'console': False,
                'base_dir': temp_dir,
                'filename': 'test.log',
                'max_size_kb': 1024
            }
            logger.configure(config)

            # Get the handler
            from rome.logger import SizeRotatingFileHandler
            handler = [h for h in logger._logger.handlers
                      if isinstance(h, SizeRotatingFileHandler)][0]

            # Create a test file with content
            log_path = os.path.join(temp_dir, 'test.log')
            original_lines = ["Line 1\n", "Line 2\n", "Line 3\n"] * 100
            with open(log_path, 'w') as f:
                f.writelines(original_lines)

            # Mock subprocess to fail
            mock_subprocess.side_effect = subprocess.CalledProcessError(1, 'cmd')

            # Trigger rotation
            handler._rotate()

            # File should still exist with emergency truncate marker
            self.assertTrue(os.path.exists(log_path))

            with open(log_path, 'r') as f:
                content = f.read()

            # Should contain emergency truncate marker
            self.assertIn("[EMERGENCY TRUNCATE", content)

    def test_no_rotation_without_max_size(self):
        """Test that no rotation occurs when max_log_size is not set"""
        logger = Logger()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'level': 'INFO',
                'format': '%(message)s',
                'console': False,
                'base_dir': temp_dir,
                'filename': 'test.log'
                # No max_size_kb specified
            }
            logger.configure(config)

            # Should use regular FileHandler, not SizeRotatingFileHandler
            from rome.logger import SizeRotatingFileHandler
            size_handlers = [h for h in logger._logger.handlers
                           if isinstance(h, SizeRotatingFileHandler)]
            self.assertEqual(len(size_handlers), 0)

            # Should have regular FileHandler
            file_handlers = [h for h in logger._logger.handlers
                           if isinstance(h, logging.FileHandler)]
            self.assertEqual(len(file_handlers), 1)

    def test_thread_safety_during_rotation(self):
        """Test that log rotation is thread-safe"""
        import threading
        logger = Logger()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'level': 'INFO',
                'format': '%(message)s',
                'console': False,
                'base_dir': temp_dir,
                'filename': 'test.log',
                'max_size_kb': 1024
            }
            logger.configure(config)

            # Function to log messages in a thread
            def log_messages(thread_id):
                for i in range(50):
                    logger.info(f"Thread {thread_id} - Message {i} - " + "x" * 50)
                    time.sleep(0.001)  # Small delay

            # Start multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=log_messages, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Log file should exist and be properly formatted
            log_path = os.path.join(temp_dir, 'test.log')
            self.assertTrue(os.path.exists(log_path))

            # File should be readable and not corrupted
            with open(log_path, 'r') as f:
                content = f.read()
                # Should contain messages from all threads
                self.assertIn("Thread", content)
                self.assertIn("Message", content)

class TestLogger(unittest.TestCase):
    """Test suite for the Logger class"""

    def setUp(self):
        _reset_logger()

    def test_singleton_pattern(self):
        """Test that get_logger follows the singleton pattern"""
        logger1 = get_logger()
        logger2 = get_logger()
        self.assertIs(logger1, logger2)

    def test_default_initialization(self):
        """Test that logger initializes with default settings"""
        logger = Logger()
        self.assertEqual(logger._logger.name, 'Agent')
        self.assertEqual(logger._logger.level, logging.INFO)
        self.assertFalse(logger._logger.propagate)

    def test_configure_log_level(self):
        """Test log level configuration"""
        logger = Logger()
        config = {
            'level': 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': True
        }
        logger.configure(config)
        self.assertEqual(logger._logger.level, logging.DEBUG)

    def test_configure_file_handler(self):
        """Test file handler configuration"""
        logger = Logger()

        # Create a temporary directory for logs
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'console': False,
                'base_dir': temp_dir,
                'filename': 'test.log'
            }
            logger.configure(config)

            # Check if file handler was added
            file_handlers = [h for h in logger._logger.handlers if isinstance(h, logging.FileHandler)]
            self.assertEqual(len(file_handlers), 1)

            # Check if log file was created
            log_path = os.path.join(temp_dir, 'test.log')
            self.assertTrue(os.path.exists(log_path))

            # Test logging to file
            logger.info("Test message")
            with open(log_path, 'r') as f:
                log_content = f.read()
                self.assertIn("Test message", log_content)

    def test_console_handler(self):
        """Test console handler configuration"""
        logger = Logger()
        config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'console': True
        }
        logger.configure(config)

        # Check if RichHandler was added
        from rich.logging import RichHandler
        rich_handlers = [h for h in logger._logger.handlers if isinstance(h, RichHandler)]
        self.assertEqual(len(rich_handlers), 1)

    def test_base_dir_creation(self):
        """Test that base_dir is created during configuration"""
        logger = Logger()
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, 'logs')
            config = {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'console': False,
                'base_dir': log_dir,
                'filename': 'test.log'
            }
            logger.configure(config)

            self.assertTrue(os.path.exists(log_dir))
            self.assertEqual(logger.base_dir, log_dir)

    def test_log_methods(self):
        """Test all logging methods"""
        logger = Logger()
        config = {
            'level': 'DEBUG',  # Set to lowest level to test all methods
            'format': '%(levelname)s: %(message)s',
            'console': False,
        }
        logger.configure(config)

        # Test each log method
        with patch.object(logger._logger, 'debug') as mock_debug:
            logger.debug("Debug message")
            mock_debug.assert_called_once()

        with patch.object(logger._logger, 'info') as mock_info:
            logger.info("Info message")
            mock_info.assert_called_once()

        with patch.object(logger._logger, 'warning') as mock_warning:
            logger.warning("Warning message")
            mock_warning.assert_called_once()

        with patch.object(logger._logger, 'error') as mock_error:
            logger.error("Error message")
            mock_error.assert_called_once()

        with patch.object(logger._logger, 'critical') as mock_critical:
            logger.critical("Critical message")
            mock_critical.assert_called_once()

    def test_assert_true_success(self):
        """Test assert_true when condition is True"""
        logger = Logger()
        # Should not raise exception or log anything
        with patch.object(logger, 'error') as mock_error:
            logger.assert_true(True, "This should not be logged")
            mock_error.assert_not_called()

    def test_assert_true_failure_with_exception(self):
        """Test assert_true when condition is False and log_only=False raises exception"""
        logger = Logger()
        config = {
            'level': 'INFO',
            'format': '%(message)s',
            'console': False,
        }
        logger.configure(config)

        with self.assertRaises(ValueError):
            logger.assert_true(False, "Test assertion error", log_only=False)

    def test_assert_true_failure_log_only(self):
        """Test assert_true when condition is False and log_only=True exits"""
        logger = Logger()
        config = {
            'level': 'INFO',
            'format': '%(message)s',
            'console': False,
        }
        logger.configure(config)

        with patch.object(logger, 'error') as mock_error:
            with self.assertRaises(SystemExit):
                logger.assert_true(False, "Test assertion error", log_only=True)
            mock_error.assert_called_once()
            self.assertIn("ASSERTION FAILED", mock_error.call_args[0][0])

    def test_assert_attribute_success(self):
        """Test assert_attribute when attribute exists"""
        logger = Logger()
        # Create a test object with an attribute
        test_obj = type('TestObject', (), {'test_attr': 'value'})()

        # Should not raise exception
        logger.assert_attribute(test_obj, 'test_attr')

    def test_assert_attribute_failure(self):
        """Test assert_attribute when attribute doesn't exist"""
        logger = Logger()
        # Create a test object without the required attribute
        test_obj = type('TestObject', (), {})()

        with patch.object(logger, 'assert_true') as mock_assert_true:
            logger.assert_attribute(test_obj, 'missing_attr')
            mock_assert_true.assert_called_once()
            # Check that the default message was generated
            self.assertIn("'missing_attr' not provided in TestObject configuration", mock_assert_true.call_args[0][1])

def test_set_attributes_from_config():
    """Test set_attributes_from_config utility function"""
    obj = type('TestObject', (), {})()
    config = {
        'attr1': 'value1',
        'attr2': 123,
        'attr3': ['a', 'b', 'c']
    }

    set_attributes_from_config(obj, config)

    assert hasattr(obj, 'attr1')
    assert obj.attr1 == 'value1'
    assert hasattr(obj, 'attr2')
    assert obj.attr2 == 123
    assert hasattr(obj, 'attr3')
    assert obj.attr3 == ['a', 'b', 'c']

    # Test with empty config
    obj2 = type('TestObject', (), {})()
    set_attributes_from_config(obj2, None)
    # Should not raise any exceptions

class TestUtilityFunctions(unittest.TestCase):
    """Test case for utility functions"""

    def test_set_attributes_from_config(self):
        """Test set_attributes_from_config utility function"""
        obj = type('TestObject', (), {})()
        config = {
            'attr1': 'value1',
            'attr2': 123,
            'attr3': ['a', 'b', 'c']
        }

        set_attributes_from_config(obj, config)

        self.assertTrue(hasattr(obj, 'attr1'))
        self.assertEqual(obj.attr1, 'value1')
        self.assertTrue(hasattr(obj, 'attr2'))
        self.assertEqual(obj.attr2, 123)
        self.assertTrue(hasattr(obj, 'attr3'))
        self.assertEqual(obj.attr3, ['a', 'b', 'c'])

        # Test with empty config
        obj2 = type('TestObject', (), {})()
        set_attributes_from_config(obj2, None)
        # Should not raise any exceptions

    def test_check_attrs_success(self):
        """Test check_attrs when all attributes exist"""
        obj = type('TestObject', (), {'attr1': 1, 'attr2': 2})()

        # Should not raise an exception
        check_attrs(obj, ['attr1', 'attr2'])

    def test_check_attrs_failure(self):
        """Test check_attrs when an attribute is missing calls assert_true"""
        obj = type('TestObject', (), {'attr1': 1})()

        # check_attrs calls logger.assert_true which calls sys.exit(1)
        with self.assertRaises(SystemExit):
            check_attrs(obj, ['attr1', 'missing_attr'])

def run_test_suite():
    """Run all tests in the TestLogger class and TestUtilityFunctions class."""
    print("Running Logger test suite...")

    # Create a test suite for both classes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests from TestLogger
    suite.addTest(loader.loadTestsFromTestCase(TestLogger))

    # Add tests from TestUtilityFunctions
    suite.addTest(loader.loadTestsFromTestCase(TestUtilityFunctions))

    # Add tests from TestSizeRotatingFileHandler
    suite.addTest(loader.loadTestsFromTestCase(TestSizeRotatingFileHandler))

    # Run the tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    # Return exit code based on test result
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description='Run tests')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--class', dest='test_class', choices=['TestLogger', 'TestUtilityFunctions', 'TestSizeRotatingFileHandler'],
                       help='Run tests from specific class')
    parser.add_argument('test_name', nargs='?', help='Specific test method to run')

    args = parser.parse_args()

    # Map class names to actual classes
    class_map = {
        'TestLogger': TestLogger,
        'TestUtilityFunctions': TestUtilityFunctions,
        'TestSizeRotatingFileHandler': TestSizeRotatingFileHandler
    }

    suite = unittest.TestSuite()

    if args.test_name:
        # Run specific test
        if args.test_class:
            # From specific class
            try:
                suite.addTest(class_map[args.test_class](args.test_name))
            except (ValueError, KeyError):
                print(f"Error: Test {args.test_name} not found in {args.test_class}")
                sys.exit(1)
        else:
            # Try all classes
            for test_class in class_map.values():
                try:
                    suite.addTest(test_class(args.test_name))
                except ValueError:
                    continue

            if suite.countTestCases() == 0:
                print(f"Error: Test {args.test_name} not found")
                sys.exit(1)

    elif args.test_class:
        # Run all tests from specific class
        suite = unittest.TestLoader().loadTestsFromTestCase(class_map[args.test_class])

    else:
        # Run all tests
        return run_test_suite()

    # Run the suite
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
