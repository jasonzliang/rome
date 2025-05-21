import logging
import threading
import os
import inspect
import traceback
from typing import Optional, Dict, Any, Type, Callable
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

# Don't import from config, needed to prevent circular imports
def set_attributes_from_config(obj, config):
    if config:
        for key, value in config.items():
            setattr(obj, key, value)

def check_attrs(obj, attrs):
    for attr in attrs:
        assert hasattr(obj, attr), f"{attr} not provided in {obj.__class__.__name__} config"

class Logger:
    """Thread-safe singleton logger with Rich console output and caller information"""
    _instance: Optional['Logger'] = None
    _lock = threading.Lock()
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once per instance
        if self._logger is not None:
            return
        with self._lock:
            if self._logger is not None:
                return
            # Create logger with default settings
            self._logger = logging.getLogger('Agent')
            self._logger.setLevel(logging.INFO)
            # Prevent propagation to avoid duplicate logs
            self._logger.propagate = False

            # Initialize the base_dir and filename attributes
            self.base_dir = None
            self.filename = None
            self.include_caller_info = True  # Default to including caller info

    def configure(self, log_config: Dict):
        """Configure logger with provided settings, sets up console and file handlers."""
        with self._lock:
            # Clear existing handlers to avoid duplicates
            for handler in self._logger.handlers[:]:
                self._logger.removeHandler(handler)

            # Set attributes from config
            set_attributes_from_config(self, log_config)
            check_attrs(self, ['level', 'format', 'console'])

            # Set log level
            level = getattr(logging, self.level.upper())
            self._logger.setLevel(level)

            # Ensure 'include_caller_info' is set
            if not hasattr(self, 'include_caller_info'):
                self.include_caller_info = False

            # Create formatter
            formatter = logging.Formatter(self.format)

            # Add file handler if base_dir and filename are specified
            if hasattr(self, 'base_dir') and self.base_dir:
                # Create log directory if it doesn't exist
                os.makedirs(self.base_dir, exist_ok=True)

                if hasattr(self, 'filename') and self.filename:
                    # Construct full log file path
                    log_file_path = os.path.join(self.base_dir, self.filename)

                    file_handler = logging.FileHandler(log_file_path, mode='a')
                    file_handler.setFormatter(formatter)
                    file_handler.setLevel(level)
                    self._logger.addHandler(file_handler)
                    self.info(f"Logging to file: {log_file_path}")

            # Add Rich console handler if enabled
            if self.console:
                console = Console()
                rich_handler = RichHandler(
                    console=console,
                    show_time=True,
                    show_path=False,
                    rich_tracebacks=True,
                    markup=True
                )
                rich_handler.setLevel(level)
                self._logger.addHandler(rich_handler)

    def get_log_dir(self) -> Optional[str]:
        """Get the current log directory, creating it if needed."""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        return self.base_dir

    def _get_caller_info(self):
        """Get information about the caller of the logging method."""
        if not self.include_caller_info:
            return ""

        # Get the stack frame of the caller
        frame = inspect.currentframe()
        try:
            # Go back 3 frames to get the actual caller (1: _get_caller_info, 2: log method, 3: actual caller)
            frame = inspect.getouterframes(frame)[3]
            file_path = os.path.abspath(frame.filename)
            file_name = os.path.basename(file_path)
            line_num = frame.lineno
            function_name = frame.function
            return f"[{file_name}:{line_num} in {function_name}]"
        except (IndexError, AttributeError):
            return "[unknown caller]"
        finally:
            # Always delete frame references to avoid reference cycles
            del frame

    def info(self, message: str):
        """Log info message with caller information"""
        caller_info = self._get_caller_info()
        self._logger.info(f"{caller_info} {message}" if caller_info else message)

    def warning(self, message: str):
        """Log warning message with caller information"""
        caller_info = self._get_caller_info()
        self._logger.warning(f"{caller_info} {message}" if caller_info else message)

    def error(self, message: str):
        """Log error message with caller information"""
        caller_info = self._get_caller_info()
        self._logger.error(f"{caller_info} {message}" if caller_info else message)

    def debug(self, message: str):
        """Log debug message with caller information"""
        caller_info = self._get_caller_info()
        self._logger.debug(f"{caller_info} {message}" if caller_info else message)

    def critical(self, message: str):
        """Log critical message with caller information"""
        caller_info = self._get_caller_info()
        self._logger.critical(f"{caller_info} {message}" if caller_info else message)

    def assert_true(self, condition: bool, message: str,
                    exception_type: Type[Exception] = ValueError,
                    log_only: bool = True) -> None:
        """
        Assert condition is true, log error with traceback if false.

        Args:
            condition: The condition to check
            message: Error message to log if condition is False
            exception_type: Exception type to pass to the caller
            log_only: If True, logs the error and exits program without raising exception
        """
        if condition:
            return

        # Condition failed - capture stack trace
        stack = traceback.extract_stack()[:-1]  # Exclude this function from trace
        stack_trace = ''.join(traceback.format_list(stack))

        # Log the error with stack trace
        caller_info = self._get_caller_info()
        error_msg = f"{caller_info} {message}" if caller_info else message
        self.error(f"{error_msg}\nStack trace:\n{stack_trace}")

        # If we're just logging, don't raise an exception but exit program
        if log_only:
            self.critical(f"Exiting program due to assertion failure: {message}")
            exit(1)  # Exit with error code 1

        # Raise the exception
        try:
            raise exception_type(message)
        except exception_type as e:
            # Re-raise with the original message but without a new traceback
            # The caller will see the exception but Python won't print a duplicate stack trace
            raise exception_type(str(e)) from None

    def assert_attribute(self, obj: Any, attr_name: str,
                         message: str = None,
                         exception_type: Type[Exception] = ValueError) -> None:
        """Assert object has specified attribute, generates default message if none provided."""
        if message is None:
            obj_name = obj.__class__.__name__
            message = f"'{attr_name}' not provided in {obj_name} configuration"

        self.assert_true(hasattr(obj, attr_name), message, exception_type)

    def assert_condition(self, condition: Callable[[], bool],
                      message: str,
                      exception_type: Type[Exception] = ValueError) -> None:
        """Assert callable returns True, useful for lazy evaluation of complex conditions."""
        self.assert_true(condition(), message, exception_type)

# Global instance and convenience functions
_logger_instance = None

def get_logger() -> Logger:
    """Get the singleton logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger()
    return _logger_instance
