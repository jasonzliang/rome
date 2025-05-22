import logging
import threading
import os
import inspect
import traceback
from typing import Optional, Dict, Any, Type, Callable
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text


def check_attrs(obj, required_attrs):
    """Helper function to check if required attributes have been set"""
    for attr in required_attrs:
        assert hasattr(obj, attr), f"{attr} not provided in {obj.__class__.__name__} config"


def check_opt_attrs(obj, optional_attrs):
    """Helper function to check if optional attributes have been set"""
    for attr in optional_attrs:
        if not hasattr(obj, attr):
            setattr(obj, attr, None), f"{attr} (optional) not provided in {obj.__class__.__name__} config"


def set_attributes_from_config(obj, config=None, required_attrs=None, optional_attrs=None):
    """Helper function to convert configuration dictionary entries to object attributes"""
    if config:
        for key, value in config.items():
            setattr(obj, key, value)


class ParentPathRichHandler(RichHandler):
    """Custom RichHandler that shows the path of the parent function (1 level above caller)."""

    def __init__(self, *args, **kwargs):
        kwargs['show_path'] = False  # Disable default path
        super().__init__(*args, **kwargs)

    def emit(self, record):
        """Add parent path info before emitting."""
        # Get the current call stack
        stack = inspect.stack()

        # Find the parent caller (1 level above the logging call)
        parent_info = None

        # Look through stack to find user code frames
        user_frames = []
        for frame in stack:
            frame_file = frame.filename

            # Skip logging internals and this handler file
            if ('__init__.py' not in frame_file and
                'logging' not in frame_file and
                frame.function not in ['emit', 'handle', 'callHandlers', '_log']):
                user_frames.append({
                    'filename': os.path.basename(frame_file),
                    'lineno': frame.lineno
                })

        # If we have at least 2 user frames, the second one is the parent
        if len(user_frames) >= 2:
            parent_info = user_frames[1]  # Parent of the logging caller

        # Add parent path to the message (file:line format only)
        if parent_info:
            parent_path = f"\\[{parent_info['filename']}:{parent_info['lineno']}]"
            record.msg = f"{parent_path} {record.msg}"

        super().emit(record)


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

    def configure(self, log_config: Dict):
        """Configure logger with provided settings, sets up console and file handlers."""
        with self._lock:
            # Clear existing handlers to avoid duplicates
            for handler in self._logger.handlers[:]:
                self._logger.removeHandler(handler)

            # Set attributes from config
            set_attributes_from_config(self, log_config)
            check_attrs(self, ['level', 'format', 'console'])
            check_opt_attrs(self, ['include_caller_info', 'base_dir', 'filename'])

            # Set log level
            level = getattr(logging, self.level.upper())
            self._logger.setLevel(level)

            # Create formatter
            formatter = logging.Formatter(self.format)

            # Add file handler if base_dir and filename are specified
            if self.base_dir:
                # Create log directory if it doesn't exist
                os.makedirs(self.base_dir, exist_ok=True)

                if self.filename:
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

                # Determine handler type based on include_caller_info setting
                caller_info = self.include_caller_info

                if caller_info == "rome":
                    # Use custom handler that shows parent caller path
                    rich_handler = ParentPathRichHandler(
                        console=console,
                        show_time=True,
                        show_path=False,  # We handle path info ourselves
                        show_level=True,
                        rich_tracebacks=True,
                        markup=True,
                        log_time_format="[%H:%M:%S]"
                    )
                elif caller_info == "rich" or caller_info == True:
                    # Use standard RichHandler with built-in path display
                    rich_handler = RichHandler(
                        console=console,
                        show_time=True,
                        show_path=True,  # Show Rich's default path info
                        show_level=True,
                        rich_tracebacks=True,
                        markup=True,
                        log_time_format="[%H:%M:%S]"
                    )
                else:
                    # include_caller_info is None or other values - no caller info
                    rich_handler = RichHandler(
                        console=console,
                        show_time=True,
                        show_path=False,  # No path info
                        show_level=True,
                        rich_tracebacks=True,
                        markup=True,
                        log_time_format="[%H:%M:%S]"
                    )

                rich_handler.setLevel(level)
                self._logger.addHandler(rich_handler)

    def get_log_dir(self) -> Optional[str]:
        """Get the current log directory, creating it if needed."""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        return self.base_dir

    def info(self, message: str):
        """Log info message with caller information"""
        self._logger.info(message)

    def warning(self, message: str):
        """Log warning message with caller information"""
        self._logger.warning(message)

    def error(self, message: str):
        """Log error message with caller information"""
        self._logger.error(message)

    def debug(self, message: str):
        """Log debug message with caller information"""
        self._logger.debug(message)

    def critical(self, message: str):
        """Log critical message with caller information"""
        self._logger.critical(message)

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
        self.error(f"{message}\nStack trace:\n{stack_trace}")

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


# Example usage
if __name__ == "__main__":
    pass