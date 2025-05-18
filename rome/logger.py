import logging
import threading
from typing import Optional, Dict
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

# Needed to prevent circular imports
def set_attributes_from_config(obj, config):
    for key, value in config.items():
        setattr(obj, key, value)

class Logger:
    """Thread-safe singleton logger with Rich console output"""
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
        """Configure the logger with provided configuration"""
        with self._lock:
            # Clear existing handlers to avoid duplicates
            for handler in self._logger.handlers[:]:
                self._logger.removeHandler(handler)

            # Set attributes from config
            set_attributes_from_config(self, log_config)

            # Validate required attributes with a more compact assertion
            required_attrs = ['level', 'format', 'console']
            for attr in required_attrs:
                assert hasattr(self, attr), f"{attr} not provided in Logger config"

            # Set log level
            level = getattr(logging, self.level.upper())
            self._logger.setLevel(level)

            # Create formatter
            formatter = logging.Formatter(self.format)

            # Add file handler if specified
            if hasattr(self, 'file') and self.file:
                file_handler = logging.FileHandler(self.file, mode='a')
                file_handler.setFormatter(formatter)
                file_handler.setLevel(level)
                self._logger.addHandler(file_handler)

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

    def info(self, message: str):
        """Log info message"""
        self._logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self._logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self._logger.error(message)

    def debug(self, message: str):
        """Log debug message"""
        self._logger.debug(message)

    def critical(self, message: str):
        """Log critical message"""
        self._logger.critical(message)

# Global instance and convenience functions
_logger_instance = None

def get_logger() -> Logger:
    """Get the singleton logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger()
    return _logger_instance
