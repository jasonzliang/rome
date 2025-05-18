import logging
import threading
import os
from typing import Optional, Dict
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

# Don't import from config, needed to prevent circular imports
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

            # Initialize the base_dir and filename attributes
            self.base_dir = None
            self.filename = None

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
        """Get the current log directory"""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        return self.base_dir

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
