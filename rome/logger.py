import logging
import threading
from typing import Optional, Dict
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

class SingletonLogger:
    """Thread-safe singleton logger with Rich console output"""
    _instance: Optional['SingletonLogger'] = None
    _lock = threading.Lock()
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SingletonLogger, cls).__new__(cls)
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

            # Set log level
            level = getattr(logging, log_config.get('level', 'INFO').upper())
            self._logger.setLevel(level)

            # Create formatter
            formatter = logging.Formatter(
                log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )

            # Add file handler if specified
            if log_config.get('file'):
                file_handler = logging.FileHandler(log_config['file'], mode='a')
                file_handler.setFormatter(formatter)
                file_handler.setLevel(level)
                self._logger.addHandler(file_handler)

            # Add Rich console handler if enabled (default: True)
            if log_config.get('console', True):
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

def get_logger() -> SingletonLogger:
    """Get the singleton logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SingletonLogger()
    return _logger_instance

# Example usage
if __name__ == "__main__":
    # Configure the logger
    logger = get_logger()
    logger.configure({
        'level': 'DEBUG',
        'console': True,
        'format': '%(message)s'  # Rich handles the formatting
    })

    # Test all log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
