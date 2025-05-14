# logger.py
import logging
import threading
from typing import Optional, Dict


class SingletonLogger:
    """Thread-safe singleton logger that can be configured once and accessed from any class"""

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

            # Add console handler if enabled (default: True)
            if log_config.get('console', True):
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                console_handler.setLevel(level)
                self._logger.addHandler(console_handler)

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
