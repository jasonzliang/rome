# Modified logger.py - Update the get_logger function and Logger class

import logging
import sys
import threading
import tempfile
import os
import subprocess
import pytz
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler

# Global bootstrap logger instance
_bootstrap_logger = None
_bootstrap_lock = threading.Lock()

# Global instances
_main_logger_instance = None
_bootstrap_logger_instance = None


class SizeRotatingFileHandler(logging.FileHandler):
    """Deadlock-safe size-rotating file handler with cross-platform support"""

    def __init__(self, filename, max_size_kb, keep_ratio=0.7, *args, **kwargs):
        self.max_size_kb = max_size_kb
        self.max_size_bytes = max_size_kb * 1024
        self.keep_ratio = max(0.1, min(0.9, keep_ratio))
        self._lock = threading.RLock()  # Use RLock for reentrant locking
        self._rotating = False  # Track rotation state
        super().__init__(filename, *args, **kwargs)

    def emit(self, record):
        """Emit record with deadlock-safe rotation"""
        with self._lock:
            try:
                # Check if rotation needed (but don't rotate during rotation)
                if (not self._rotating and
                    self.max_size_bytes and
                    os.path.exists(self.baseFilename) and
                    os.path.getsize(self.baseFilename) >= self.max_size_bytes):
                    self._rotate()

                super().emit(record)

            except Exception:
                # Handle errors without triggering more logging
                try:
                    import sys
                    sys.stderr.write(f"Logging error: {record.getMessage()}\n")
                except:
                    pass

    def _rotate(self):
        """Rotate log efficiently with Unix tools fallback to Python"""
        try:
            self._rotating = True  # Set rotation flag
            self.stream and self.stream.close()
            self.stream = None

            if not self._unix_rotate():
                self._python_rotate()

            self.stream = self._open()
        except Exception as e:
            # Emergency: ensure we have a working stream
            self.stream = self.stream or self._open()
        finally:
            self._rotating = False  # Clear rotation flag

    def _unix_rotate(self) -> bool:
        """Fast Unix-based rotation with timeout protection"""
        if os.name != 'posix':
            return False
        try:
            lines = max(int(self.max_size_bytes * self.keep_ratio) // 80, 50)
            fd, tmp = tempfile.mkstemp(suffix='.log')
            with os.fdopen(fd, 'w') as f:
                f.write(f"[ROTATED - {self.max_size_kb}KB limit]\n")
                f.write(subprocess.run(['tail', '-n', str(lines), self.baseFilename],
                                     capture_output=True, text=True, check=True,
                                     timeout=5).stdout)  # Reduced timeout
            subprocess.run(['mv', tmp, self.baseFilename], check=True,
                          timeout=2)  # Reduced timeout
            return True
        except:
            try: os.unlink(tmp)
            except: pass
            return False

    def _python_rotate(self):
        """Cross-platform Python rotation"""
        try:
            size = os.path.getsize(self.baseFilename)
            keep_size = int(size * self.keep_ratio)

            with open(self.baseFilename, 'rb') as f:
                f.seek(max(0, size - keep_size))
                size > keep_size and f.readline()  # Skip partial line
                data = f.read()

            with open(self.baseFilename, 'wb') as f:
                f.write(f"[ROTATED - {self.max_size_kb}KB limit]\n".encode())
                f.write(data)
        except Exception:
            # Emergency truncation
            with open(self.baseFilename, 'w') as f:
                f.write(f"[TRUNCATED - {self.max_size_kb}KB limit]\n")


class ParentPathRichHandler(RichHandler):
    """Custom RichHandler that shows the path of the parent function (1 level above caller)."""

    def __init__(self, *args, **kwargs):
        kwargs['show_path'] = False  # Disable default path
        super().__init__(*args, **kwargs)

    def emit(self, record):
        """Add parent path info before emitting."""
        # Get the current call stack
        import inspect
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
    """Thread-safe logger with Rich console output and caller information"""

    def __init__(self, logger_name='Agent'):
        """Initialize logger with specified name"""
        self.logger_name = logger_name
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._configured = False
        self._lock = threading.Lock()

    def is_configured(self) -> bool:
        """Check if the logger has been properly configured"""
        return (self._configured and
                self._logger is not None and
                len(self._logger.handlers) > 0)

    def configure_simple(self):
        """Configure as a simple bootstrap logger"""
        self.configure({})

    def configure(self, log_config: dict):
        """Configure logger with provided settings, sets up console and file handlers."""
        with self._lock:
            # Clear existing handlers to avoid duplicates
            for handler in self._logger.handlers[:]:
                self._logger.removeHandler(handler)

            # Clear existing filters to avoid duplicates
            for filter_obj in self._logger.filters[:]:
                self._logger.removeFilter(filter_obj)

            # Set attributes from config - we need to import this function
            # For now, manually set attributes
            if log_config:
                for key, value in log_config.items():
                    setattr(self, key, value)

            # Handle missing config gracefully with defaults
            self.level = getattr(self, 'level', 'INFO')
            self.format = getattr(self,
                'format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.console = getattr(self, 'console', True)
            self.include_caller_info = getattr(self, 'include_caller_info', None)
            self.base_dir = getattr(self, 'base_dir', None)
            self.filename = getattr(self, 'filename', None)
            self.max_size_kb = getattr(self, 'max_size_kb', None)
            self.timezone = getattr(self, 'timezone', 'US/Pacific')

            try:
                # Set log level
                level = getattr(logging, self.level.upper())
                self._logger.setLevel(level)

                # Set up timezone filter
                if hasattr(self, 'timezone') and self.timezone:
                    pacific_tz = pytz.timezone(self.timezone)
                    class PacificTimezoneFilter(logging.Filter):
                        def filter(self, record):
                            dt = datetime.fromtimestamp(record.created, tz=pacific_tz)
                            record.created = dt.timestamp()
                            return True
                    pacific_filter = PacificTimezoneFilter()
                    self._logger.addFilter(pacific_filter)

                # Create formatter
                formatter = logging.Formatter(self.format, datefmt="%H:%M:%S")

                # Add file handler if base_dir and filename are specified
                if self.base_dir:
                    os.makedirs(self.base_dir, exist_ok=True)

                    if self.filename:
                        log_file_path = os.path.join(self.base_dir, self.filename)

                        if self.max_size_kb:
                            file_handler = SizeRotatingFileHandler(
                                log_file_path,
                                max_size_kb=max(self.max_size_kb, 1024),
                                mode='a'
                            )
                        else:
                            file_handler = logging.FileHandler(log_file_path, mode='a')

                        file_handler.setFormatter(formatter)
                        file_handler.setLevel(level)
                        self._logger.addHandler(file_handler)

                # Add Rich console handler if enabled
                if self.console:
                    self._add_console_handler(level, formatter)

                # Mark as configured
                self._configured = True

                # Log successful configuration
                self.info("Main logger configured successfully")

            except Exception as e:
                # Emergency fallback - create basic working logger
                print(f"Logger configuration failed: {e}")
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(levelname)s: %(message)s')
                handler.setFormatter(formatter)
                handler.setLevel(logging.DEBUG)
                self._logger.addHandler(handler)
                self._configured = True

    def _add_console_handler(self, level, formatter):
        """Add appropriate console handler based on caller_info setting"""
        console = Console()
        caller_info = getattr(self, 'include_caller_info', None)

        # Validate caller_info value safely
        if caller_info not in ["rome", "rich", None]:
            print(f"Warning: Invalid caller_info '{caller_info}', using None")
            caller_info = None

        try:
            if caller_info == "rome":
                rich_handler = ParentPathRichHandler(
                    console=console,
                    show_time=True,
                    show_path=False,
                    show_level=True,
                    rich_tracebacks=True,
                    markup=True,
                    log_time_format="[%H:%M:%S]"
                )
            elif caller_info == "rich":
                rich_handler = RichHandler(
                    console=console,
                    show_time=True,
                    show_path=True,
                    show_level=True,
                    rich_tracebacks=True,
                    markup=True,
                    log_time_format="[%H:%M:%S]"
                )
            else:
                rich_handler = RichHandler(
                    console=console,
                    show_time=True,
                    show_path=False,
                    show_level=True,
                    rich_tracebacks=True,
                    markup=True,
                    log_time_format="[%H:%M:%S]"
                )

            rich_handler.setLevel(level)
            self._logger.addHandler(rich_handler)
            return rich_handler

        except Exception as e:
            # Fallback to basic console handler
            print(f"Failed to create RichHandler: {e}")
            basic_handler = logging.StreamHandler()
            basic_handler.setFormatter(formatter)
            basic_handler.setLevel(level)
            self._logger.addHandler(basic_handler)
            return basic_handler

    # Standard logging methods
    def info(self, message: str):
        self._logger.info(message)

    def warning(self, message: str):
        self._logger.warning(message)

    def error(self, message: str):
        self._logger.error(message)

    def debug(self, message: str):
        self._logger.debug(message)

    def critical(self, message: str):
        self._logger.critical(message)

    def assert_true(self, condition: bool, message: str,
                    exception_type = ValueError,
                    log_only: bool = True) -> None:
        """Assert condition is true, log error with traceback if false."""
        if condition:
            return

        if log_only:
            try:
                self.error(f"ASSERTION FAILED: {message}")
            except:
                print(f"ASSERTION FAILED: {message}")
            exit(1)
        else:
            raise exception_type(message)

    def assert_attribute(self, obj, attr_name: str,
                         message: str = None,
                         exception_type = ValueError) -> None:
        """Assert object has specified attribute"""
        if message is None:
            obj_name = obj.__class__.__name__
            message = f"'{attr_name}' not provided in {obj_name} configuration, did you merge your config with default config?"

        self.assert_true(hasattr(obj, attr_name), message, exception_type)


def get_logger():
    """
    Get the appropriate logger instance:
    - Returns bootstrap logger if main logger not configured
    - Returns main logger if it has been configured
    - Automatically transitions when main logger becomes available
    """
    global _main_logger_instance, _bootstrap_logger_instance

    # Create main logger instance if it doesn't exist (singleton pattern)
    if _main_logger_instance is None:
        _main_logger_instance = Logger('Agent')  # Main logger uses 'Agent' name

    # Check if main logger is configured
    if _main_logger_instance.is_configured():
        # Log transition message only once
        if _bootstrap_logger_instance is not None:
            _main_logger_instance.info("Transitioned from bootstrap to main logger")
            _bootstrap_logger_instance = None  # Clear bootstrap reference
        return _main_logger_instance
    else:
        # Create bootstrap logger only once
        if _bootstrap_logger_instance is None:
            _bootstrap_logger_instance = Logger('bootstrap')  # Bootstrap uses 'bootstrap' name
            _bootstrap_logger_instance.configure_simple()      # Configure with simple settings
        return _bootstrap_logger_instance
