"""
Process management utility with automatic child process cleanup and signal handling.

Provides a decorator that adds robust process management to any class:
- Automatic discovery and cleanup of all child processes
- Signal handling (SIGTERM, SIGINT) with graceful shutdown
- Preserves existing shutdown() methods while adding process management
- Zero changes required to existing code

Usage:
    @process_managed
    class MyClass:
        def shutdown(self):
            # Your existing cleanup code
            pass

    # Both manual and automatic shutdown work:
    obj = MyClass()
    obj.shutdown()  # Manual shutdown
    # SIGTERM/SIGINT also trigger automatic shutdown

Dependencies:
    pip install psutil
"""
import signal
import sys
import os
import threading
from typing import List, Callable, Optional
import psutil


class ProcessManager:
    """Automatic process management with signal handling"""

    def __init__(self, name: str = "ProcessManager", timeout: int = 10):
        self.name = name
        self.timeout = timeout
        self.cleanup_callbacks: List[Callable] = []
        self.shutdown_called = False
        self._lock = threading.Lock()
        self.parent_pid = os.getpid()

    def add_cleanup_callback(self, callback: Callable) -> None:
        """Add callback to run during cleanup"""
        if callback and callable(callback):
            self.cleanup_callbacks.append(callback)

    def setup_signal_handlers(self, logger=None) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            if logger:
                logger.info(f"{self.name} received signal {signum}, initiating shutdown")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def get_child_processes(self) -> List[psutil.Process]:
        """Discover all child processes automatically"""
        try:
            parent = psutil.Process(self.parent_pid)
            return parent.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return []

    def terminate_process(self, process: psutil.Process, timeout: Optional[int] = None) -> bool:
        """Terminate a process with escalating force"""
        if timeout is None:
            timeout = self.timeout

        try:
            if not process.is_running():
                return True
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return True

        try:
            # Escalating termination: terminate -> kill
            process.terminate()
            try:
                process.wait(timeout=min(5, timeout))
                return True
            except psutil.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=timeout)
                    return True
                except psutil.TimeoutExpired:
                    return False
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return True
        except Exception:
            return False

    def shutdown_all_processes(self, timeout: Optional[int] = None) -> None:
        """Shutdown all child processes automatically"""
        if timeout is None:
            timeout = self.timeout

        children = self.get_child_processes()
        if not children:
            return

        # Terminate all children with escalating force
        for child in children:
            try:
                self.terminate_process(child, timeout)
            except Exception:
                # Continue with other processes even if one fails
                continue

    def shutdown(self) -> None:
        """Complete shutdown with callbacks and process cleanup"""
        with self._lock:
            if self.shutdown_called:
                return
            self.shutdown_called = True

        # Run cleanup callbacks first (user's shutdown logic)
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception:
                # Silent failure for callbacks to prevent cascade failures
                pass

        # Shutdown all child processes
        self.shutdown_all_processes()


def process_managed(timeout: int = 10, auto_signal_handling: bool = True):
    """
    Decorator to add automatic process management to any class.

    Args:
        timeout: Timeout in seconds for process termination (default: 10)
        auto_signal_handling: Whether to setup SIGTERM/SIGINT handlers (default: True)

    Usage:
        @process_managed
        class MyClass:
            def shutdown(self):
                # Your cleanup code here
                pass

        # Or with configuration:
        @process_managed(timeout=30, auto_signal_handling=True)
        class MyClass:
            pass

    The decorator preserves existing shutdown() methods and enhances them with
    automatic process cleanup. Both manual and signal-triggered shutdown work.
    """

    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # Initialize the original class first
            original_init(self, *args, **kwargs)

            # Add process manager with custom timeout
            manager_name = f"{cls.__name__}_{getattr(self, 'name', 'unknown')}"
            self.process_manager = ProcessManager(name=manager_name, timeout=timeout)

            # Preserve original shutdown method if it exists
            self._original_shutdown = None
            if hasattr(self, 'shutdown') and callable(self.shutdown):
                self._original_shutdown = self.shutdown
                # Add it as a cleanup callback
                self.process_manager.add_cleanup_callback(self._original_shutdown)

            # Replace shutdown with enhanced version
            self.shutdown = self._enhanced_shutdown

            # Setup signal handlers if requested
            if auto_signal_handling:
                logger = getattr(self, 'logger', None)
                self.process_manager.setup_signal_handlers(logger)

        def _enhanced_shutdown(self):
            """Enhanced shutdown that handles both processes and user cleanup"""
            self.process_manager.shutdown()

        # Add methods to the class
        cls._enhanced_shutdown = _enhanced_shutdown
        cls.__init__ = new_init
        return cls

    # Support both @process_managed and @process_managed() syntax
    if callable(timeout):
        # Called as @process_managed (no parentheses)
        cls = timeout
        timeout = 10
        return decorator(cls)
    else:
        # Called as @process_managed() or @process_managed(timeout=5)
        return decorator


# Convenience function for manual process management
def cleanup_child_processes(timeout: int = 10) -> None:
    """
    Standalone function to cleanup all child processes.

    Args:
        timeout: Timeout in seconds for process termination

    Usage:
        # Manual cleanup anywhere in your code
        cleanup_child_processes()
    """
    manager = ProcessManager("manual_cleanup", timeout)
    manager.shutdown_all_processes()
