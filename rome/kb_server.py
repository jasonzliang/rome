# chroma_server.py
import os
import sys
import time
import subprocess
import signal
import threading
import requests
import atexit
from typing import Optional, Callable, List, Dict

import chromadb

from .config import set_attributes_from_config
from .logger import get_logger

TIMEOUT_LEN = 2

class ChromaServerManager:
    """Independent ChromaDB server lifecycle manager with thread safety"""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, config: Dict = None):
        """Get singleton instance of ChromaServerManager"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)"""
        with cls._lock:
            if cls._instance:
                cls._instance.force_stop()
            cls._instance = None

    def __init__(self, config: Dict = None):
        """Initialize ChromaDB server manager"""
        if ChromaServerManager._instance is not None:
            raise RuntimeError("ChromaServerManager is a singleton. Use get_instance() instead.")

        self.config = config or {}
        self.logger = get_logger()

        # Set attributes from config
        set_attributes_from_config(self, self.config,
            ['host', 'port', 'persist_path'],
            ['startup_timeout', 'shutdown_timeout'])

        if not self.startup_timeout:
            self.startup_timeout = TIMEOUT_LEN * 2
        if not self.shutdown_timeout:
            self.shutdown_timeout = TIMEOUT_LEN

        # Handle persist_path intelligently
        self.persist_path = self._resolve_persist_path()

        self.server_process = None
        self.server_url = f"http://{self.host}:{self.port}"

        self._clients = set()
        self._clients_lock = threading.Lock()  # New lock for client operations

        self._shutdown_registered = False

        # Register cleanup on exit
        self._register_cleanup()

        self.logger.debug(f"ChromaServerManager initialized for {self.server_url}")
        self.logger.debug(f"Data will persist to: {self.persist_path}")

        while not self.is_running():
            self.logger.debug("Waiting for server to start up...")
            time.sleep(1)

    def _resolve_persist_path(self) -> str:
        """Resolve the best persist path for ChromaDB data"""
        import tempfile
        import platform

        # If explicitly set in config, use that
        if self.persist_path:
            return os.path.expanduser(self.persist_path)

        # Create agent-specific directory
        base_dir = os.path.expanduser("~/.rome")
        agent_data_dir = os.path.join(base_dir, "agent-chroma-db")

        # Fallback to temp if home directory issues
        try:
            os.makedirs(agent_data_dir, exist_ok=True)
            # Test write access
            test_file = os.path.join(agent_data_dir, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return agent_data_dir
        except (OSError, PermissionError):
            # Fallback to session-specific temp directory
            session_dir = os.path.join(tempfile.gettempdir(), f"agent-chroma-db-{os.getpid()}")
            self.logger.warning(f"Could not use user data directory, falling back to: {session_dir}")
            return session_dir

    def _register_cleanup(self):
        """Register cleanup handlers"""
        if not self._shutdown_registered and hasattr(self, '_started_by_me'):
            atexit.register(self.stop)
            self._shutdown_registered = True

    def is_running(self) -> bool:
        """Check if ChromaDB server is running using v2 API"""
        try:
            # Try v2 heartbeat first
            response = requests.get(f"{self.server_url}/api/v2/heartbeat", timeout=TIMEOUT_LEN)
            if response.status_code == 200:
                return True
        except:
            pass

        try:
            # Fallback: try getting version info
            response = requests.get(f"{self.server_url}/api/v2/version", timeout=TIMEOUT_LEN)
            if response.status_code == 200:
                return True
        except:
            pass

        try:
            # Last resort: try to list collections
            response = requests.get(f"{self.server_url}/api/v2/collections", timeout=TIMEOUT_LEN)
            return response.status_code in [200, 404]  # 404 is ok if no collections exist
        except:
            return False

    def start(self, force_restart: bool = False) -> bool:
        """
        Start ChromaDB server

        Args:
            force_restart: If True, stop existing server before starting

        Returns:
            True if server started successfully, False otherwise
        """
        if force_restart and self.is_running():
            self.logger.info("Force restart requested, stopping existing server")
            self.stop()
            time.sleep(self.shutdown_timeout)  # Give server time to fully stop

        if self.is_running():
            self.logger.info(f"ChromaDB server already running at {self.server_url}")
            return True

        self.logger.info(f"Starting ChromaDB server at {self.server_url}...")

        # Ensure persist directory exists
        os.makedirs(self.persist_path, exist_ok=True)

        # Try different command variations for ChromaDB
        commands = [
            ["chroma", "run", "--host", self.host, "--port", str(self.port),
                "--path", self.persist_path],
        ]

        for cmd in commands:
            if self._try_start_command(cmd):
                return True

        self.logger.error("All ChromaDB server start attempts failed")
        self.logger.info(f"Try manually: chroma run --host {self.host} --port {self.port} --path {self.persist_path}")
        return False

    def _try_start_command(self, cmd: list) -> bool:
        """Try to start server with a specific command"""
        try:
            self.logger.debug(f"Trying command: {' '.join(cmd)}")

            # Start process
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )

            # Wait for server to be ready
            for i in range(self.startup_timeout):
                time.sleep(1)

                if self.is_running():
                    self.logger.info(f"ChromaDB server started successfully (took {i+1}s)")
                    return True

                # Check if process died
                if self.server_process.poll() is not None:
                    stdout, stderr = self.server_process.communicate()
                    self.logger.debug(f"Command failed: {stderr.decode().strip()}")
                    self.server_process = None
                    return False

            # Timeout - kill process and try next command
            self.logger.debug(f"Command timed out after {self.startup_timeout}s")
            self._kill_process()
            return False

        except FileNotFoundError:
            self.logger.debug(f"Command not found: {cmd[0]}")
            return False
        except Exception as e:
            self.logger.debug(f"Error with command {cmd[0]}: {e}")
            self._kill_process()
            return False

    def _kill_process(self):
        """Forcefully kill the server process"""
        if self.server_process:
            try:
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                else:
                    self.server_process.terminate()

                try:
                    self.server_process.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    if hasattr(os, 'killpg'):
                        os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                    else:
                        self.server_process.kill()
            except Exception as e:
                self.logger.debug(f"Error killing process: {e}")
            finally:
                self.server_process = None

    def stop(self, force: bool = False) -> bool:
        """Stop ChromaDB server with improved reliability and thread safety"""
        with self._clients_lock:
            active_clients = len(self._clients)

        if not force and active_clients > 0:
            self.logger.warning(f"Cannot stop server: {active_clients} active clients remain")
            return False

        # Get all possible PIDs to stop
        pids = self._get_target_pids()
        if not pids:
            self.logger.info("No ChromaDB processes found to stop")
            return True

        # Stop all found processes
        success = all(self._stop_process(pid, force) for pid in pids)
        self.server_process = None

        # Verify server stopped
        return self._verify_stopped() if success else False

    def _get_target_pids(self) -> List[int]:
        """Get all PIDs that need to be stopped"""
        pids = []

        # Add managed process PID
        if self.server_process and self.server_process.poll() is None:
            pids.append(self.server_process.pid)

        # Add external process PID
        external_pid = self._find_external_pid()
        if external_pid and external_pid not in pids:
            pids.append(external_pid)

        return pids

    def _stop_process(self, pid: int, force: bool) -> bool:
        """Stop a single process by PID"""
        try:
            self.logger.info(f"Stopping ChromaDB process (PID: {pid})")

            if force:
                return self._kill_process_by_pid(pid, signal.SIGKILL, timeout=2)

            # Try graceful then force
            return (self._kill_process_by_pid(pid, signal.SIGTERM, self.shutdown_timeout) or
                   self._kill_process_by_pid(pid, signal.SIGKILL, timeout=2))

        except (OSError, ProcessLookupError):
            return True  # Already dead
        except Exception as e:
            self.logger.error(f"Error stopping process {pid}: {e}")
            return False

    def _kill_process_by_pid(self, pid: int, sig: int, timeout: int) -> bool:
        """Kill process with signal and wait for termination"""
        try:
            # Try psutil first (most reliable)
            try:
                import psutil
                proc = psutil.Process(pid)
                if sig == signal.SIGKILL:
                    proc.kill()
                else:
                    proc.terminate()
                proc.wait(timeout=timeout)
                return True
            except ImportError:
                pass

            # Fallback to os.kill
            os.kill(pid, sig)

            # Wait for process death
            for _ in range(timeout * 2):  # Check every 0.5s
                try:
                    os.kill(pid, 0)  # Check if alive
                    time.sleep(0.5)
                except OSError:
                    return True  # Process died

            return False  # Timeout

        except (OSError, ProcessLookupError):
            return True  # Already dead

    def _find_external_pid(self) -> Optional[int]:
        """Find external ChromaDB process PID using multiple methods"""
        finders: List[Callable[[], Optional[int]]] = [
            lambda: self._run_cmd(['lsof', '-ti', f':{self.port}']),
            lambda: self._parse_netstat(),
            lambda: self._check_psutil_connections()
        ]

        for finder in finders:
            try:
                pid = finder()
                if pid:
                    return pid
            except Exception:
                continue
        return None

    def _run_cmd(self, cmd: List[str]) -> Optional[int]:
        """Run command and extract first PID from output"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip().split('\n')[0])
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        return None

    def _parse_netstat(self) -> Optional[int]:
        """Extract PID from netstat output"""
        try:
            result = subprocess.run(['netstat', '-tlnp'], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return None

            for line in result.stdout.split('\n'):
                if f':{self.port} ' in line and 'LISTEN' in line:
                    for part in line.split():
                        if '/' in part:
                            try:
                                return int(part.split('/')[0])
                            except ValueError:
                                continue
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _check_psutil_connections(self) -> Optional[int]:
        """Find PID using psutil network connections"""
        try:
            import psutil
            for conn in psutil.net_connections(kind='inet'):
                if (conn.status == psutil.CONN_LISTEN and
                    conn.laddr.port == self.port):
                    return conn.pid
        except ImportError:
            pass
        return None

    def _verify_stopped(self) -> bool:
        """Verify server actually stopped"""
        for _ in range(10):  # Check for 5 seconds
            if not self.is_running():
                self.logger.info("ChromaDB server stopped successfully")
                return True
            time.sleep(0.5)

        self.logger.warning("Server appears to still be running after stop attempt")
        return False

    def force_stop(self) -> bool:
        """Force stop the server regardless of active clients"""
        return self.stop(force=True)

    def restart(self) -> bool:
        """Restart the ChromaDB server"""
        self.logger.info("Restarting ChromaDB server...")
        self.stop(force=True)
        time.sleep(self.shutdown_timeout)  # Give server time to fully stop
        return self.start()

    def get_client(self) -> chromadb.HttpClient:
        """
        Get ChromaDB client and register it as active (thread-safe)

        Returns:
            chromadb.HttpClient instance

        Raises:
            RuntimeError: If server is not running
        """
        if not self.is_running():
            raise RuntimeError(f"ChromaDB server not running at {self.server_url}")

        client = chromadb.HttpClient(host=self.host, port=self.port)

        with self._clients_lock:
            self._clients.add(client)
            client_count = len(self._clients)

        self.logger.debug(f"Created ChromaDB client ({client_count} active)")
        return client

    def release_client(self, client: chromadb.HttpClient):
        """Release a client (remove from active clients tracking) - thread-safe"""
        with self._clients_lock:
            self._clients.discard(client)
            client_count = len(self._clients)

        self.logger.debug(f"Released ChromaDB client ({client_count} active)")

    def get_status(self) -> Dict:
        """Get detailed server status with external PID detection (thread-safe)"""
        external_pid = None
        if self.is_running() and not self.server_process:
            external_pid = self._find_external_pid()

        with self._clients_lock:
            active_clients = len(self._clients)

        return {
            "running": self.is_running(),
            "url": self.server_url,
            "host": self.host,
            "port": self.port,
            "persist_path": self.persist_path,
            "process_id": self.server_process.pid if self.server_process else external_pid,
            "managed_externally": self.server_process is not None,
            "active_clients": active_clients,
            "startup_timeout": self.startup_timeout,
            "shutdown_timeout": self.shutdown_timeout
        }

    def get_database_info(self) -> Dict:
        """Get information about database location and collections"""
        try:
            if self.is_running():
                client = chromadb.HttpClient(host=self.host, port=self.port)
                collections = client.list_collections()
                collection_info = [{"name": c.name, "count": c.count()} for c in collections]
            else:
                collection_info = []

            return {
                "persist_path": self.persist_path,
                "server_url": self.server_url,
                "running": self.is_running(),
                "collections": collection_info,
                "total_collections": len(collection_info)
            }
        except Exception as e:
            return {
                "persist_path": self.persist_path,
                "server_url": self.server_url,
                "running": self.is_running(),
                "error": str(e)
            }

    def health_check(self) -> Dict:
        """Perform health check and return detailed status"""
        status = self.get_status()

        if status["running"]:
            try:
                # Try to create a test client
                test_client = chromadb.HttpClient(host=self.host, port=self.port)

                # Test basic operations
                collections = test_client.list_collections()
                status.update({
                    "health": "healthy",
                    "collections_count": len(collections),
                    "error": None
                })
            except Exception as e:
                status.update({
                    "health": "unhealthy",
                    "error": str(e)
                })
        else:
            status.update({
                "health": "not_running",
                "error": "Server is not running"
            })

        return status

    def clear_database(self, force: bool = False) -> bool:
        """
        Clear all data from the ChromaDB database

        Args:
            force: If True, stop server first to ensure clean deletion

        Returns:
            True if database was cleared successfully
        """
        import shutil

        was_running = self.is_running()

        try:
            # Stop server if running (database files are locked while server runs)
            if was_running:
                if not force:
                    self.logger.error("Cannot clear database while server is running. Use force=True to stop server first.")
                    return False

                self.logger.info("Stopping server to clear database...")
                if not self.stop(force=True):
                    self.logger.error("Failed to stop server for database clearing")
                    return False

                # Give server time to fully shutdown
                time.sleep(self.shutdown_timeout)

            # Clear the database directory
            if os.path.exists(self.persist_path):
                self.logger.info(f"Clearing database at: {self.persist_path}")
                shutil.rmtree(self.persist_path)
                self.logger.info("Database cleared successfully")
            else:
                self.logger.info("Database directory doesn't exist - nothing to clear")

            # Restart server if it was running before
            if was_running:
                self.logger.info("Restarting server after database clear...")
                return self.start()

            return True

        except Exception as e:
            self.logger.error(f"Failed to clear database: {e}")

            # Try to restart server if it was running
            if was_running:
                try:
                    self.start()
                except:
                    self.logger.error("Failed to restart server after failed database clear")

            return False

    def clear_collection(self, collection_name: str) -> bool:
        """
        Clear a specific collection from the database

        Args:
            collection_name: Name of collection to clear

        Returns:
            True if collection was cleared successfully
        """
        try:
            if not self.is_running():
                self.logger.error("Server must be running to clear collections")
                return False

            client = chromadb.HttpClient(host=self.host, port=self.port)

            # Check if collection exists
            try:
                collection = client.get_collection(collection_name)
                # Delete all documents in the collection
                collection.delete()  # This deletes all documents
                self.logger.info(f"Cleared collection '{collection_name}'")
                return True
            except Exception:
                # Collection doesn't exist
                self.logger.info(f"Collection '{collection_name}' doesn't exist - nothing to clear")
                return True

        except Exception as e:
            self.logger.error(f"Failed to clear collection '{collection_name}': {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """
        Completely delete a collection from the database

        Args:
            collection_name: Name of collection to delete

        Returns:
            True if collection was deleted successfully
        """
        try:
            if not self.is_running():
                self.logger.error("Server must be running to delete collections")
                return False

            client = chromadb.HttpClient(host=self.host, port=self.port)

            try:
                client.delete_collection(collection_name)
                self.logger.info(f"Deleted collection '{collection_name}'")
                return True
            except Exception:
                self.logger.info(f"Collection '{collection_name}' doesn't exist - nothing to delete")
                return True

        except Exception as e:
            self.logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if hasattr(self, 'server_process') and self.server_process:
                self.force_stop()
        except:
            pass  # Ignore errors during cleanup
