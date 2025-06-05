import logging
import socket
from typing import Optional, Dict, Any
import threading

from .logger import get_logger
from .config import API_PORT_RANGE, set_attributes_from_config


class AgentApi:
    def __init__(self, agent: Any, config: Dict = None):
        """
        Initialize the Agent API with configuration

        Args:
            agent: The agent instance (type annotation is Any to avoid circular imports)
            config: Configuration dictionary
        """
        from fastapi import FastAPI
        import uvicorn

        self.agent = agent
        self.config = config or {}

        # Initialize logger
        self.logger = get_logger()

        # Set attributes from config if provided
        set_attributes_from_config(self, self.config, ['host', 'port'])

        self._setup_port()
        self._setup_fastapi()

        # Disable Uvicorn's default loggers
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

        # Configure Uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",  # Minimize Uvicorn's own logging
            access_log=False,     # Disable access logs entirely
        )

        self.server = uvicorn.Server(config)
        self.server_thread: Optional[threading.Thread] = None

        self.logger.info(f"AgentApi initialized")

    def _setup_port(self):
        """Check for a valid port to use"""
        def _is_port_in_use(port: int) -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0

        min_port, max_port = API_PORT_RANGE
        self.logger.assert_true(min_port <= self.port <= max_port,
            f"Ports must be within {min_port}-{max_port} range")

        new_port = self.port
        while _is_port_in_use(new_port):
            new_port += 1
            if new_port > max_port:
                raise ValueError("Cannot find a valid unused port for API server")

        if new_port != self.port:
            self.logger.error(f"Port {self.port} is used already, using {new_port} instead")
            self.port = new_port

    def _setup_fastapi(self):
        """Add API routes to the FastAPI app"""
        from fastapi.responses import JSONResponse

        self.app = FastAPI(title="Agent API", description="API to expose agent state.", version="1.2")
        @self.app.get("/agent", response_class=JSONResponse)
        async def get_state():
            try:
                fsm = self.agent.fsm
                return {
                    "name": self.agent.name,
                    "role": self.agent.role,
                    "context": getattr(self.agent, "context", {}),
                    "fsm": {
                        "current_state": fsm.current_state,
                        "current_action": fsm.current_action,
                        "graph": fsm.get_graph()
                    }
                }
            except Exception as e:
                self.logger.error(f"Error in get_state endpoint: {str(e)}")
                return JSONResponse(content={"error": str(e)}, status_code=500)

    def run(self):
        """Run FastAPI in a background thread (non-blocking)."""
        self.logger.info(f"Starting AgentApi server on {self.host}:{self.port}")

        def run_server():
            """Internal method to run the server loop."""
            self.server.run()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.logger.info("AgentApi server started in background thread")

    def shutdown(self):
        """Gracefully shutdown the Uvicorn server."""
        self.logger.info("Shutting down AgentApi server")
        self.server.should_exit = True  # Signal server to stop
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join()   # Wait for server to stop
        self.logger.info("AgentApi server shutdown complete")
