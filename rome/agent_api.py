import logging
from typing import Optional, Dict, Any
import threading

from .logger import get_logger
from .config import set_attributes_from_config


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

        self.app = FastAPI(title="Agent API", description="API to expose agent state.", version="1.2")
        self._add_routes()

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

    def _add_routes(self):
        """Add API routes to the FastAPI app"""
        from fastapi.responses import JSONResponse
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
