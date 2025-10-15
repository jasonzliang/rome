"""Agent Memory - Intelligent memory layer using Mem0 with ChromaDB +/- Graph backend"""
import os
import time
import traceback
from typing import Dict, Optional

from mem0 import Memory

from .config import set_attributes_from_config, LONGER_SUMMARY_LEN
from .logger import get_logger
from .kb_server import ChromaServerManager, CHROMA_BASE_DIR

class AgentMemory:
    """Intelligent memory layer for agents using Mem0 with optional graph storage"""

    def __init__(self, agent_name: str, repo_name: str, config: Dict = None):
        """Initialize agent memory with vector +/- graph storage"""

        self.agent_name = f"{repo_name}_{agent_name}"
        self.logger = get_logger()

        # Apply config (defaults come from DEFAULT_CONFIG)
        set_attributes_from_config(
            self, config,
            ['enabled', 'auto_inject', 'auto_remember', 'remember_threshold',
                'embedding_model', 'recall_limit', 'use_graph', 'graph_url',
                'graph_username', 'graph_password', 'chroma_host', 'chroma_port']
        )

        self.memory = None
        if self.enabled:
            self._initialize_mem0()
            self.clear()

    def _initialize_mem0(self) -> None:
        """Initialize Mem0 with ChromaDB vector store and optional graph store"""
        try:
            # Build base config
            mem0_config = {
                "version": "v1.1",
                "llm": {
                    "provider": "openai",
                    "config": {"model": "gpt-5-mini", "temperature": 0.1, "max_tokens": 4000}
                }
            }

            # Add embedder if specified
            if self.embedding_model:
                mem0_config["embedder"] = {
                    "provider": "openai",
                    "config": {"model": self.embedding_model}
                }

            # Configure vector store (server-based unless using graph)
            if self.use_graph:
                # File-based storage for graph mode

                # Add graph store
                if not self.graph_password:
                    self.logger.error("Graph enabled but no password provided")

                provider = "neo4j" if "neo4j" in self.graph_url else "memgraph"
                mem0_config["graph_store"] = {
                    "provider": provider,
                    "config": {
                        "url": self.graph_url,
                        "username": self.graph_username,
                        "password": self.graph_password or ""
                    }
                }
                self.logger.info(f"Mem0 initialized: file-based ChromaDB + {provider} graph")
            else:
                # Server-based storage
                server_config = {
                    'host': self.chroma_host,
                    'port': self.chroma_port,
                    'persist_path': None
                }

                server_manager = ChromaServerManager.get_instance(config=server_config)
                if not server_manager.is_running() and not server_manager.start():
                    raise RuntimeError("Failed to start ChromaDB server")

                collection_name = f"mem0_{self.agent_name}"
                mem0_config["vector_store"] = {
                    "provider": "chroma",
                    "config": {
                        "collection_name": collection_name,
                        "host": server_manager.host,
                        "port": server_manager.port
                    }
                }
                self.logger.info(f"Mem0 initialized: ChromaDB server @ {server_manager.server_url}")

            self.memory = Memory.from_config(config_dict=mem0_config)

        except Exception as e:
            self.logger.error(f"Memory initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            self.enabled = False
            self.memory = None

    def is_enabled(self) -> bool:
        """Check if memory is enabled and initialized"""
        return self.enabled and self.memory is not None

    def remember(self, content: str, context: str = None, metadata: Dict = None) -> bool:
        """Store content in memory (vector + graph if enabled)"""
        if not self.is_enabled():
            self.logger.error("Memory not enabled, skipping remember")
            return False

        try:
            # Make content more "memorable" by adding personal/experiential context
            message = f"{context}: {content}" if context else content

            meta = metadata or {}
            if context: meta['context'] = context
            meta['timestamp'] = time.time()

            # Structure as conversation for entity extraction
            messages = [{"role": "user", "content": message}]
            result = self.memory.add(
                messages,
                user_id=self.agent_name,
                metadata=meta,
                infer=False  # Bypass LLM extraction - store directly!
            )

            self.logger.info(f"Memory add result: {result}")

            # Check if anything was actually stored
            if result and result.get('results'):
                self.logger.info(f"Successfully stored {len(result['results'])} memories")
                return True
            else:
                self.logger.error(f"No memories extracted from content (may not be memorable enough)")
                return False

        except Exception as e:
            self.logger.error(f"Failed to remember: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def recall(self, query: str, context: str = None) -> str:
        """
        Query memory and return most relevant results
        Uses vector similarity + graph traversal (if enabled)

        Args:
            query: Query for semantic similarity + graph search
            context: Optional context filter

        Returns:
            Concatenated relevant memories (empty string if none found)
        """
        if not self.is_enabled():
            self.logger.error("Memory not enabled, skipping recall")
            return ""

        try:
            filters = {"context": context} if context else None
            results = self.memory.search(
                query=query,
                user_id=self.agent_name,
                limit=self.recall_limit,
                filters=filters
            )

            if results and results.get('results'):
                recalled_mem =  '\n\n'.join(r['memory'] for r in results['results'])
                self.logger.debug(f"Recalled: {recalled_mem}...")
                return recalled_mem
            else:
                raise
        except Exception as e:
            self.logger.error(f"Failed to recall: {e}")
            return ""

    def should_remember(self, prompt: str, response: str) -> bool:
        """Heuristic check if interaction should be remembered"""
        if len(prompt) + len(response) < self.remember_threshold:
            return False

        memorable_patterns = [
            'decided', 'chose', 'learned', 'discovered', 'found', 'realized',
            'prefer', 'strategy', 'failed', 'succeeded', 'worked', "didn't work",
            'important', 'critical', 'key', 'pattern', 'should', 'avoid',
            'remember', 'note', 'always', 'never', 'visited', 'explored',
            'analyzed', 'edited', 'file', 'directory'
        ]

        combined = (prompt + ' ' + response).lower()
        return any(pattern in combined for pattern in memorable_patterns)

    def clear(self) -> bool:
        """Clear all memories and verify deletion"""
        if not self.is_enabled():
            return False

        try:
            # Delete and verify
            self.memory.delete_all(user_id=self.agent_name)
            results = self.memory.get_all(user_id=self.agent_name, limit=10000)
            count = len(results.get('results', [])) if results else 0

            if count == 0:
                self.logger.info(f"Cleared all memories for {self.agent_name}")
                return True

            self.logger.error(f"Clear failed: {count} memories remain")
            return False

        except Exception as e:
            self.logger.error(f"Failed to clear memories: {e}")
            return False

    def shutdown(self) -> None:
        """Clean up memory resources"""
        if self.memory:
            # self.clear()
            self.logger.debug(f"Memory resources released for {self.agent_name}")
