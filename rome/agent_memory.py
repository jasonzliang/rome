"""Agent Memory - Intelligent memory layer using Mem0 with ChromaDB + Neo4j backend"""
import os
import time
import traceback
from typing import Dict, Optional

from mem0 import Memory

from .config import set_attributes_from_config, DEFAULT_CONFIG, LONGER_SUMMARY_LEN
from .logger import get_logger
from .kb_server import ChromaServerManager, CHROMA_BASE_DIR
from .kb_client import EMBEDDING_MODELS

OPENAI_CONFIG =  {
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 1337,
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "openai_base_url": "https://api.openai.com/v1"
}


class AgentMemory:
    """Intelligent memory layer for agents using Mem0 with optional graph storage"""

    def __init__(self, agent, config: Dict = None):
        """Initialize agent memory with vector +/- graph storage"""
        self.agent = agent
        self.mem_id = f"{os.path.basename(agent.get_repo())}_{agent.get_id()}"
        self.logger = get_logger()

        # Apply config (defaults come from DEFAULT_CONFIG)
        set_attributes_from_config(
            self, config, DEFAULT_CONFIG['AgentMemory'].keys(), ['openai_config']
        )

        self.memory = None
        if self.enabled:
            self._initialize_mem0()
            self.clear()

    # def _test_neo4j_connection(self) -> bool:
    #     """Test if Neo4j is accessible"""
    #     if not self.use_graph or not self.memory:
    #         return False

    #     try:
    #         # Try to access the graph store directly
    #         if hasattr(self.memory, 'graph'):
    #             # Run a simple test query
    #             test_query = "MATCH (n) RETURN count(n) as total LIMIT 1"
    #             result = self.memory.graph.graph.query(test_query)
    #             self.logger.info(f"Neo4j connection test successful: {result}")
    #             return True
    #     except Exception as e:
    #         self.logger.error(f"Neo4j connection test failed: {e}")
    #         self.logger.error(traceback.format_exc())
    #         return False

    def _initialize_mem0(self) -> None:
        """Initialize Mem0 with ChromaDB vector store and optional Neo4j graph store"""
        try:
            # Validate embedding model
            if self.embedding_model not in EMBEDDING_MODELS:
                raise RuntimeError(f"Invalid embedding model {self.embedding_model}")

            if "OPENROUTER_API_KEY" in os.environ:
                self.logger.debug("Removing OPENROUTER_API_KEY to prevent auto-routing")
                del os.environ["OPENROUTER_API_KEY"]

            # Build base config
            openai_config = OPENAI_CONFIG | self.openai_config if self.openai_config else OPENAI_CONFIG
            llm_config = {
                "provider": "openai",
                "config": openai_config
            }
            mem0_config = {
                "version": "v1.1",
                "llm": llm_config
            }

            # Embedder configuration with explicit dimensions
            embedder_config = {
                "provider": "openai",
                "config": {
                    "model": self.embedding_model,
                    "embedding_dims": EMBEDDING_MODELS[self.embedding_model]
                }
            }
            mem0_config["embedder"] = embedder_config

            # ALWAYS configure vector store (required for both modes)
            server_config = {
                'host': self.vector_host,
                'port': self.vector_port,
                'persist_path': None
            }

            server_manager = ChromaServerManager.get_instance(config=server_config)
            if not server_manager.is_running() and not server_manager.start():
                raise RuntimeError("Failed to start ChromaDB server")

            collection_name = f"mem0_{self.mem_id}"
            mem0_config["vector_store"] = {
                "provider": "chroma",
                "config": {
                    "collection_name": collection_name,
                    "host": server_manager.host,
                    "port": server_manager.port
                }
            }

            # Optionally ADD graph store (works alongside vector store)
            if self.use_graph:
                if not self.graph_password:
                    self.logger.error("Graph enabled but no password provided")
                    return

                mem0_config["graph_store"] = {
                    "provider": "neo4j",
                    "config": {
                        "url": self.graph_url,
                        "username": self.graph_username,
                        "password": self.graph_password,
                        "llm": llm_config
                    }
                }
                self.logger.info(f"Mem0 initialized: ChromaDB @ {server_manager.server_url} + Neo4j @ {self.graph_url}")
            else:
                self.logger.info(f"Mem0 initialized: ChromaDB @ {server_manager.server_url}")

            # Initialize memory with the complete configuration
            self.memory = Memory.from_config(config_dict=mem0_config)

        except Exception as e:
            self.logger.error(f"Memory initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            self.enabled = False; self.memory = None

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
                user_id=self.mem_id,
                metadata=meta,
                infer=False
            )

            if not result:
                self.logger.error("No result returned from memory.add()")
                return False

            results = result.get('results', [])
            relations = result.get('relations', [])

            if results:
                self.logger.info(
                    f"Vector DB: {len(results)} memories | graph DB: {len(relations)} relations")
                if not relations and self.use_graph:
                    self.logger.debug("No graph relations extracted")
                self.logger.debug(f"Remembered memory: {result}")
                return True
            else:
                self.logger.error("No memories or graph relations extracted")
                return False

        except Exception as e:
            self.logger.error(f"Failed to remember: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def recall(self, query: str, context: str = None) -> str:
        """
        Query memory and return most relevant results
        Uses vector similarity + graph traversal (if enabled)
        """
        if not self.is_enabled():
            self.logger.error("Memory not enabled, skipping recall")
            return ""

        try:
            filters = {"context": context} if context else None
            results = self.memory.search(
                query=query,
                user_id=self.mem_id,
                limit=self.recall_limit,
                filters=filters,
            )

            if results and results.get('results'):
                recalled_mem =  '\n\n'.join(r['memory'] for r in results['results'])
                self.logger.debug(f"Recalled: {recalled_mem}...")
                return recalled_mem
            else:
                self.logger.error(f"No memories recalled, results empty")
                return ""

        except Exception as e:
            self.logger.error(f"Failed to recall: {e}")
            self.logger.error(traceback.format_exc())
            return ""

    def should_remember(self, prompt: str, response: str) -> bool:
        """Heuristic check if interaction should be remembered"""
        if len(prompt) + len(response) < self.auto_remember_len:
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

    def chat_completion(self, prompt: str, system_message: str = None,
                       override_config: Dict = None, response_format: Dict = None) -> str:
        """Enhanced chat completion with automatic memory integration"""
        system_message = system_message or self.agent.role
        # Get memory context if enabled
        if self.is_enabled() and self.auto_recall:
            # Use prompt as query for recall
            memory_context = self.recall(prompt[:LONGEST_SUMMARY_LEN])

            if memory_context:
                # Inject memory into system message
                system_message = f"{system_message}\n\n[Relevant Memory Context]\n{memory_context}"
                self.logger.debug(f"Injected memory: {len(memory_context)} chars")

        # Call original chat completion
        response = self.agent.openai_handler.chat_completion(
            prompt=prompt,
            system_message=system_message,
            override_config=override_config,
            response_format=response_format,
        )

        # Auto-remember after getting response
        if self.is_enabled() and self.auto_remember:
            if self.should_remember(prompt, response):
                summary = \
                    f"Q: {prompt[:LONGEST_SUMMARY_LEN]}... A: {response[:LONGEST_SUMMARY_LEN]}..."
                self.remember(summary, context="interaction")

        return response

    def clear(self) -> bool:
        """Clear all memories from vector store and Neo4j graph"""
        if not self.is_enabled():
            return False

        try:
            # Delete from both vector and graph stores
            self.memory.delete_all(user_id=self.mem_id)

            # Verify vector store cleanup
            results = self.memory.get_all(user_id=self.mem_id, limit=10000)
            vector_count = len(results.get('results', [])) if results else 0

            # Verify Neo4j cleanup if graph enabled
            graph_count = 0
            if self.use_graph and hasattr(self.memory, 'graph'):
                query = "MATCH (n {user_id: $user_id}) RETURN count(n) as total"
                result = self.memory.graph.graph.query(query, {"user_id": self.mem_id})
                graph_count = result[0]['total'] if result else 0

            if vector_count == 0 and graph_count == 0:
                self.logger.info(f"Cleared all memories for {self.mem_id} for both vector/graph")
                return True
            else:
                self.logger.error(f"Clear incomplete: vector={vector_count}, graph={graph_count} memories remain")
                return False

        except Exception as e:
            self.logger.error(f"Failed to clear memories: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def shutdown(self) -> None:
        """Clean up memory resources"""
        if self.memory:
            # self.clear()
            self.logger.debug(f"Memory resources released for {self.mem_id}")
