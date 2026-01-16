# knowledge_base.py
import json
import hashlib
from typing import Optional, List, Dict
import os
from pathlib import Path
import re
import time
import sys
import warnings
warnings.filterwarnings('ignore', message='.*validate_default.*', category=UserWarning)

try:
    from chromadb.utils.embedding_functions import (
       SentenceTransformerEmbeddingFunction, OpenAIEmbeddingFunction)
    from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.postprocessor.llm_rerank import LLMRerank
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.chroma import ChromaVectorStore
    # Needed for OpenAIEmbeddingFunction
    os.environ['CHROMA_OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']
except ImportError as e:
    print(f"Import error: {e}, install with: pip install chromadb llama-index llama-index-vector-stores-chroma llama-index-embeddings-openai llama-index-llms-openai")
    exit(1)

import openai

from .config import set_attributes_from_config, DEFAULT_CONFIG
from .logger import get_logger
from .kb_server import ChromaServerManager
from .parsing import parse_json_response

# Embedding model configurations
EMBEDDING_MODELS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "text-embedding-ada-001": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "paraphrase-MiniLM-L6-v2": 384
}


class ChromaClientManager:
    """Enhanced ChromaDB + LlamaIndex knowledge base with reranking"""

    def __init__(self, agent=None):
        self.agent = agent

        client_config = self.agent.config.get('ChromaClientManager', {})
        server_config = self.agent.config.get('ChromaServerManager', {})
        self.config = client_config or {}
        self.server_config = server_config or {}
        self.logger = get_logger()

        # Set attributes from config
        set_attributes_from_config(self, self.config,
            DEFAULT_CONFIG['ChromaClientManager'].keys())

        self.logger.assert_true(self.embedding_model in EMBEDDING_MODELS,
            f"Invalid embedding model: {self.embedding_model}")

        # Get or create server manager instance
        if self.use_shared_server:
            # Use shared singleton server
            self.server = ChromaServerManager.get_instance(server_config)
            self._owns_server = False
        else:
            # Create dedicated server instance
            self.server = ChromaServerManager(server_config)
            self._owns_server = True

        if not self.server.is_running():
            self.logger.info("ChromaDB server not running, starting it...")
            if not self.server.start():
                raise RuntimeError(f"Failed to start ChromaDB server at {self.server.server_url}")

        # Initialize ChromaDB client and collection
        self._setup_chroma_client()

        # Initialize LlamaIndex components
        self._setup_llamaindex()

        # Initialize reranker if enabled
        self._setup_reranker()

        self.logger.info(f"ChromaClientManager initialized: collection={self.collection_name}, reranking={self.enable_reranking}")

    def _path_to_collection_name(self, file_path: str, max_len: int = 128) -> str:
        """Convert file path to valid Chroma collection name."""
        name = re.sub(r'[^a-z0-9._-]', '_', Path(file_path).stem.lower())
        name = re.sub(r'(^[^a-z0-9]+|[^a-z0-9]+$|_{2,}|\.{2,})', '_', name).strip('_')
        return (name if len(name) >= 3 else f"doc_{name}".ljust(3, '0'))[:max_len]

    def _validate_dimensions(self, expected_dim):
        """Validate collection embedding dimensions"""
        if self.collection.count() == 0:
            return

        result = self.collection.get(limit=1, include=["embeddings"])

        # More defensive approach - handle various array types
        embeddings = result.get("embeddings")
        # Check for None first
        if embeddings is None:
            return

        # Convert to list if it's a numpy array or similar
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()

        # Now safely check length and content
        if len(embeddings) > 0 and embeddings[0] is not None:
            actual_dim = len(embeddings[0])
            if actual_dim != expected_dim:
                compatible = [m for m, d in EMBEDDING_MODELS.items() if d == actual_dim]
                raise ValueError(
                    f"Dimension mismatch: collection={actual_dim}d, model={expected_dim}d. "
                    f"Compatible models: {compatible} or clear collection."
                )
            else:
                self.logger.debug(f"Collection/model embedding dimension validated: {actual_dim}")

    def _create_collection(self):
        """Create collection with appropriate embedding function"""
        expected_dim = EMBEDDING_MODELS[self.embedding_model]
        is_sentence_transformer = expected_dim == 384

        if is_sentence_transformer:
            embedding_fn = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        else:
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError(f"OPENAI_API_KEY required for {self.embedding_model}")
            embedding_fn = OpenAIEmbeddingFunction(model_name=self.embedding_model)

        try:
            self.collection = self.client.create_collection(
                name=self.collection_name, embedding_function=embedding_fn)
            self.logger.debug(f"Created new collection: {self.collection_name} ({self.embedding_model} | {expected_dim}d)")

        except Exception as e:
            if "already exists" in str(e).lower():
                self.collection = self.client.get_collection(
                    name=self.collection_name, embedding_function=embedding_fn)
                self.logger.debug(f"Using existing collection: {self.collection_name} ({self.embedding_model} | {expected_dim}d | {self.collection.count()}n)")
            else: raise

        self._validate_dimensions(expected_dim)

    def _setup_chroma_client(self):
        """Setup ChromaDB client and collection with validation"""
        # Validate model and get expected dimensions
        if not self.embedding_model or self.embedding_model not in EMBEDDING_MODELS:
            models = list(EMBEDDING_MODELS.keys())
            raise ValueError(f"Invalid embedding_model '{self.embedding_model}'. Supported: {models}")

        if not self.collection_name:
            self.collection_name = self._path_to_collection_name(self.agent.repository)

        self.client = self.server.get_client()
        self._create_collection()

    def _setup_llamaindex(self):
        """Setup LlamaIndex components with instance isolation"""
        self._original_settings = {
            'llm': getattr(Settings, 'llm', None),
            'embed_model': getattr(Settings, 'embed_model', None),
            'node_parser': getattr(Settings, 'node_parser', None)
        }

        # Use agent's OpenAI configuration if available, otherwise use defaults
        if self.agent:
            self.llm = OpenAI(
                model=self.llm_model,
                temperature=self.llm_temperature,
                max_tokens=DEFAULT_CONFIG['OpenAIHandler']['max_completion_tokens'],
                additional_kwargs={"reasoning_effort": self.llm_reasoning_effort} \
                    if self.llm_reasoning_effort else None
            )
            self.embed_model = OpenAIEmbedding(model=self.embedding_model)

            self.logger.debug(f"Using agent's OpenAI config: model={self.agent.openai_handler.model}")
        else:
            # Fallback to default LlamaIndex settings
            self.llm = OpenAI()  # Instance-specific LLM
            self.embed_model = OpenAIEmbedding(model=self.embedding_model)

            self.logger.debug("Using LlamaIndex default OpenAI config")

        # Configure chunking if specified
        if self.chunk_size and self.chunk_overlap is not None:
            self.node_parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            self.logger.debug(f"Configured chunking: size={self.chunk_size}, overlap={self.chunk_overlap}")

        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Create index with explicit embed model to avoid global setting conflicts
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context,
            embed_model=self.embed_model  # Use instance-specific embed model
        )

    def _setup_reranker(self):
        """Setup LLMRerank if enabled"""
        if self.enable_reranking:
            self.reranker = LLMRerank(
                # choice_batch_size=self.rerank_batch_size,
                top_n=self.rerank_top_n,
                llm=self.llm
            )
            self.response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                llm=self.llm,
                # streaming=False,
                # use_async=False
            )
            self.logger.debug("LLMRerank enabled")
        else:
            self.reranker = None
            self.response_synthesizer = None
            self.logger.debug("Reranking disabled")

    def size(self):
        return self.collection.count()

    def add_text(self, text, metadata=None):
        """Add a single text document with automatic deduplication"""

        # Validate one more time before adding text
        # self._validate_dimensions(EMBEDDING_MODELS[self.embedding_model])

        # Generate deterministic ID from content for deduplication
        content_hash = hashlib.sha256(text.encode()).hexdigest()

        # Get count before operation
        count_before = self.collection.count()

        # Use upsert for automatic deduplication
        self.collection.upsert(
            ids=[content_hash],
            documents=[text],
            metadatas=[metadata or {}]
        )

        # Get count after operation
        count_after = self.collection.count()

        # Log if it was a new document
        if count_after > count_before:
            self.logger.info(f"Added new document with hash {content_hash[:8]}...")
            if self.log_db:
                with open(os.path.join(self.agent.get_log_dir(),
                    self.agent.get_id() + ".db-doc.log"), "a") as f:
                    f.write("="*80)
                    f.write(f"\n\nDOCUMENT: {text}\n\nMETADATA: {metadata}\n\n")
        else:
            self.logger.info(f"Updated existing document with hash {content_hash[:8]}...")

        return True

    def query(self, question, top_k=None, top_n=None, return_sources=False, filters=None):
        """Enhanced query with optional source URLs"""
        try:
            if self.size() == 0:
                return ("", []) if return_sources else ""

            should_rerank = self.reranker is not None
            top_k = max([top_k or self.rerank_top_k, self.rerank_top_n, 1]) if should_rerank else max(top_k or self.top_k, 1)

            nodes = self._retrieve_nodes(question, top_k, filters)

            if should_rerank:
                top_n = max([top_n or self.rerank_top_n, self.rerank_top_n, 1])
                if top_n != self.rerank_top_n:
                    self.reranker = LLMRerank(top_n=top_n, llm=self.llm)
                response, nodes = self._rerank_and_respond(question, nodes, top_n)
            else:
                response = self._standard_query(question, top_k, filters)

            if self.log_db:
                with open(os.path.join(self.agent.get_log_dir(),
                    self.agent.get_id() + ".db-query.log"), "a") as f:
                    f.write("="*80)
                    f.write(f"\n\nQUERY: {question}\n\nRESPONSE: {response}\n\n")

            if return_sources:
                sources = [
                    {'url': m['url'], 'depth': m.get('depth'), 'iteration': m.get('iteration')}
                    for node in nodes
                    if (m := getattr(node.node if hasattr(node, 'node') else node, 'metadata', {}))
                    and 'url' in m
                ]
                return response, sources

            return response

        except openai.APITimeoutError as e:
            self.logger.error(f"KB query timed out (OpenAI API): {e}")
            return ("", []) if return_sources else ""

    def _retrieve_nodes(self, question: str, retrieval_k: int, filters: MetadataFilters):
        """Retrieve nodes with instance-specific embed model"""
        retriever = self.index.as_retriever(
            similarity_top_k=retrieval_k,
            embed_model=self.embed_model,
            filters=filters)

        self.logger.debug(f"Using retriever (n={retrieval_k}) for query")
        return retriever.retrieve(question)

    def _rerank_and_respond(self, question: str, nodes: list, top_n=None):
        """Handle reranking with LLMRerank and response generation"""
        top_n = max([top_n or self.rerank_top_n, self.rerank_top_n, 1])
        if top_n != self.rerank_top_n: self.reranker = LLMRerank(top_n=top_n, llm=self.llm)

        # Use LLMRerank to rerank nodes
        reranked_nodes = self.reranker.postprocess_nodes(nodes, query_str=question)

        # Build context from reranked nodes
        # context = "\n\n".join([node.node.get_content() for node in reranked_nodes])

        # Generate response
        question = f"{question}\n\nIMPORTANT: Keep your response under {self.response_max_length} words!" if self.response_max_length else question
        response = self.response_synthesizer.synthesize(question, nodes=reranked_nodes)

        self.logger.debug(f"Using reranker (n={top_n}) for query")
        return str(response), reranked_nodes

    def _standard_query(self, question: str, top_k: int, filters: MetadataFilters):
        """Standard query with optional system prompt prepended"""

        # Simply prepend system prompt to the question
        if self.agent.role and len(self.agent.role) > 0:
            question = f"{self.agent.role}\n\n{question}"

        engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
            llm=self.llm,
            embed_model=self.embed_model,
            filters=filters)
        question = f"{question}\n\nIMPORTANT: Keep your response under {self.response_max_length} tokens!" if self.response_max_length else question
        return str(engine.query(question))

    def info(self):
        """Get knowledge base information"""
        try:
            return {
                "name": self.collection_name,
                "count": self.collection.count(),
                "url": self.server.server_url,
                "running": self.server.is_running(),
                "reranking": self.reranker is not None,
                "chunk_size": getattr(self, 'chunk_size', None),
                "chunk_overlap": getattr(self, 'chunk_overlap', None)
            }
        except Exception as e:
            self.logger.error(f"Failed to get info: {e}")
            return {"error": str(e)}

    def shutdown(self):
        """Shutdown the knowledge base"""
        # Release client from server tracking
        if hasattr(self, 'client'):
            self.server.release_client(self.client)

        # Only stop server if we own it (not shared)
        if self._owns_server and hasattr(self, 'server'):
            self.server.stop()

        self.logger.info("ChromaClientManager shutdown completed")