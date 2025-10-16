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
    import chromadb; import llama_index
except ImportError as e:
    print(f"Import error: {e}, install with: pip install chromadb llama-index llama-index-vector-stores-chroma llama-index-embeddings-openai llama-index-llms-openai")
    exit(1)

import openai
from chromadb.utils.embedding_functions import (
   SentenceTransformerEmbeddingFunction, OpenAIEmbeddingFunction)
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter

from .config import set_attributes_from_config
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


class OpenAIReranker:
    """Simplified OpenAI reranker with hierarchical scaling for large document sets"""

    def __init__(self, config: Dict = None, agent=None):
        self.config = config or {}
        self.agent = agent
        self.logger = get_logger()

        # Set attributes from config
        set_attributes_from_config(self, self.config,
            ['batch_size', 'direct_rerank_limit', 'min_score_threshold'])
        self.logger.info(f"OpenAI reranker initialized")

    def rerank_with_scores(self, query: str, texts: List[str], top_k: int = 5) -> List[dict]:
        """Returns list of {text, score, index} sorted by relevance"""
        if not texts:
            return []

        # Use hierarchical approach for large document sets
        if len(texts) > self.direct_rerank_limit:
            return self._hierarchical_rerank(query, texts, top_k)
        else:
            return self._direct_rerank(query, texts, top_k)

    def _hierarchical_rerank(self, query: str, texts: List[str], top_k: int) -> List[dict]:
        """Two-stage reranking for large document sets"""
        # Stage 1: Quick filtering to manageable size
        stage1_size = min(self.direct_rerank_limit, len(texts) // 2)
        stage1_results = self._score_texts(query, texts, batch_size=10, quick_mode=True)
        stage1_results.sort(key=lambda x: x['score'], reverse=True)

        # Stage 2: Detailed reranking of filtered set
        stage1_texts = [item['text'] for item in stage1_results[:stage1_size]]
        stage2_results = self._direct_rerank(query, stage1_texts, top_k)

        # Map back to original indices
        index_map = {i: stage1_results[i]['index'] for i in range(len(stage1_results[:stage1_size]))}
        for result in stage2_results:
            result['index'] = index_map.get(result['index'], result['index'])

        return stage2_results

    def _direct_rerank(self, query: str, texts: List[str], top_k: int) -> List[dict]:
        """Direct reranking for smaller document sets"""
        scored_docs = self._score_texts(query, texts, self.batch_size)
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:top_k]

    def _score_texts(self, query: str, texts: List[str], batch_size: int, quick_mode: bool = False) -> List[dict]:
        """Score texts in batches with LLM"""
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self._score_batch(query, batch, i, quick_mode)
            all_results.extend(batch_results)

        return all_results

    def _score_batch(self, query: str, texts: List[str], start_idx: int, quick_mode: bool = False) -> List[dict]:
        """Score a single batch of documents"""
        # Build document list for prompt
        doc_list = ""
        max_len = 300 if quick_mode else 800
        for i, text in enumerate(texts):
            content = text[:max_len] + "..." if len(text) > max_len else text
            doc_list += f"\nDoc {start_idx + i + 1}: {content}\n"

        # Create scoring prompt
        if quick_mode:
            prompt = f"Rate relevance (0.0-1.0) for filtering:\nQuery: {query}\n{doc_list}\nJSON array only: [0.8, 0.2, ...]"
        else:
            prompt = f"""Rate each document's relevance to the query (0.0-1.0):
1.0 = Perfect match, 0.7-0.9 = Highly relevant, 0.4-0.6 = Somewhat relevant, 0.1-0.3 = Barely relevant, 0.0 = Not relevant

Query: {query}
{doc_list}

Respond with ONLY a JSON array of scores: [0.8, 0.2, 0.9]"""

        # Get LLM response
        response = self.agent.chat_completion(prompt=prompt)
        scores = parse_json_response(response)

        # Validate and format results
        if not isinstance(scores, list) or len(scores) < len(texts):
            raise ValueError(f"Expected {len(texts)} scores, got {len(scores) if isinstance(scores, list) else type(scores)}")

        return [{'text': text,
                'score': max(0.0, min(1.0, float(scores[i]))),
                'index': start_idx + i}
                for i, text in enumerate(texts)]

    def rerank_texts(self, query: str, texts: List[str], top_k: int = 5) -> List[str]:
        """Simple interface that returns only reranked texts"""
        scored = self.rerank_with_scores(query, texts, top_k)
        return [item['text'] for item in scored]


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
            ['enable_reranking', 'chunk_size', 'chunk_overlap', 'embedding_model', 'use_shared_server'],
            ['top_k', 'log_db', 'collection_name'])

        assert self.embedding_model in EMBEDDING_MODELS, \
            f"Invalid embedding model: {self.embedding_model}"

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
            embedding_fn = OpenAIEmbeddingFunction(
                model_name=self.embedding_model, api_key=os.getenv('OPENAI_API_KEY'))

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
                model=self.agent.openai_handler.model,
                temperature=self.agent.openai_handler.temperature
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
        """Setup reranker if enabled"""
        if self.enable_reranking:
            reranker_config = self.agent.config.get('OpenAIReranker', {})
            self.reranker = OpenAIReranker(reranker_config, self.agent)
            self.logger.debug("OpenAI reranker enabled")
        else:
            self.reranker = None
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

    def query(self, question, top_k=None, use_reranking=None, show_scores=False):
        """Enhanced query with simplified reranking logic and empty collection validation"""
        try:
            # Check if collection is empty before proceeding
            if self.size() == 0:
                return ""

            top_k = top_k or self.top_k or 5
            should_rerank = (use_reranking is True) or \
                (use_reranking is None and self.reranker is not None)

            if should_rerank and self.reranker:
                # Calculate how many docs to retrieve for reranking
                retrieval_k = self._calculate_retrieval_size(top_k)
                # Get documents
                nodes = self._retrieve_nodes(question, retrieval_k)
                # Rerank and generate response
                response = self._rerank_and_respond(question, nodes, top_k, show_scores)
            else:
                # Standard query without reranking
                response = self._standard_query(question, top_k)

            if self.log_db:
                with open(os.path.join(self.agent.get_log_dir(),
                    self.agent.get_id() + ".db-query.log"), "a") as f:
                    f.write("="*80)
                    f.write(f"\n\nQUERY: {question}\n\nRESPONSE: {response}\n\n")

            return response

        except openai.APITimeoutError as e:
            self.logger.error(f"KB query timed out (OpenAI API): {e}")
            return ""

    def _calculate_retrieval_size(self, top_k: int) -> int:
        """Calculate optimal retrieval size for reranking"""
        multiplier = 4 if self.reranker.direct_rerank_limit >= 100 else 3 if self.reranker.direct_rerank_limit >= 50 else 2
        return max(top_k, min(top_k * multiplier, self.reranker.direct_rerank_limit))

    def _retrieve_nodes(self, question: str, retrieval_k: int):
        """Retrieve nodes with instance-specific embed model"""
        retriever = self.index.as_retriever(
            similarity_top_k=retrieval_k,
            embed_model=self.embed_model)
        return retriever.retrieve(question)

    def _rerank_and_respond(self, question: str, nodes, top_k: int, show_scores: bool) -> str:
        """Handle reranking and response generation"""
        if len(nodes) <= 1:
            # Not enough nodes for reranking, use standard query
            return self._standard_query(question, top_k)

        # Extract and rerank
        texts = [node.node.get_content() for node in nodes]
        scored_results = self.reranker.rerank_with_scores(question, texts, top_k)

        # Optional score logging
        if show_scores:
            self._log_rerank_scores(question, scored_results)

        # Filter by score threshold and create context
        context = self._build_context(scored_results)

        # Generate response
        response = self.agent.chat_completion(
            prompt=f"Context:\n{context}\n\nQuestion: {question}",
            system_message="Answer based on the provided context. Use the relevance scores to prioritize information. Be concise and cite relevant details.")
        return response

    def _log_rerank_scores(self, question: str, scored_results: List[dict]):
        """Log reranking scores for debugging"""
        self.logger.info(f"Reranking scores for: {question}")
        for i, item in enumerate(scored_results):
            self.logger.info(f"{i+1}. Score: {item['score']:.2f} | {item['text'][:100]}...")

    def _build_context(self, scored_results: List[dict]) -> str:
        """Build context string from scored results"""
        context_parts = [
            f"[Relevance: {item['score']:.1f}] {item['text']}"
            for item in scored_results
            if item['score'] >= self.reranker.min_score_threshold
        ]
        return "\n\n".join(context_parts)

    def _standard_query(self, question: str, top_k: int) -> str:
        """Standard query with optional system prompt prepended"""

        # Simply prepend system prompt to the question
        if self.agent.role and len(self.agent.role) > 0:
            question = f"{self.agent.role}\n\n{question}"

        engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
            llm=self.llm,
            embed_model=self.embed_model
        )
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
