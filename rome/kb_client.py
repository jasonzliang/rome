# knowledge_base.py
import json
from typing import Optional, List, Dict

# ChromaDB and LlamaIndex imports
try:
    import chromadb
    from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.core.node_parser import SentenceSplitter
except ImportError as e:
    print(f"Import error: {e}")
    print("Install with: pip install chromadb llama-index llama-index-vector-stores-chroma llama-index-embeddings-openai llama-index-llms-openai")
    exit(1)

from .config import set_attributes_from_config
from .logger import get_logger
from .kb_server import ChromaServerManager


class OpenAIReranker:
    """Effective OpenAI reranker with hierarchical scaling for large document sets"""

    def __init__(self, config: Dict = None, agent=None):
        self.config = config or {}
        self.agent = agent
        self.logger = get_logger()

        # Set attributes from config
        set_attributes_from_config(self, self.config,
            ['batch_size', 'max_docs', 'min_score_threshold'])

        self.logger.info(f"OpenAI reranker initialized: max_docs={self.max_docs}, agent_available={self.agent is not None}")

    def rerank_with_scores(self, query: str, texts: List[str], top_k: int = 5) -> List[dict]:
        """Returns list of {text, score, index} sorted by relevance"""
        if not texts:
            return []

        # For large document sets, use hierarchical approach
        if len(texts) > self.max_docs:
            return self._hierarchical_rerank(query, texts, top_k)
        else:
            return self._direct_rerank(query, texts, top_k)

    def _hierarchical_rerank(self, query: str, texts: List[str], top_k: int) -> List[dict]:
        """Two-stage reranking for large document sets"""
        self.logger.debug(f"Hierarchical reranking {len(texts)} documents...")

        # Stage 1: Quick scoring in larger batches to filter down
        stage1_batch_size = 10
        stage1_keep = min(self.max_docs, len(texts) // 2)  # Keep top half, max configured

        self.logger.debug(f"Stage 1: Quick filtering to top {stage1_keep}")
        quick_scored = []

        for i in range(0, len(texts), stage1_batch_size):
            batch = texts[i:i + stage1_batch_size]
            batch_scores = self._quick_score_batch(query, batch, i)
            quick_scored.extend(batch_scores)

        # Sort and keep top candidates
        quick_scored.sort(key=lambda x: x['score'], reverse=True)
        stage1_texts = [item['text'] for item in quick_scored[:stage1_keep]]

        # Stage 2: Detailed reranking of filtered set
        self.logger.debug(f"Stage 2: Detailed reranking of top {len(stage1_texts)}")
        final_results = self._direct_rerank(query, stage1_texts, top_k)

        # Map back to original indices
        for result in final_results:
            stage1_item = quick_scored[result['original_index']]
            result['original_index'] = stage1_item['original_index']
            result['index'] = stage1_item['original_index']

        return final_results

    def _direct_rerank(self, query: str, texts: List[str], top_k: int) -> List[dict]:
        """Direct reranking for smaller document sets"""
        scored_docs = []

        # Process in batches for better reliability
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_scores = self._score_batch(query, batch, i)
            scored_docs.extend(batch_scores)

        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:top_k]

    def _quick_score_batch(self, query: str, texts: List[str], start_idx: int = 0) -> List[dict]:
        """Quick scoring with shorter prompts for stage 1 filtering"""
        doc_list = ""
        for i, text in enumerate(texts):
            doc_content = text[:300] + "..." if len(text) > 300 else text
            doc_list += f"\nDoc {start_idx + i + 1}: {doc_content}\n"

        prompt = f"""Quickly rate relevance (0.0-1.0) for filtering:

Query: {query}
{doc_list}

JSON array only: [0.8, 0.2, 0.9, ...]"""

        try:
            if self.agent:
                # Use agent's chat completion
                response = self.agent.chat_completion(prompt=prompt)
            else:
                # Should not happen in normal usage, but provide fallback
                self.logger.warning("No agent available for quick scoring, using default scores")
                return [
                    {
                        'text': text,
                        'score': max(0.1, 0.8 - (i * 0.1)),
                        'original_index': start_idx + i,
                        'index': start_idx + i
                    }
                    for i, text in enumerate(texts)
                ]

            # Parse JSON response
            content = response.strip()
            if content.startswith('```'):
                content = content.split('\n')[1:-1]
                content = '\n'.join(content)

            scores = json.loads(content)

            return [
                {
                    'text': text,
                    'score': float(score),
                    'original_index': start_idx + i,
                    'index': start_idx + i
                }
                for i, (text, score) in enumerate(zip(texts, scores))
            ]

        except Exception as e:
            self.logger.error(f"Quick scoring failed: {e}")
            # Fallback: return with decreasing scores
            return [
                {
                    'text': text,
                    'score': max(0.1, 0.8 - (i * 0.1)),
                    'original_index': start_idx + i,
                    'index': start_idx + i
                }
                for i, text in enumerate(texts)
            ]

    def _score_batch(self, query: str, texts: List[str], start_idx: int = 0) -> List[dict]:
        """Score a batch of documents"""
        doc_list = ""
        for i, text in enumerate(texts):
            doc_content = text[:800] + "..." if len(text) > 800 else text
            doc_list += f"\nDocument {start_idx + i + 1}:\n{doc_content}\n"

        prompt = f"""Rate each document's relevance to the query on a scale of 0.0 to 1.0, where:
- 1.0 = Perfectly relevant, directly answers the query
- 0.7-0.9 = Highly relevant, contains key information
- 0.4-0.6 = Somewhat relevant, tangentially related
- 0.1-0.3 = Barely relevant, mentions related concepts
- 0.0 = Not relevant at all

Query: {query}
{doc_list}

Respond with ONLY a JSON array of scores in order. Example: [0.8, 0.2, 0.9]
Do not include any other text, backticks, or formatting."""

        try:
            if self.agent:
                # Use agent's chat completion
                response = self.agent.chat_completion(prompt=prompt)
            else:
                # Should not happen in normal usage, but provide fallback
                self.logger.warning("No agent available for scoring, using default scores")
                return [
                    {
                        'text': text,
                        'score': max(0.1, 1.0 - (i * 0.15)),
                        'original_index': start_idx + i,
                        'index': start_idx + i
                    }
                    for i, text in enumerate(texts)
                ]

            # Parse JSON scores with better error handling
            content = response.strip()
            content = content.replace('```json', '').replace('```', '').strip()

            if content.startswith('[') and content.endswith(']'):
                scores = json.loads(content)
            else:
                # Try to extract just the array part
                import re
                match = re.search(r'\[([\d\s,\.]+)\]', content)
                if match:
                    scores = json.loads(match.group(0))
                else:
                    raise ValueError(f"Could not parse JSON from: {content}")

            # Validate scores
            if len(scores) != len(texts):
                self.logger.warning(f"Expected {len(texts)} scores, got {len(scores)}")
                # Pad or truncate as needed
                while len(scores) < len(texts):
                    scores.append(0.1)
                scores = scores[:len(texts)]

            # Create scored results
            results = []
            for i, (text, score) in enumerate(zip(texts, scores)):
                results.append({
                    'text': text,
                    'score': max(0.0, min(1.0, float(score))),  # Clamp to [0,1]
                    'original_index': start_idx + i,
                    'index': start_idx + i
                })

            return results

        except Exception as e:
            self.logger.error(f"Batch scoring failed: {e}")
            # Fallback: return with decreasing scores
            return [
                {
                    'text': text,
                    'score': max(0.1, 1.0 - (i * 0.15)),
                    'original_index': start_idx + i,
                    'index': start_idx + i
                }
                for i, text in enumerate(texts)
            ]

    def rerank_texts(self, query: str, texts: List[str], top_k: int = 5) -> List[str]:
        """Simple interface that just returns reranked texts"""
        scored = self.rerank_with_scores(query, texts, top_k)
        return [item['text'] for item in scored]


class ChromaClientManager:
    """Enhanced ChromaDB + LlamaIndex knowledge base with reranking"""

    def __init__(self, config: Dict = None, agent=None):
        self.config = config or {}
        self.agent = agent
        self.logger = get_logger()

        # Set attributes from config
        set_attributes_from_config(self, self.config,
            ['collection_name', 'enable_reranking', 'chunk_size', 'chunk_overlap',
             'embedding_model', 'use_shared_server'])

        # Get or create server manager instance
        if self.use_shared_server:
            # Use shared singleton server
            server_config = self.config.get('ChromaServerManager', {})
            self.server = ChromaServerManager.get_instance(server_config)
            self._owns_server = False
        else:
            # Create dedicated server instance
            server_config = self.config.get('ChromaServerManager', {})
            self.server = ChromaServerManager(server_config)
            self._owns_server = True

        # Initialize ChromaDB client and collection
        self._setup_chroma_client()

        # Initialize LlamaIndex components
        self._setup_llamaindex()

        # Initialize reranker if enabled
        self._setup_reranker()

        self.logger.info(f"ChromaClientManager initialized: collection={self.collection_name}, reranking={self.enable_reranking}")

    def _setup_chroma_client(self):
        """Setup ChromaDB client and collection"""
        self.client = self.server.get_client()

        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            self.logger.debug(f"Connected to existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(self.collection_name)
            self.logger.info(f"Created new collection: {self.collection_name}")

    def _setup_llamaindex(self):
        """Setup LlamaIndex components"""
        # Use agent's OpenAI configuration if available, otherwise use defaults
        if self.agent and hasattr(self.agent, 'openai_handler'):
            # Use agent's model and temperature settings
            Settings.llm = OpenAI(
                model=self.agent.openai_handler.model,
                temperature=self.agent.openai_handler.temperature
            )
            Settings.embed_model = OpenAIEmbedding(model=self.embedding_model)
            self.logger.debug(f"Using agent's OpenAI config: model={self.agent.openai_handler.model}")
        else:
            # Fallback to default LlamaIndex settings
            Settings.llm = OpenAI()  # Uses LlamaIndex defaults
            Settings.embed_model = OpenAIEmbedding(model=self.embedding_model)
            self.logger.debug("Using LlamaIndex default OpenAI config")

        # Configure chunking if specified
        if self.chunk_size and self.chunk_overlap is not None:
            Settings.node_parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            self.logger.debug(f"Configured chunking: size={self.chunk_size}, overlap={self.chunk_overlap}")

        # Initialize vector store and index
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store, storage_context=self.storage_context)

    def _setup_reranker(self):
        """Setup reranker if enabled"""
        if self.enable_reranking:
            reranker_config = self.config.get('OpenAIReranker', {})
            self.reranker = OpenAIReranker(reranker_config, self.agent)
            self.logger.debug("OpenAI reranker enabled")
        else:
            self.reranker = None
            self.logger.debug("Reranking disabled")

    def add_text(self, text, metadata=None):
        """Add a single text document"""
        try:
            self.index.insert(Document(text=text, metadata=metadata or {}))
            return True
        except Exception as e:
            self.logger.error(f"Failed to add text: {e}")
            return False

    def add_docs(self, docs):
        """Add multiple documents"""
        try:
            for doc in docs:
                self.index.insert(doc)
            self.logger.info(f"Added {len(docs)} documents")
            return len(docs)
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return 0

    def query(self, question, top_k=5, use_reranking=None, show_scores=False):
        """Enhanced query with optional reranking"""
        try:
            should_rerank = (use_reranking is True) or (use_reranking is None and self.reranker is not None)

            if should_rerank and self.reranker:
                # Smart retrieval sizing based on reranking capability
                if self.reranker.max_docs >= 50:
                    retrieval_k = min(top_k * 4, 100)
                else:
                    retrieval_k = min(top_k * 2, 20)

                retriever = self.index.as_retriever(similarity_top_k=retrieval_k)
                nodes = retriever.retrieve(question)

                if len(nodes) > 1:
                    # Extract texts and rerank
                    texts = [node.node.get_content() for node in nodes]
                    scored_results = self.reranker.rerank_with_scores(question, texts, top_k)

                    if show_scores:
                        self.logger.info(f"Reranking scores for: {question}")
                        for i, item in enumerate(scored_results):
                            self.logger.info(f"{i+1}. Score: {item['score']:.2f} | {item['text'][:100]}...")

                    # Create context from top reranked results
                    context_parts = []
                    for item in scored_results:
                        if item['score'] >= self.reranker.min_score_threshold:
                            context_parts.append(f"[Relevance: {item['score']:.1f}] {item['text']}")

                    context = "\n\n".join(context_parts)

                    # Generate final answer using agent's chat completion
                    if self.agent:
                        response = self.agent.chat_completion(
                            prompt=f"Context:\n{context}\n\nQuestion: {question}",
                            system_message="Answer based on the provided context. Use the relevance scores to prioritize information. Be concise and cite relevant details."
                        )
                        return response
                    else:
                        # Fallback: use LlamaIndex engine when no agent available
                        from llama_index.core import get_response_synthesizer
                        synthesizer = get_response_synthesizer(response_mode="compact")

                        # Create mock nodes with context
                        from llama_index.core.schema import NodeWithScore, TextNode
                        nodes_with_scores = []
                        for item in scored_results:
                            if item['score'] >= self.reranker.min_score_threshold:
                                text_node = TextNode(text=item['text'])
                                node_with_score = NodeWithScore(node=text_node, score=item['score'])
                                nodes_with_scores.append(node_with_score)

                        response = synthesizer.synthesize(question, nodes_with_scores)
                        return str(response)

                else:
                    # Single result - no reranking needed
                    engine = self.index.as_query_engine(similarity_top_k=top_k, response_mode="compact")
                    return str(engine.query(question))
            else:
                # Regular query without reranking
                engine = self.index.as_query_engine(similarity_top_k=top_k, response_mode="compact")
                return str(engine.query(question))

        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return f"Error: {e}"

    def chat(self, message):
        """Chat interface"""
        try:
            engine = self.index.as_chat_engine(chat_mode="context")
            return str(engine.chat(message))
        except Exception as e:
            self.logger.error(f"Chat failed: {e}")
            return f"Error: {e}"

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
        try:
            # Release client from server tracking
            if hasattr(self, 'client'):
                self.server.release_client(self.client)

            # Only stop server if we own it (not shared)
            if self._owns_server and hasattr(self, 'server'):
                self.server.stop()

            self.logger.info("ChromaClientManager shutdown completed")
        except Exception as e:
            self.logger.error(f"ChromaClientManager shutdown error: {e}")
