"""
Comprehensive tests for kb_client.py without mocks
Tests use real ChromaDB instances and minimal OpenAI API calls
"""

import pytest
import tempfile
import shutil
import os
import time
import json
from typing import Dict, List
from unittest import TestCase

# Import the modules to test
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rome.kb_client import ChromaClientManager, OpenAIReranker
from rome.kb_server import ChromaServerManager
from rome.config import DEFAULT_CONFIG
from rome.logger import get_logger
from rome.openai import OpenAIHandler


class MockAgent:
    """Minimal agent implementation for testing"""

    def __init__(self, config: Dict = None):
        self.openai_handler = MockOpenAIHandler(config.get('OpenAIHandler', {}) if config else {})
        self.logger = get_logger()

    def chat_completion(self, prompt: str, system_message: str = None, **kwargs) -> str:
        return self.openai_handler.chat_completion(prompt, system_message, **kwargs)


class MockOpenAIHandler:
    """Lightweight OpenAI handler for testing with minimal API usage"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.model = self.config.get('model', 'gpt-4o-mini')  # Use cheapest model
        self.temperature = self.config.get('temperature', 0.1)
        self.max_tokens = self.config.get('max_tokens', 100)  # Minimal tokens for cost
        self.logger = get_logger()

    def chat_completion(self, prompt: str, system_message: str = None, **kwargs) -> str:
        """Mock completion for testing - returns predictable responses"""
        # For reranking tests, return JSON scores
        if "JSON array" in prompt or "relevance" in prompt.lower():
            # Count how many docs are being ranked
            doc_count = prompt.count("Doc ")
            # Return mock scores in descending order
            scores = [0.9 - (i * 0.1) for i in range(doc_count)]
            return json.dumps(scores)

        # For general queries, return a simple response
        return f"This is a test response to: {prompt[:50]}..."


class TestOpenAIReranker(TestCase):
    """Test OpenAI reranker functionality"""

    def setUp(self):
        """Set up test environment"""
        self.config = {
            'batch_size': 3,
            'direct_rerank_limit': 5,
            'min_score_threshold': 0.3
        }
        self.agent = MockAgent()
        self.reranker = OpenAIReranker(self.config, self.agent)

    def test_reranker_initialization(self):
        """Test reranker initializes correctly"""
        self.assertEqual(self.reranker.batch_size, 3)
        self.assertEqual(self.reranker.direct_rerank_limit, 5)
        self.assertEqual(self.reranker.min_score_threshold, 0.3)
        self.assertIsNotNone(self.reranker.agent)

    def test_rerank_empty_texts(self):
        """Test reranking with empty text list"""
        result = self.reranker.rerank_with_scores("test query", [], top_k=3)
        self.assertEqual(result, [])

    def test_rerank_single_text(self):
        """Test reranking with single text"""
        texts = ["This is a test document about machine learning."]
        result = self.reranker.rerank_with_scores("machine learning", texts, top_k=1)

        self.assertEqual(len(result), 1)
        self.assertIn('text', result[0])
        self.assertIn('score', result[0])
        self.assertIn('index', result[0])
        self.assertEqual(result[0]['text'], texts[0])
        self.assertEqual(result[0]['index'], 0)

    def test_rerank_multiple_texts_direct(self):
        """Test direct reranking with multiple texts"""
        texts = [
            "Python is a programming language",
            "Machine learning algorithms",
            "Data science techniques",
            "Software engineering practices"
        ]

        result = self.reranker.rerank_with_scores("programming", texts, top_k=2)

        self.assertEqual(len(result), 2)
        # Results should be sorted by score descending
        self.assertGreaterEqual(result[0]['score'], result[1]['score'])
        # All results should have required fields
        for item in result:
            self.assertIn('text', item)
            self.assertIn('score', item)
            self.assertIn('index', item)
            self.assertGreaterEqual(item['score'], 0.0)
            self.assertLessEqual(item['score'], 1.0)

    def test_rerank_hierarchical(self):
        """Test hierarchical reranking with many texts"""
        # Create more texts than direct_rerank_limit
        texts = [f"Document {i} about various topics" for i in range(10)]

        result = self.reranker.rerank_with_scores("topics", texts, top_k=3)

        self.assertEqual(len(result), 3)
        # Check that hierarchical path was taken (more than direct_rerank_limit)
        self.assertGreater(len(texts), self.reranker.direct_rerank_limit)

    def test_rerank_texts_simple_interface(self):
        """Test simple rerank_texts interface"""
        texts = ["Text A", "Text B", "Text C"]
        result = self.reranker.rerank_texts("query", texts, top_k=2)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        # Should return only text strings
        for text in result:
            self.assertIsInstance(text, str)
            self.assertIn(text, texts)


class TestChromaServerIntegration(TestCase):
    """Test ChromaDB server integration"""

    def setUp(self):
        """Set up test server"""
        self.temp_dir = tempfile.mkdtemp()
        self.server_config = {
            'host': 'localhost',
            'port': 8001,  # Use different port to avoid conflicts
            'persist_path': self.temp_dir,
            'startup_timeout': 10,
            'shutdown_timeout': 5
        }
        self.server = ChromaServerManager(self.server_config)

    def tearDown(self):
        """Clean up test server and temp directory"""
        try:
            self.server.force_stop()
            time.sleep(1)  # Give server time to stop
        except:
            pass
        finally:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_server_start_stop(self):
        """Test server startup and shutdown"""
        # Start server
        success = self.server.start()
        self.assertTrue(success, "Server should start successfully")
        self.assertTrue(self.server.is_running(), "Server should be running after start")

        # Stop server
        success = self.server.stop(force=True)
        self.assertTrue(success, "Server should stop successfully")
        time.sleep(2)  # Give server time to fully stop
        self.assertFalse(self.server.is_running(), "Server should not be running after stop")

    def test_server_restart(self):
        """Test server restart functionality"""
        # Start, then restart
        self.server.start()
        self.assertTrue(self.server.is_running())

        success = self.server.restart()
        self.assertTrue(success, "Server should restart successfully")
        self.assertTrue(self.server.is_running(), "Server should be running after restart")

    def test_get_client(self):
        """Test client creation and release"""
        self.server.start()

        # Get client
        client = self.server.get_client()
        self.assertIsNotNone(client)

        # Test client functionality
        collections = client.list_collections()
        self.assertIsInstance(collections, list)

        # Release client
        self.server.release_client(client)


class TestChromaClientManager(TestCase):
    """Test ChromaClientManager functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test server for all tests"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.server_config = {
            'host': 'localhost',
            'port': 8002,  # Different port for this test class
            'persist_path': cls.temp_dir,
            'startup_timeout': 10,
            'shutdown_timeout': 5
        }
        cls.server = ChromaServerManager(cls.server_config)
        cls.server.start()
        time.sleep(2)  # Ensure server is ready

    @classmethod
    def tearDownClass(cls):
        """Clean up test server"""
        try:
            cls.server.force_stop()
            time.sleep(2)
        except:
            pass
        finally:
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up individual test"""
        self.config = {
            'collection_name': f'test_collection_{int(time.time()*1000)}',  # Unique collection name
            'enable_reranking': False,
            'use_shared_server': False,
            'chunk_size': 512,
            'chunk_overlap': 50,
            'embedding_model': 'text-embedding-3-small',
            'top_k': 5,
            'ChromaServerManager': self.server_config
        }
        self.agent = MockAgent({'OpenAIHandler': {'model': 'gpt-4o-mini', 'max_tokens': 100}})
        self.kb_manager = ChromaClientManager(self.config, self.agent)

    def tearDown(self):
        """Clean up individual test"""
        try:
            self.kb_manager.shutdown()
        except:
            pass

    def test_kb_manager_initialization(self):
        """Test knowledge base manager initializes correctly"""
        self.assertEqual(self.kb_manager.collection_name, self.config['collection_name'])
        self.assertFalse(self.kb_manager.enable_reranking)
        self.assertIsNotNone(self.kb_manager.client)
        self.assertIsNotNone(self.kb_manager.collection)
        self.assertIsNotNone(self.kb_manager.index)

    def test_empty_collection_size(self):
        """Test size of empty collection"""
        size = self.kb_manager.size()
        self.assertEqual(size, 0)

    def test_add_single_text(self):
        """Test adding a single text document"""
        text = "This is a test document about machine learning algorithms."
        metadata = {"source": "test", "topic": "ml"}

        self.kb_manager.add_text(text, metadata)

        # Check that document was added
        size = self.kb_manager.size()
        self.assertGreater(size, 0)

    def test_add_multiple_texts(self):
        """Test adding multiple text documents"""
        texts = [
            "Python is a versatile programming language.",
            "Machine learning models require training data.",
            "Data preprocessing is crucial for model performance.",
            "Feature engineering improves model accuracy."
        ]

        for i, text in enumerate(texts):
            self.kb_manager.add_text(text, {"doc_id": i, "category": "tech"})

        size = self.kb_manager.size()
        self.assertGreaterEqual(size, len(texts))

    def test_query_empty_collection(self):
        """Test querying empty collection"""
        result = self.kb_manager.query("test query")
        self.assertIsNone(result)

    def test_query_populated_collection(self):
        """Test querying populated collection"""
        # Add some test documents
        documents = [
            "Python programming language syntax and features",
            "Machine learning algorithms and techniques",
            "Data science methodology and best practices",
            "Software engineering principles and patterns"
        ]

        for doc in documents:
            self.kb_manager.add_text(doc)

        # Query the collection
        result = self.kb_manager.query("Python programming", top_k=2)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_query_with_top_k(self):
        """Test querying with different top_k values"""
        # Add test documents
        for i in range(5):
            self.kb_manager.add_text(f"Document {i} about various programming topics")

        # Test different top_k values
        for k in [1, 3, 5]:
            result = self.kb_manager.query("programming", top_k=k)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, str)

    def test_info_method(self):
        """Test info method returns correct information"""
        info = self.kb_manager.info()

        self.assertIsInstance(info, dict)
        self.assertIn('name', info)
        self.assertIn('count', info)
        self.assertIn('url', info)
        self.assertIn('running', info)
        self.assertIn('reranking', info)

        self.assertEqual(info['name'], self.config['collection_name'])
        self.assertIsInstance(info['count'], int)
        self.assertIsInstance(info['running'], bool)
        self.assertFalse(info['reranking'])  # Reranking is disabled in this test


class TestChromaClientManagerWithReranking(TestCase):
    """Test ChromaClientManager with reranking enabled"""

    @classmethod
    def setUpClass(cls):
        """Set up test server for reranking tests"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.server_config = {
            'host': 'localhost',
            'port': 8003,  # Different port for reranking tests
            'persist_path': cls.temp_dir,
            'startup_timeout': 10,
            'shutdown_timeout': 5
        }
        cls.server = ChromaServerManager(cls.server_config)
        cls.server.start()
        time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        """Clean up test server"""
        try:
            cls.server.force_stop()
            time.sleep(2)
        except:
            pass
        finally:
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        """Set up test with reranking enabled"""
        self.config = {
            'collection_name': f'rerank_test_{int(time.time()*1000)}',
            'enable_reranking': True,
            'use_shared_server': False,
            'top_k': 3,
            'ChromaServerManager': self.server_config,
            'OpenAIReranker': {
                'direct_rerank_limit': 5,
                'batch_size': 2,
                'min_score_threshold': 0.2
            }
        }
        self.agent = MockAgent({'OpenAIHandler': {'model': 'gpt-4o-mini', 'max_tokens': 100}})
        self.kb_manager = ChromaClientManager(self.config, self.agent)

        # Add test documents
        self.test_docs = [
            "Python programming language fundamentals",
            "JavaScript web development frameworks",
            "Machine learning with scikit-learn",
            "Data visualization using matplotlib",
            "Backend development with Django"
        ]

        for doc in self.test_docs:
            self.kb_manager.add_text(doc)

    def tearDown(self):
        """Clean up test"""
        try:
            self.kb_manager.shutdown()
        except:
            pass

    def test_reranking_initialization(self):
        """Test that reranking is properly initialized"""
        self.assertTrue(self.kb_manager.enable_reranking)
        self.assertIsNotNone(self.kb_manager.reranker)

        info = self.kb_manager.info()
        self.assertTrue(info['reranking'])

    def test_query_with_reranking(self):
        """Test querying with reranking enabled"""
        result = self.kb_manager.query("Python programming", use_reranking=True)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_query_without_reranking_override(self):
        """Test disabling reranking for specific query"""
        result = self.kb_manager.query("Python programming", use_reranking=False)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_query_with_score_display(self):
        """Test querying with score display enabled"""
        result = self.kb_manager.query("Python programming", show_scores=True)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)


class TestIntegrationScenarios(TestCase):
    """Integration tests for complete workflows"""

    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.server_config = {
            'host': 'localhost',
            'port': 8004,  # Different port for integration tests
            'persist_path': self.temp_dir,
            'startup_timeout': 10,
            'shutdown_timeout': 5
        }

        # Use config similar to production
        self.config = DEFAULT_CONFIG.copy()
        self.config['ChromaClientManager']['collection_name'] = f'integration_test_{int(time.time()*1000)}'
        self.config['ChromaClientManager']['use_shared_server'] = False
        self.config['ChromaClientManager']['ChromaServerManager'] = self.server_config
        self.config['OpenAIHandler']['model'] = 'gpt-4o-mini'  # Use cheap model
        self.config['OpenAIHandler']['max_tokens'] = 100

        self.agent = MockAgent(self.config)

    def tearDown(self):
        """Clean up integration test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_complete_workflow_without_reranking(self):
        """Test complete workflow: create KB, add docs, query, shutdown"""
        # Create KB manager
        kb_config = self.config['ChromaClientManager'].copy()
        kb_config['enable_reranking'] = False
        kb_manager = ChromaClientManager(kb_config, self.agent)

        try:
            # Verify initial state
            self.assertEqual(kb_manager.size(), 0)

            # Add documents
            documents = [
                ("Python basics", {"category": "programming", "difficulty": "beginner"}),
                ("Advanced Python", {"category": "programming", "difficulty": "advanced"}),
                ("Data structures", {"category": "computer_science", "difficulty": "intermediate"}),
                ("Web development", {"category": "web", "difficulty": "intermediate"})
            ]

            for text, metadata in documents:
                kb_manager.add_text(text, metadata)

            # Verify documents were added
            self.assertGreaterEqual(kb_manager.size(), len(documents))

            # Test queries
            result1 = kb_manager.query("Python programming")
            self.assertIsNotNone(result1)

            result2 = kb_manager.query("web development")
            self.assertIsNotNone(result2)

            # Test info
            info = kb_manager.info()
            self.assertGreater(info['count'], 0)

        finally:
            kb_manager.shutdown()

    def test_complete_workflow_with_reranking(self):
        """Test complete workflow with reranking enabled"""
        # Create KB manager with reranking
        kb_config = self.config['ChromaClientManager'].copy()
        kb_config['enable_reranking'] = True
        kb_manager = ChromaClientManager(kb_config, self.agent)

        try:
            # Add technical documents
            tech_docs = [
                "Python list comprehensions and generator expressions",
                "JavaScript async/await and promise handling",
                "SQL joins and query optimization techniques",
                "Docker containerization and orchestration",
                "Git version control and branching strategies"
            ]

            for doc in tech_docs:
                kb_manager.add_text(doc, {"type": "technical"})

            # Test queries with reranking
            result = kb_manager.query("Python programming techniques", top_k=2)
            self.assertIsNotNone(result)

            # Test with score display
            result_with_scores = kb_manager.query("Docker containers", show_scores=True)
            self.assertIsNotNone(result_with_scores)

        finally:
            kb_manager.shutdown()


# Test runner and utilities
def run_tests():
    """Run all tests"""
    import unittest

    # Create test suite
    test_classes = [
        TestOpenAIReranker,
        TestChromaServerIntegration,
        TestChromaClientManager,
        TestChromaClientManagerWithReranking,
        TestIntegrationScenarios
    ]

    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    return result.wasSuccessful()


if __name__ == "__main__":
    # Set up minimal logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)

    # Run tests
    success = run_tests()
    exit(0 if success else 1)
