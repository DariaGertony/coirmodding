"""
Unit tests for LLM components.

This module tests the AsyncOllama and QueryExpander classes,
including error handling and edge cases.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import threading
import time
from coir.beir.retrieval.search.llm import AsyncOllama, QueryExpander


class TestAsyncOllama(unittest.TestCase):
    """Test cases for AsyncOllama class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm_name = "test-model"
        self.prompt = "Test prompt:"
        self.async_ollama = AsyncOllama(self.llm_name, self.prompt)
    
    def test_init(self):
        """Test AsyncOllama initialization."""
        self.assertEqual(self.async_ollama.llm, self.llm_name)
        self.assertEqual(self.async_ollama.prompt, self.prompt)
        self.assertIsNone(self.async_ollama.result)
        self.assertEqual(len(self.async_ollama.threads), 0)
    
    @patch('coir.beir.retrieval.search.llm.async_ollama.ollama.chat')
    def test_ask_async_success(self, mock_chat):
        """Test successful async query processing."""
        # Mock ollama response
        mock_response = {
            'message': {
                'content': 'Test response'
            }
        }
        mock_chat.return_value = mock_response
        
        # Test callback
        callback_results = []
        def test_callback(i, query, result):
            callback_results.append((i, query, result))
        
        # Submit async query
        test_query = "test query"
        self.async_ollama.ask_async(0, test_query, test_callback)
        
        # Wait for completion
        self.async_ollama.wait()
        
        # Verify results
        self.assertEqual(len(callback_results), 1)
        self.assertEqual(callback_results[0], (0, test_query, 'Test response'))
        self.assertEqual(self.async_ollama.result, 'Test response')
        
        # Verify ollama.chat was called with correct parameters
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args
        self.assertEqual(call_args[1]['model'], self.llm_name)
        self.assertEqual(
            call_args[1]['messages'][0]['content'], 
            f"{self.prompt}\n{test_query}"
        )
    
    @patch('coir.beir.retrieval.search.llm.async_ollama.ollama.chat')
    def test_ask_async_error_handling(self, mock_chat):
        """Test error handling in async query processing."""
        # Mock ollama to raise an exception
        mock_chat.side_effect = Exception("Connection error")
        
        # Test callback
        callback_results = []
        def test_callback(i, query, result):
            callback_results.append((i, query, result))
        
        # Submit async query
        test_query = "test query"
        self.async_ollama.ask_async(0, test_query, test_callback)
        
        # Wait for completion
        self.async_ollama.wait()
        
        # Verify error handling
        self.assertEqual(len(callback_results), 1)
        self.assertEqual(callback_results[0][0], 0)
        self.assertEqual(callback_results[0][1], test_query)
        self.assertIn("Error processing query", callback_results[0][2])
    
    def test_ask_async_without_callback(self):
        """Test async query without callback."""
        with patch('coir.beir.retrieval.search.llm.async_ollama.ollama.chat') as mock_chat:
            mock_response = {'message': {'content': 'Test response'}}
            mock_chat.return_value = mock_response
            
            # Submit async query without callback
            self.async_ollama.ask_async(0, "test query")
            self.async_ollama.wait()
            
            # Should complete without errors
            self.assertEqual(self.async_ollama.result, 'Test response')
    
    def test_multiple_async_queries(self):
        """Test multiple concurrent async queries."""
        with patch('coir.beir.retrieval.search.llm.async_ollama.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Response'}}
            
            callback_results = []
            def test_callback(i, query, result):
                callback_results.append((i, query, result))
            
            # Submit multiple queries
            queries = ["query1", "query2", "query3"]
            for i, query in enumerate(queries):
                self.async_ollama.ask_async(i, query, test_callback)
            
            # Wait for all to complete
            self.async_ollama.wait()
            
            # Verify all queries were processed
            self.assertEqual(len(callback_results), 3)
            self.assertEqual(mock_chat.call_count, 3)
    
    def test_is_busy(self):
        """Test is_busy method."""
        # Initially not busy
        self.assertFalse(self.async_ollama.is_busy())
        
        with patch('coir.beir.retrieval.search.llm.async_ollama.ollama.chat') as mock_chat:
            # Mock a slow response
            def slow_response(*args, **kwargs):
                time.sleep(0.1)
                return {'message': {'content': 'Response'}}
            
            mock_chat.side_effect = slow_response
            
            # Submit query
            self.async_ollama.ask_async(0, "test query")
            
            # Should be busy immediately after submission
            self.assertTrue(self.async_ollama.is_busy())
            
            # Wait and check again
            self.async_ollama.wait()
            self.assertFalse(self.async_ollama.is_busy())
    
    def test_get_active_count(self):
        """Test get_active_count method."""
        self.assertEqual(self.async_ollama.get_active_count(), 0)
        
        with patch('coir.beir.retrieval.search.llm.async_ollama.ollama.chat') as mock_chat:
            mock_chat.return_value = {'message': {'content': 'Response'}}
            
            # Submit multiple queries
            for i in range(3):
                self.async_ollama.ask_async(i, f"query{i}")
            
            # Wait for completion
            self.async_ollama.wait()
            
            # After completion, active count should be 0
            self.assertEqual(self.async_ollama.get_active_count(), 0)


class TestQueryExpander(unittest.TestCase):
    """Test cases for QueryExpander class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm_name = "test-model"
        self.prompt = "Expand this query:"
        self.expander = QueryExpander(self.llm_name, self.prompt)
    
    def test_init(self):
        """Test QueryExpander initialization."""
        self.assertEqual(self.expander.llm_name, self.llm_name)
        self.assertEqual(self.expander.prompt, self.prompt)
    
    @patch('coir.beir.retrieval.search.llm.query_expander.AsyncOllama')
    def test_expand_queries_success(self, mock_async_ollama_class):
        """Test successful query expansion."""
        # Mock AsyncOllama instance
        mock_ollama = Mock()
        mock_async_ollama_class.return_value = mock_ollama
        
        # Mock the ask_async method to call callback immediately
        def mock_ask_async(i, query, callback):
            if callback:
                callback(i, query, f"expanded {query}")
        
        mock_ollama.ask_async.side_effect = mock_ask_async
        
        # Test query expansion
        queries = ["query1", "query2", "query3"]
        result = self.expander.expand_queries(queries)
        
        # Verify results
        expected = [
            "query1\nexpanded query1",
            "query2\nexpanded query2", 
            "query3\nexpanded query3"
        ]
        self.assertEqual(result, expected)
        
        # Verify AsyncOllama was used correctly
        mock_async_ollama_class.assert_called_once_with(self.llm_name, self.prompt)
        self.assertEqual(mock_ollama.ask_async.call_count, 3)
        mock_ollama.wait.assert_called_once()
    
    def test_expand_queries_empty_list(self):
        """Test expanding empty query list."""
        result = self.expander.expand_queries([])
        self.assertEqual(result, [])
    
    @patch('coir.beir.retrieval.search.llm.query_expander.AsyncOllama')
    def test_expand_query_dict(self, mock_async_ollama_class):
        """Test expanding queries from dictionary format."""
        # Mock AsyncOllama instance
        mock_ollama = Mock()
        mock_async_ollama_class.return_value = mock_ollama
        
        def mock_ask_async(i, query, callback):
            if callback:
                callback(i, query, f"expanded {query}")
        
        mock_ollama.ask_async.side_effect = mock_ask_async
        
        # Test dictionary expansion
        query_dict = {
            "q1": "first query",
            "q2": "second query"
        }
        result = self.expander.expand_query_dict(query_dict)
        
        # Verify results maintain ID mapping
        expected = {
            "q1": "first query\nexpanded first query",
            "q2": "second query\nexpanded second query"
        }
        self.assertEqual(result, expected)
    
    def test_expand_query_dict_empty(self):
        """Test expanding empty query dictionary."""
        result = self.expander.expand_query_dict({})
        self.assertEqual(result, {})
    
    def test_set_prompt(self):
        """Test updating prompt template."""
        new_prompt = "New prompt template:"
        self.expander.set_prompt(new_prompt)
        self.assertEqual(self.expander.prompt, new_prompt)
    
    def test_set_llm_model(self):
        """Test updating LLM model."""
        new_model = "new-model"
        self.expander.set_llm_model(new_model)
        self.assertEqual(self.expander.llm_name, new_model)


if __name__ == '__main__':
    unittest.main()