"""
Integration tests for the evaluation layer with new search methods.

This module tests the integration between the EvaluateRetrieval class,
the SearchMethodFactory, and all search method implementations.
"""

import pytest
import numpy as np
from typing import Dict, Any
from unittest.mock import Mock, patch

# Import the components we're testing
from coir.beir.retrieval.evaluation import EvaluateRetrieval
from coir.beir.retrieval.search.factory import SearchMethodFactory
from coir.beir.retrieval.search.base import BaseSearch
from coir.beir.retrieval.search.dense.exact_search import DenseRetrievalExactSearch
from coir.beir.retrieval.search.lexical.jaccard_search import LexicalJaccardSearch
from coir.beir.retrieval.search.lexical.bm25_search import LexicalBM25Search
from coir.beir.retrieval.search.hybrid.simple_hybrid_search import SimpleHybridSearch


class TestSearchMethodFactory:
    """Test the SearchMethodFactory functionality."""
    
    def test_list_available_methods(self):
        """Test that all expected search methods are available."""
        methods = SearchMethodFactory.list_available_methods()
        expected_methods = ['dense', 'semantic', 'jaccard', 'bm25', 'simple_hybrid', 'advanced_hybrid']
        
        for method in expected_methods:
            assert method in methods, f"Method {method} not found in available methods"
    
    def test_create_jaccard_search(self):
        """Test creating a Jaccard search method."""
        search_method = SearchMethodFactory.create_search_method('jaccard', batch_size=64)
        
        assert isinstance(search_method, LexicalJaccardSearch)
        assert search_method.batch_size == 64
    
    def test_create_bm25_search(self):
        """Test creating a BM25 search method."""
        search_method = SearchMethodFactory.create_search_method('bm25', batch_size=128)
        
        assert isinstance(search_method, LexicalBM25Search)
        assert search_method.batch_size == 128
    
    def test_create_dense_search_with_alias(self):
        """Test creating dense search using both 'dense' and 'semantic' aliases."""
        # Mock model for dense search
        mock_model = Mock()
        mock_model.encode_corpus.return_value = np.random.rand(10, 768)
        mock_model.encode_queries.return_value = np.random.rand(5, 768)
        
        dense_search = SearchMethodFactory.create_search_method('dense', model=mock_model)
        semantic_search = SearchMethodFactory.create_search_method('semantic', model=mock_model)
        
        assert isinstance(dense_search, DenseRetrievalExactSearch)
        assert isinstance(semantic_search, DenseRetrievalExactSearch)
        assert type(dense_search) == type(semantic_search)
    
    def test_create_hybrid_search(self):
        """Test creating a hybrid search method."""
        mock_model = Mock()
        mock_model.encode_corpus.return_value = np.random.rand(10, 768)
        mock_model.encode_queries.return_value = np.random.rand(5, 768)
        
        hybrid_search = SearchMethodFactory.create_search_method('simple_hybrid', model=mock_model)
        
        assert isinstance(hybrid_search, SimpleHybridSearch)
    
    def test_unknown_search_method_raises_error(self):
        """Test that requesting an unknown search method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown search method: unknown_method"):
            SearchMethodFactory.create_search_method('unknown_method')
    
    def test_register_new_search_method(self):
        """Test registering a new search method."""
        class CustomSearch(BaseSearch):
            def search(self, corpus, queries, top_k, **kwargs):
                return {}
        
        SearchMethodFactory.register_search_method('custom', CustomSearch)
        
        assert 'custom' in SearchMethodFactory.list_available_methods()
        
        custom_search = SearchMethodFactory.create_search_method('custom')
        assert isinstance(custom_search, CustomSearch)
    
    def test_register_invalid_search_method_raises_error(self):
        """Test that registering a non-BaseSearch class raises ValueError."""
        class InvalidSearch:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseSearch"):
            SearchMethodFactory.register_search_method('invalid', InvalidSearch)
    
    def test_get_search_method_class(self):
        """Test getting search method class without instantiation."""
        jaccard_class = SearchMethodFactory.get_search_method_class('jaccard')
        assert jaccard_class == LexicalJaccardSearch
        
        with pytest.raises(ValueError, match="Unknown search method"):
            SearchMethodFactory.get_search_method_class('unknown')


class TestEvaluateRetrievalIntegration:
    """Test the EvaluateRetrieval class with different search methods."""
    
    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus for testing."""
        return {
            "doc1": {"title": "Machine Learning", "text": "Machine learning is a subset of artificial intelligence"},
            "doc2": {"title": "Deep Learning", "text": "Deep learning uses neural networks with multiple layers"},
            "doc3": {"title": "Natural Language Processing", "text": "NLP helps computers understand human language"},
        }
    
    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing."""
        return {
            "q1": "What is machine learning?",
            "q2": "How does deep learning work?",
        }
    
    def test_evaluate_retrieval_with_jaccard_search(self, sample_corpus, sample_queries):
        """Test EvaluateRetrieval with Jaccard search method."""
        jaccard_search = SearchMethodFactory.create_search_method('jaccard')
        evaluator = EvaluateRetrieval(retriever=jaccard_search, k_values=[1, 3, 5])
        
        results = evaluator.retrieve(sample_corpus, sample_queries)
        
        assert isinstance(results, dict)
        assert len(results) == len(sample_queries)
        for query_id in sample_queries:
            assert query_id in results
            assert isinstance(results[query_id], dict)
    
    def test_evaluate_retrieval_with_bm25_search(self, sample_corpus, sample_queries):
        """Test EvaluateRetrieval with BM25 search method."""
        bm25_search = SearchMethodFactory.create_search_method('bm25')
        evaluator = EvaluateRetrieval(retriever=bm25_search, k_values=[1, 3, 5])
        
        results = evaluator.retrieve(sample_corpus, sample_queries)
        
        assert isinstance(results, dict)
        assert len(results) == len(sample_queries)
        for query_id in sample_queries:
            assert query_id in results
            assert isinstance(results[query_id], dict)
    
    @patch('coir.beir.retrieval.search.llm.QueryExpander')
    def test_evaluate_retrieval_with_llm_expansion(self, mock_expander_class, sample_corpus, sample_queries):
        """Test EvaluateRetrieval with LLM query expansion."""
        # Mock the QueryExpander
        mock_expander = Mock()
        mock_expander.expand_queries.return_value = [
            "What is machine learning and artificial intelligence?",
            "How does deep learning work with neural networks?"
        ]
        mock_expander_class.return_value = mock_expander
        
        jaccard_search = SearchMethodFactory.create_search_method('jaccard')
        evaluator = EvaluateRetrieval(retriever=jaccard_search, k_values=[1, 3, 5])
        
        results = evaluator.retrieve(
            sample_corpus, 
            sample_queries, 
            use_llm=True, 
            llm_name='test_model',
            prompt="Expand this query:"
        )
        
        assert isinstance(results, dict)
        mock_expander_class.assert_called_once_with('test_model', 'Expand this query:')
        mock_expander.expand_queries.assert_called_once()
    
    def test_evaluate_retrieval_llm_import_error_handling(self, sample_corpus, sample_queries):
        """Test graceful handling when LLM components are not available."""
        jaccard_search = SearchMethodFactory.create_search_method('jaccard')
        evaluator = EvaluateRetrieval(retriever=jaccard_search, k_values=[1, 3, 5])
        
        # This should work even if LLM components fail to import
        with patch('coir.beir.retrieval.evaluation.logger') as mock_logger:
            results = evaluator.retrieve(
                sample_corpus, 
                sample_queries, 
                use_llm=True, 
                llm_name='test_model'
            )
            
            assert isinstance(results, dict)
            # Should log a warning about LLM not being available
            mock_logger.warning.assert_called()
    
    def test_rerank_functionality(self, sample_corpus, sample_queries):
        """Test the rerank functionality."""
        jaccard_search = SearchMethodFactory.create_search_method('jaccard')
        evaluator = EvaluateRetrieval(retriever=jaccard_search, k_values=[1, 3, 5])
        
        # Get initial results
        initial_results = evaluator.retrieve(sample_corpus, sample_queries)
        
        # Test reranking
        reranked_results = evaluator.rerank(
            sample_corpus, 
            sample_queries, 
            initial_results, 
            top_k=2
        )
        
        assert isinstance(reranked_results, dict)
        assert len(reranked_results) == len(sample_queries)
    
    @patch('coir.beir.retrieval.search.llm.QueryExpander')
    def test_rerank_with_llm(self, mock_expander_class, sample_corpus, sample_queries):
        """Test reranking with LLM query expansion."""
        mock_expander = Mock()
        mock_expander.expand_queries.return_value = [
            "What is machine learning expanded?",
            "How does deep learning work expanded?"
        ]
        mock_expander_class.return_value = mock_expander
        
        jaccard_search = SearchMethodFactory.create_search_method('jaccard')
        evaluator = EvaluateRetrieval(retriever=jaccard_search, k_values=[1, 3, 5])
        
        initial_results = evaluator.retrieve(sample_corpus, sample_queries)
        
        reranked_results = evaluator.rerank(
            sample_corpus, 
            sample_queries, 
            initial_results, 
            top_k=2,
            use_llm=True,
            llm_name='test_model'
        )
        
        assert isinstance(reranked_results, dict)
        mock_expander_class.assert_called_once_with('test_model', '')  # Empty prompt for reranking


class TestBackwardCompatibility:
    """Test that existing code continues to work unchanged."""
    
    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus for testing."""
        return {
            "doc1": {"title": "Test", "text": "This is a test document"},
            "doc2": {"title": "Another", "text": "This is another test document"},
        }
    
    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing."""
        return {
            "q1": "test query",
        }
    
    def test_original_interface_still_works(self, sample_corpus, sample_queries):
        """Test that the original EvaluateRetrieval interface still works."""
        # Create a mock retriever that follows the original interface
        mock_retriever = Mock(spec=BaseSearch)
        mock_retriever.search.return_value = {"q1": {"doc1": 0.8, "doc2": 0.6}}
        
        # Test original constructor signature
        evaluator = EvaluateRetrieval(retriever=mock_retriever, k_values=[1, 3, 5])
        
        # Test original retrieve method signature (without LLM parameters)
        results = evaluator.retrieve(sample_corpus, sample_queries)
        
        assert isinstance(results, dict)
        mock_retriever.search.assert_called_once()
    
    def test_original_rerank_interface_still_works(self, sample_corpus, sample_queries):
        """Test that the original rerank interface still works."""
        mock_retriever = Mock(spec=BaseSearch)
        mock_retriever.search.return_value = {"q1": {"doc1": 0.9}}
        
        evaluator = EvaluateRetrieval(retriever=mock_retriever, k_values=[1, 3, 5])
        initial_results = {"q1": {"doc1": 0.8, "doc2": 0.6}}
        
        # Test original rerank method signature (without LLM parameters)
        reranked_results = evaluator.rerank(sample_corpus, sample_queries, initial_results, top_k=1)
        
        assert isinstance(reranked_results, dict)
        mock_retriever.search.assert_called()
    
    def test_default_parameters_work(self, sample_corpus, sample_queries):
        """Test that default parameters maintain backward compatibility."""
        jaccard_search = SearchMethodFactory.create_search_method('jaccard')
        evaluator = EvaluateRetrieval(retriever=jaccard_search)
        
        # Should work with default k_values and score_function
        results = evaluator.retrieve(sample_corpus, sample_queries)
        
        assert isinstance(results, dict)
        assert evaluator.k_values == [1, 3, 5, 10, 100, 1000]  # Default values
        assert evaluator.score_function == "cos_sim"  # Default value


if __name__ == "__main__":
    pytest.main([__file__])