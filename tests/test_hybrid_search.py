import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from typing import Dict

# Import the hybrid search classes
from coir.beir.retrieval.search.hybrid import SimpleHybridSearch, AdvancedHybridSearch, FusionStrategies


class TestFusionStrategies(unittest.TestCase):
    """Test cases for fusion strategy methods."""
    
    def setUp(self):
        """Set up test data."""
        self.semantic_results = {
            'q1': {'doc1': 0.9, 'doc2': 0.8, 'doc3': 0.7},
            'q2': {'doc4': 0.85, 'doc5': 0.75}
        }
        self.lexical_results = {
            'q1': {'doc2': 0.6, 'doc3': 0.5, 'doc4': 0.4},
            'q2': {'doc4': 0.7, 'doc6': 0.6}
        }
        self.top_k = 3
    
    def test_reciprocal_rank_fusion(self):
        """Test RRF fusion strategy."""
        results = FusionStrategies.reciprocal_rank_fusion(
            self.semantic_results, self.lexical_results, self.top_k, k=60
        )
        
        # Check that results are returned for all queries
        self.assertIn('q1', results)
        self.assertIn('q2', results)
        
        # Check that RRF scores are calculated (should be positive)
        for qid in results:
            for doc_id, score in results[qid].items():
                self.assertGreater(score, 0)
        
        # Check that top_k limit is respected
        for qid in results:
            self.assertLessEqual(len(results[qid]), self.top_k)
    
    def test_weighted_score_fusion(self):
        """Test weighted score fusion strategy."""
        results = FusionStrategies.weighted_score_fusion(
            self.semantic_results, self.lexical_results, self.top_k, alpha=0.7
        )
        
        # Check that results are returned for all queries
        self.assertIn('q1', results)
        self.assertIn('q2', results)
        
        # Check that scores are in [0, 1] range (normalized)
        for qid in results:
            for doc_id, score in results[qid].items():
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 1)
        
        # Check that top_k limit is respected
        for qid in results:
            self.assertLessEqual(len(results[qid]), self.top_k)
    
    def test_score_interpolation(self):
        """Test score interpolation fusion strategy."""
        results = FusionStrategies.score_interpolation(
            self.semantic_results, self.lexical_results, self.top_k, alpha=0.5
        )
        
        # Check that results are returned for all queries
        self.assertIn('q1', results)
        self.assertIn('q2', results)
        
        # Check that top_k limit is respected
        for qid in results:
            self.assertLessEqual(len(results[qid]), self.top_k)
    
    def test_combMNZ_fusion(self):
        """Test CombMNZ fusion strategy."""
        results = FusionStrategies.combMNZ_fusion(
            self.semantic_results, self.lexical_results, self.top_k
        )
        
        # Check that results are returned for all queries
        self.assertIn('q1', results)
        self.assertIn('q2', results)
        
        # Check that top_k limit is respected
        for qid in results:
            self.assertLessEqual(len(results[qid]), self.top_k)
    
    def test_empty_results(self):
        """Test fusion strategies with empty results."""
        empty_semantic = {'q1': {}}
        empty_lexical = {'q1': {}}
        
        # Test that empty results don't cause errors
        for fusion_func in [
            FusionStrategies.reciprocal_rank_fusion,
            FusionStrategies.weighted_score_fusion,
            FusionStrategies.score_interpolation,
            FusionStrategies.combMNZ_fusion
        ]:
            results = fusion_func(empty_semantic, empty_lexical, self.top_k)
            self.assertIn('q1', results)
            self.assertEqual(len(results['q1']), 0)


class TestSimpleHybridSearch(unittest.TestCase):
    """Test cases for SimpleHybridSearch class."""
    
    def setUp(self):
        """Set up test data and mocks."""
        # Mock model
        self.mock_model = Mock()
        
        # Sample corpus and queries
        self.corpus = {
            'doc1': {'title': 'Title 1', 'text': 'This is document one'},
            'doc2': {'title': 'Title 2', 'text': 'This is document two'},
            'doc3': {'title': 'Title 3', 'text': 'This is document three'}
        }
        self.queries = {
            'q1': 'document one',
            'q2': 'document two'
        }
        self.top_k = 2
    
    @patch('coir.beir.retrieval.search.hybrid.simple_hybrid_search.DenseRetrievalExactSearch')
    @patch('coir.beir.retrieval.search.hybrid.simple_hybrid_search.LexicalJaccardSearch')
    def test_simple_hybrid_search_initialization(self, mock_jaccard, mock_dense):
        """Test SimpleHybridSearch initialization."""
        search = SimpleHybridSearch(self.mock_model, batch_size=64)
        
        # Check that both search methods are initialized
        mock_dense.assert_called_once_with(self.mock_model, 64)
        mock_jaccard.assert_called_once_with(64)
        
        self.assertEqual(search.results, {})
    
    @patch('coir.beir.retrieval.search.hybrid.simple_hybrid_search.DenseRetrievalExactSearch')
    @patch('coir.beir.retrieval.search.hybrid.simple_hybrid_search.LexicalJaccardSearch')
    def test_simple_hybrid_search_execution(self, mock_jaccard_class, mock_dense_class):
        """Test SimpleHybridSearch search execution."""
        # Mock the search methods
        mock_dense = Mock()
        mock_jaccard = Mock()
        mock_dense_class.return_value = mock_dense
        mock_jaccard_class.return_value = mock_jaccard
        
        # Mock search results
        semantic_results = {
            'q1': {'doc1': 0.9, 'doc2': 0.8},
            'q2': {'doc2': 0.85, 'doc3': 0.75}
        }
        lexical_results = {
            'q1': {'doc2': 0.6, 'doc3': 0.5},
            'q2': {'doc1': 0.7, 'doc3': 0.65}
        }
        
        mock_dense.search.return_value = semantic_results
        mock_jaccard.search.return_value = lexical_results
        
        # Create and run search
        search = SimpleHybridSearch(self.mock_model)
        results = search.search(self.corpus, self.queries, self.top_k)
        
        # Check that both search methods were called
        mock_dense.search.assert_called_once()
        mock_jaccard.search.assert_called_once()
        
        # Check that results are returned for all queries
        self.assertIn('q1', results)
        self.assertIn('q2', results)
        
        # Check that top_k limit is respected
        for qid in results:
            self.assertLessEqual(len(results[qid]), self.top_k)


class TestAdvancedHybridSearch(unittest.TestCase):
    """Test cases for AdvancedHybridSearch class."""
    
    def setUp(self):
        """Set up test data and mocks."""
        # Mock model
        self.mock_model = Mock()
        
        # Sample corpus and queries
        self.corpus = {
            'doc1': {'title': 'Title 1', 'text': 'This is document one'},
            'doc2': {'title': 'Title 2', 'text': 'This is document two'},
            'doc3': {'title': 'Title 3', 'text': 'This is document three'}
        }
        self.queries = {
            'q1': 'document one',
            'q2': 'document two'
        }
        self.top_k = 2
    
    @patch('coir.beir.retrieval.search.hybrid.advanced_hybrid_search.DenseRetrievalExactSearch')
    @patch('coir.beir.retrieval.search.hybrid.advanced_hybrid_search.LexicalBM25Search')
    def test_advanced_hybrid_search_initialization(self, mock_bm25, mock_dense):
        """Test AdvancedHybridSearch initialization."""
        search = AdvancedHybridSearch(
            self.mock_model, 
            batch_size=64, 
            fusion_method='rrf',
            use_reranking=False
        )
        
        # Check that both search methods are initialized
        mock_dense.assert_called_once_with(self.mock_model, 64)
        mock_bm25.assert_called_once_with(64)
        
        self.assertEqual(search.fusion_method, 'rrf')
        self.assertEqual(search.use_reranking, False)
        self.assertEqual(search.results, {})
    
    @patch('coir.beir.retrieval.search.hybrid.advanced_hybrid_search.DenseRetrievalExactSearch')
    @patch('coir.beir.retrieval.search.hybrid.advanced_hybrid_search.LexicalBM25Search')
    @patch('coir.beir.retrieval.search.hybrid.advanced_hybrid_search.FusionStrategies')
    def test_advanced_hybrid_search_rrf_fusion(self, mock_fusion, mock_bm25_class, mock_dense_class):
        """Test AdvancedHybridSearch with RRF fusion."""
        # Mock the search methods
        mock_dense = Mock()
        mock_bm25 = Mock()
        mock_dense_class.return_value = mock_dense
        mock_bm25_class.return_value = mock_bm25
        
        # Mock search results
        semantic_results = {
            'q1': {'doc1': 0.9, 'doc2': 0.8},
            'q2': {'doc2': 0.85, 'doc3': 0.75}
        }
        lexical_results = {
            'q1': {'doc2': 0.6, 'doc3': 0.5},
            'q2': {'doc1': 0.7, 'doc3': 0.65}
        }
        fused_results = {
            'q1': {'doc1': 0.5, 'doc2': 0.4},
            'q2': {'doc2': 0.6, 'doc3': 0.5}
        }
        
        mock_dense.search.return_value = semantic_results
        mock_bm25.search.return_value = lexical_results
        mock_fusion.reciprocal_rank_fusion.return_value = fused_results
        
        # Create and run search
        search = AdvancedHybridSearch(self.mock_model, fusion_method='rrf', use_reranking=False)
        results = search.search(self.corpus, self.queries, self.top_k)
        
        # Check that both search methods were called
        mock_dense.search.assert_called_once()
        mock_bm25.search.assert_called_once()
        
        # Check that RRF fusion was called
        mock_fusion.reciprocal_rank_fusion.assert_called_once()
        
        # Check results
        self.assertEqual(results, fused_results)
    
    @patch('coir.beir.retrieval.search.hybrid.advanced_hybrid_search.DenseRetrievalExactSearch')
    @patch('coir.beir.retrieval.search.hybrid.advanced_hybrid_search.LexicalBM25Search')
    def test_fusion_method_setter(self, mock_bm25, mock_dense):
        """Test fusion method setter and getter."""
        search = AdvancedHybridSearch(self.mock_model, use_reranking=False)
        
        # Test setting valid fusion method
        search.set_fusion_method('weighted')
        self.assertEqual(search.get_fusion_method(), 'weighted')
        
        # Test setting invalid fusion method
        with self.assertRaises(ValueError):
            search.set_fusion_method('invalid_method')
    
    @patch('coir.beir.retrieval.search.hybrid.advanced_hybrid_search.DenseRetrievalExactSearch')
    @patch('coir.beir.retrieval.search.hybrid.advanced_hybrid_search.LexicalBM25Search')
    def test_reranking_control(self, mock_bm25, mock_dense):
        """Test reranking enable/disable functionality."""
        search = AdvancedHybridSearch(self.mock_model, use_reranking=False)
        
        # Initially disabled
        self.assertFalse(search.is_reranking_enabled())
        
        # Test disable (should not change anything)
        search.disable_reranking()
        self.assertFalse(search.is_reranking_enabled())
    
    @patch('coir.beir.retrieval.search.hybrid.advanced_hybrid_search.DenseRetrievalExactSearch')
    @patch('coir.beir.retrieval.search.hybrid.advanced_hybrid_search.LexicalBM25Search')
    def test_invalid_fusion_method_in_search(self, mock_bm25_class, mock_dense_class):
        """Test search with invalid fusion method."""
        # Mock the search methods
        mock_dense = Mock()
        mock_bm25 = Mock()
        mock_dense_class.return_value = mock_dense
        mock_bm25_class.return_value = mock_bm25
        
        mock_dense.search.return_value = {'q1': {}}
        mock_bm25.search.return_value = {'q1': {}}
        
        # Create search with invalid fusion method
        search = AdvancedHybridSearch(self.mock_model, use_reranking=False)
        search.fusion_method = 'invalid'
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            search.search(self.corpus, self.queries, self.top_k)


class TestHybridSearchIntegration(unittest.TestCase):
    """Integration tests for hybrid search functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.corpus = {
            'doc1': {'title': 'Machine Learning', 'text': 'Machine learning is a subset of artificial intelligence'},
            'doc2': {'title': 'Deep Learning', 'text': 'Deep learning uses neural networks with multiple layers'},
            'doc3': {'title': 'Natural Language Processing', 'text': 'NLP helps computers understand human language'}
        }
        self.queries = {
            'q1': 'artificial intelligence machine learning',
            'q2': 'neural networks deep learning'
        }
        self.top_k = 2
    
    def test_fusion_strategies_consistency(self):
        """Test that all fusion strategies return consistent results."""
        semantic_results = {
            'q1': {'doc1': 0.9, 'doc2': 0.7, 'doc3': 0.5},
            'q2': {'doc2': 0.8, 'doc3': 0.6, 'doc1': 0.4}
        }
        lexical_results = {
            'q1': {'doc1': 0.6, 'doc3': 0.8, 'doc2': 0.4},
            'q2': {'doc2': 0.7, 'doc1': 0.5, 'doc3': 0.3}
        }
        
        # Test all fusion strategies
        strategies = [
            ('rrf', FusionStrategies.reciprocal_rank_fusion),
            ('weighted', FusionStrategies.weighted_score_fusion),
            ('interpolation', FusionStrategies.score_interpolation),
            ('combMNZ', FusionStrategies.combMNZ_fusion)
        ]
        
        for strategy_name, strategy_func in strategies:
            if strategy_name == 'rrf':
                results = strategy_func(semantic_results, lexical_results, self.top_k, 60)
            elif strategy_name in ['weighted', 'interpolation']:
                results = strategy_func(semantic_results, lexical_results, self.top_k, 0.5)
            else:  # combMNZ
                results = strategy_func(semantic_results, lexical_results, self.top_k)
            
            # Check basic consistency
            self.assertIn('q1', results)
            self.assertIn('q2', results)
            
            for qid in results:
                self.assertLessEqual(len(results[qid]), self.top_k)
                for doc_id, score in results[qid].items():
                    self.assertIsInstance(score, (int, float))


if __name__ == '__main__':
    unittest.main()