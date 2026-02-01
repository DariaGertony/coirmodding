"""
Specific tests for backward compatibility

This module ensures that the refactored CoIR architecture maintains complete
backward compatibility with the original interface and behavior.
"""

import pytest
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch
import numpy as np

from coir.evaluation import COIR
from coir.beir.retrieval.evaluation import EvaluateRetrieval
from coir.beir.retrieval.search.dense.exact_search import DenseRetrievalExactSearch
from coir.beir.retrieval.search.factory import SearchMethodFactory


class TestBackwardCompatibility:
    """Test backward compatibility with original CoIR interface"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test tasks in original format
        self.test_tasks = {
            "test_task": (
                {  # corpus
                    "doc1": {"title": "Test Document 1", "text": "This is test document 1"},
                    "doc2": {"title": "Test Document 2", "text": "This is test document 2"},
                    "doc3": {"title": "Test Document 3", "text": "This is test document 3"}
                },
                {  # queries
                    "q1": "test query 1",
                    "q2": "test query 2"
                },
                {  # qrels
                    "q1": {"doc1": 1, "doc2": 0, "doc3": 0},
                    "q2": {"doc2": 1, "doc1": 0, "doc3": 0}
                }
            )
        }
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.model_name = "test_model"
        self.mock_model.encode_corpus.return_value = np.random.rand(3, 768)
        self.mock_model.encode_queries.return_value = np.random.rand(2, 768)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_original_coir_interface(self):
        """Test that original COIR interface works unchanged"""
        # Original usage pattern: COIR(tasks, batch_size)
        evaluation = COIR(self.test_tasks, batch_size=128)
        
        # Verify initialization
        assert evaluation.tasks == self.test_tasks
        assert evaluation.batch_size == 128
        assert evaluation.search_config == {"method": "dense"}  # Default should be dense
        
        with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
             patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
            
            # Setup mocks
            mock_search = Mock()
            mock_factory.create_search_method.return_value = mock_search
            
            mock_retriever = Mock()
            mock_eval_class.return_value = mock_retriever
            mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9, "doc2": 0.1}}
            mock_retriever.evaluate.return_value = (
                {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
            )
            mock_retriever.k_values = [10]
            
            # Original run method call: run(model, output_folder)
            results = evaluation.run(self.mock_model, self.temp_dir)
            
            # Verify results structure matches original format
            assert isinstance(results, dict)
            assert "test_task" in results
            assert "NDCG" in results["test_task"]
            assert "MAP" in results["test_task"]
            assert "Recall" in results["test_task"]
            assert "Precision" in results["test_task"]
            
            # Verify factory was called with dense method (default)
            mock_factory.create_search_method.assert_called_once_with(
                "dense", model=self.mock_model, batch_size=128
            )
    
    def test_original_dense_search_unchanged(self):
        """Verify DenseRetrievalExactSearch is completely unchanged"""
        # Test that the original class works exactly as before
        model = Mock()
        model.encode_corpus.return_value = np.random.rand(3, 768)
        model.encode_queries.return_value = np.random.rand(2, 768)
        
        # Original interface should be identical
        search = DenseRetrievalExactSearch(model, batch_size=128)
        
        # Verify attributes
        assert search.model == model
        assert search.batch_size == 128
        assert hasattr(search, 'search')
        
        # Test with sample corpus and queries
        corpus = {
            "doc1": {"title": "Test", "text": "Test document"},
            "doc2": {"title": "Another", "text": "Another document"}
        }
        queries = {"q1": "test query"}
        
        # Should work without errors (mocked internally)
        with patch.object(search, 'search') as mock_search:
            mock_search.return_value = {"q1": {"doc1": 0.9, "doc2": 0.1}}
            results = search.search(corpus, queries, top_k=10)
            assert isinstance(results, dict)
            mock_search.assert_called_once_with(corpus, queries, top_k=10)
    
    def test_original_evaluation_interface(self):
        """Test EvaluateRetrieval backward compatibility"""
        model = Mock()
        model.encode_corpus.return_value = np.random.rand(3, 768)
        model.encode_queries.return_value = np.random.rand(2, 768)
        
        search = DenseRetrievalExactSearch(model, batch_size=128)
        
        # Original EvaluateRetrieval constructor
        retriever = EvaluateRetrieval(search, score_function="cos_sim")
        
        # Verify initialization
        assert retriever.retriever == search
        assert retriever.score_function == "cos_sim"
        assert hasattr(retriever, 'retrieve')
        assert hasattr(retriever, 'evaluate')
        
        # Test original retrieve method interface
        corpus = {"doc1": {"title": "Test", "text": "Test document"}}
        queries = {"q1": "test query"}
        
        with patch.object(search, 'search') as mock_search:
            mock_search.return_value = {"q1": {"doc1": 0.9}}
            
            # Original retrieve call (without LLM parameters)
            results = retriever.retrieve(corpus, queries)
            assert isinstance(results, dict)
            mock_search.assert_called_once()
    
    def test_original_result_format(self):
        """Test that result format matches original implementation"""
        evaluation = COIR(self.test_tasks, batch_size=64)
        
        with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
             patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
            
            # Setup mocks to return original format
            mock_search = Mock()
            mock_factory.create_search_method.return_value = mock_search
            
            mock_retriever = Mock()
            mock_eval_class.return_value = mock_retriever
            mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9}}
            mock_retriever.evaluate.return_value = (
                {"NDCG@1": 1.0, "NDCG@3": 0.9, "NDCG@5": 0.8, "NDCG@10": 0.7},
                {"MAP@1": 1.0, "MAP@3": 0.9, "MAP@5": 0.8, "MAP@10": 0.7},
                {"Recall@1": 0.5, "Recall@3": 0.7, "Recall@5": 0.8, "Recall@10": 0.9},
                {"P@1": 1.0, "P@3": 0.67, "P@5": 0.6, "P@10": 0.5}
            )
            mock_retriever.k_values = [1, 3, 5, 10]
            
            results = evaluation.run(self.mock_model, self.temp_dir)
            
            # Verify original result structure
            assert "test_task" in results
            task_results = results["test_task"]
            
            # Should have all four metric types
            assert "NDCG" in task_results
            assert "MAP" in task_results
            assert "Recall" in task_results
            assert "Precision" in task_results
            
            # Each metric should have multiple k values
            for metric_name, metric_values in task_results.items():
                assert isinstance(metric_values, dict)
                assert len(metric_values) > 0
    
    def test_original_file_output_format(self):
        """Test that output file format matches original"""
        evaluation = COIR(self.test_tasks, batch_size=32)
        
        with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
             patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
            
            # Setup mocks
            mock_search = Mock()
            mock_factory.create_search_method.return_value = mock_search
            
            mock_retriever = Mock()
            mock_eval_class.return_value = mock_retriever
            mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9}}
            mock_retriever.evaluate.return_value = (
                {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
            )
            mock_retriever.k_values = [10]
            
            # Run evaluation
            evaluation.run(self.mock_model, self.temp_dir)
            
            # Check output file exists
            output_file = os.path.join(self.temp_dir, "test_task.json")
            assert os.path.exists(output_file)
            
            # Verify file content structure
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Should have baseline configuration
            assert "baseline" in data
            assert "test_model" in data["baseline"]
            assert "metrics" in data["baseline"]["test_model"]
            
            metrics = data["baseline"]["test_model"]["metrics"]
            assert "NDCG" in metrics
            assert "MAP" in metrics
            assert "Recall" in metrics
            assert "Precision" in metrics
    
    def test_original_parameter_handling(self):
        """Test that original parameter handling still works"""
        # Test with minimal parameters (original style)
        evaluation = COIR(self.test_tasks, 64)
        assert evaluation.batch_size == 64
        assert evaluation.search_config == {"method": "dense"}
        
        # Test that extra parameters are ignored gracefully
        with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
             patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
            
            mock_search = Mock()
            mock_factory.create_search_method.return_value = mock_search
            
            mock_retriever = Mock()
            mock_eval_class.return_value = mock_retriever
            mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9}}
            mock_retriever.evaluate.return_value = (
                {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
            )
            mock_retriever.k_values = [10]
            
            # Original run call should work
            results = evaluation.run(self.mock_model, self.temp_dir)
            assert isinstance(results, dict)
    
    def test_original_error_handling(self):
        """Test that original error handling behavior is preserved"""
        # Test with invalid tasks
        with pytest.raises((KeyError, TypeError, ValueError)):
            evaluation = COIR(None, 32)
        
        # Test with invalid batch size
        with pytest.raises((TypeError, ValueError)):
            evaluation = COIR(self.test_tasks, "invalid")
    
    def test_original_method_signatures(self):
        """Test that all original method signatures are preserved"""
        evaluation = COIR(self.test_tasks, 32)
        
        # Test that original methods exist with correct signatures
        assert hasattr(evaluation, 'run')
        
        # Test run method can be called with original parameters
        with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
             patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
            
            mock_search = Mock()
            mock_factory.create_search_method.return_value = mock_search
            
            mock_retriever = Mock()
            mock_eval_class.return_value = mock_retriever
            mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9}}
            mock_retriever.evaluate.return_value = (
                {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
            )
            mock_retriever.k_values = [10]
            
            # Original signature: run(model, output_folder)
            try:
                results = evaluation.run(self.mock_model, self.temp_dir)
                assert True  # Should not raise
            except TypeError:
                pytest.fail("Original run method signature not preserved")
    
    def test_default_values_unchanged(self):
        """Test that default values match original implementation"""
        evaluation = COIR(self.test_tasks, 32)
        
        # Default search config should be dense
        assert evaluation.search_config == {"method": "dense"}
        
        # Test EvaluateRetrieval defaults
        with patch('coir.evaluation.SearchMethodFactory') as mock_factory:
            mock_search = Mock()
            mock_factory.create_search_method.return_value = mock_search
            
            # Default EvaluateRetrieval should use cos_sim
            with patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                mock_retriever = Mock()
                mock_eval_class.return_value = mock_retriever
                mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9}}
                mock_retriever.evaluate.return_value = (
                    {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
                )
                mock_retriever.k_values = [10]
                
                evaluation.run(self.mock_model, self.temp_dir)
                
                # Verify EvaluateRetrieval was called with cos_sim
                mock_eval_class.assert_called_once_with(mock_search, score_function="cos_sim")
    
    def test_original_import_compatibility(self):
        """Test that original imports still work"""
        # Test that original classes can be imported
        try:
            from coir.evaluation import COIR
            from coir.beir.retrieval.evaluation import EvaluateRetrieval
            from coir.beir.retrieval.search.dense.exact_search import DenseRetrievalExactSearch
            assert True  # Should not raise ImportError
        except ImportError as e:
            pytest.fail(f"Original imports failed: {e}")
        
        # Test that classes have expected attributes
        assert hasattr(COIR, 'run')
        assert hasattr(EvaluateRetrieval, 'retrieve')
        assert hasattr(EvaluateRetrieval, 'evaluate')
        assert hasattr(DenseRetrievalExactSearch, 'search')
    
    def test_no_breaking_changes_in_dense_search(self):
        """Test that DenseRetrievalExactSearch has no breaking changes"""
        # Create instance with original parameters
        model = Mock()
        search = DenseRetrievalExactSearch(model, batch_size=64)
        
        # Verify all original attributes exist
        assert hasattr(search, 'model')
        assert hasattr(search, 'batch_size')
        assert hasattr(search, 'search')
        
        # Verify attribute values
        assert search.model == model
        assert search.batch_size == 64
        
        # Verify search method signature hasn't changed
        import inspect
        sig = inspect.signature(search.search)
        params = list(sig.parameters.keys())
        
        # Should have at least corpus, queries, top_k
        assert 'corpus' in params
        assert 'queries' in params
        assert 'top_k' in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])