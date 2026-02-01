"""
Comprehensive integration tests for the refactored CoIR architecture

This module tests the complete integration of all components including:
- Search method factory
- All search method implementations
- COIR evaluation orchestrator
- LLM integration (mocked)
- Configuration system
- Result storage and organization
"""

import pytest
import tempfile
import os
import json
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import the components we're testing
from coir.evaluation import COIR
from coir.data_loader import get_tasks
from coir.beir.retrieval.search.factory import SearchMethodFactory
from coir.beir.retrieval.evaluation import EvaluateRetrieval


class TestFullIntegration:
    """Comprehensive integration tests for the complete system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal test dataset
        self.test_tasks = self._create_test_tasks()
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.model_name = "test_model"
        self.mock_model.encode_corpus.return_value = np.random.rand(3, 768)
        self.mock_model.encode_queries.return_value = np.random.rand(2, 768)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_tasks(self):
        """Create minimal test tasks for integration testing"""
        return {
            "test_task": (
                {  # corpus
                    "doc1": {"title": "Machine Learning", "text": "Machine learning is a subset of artificial intelligence"},
                    "doc2": {"title": "Deep Learning", "text": "Deep learning uses neural networks with multiple layers"},
                    "doc3": {"title": "Natural Language Processing", "text": "NLP helps computers understand human language"}
                },
                {  # queries
                    "q1": "What is machine learning?",
                    "q2": "How does deep learning work?"
                },
                {  # qrels
                    "q1": {"doc1": 1, "doc2": 0},
                    "q2": {"doc2": 1, "doc1": 0}
                }
            )
        }
    
    def test_backward_compatibility_dense_search(self):
        """Test that original dense search still works unchanged"""
        # Original interface should work exactly as before
        evaluation = COIR(self.test_tasks, batch_size=32)
        
        # Mock the search method and evaluation components
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
            
            # Run evaluation
            results = evaluation.run(self.mock_model, self.temp_dir)
            
            # Verify results structure matches original format
            assert isinstance(results, dict)
            assert "test_task" in results
            assert "NDCG" in results["test_task"]
            assert "MAP" in results["test_task"]
            assert "Recall" in results["test_task"]
            assert "Precision" in results["test_task"]
            
            # Verify factory was called with correct parameters
            mock_factory.create_search_method.assert_called_once_with(
                "dense", model=self.mock_model, batch_size=32
            )
    
    def test_all_search_methods(self):
        """Test each search method individually"""
        search_configs = [
            {"method": "dense"},
            {"method": "jaccard"},
            {"method": "bm25"},
            {"method": "simple_hybrid"},
            {"method": "advanced_hybrid", "params": {"fusion_method": "rrf"}}
        ]
        
        for config in search_configs:
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
                
                # Create evaluation with specific config
                evaluation = COIR(self.test_tasks, batch_size=32, search_config=config)
                results = evaluation.run(self.mock_model, self.temp_dir)
                
                # Verify each method produces valid results
                assert isinstance(results, dict)
                assert len(results) > 0
                assert "test_task" in results
                
                # Verify correct method was requested
                if config["method"] in ["dense", "simple_hybrid", "advanced_hybrid"]:
                    # These methods need a model
                    mock_factory.create_search_method.assert_called_with(
                        config["method"], model=self.mock_model, batch_size=32,
                        **config.get("params", {})
                    )
                else:
                    # Lexical methods don't need a model
                    mock_factory.create_search_method.assert_called_with(
                        config["method"], batch_size=32, **config.get("params", {})
                    )
    
    @patch('coir.beir.retrieval.evaluation.QueryExpander')
    def test_llm_integration(self, mock_expander_class):
        """Test LLM query expansion (mock LLM calls)"""
        # Setup LLM mock
        mock_expander = Mock()
        mock_expander.expand_queries.return_value = [
            "What is machine learning and artificial intelligence?",
            "How does deep learning work with neural networks?"
        ]
        mock_expander_class.return_value = mock_expander
        
        config = {"method": "dense"}
        evaluation = COIR(self.test_tasks, batch_size=32, search_config=config)
        
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
            
            # Run with LLM enhancement
            results = evaluation.run(
                self.mock_model, self.temp_dir,
                use_llm=True, llm_name="test_llm", prompt="Expand this query:"
            )
            
            assert isinstance(results, dict)
            # Verify LLM parameters were passed to retrieve method
            mock_retriever.retrieve.assert_called_once()
            call_args = mock_retriever.retrieve.call_args
            assert len(call_args[0]) >= 2  # corpus, queries
    
    def test_result_organization(self):
        """Test enhanced result storage and organization"""
        config = {"method": "dense"}
        evaluation = COIR(self.test_tasks, batch_size=32, search_config=config)
        
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
            
            # Run multiple configurations
            evaluation.run(self.mock_model, self.temp_dir)
            evaluation.run(self.mock_model, self.temp_dir, 
                         use_llm=True, llm_name="test_llm", prompt="test prompt")
            
            # Check result file structure
            result_files = os.listdir(self.temp_dir)
            assert len(result_files) > 0
            
            # Verify JSON structure
            result_file = os.path.join(self.temp_dir, "test_task.json")
            assert os.path.exists(result_file)
            
            with open(result_file) as f:
                data = json.load(f)
                assert isinstance(data, dict)
                # Verify multi-dimensional organization
                assert "baseline" in data
                assert "test_llm\ntest prompt" in data
                assert "test_model" in data["baseline"]
                assert "metrics" in data["baseline"]["test_model"]
    
    def test_configuration_validation(self):
        """Test configuration validation throughout the system"""
        # Valid configurations should work
        valid_configs = [
            {"method": "dense"},
            {"method": "bm25"},
            {"method": "advanced_hybrid", "params": {"fusion_method": "rrf"}}
        ]
        
        for config in valid_configs:
            evaluation = COIR(self.test_tasks, batch_size=32, search_config=config)
            assert evaluation.get_search_config() == config
        
        # Invalid configurations should raise errors
        with pytest.raises(ValueError, match="Unknown search method"):
            COIR(self.test_tasks, batch_size=32, search_config={"method": "invalid_method"})
    
    def test_factory_integration(self):
        """Test SearchMethodFactory integration with COIR"""
        # Test that factory methods are available
        methods = SearchMethodFactory.list_available_methods()
        expected_methods = ['dense', 'semantic', 'jaccard', 'bm25', 'simple_hybrid', 'advanced_hybrid']
        
        for method in expected_methods:
            assert method in methods
        
        # Test that COIR can list methods through factory
        coir_methods = COIR.list_available_search_methods()
        assert coir_methods == methods
    
    def test_error_handling(self):
        """Test error handling throughout the integration"""
        evaluation = COIR(self.test_tasks, batch_size=32)
        
        # Test with invalid model (should handle gracefully)
        with patch('coir.evaluation.SearchMethodFactory') as mock_factory:
            mock_factory.create_search_method.side_effect = Exception("Model error")
            
            with pytest.raises(Exception):
                evaluation.run(None, self.temp_dir)
    
    def test_search_config_management(self):
        """Test search configuration management integration"""
        evaluation = COIR(self.test_tasks, batch_size=32)
        
        # Test getting configuration
        config = evaluation.get_search_config()
        assert config == {"method": "dense"}
        
        # Test setting new configuration
        new_config = {"method": "bm25", "params": {"k1": 1.5}}
        evaluation.set_search_config(new_config)
        assert evaluation.get_search_config() == new_config
        
        # Test creating configuration
        created_config = COIR.create_search_config("jaccard", threshold=0.5)
        expected = {"method": "jaccard", "params": {"threshold": 0.5}}
        assert created_config == expected
    
    def test_method_identifier_generation(self):
        """Test method identifier generation for different search types"""
        evaluation = COIR(self.test_tasks, batch_size=32)
        
        # Test dense method
        evaluation.search_config = {"method": "dense"}
        identifier = evaluation._get_method_identifier(self.mock_model)
        assert identifier == "test_model"
        
        # Test lexical method
        evaluation.search_config = {"method": "bm25"}
        identifier = evaluation._get_method_identifier(self.mock_model)
        assert identifier == "bm25"
        
        # Test hybrid method
        evaluation.search_config = {"method": "simple_hybrid"}
        identifier = evaluation._get_method_identifier(self.mock_model)
        assert identifier == "simple_hybrid(test_model)"
    
    def test_config_label_creation(self):
        """Test configuration label creation for result organization"""
        evaluation = COIR(self.test_tasks, batch_size=32)
        
        # Test baseline label
        label = evaluation._create_config_label(False, "", "", False)
        assert label == "baseline"
        
        # Test LLM label
        label = evaluation._create_config_label(True, "llama2", "", False)
        assert label == "llama2"
        
        # Test LLM with prompt
        label = evaluation._create_config_label(True, "llama2", "test prompt", False)
        assert label == "llama2\ntest prompt"
        
        # Test with reranking
        label = evaluation._create_config_label(True, "llama2", "", True)
        assert label == "llama2 + reranking"
        
        # Test full configuration
        label = evaluation._create_config_label(True, "llama2", "test prompt", True)
        assert label == "llama2\ntest prompt + reranking"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])