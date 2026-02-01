"""
Test the configuration system thoroughly

This module tests the SearchMethodFactory functionality, configuration validation,
and the overall configuration management system of the refactored CoIR architecture.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from coir.evaluation import COIR
from coir.beir.retrieval.search.factory import SearchMethodFactory
from coir.beir.retrieval.search.base import BaseSearch
from coir.beir.retrieval.search.dense.exact_search import DenseRetrievalExactSearch
from coir.beir.retrieval.search.lexical.jaccard_search import LexicalJaccardSearch
from coir.beir.retrieval.search.lexical.bm25_search import LexicalBM25Search
from coir.beir.retrieval.search.hybrid.simple_hybrid_search import SimpleHybridSearch
from coir.beir.retrieval.search.hybrid.advanced_hybrid_search import AdvancedHybridSearch


class TestConfigurationSystem:
    """Test the configuration system thoroughly"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_tasks = {
            "test_task": (
                {"doc1": {"title": "Test", "text": "Test document"}},
                {"q1": "test query"},
                {"q1": {"doc1": 1}}
            )
        }
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.model_name = "test_model"
        self.mock_model.encode_corpus.return_value = np.random.rand(1, 768)
        self.mock_model.encode_queries.return_value = np.random.rand(1, 768)
    
    def test_search_method_factory(self):
        """Test SearchMethodFactory functionality"""
        # Test all available methods
        methods = SearchMethodFactory.list_available_methods()
        expected_methods = ['dense', 'semantic', 'jaccard', 'bm25', 'simple_hybrid', 'advanced_hybrid']
        
        for method in expected_methods:
            assert method in methods, f"Method {method} not found in available methods"
        
        # Test method creation for each type
        
        # Dense/Semantic methods (require model)
        for method in ['dense', 'semantic']:
            search_instance = SearchMethodFactory.create_search_method(
                method, model=self.mock_model, batch_size=32
            )
            assert isinstance(search_instance, DenseRetrievalExactSearch)
            assert search_instance.model == self.mock_model
            assert search_instance.batch_size == 32
        
        # Lexical methods (don't require model)
        jaccard_search = SearchMethodFactory.create_search_method('jaccard', batch_size=64)
        assert isinstance(jaccard_search, LexicalJaccardSearch)
        assert jaccard_search.batch_size == 64
        
        bm25_search = SearchMethodFactory.create_search_method('bm25', batch_size=128)
        assert isinstance(bm25_search, LexicalBM25Search)
        assert bm25_search.batch_size == 128
        
        # Hybrid methods (require model)
        simple_hybrid = SearchMethodFactory.create_search_method(
            'simple_hybrid', model=self.mock_model, batch_size=32
        )
        assert isinstance(simple_hybrid, SimpleHybridSearch)
        
        advanced_hybrid = SearchMethodFactory.create_search_method(
            'advanced_hybrid', model=self.mock_model, batch_size=32
        )
        assert isinstance(advanced_hybrid, AdvancedHybridSearch)
    
    def test_factory_error_handling(self):
        """Test factory error handling"""
        # Unknown method should raise ValueError
        with pytest.raises(ValueError, match="Unknown search method: unknown_method"):
            SearchMethodFactory.create_search_method('unknown_method')
        
        # Missing required parameters should raise appropriate errors
        with pytest.raises(TypeError):
            SearchMethodFactory.create_search_method('dense')  # Missing model
        
        with pytest.raises(TypeError):
            SearchMethodFactory.create_search_method('simple_hybrid')  # Missing model
    
    def test_factory_registration(self):
        """Test registering new search methods"""
        # Create a custom search method
        class CustomSearch(BaseSearch):
            def __init__(self, custom_param=None, **kwargs):
                super().__init__(**kwargs)
                self.custom_param = custom_param
            
            def search(self, corpus, queries, top_k, **kwargs):
                return {"q1": {"doc1": 0.5}}
        
        # Register the custom method
        SearchMethodFactory.register_search_method('custom', CustomSearch)
        
        # Verify it's available
        assert 'custom' in SearchMethodFactory.list_available_methods()
        
        # Test creation
        custom_search = SearchMethodFactory.create_search_method('custom', custom_param="test")
        assert isinstance(custom_search, CustomSearch)
        assert custom_search.custom_param == "test"
        
        # Test getting class without instantiation
        custom_class = SearchMethodFactory.get_search_method_class('custom')
        assert custom_class == CustomSearch
    
    def test_factory_registration_validation(self):
        """Test that factory validates registered classes"""
        # Non-BaseSearch class should raise error
        class InvalidSearch:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseSearch"):
            SearchMethodFactory.register_search_method('invalid', InvalidSearch)
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Valid configurations should work
        valid_configs = [
            {"method": "dense"},
            {"method": "bm25"},
            {"method": "jaccard"},
            {"method": "simple_hybrid"},
            {"method": "advanced_hybrid", "params": {"fusion_method": "rrf"}}
        ]
        
        for config in valid_configs:
            evaluation = COIR(self.test_tasks, batch_size=32, search_config=config)
            assert evaluation.get_search_config() == config
        
        # Invalid configurations should raise errors
        invalid_configs = [
            {"method": "invalid_method"},
            {"method": "nonexistent"},
            {"invalid_key": "dense"}  # Missing method key
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                COIR(self.test_tasks, batch_size=32, search_config=config)
    
    def test_configuration_parameter_passing(self):
        """Test that configuration parameters are passed correctly"""
        # Test BM25 with custom parameters
        bm25_config = {
            "method": "bm25",
            "params": {"k1": 1.5, "b": 0.8}
        }
        
        with patch.object(SearchMethodFactory, 'create_search_method') as mock_create:
            mock_search = Mock()
            mock_create.return_value = mock_search
            
            evaluation = COIR(self.test_tasks, batch_size=32, search_config=bm25_config)
            
            with patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                mock_retriever = Mock()
                mock_eval_class.return_value = mock_retriever
                mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9}}
                mock_retriever.evaluate.return_value = (
                    {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
                )
                mock_retriever.k_values = [10]
                
                evaluation.run(self.mock_model, "/tmp")
                
                # Verify parameters were passed
                mock_create.assert_called_once_with(
                    "bm25", batch_size=32, k1=1.5, b=0.8
                )
    
    def test_configuration_with_additional_kwargs(self):
        """Test configuration with additional keyword arguments"""
        # Test that additional kwargs are passed through
        config = {"method": "dense"}
        
        with patch.object(SearchMethodFactory, 'create_search_method') as mock_create:
            mock_search = Mock()
            mock_create.return_value = mock_search
            
            evaluation = COIR(self.test_tasks, batch_size=32, search_config=config)
            
            with patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                mock_retriever = Mock()
                mock_eval_class.return_value = mock_retriever
                mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9}}
                mock_retriever.evaluate.return_value = (
                    {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
                )
                mock_retriever.k_values = [10]
                
                # Run with additional kwargs
                evaluation.run(
                    self.mock_model, "/tmp",
                    custom_param="test_value",
                    another_param=42
                )
                
                # Verify all parameters were passed
                mock_create.assert_called_once_with(
                    "dense", 
                    model=self.mock_model, 
                    batch_size=32,
                    custom_param="test_value",
                    another_param=42
                )
    
    def test_coir_configuration_management(self):
        """Test COIR configuration management methods"""
        evaluation = COIR(self.test_tasks, batch_size=32)
        
        # Test getting configuration
        config = evaluation.get_search_config()
        assert config == {"method": "dense"}  # Default
        
        # Test setting new configuration
        new_config = {"method": "bm25", "params": {"k1": 1.5}}
        evaluation.set_search_config(new_config)
        assert evaluation.get_search_config() == new_config
        
        # Test that setting invalid config raises error
        with pytest.raises(ValueError):
            evaluation.set_search_config({"method": "invalid_method"})
        
        # Test creating configuration
        created_config = COIR.create_search_config("jaccard", threshold=0.5)
        expected = {"method": "jaccard", "params": {"threshold": 0.5}}
        assert created_config == expected
        
        # Test listing available methods
        methods = COIR.list_available_search_methods()
        assert isinstance(methods, list)
        assert "dense" in methods
        assert "bm25" in methods
        assert "jaccard" in methods
    
    def test_configuration_immutability(self):
        """Test that configuration objects are properly isolated"""
        config = {"method": "dense", "params": {"test": "value"}}
        evaluation = COIR(self.test_tasks, batch_size=32, search_config=config)
        
        # Get configuration copy
        retrieved_config = evaluation.get_search_config()
        
        # Modify the retrieved copy
        retrieved_config["method"] = "modified"
        retrieved_config["params"]["test"] = "modified"
        
        # Original should be unchanged
        original_config = evaluation.get_search_config()
        assert original_config["method"] == "dense"
        assert original_config["params"]["test"] == "value"
    
    def test_complex_configuration_scenarios(self):
        """Test complex configuration scenarios"""
        # Test advanced hybrid with multiple parameters
        complex_config = {
            "method": "advanced_hybrid",
            "params": {
                "fusion_method": "rrf",
                "weights": {"dense": 0.7, "lexical": 0.3},
                "rerank_model": "BAAI/bge-reranker-base",
                "top_k_fusion": 100
            }
        }
        
        evaluation = COIR(self.test_tasks, batch_size=32, search_config=complex_config)
        assert evaluation.get_search_config() == complex_config
        
        with patch.object(SearchMethodFactory, 'create_search_method') as mock_create:
            mock_search = Mock()
            mock_create.return_value = mock_search
            
            with patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                mock_retriever = Mock()
                mock_eval_class.return_value = mock_retriever
                mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9}}
                mock_retriever.evaluate.return_value = (
                    {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
                )
                mock_retriever.k_values = [10]
                
                evaluation.run(self.mock_model, "/tmp")
                
                # Verify all complex parameters were passed
                mock_create.assert_called_once_with(
                    "advanced_hybrid",
                    model=self.mock_model,
                    batch_size=32,
                    fusion_method="rrf",
                    weights={"dense": 0.7, "lexical": 0.3},
                    rerank_model="BAAI/bge-reranker-base",
                    top_k_fusion=100
                )
    
    def test_configuration_edge_cases(self):
        """Test configuration edge cases"""
        # Empty params should work
        config_empty_params = {"method": "dense", "params": {}}
        evaluation = COIR(self.test_tasks, batch_size=32, search_config=config_empty_params)
        assert evaluation.get_search_config() == config_empty_params
        
        # None params should be handled
        config_none_params = {"method": "dense", "params": None}
        with pytest.raises((TypeError, ValueError)):
            COIR(self.test_tasks, batch_size=32, search_config=config_none_params)
        
        # Missing params key should work (defaults to empty)
        config_no_params = {"method": "dense"}
        evaluation = COIR(self.test_tasks, batch_size=32, search_config=config_no_params)
        assert evaluation.get_search_config() == config_no_params
    
    def test_configuration_validation_integration(self):
        """Test configuration validation integration with factory"""
        # Test that COIR validation aligns with factory capabilities
        factory_methods = SearchMethodFactory.list_available_methods()
        
        for method in factory_methods:
            config = {"method": method}
            
            # Should not raise validation error
            try:
                evaluation = COIR(self.test_tasks, batch_size=32, search_config=config)
                assert evaluation.get_search_config() == config
            except ValueError as e:
                pytest.fail(f"Valid method {method} failed validation: {e}")
        
        # Test that unknown methods are rejected by both
        unknown_method = "definitely_unknown_method"
        
        # Should fail in COIR validation
        with pytest.raises(ValueError):
            COIR(self.test_tasks, batch_size=32, search_config={"method": unknown_method})
        
        # Should also fail in factory
        with pytest.raises(ValueError):
            SearchMethodFactory.create_search_method(unknown_method)
    
    def test_configuration_serialization_compatibility(self):
        """Test that configurations can be serialized/deserialized"""
        import json
        
        configs = [
            {"method": "dense"},
            {"method": "bm25", "params": {"k1": 1.2, "b": 0.75}},
            {"method": "advanced_hybrid", "params": {"fusion_method": "rrf", "weights": {"dense": 0.6, "lexical": 0.4}}}
        ]
        
        for config in configs:
            # Test JSON serialization
            json_str = json.dumps(config)
            deserialized_config = json.loads(json_str)
            
            # Should work with deserialized config
            evaluation = COIR(self.test_tasks, batch_size=32, search_config=deserialized_config)
            assert evaluation.get_search_config() == config
    
    def test_dynamic_configuration_updates(self):
        """Test dynamic configuration updates during runtime"""
        evaluation = COIR(self.test_tasks, batch_size=32)
        
        # Start with default
        assert evaluation.get_search_config() == {"method": "dense"}
        
        # Update to BM25
        bm25_config = {"method": "bm25", "params": {"k1": 1.5}}
        evaluation.set_search_config(bm25_config)
        assert evaluation.get_search_config() == bm25_config
        
        # Update to hybrid
        hybrid_config = {"method": "simple_hybrid"}
        evaluation.set_search_config(hybrid_config)
        assert evaluation.get_search_config() == hybrid_config
        
        # Each update should be validated
        with pytest.raises(ValueError):
            evaluation.set_search_config({"method": "invalid"})
        
        # Previous valid config should be preserved after failed update
        assert evaluation.get_search_config() == hybrid_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])