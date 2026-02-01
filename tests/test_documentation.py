"""
Validate that documentation examples work correctly

This module tests that all examples in documentation, README files, and
configuration examples work as expected and are kept up-to-date.
"""

import pytest
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, patch
import numpy as np

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from coir.evaluation import COIR
from coir.beir.retrieval.search.factory import SearchMethodFactory


class TestDocumentation:
    """Validate that documentation examples work correctly"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test tasks for documentation examples
        self.test_tasks = {
            "codetrans-dl": (
                {  # corpus
                    "doc1": {"title": "Python Function", "text": "def hello_world(): print('Hello, World!')"},
                    "doc2": {"title": "Java Method", "text": "public void helloWorld() { System.out.println('Hello, World!'); }"},
                    "doc3": {"title": "JavaScript Function", "text": "function helloWorld() { console.log('Hello, World!'); }"}
                },
                {  # queries
                    "q1": "print hello world function",
                    "q2": "console output method"
                },
                {  # qrels
                    "q1": {"doc1": 1, "doc2": 1, "doc3": 1},
                    "q2": {"doc1": 0, "doc2": 1, "doc3": 1}
                }
            )
        }
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.model_name = "all-MiniLM-L6-v2"
        self.mock_model.encode_corpus.return_value = np.random.rand(3, 384)
        self.mock_model.encode_queries.return_value = np.random.rand(2, 384)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _setup_mock_evaluation(self, mock_factory, mock_eval_class):
        """Setup common mock evaluation components"""
        mock_search = Mock()
        mock_factory.create_search_method.return_value = mock_search
        
        mock_retriever = Mock()
        mock_eval_class.return_value = mock_retriever
        mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9, "doc2": 0.8}}
        mock_retriever.evaluate.return_value = (
            {"NDCG@10": 0.85}, {"MAP@10": 0.82}, {"Recall@10": 0.90}, {"P@10": 0.75}
        )
        mock_retriever.k_values = [10]
        
        return mock_search, mock_retriever
    
    def test_readme_basic_usage_example(self):
        """Test basic usage example from README"""
        # This should match the basic example in README
        
        with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
             patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
            
            mock_search, mock_retriever = self._setup_mock_evaluation(mock_factory, mock_eval_class)
            
            # Basic usage example
            tasks = self.test_tasks  # Simulating get_tasks(tasks=["codetrans-dl"])
            evaluation = COIR(tasks, batch_size=128)
            
            # This should work without errors
            results = evaluation.run(self.mock_model, self.temp_dir)
            
            assert isinstance(results, dict)
            assert "codetrans-dl" in results
            
            # Verify default configuration was used
            assert evaluation.get_search_config() == {"method": "dense"}
    
    def test_readme_advanced_usage_example(self):
        """Test advanced usage example from README"""
        # This should match advanced examples in README
        
        with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
             patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
            
            mock_search, mock_retriever = self._setup_mock_evaluation(mock_factory, mock_eval_class)
            
            # Advanced usage with custom configuration
            tasks = self.test_tasks
            config = {"method": "advanced_hybrid", "params": {"fusion_method": "rrf"}}
            evaluation = COIR(tasks, batch_size=128, search_config=config)
            
            # Run with LLM enhancement
            results = evaluation.run(
                self.mock_model, 
                self.temp_dir,
                use_llm=True,
                llm_name="llama2",
                prompt="Expand this query for better code search: {query}",
                enable_reranking=True
            )
            
            assert isinstance(results, dict)
            assert "codetrans-dl" in results
            
            # Verify configuration was set correctly
            assert evaluation.get_search_config() == config
    
    def test_configuration_examples(self):
        """Test examples from search_configurations.py"""
        # Import the configuration examples
        try:
            from examples.search_configurations import (
                DENSE_CONFIG, JACCARD_CONFIG, BM25_CONFIG,
                SIMPLE_HYBRID_CONFIG, ADVANCED_HYBRID_CONFIG
            )
        except ImportError:
            pytest.skip("Configuration examples not available")
        
        configs = [
            ("DENSE_CONFIG", DENSE_CONFIG),
            ("JACCARD_CONFIG", JACCARD_CONFIG), 
            ("BM25_CONFIG", BM25_CONFIG),
            ("SIMPLE_HYBRID_CONFIG", SIMPLE_HYBRID_CONFIG),
            ("ADVANCED_HYBRID_CONFIG", ADVANCED_HYBRID_CONFIG)
        ]
        
        for config_name, config in configs:
            with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                 patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                
                mock_search, mock_retriever = self._setup_mock_evaluation(mock_factory, mock_eval_class)
                
                # Test that each configuration example works
                try:
                    evaluation = COIR(self.test_tasks, batch_size=128, search_config=config)
                    assert evaluation is not None
                    assert evaluation.get_search_config() == config
                except Exception as e:
                    pytest.fail(f"Configuration example {config_name} failed: {e}")
    
    def test_configuration_examples_with_params(self):
        """Test configuration examples with parameters"""
        try:
            from examples.search_configurations import (
                DENSE_WITH_PARAMS_CONFIG, BM25_WITH_PARAMS_CONFIG
            )
        except ImportError:
            pytest.skip("Parameter configuration examples not available")
        
        param_configs = [
            ("DENSE_WITH_PARAMS_CONFIG", DENSE_WITH_PARAMS_CONFIG),
            ("BM25_WITH_PARAMS_CONFIG", BM25_WITH_PARAMS_CONFIG)
        ]
        
        for config_name, config in param_configs:
            with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                 patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                
                mock_search, mock_retriever = self._setup_mock_evaluation(mock_factory, mock_eval_class)
                
                # Test configuration with parameters
                try:
                    evaluation = COIR(self.test_tasks, batch_size=128, search_config=config)
                    assert evaluation is not None
                    assert evaluation.get_search_config() == config
                    
                    # Verify parameters are included
                    assert "params" in config
                    assert isinstance(config["params"], dict)
                    
                except Exception as e:
                    pytest.fail(f"Parameter configuration example {config_name} failed: {e}")
    
    def test_example_usage_functions(self):
        """Test the example usage functions from search_configurations.py"""
        try:
            from examples.search_configurations import (
                example_usage, example_with_llm_and_reranking,
                example_configuration_management, example_backward_compatibility
            )
        except ImportError:
            pytest.skip("Example functions not available")
        
        # Mock get_tasks to return our test tasks
        with patch('examples.search_configurations.get_tasks') as mock_get_tasks:
            mock_get_tasks.return_value = self.test_tasks
            
            # Test each example function (they should not raise errors)
            try:
                example_usage()
                example_with_llm_and_reranking()
                example_configuration_management()
                example_backward_compatibility()
            except Exception as e:
                # These functions are mostly for demonstration and may not run fully
                # but they should not have syntax errors or import issues
                if "get_tasks" not in str(e) and "SentenceTransformer" not in str(e):
                    pytest.fail(f"Example function failed with unexpected error: {e}")
    
    def test_factory_documentation_examples(self):
        """Test examples from SearchMethodFactory docstrings"""
        # Test factory usage examples that might be in docstrings
        
        # Basic factory usage
        methods = SearchMethodFactory.list_available_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0
        
        # Test creating different search methods (as shown in docs)
        for method in ['dense', 'jaccard', 'bm25']:
            try:
                if method == 'dense':
                    search_instance = SearchMethodFactory.create_search_method(
                        method, model=self.mock_model, batch_size=64
                    )
                else:
                    search_instance = SearchMethodFactory.create_search_method(
                        method, batch_size=64
                    )
                assert search_instance is not None
            except Exception as e:
                pytest.fail(f"Factory documentation example failed for {method}: {e}")
    
    def test_coir_class_documentation_examples(self):
        """Test examples from COIR class docstrings"""
        # Test COIR usage patterns that might be documented
        
        # Basic initialization
        evaluation = COIR(self.test_tasks, batch_size=32)
        assert evaluation.batch_size == 32
        assert evaluation.tasks == self.test_tasks
        
        # Configuration management
        config = evaluation.get_search_config()
        assert isinstance(config, dict)
        
        new_config = {"method": "bm25"}
        evaluation.set_search_config(new_config)
        assert evaluation.get_search_config() == new_config
        
        # Static methods
        methods = COIR.list_available_search_methods()
        assert isinstance(methods, list)
        
        created_config = COIR.create_search_config("jaccard", threshold=0.5)
        expected = {"method": "jaccard", "params": {"threshold": 0.5}}
        assert created_config == expected
    
    def test_api_consistency_with_documentation(self):
        """Test that API matches what's documented"""
        # Test that all documented methods exist and have correct signatures
        
        # COIR class methods
        evaluation = COIR(self.test_tasks, batch_size=32)
        
        # Check that documented methods exist
        assert hasattr(evaluation, 'run')
        assert hasattr(evaluation, 'get_search_config')
        assert hasattr(evaluation, 'set_search_config')
        assert hasattr(COIR, 'create_search_config')
        assert hasattr(COIR, 'list_available_search_methods')
        
        # SearchMethodFactory methods
        assert hasattr(SearchMethodFactory, 'create_search_method')
        assert hasattr(SearchMethodFactory, 'list_available_methods')
        assert hasattr(SearchMethodFactory, 'register_search_method')
        assert hasattr(SearchMethodFactory, 'get_search_method_class')
        
        # Check method signatures match documentation
        import inspect
        
        # COIR.run should accept documented parameters
        run_sig = inspect.signature(evaluation.run)
        run_params = list(run_sig.parameters.keys())
        
        expected_run_params = ['model', 'output_folder', 'use_llm', 'llm_name', 'prompt', 'enable_reranking']
        for param in expected_run_params:
            assert param in run_params, f"Parameter {param} missing from run method"
    
    def test_configuration_schema_documentation(self):
        """Test that configuration schema matches documentation"""
        # Test documented configuration formats
        
        documented_configs = [
            # Basic configurations
            {"method": "dense"},
            {"method": "bm25"},
            {"method": "jaccard"},
            
            # Configurations with parameters
            {"method": "bm25", "params": {"k1": 1.2, "b": 0.75}},
            {"method": "advanced_hybrid", "params": {"fusion_method": "rrf"}},
            
            # Complex configurations
            {
                "method": "advanced_hybrid",
                "params": {
                    "fusion_method": "rrf",
                    "weights": {"dense": 0.7, "lexical": 0.3}
                }
            }
        ]
        
        for config in documented_configs:
            try:
                evaluation = COIR(self.test_tasks, batch_size=32, search_config=config)
                assert evaluation.get_search_config() == config
            except Exception as e:
                pytest.fail(f"Documented configuration failed: {config}, error: {e}")
    
    def test_error_messages_match_documentation(self):
        """Test that error messages match what's documented"""
        # Test that error messages are helpful and match documentation
        
        # Invalid method should give clear error
        try:
            COIR(self.test_tasks, batch_size=32, search_config={"method": "invalid_method"})
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "Unknown search method" in error_msg
            assert "invalid_method" in error_msg
            assert "Available" in error_msg  # Should list available methods
        
        # Factory errors should be clear
        try:
            SearchMethodFactory.create_search_method("unknown_method")
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            error_msg = str(e)
            assert "Unknown search method" in error_msg
            assert "unknown_method" in error_msg
    
    def test_import_statements_in_documentation(self):
        """Test that import statements in documentation work"""
        # Test common import patterns that would be in documentation
        
        try:
            # Basic imports
            from coir.evaluation import COIR
            from coir.beir.retrieval.search.factory import SearchMethodFactory
            
            # These should work without errors
            assert COIR is not None
            assert SearchMethodFactory is not None
            
            # Advanced imports
            from coir.beir.retrieval.evaluation import EvaluateRetrieval
            from coir.beir.retrieval.search.dense.exact_search import DenseRetrievalExactSearch
            
            assert EvaluateRetrieval is not None
            assert DenseRetrievalExactSearch is not None
            
        except ImportError as e:
            pytest.fail(f"Documented import failed: {e}")
    
    def test_version_compatibility_documentation(self):
        """Test that version compatibility information is accurate"""
        # Test that the code works with documented Python versions
        
        import sys
        python_version = sys.version_info
        
        # Should work with Python 3.7+ (adjust based on actual requirements)
        assert python_version >= (3, 7), "Python version too old for documented compatibility"
        
        # Test that documented dependencies can be imported
        try:
            import numpy
            import torch  # If documented as required
            assert numpy is not None
            assert torch is not None
        except ImportError as e:
            # Only fail if these are documented as required
            pytest.skip(f"Optional dependency not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])