"""
Unit tests for the main CoIR evaluation orchestrator.

This module tests the updated COIR class with support for different search methods,
configuration management, and backward compatibility.
"""

import unittest
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from coir.evaluation import COIR


class TestCOIRClass(unittest.TestCase):
    """Test cases for the COIR evaluation class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_tasks = {
            "test_task": (
                {"doc1": "test document 1", "doc2": "test document 2"},  # corpus
                {"q1": "test query 1"},  # queries
                {"q1": {"doc1": 1}}  # qrels
            )
        }
        self.batch_size = 32
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_default_config(self):
        """Test COIR initialization with default configuration"""
        evaluator = COIR(self.test_tasks, self.batch_size)
        
        self.assertEqual(evaluator.tasks, self.test_tasks)
        self.assertEqual(evaluator.batch_size, self.batch_size)
        self.assertEqual(evaluator.search_config, {"method": "dense"})
    
    def test_init_custom_config(self):
        """Test COIR initialization with custom configuration"""
        config = {"method": "bm25", "params": {"k1": 1.5}}
        evaluator = COIR(self.test_tasks, self.batch_size, config)
        
        self.assertEqual(evaluator.search_config, config)
    
    def test_init_invalid_config(self):
        """Test COIR initialization with invalid configuration"""
        config = {"method": "invalid_method"}
        
        with self.assertRaises(ValueError) as context:
            COIR(self.test_tasks, self.batch_size, config)
        
        self.assertIn("Unknown search method", str(context.exception))
    
    def test_validate_search_config(self):
        """Test search configuration validation"""
        evaluator = COIR(self.test_tasks, self.batch_size)
        
        # Valid configuration should not raise
        evaluator.search_config = {"method": "dense"}
        evaluator._validate_search_config()  # Should not raise
        
        # Invalid configuration should raise
        evaluator.search_config = {"method": "invalid"}
        with self.assertRaises(ValueError):
            evaluator._validate_search_config()
    
    def test_get_search_config(self):
        """Test getting search configuration"""
        config = {"method": "jaccard", "params": {"threshold": 0.5}}
        evaluator = COIR(self.test_tasks, self.batch_size, config)
        
        retrieved_config = evaluator.get_search_config()
        self.assertEqual(retrieved_config, config)
        
        # Ensure it's a copy, not the original
        retrieved_config["method"] = "modified"
        self.assertEqual(evaluator.search_config["method"], "jaccard")
    
    def test_set_search_config(self):
        """Test setting search configuration"""
        evaluator = COIR(self.test_tasks, self.batch_size)
        
        new_config = {"method": "bm25"}
        evaluator.set_search_config(new_config)
        
        self.assertEqual(evaluator.search_config, new_config)
    
    def test_set_invalid_search_config(self):
        """Test setting invalid search configuration"""
        evaluator = COIR(self.test_tasks, self.batch_size)
        
        invalid_config = {"method": "invalid_method"}
        with self.assertRaises(ValueError):
            evaluator.set_search_config(invalid_config)
    
    def test_create_search_config(self):
        """Test static method for creating search configuration"""
        config = COIR.create_search_config("bm25", k1=1.2, b=0.75)
        
        expected = {
            "method": "bm25",
            "params": {"k1": 1.2, "b": 0.75}
        }
        self.assertEqual(config, expected)
    
    @patch('coir.evaluation.SearchMethodFactory')
    def test_list_available_search_methods(self, mock_factory):
        """Test listing available search methods"""
        mock_factory.list_available_methods.return_value = ["dense", "bm25", "jaccard"]
        
        methods = COIR.list_available_search_methods()
        self.assertEqual(methods, ["dense", "bm25", "jaccard"])
        mock_factory.list_available_methods.assert_called_once()
    
    def test_create_config_label(self):
        """Test configuration label creation"""
        evaluator = COIR(self.test_tasks, self.batch_size)
        
        # Test baseline label
        label = evaluator._create_config_label(False, "", "", False)
        self.assertEqual(label, "baseline")
        
        # Test LLM label
        label = evaluator._create_config_label(True, "llama2", "", False)
        self.assertEqual(label, "llama2")
        
        # Test LLM with prompt
        label = evaluator._create_config_label(True, "llama2", "test prompt", False)
        self.assertEqual(label, "llama2\ntest prompt")
        
        # Test with reranking
        label = evaluator._create_config_label(True, "llama2", "", True)
        self.assertEqual(label, "llama2 + reranking")
        
        # Test full configuration
        label = evaluator._create_config_label(True, "llama2", "test prompt", True)
        self.assertEqual(label, "llama2\ntest prompt + reranking")
    
    def test_get_method_identifier(self):
        """Test method identifier generation"""
        evaluator = COIR(self.test_tasks, self.batch_size)
        
        # Test dense method
        evaluator.search_config = {"method": "dense"}
        mock_model = Mock()
        mock_model.model_name = "test_model"
        identifier = evaluator._get_method_identifier(mock_model)
        self.assertEqual(identifier, "test_model")
        
        # Test lexical method
        evaluator.search_config = {"method": "bm25"}
        identifier = evaluator._get_method_identifier(mock_model)
        self.assertEqual(identifier, "bm25")
        
        # Test hybrid method
        evaluator.search_config = {"method": "simple_hybrid"}
        identifier = evaluator._get_method_identifier(mock_model)
        self.assertEqual(identifier, "simple_hybrid(test_model)")
        
        # Test model without model_name
        mock_model_no_name = Mock(spec=[])
        evaluator.search_config = {"method": "dense"}
        identifier = evaluator._get_method_identifier(mock_model_no_name)
        self.assertEqual(identifier, "dense_model")
    
    @patch('coir.evaluation.SearchMethodFactory')
    @patch('coir.evaluation.EvaluateRetrieval')
    def test_run_dense_search(self, mock_eval_class, mock_factory):
        """Test running evaluation with dense search"""
        # Setup mocks
        mock_search_method = Mock()
        mock_factory.create_search_method.return_value = mock_search_method
        
        mock_retriever = Mock()
        mock_eval_class.return_value = mock_retriever
        mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9}}
        mock_retriever.evaluate.return_value = (
            {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
        )
        mock_retriever.k_values = [10]
        
        # Run evaluation
        evaluator = COIR(self.test_tasks, self.batch_size, {"method": "dense"})
        mock_model = Mock()
        mock_model.model_name = "test_model"
        
        results = evaluator.run(mock_model, self.temp_dir)
        
        # Verify results
        self.assertIn("test_task", results)
        self.assertIn("NDCG", results["test_task"])
        
        # Verify factory was called correctly
        mock_factory.create_search_method.assert_called_once_with(
            "dense", model=mock_model, batch_size=self.batch_size
        )
    
    @patch('coir.evaluation.SearchMethodFactory')
    @patch('coir.evaluation.EvaluateRetrieval')
    def test_run_lexical_search(self, mock_eval_class, mock_factory):
        """Test running evaluation with lexical search"""
        # Setup mocks
        mock_search_method = Mock()
        mock_factory.create_search_method.return_value = mock_search_method
        
        mock_retriever = Mock()
        mock_eval_class.return_value = mock_retriever
        mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9}}
        mock_retriever.evaluate.return_value = (
            {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
        )
        mock_retriever.k_values = [10]
        
        # Run evaluation
        evaluator = COIR(self.test_tasks, self.batch_size, {"method": "bm25"})
        mock_model = Mock()
        
        results = evaluator.run(mock_model, self.temp_dir)
        
        # Verify results
        self.assertIn("test_task", results)
        
        # Verify factory was called without model for lexical search
        mock_factory.create_search_method.assert_called_once_with(
            "bm25", batch_size=self.batch_size
        )
    
    @patch('coir.evaluation.SearchMethodFactory')
    @patch('coir.evaluation.EvaluateRetrieval')
    def test_run_with_llm_enhancement(self, mock_eval_class, mock_factory):
        """Test running evaluation with LLM enhancement"""
        # Setup mocks
        mock_search_method = Mock()
        mock_factory.create_search_method.return_value = mock_search_method
        
        mock_retriever = Mock()
        mock_eval_class.return_value = mock_retriever
        mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9}}
        mock_retriever.evaluate.return_value = (
            {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
        )
        mock_retriever.k_values = [10]
        
        # Run evaluation with LLM
        evaluator = COIR(self.test_tasks, self.batch_size)
        mock_model = Mock()
        mock_model.model_name = "test_model"
        
        results = evaluator.run(
            mock_model, self.temp_dir,
            use_llm=True, llm_name="llama2", prompt="test prompt"
        )
        
        # Verify LLM parameters were passed to retrieve
        mock_retriever.retrieve.assert_called_once_with(
            self.test_tasks["test_task"][0],  # corpus
            self.test_tasks["test_task"][1],  # queries
            True,  # use_llm
            "llama2",  # llm_name
            "test prompt"  # prompt
        )
    
    def test_save_results(self):
        """Test result saving functionality"""
        evaluator = COIR(self.test_tasks, self.batch_size)
        mock_model = Mock()
        mock_model.model_name = "test_model"
        
        metrics = {
            "NDCG": {"NDCG@10": 0.8},
            "MAP": {"MAP@10": 0.7},
            "Recall": {"Recall@10": 0.6},
            "Precision": {"P@10": 0.5}
        }
        
        # Save results
        evaluator._save_results(
            "test_task", self.temp_dir, metrics, mock_model,
            False, "", "", False
        )
        
        # Verify file was created
        output_file = os.path.join(self.temp_dir, "test_task.json")
        self.assertTrue(os.path.exists(output_file))
        
        # Verify file content
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn("baseline", data)
        self.assertIn("test_model", data["baseline"])
        self.assertEqual(data["baseline"]["test_model"]["metrics"], metrics)
    
    def test_save_results_incremental(self):
        """Test incremental result saving"""
        evaluator = COIR(self.test_tasks, self.batch_size)
        mock_model = Mock()
        mock_model.model_name = "test_model"
        
        metrics1 = {"NDCG": {"NDCG@10": 0.8}}
        metrics2 = {"NDCG": {"NDCG@10": 0.9}}
        
        # Save first result
        evaluator._save_results(
            "test_task", self.temp_dir, metrics1, mock_model,
            False, "", "", False
        )
        
        # Save second result with different configuration
        evaluator._save_results(
            "test_task", self.temp_dir, metrics2, mock_model,
            True, "llama2", "test prompt", False
        )
        
        # Verify both results are saved
        output_file = os.path.join(self.temp_dir, "test_task.json")
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn("baseline", data)
        self.assertIn("llama2\ntest prompt", data)
        self.assertEqual(data["baseline"]["test_model"]["metrics"], metrics1)
        self.assertEqual(data["llama2\ntest prompt"]["test_model"]["metrics"], metrics2)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with original interface"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_tasks = {
            "test_task": (
                {"doc1": "test document 1"},
                {"q1": "test query 1"},
                {"q1": {"doc1": 1}}
            )
        }
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_original_constructor(self):
        """Test that original constructor interface still works"""
        # Original interface: COIR(tasks, batch_size)
        evaluator = COIR(self.test_tasks, 32)
        
        self.assertEqual(evaluator.tasks, self.test_tasks)
        self.assertEqual(evaluator.batch_size, 32)
        self.assertEqual(evaluator.search_config, {"method": "dense"})
    
    @patch('coir.evaluation.SearchMethodFactory')
    @patch('coir.evaluation.EvaluateRetrieval')
    def test_original_run_method(self, mock_eval_class, mock_factory):
        """Test that original run method interface still works"""
        # Setup mocks
        mock_search_method = Mock()
        mock_factory.create_search_method.return_value = mock_search_method
        
        mock_retriever = Mock()
        mock_eval_class.return_value = mock_retriever
        mock_retriever.retrieve.return_value = {"q1": {"doc1": 0.9}}
        mock_retriever.evaluate.return_value = (
            {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
        )
        mock_retriever.k_values = [10]
        
        # Original interface: run(model, output_folder)
        evaluator = COIR(self.test_tasks, 32)
        mock_model = Mock()
        
        results = evaluator.run(mock_model, self.temp_dir)
        
        # Should work without errors
        self.assertIn("test_task", results)


if __name__ == '__main__':
    unittest.main()