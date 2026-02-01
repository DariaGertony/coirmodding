"""
End-to-end workflow tests for common usage patterns

This module tests complete workflows that researchers and practitioners
would use with the CoIR framework, ensuring all components work together
seamlessly in realistic scenarios.
"""

import pytest
import tempfile
import shutil
import os
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from coir.evaluation import COIR
from coir.beir.retrieval.search.factory import SearchMethodFactory


class TestE2EWorkflows:
    """End-to-end workflow tests for common usage patterns"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create realistic test tasks
        self.test_tasks = self._create_realistic_tasks()
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.mock_model.encode_corpus.return_value = np.random.rand(10, 384)
        self.mock_model.encode_queries.return_value = np.random.rand(5, 384)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_realistic_tasks(self):
        """Create realistic test tasks for workflow testing"""
        return {
            "code_search_task": (
                {  # corpus - code snippets
                    "func1": {
                        "title": "Binary Search Implementation",
                        "text": "def binary_search(arr, target): left, right = 0, len(arr) - 1"
                    },
                    "func2": {
                        "title": "Quick Sort Algorithm", 
                        "text": "def quicksort(arr): if len(arr) <= 1: return arr"
                    },
                    "func3": {
                        "title": "Merge Sort Implementation",
                        "text": "def merge_sort(arr): if len(arr) <= 1: return arr"
                    },
                    "func4": {
                        "title": "Linear Search Function",
                        "text": "def linear_search(arr, target): for i, val in enumerate(arr):"
                    },
                    "func5": {
                        "title": "Bubble Sort Algorithm",
                        "text": "def bubble_sort(arr): n = len(arr); for i in range(n):"
                    }
                },
                {  # queries - search intents
                    "q1": "find binary search algorithm",
                    "q2": "sorting algorithm implementation",
                    "q3": "search function in array",
                    "q4": "efficient sorting method",
                    "q5": "divide and conquer algorithm"
                },
                {  # qrels - relevance judgments
                    "q1": {"func1": 1, "func4": 1, "func2": 0, "func3": 0, "func5": 0},
                    "q2": {"func2": 1, "func3": 1, "func5": 1, "func1": 0, "func4": 0},
                    "q3": {"func1": 1, "func4": 1, "func2": 0, "func3": 0, "func5": 0},
                    "q4": {"func2": 1, "func3": 1, "func1": 0, "func4": 0, "func5": 0},
                    "q5": {"func2": 1, "func3": 1, "func1": 1, "func4": 0, "func5": 0}
                }
            )
        }
    
    def _setup_mock_evaluation(self, mock_factory, mock_eval_class):
        """Setup common mock evaluation components"""
        mock_search = Mock()
        mock_factory.create_search_method.return_value = mock_search
        
        mock_retriever = Mock()
        mock_eval_class.return_value = mock_retriever
        
        # Create realistic retrieval results
        mock_retriever.retrieve.return_value = {
            "q1": {"func1": 0.95, "func4": 0.85, "func2": 0.3, "func3": 0.2, "func5": 0.1},
            "q2": {"func2": 0.92, "func3": 0.88, "func5": 0.75, "func1": 0.2, "func4": 0.1},
            "q3": {"func1": 0.90, "func4": 0.87, "func2": 0.25, "func3": 0.15, "func5": 0.1},
            "q4": {"func2": 0.89, "func3": 0.86, "func1": 0.3, "func4": 0.2, "func5": 0.4},
            "q5": {"func2": 0.91, "func3": 0.88, "func1": 0.82, "func4": 0.1, "func5": 0.2}
        }
        
        # Realistic evaluation metrics
        mock_retriever.evaluate.return_value = (
            {"NDCG@1": 0.85, "NDCG@3": 0.82, "NDCG@5": 0.80, "NDCG@10": 0.78},
            {"MAP@1": 0.85, "MAP@3": 0.83, "MAP@5": 0.81, "MAP@10": 0.79},
            {"Recall@1": 0.45, "Recall@3": 0.72, "Recall@5": 0.85, "Recall@10": 0.92},
            {"P@1": 0.85, "P@3": 0.78, "P@5": 0.72, "P@10": 0.65}
        )
        mock_retriever.k_values = [1, 3, 5, 10]
        
        return mock_search, mock_retriever
    
    def test_research_comparison_workflow(self):
        """Test workflow for comparing different search methods"""
        # Simulate researcher comparing semantic vs lexical vs hybrid approaches
        methods_to_compare = [
            {"method": "dense", "name": "Semantic Search"},
            {"method": "bm25", "name": "Lexical Search (BM25)"},
            {"method": "jaccard", "name": "Lexical Search (Jaccard)"},
            {"method": "simple_hybrid", "name": "Simple Hybrid"},
            {"method": "advanced_hybrid", "name": "Advanced Hybrid", "params": {"fusion_method": "rrf"}}
        ]
        
        comparison_results = {}
        
        for method_config in methods_to_compare:
            method_name = method_config["name"]
            search_config = {k: v for k, v in method_config.items() if k not in ["name"]}
            
            with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                 patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                
                mock_search, mock_retriever = self._setup_mock_evaluation(mock_factory, mock_eval_class)
                
                # Create evaluation with specific method
                evaluation = COIR(self.test_tasks, batch_size=32, search_config=search_config)
                results = evaluation.run(self.mock_model, os.path.join(self.temp_dir, method_config["method"]))
                
                comparison_results[method_name] = results
                
                # Verify method was called correctly
                if search_config["method"] in ["dense", "simple_hybrid", "advanced_hybrid"]:
                    mock_factory.create_search_method.assert_called_with(
                        search_config["method"], 
                        model=self.mock_model, 
                        batch_size=32,
                        **search_config.get("params", {})
                    )
                else:
                    mock_factory.create_search_method.assert_called_with(
                        search_config["method"], 
                        batch_size=32,
                        **search_config.get("params", {})
                    )
        
        # Verify all methods produced comparable results
        assert len(comparison_results) == len(methods_to_compare)
        
        for method_name, method_results in comparison_results.items():
            assert isinstance(method_results, dict)
            assert "code_search_task" in method_results
            
            task_metrics = method_results["code_search_task"]
            assert "NDCG" in task_metrics
            assert "MAP" in task_metrics
            assert "Recall" in task_metrics
            assert "Precision" in task_metrics
        
        # Verify result files were created for each method
        for method_config in methods_to_compare:
            result_dir = os.path.join(self.temp_dir, method_config["method"])
            result_file = os.path.join(result_dir, "code_search_task.json")
            assert os.path.exists(result_file)
    
    def test_ablation_study_workflow(self):
        """Test workflow for ablation studies"""
        # Test with/without LLM, with/without reranking
        base_config = {"method": "advanced_hybrid", "params": {"fusion_method": "rrf"}}
        evaluation = COIR(self.test_tasks, batch_size=32, search_config=base_config)
        
        ablation_configs = [
            {"name": "baseline", "use_llm": False, "enable_reranking": False},
            {"name": "with_llm", "use_llm": True, "llm_name": "llama2", "prompt": "Expand this code search query:", "enable_reranking": False},
            {"name": "with_reranking", "use_llm": False, "enable_reranking": True},
            {"name": "full_system", "use_llm": True, "llm_name": "llama2", "prompt": "Expand this code search query:", "enable_reranking": True}
        ]
        
        ablation_results = {}
        
        for config in ablation_configs:
            config_name = config.pop("name")
            
            with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                 patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                
                mock_search, mock_retriever = self._setup_mock_evaluation(mock_factory, mock_eval_class)
                
                # Simulate different performance for different configurations
                if config.get("use_llm", False):
                    # LLM should improve performance slightly
                    mock_retriever.evaluate.return_value = (
                        {"NDCG@1": 0.88, "NDCG@3": 0.85, "NDCG@5": 0.83, "NDCG@10": 0.81},
                        {"MAP@1": 0.88, "MAP@3": 0.86, "MAP@5": 0.84, "MAP@10": 0.82},
                        {"Recall@1": 0.48, "Recall@3": 0.75, "Recall@5": 0.88, "Recall@10": 0.95},
                        {"P@1": 0.88, "P@3": 0.81, "P@5": 0.75, "P@10": 0.68}
                    )
                
                if config.get("enable_reranking", False):
                    # Reranking should improve precision
                    mock_retriever.evaluate.return_value = (
                        {"NDCG@1": 0.90, "NDCG@3": 0.87, "NDCG@5": 0.85, "NDCG@10": 0.83},
                        {"MAP@1": 0.90, "MAP@3": 0.88, "MAP@5": 0.86, "MAP@10": 0.84},
                        {"Recall@1": 0.50, "Recall@3": 0.77, "Recall@5": 0.90, "Recall@10": 0.97},
                        {"P@1": 0.90, "P@3": 0.83, "P@5": 0.77, "P@10": 0.70}
                    )
                
                # Run evaluation with specific configuration
                results = evaluation.run(
                    self.mock_model, 
                    os.path.join(self.temp_dir, "ablation", config_name),
                    **config
                )
                
                ablation_results[config_name] = results
        
        # Verify all configurations work
        assert len(ablation_results) == 4
        
        for config_name, results in ablation_results.items():
            assert isinstance(results, dict)
            assert "code_search_task" in results
            
            # Verify result files were created with proper organization
            result_dir = os.path.join(self.temp_dir, "ablation", config_name)
            result_file = os.path.join(result_dir, "code_search_task.json")
            assert os.path.exists(result_file)
            
            # Verify JSON structure includes configuration information
            with open(result_file, 'r') as f:
                data = json.load(f)
                assert isinstance(data, dict)
                # Should have different configuration labels
                assert len(data) > 0
    
    def test_model_comparison_workflow(self):
        """Test workflow for comparing different embedding models"""
        # Simulate comparing different embedding models
        models_to_test = [
            {"name": "all-MiniLM-L6-v2", "dim": 384},
            {"name": "all-mpnet-base-v2", "dim": 768},
            {"name": "e5-base-v2", "dim": 768}
        ]
        
        model_results = {}
        
        for model_info in models_to_test:
            # Create mock model for each
            mock_model = Mock()
            mock_model.model_name = f"sentence-transformers/{model_info['name']}"
            mock_model.encode_corpus.return_value = np.random.rand(10, model_info['dim'])
            mock_model.encode_queries.return_value = np.random.rand(5, model_info['dim'])
            
            with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                 patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                
                mock_search, mock_retriever = self._setup_mock_evaluation(mock_factory, mock_eval_class)
                
                # Simulate different performance for different models
                if "mpnet" in model_info['name']:
                    # Larger model should perform better
                    mock_retriever.evaluate.return_value = (
                        {"NDCG@1": 0.90, "NDCG@3": 0.87, "NDCG@5": 0.85, "NDCG@10": 0.83},
                        {"MAP@1": 0.90, "MAP@3": 0.88, "MAP@5": 0.86, "MAP@10": 0.84},
                        {"Recall@1": 0.50, "Recall@3": 0.77, "Recall@5": 0.90, "Recall@10": 0.97},
                        {"P@1": 0.90, "P@3": 0.83, "P@5": 0.77, "P@10": 0.70}
                    )
                elif "e5" in model_info['name']:
                    # E5 model should also perform well
                    mock_retriever.evaluate.return_value = (
                        {"NDCG@1": 0.87, "NDCG@3": 0.84, "NDCG@5": 0.82, "NDCG@10": 0.80},
                        {"MAP@1": 0.87, "MAP@3": 0.85, "MAP@5": 0.83, "MAP@10": 0.81},
                        {"Recall@1": 0.47, "Recall@3": 0.74, "Recall@5": 0.87, "Recall@10": 0.94},
                        {"P@1": 0.87, "P@3": 0.80, "P@5": 0.74, "P@10": 0.67}
                    )
                
                # Run evaluation
                evaluation = COIR(self.test_tasks, batch_size=32, search_config={"method": "dense"})
                results = evaluation.run(
                    mock_model, 
                    os.path.join(self.temp_dir, "models", model_info['name'])
                )
                
                model_results[model_info['name']] = results
        
        # Verify all models were tested
        assert len(model_results) == len(models_to_test)
        
        for model_name, results in model_results.items():
            assert isinstance(results, dict)
            assert "code_search_task" in results
            
            # Verify result files
            result_dir = os.path.join(self.temp_dir, "models", model_name)
            result_file = os.path.join(result_dir, "code_search_task.json")
            assert os.path.exists(result_file)
    
    def test_parameter_tuning_workflow(self):
        """Test workflow for parameter tuning"""
        # Test different BM25 parameters
        bm25_params = [
            {"k1": 1.2, "b": 0.75},  # Default
            {"k1": 1.5, "b": 0.75},  # Higher k1
            {"k1": 1.2, "b": 0.5},   # Lower b
            {"k1": 2.0, "b": 0.9}    # Aggressive
        ]
        
        tuning_results = {}
        
        for i, params in enumerate(bm25_params):
            config = {"method": "bm25", "params": params}
            param_name = f"bm25_k1_{params['k1']}_b_{params['b']}"
            
            with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                 patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                
                mock_search, mock_retriever = self._setup_mock_evaluation(mock_factory, mock_eval_class)
                
                # Simulate parameter impact on performance
                base_ndcg = 0.75
                k1_impact = (params['k1'] - 1.2) * 0.02  # k1 impact
                b_impact = (0.75 - params['b']) * 0.01   # b impact
                adjusted_ndcg = base_ndcg + k1_impact + b_impact
                
                mock_retriever.evaluate.return_value = (
                    {"NDCG@10": adjusted_ndcg},
                    {"MAP@10": adjusted_ndcg - 0.02},
                    {"Recall@10": min(0.95, adjusted_ndcg + 0.1)},
                    {"P@10": adjusted_ndcg - 0.05}
                )
                
                # Run evaluation
                evaluation = COIR(self.test_tasks, batch_size=32, search_config=config)
                results = evaluation.run(
                    self.mock_model, 
                    os.path.join(self.temp_dir, "tuning", param_name)
                )
                
                tuning_results[param_name] = results
                
                # Verify parameters were passed correctly
                mock_factory.create_search_method.assert_called_with(
                    "bm25", batch_size=32, **params
                )
        
        # Verify all parameter combinations were tested
        assert len(tuning_results) == len(bm25_params)
        
        for param_name, results in tuning_results.items():
            assert isinstance(results, dict)
            assert "code_search_task" in results
    
    def test_cross_domain_evaluation_workflow(self):
        """Test workflow for cross-domain evaluation"""
        # Simulate evaluating on multiple domains
        domains = ["code_search", "academic_papers", "web_documents"]
        
        domain_tasks = {}
        for domain in domains:
            domain_tasks[f"{domain}_task"] = self.test_tasks["code_search_task"]  # Reuse for simplicity
        
        cross_domain_results = {}
        
        # Test same model across different domains
        evaluation = COIR(domain_tasks, batch_size=32, search_config={"method": "dense"})
        
        with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
             patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
            
            mock_search, mock_retriever = self._setup_mock_evaluation(mock_factory, mock_eval_class)
            
            # Run evaluation across all domains
            results = evaluation.run(
                self.mock_model, 
                os.path.join(self.temp_dir, "cross_domain")
            )
            
            cross_domain_results = results
        
        # Verify results for all domains
        assert len(cross_domain_results) == len(domains)
        
        for domain in domains:
            task_name = f"{domain}_task"
            assert task_name in cross_domain_results
            assert isinstance(cross_domain_results[task_name], dict)
            
            # Verify result files
            result_file = os.path.join(self.temp_dir, "cross_domain", f"{task_name}.json")
            assert os.path.exists(result_file)
    
    def test_production_deployment_workflow(self):
        """Test workflow for production deployment validation"""
        # Simulate validating a model before production deployment
        
        # Test with production-like configuration
        production_config = {
            "method": "advanced_hybrid",
            "params": {
                "fusion_method": "rrf",
                "weights": {"dense": 0.7, "lexical": 0.3}
            }
        }
        
        with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
             patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
            
            mock_search, mock_retriever = self._setup_mock_evaluation(mock_factory, mock_eval_class)
            
            # Simulate production-quality metrics
            mock_retriever.evaluate.return_value = (
                {"NDCG@1": 0.92, "NDCG@3": 0.89, "NDCG@5": 0.87, "NDCG@10": 0.85},
                {"MAP@1": 0.92, "MAP@3": 0.90, "MAP@5": 0.88, "MAP@10": 0.86},
                {"Recall@1": 0.52, "Recall@3": 0.79, "Recall@5": 0.92, "Recall@10": 0.98},
                {"P@1": 0.92, "P@3": 0.85, "P@5": 0.79, "P@10": 0.72}
            )
            
            # Run production validation
            evaluation = COIR(self.test_tasks, batch_size=64, search_config=production_config)
            
            # Test with multiple enhancements
            results = evaluation.run(
                self.mock_model,
                os.path.join(self.temp_dir, "production"),
                use_llm=True,
                llm_name="production_llm",
                prompt="Optimize this search query for production:",
                enable_reranking=True
            )
            
            # Verify production-quality results
            assert isinstance(results, dict)
            assert "code_search_task" in results
            
            task_metrics = results["code_search_task"]
            
            # Production should meet quality thresholds
            assert task_metrics["NDCG"]["NDCG@10"] >= 0.8  # High quality threshold
            assert task_metrics["MAP"]["MAP@10"] >= 0.8
            
            # Verify production result file structure
            result_file = os.path.join(self.temp_dir, "production", "code_search_task.json")
            assert os.path.exists(result_file)
            
            with open(result_file, 'r') as f:
                data = json.load(f)
                # Should have enhanced configuration
                assert "production_llm" in str(data)  # LLM configuration should be recorded
    
    def test_continuous_evaluation_workflow(self):
        """Test workflow for continuous evaluation and monitoring"""
        # Simulate continuous evaluation with different time periods
        
        time_periods = ["week1", "week2", "week3", "week4"]
        continuous_results = {}
        
        for period in time_periods:
            with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                 patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                
                mock_search, mock_retriever = self._setup_mock_evaluation(mock_factory, mock_eval_class)
                
                # Simulate slight performance drift over time
                week_num = int(period.replace("week", ""))
                drift_factor = 1.0 - (week_num - 1) * 0.01  # 1% degradation per week
                
                base_metrics = (
                    {"NDCG@10": 0.85 * drift_factor},
                    {"MAP@10": 0.83 * drift_factor},
                    {"Recall@10": 0.92 * drift_factor},
                    {"P@10": 0.70 * drift_factor}
                )
                mock_retriever.evaluate.return_value = base_metrics
                
                # Run evaluation for this time period
                evaluation = COIR(self.test_tasks, batch_size=32, search_config={"method": "dense"})
                results = evaluation.run(
                    self.mock_model,
                    os.path.join(self.temp_dir, "continuous", period)
                )
                
                continuous_results[period] = results
        
        # Verify continuous monitoring results
        assert len(continuous_results) == len(time_periods)
        
        # Check for performance trends
        ndcg_scores = []
        for period in time_periods:
            ndcg_score = continuous_results[period]["code_search_task"]["NDCG"]["NDCG@10"]
            ndcg_scores.append(ndcg_score)
        
        # Should show gradual decline (simulated drift)
        assert ndcg_scores[0] > ndcg_scores[-1]  # Performance should decline over time
        
        # Verify all result files exist
        for period in time_periods:
            result_file = os.path.join(self.temp_dir, "continuous", period, "code_search_task.json")
            assert os.path.exists(result_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])