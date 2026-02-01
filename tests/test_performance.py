"""
Performance tests to ensure refactoring doesn't degrade performance

This module tests performance characteristics of the refactored CoIR architecture
to ensure that the new flexibility doesn't come at the cost of performance degradation.
"""

import pytest
import time
import tempfile
import shutil
import psutil
import os
from unittest.mock import Mock, patch
import numpy as np
from typing import Dict, Any

from coir.evaluation import COIR
from coir.beir.retrieval.search.factory import SearchMethodFactory


class TestPerformance:
    """Performance tests for the refactored architecture"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data of various sizes
        self.small_tasks = self._create_test_tasks(num_docs=100, num_queries=10)
        self.medium_tasks = self._create_test_tasks(num_docs=1000, num_queries=50)
        self.large_tasks = self._create_test_tasks(num_docs=5000, num_queries=100)
        
        # Create mock model
        self.mock_model = Mock()
        self.mock_model.model_name = "performance_test_model"
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_tasks(self, num_docs: int, num_queries: int) -> Dict[str, Any]:
        """Create test tasks of specified size"""
        corpus = {}
        queries = {}
        qrels = {}
        
        # Generate corpus
        for i in range(num_docs):
            corpus[f"doc{i}"] = {
                "title": f"Document {i}",
                "text": f"This is test document {i} with some content for testing performance."
            }
        
        # Generate queries
        for i in range(num_queries):
            queries[f"q{i}"] = f"Test query {i} for performance testing"
            # Create some relevant documents for each query
            qrels[f"q{i}"] = {f"doc{i % num_docs}": 1}
        
        return {
            "performance_task": (corpus, queries, qrels)
        }
    
    def _setup_mock_model_for_size(self, num_docs: int, num_queries: int):
        """Setup mock model with appropriate return sizes"""
        self.mock_model.encode_corpus.return_value = np.random.rand(num_docs, 768)
        self.mock_model.encode_queries.return_value = np.random.rand(num_queries, 768)
    
    def _measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    def _measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage during function execution"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        return result, memory_used
    
    def test_dense_search_performance(self):
        """Ensure dense search performance is maintained"""
        with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
             patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
            
            # Setup mocks for different data sizes
            test_cases = [
                ("small", self.small_tasks, 100, 10),
                ("medium", self.medium_tasks, 1000, 50),
                ("large", self.large_tasks, 5000, 100)
            ]
            
            performance_results = {}
            
            for size_name, tasks, num_docs, num_queries in test_cases:
                self._setup_mock_model_for_size(num_docs, num_queries)
                
                # Setup mocks
                mock_search = Mock()
                mock_factory.create_search_method.return_value = mock_search
                
                mock_retriever = Mock()
                mock_eval_class.return_value = mock_retriever
                
                # Create realistic return data
                mock_results = {}
                for i in range(num_queries):
                    mock_results[f"q{i}"] = {f"doc{j}": 0.9 - (j * 0.01) for j in range(min(10, num_docs))}
                
                mock_retriever.retrieve.return_value = mock_results
                mock_retriever.evaluate.return_value = (
                    {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
                )
                mock_retriever.k_values = [10]
                
                # Measure performance
                evaluation = COIR(tasks, batch_size=32, search_config={"method": "dense"})
                
                result, execution_time = self._measure_execution_time(
                    evaluation.run, self.mock_model, self.temp_dir
                )
                
                performance_results[size_name] = {
                    "execution_time": execution_time,
                    "num_docs": num_docs,
                    "num_queries": num_queries
                }
                
                # Performance assertions
                assert execution_time < 10.0, f"Dense search took too long for {size_name} dataset: {execution_time}s"
                assert isinstance(result, dict), "Result should be a dictionary"
                assert "performance_task" in result, "Task results should be present"
            
            # Check that performance scales reasonably
            small_time = performance_results["small"]["execution_time"]
            medium_time = performance_results["medium"]["execution_time"]
            large_time = performance_results["large"]["execution_time"]
            
            # Performance should scale sub-linearly (allowing for some overhead)
            assert medium_time < small_time * 20, "Medium dataset performance degradation too high"
            assert large_time < medium_time * 10, "Large dataset performance degradation too high"
    
    def test_lexical_search_performance(self):
        """Test performance of lexical search methods"""
        lexical_methods = ["jaccard", "bm25"]
        
        for method in lexical_methods:
            with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                 patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                
                # Setup mocks
                mock_search = Mock()
                mock_factory.create_search_method.return_value = mock_search
                
                mock_retriever = Mock()
                mock_eval_class.return_value = mock_retriever
                mock_retriever.retrieve.return_value = {"q0": {"doc0": 0.9}}
                mock_retriever.evaluate.return_value = (
                    {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
                )
                mock_retriever.k_values = [10]
                
                # Measure performance
                evaluation = COIR(self.medium_tasks, batch_size=32, search_config={"method": method})
                
                result, execution_time = self._measure_execution_time(
                    evaluation.run, self.mock_model, self.temp_dir
                )
                
                # Lexical methods should be fast
                assert execution_time < 5.0, f"{method} search took too long: {execution_time}s"
                assert isinstance(result, dict), f"{method} result should be a dictionary"
    
    def test_hybrid_search_performance(self):
        """Test performance of hybrid search methods"""
        hybrid_methods = ["simple_hybrid", "advanced_hybrid"]
        
        for method in hybrid_methods:
            with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                 patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                
                self._setup_mock_model_for_size(1000, 50)
                
                # Setup mocks
                mock_search = Mock()
                mock_factory.create_search_method.return_value = mock_search
                
                mock_retriever = Mock()
                mock_eval_class.return_value = mock_retriever
                mock_retriever.retrieve.return_value = {"q0": {"doc0": 0.9}}
                mock_retriever.evaluate.return_value = (
                    {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
                )
                mock_retriever.k_values = [10]
                
                # Measure performance
                evaluation = COIR(self.medium_tasks, batch_size=32, search_config={"method": method})
                
                result, execution_time = self._measure_execution_time(
                    evaluation.run, self.mock_model, self.temp_dir
                )
                
                # Hybrid methods may be slower but should still be reasonable
                assert execution_time < 15.0, f"{method} search took too long: {execution_time}s"
                assert isinstance(result, dict), f"{method} result should be a dictionary"
    
    def test_memory_usage(self):
        """Test memory usage across different search methods"""
        methods = ["dense", "jaccard", "bm25", "simple_hybrid"]
        memory_results = {}
        
        for method in methods:
            with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                 patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                
                if method in ["dense", "simple_hybrid"]:
                    self._setup_mock_model_for_size(1000, 50)
                
                # Setup mocks
                mock_search = Mock()
                mock_factory.create_search_method.return_value = mock_search
                
                mock_retriever = Mock()
                mock_eval_class.return_value = mock_retriever
                mock_retriever.retrieve.return_value = {"q0": {"doc0": 0.9}}
                mock_retriever.evaluate.return_value = (
                    {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
                )
                mock_retriever.k_values = [10]
                
                # Measure memory usage
                evaluation = COIR(self.medium_tasks, batch_size=32, search_config={"method": method})
                
                result, memory_used = self._measure_memory_usage(
                    evaluation.run, self.mock_model, self.temp_dir
                )
                
                memory_results[method] = memory_used
                
                # Memory usage should be reasonable (less than 500MB for test data)
                assert memory_used < 500, f"{method} used too much memory: {memory_used}MB"
        
        # Log memory usage for analysis
        print(f"Memory usage by method: {memory_results}")
    
    def test_scalability(self):
        """Test with progressively larger datasets"""
        dataset_sizes = [
            (100, 10, "tiny"),
            (500, 25, "small"),
            (1000, 50, "medium"),
            (2000, 100, "large")
        ]
        
        scalability_results = []
        
        for num_docs, num_queries, size_name in dataset_sizes:
            tasks = self._create_test_tasks(num_docs, num_queries)
            
            with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                 patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                
                self._setup_mock_model_for_size(num_docs, num_queries)
                
                # Setup mocks
                mock_search = Mock()
                mock_factory.create_search_method.return_value = mock_search
                
                mock_retriever = Mock()
                mock_eval_class.return_value = mock_retriever
                mock_retriever.retrieve.return_value = {"q0": {"doc0": 0.9}}
                mock_retriever.evaluate.return_value = (
                    {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
                )
                mock_retriever.k_values = [10]
                
                # Measure performance
                evaluation = COIR(tasks, batch_size=32, search_config={"method": "dense"})
                
                result, execution_time = self._measure_execution_time(
                    evaluation.run, self.mock_model, self.temp_dir
                )
                
                scalability_results.append({
                    "size": size_name,
                    "num_docs": num_docs,
                    "num_queries": num_queries,
                    "execution_time": execution_time
                })
                
                # Each size should complete in reasonable time
                max_time = 20.0  # 20 seconds max for any size in this test
                assert execution_time < max_time, f"{size_name} dataset took too long: {execution_time}s"
        
        # Check that scaling is reasonable (not exponential)
        for i in range(1, len(scalability_results)):
            prev_result = scalability_results[i-1]
            curr_result = scalability_results[i]
            
            size_ratio = curr_result["num_docs"] / prev_result["num_docs"]
            time_ratio = curr_result["execution_time"] / prev_result["execution_time"]
            
            # Time should not increase faster than size squared
            assert time_ratio < size_ratio ** 2, f"Performance degradation too high between {prev_result['size']} and {curr_result['size']}"
    
    def test_batch_size_impact(self):
        """Test impact of different batch sizes on performance"""
        batch_sizes = [16, 32, 64, 128]
        batch_results = {}
        
        for batch_size in batch_sizes:
            with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                 patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                
                self._setup_mock_model_for_size(1000, 50)
                
                # Setup mocks
                mock_search = Mock()
                mock_factory.create_search_method.return_value = mock_search
                
                mock_retriever = Mock()
                mock_eval_class.return_value = mock_retriever
                mock_retriever.retrieve.return_value = {"q0": {"doc0": 0.9}}
                mock_retriever.evaluate.return_value = (
                    {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
                )
                mock_retriever.k_values = [10]
                
                # Measure performance
                evaluation = COIR(self.medium_tasks, batch_size=batch_size, search_config={"method": "dense"})
                
                result, execution_time = self._measure_execution_time(
                    evaluation.run, self.mock_model, self.temp_dir
                )
                
                batch_results[batch_size] = execution_time
                
                # All batch sizes should complete in reasonable time
                assert execution_time < 10.0, f"Batch size {batch_size} took too long: {execution_time}s"
        
        # Log batch size impact for analysis
        print(f"Execution time by batch size: {batch_results}")
    
    def test_concurrent_evaluations(self):
        """Test performance when running multiple evaluations"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def run_evaluation(method_name):
            """Run a single evaluation and put result in queue"""
            try:
                with patch('coir.evaluation.SearchMethodFactory') as mock_factory, \
                     patch('coir.evaluation.EvaluateRetrieval') as mock_eval_class:
                    
                    if method_name in ["dense", "simple_hybrid"]:
                        self._setup_mock_model_for_size(500, 25)
                    
                    # Setup mocks
                    mock_search = Mock()
                    mock_factory.create_search_method.return_value = mock_search
                    
                    mock_retriever = Mock()
                    mock_eval_class.return_value = mock_retriever
                    mock_retriever.retrieve.return_value = {"q0": {"doc0": 0.9}}
                    mock_retriever.evaluate.return_value = (
                        {"NDCG@10": 0.8}, {"MAP@10": 0.7}, {"Recall@10": 0.6}, {"P@10": 0.5}
                    )
                    mock_retriever.k_values = [10]
                    
                    # Run evaluation
                    evaluation = COIR(self.small_tasks, batch_size=32, search_config={"method": method_name})
                    
                    start_time = time.time()
                    result = evaluation.run(self.mock_model, self.temp_dir)
                    end_time = time.time()
                    
                    results_queue.put({
                        "method": method_name,
                        "execution_time": end_time - start_time,
                        "success": True
                    })
            except Exception as e:
                results_queue.put({
                    "method": method_name,
                    "error": str(e),
                    "success": False
                })
        
        # Run multiple evaluations concurrently
        methods = ["dense", "jaccard", "bm25"]
        threads = []
        
        start_time = time.time()
        
        for method in methods:
            thread = threading.Thread(target=run_evaluation, args=(method,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify all evaluations completed successfully
        assert len(results) == len(methods), "Not all evaluations completed"
        
        for result in results:
            assert result["success"], f"Evaluation failed for {result['method']}: {result.get('error', 'Unknown error')}"
            assert result["execution_time"] < 10.0, f"Concurrent evaluation took too long for {result['method']}"
        
        # Concurrent execution should not take much longer than sequential
        assert total_time < 15.0, f"Concurrent evaluations took too long: {total_time}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])