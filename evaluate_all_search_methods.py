#!/usr/bin/env python3
"""
Comprehensive Search Methods Evaluation Script

This script evaluates all available search methods in the refactored CoIR architecture:
- Dense/Semantic Search (original)
- Lexical Search (Jaccard, BM25)
- Hybrid Search (Simple, Advanced)

It provides a comprehensive comparison of different search approaches with detailed metrics.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coir.evaluation import COIR
from coir.data_loader import get_tasks
from coir.models import YourCustomDEModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('search_evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SearchMethodsEvaluator:
    """
    Comprehensive evaluator for all search methods in CoIR
    """
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2", batch_size: int = 32):
        """
        Initialize the evaluator
        
        Args:
            model_name: Name of the model to use for dense/hybrid search
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.results = {}
        
        # Define all search configurations to test
        self.search_configs = {
            "dense": {
                "method": "dense",
                "description": "Dense/Semantic Search (Original CoIR)",
                "requires_model": True
            },
            "semantic": {
                "method": "semantic", 
                "description": "Semantic Search (Alias for Dense)",
                "requires_model": True
            },
            "jaccard": {
                "method": "jaccard",
                "description": "Lexical Search using Jaccard Similarity",
                "requires_model": False
            },
            "bm25": {
                "method": "bm25",
                "description": "Lexical Search using BM25 Algorithm",
                "requires_model": False
            },
            "bm25_tuned": {
                "method": "bm25",
                "params": {"k1": 1.5, "b": 0.8},
                "description": "BM25 with Tuned Parameters",
                "requires_model": False
            },
            "simple_hybrid": {
                "method": "simple_hybrid",
                "description": "Simple Hybrid Search (Dense + Lexical)",
                "requires_model": True
            },
            "advanced_hybrid_rrf": {
                "method": "advanced_hybrid",
                "params": {"fusion_method": "rrf"},
                "description": "Advanced Hybrid with Reciprocal Rank Fusion",
                "requires_model": True
            },
            "advanced_hybrid_weighted": {
                "method": "advanced_hybrid", 
                "params": {
                    "fusion_method": "weighted",
                    "weights": {"dense": 0.7, "lexical": 0.3}
                },
                "description": "Advanced Hybrid with Weighted Fusion",
                "requires_model": True
            }
        }
        
        logger.info(f"Initialized evaluator with model: {model_name}")
        logger.info(f"Available search methods: {list(self.search_configs.keys())}")
    
    def load_model(self):
        """Load the embedding model for dense/hybrid search methods"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = YourCustomDEModel(model_name=self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_test_tasks(self, task_names: List[str] = None) -> Dict:
        """
        Load test tasks for evaluation
        
        Args:
            task_names: List of task names to load. If None, loads a default subset.
            
        Returns:
            Dictionary of loaded tasks
        """
        if task_names is None:
            # Use a smaller subset for comprehensive testing
            task_names = ["codetrans-dl"]  # Start with one task for testing
        
        try:
            logger.info(f"Loading tasks: {task_names}")
            tasks = get_tasks(tasks=task_names)
            logger.info(f"Loaded {len(tasks)} tasks successfully")
            return tasks
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")
            raise
    
    def evaluate_search_method(self, config_name: str, config: Dict, tasks: Dict, output_dir: str) -> Dict:
        """
        Evaluate a single search method
        
        Args:
            config_name: Name of the configuration
            config: Search configuration dictionary
            tasks: Tasks to evaluate on
            output_dir: Output directory for results
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {config_name}: {config['description']}")
        
        try:
            # Create search configuration
            search_config = {
                "method": config["method"]
            }
            if "params" in config:
                search_config["params"] = config["params"]
            
            # Initialize COIR evaluation
            evaluation = COIR(
                tasks=tasks,
                batch_size=self.batch_size,
                search_config=search_config
            )
            
            # Create output directory for this method
            method_output_dir = os.path.join(output_dir, config_name)
            os.makedirs(method_output_dir, exist_ok=True)
            
            # Run evaluation
            start_time = time.time()
            
            if config.get("requires_model", False):
                if self.model is None:
                    self.load_model()
                results = evaluation.run(self.model, method_output_dir)
            else:
                # For lexical methods, we still need to pass a model but it won't be used
                if self.model is None:
                    self.load_model()
                results = evaluation.run(self.model, method_output_dir)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"✅ {config_name} completed in {execution_time:.2f} seconds")
            
            # Add metadata to results
            results_with_metadata = {
                "config_name": config_name,
                "description": config["description"],
                "search_config": search_config,
                "execution_time": execution_time,
                "results": results
            }
            
            return results_with_metadata
            
        except Exception as e:
            logger.error(f"❌ {config_name} failed: {e}")
            return {
                "config_name": config_name,
                "description": config["description"],
                "error": str(e),
                "execution_time": 0,
                "results": {}
            }
    
    def run_comprehensive_evaluation(self, 
                                   task_names: List[str] = None,
                                   output_dir: str = "search_methods_evaluation",
                                   methods_to_test: List[str] = None) -> Dict:
        """
        Run comprehensive evaluation of all search methods
        
        Args:
            task_names: List of task names to evaluate on
            output_dir: Output directory for all results
            methods_to_test: List of method names to test. If None, tests all.
            
        Returns:
            Complete evaluation results
        """
        logger.info("🚀 Starting comprehensive search methods evaluation")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tasks
        tasks = self.get_test_tasks(task_names)
        
        # Determine which methods to test
        if methods_to_test is None:
            methods_to_test = list(self.search_configs.keys())
        
        logger.info(f"Testing {len(methods_to_test)} search methods: {methods_to_test}")
        
        # Run evaluations
        all_results = {}
        total_start_time = time.time()
        
        for i, config_name in enumerate(methods_to_test, 1):
            if config_name not in self.search_configs:
                logger.warning(f"Unknown search method: {config_name}")
                continue
            
            logger.info(f"[{i}/{len(methods_to_test)}] Testing {config_name}")
            
            config = self.search_configs[config_name]
            result = self.evaluate_search_method(config_name, config, tasks, output_dir)
            all_results[config_name] = result
        
        total_time = time.time() - total_start_time
        
        # Generate comprehensive report
        report = self.generate_comparison_report(all_results, total_time)
        
        # Save complete results
        results_file = os.path.join(output_dir, "comprehensive_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "evaluation_metadata": {
                    "model_name": self.model_name,
                    "batch_size": self.batch_size,
                    "total_execution_time": total_time,
                    "tasks_evaluated": list(tasks.keys()) if tasks else [],
                    "methods_tested": methods_to_test
                },
                "results": all_results,
                "comparison_report": report
            }, f, indent=2)
        
        # Save human-readable report
        report_file = os.path.join(output_dir, "evaluation_report.md")
        self.save_markdown_report(report, report_file, all_results)
        
        logger.info(f"🎉 Comprehensive evaluation completed in {total_time:.2f} seconds")
        logger.info(f"📊 Results saved to: {output_dir}")
        logger.info(f"📋 Report available at: {report_file}")
        
        return all_results
    
    def generate_comparison_report(self, all_results: Dict, total_time: float) -> Dict:
        """Generate a comparison report of all methods"""
        
        report = {
            "summary": {
                "total_methods_tested": len(all_results),
                "successful_evaluations": len([r for r in all_results.values() if "error" not in r]),
                "failed_evaluations": len([r for r in all_results.values() if "error" in r]),
                "total_execution_time": total_time
            },
            "method_performance": {},
            "metric_comparison": {}
        }
        
        # Analyze performance for each method
        for method_name, result in all_results.items():
            if "error" in result:
                report["method_performance"][method_name] = {
                    "status": "failed",
                    "error": result["error"]
                }
                continue
            
            method_perf = {
                "status": "success",
                "execution_time": result["execution_time"],
                "description": result["description"]
            }
            
            # Extract metrics if available
            if "results" in result and result["results"]:
                for task_name, task_results in result["results"].items():
                    if isinstance(task_results, dict):
                        method_perf[f"{task_name}_metrics"] = task_results
            
            report["method_performance"][method_name] = method_perf
        
        # Compare metrics across methods
        successful_methods = [name for name, result in all_results.items() if "error" not in result]
        
        if successful_methods:
            # Find common tasks across all successful methods
            common_tasks = None
            for method_name in successful_methods:
                method_tasks = set(all_results[method_name]["results"].keys())
                if common_tasks is None:
                    common_tasks = method_tasks
                else:
                    common_tasks = common_tasks.intersection(method_tasks)
            
            # Compare metrics for common tasks
            for task_name in common_tasks or []:
                task_comparison = {}
                for method_name in successful_methods:
                    task_results = all_results[method_name]["results"].get(task_name, {})
                    if isinstance(task_results, dict):
                        task_comparison[method_name] = task_results
                
                if task_comparison:
                    report["metric_comparison"][task_name] = task_comparison
        
        return report
    
    def save_markdown_report(self, report: Dict, report_file: str, all_results: Dict):
        """Save a human-readable markdown report"""
        
        with open(report_file, 'w') as f:
            f.write("# CoIR Search Methods Evaluation Report\n\n")
            
            # Summary
            f.write("## 📊 Evaluation Summary\n\n")
            summary = report["summary"]
            f.write(f"- **Total Methods Tested**: {summary['total_methods_tested']}\n")
            f.write(f"- **Successful Evaluations**: {summary['successful_evaluations']}\n")
            f.write(f"- **Failed Evaluations**: {summary['failed_evaluations']}\n")
            f.write(f"- **Total Execution Time**: {summary['total_execution_time']:.2f} seconds\n\n")
            
            # Method Details
            f.write("## 🔍 Search Methods Tested\n\n")
            for method_name, config in self.search_configs.items():
                if method_name in all_results:
                    result = all_results[method_name]
                    status = "✅ Success" if "error" not in result else "❌ Failed"
                    f.write(f"### {method_name}\n")
                    f.write(f"- **Description**: {config['description']}\n")
                    f.write(f"- **Status**: {status}\n")
                    f.write(f"- **Requires Model**: {config.get('requires_model', False)}\n")
                    
                    if "error" not in result:
                        f.write(f"- **Execution Time**: {result['execution_time']:.2f} seconds\n")
                    else:
                        f.write(f"- **Error**: {result['error']}\n")
                    f.write("\n")
            
            # Performance Comparison
            if "method_performance" in report:
                f.write("## ⚡ Performance Comparison\n\n")
                f.write("| Method | Status | Execution Time (s) | Description |\n")
                f.write("|--------|--------|-------------------|-------------|\n")
                
                for method_name, perf in report["method_performance"].items():
                    status = perf["status"]
                    exec_time = perf.get("execution_time", "N/A")
                    description = perf.get("description", "")
                    f.write(f"| {method_name} | {status} | {exec_time} | {description} |\n")
                f.write("\n")
            
            # Metrics Comparison
            if "metric_comparison" in report and report["metric_comparison"]:
                f.write("## 📈 Metrics Comparison\n\n")
                for task_name, task_metrics in report["metric_comparison"].items():
                    f.write(f"### {task_name}\n\n")
                    
                    # Extract NDCG@10 for comparison if available
                    ndcg_comparison = {}
                    for method_name, metrics in task_metrics.items():
                        if isinstance(metrics, dict) and "NDCG" in metrics:
                            ndcg_dict = metrics["NDCG"]
                            if isinstance(ndcg_dict, dict) and "NDCG@10" in ndcg_dict:
                                ndcg_comparison[method_name] = ndcg_dict["NDCG@10"]
                    
                    if ndcg_comparison:
                        f.write("**NDCG@10 Comparison:**\n\n")
                        f.write("| Method | NDCG@10 |\n")
                        f.write("|--------|----------|\n")
                        
                        # Sort by NDCG@10 score
                        sorted_methods = sorted(ndcg_comparison.items(), key=lambda x: x[1], reverse=True)
                        for method_name, score in sorted_methods:
                            f.write(f"| {method_name} | {score:.4f} |\n")
                        f.write("\n")
            
            # Configuration Details
            f.write("## ⚙️ Search Configurations\n\n")
            for method_name, result in all_results.items():
                if "search_config" in result:
                    f.write(f"### {method_name}\n")
                    f.write("```json\n")
                    f.write(json.dumps(result["search_config"], indent=2))
                    f.write("\n```\n\n")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate all CoIR search methods")
    parser.add_argument("--model", default="intfloat/e5-base-v2", 
                       help="Model name for dense/hybrid search")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--output-dir", default="search_methods_evaluation",
                       help="Output directory for results")
    parser.add_argument("--tasks", nargs="+", default=["codetrans-dl"],
                       help="Tasks to evaluate on")
    parser.add_argument("--methods", nargs="+", 
                       help="Specific methods to test (default: all)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick evaluation with basic methods only")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SearchMethodsEvaluator(
        model_name=args.model,
        batch_size=args.batch_size
    )
    
    # Determine methods to test
    methods_to_test = args.methods
    if args.quick:
        methods_to_test = ["dense", "jaccard", "bm25", "simple_hybrid"]
        logger.info("Quick mode: testing basic methods only")
    
    try:
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation(
            task_names=args.tasks,
            output_dir=args.output_dir,
            methods_to_test=methods_to_test
        )
        
        print("\n" + "="*80)
        print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📊 Results directory: {args.output_dir}")
        print(f"📋 Report file: {os.path.join(args.output_dir, 'evaluation_report.md')}")
        print(f"📄 Raw results: {os.path.join(args.output_dir, 'comprehensive_results.json')}")
        
        # Print quick summary
        successful = len([r for r in results.values() if "error" not in r])
        total = len(results)
        print(f"✅ Successful evaluations: {successful}/{total}")
        
        if successful < total:
            failed_methods = [name for name, result in results.items() if "error" in result]
            print(f"❌ Failed methods: {', '.join(failed_methods)}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()