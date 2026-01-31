#!/usr/bin/env python3
"""
Simple Search Methods Evaluation Script

This script demonstrates all available search methods in the refactored CoIR architecture,
following the same pattern as the README examples.
"""

import os
import sys
import json
import time

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coir.evaluation import COIR
from coir.data_loader import get_tasks
from coir.models import YourCustomDEModel


def main():
    """Simple evaluation of all search methods"""
    
    print("🚀 CoIR Search Methods Evaluation")
    print("=" * 50)
    
    # Model setup (following README pattern)
    model_name = "intfloat/e5-base-v2"
    print(f"📦 Loading model: {model_name}")
    model = YourCustomDEModel(model_name=model_name)
    
    # Get tasks (following README pattern)
    print("📋 Loading tasks...")
    tasks = get_tasks(tasks=["codetrans-dl"])
    
    # Define search configurations to test
    search_configs = [
        {"method": "dense", "name": "Dense Search (Original)"},
        {"method": "jaccard", "name": "Jaccard Similarity"},
        {"method": "bm25", "name": "BM25 Algorithm"},
        {"method": "simple_hybrid", "name": "Simple Hybrid"},
        {"method": "advanced_hybrid", "params": {"fusion_method": "rrf"}, "name": "Advanced Hybrid (RRF)"}
    ]
    
    results = {}
    
    print(f"\n🔍 Testing {len(search_configs)} search methods:")
    print("-" * 50)
    
    for i, config in enumerate(search_configs, 1):
        method_name = config["name"]
        search_config = {k: v for k, v in config.items() if k not in ["name"]}
        
        print(f"[{i}/{len(search_configs)}] {method_name}")
        
        try:
            # Initialize evaluation (following README pattern)
            evaluation = COIR(tasks=tasks, batch_size=32, search_config=search_config)
            
            # Run evaluation (following README pattern)
            start_time = time.time()
            result = evaluation.run(model, output_folder=f"results/{config['method']}")
            end_time = time.time()
            
            execution_time = end_time - start_time
            print(f"  ✅ Success ({execution_time:.2f}s)")
            
            # Store results
            results[config['method']] = {
                "name": method_name,
                "status": "success",
                "execution_time": execution_time,
                "results": result
            }
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            results[config['method']] = {
                "name": method_name,
                "status": "failed",
                "error": str(e)
            }
    
    # Print summary
    print("\n📊 Results Summary:")
    print("-" * 50)
    
    successful = 0
    for method, result in results.items():
        status = "✅" if result["status"] == "success" else "❌"
        name = result["name"]
        
        if result["status"] == "success":
            time_str = f"({result['execution_time']:.2f}s)"
            successful += 1
        else:
            time_str = "(failed)"
        
        print(f"{status} {name} {time_str}")
    
    print(f"\n🎯 Final Score: {successful}/{len(search_configs)} methods successful")
    
    # Save simple results
    os.makedirs("results", exist_ok=True)
    with open("results/simple_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: results/simple_evaluation_results.json")
    
    if successful == len(search_configs):
        print("🎉 All search methods working correctly!")
    else:
        print("⚠️  Some methods need attention - check the error messages above")


if __name__ == "__main__":
    main()