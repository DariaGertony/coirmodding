# Simple Search Methods Evaluation

A straightforward script to test all search methods in the refactored CoIR architecture.

## 🚀 Quick Start

```bash
# Run the simple evaluation
python3 simple_search_evaluation.py
```

## 📋 What it does

The script follows the same pattern as the README examples and tests:

1. **Dense Search** - Original CoIR semantic search
2. **Jaccard Similarity** - Lexical search using Jaccard
3. **BM25 Algorithm** - Lexical search using BM25
4. **Simple Hybrid** - Combination of dense + lexical
5. **Advanced Hybrid** - Advanced fusion with RRF

## 📊 Output

```
🚀 CoIR Search Methods Evaluation
==================================================
📦 Loading model: intfloat/e5-base-v2
📋 Loading tasks...

🔍 Testing 5 search methods:
--------------------------------------------------
[1/5] Dense Search (Original)
  ✅ Success (15.23s)
[2/5] Jaccard Similarity
  ✅ Success (2.45s)
[3/5] BM25 Algorithm
  ✅ Success (3.12s)
[4/5] Simple Hybrid
  ✅ Success (18.67s)
[5/5] Advanced Hybrid (RRF)
  ✅ Success (21.34s)

📊 Results Summary:
--------------------------------------------------
✅ Dense Search (Original) (15.23s)
✅ Jaccard Similarity (2.45s)
✅ BM25 Algorithm (3.12s)
✅ Simple Hybrid (18.67s)
✅ Advanced Hybrid (RRF) (21.34s)

🎯 Final Score: 5/5 methods successful
💾 Results saved to: results/simple_evaluation_results.json
🎉 All search methods working correctly!
```

## 📁 Generated Files

- `results/simple_evaluation_results.json` - Complete results in JSON format
- `results/dense/` - Dense search detailed results
- `results/jaccard/` - Jaccard search detailed results  
- `results/bm25/` - BM25 search detailed results
- `results/simple_hybrid/` - Simple hybrid detailed results
- `results/advanced_hybrid/` - Advanced hybrid detailed results

## 🔧 Code Pattern

The script follows the exact same pattern as the README:

```python
# Load the model (same as README)
model = YourCustomDEModel(model_name="intfloat/e5-base-v2")

# Get tasks (same as README)
tasks = get_tasks(tasks=["codetrans-dl"])

# Initialize evaluation (same as README)
evaluation = COIR(tasks=tasks, batch_size=32, search_config=search_config)

# Run evaluation (same as README)
results = evaluation.run(model, output_folder=f"results/{method}")
```

## ✨ Features

- **Simple**: Just run one command
- **Clear Output**: Easy to read progress and results
- **Error Handling**: Shows which methods work and which need fixes
- **JSON Results**: Machine-readable output for further analysis
- **README Compatible**: Uses the exact same API as documented

This script validates that the refactored CoIR architecture works correctly and all search methods are properly integrated.