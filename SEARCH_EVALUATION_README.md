# CoIR Search Methods Evaluation Script

This script provides comprehensive evaluation of all available search methods in the refactored CoIR architecture.

## 🔍 Available Search Methods

The script evaluates the following search methods:

### Dense/Semantic Search
- **dense**: Original CoIR dense search using embeddings
- **semantic**: Alias for dense search

### Lexical Search  
- **jaccard**: Jaccard similarity-based lexical search
- **bm25**: BM25 algorithm-based lexical search
- **bm25_tuned**: BM25 with optimized parameters (k1=1.5, b=0.8)

### Hybrid Search
- **simple_hybrid**: Simple combination of dense and lexical search
- **advanced_hybrid_rrf**: Advanced hybrid using Reciprocal Rank Fusion
- **advanced_hybrid_weighted**: Advanced hybrid with weighted fusion (70% dense, 30% lexical)

## 🚀 Usage

### Basic Usage
```bash
# Evaluate all search methods on default task
python3 evaluate_all_search_methods.py

# Quick evaluation with basic methods only
python3 evaluate_all_search_methods.py --quick
```

### Advanced Usage
```bash
# Specify custom model and tasks
python3 evaluate_all_search_methods.py \
    --model "intfloat/e5-base-v2" \
    --tasks codetrans-dl stackoverflow-qa \
    --batch-size 64 \
    --output-dir my_evaluation_results

# Test specific methods only
python3 evaluate_all_search_methods.py \
    --methods dense jaccard bm25 simple_hybrid \
    --output-dir focused_evaluation

# Evaluate on multiple tasks
python3 evaluate_all_search_methods.py \
    --tasks codetrans-dl apps cosqa \
    --output-dir multi_task_evaluation
```

## 📊 Output

The script generates:

1. **comprehensive_results.json**: Complete raw results in JSON format
2. **evaluation_report.md**: Human-readable markdown report with:
   - Evaluation summary
   - Method descriptions and status
   - Performance comparison table
   - Metrics comparison (NDCG@10, MAP@10, etc.)
   - Configuration details

3. **Individual method directories**: Separate folders for each search method with detailed results

4. **search_evaluation.log**: Detailed execution log

## 📈 Example Report Structure

```
search_methods_evaluation/
├── comprehensive_results.json
├── evaluation_report.md
├── search_evaluation.log
├── dense/
│   └── codetrans-dl.json
├── jaccard/
│   └── codetrans-dl.json
├── bm25/
│   └── codetrans-dl.json
├── simple_hybrid/
│   └── codetrans-dl.json
└── advanced_hybrid_rrf/
    └── codetrans-dl.json
```

## 🎯 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model name for dense/hybrid search | `intfloat/e5-base-v2` |
| `--batch-size` | Batch size for processing | `32` |
| `--output-dir` | Output directory for results | `search_methods_evaluation` |
| `--tasks` | Tasks to evaluate on | `["codetrans-dl"]` |
| `--methods` | Specific methods to test | All methods |
| `--quick` | Run quick evaluation with basic methods | `False` |

## 📋 Available Tasks

Based on the CoIR benchmark, you can evaluate on:

- `codetrans-dl`: Code translation dataset
- `stackoverflow-qa`: Stack Overflow Q&A dataset  
- `apps`: Programming problems dataset
- `codefeedback-mt`: Code feedback (multi-turn)
- `codefeedback-st`: Code feedback (single-turn)
- `codetrans-contest`: Contest code translation
- `synthetic-text2sql`: Text-to-SQL synthesis
- `cosqa`: Code search Q&A
- `codesearchnet`: CodeSearchNet dataset
- `codesearchnet-ccr`: CodeSearchNet code-code retrieval

## 🔧 Requirements

The script requires all dependencies from the refactored CoIR architecture:

```bash
pip install torch datasets sentence-transformers faiss-cpu pytrec_eval bm25s ollama
```

## 💡 Example Workflows

### Research Comparison
Compare all search methods to understand their relative performance:
```bash
python3 evaluate_all_search_methods.py \
    --tasks codetrans-dl cosqa \
    --output-dir research_comparison
```

### Production Evaluation
Test specific methods for production deployment:
```bash
python3 evaluate_all_search_methods.py \
    --methods dense bm25 advanced_hybrid_rrf \
    --tasks codetrans-dl stackoverflow-qa apps \
    --output-dir production_evaluation
```

### Parameter Tuning
Compare different configurations:
```bash
python3 evaluate_all_search_methods.py \
    --methods bm25 bm25_tuned advanced_hybrid_rrf advanced_hybrid_weighted \
    --output-dir parameter_tuning
```

## 📊 Metrics Reported

For each search method and task, the script reports:

- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MAP@k**: Mean Average Precision  
- **Recall@k**: Recall at k
- **Precision@k**: Precision at k

Where k typically includes values: 1, 3, 5, 10, 100, 1000

## 🚨 Troubleshooting

### Common Issues

1. **Model loading fails**: Ensure you have sufficient memory and the model name is correct
2. **Import errors**: Install all required dependencies
3. **CUDA out of memory**: Reduce batch size with `--batch-size 16`
4. **Task loading fails**: Check internet connection for Hugging Face dataset downloads

### Performance Tips

- Use `--quick` for faster evaluation during development
- Reduce batch size if running out of memory
- Start with a single task before evaluating multiple tasks
- Use specific `--methods` to focus on methods of interest

## 🎉 Integration with CoIR

This script is fully integrated with the refactored CoIR architecture and:

- ✅ Maintains backward compatibility with original CoIR interface
- ✅ Supports all new search methods from the refactoring
- ✅ Uses the SearchMethodFactory for consistent method creation
- ✅ Provides comprehensive metrics comparison
- ✅ Generates publication-ready evaluation reports

The script serves as both a validation tool for the refactoring and a practical utility for researchers and practitioners to compare different search approaches on code retrieval tasks.