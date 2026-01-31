# CoIR Search Methods Evaluation Report

## 📊 Evaluation Summary

- **Total Methods Tested**: 4
- **Successful Evaluations**: 2
- **Failed Evaluations**: 2
- **Total Execution Time**: 840.40 seconds

## 🔍 Search Methods Tested

### dense
- **Description**: Dense/Semantic Search (Original CoIR)
- **Status**: ✅ Success
- **Requires Model**: True
- **Execution Time**: 606.82 seconds

### jaccard
- **Description**: Lexical Search using Jaccard Similarity
- **Status**: ❌ Failed
- **Requires Model**: False
- **Error**: LexicalJaccardSearch.search() takes 4 positional arguments but 5 were given

### bm25
- **Description**: Lexical Search using BM25 Algorithm
- **Status**: ❌ Failed
- **Requires Model**: False
- **Error**: LexicalBM25Search.search() takes 4 positional arguments but 5 were given

### simple_hybrid
- **Description**: Simple Hybrid Search (Dense + Lexical)
- **Status**: ✅ Success
- **Requires Model**: True
- **Execution Time**: 233.58 seconds

## ⚡ Performance Comparison

| Method | Status | Execution Time (s) | Description |
|--------|--------|-------------------|-------------|
| dense | success | 606.8173499107361 | Dense/Semantic Search (Original CoIR) |
| jaccard | failed | N/A |  |
| bm25 | failed | N/A |  |
| simple_hybrid | success | 233.5829758644104 | Simple Hybrid Search (Dense + Lexical) |

## 📈 Metrics Comparison

### codetrans-dl

**NDCG@10 Comparison:**

| Method | NDCG@10 |
|--------|----------|
| dense | 0.1098 |
| simple_hybrid | 0.1098 |

## ⚙️ Search Configurations

### dense
```json
{
  "method": "dense"
}
```

### simple_hybrid
```json
{
  "method": "simple_hybrid"
}
```

