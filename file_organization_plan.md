# CoIR Search Directory Organization Plan

## Overview
This document outlines the planned file structure for the CoIR search module after refactoring to support clean multi-modal search architecture.

## Current Directory Structure
```
coir/beir/retrieval/search/
├── __init__.py (existing - will be updated)
├── base.py (existing - minimal changes)
├── dense/
│   ├── __init__.py (existing)
│   ├── exact_search.py (PRESERVE ORIGINAL - no changes)
│   ├── exact_search_multi_gpu.py (existing)
│   ├── faiss_index.py (existing)
│   ├── faiss_search.py (existing)
│   └── util.py (existing)
├── lexical/
│   ├── __init__.py (existing)
│   ├── bm25_search.py (existing)
│   └── elastic_search.py (existing)
├── sparse/
│   ├── __init__.py (existing)
│   └── sparse_search.py (existing)
├── hybrid/ (NEW)
│   ├── __init__.py (new)
│   ├── simple_hybrid_search.py (planned)
│   └── advanced_hybrid_search.py (planned)
└── llm/ (NEW)
    ├── __init__.py (new)
    ├── async_ollama.py (planned)
    └── query_expander.py (planned)
```

## Search Paradigm Organization

### Dense Search (`dense/`)
- **Purpose**: Vector-based semantic search methods
- **Status**: Existing implementation - PRESERVE AS-IS
- **Files**: 
  - `exact_search.py` - Core dense retrieval (DO NOT MODIFY)
  - `exact_search_multi_gpu.py` - Multi-GPU support
  - `faiss_index.py` - FAISS indexing utilities
  - `faiss_search.py` - FAISS-based search
  - `util.py` - Dense search utilities

### Lexical Search (`lexical/`)
- **Purpose**: Traditional keyword-based search methods
- **Status**: Existing implementation with planned additions
- **Files**:
  - `bm25_search.py` - Existing BM25 implementation
  - `elastic_search.py` - Existing Elasticsearch integration
  - `jaccard_search.py` - Planned Jaccard similarity search

### Sparse Search (`sparse/`)
- **Purpose**: Sparse vector representations (SPLADE, etc.)
- **Status**: Existing implementation
- **Files**:
  - `sparse_search.py` - Existing sparse search implementation

### Hybrid Search (`hybrid/`) - NEW
- **Purpose**: Methods combining multiple search paradigms
- **Status**: New module to be implemented
- **Files**:
  - `simple_hybrid_search.py` - Basic hybrid combining dense + lexical
  - `advanced_hybrid_search.py` - Advanced fusion strategies

### LLM Search (`llm/`) - NEW
- **Purpose**: LLM-powered search and query enhancement
- **Status**: New module to be implemented
- **Files**:
  - `async_ollama.py` - Asynchronous Ollama integration
  - `query_expander.py` - LLM-based query expansion

## Import Strategy

### Main Search Module (`__init__.py`)
The main search module will expose key classes from each paradigm:

```python
# Existing imports
from .base import BaseSearch

# Dense search (preserve existing)
from .dense.exact_search import DenseRetrievalExactSearch

# Lexical search
from .lexical.bm25_search import BM25Search
# from .lexical.jaccard_search import LexicalJaccardSearch  # Future

# Sparse search (existing)
# from .sparse.sparse_search import SparseSearch  # If needed

# Hybrid search (new)
# from .hybrid.simple_hybrid_search import SimpleHybridSearch  # Future
# from .hybrid.advanced_hybrid_search import AdvancedHybridSearch  # Future

# LLM search (new)
# from .llm.async_ollama import AsyncOllamaSearch  # Future
# from .llm.query_expander import QueryExpander  # Future
```

## Migration Strategy

### Phase 1: Directory Setup (Current Subtask)
- ✅ Create `hybrid/` and `llm/` directories
- ✅ Add `__init__.py` files with placeholder imports
- ✅ Update main `__init__.py` with commented future imports

### Phase 2: Extract LLM Components (Next Subtask)
- Move LLM-related code to `llm/` module
- Implement async Ollama integration
- Create query expansion utilities

### Phase 3: Implement Hybrid Methods (Future Subtask)
- Create simple hybrid search combining dense + lexical
- Implement advanced fusion strategies
- Add configuration management for hybrid approaches

### Phase 4: Enhance Lexical Methods (Future Subtask)
- Add Jaccard similarity search
- Optimize existing BM25 implementation
- Add more lexical search variants

## Design Principles

1. **Separation of Concerns**: Each directory handles one search paradigm
2. **Backward Compatibility**: Existing imports continue to work
3. **Extensibility**: Easy to add new search methods within each paradigm
4. **Clean Interfaces**: All search classes inherit from `BaseSearch`
5. **Minimal Dependencies**: Each module has minimal cross-dependencies

## Notes

- The `dense/exact_search.py` file should remain unchanged to preserve existing functionality
- All new modules should follow the existing `BaseSearch` interface
- Import statements are initially commented to avoid import errors during development
- Each module's `__init__.py` exposes the main classes for that paradigm