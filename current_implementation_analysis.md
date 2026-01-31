# Current Implementation Analysis

## Overview
This document analyzes the current state of the CoIR framework before refactoring. The implementation contains messy multi-modal functionality that needs to be cleaned up and properly structured.

## Modified Files Summary

### 1. [`coir/beir/retrieval/search/dense/exact_search.py`](coir/beir/retrieval/search/dense/exact_search.py)
**Status**: Heavily modified with extensive new functionality
**Changes**:
- Added new imports: `bm25s`, `ollama`, `threading`, `tqdm`
- Added reranking imports: `Rerank`, `CrossEncoder`
- Introduced `AsyncOllama` class for LLM query expansion
- Added multiple search methods:
  - `lexical_search_bm25()` - BM25-based lexical search
  - `lexical_search()` - Jaccard similarity-based lexical search
  - `lexical_search_fh()` - Heap-based lexical search variant
  - `search_fh()` - Heap-based semantic search variant
  - `hibrid_search()` - Hybrid search combining semantic and lexical
  - `hibrid_search_with_bm25()` - Hybrid search with BM25 and reranking
- Added fusion methods:
  - `_reciprocal_rank_fusion()` - RRF fusion algorithm
  - `_weighted_score_fusion()` - Weighted score fusion
  - `_score_interpolation()` - Linear interpolation fusion
  - `_combMNZ_fusion()` - CombMNZ fusion algorithm
- Modified main `search()` method to support:
  - LLM query expansion via Ollama
  - Multiple search types (semantic, lexical, hybrid, hybrid_bm25)
  - Progress bars and async processing

### 2. [`coir/beir/retrieval/search/base.py`](coir/beir/retrieval/search/base.py)
**Status**: Extended with new abstract methods
**Changes**:
- Added abstract methods for new search functionality:
  - `lexical_search()`
  - `lexical_search_bm25()`
  - `hibrid_search()`
  - `hibrid_search_with_bm25()`

### 3. [`coir/beir/retrieval/evaluation.py`](coir/beir/retrieval/evaluation.py)
**Status**: Modified to support LLM integration
**Changes**:
- Removed logger initialization
- Modified `retrieve()` method to accept LLM parameters: `useLLm`, `llmname`, `prompt`
- Updated `rerank()` method with LLM support
- Added commented-out alternative search method calls
- Removed logging statements from evaluation methods

### 4. [`coir/evaluation.py`](coir/evaluation.py)
**Status**: Completely rewritten for multi-modal support
**Changes**:
- Added torch import and device detection
- Modified `COIR` class constructor to accept `type` parameter
- Completely rewrote `run()` method to support:
  - LLM parameters: `useLLm`, `llmname`, `prompt`, `to_rerank`
  - Different search types (semantic, lexical, hybrid)
  - Complex JSON output structure with nested results
  - Conditional result saving based on LLM usage

## New Dependencies Introduced

### External Libraries
- **`bm25s`**: BM25 search implementation
- **`ollama`**: LLM integration for query expansion
- **`threading`**: Async LLM processing
- **`tqdm`**: Progress bars

### Internal Dependencies
- **`coir.beir.reranking.rerank.Rerank`**: Reranking functionality
- **`coir.beir.reranking.models.cross_encoder.CrossEncoder`**: Cross-encoder reranking

## New Functionality Added

### 1. Multi-Modal Search Support
- **Semantic Search**: Original dense retrieval functionality
- **Lexical Search**: Jaccard similarity and BM25-based search
- **Hybrid Search**: Multiple fusion algorithms combining semantic and lexical results

### 2. LLM Query Expansion
- Async query expansion using Ollama
- Configurable LLM models and prompts
- Progress tracking for query processing

### 3. Advanced Fusion Algorithms
- **Reciprocal Rank Fusion (RRF)**: Standard rank-based fusion
- **Weighted Score Fusion**: Normalized score combination
- **Score Interpolation**: Linear interpolation of scores
- **CombMNZ**: Score sum multiplied by number of contributing systems

### 4. Reranking Integration
- Cross-encoder reranking for lexical results
- Configurable reranking models
- GPU/CPU device detection

### 5. Enhanced Result Management
- Complex JSON output structure
- Support for multiple model comparisons
- Conditional result saving and loading

## Code Quality Issues Identified

### 1. Architecture Problems
- Massive monolithic class with too many responsibilities
- Mixed concerns (search, fusion, LLM integration, reranking)
- Inconsistent naming conventions (`hibrid` vs `hybrid`)
- Hard-coded parameters scattered throughout

### 2. Code Duplication
- Multiple similar search methods with slight variations
- Repeated corpus/query processing logic
- Duplicated heap management code

### 3. Error Handling
- Minimal error handling for LLM operations
- No validation for fusion method parameters
- Missing checks for empty results

### 4. Performance Issues
- Inefficient corpus sorting in multiple places
- Potential memory issues with large corpora
- No caching of embeddings or results

### 5. Maintainability Issues
- Russian comments mixed with English
- Commented-out code blocks
- Complex nested conditionals
- No clear separation of concerns

## Recommendations for Refactoring

### 1. Modular Architecture
- Separate search strategies into individual classes
- Create dedicated fusion algorithm implementations
- Extract LLM integration into separate service
- Implement proper dependency injection

### 2. Clean Interfaces
- Define clear abstractions for each component
- Standardize method signatures
- Implement proper error handling
- Add comprehensive logging

### 3. Configuration Management
- Externalize all configuration parameters
- Implement validation for configuration values
- Support for different deployment environments

### 4. Performance Optimization
- Implement result caching
- Optimize corpus processing
- Add batch processing capabilities
- Memory usage optimization

### 5. Testing Strategy
- Unit tests for each component
- Integration tests for search pipelines
- Performance benchmarks
- Mock LLM services for testing

## Backup Information
- **Backup Branch**: `backup-messy-implementation`
- **Commit Hash**: `4804d60`
- **Files Backed Up**: 18 files with 5,238 insertions and 76 deletions
- **Backup Date**: 2026-01-31

This backup preserves all current functionality while allowing for clean refactoring on the main branch.