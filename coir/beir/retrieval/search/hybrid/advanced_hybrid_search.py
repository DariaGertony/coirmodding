from typing import Dict
import torch
from ..base import BaseSearch
from ..dense.exact_search import DenseRetrievalExactSearch
from ..lexical.bm25_search import LexicalBM25Search
from ....reranking.rerank import Rerank
from ....reranking.models.cross_encoder import CrossEncoder
from .fusion_strategies import FusionStrategies


class AdvancedHybridSearch(BaseSearch):
    """
    Advanced hybrid search that combines semantic and lexical search results using multiple fusion strategies.
    
    This implementation supports:
    - Multiple fusion strategies (RRF, weighted, interpolation, CombMNZ)
    - BM25 + semantic combination for better performance
    - Optional reranking integration with cross-encoder models
    - Configurable fusion method selection
    
    Algorithm:
    1. Run semantic search (dense retrieval)
    2. Run BM25 lexical search
    3. Optional: Rerank lexical results with cross-encoder
    4. Apply selected fusion strategy
    5. Return fused top-k results
    """
    
    def __init__(self, model, batch_size: int = 128, 
                 fusion_method: str = 'rrf',
                 rerank_model: str = "BAAI/bge-reranker-base",
                 use_reranking: bool = False,
                 **kwargs):
        """
        Initialize the advanced hybrid search.
        
        Args:
            model: Dense retrieval model that provides encode_corpus() and encode_queries()
            batch_size: Batch size for processing
            fusion_method: Fusion strategy to use ('rrf', 'weighted', 'interpolation', 'combMNZ')
            rerank_model: Cross-encoder model for reranking
            use_reranking: Whether to use reranking on lexical results
            **kwargs: Additional keyword arguments passed to underlying search methods
        """
        self.semantic_search = DenseRetrievalExactSearch(model, batch_size, **kwargs)
        self.lexical_search = LexicalBM25Search(batch_size, **kwargs)
        self.fusion_method = fusion_method
        self.rerank_model = rerank_model
        self.use_reranking = use_reranking
        self.results = {}
        
        # Initialize reranker if needed
        if self.use_reranking:
            try:
                cross_encoder = CrossEncoder(self.rerank_model)
                self.reranker = Rerank(cross_encoder, batch_size)
            except Exception as e:
                print(f"Warning: Could not initialize reranker with {rerank_model}: {e}")
                self.use_reranking = False
                self.reranker = None
        else:
            self.reranker = None
    
    def search(self, corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str = "cos_sim",
               fusion_alpha: float = 0.5,
               rrf_k: int = 60,
               **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Implement advanced hybrid search with fusion strategies.
        
        Args:
            corpus: Dictionary mapping corpus IDs to documents with 'title' and 'text' fields
            queries: Dictionary mapping query IDs to query strings
            top_k: Number of top results to return per query
            score_function: Scoring function for semantic search ("cos_sim" or "dot")
            fusion_alpha: Alpha parameter for weighted/interpolation fusion (default: 0.5)
            rrf_k: K parameter for RRF fusion (default: 60)
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary mapping query IDs to dictionaries of document IDs and scores
        """
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        
        # Run semantic search (dense retrieval)
        semantic_results = self.semantic_search.search(
            corpus, queries, top_k, score_function, **kwargs
        )
        
        # Run lexical search (BM25)
        lexical_results = self.lexical_search.search(
            corpus, queries, top_k, **kwargs
        )
        
        # Optional: Rerank lexical results
        if self.use_reranking and self.reranker is not None:
            try:
                lexical_results = self.reranker.rerank(
                    corpus, queries, lexical_results, top_k
                )
            except Exception as e:
                print(f"Warning: Reranking failed: {e}")
                # Continue without reranking
        
        # Apply fusion strategy
        if self.fusion_method == 'rrf':
            fused_results = FusionStrategies.reciprocal_rank_fusion(
                semantic_results, lexical_results, top_k, rrf_k
            )
        elif self.fusion_method == 'weighted':
            fused_results = FusionStrategies.weighted_score_fusion(
                semantic_results, lexical_results, top_k, fusion_alpha
            )
        elif self.fusion_method == 'interpolation':
            fused_results = FusionStrategies.score_interpolation(
                semantic_results, lexical_results, top_k, fusion_alpha
            )
        elif self.fusion_method == 'combMNZ':
            fused_results = FusionStrategies.combMNZ_fusion(
                semantic_results, lexical_results, top_k
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}. "
                           f"Supported methods: 'rrf', 'weighted', 'interpolation', 'combMNZ'")
        
        self.results = fused_results
        return self.results
    
    def set_fusion_method(self, fusion_method: str):
        """
        Change the fusion method at runtime.
        
        Args:
            fusion_method: New fusion strategy ('rrf', 'weighted', 'interpolation', 'combMNZ')
        """
        supported_methods = ['rrf', 'weighted', 'interpolation', 'combMNZ']
        if fusion_method not in supported_methods:
            raise ValueError(f"Unknown fusion method: {fusion_method}. "
                           f"Supported methods: {supported_methods}")
        self.fusion_method = fusion_method
    
    def enable_reranking(self, rerank_model: str = None):
        """
        Enable reranking with optional model change.
        
        Args:
            rerank_model: Optional new reranking model to use
        """
        if rerank_model:
            self.rerank_model = rerank_model
        
        try:
            cross_encoder = CrossEncoder(self.rerank_model)
            self.reranker = Rerank(cross_encoder, self.semantic_search.batch_size)
            self.use_reranking = True
        except Exception as e:
            print(f"Warning: Could not initialize reranker with {self.rerank_model}: {e}")
            self.use_reranking = False
            self.reranker = None
    
    def disable_reranking(self):
        """
        Disable reranking functionality.
        """
        self.use_reranking = False
        self.reranker = None
    
    def get_fusion_method(self) -> str:
        """
        Get the current fusion method.
        
        Returns:
            Current fusion method name
        """
        return self.fusion_method
    
    def is_reranking_enabled(self) -> bool:
        """
        Check if reranking is currently enabled.
        
        Returns:
            True if reranking is enabled, False otherwise
        """
        return self.use_reranking and self.reranker is not None