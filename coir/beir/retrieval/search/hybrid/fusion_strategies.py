from typing import Dict
import numpy as np
from scipy import stats


class FusionStrategies:
    """
    Collection of fusion strategies for combining semantic and lexical search results.
    
    This class provides static methods for different fusion approaches:
    - Reciprocal Rank Fusion (RRF)
    - Weighted Score Fusion
    - Score Interpolation
    - CombMNZ Fusion
    """
    
    @staticmethod
    def reciprocal_rank_fusion(semantic_results: Dict, lexical_results: Dict, 
                              top_k: int, k: int = 60) -> Dict:
        """
        Reciprocal Rank Fusion (RRF): score = 1/(rank + k)
        
        Args:
            semantic_results: Dictionary mapping query IDs to document scores from semantic search
            lexical_results: Dictionary mapping query IDs to document scores from lexical search
            top_k: Number of top results to return
            k: RRF parameter (default: 60)
            
        Returns:
            Dictionary mapping query IDs to fused document scores
        """
        fused_results = {}
        
        for qid in semantic_results:
            fused_results[qid] = {}
            
            # Get ranked lists from both methods
            semantic_ranked = sorted(semantic_results[qid].items(), key=lambda x: x[1], reverse=True)
            lexical_ranked = sorted(lexical_results.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
            
            # Create rank mappings
            semantic_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(semantic_ranked)}
            lexical_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(lexical_ranked)}
            
            # Get all unique document IDs
            all_docs = set(semantic_ranks.keys()) | set(lexical_ranks.keys())
            
            # Calculate RRF scores
            for doc_id in all_docs:
                semantic_rank = semantic_ranks.get(doc_id, len(semantic_ranked) + 1)
                lexical_rank = lexical_ranks.get(doc_id, len(lexical_ranked) + 1)
                
                rrf_score = (1.0 / (semantic_rank + k)) + (1.0 / (lexical_rank + k))
                fused_results[qid][doc_id] = rrf_score
            
            # Keep only top-k results
            if len(fused_results[qid]) > top_k:
                top_items = sorted(fused_results[qid].items(), key=lambda x: x[1], reverse=True)[:top_k]
                fused_results[qid] = dict(top_items)
        
        return fused_results
    
    @staticmethod
    def weighted_score_fusion(semantic_results: Dict, lexical_results: Dict, 
                             top_k: int, alpha: float = 0.5) -> Dict:
        """
        Weighted average of normalized scores.
        
        Args:
            semantic_results: Dictionary mapping query IDs to document scores from semantic search
            lexical_results: Dictionary mapping query IDs to document scores from lexical search
            top_k: Number of top results to return
            alpha: Weight for semantic scores (1-alpha for lexical scores)
            
        Returns:
            Dictionary mapping query IDs to fused document scores
        """
        fused_results = {}
        
        for qid in semantic_results:
            fused_results[qid] = {}
            
            semantic_scores = semantic_results[qid]
            lexical_scores = lexical_results.get(qid, {})
            
            # Normalize scores to [0, 1] range
            if semantic_scores:
                sem_values = list(semantic_scores.values())
                sem_min, sem_max = min(sem_values), max(sem_values)
                sem_range = sem_max - sem_min if sem_max != sem_min else 1.0
                semantic_norm = {doc_id: (score - sem_min) / sem_range 
                               for doc_id, score in semantic_scores.items()}
            else:
                semantic_norm = {}
            
            if lexical_scores:
                lex_values = list(lexical_scores.values())
                lex_min, lex_max = min(lex_values), max(lex_values)
                lex_range = lex_max - lex_min if lex_max != lex_min else 1.0
                lexical_norm = {doc_id: (score - lex_min) / lex_range 
                              for doc_id, score in lexical_scores.items()}
            else:
                lexical_norm = {}
            
            # Get all unique document IDs
            all_docs = set(semantic_norm.keys()) | set(lexical_norm.keys())
            
            # Calculate weighted fusion
            for doc_id in all_docs:
                sem_score = semantic_norm.get(doc_id, 0.0)
                lex_score = lexical_norm.get(doc_id, 0.0)
                
                fused_score = alpha * sem_score + (1 - alpha) * lex_score
                fused_results[qid][doc_id] = fused_score
            
            # Keep only top-k results
            if len(fused_results[qid]) > top_k:
                top_items = sorted(fused_results[qid].items(), key=lambda x: x[1], reverse=True)[:top_k]
                fused_results[qid] = dict(top_items)
        
        return fused_results
    
    @staticmethod
    def score_interpolation(semantic_results: Dict, lexical_results: Dict, 
                           top_k: int, alpha: float = 0.5) -> Dict:
        """
        Linear interpolation of scores without normalization.
        
        Args:
            semantic_results: Dictionary mapping query IDs to document scores from semantic search
            lexical_results: Dictionary mapping query IDs to document scores from lexical search
            top_k: Number of top results to return
            alpha: Weight for semantic scores (1-alpha for lexical scores)
            
        Returns:
            Dictionary mapping query IDs to fused document scores
        """
        fused_results = {}
        
        for qid in semantic_results:
            fused_results[qid] = {}
            
            semantic_scores = semantic_results[qid]
            lexical_scores = lexical_results.get(qid, {})
            
            # Get all unique document IDs
            all_docs = set(semantic_scores.keys()) | set(lexical_scores.keys())
            
            # Calculate interpolated scores
            for doc_id in all_docs:
                sem_score = semantic_scores.get(doc_id, 0.0)
                lex_score = lexical_scores.get(doc_id, 0.0)
                
                interpolated_score = alpha * sem_score + (1 - alpha) * lex_score
                fused_results[qid][doc_id] = interpolated_score
            
            # Keep only top-k results
            if len(fused_results[qid]) > top_k:
                top_items = sorted(fused_results[qid].items(), key=lambda x: x[1], reverse=True)[:top_k]
                fused_results[qid] = dict(top_items)
        
        return fused_results
    
    @staticmethod
    def combMNZ_fusion(semantic_results: Dict, lexical_results: Dict, 
                       top_k: int) -> Dict:
        """
        CombMNZ with z-score normalization.
        Formula: sum(scores) * num_systems
        
        Args:
            semantic_results: Dictionary mapping query IDs to document scores from semantic search
            lexical_results: Dictionary mapping query IDs to document scores from lexical search
            top_k: Number of top results to return
            
        Returns:
            Dictionary mapping query IDs to fused document scores
        """
        fused_results = {}
        
        for qid in semantic_results:
            fused_results[qid] = {}
            
            semantic_scores = semantic_results[qid]
            lexical_scores = lexical_results.get(qid, {})
            
            # Z-score normalization
            if semantic_scores:
                sem_values = list(semantic_scores.values())
                if len(sem_values) > 1:
                    sem_mean = np.mean(sem_values)
                    sem_std = np.std(sem_values)
                    if sem_std > 0:
                        semantic_norm = {doc_id: (score - sem_mean) / sem_std 
                                       for doc_id, score in semantic_scores.items()}
                    else:
                        semantic_norm = {doc_id: 0.0 for doc_id in semantic_scores}
                else:
                    semantic_norm = {doc_id: 0.0 for doc_id in semantic_scores}
            else:
                semantic_norm = {}
            
            if lexical_scores:
                lex_values = list(lexical_scores.values())
                if len(lex_values) > 1:
                    lex_mean = np.mean(lex_values)
                    lex_std = np.std(lex_values)
                    if lex_std > 0:
                        lexical_norm = {doc_id: (score - lex_mean) / lex_std 
                                      for doc_id, score in lexical_scores.items()}
                    else:
                        lexical_norm = {doc_id: 0.0 for doc_id in lexical_scores}
                else:
                    lexical_norm = {doc_id: 0.0 for doc_id in lexical_scores}
            else:
                lexical_norm = {}
            
            # Get all unique document IDs
            all_docs = set(semantic_norm.keys()) | set(lexical_norm.keys())
            
            # Calculate CombMNZ scores
            for doc_id in all_docs:
                sem_score = semantic_norm.get(doc_id, 0.0)
                lex_score = lexical_norm.get(doc_id, 0.0)
                
                # Count number of systems that retrieved this document
                num_systems = (1 if doc_id in semantic_norm else 0) + (1 if doc_id in lexical_norm else 0)
                
                # CombMNZ: sum of scores * number of systems
                combmnz_score = (sem_score + lex_score) * num_systems
                fused_results[qid][doc_id] = combmnz_score
            
            # Keep only top-k results
            if len(fused_results[qid]) > top_k:
                top_items = sorted(fused_results[qid].items(), key=lambda x: x[1], reverse=True)[:top_k]
                fused_results[qid] = dict(top_items)
        
        return fused_results