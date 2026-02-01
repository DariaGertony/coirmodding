from typing import Dict
import heapq
from ..base import BaseSearch
from ..dense.exact_search import DenseRetrievalExactSearch
from ..lexical.jaccard_search import LexicalJaccardSearch


class SimpleHybridSearch(BaseSearch):
    """
    Simple hybrid search that combines semantic and lexical search results using a 50/50 merge strategy.
    
    This implementation alternates between taking results from semantic search (dense retrieval)
    and lexical search (Jaccard similarity) to create a balanced hybrid approach.
    
    Algorithm:
    1. Run semantic search (dense retrieval)
    2. Run lexical search (Jaccard similarity)
    3. Convert results to max-heaps for efficient processing
    4. Alternate selection: semantic → lexical → semantic...
    5. Deduplicate and return top-k results
    """
    
    def __init__(self, model, batch_size: int = 128, **kwargs):
        """
        Initialize the simple hybrid search.
        
        Args:
            model: Dense retrieval model that provides encode_corpus() and encode_queries()
            batch_size: Batch size for processing
            **kwargs: Additional keyword arguments passed to underlying search methods
        """
        self.semantic_search = DenseRetrievalExactSearch(model, batch_size, **kwargs)
        self.lexical_search = LexicalJaccardSearch(batch_size, **kwargs)
        self.results = {}
    
    def search(self, corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str = "cos_sim",
               **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Implement simple 50/50 hybrid search.
        
        Args:
            corpus: Dictionary mapping corpus IDs to documents with 'title' and 'text' fields
            queries: Dictionary mapping query IDs to query strings
            top_k: Number of top results to return per query
            score_function: Scoring function for semantic search ("cos_sim" or "dot")
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
        
        # Run lexical search (Jaccard similarity)
        lexical_results = self.lexical_search.search(
            corpus, queries, top_k, **kwargs
        )
        
        # Merge results using 50/50 strategy
        for qid in query_ids:
            semantic_scores = semantic_results.get(qid, {})
            lexical_scores = lexical_results.get(qid, {})
            
            # Convert to max-heaps for efficient processing
            semantic_heap = [(score, corpus_id) for corpus_id, score in semantic_scores.items()]
            lexical_heap = [(score, corpus_id) for corpus_id, score in lexical_scores.items()]
            
            # Convert to max-heaps (negate scores for min-heap to work as max-heap)
            semantic_heap = [(-score, corpus_id) for score, corpus_id in semantic_heap]
            lexical_heap = [(-score, corpus_id) for score, corpus_id in lexical_heap]
            
            heapq.heapify(semantic_heap)
            heapq.heapify(lexical_heap)
            
            # Determine the border (how many results to take)
            max_lexical = len(lexical_heap)
            max_semantic = len(semantic_heap)
            border = min(top_k, max_lexical + max_semantic)
            
            count = 0
            added_corpus_ids = set()
            
            # Alternate between semantic and lexical results
            while count < border and (semantic_heap or lexical_heap):
                # Try to take from semantic first (if we haven't reached half)
                if count < border // 2 and semantic_heap:
                    neg_score, corpus_id = heapq.heappop(semantic_heap)
                    score = -neg_score  # Convert back to positive
                    
                    if corpus_id not in added_corpus_ids:
                        self.results[qid][corpus_id] = score
                        added_corpus_ids.add(corpus_id)
                        count += 1
                
                # Then try to take from lexical
                elif lexical_heap:
                    neg_score, corpus_id = heapq.heappop(lexical_heap)
                    score = -neg_score  # Convert back to positive
                    
                    if corpus_id not in added_corpus_ids:
                        self.results[qid][corpus_id] = score
                        added_corpus_ids.add(corpus_id)
                        count += 1
                
                # If lexical is exhausted, continue with semantic
                elif semantic_heap:
                    neg_score, corpus_id = heapq.heappop(semantic_heap)
                    score = -neg_score  # Convert back to positive
                    
                    if corpus_id not in added_corpus_ids:
                        self.results[qid][corpus_id] = score
                        added_corpus_ids.add(corpus_id)
                        count += 1
                
                # If both are exhausted, break
                else:
                    break
        
        return self.results