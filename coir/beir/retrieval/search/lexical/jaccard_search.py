from typing import Dict
import heapq
from ..base import BaseSearch


class LexicalJaccardSearch(BaseSearch):
    """
    Lexical search implementation using Jaccard similarity coefficient.
    
    The Jaccard similarity is calculated as the size of the intersection 
    divided by the size of the union of two sets of words.
    """
    
    def __init__(self, batch_size: int = 128, **kwargs):
        """
        Initialize the Jaccard search.
        
        Args:
            batch_size: Batch size for processing (kept for interface compatibility)
            **kwargs: Additional keyword arguments
        """
        self.batch_size = batch_size
        self.results = {}
    
    def search(self,
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str],
               top_k: int,
               score_function: str = "cos_sim",
               **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Perform Jaccard similarity-based search.
        
        Args:
            corpus: Dictionary mapping corpus IDs to documents with 'title' and 'text' fields
            queries: Dictionary mapping query IDs to query strings
            top_k: Number of top results to return per query
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary mapping query IDs to dictionaries of document IDs and scores
        """
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        
        # Convert queries to list for iteration
        query_texts = [queries[qid] for qid in query_ids]
        
        # Sort corpus by document length (longest first) for consistency with other implementations
        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus_docs = [corpus[cid] for cid in corpus_ids]
        
        # Extract document texts
        docs = [doc.get('text', '') for doc in corpus_docs]
        
        # Use heaps to maintain top-k results for each query
        result_heaps = {qid: [] for qid in query_ids}
        
        for query_iter in range(len(query_texts)):
            query_id = query_ids[query_iter]
            # Convert query to word set
            q_set = set(query_texts[query_iter].split())
            
            for doc_itr in range(len(docs)):
                corpus_id = corpus_ids[doc_itr]
                # Convert document to word set
                d_set = set(docs[doc_itr].split())
                
                # Calculate Jaccard similarity: intersection / union
                intersection = len(q_set.intersection(d_set))
                union = len(q_set.union(d_set))
                score = intersection / union if union > 0 else 0.0
                
                # Skip self-matches
                if corpus_id != query_id:
                    if len(result_heaps[query_id]) < top_k:
                        # Push item on the heap
                        heapq.heappush(result_heaps[query_id], (score, corpus_id))
                    else:
                        # If item is larger than the smallest in the heap, 
                        # push it on the heap then pop the smallest element
                        heapq.heappushpop(result_heaps[query_id], (score, corpus_id))
        
        # Convert heap results to final format
        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score
        
        return self.results