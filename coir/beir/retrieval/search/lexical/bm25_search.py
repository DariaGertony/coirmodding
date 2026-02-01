from typing import Dict
import bm25s
from ..base import BaseSearch


class LexicalBM25Search(BaseSearch):
    """
    Lexical search implementation using BM25 algorithm.
    
    Uses the bm25s library for efficient BM25 implementation with
    English stopwords removal and tokenization.
    """
    
    def __init__(self, batch_size: int = 128, **kwargs):
        """
        Initialize the BM25 search.
        
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
        Perform BM25-based search.
        
        Args:
            corpus: Dictionary mapping corpus IDs to documents with 'title' and 'text' fields
            queries: Dictionary mapping query IDs to query strings
            top_k: Number of top results to return per query
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary mapping query IDs to dictionaries of document IDs and scores
        """
        query_ids = list(queries.keys())
        
        # Initialize results dictionary
        results = {qid: {} for qid in query_ids}
        
        # Handle empty corpus case
        if not corpus:
            return results
        
        # Sort corpus by document length (longest first) for consistency with other implementations
        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        
        # Extract corpus texts
        corpus_texts = [corpus[cid].get("text", "") for cid in corpus_ids]
        
        # Handle case where all corpus texts are empty
        if not any(text.strip() for text in corpus_texts):
            return results
        
        # Tokenize corpus with English stopwords removal
        tokenized_corpus = bm25s.tokenize(corpus_texts, stopwords="en")
        
        # Create and index BM25 retriever
        retriever = bm25s.BM25()
        retriever.index(tokenized_corpus)
        
        # Process each query
        for qid, query_text in queries.items():
            # Skip empty queries
            if not query_text.strip():
                continue
                
            # Tokenize query with English stopwords removal
            tokenized_query = bm25s.tokenize(query_text, stopwords="en")
            
            # Adjust top_k if corpus is smaller
            effective_top_k = min(top_k, len(corpus_texts))
            
            # Skip if no effective top_k
            if effective_top_k <= 0:
                continue
            
            # Retrieve top-k documents
            doc_indexes, scores = retriever.retrieve(tokenized_query, k=effective_top_k)
            
            # Convert results to the expected format
            for idx, score in zip(doc_indexes[0], scores[0]):
                corpus_id = corpus_ids[idx]
                # Skip self-matches (when query ID matches document ID)
                if corpus_id != qid:
                    results[qid][corpus_id] = float(score)
        
        return results
