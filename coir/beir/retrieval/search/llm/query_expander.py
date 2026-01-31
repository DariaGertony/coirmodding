"""
QueryExpander class for LLM-based query expansion.

This module provides a clean interface for expanding search queries using LLMs,
with progress tracking and batch processing capabilities.
"""

from typing import List, Dict
from tqdm import tqdm
from .async_ollama import AsyncOllama


class QueryExpander:
    """
    Query expansion orchestrator using LLM for enhanced search queries.
    
    This class manages the process of expanding queries using an LLM,
    providing progress tracking and result aggregation.
    """
    
    def __init__(self, llm_name: str, prompt: str):
        """
        Initialize the QueryExpander.
        
        Args:
            llm_name (str): Name of the LLM model to use for expansion
            prompt (str): Prompt template for query expansion
        """
        self.llm_name = llm_name
        self.prompt = prompt
    
    def expand_queries(self, queries: List[str]) -> List[str]:
        """
        Expand a list of queries using LLM and return enhanced queries.
        
        Args:
            queries (List[str]): List of original queries to expand
            
        Returns:
            List[str]: List of expanded queries in the same order
        """
        if not queries:
            return []
        
        # Initialize progress tracking
        with tqdm(total=len(queries), desc="Expanding queries", unit="query") as pbar:
            expanded_queries = {}
            completed_count = []
            
            # Initialize expanded queries dictionary
            for i, query in enumerate(queries):
                expanded_queries[i] = query
            
            # Create AsyncOllama client
            ollama_client = AsyncOllama(self.llm_name, self.prompt)
            
            def callback(i: int, original_query: str, llm_result: str):
                """Callback function to handle LLM results."""
                # Combine original query with LLM expansion
                expanded_queries[i] = original_query + "\n" + llm_result
                completed_count.append(True)
                pbar.update(1)
            
            # Submit all queries for async processing
            for i, query in enumerate(queries):
                ollama_client.ask_async(i, query, callback)
            
            # Wait for all queries to complete
            ollama_client.wait()
        
        # Return expanded queries in original order
        return [expanded_queries[i] for i in range(len(queries))]
    
    def expand_query_dict(self, query_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Expand queries from a dictionary format.
        
        Args:
            query_dict (Dict[str, str]): Dictionary mapping query IDs to query text
            
        Returns:
            Dict[str, str]: Dictionary with same IDs but expanded query text
        """
        if not query_dict:
            return {}
        
        # Extract queries and IDs
        query_ids = list(query_dict.keys())
        queries = [query_dict[qid] for qid in query_ids]
        
        # Expand queries
        expanded_queries = self.expand_queries(queries)
        
        # Reconstruct dictionary with expanded queries
        return dict(zip(query_ids, expanded_queries))
    
    def set_prompt(self, new_prompt: str):
        """
        Update the prompt template for query expansion.
        
        Args:
            new_prompt (str): New prompt template to use
        """
        self.prompt = new_prompt
    
    def set_llm_model(self, new_llm_name: str):
        """
        Update the LLM model to use for expansion.
        
        Args:
            new_llm_name (str): New LLM model name
        """
        self.llm_name = new_llm_name