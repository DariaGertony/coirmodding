"""
AsyncOllama class for asynchronous LLM interactions using Ollama.

This module provides a clean interface for making parallel LLM calls
with proper thread management and callback support.
"""

import ollama
import threading
from typing import Optional, Callable, List


class AsyncOllama:
    """
    Asynchronous Ollama client for parallel LLM query processing.
    
    This class manages multiple concurrent LLM requests using threading,
    allowing for efficient batch processing of queries with callback support.
    """
    
    def __init__(self, llm_name: str, prompt: str):
        """
        Initialize the AsyncOllama client.
        
        Args:
            llm_name (str): Name of the Ollama model to use
            prompt (str): Base prompt template for queries
        """
        self.result = None
        self.threads: List[threading.Thread] = []
        self.llm = llm_name
        self.prompt = prompt
    
    def ask_async(self, i: int, query: str, callback: Optional[Callable[[int, str, str], None]] = None):
        """
        Submit an asynchronous query to the LLM.
        
        Args:
            i (int): Query index for tracking
            query (str): The query text to process
            callback (Optional[Callable]): Callback function called with (index, query, result)
        """
        def worker():
            try:
                response = ollama.chat(
                    model=self.llm,
                    messages=[{'role': 'user', 'content': self.prompt + '\n' + query}],
                    options={
                        'num_predict': 64,      
                        'temperature': 0.5,      
                        'top_k': 20,           
                        'top_p': 0.9,
                        'repeat_penalty': 1.2,
                        'num_thread': 30, 
                        'num_gpu': 80,  
                    }
                )
                self.result = response['message']['content']
                if callback:
                    callback(i, query, self.result)
            except Exception as e:
                # Handle LLM errors gracefully
                error_result = f"Error processing query: {str(e)}"
                self.result = error_result
                if callback:
                    callback(i, query, error_result)
        
        thread = threading.Thread(target=worker)
        thread.start()
        self.threads.append(thread)
    
    def wait(self):
        """
        Wait for all submitted queries to complete.
        
        This method blocks until all threads have finished processing.
        """
        for thread in self.threads:
            thread.join()
        
        # Clear threads list after completion
        self.threads.clear()
    
    def is_busy(self) -> bool:
        """
        Check if any queries are still being processed.
        
        Returns:
            bool: True if any threads are still active, False otherwise
        """
        return any(thread.is_alive() for thread in self.threads)
    
    def get_active_count(self) -> int:
        """
        Get the number of currently active threads.
        
        Returns:
            int: Number of active threads
        """
        return sum(1 for thread in self.threads if thread.is_alive())