"""
Search Method Factory for creating search method instances.

This module provides a factory pattern for creating different types of search methods,
enabling configuration-driven search method selection and easy extensibility.
"""

from typing import Dict, Any
from .base import BaseSearch
from .dense.exact_search import DenseRetrievalExactSearch
from .lexical.jaccard_search import LexicalJaccardSearch
from .lexical.bm25_search import LexicalBM25Search
from .hybrid.simple_hybrid_search import SimpleHybridSearch
from .hybrid.advanced_hybrid_search import AdvancedHybridSearch


class SearchMethodFactory:
    """
    Factory for creating search method instances.
    
    This factory provides a centralized way to create different types of search methods,
    supporting both direct instantiation and configuration-driven creation.
    """
    
    SEARCH_METHODS = {
        'dense': DenseRetrievalExactSearch,
        'semantic': DenseRetrievalExactSearch,  # alias for dense
        'jaccard': LexicalJaccardSearch,
        'bm25': LexicalBM25Search,
        'simple_hybrid': SimpleHybridSearch,
        'advanced_hybrid': AdvancedHybridSearch,
    }
    
    @classmethod
    def create_search_method(cls, method_type: str, **kwargs) -> BaseSearch:
        """
        Create a search method instance.
        
        Args:
            method_type (str): Type of search method to create
            **kwargs: Arguments to pass to the search method constructor
            
        Returns:
            BaseSearch: Instance of the requested search method
            
        Raises:
            ValueError: If the method_type is not supported
            
        Examples:
            >>> factory = SearchMethodFactory()
            >>> dense_search = factory.create_search_method('dense', model=model)
            >>> jaccard_search = factory.create_search_method('jaccard', batch_size=64)
        """
        if method_type not in cls.SEARCH_METHODS:
            raise ValueError(f"Unknown search method: {method_type}. "
                           f"Available: {list(cls.SEARCH_METHODS.keys())}")
        
        search_class = cls.SEARCH_METHODS[method_type]
        return search_class(**kwargs)
    
    @classmethod
    def list_available_methods(cls) -> list:
        """
        List all available search methods.
        
        Returns:
            list: List of available search method names
        """
        return list(cls.SEARCH_METHODS.keys())
    
    @classmethod
    def register_search_method(cls, name: str, search_class: type):
        """
        Register a new search method.
        
        Args:
            name (str): Name to register the search method under
            search_class (type): Search class that inherits from BaseSearch
            
        Raises:
            ValueError: If the search_class doesn't inherit from BaseSearch
        """
        if not issubclass(search_class, BaseSearch):
            raise ValueError(f"Search class {search_class} must inherit from BaseSearch")
        
        cls.SEARCH_METHODS[name] = search_class
    
    @classmethod
    def get_search_method_class(cls, method_type: str) -> type:
        """
        Get the class for a search method without instantiating it.
        
        Args:
            method_type (str): Type of search method
            
        Returns:
            type: The search method class
            
        Raises:
            ValueError: If the method_type is not supported
        """
        if method_type not in cls.SEARCH_METHODS:
            raise ValueError(f"Unknown search method: {method_type}. "
                           f"Available: {list(cls.SEARCH_METHODS.keys())}")
        
        return cls.SEARCH_METHODS[method_type]