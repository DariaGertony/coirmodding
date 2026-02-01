"""
Examples of different search configurations for CoIR evaluation

This module provides examples of how to configure different search methods
for the CoIR evaluation framework, demonstrating the flexibility of the
configuration-driven approach.
"""

# Dense/Semantic search (original)
DENSE_CONFIG = {
    "method": "dense"
}

# Alternative semantic search configuration (alias for dense)
SEMANTIC_CONFIG = {
    "method": "semantic"
}

# Lexical search configurations
JACCARD_CONFIG = {
    "method": "jaccard"
}

BM25_CONFIG = {
    "method": "bm25"
}

# Hybrid search configurations
SIMPLE_HYBRID_CONFIG = {
    "method": "simple_hybrid"
}

ADVANCED_HYBRID_CONFIG = {
    "method": "advanced_hybrid",
    "params": {
        "fusion_method": "rrf",
        "rerank_model": "BAAI/bge-reranker-base"
    }
}

# Custom configurations with parameters
DENSE_WITH_PARAMS_CONFIG = {
    "method": "dense",
    "params": {
        "score_function": "cos_sim",
        "normalize_embeddings": True
    }
}

BM25_WITH_PARAMS_CONFIG = {
    "method": "bm25",
    "params": {
        "k1": 1.2,
        "b": 0.75
    }
}

# Usage examples
def example_usage():
    """
    Example usage of different search configurations with CoIR evaluation
    """
    from coir.evaluation import COIR
    from coir.data_loader import get_tasks
    
    # Load tasks
    tasks = get_tasks(tasks=["codetrans-dl"])
    
    # Dense search evaluation (default)
    dense_eval = COIR(tasks, batch_size=128, search_config=DENSE_CONFIG)
    
    # Lexical search evaluation
    jaccard_eval = COIR(tasks, batch_size=128, search_config=JACCARD_CONFIG)
    bm25_eval = COIR(tasks, batch_size=128, search_config=BM25_CONFIG)
    
    # Hybrid search evaluation
    simple_hybrid_eval = COIR(tasks, batch_size=128, search_config=SIMPLE_HYBRID_CONFIG)
    advanced_hybrid_eval = COIR(tasks, batch_size=128, search_config=ADVANCED_HYBRID_CONFIG)
    
    # Example model (placeholder - replace with actual model)
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Run evaluations (commented out to avoid execution)
    # dense_results = dense_eval.run(model, "results/dense")
    # jaccard_results = jaccard_eval.run(model, "results/jaccard")
    # bm25_results = bm25_eval.run(model, "results/bm25")
    # hybrid_results = advanced_hybrid_eval.run(model, "results/hybrid", 
    #                                          use_llm=True, llm_name="llama2")

def example_with_llm_and_reranking():
    """
    Example usage with LLM enhancement and reranking
    """
    from coir.evaluation import COIR
    from coir.data_loader import get_tasks
    
    # Load tasks
    tasks = get_tasks(tasks=["codetrans-dl"])
    
    # Create evaluator with advanced hybrid search
    evaluator = COIR(tasks, batch_size=128, search_config=ADVANCED_HYBRID_CONFIG)
    
    # Example model (placeholder)
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Run with LLM enhancement and reranking
    # results = evaluator.run(
    #     model, 
    #     "results/enhanced",
    #     use_llm=True,
    #     llm_name="llama2",
    #     prompt="Expand this query for better code search: {query}",
    #     enable_reranking=True
    # )

def example_configuration_management():
    """
    Example of dynamic configuration management
    """
    from coir.evaluation import COIR
    from coir.data_loader import get_tasks
    
    # Load tasks
    tasks = get_tasks(tasks=["codetrans-dl"])
    
    # Create evaluator with default configuration
    evaluator = COIR(tasks, batch_size=128)
    
    # Check available search methods
    available_methods = COIR.list_available_search_methods()
    print(f"Available search methods: {available_methods}")
    
    # Get current configuration
    current_config = evaluator.get_search_config()
    print(f"Current configuration: {current_config}")
    
    # Update configuration dynamically
    new_config = COIR.create_search_config("bm25", k1=1.5, b=0.8)
    evaluator.set_search_config(new_config)
    
    # Verify updated configuration
    updated_config = evaluator.get_search_config()
    print(f"Updated configuration: {updated_config}")

def example_backward_compatibility():
    """
    Example showing backward compatibility with original interface
    """
    from coir.evaluation import COIR
    from coir.data_loader import get_tasks
    
    # Load tasks
    tasks = get_tasks(tasks=["codetrans-dl"])
    
    # Original interface (still works)
    evaluator = COIR(tasks, batch_size=128)
    
    # Example model (placeholder)
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Original run method call (still works)
    # results = evaluator.run(model, "results/backward_compatible")

if __name__ == "__main__":
    # Run examples (commented out to avoid execution without proper setup)
    print("CoIR Search Configuration Examples")
    print("=" * 40)
    
    print("\n1. Basic Usage Example")
    example_usage()
    
    print("\n2. LLM and Reranking Example")
    example_with_llm_and_reranking()
    
    print("\n3. Configuration Management Example")
    example_configuration_management()
    
    print("\n4. Backward Compatibility Example")
    example_backward_compatibility()
    
    print("\nAll examples completed successfully!")