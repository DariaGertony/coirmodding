import os
import json
import logging
import torch
from coir.beir.retrieval.evaluation import EvaluateRetrieval
from coir.beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from coir.beir.retrieval.search.factory import SearchMethodFactory

logger = logging.getLogger(__name__)

# Device detection for CUDA support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class COIR:
    """
    Main CoIR evaluation orchestrator with support for multiple search methods
    and configuration-driven evaluation.
    """
    
    def __init__(self, tasks, batch_size: int, search_config: dict = None):
        """
        Initialize COIR evaluation orchestrator.
        
        Args:
            tasks: Dictionary of tasks to evaluate
            batch_size: Batch size for processing
            search_config: Configuration for search method (defaults to dense search)
        """
        self.tasks = tasks
        self.batch_size = batch_size
        self.search_config = search_config or {"method": "dense"}
        
        # Validate search configuration
        self._validate_search_config()

    def _validate_search_config(self):
        """Validate search configuration parameters"""
        method = self.search_config.get("method", "dense")
        if method not in SearchMethodFactory.list_available_methods():
            raise ValueError(f"Unknown search method: {method}. "
                           f"Available: {SearchMethodFactory.list_available_methods()}")

    def run(self, model, output_folder: str, 
            use_llm: bool = False, 
            llm_name: str = "", 
            prompt: str = "",
            enable_reranking: bool = False,
            **kwargs):
        """
        Run evaluation with configurable search method and optional enhancements
        
        Args:
            model: The embedding model to evaluate
            output_folder: Directory to save results
            use_llm: Whether to use LLM for query expansion
            llm_name: Name of LLM model (e.g., 'llama2')
            prompt: Prompt template for LLM query expansion
            enable_reranking: Whether to apply reranking
            **kwargs: Additional parameters for search methods
            
        Returns:
            dict: Results for each task
        """
        results = {}
        
        for task_name, task_data in self.tasks.items():
            output_file = os.path.join(output_folder, f"{task_name}.json")

            corpus, queries, qrels = task_data

            # Create search method based on configuration
            search_params = {
                'model': model,
                'batch_size': self.batch_size,
                **self.search_config.get('params', {}),
                **kwargs
            }

            # Handle backward compatibility for dense search
            if self.search_config['method'] in ['dense', 'semantic']:
                custom_model = SearchMethodFactory.create_search_method(
                    self.search_config['method'], 
                    **search_params
                )
            else:
                # For non-dense methods, model parameter might not be needed
                if 'model' in search_params and self.search_config['method'] in ['jaccard', 'bm25']:
                    search_params.pop('model')
                custom_model = SearchMethodFactory.create_search_method(
                    self.search_config['method'], 
                    **search_params
                )

            retriever = EvaluateRetrieval(custom_model, score_function="cos_sim")

            # Retrieve results with optional LLM enhancement
            if use_llm and hasattr(retriever, 'retrieve'):
                # Check if retrieve method supports LLM parameters
                try:
                    task_results = retriever.retrieve(corpus, queries, use_llm, llm_name, prompt)
                except TypeError:
                    # Fallback for methods that don't support LLM parameters
                    logger.warning(f"Search method {self.search_config['method']} doesn't support LLM enhancement")
                    task_results = retriever.retrieve(corpus, queries)
            else:
                task_results = retriever.retrieve(corpus, queries)

            # Evaluate results
            ndcg, map_score, recall, precision = retriever.evaluate(qrels, task_results, retriever.k_values)
            metrics = {
                "NDCG": ndcg,
                "MAP": map_score,
                "Recall": recall,
                "Precision": precision
            }

            # Save results with enhanced organization
            self._save_results(
                task_name, output_folder, metrics, model, 
                use_llm, llm_name, prompt, enable_reranking
            )

            logger.info(f"Results for {task_name} saved to {output_folder}")
            results[task_name] = metrics

        return results

    def _save_results(self, task_name: str, output_folder: str, metrics: dict,
                      model, use_llm: bool, llm_name: str, prompt: str, 
                      enable_reranking: bool):
        """Save results with enhanced organization"""
        
        output_file = os.path.join(output_folder, f"{task_name}.json")
        
        # Create descriptive configuration label
        config_label = self._create_config_label(use_llm, llm_name, prompt, enable_reranking)
        
        # Determine method identifier
        method_id = self._get_method_identifier(model)
        
        # Load existing results or create new structure
        os.makedirs(output_folder, exist_ok=True)
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        
        # Organize results by configuration and method
        if config_label not in data:
            data[config_label] = {}
        
        data[config_label][method_id] = {"metrics": metrics}
        
        # Save updated results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def _create_config_label(self, use_llm: bool, llm_name: str, prompt: str, 
                            enable_reranking: bool) -> str:
        """Create descriptive label for configuration"""
        if not use_llm:
            label = 'baseline'
        else:
            label = llm_name if llm_name else 'llm'
            if prompt:
                label += f"\n{prompt}"
        
        if enable_reranking:
            label += " + reranking"
        
        return label

    def _get_method_identifier(self, model) -> str:
        """Get identifier for the search method and model"""
        method = self.search_config['method']
        
        if method in ['dense', 'semantic']:
            return getattr(model, 'model_name', 'dense_model')
        elif method in ['jaccard', 'bm25']:
            return method
        elif 'hybrid' in method:
            model_name = getattr(model, 'model_name', 'unknown_model')
            return f"{method}({model_name})"
        else:
            return method

    def get_search_config(self) -> dict:
        """Get current search configuration"""
        return self.search_config.copy()

    def set_search_config(self, config: dict):
        """Update search configuration"""
        self.search_config = config
        self._validate_search_config()

    @staticmethod
    def create_search_config(method: str, **params) -> dict:
        """Helper to create search configuration"""
        return {
            "method": method,
            "params": params
        }

    @classmethod
    def list_available_search_methods(cls) -> list:
        """List all available search methods"""
        return SearchMethodFactory.list_available_methods()