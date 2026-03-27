- stackoverflow-qa_default_semantic - семантический поиск (дефолтный пайплайн COIR'а)

- stackoverflow-qa_jaccard_lexical - лексический поиск основанный на коэфф-а Жаккарда - `self.lexical_search(corpus, dict_queries, top_k)`

- stackoverflow-qa_bm25_lexical - лексический поиск основанный на алгоритме bm25 () - `self.lexical_search_bm25(corpus, dict_queries, top_k)`

- stackoverflow-qa_jaccard_hybrid_without-rerank - гибридный поиск (внутри лексический поиск на основе коэфф-а Жаккрада; метод слияния - max_heap) - `self.hibrid_search(corpus, dict_queries, top_k, score_function,return_sorted)`

------------

все дальнейшие поиски запускаются методом:
```
return self.hibrid_search_with_bm25(corpus, dict_queries, top_k, score_function, self.to_rerak, return_sorted)
```

метод слияния задаётся при инициализации DRES; параметр **htype**

```
 class COIR:
    def __init__(self, tasks, batch_size, type="semantic"):
        self.tasks = tasks
        self.batch_size = batch_size
        self.type = type
        #######################################
        

    def run(self, model, output_folder: str, useLLm: bool, llmname: str, prompt: str,to_rerank: bool):
        results = {}
        for task_name, task_data in self.tasks.items():
            output_file = os.path.join(output_folder, f"{task_name}.json")


            corpus, queries, qrels = task_data

            # Initialize custom model
            custom_model = DRES(model, batch_size=self.batch_size, type=self.type, htype="rrf")
            retriever = EvaluateRetrieval(custom_model, score_function="cos_sim")

```

```
class DenseRetrievalExactSearch(BaseSearch):
    
    def __init__(self, model, batch_size: int = 128, type = "semantic", htype='combMNZ', corpus_chunk_size: int = 50000, to_rerank = False ,rermodel="BAAI/bge-reranker-base", **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}
        self.type = type
        self.hyb_type = htype
        self.rermodel = rermodel
        self.to_rerak = to_rerank
```    

- stackoverflow-qa_bm25_hybrid_without-rerank_combMNZ-fusion - гибридный поиск (внутри лексический поиск на основе bm25; метод слияния - combMNZ)

- stackoverflow-qa_bm25_hybrid_without-rerank_interpolation-fusion - гибридный поиск (внутри лексический поиск на основе bm25; метод слияния - interpolation)

- stackoverflow-qa_bm25_hybrid_without-rerank_rrf-fusion - гибридный поиск (внутри лексический поиск на основе bm25; метод слияния - rrf)

- stackoverflow-qa_bm25_hybrid_without-rerank_weighted-fusion - - гибридный поиск (внутри лексический поиск на основе bm25; метод слияния - weighted)