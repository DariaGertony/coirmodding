import coir
from coir.data_loader import get_tasks
from coir.evaluation import COIR
from coir.models import YourCustomDEModel
from sentence_transformers import SentenceTransformer
import ollama
import torch



poss_LLM= {"llama3.2", "gemma3", "phi3"}

# intfloat/e5-base   BAAI/bge-m3 intfloat/e5-base-v2

model_name = "BAAI/bge-m3"


useLLm = True

poss_prompts = {'Give context words for this query'}


# Load the model
model = YourCustomDEModel(model_name)

# Get tasks
all_task = ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt","codefeedback-st","codetrans-contest","synthetic-text2sql","cosqa","codesearchnet","codesearchnet-ccr"]
req_tasks = ["apps","stackoverflow-qa"]
types = ["bm25_hybrid_combMNZ", "bm25_hybrid_interpolation", "bm25_hybrid_rrf", "bm25_hybrid_weighted", "bm25_lexical", "jaccard_lexical", "default_semantic", "jaccard_hybrid"]



    
if useLLm :
    tasks = get_tasks(tasks=["codetrans-dl"])
    evaluation = COIR(tasks=tasks,batch_size=64, type ="default_semantic")
    evaluation.llm_init("nvidia/nemotron-3-nano-30b-a3b:free", "Give context words for this query'")
    results = evaluation.run(model, output_folder="results", useLLm=useLLm, to_rerank=False)
else:
    for t in req_tasks:
        tasks = get_tasks(tasks=t)
        for ty in types:
            evaluation = COIR(tasks=tasks,batch_size=64, type =ty)
            res = evaluation.run(model, output_folder="results", useLLm=useLLm, to_rerank=False)
        

