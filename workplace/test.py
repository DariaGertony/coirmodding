import coir
from coir.data_loader import get_tasks
from coir.evaluation import COIR
from coir.models import YourCustomDEModel
from sentence_transformers import SentenceTransformer
import ollama
import torch
import dotenv
import os


poss_LLM= ["meta-llama/llama-4-scout-17b-16e-instruct"]
#["qwen/qwen3-32b"]
# ["llama3", "gemma3", "phi3"] for ollama
# intfloat/e5-base   BAAI/bge-m3 intfloat/e5-base-v2

model_name = "BAAI/bge-m3"


useLLm = True

dotenv.load_dotenv()
kee = os.getenv("GK")

poss_prompts = {'Give comprehensive description for this query','Give additional information for this query to improve understanding'}


# Load the model
model = YourCustomDEModel(model_name)

# Get tasks
all_task = ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt","codefeedback-st","codetrans-contest","synthetic-text2sql","cosqa","codesearchnet","codesearchnet-ccr"]
req_tasks = ["apps","stackoverflow-qa"]
types = ["bm25_hybrid_combMNZ", "bm25_hybrid_interpolation", "bm25_hybrid_rrf", "bm25_hybrid_weighted", "bm25_lexical", "jaccard_lexical", "default_semantic", "jaccard_hybrid"]

tasks = get_tasks(tasks=["codetrans-dl"])

evaluation = COIR(tasks=tasks,batch_size=64, type ="default_semantic")
evaluation.llm_init("meta-llama/llama-4-scout-17b-16e-instruct", 'Give description of this query to improve understanding', kee)
results = evaluation.run(model, output_folder="testing_results_for_groq", useLLm=False, expanded=True)


"""
    
if useLLm :
    tasks = get_tasks(tasks=["codetrans-dl"])
    for ty in poss_LLM:
        print(ty)
        evaluation = COIR(tasks=tasks,batch_size=64, type ="default_semantic")
        evaluation.llm_init(ty, 'Give description of this query to improve understanding', kee)
        results = evaluation.run(model, output_folder="testing_results_for_groq", useLLm=useLLm, expanded=True)
else:
    for t in req_tasks:
        tasks = get_tasks(tasks=t)
        for ty in types:
            evaluation = COIR(tasks=tasks,batch_size=64, type =ty)
            res = evaluation.run(model, output_folder="results", useLLm=useLLm, to_rerank=False)
        
"""
