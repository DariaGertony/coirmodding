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
chos_LLMmodel = {"phi3"}
useLLm = False
rerank = False

poss_prompts = {'Give context words for this query', 'Give some suitable or similar code in different programming languages'}


# Load the model
model = YourCustomDEModel(model_name)

# Get tasks
all_task = ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt","codefeedback-st","codetrans-contest","synthetic-text2sql","cosqa","codesearchnet","codesearchnet-ccr"]
req_tasks = ["codetrans-dl","apps","stackoverflow-qa"]

##flag for requeueing
# Initialize evaluation

types = ["bm25_hybrid_combMNZ", "bm25_hybrid_interpolation", "bm25_hybrid_rrf", "bm25_hybrid_weighted", "bm25_lexical", "jaccard_lexical", "default_semantic", "jaccard_hybrid"]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

for t in req_tasks:
    tasks = get_tasks(tasks=["codetrans-dl"])
    for ty in types:
        evaluation = COIR(tasks=tasks,batch_size=64, type =ty)
        res = evaluation.run(model, output_folder="results", useLLm=useLLm, llmname='', prompt='', to_rerank=rerank)

    
if useLLm and chos_LLMmodel:
    for llm in chos_LLMmodel.intersection(poss_LLM):
        for pmp in poss_prompts.intersection({'Give context words for this query'}):
            results = evaluation.run(model, output_folder="results", useLLm=useLLm, llmname=llm, prompt=pmp, to_rerank=rerank)
#else:
    # Run evaluation
    #results = evaluation.run(model, output_folder="results", useLLm=useLLm, llmname='', prompt='', to_rerank=rerank)
    

