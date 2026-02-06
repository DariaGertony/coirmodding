import coir
from coir.data_loader import get_tasks
from coir.evaluation import COIR
from coir.models import YourCustomDEModel
import ollama




poss_LLM= {"llama3.2", "gemma3", "phi3"}

# intfloat/e5-base-v2   BAAI/bge-m3 intfloat/multilingual-e5-base

model_name = "intfloat/multilingual-e5-base"
chos_LLMmodel = {"phi3"}
useLLm = False
rerank = False

poss_prompts = {'Give context words for this query', 'Give some suitable or similar code in different programming languages'}


# Load the model
model = YourCustomDEModel(model_name=model_name)

# Get tasks
#all task ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt","codefeedback-st","codetrans-contest","synthetic-
# text2sql","cosqa","codesearchnet","codesearchnet-ccr"]
tasks = get_tasks(tasks=["stackoverflow-qa"])
##flag for requeueing
# Initialize evaluation
evaluation = COIR(tasks=tasks,batch_size=72, type ="semantic")

    
if useLLm and chos_LLMmodel:
    for llm in chos_LLMmodel.intersection(poss_LLM):
        for pmp in poss_prompts.intersection({'Give context words for this query'}):
            results = evaluation.run(model, output_folder="res", useLLm=useLLm, llmname=llm, prompt=pmp, to_rerank=rerank)
else:
    # Run evaluation
    results = evaluation.run(model, output_folder="res", useLLm=useLLm, llmname='', prompt='', to_rerank=rerank)
    

print(results)