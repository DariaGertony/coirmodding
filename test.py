import coir
from coir.data_loader import get_tasks
from coir.evaluation import COIR
from coir.models import YourCustomDEModel
import ollama


poss_LLM= {"llama3.2", "gemma3", "mistral"}



model_name = "intfloat/e5-base-v2"
chos_LLMmodel = {"mistral"}
useLLm = False

# Load the model
model = YourCustomDEModel(model_name=model_name)

# Get tasks
#all task ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt","codefeedback-st","codetrans-contest","synthetic-
# text2sql","cosqa","codesearchnet","codesearchnet-ccr"]
tasks = get_tasks(tasks=["codetrans-dl"])
##flag for requeueing
# Initialize evaluation
evaluation = COIR(tasks=tasks,batch_size=64, type ="hybrid")

    
if useLLm and chos_LLMmodel:
    for llm in chos_LLMmodel.intersection(poss_LLM):
        results = evaluation.run(model, output_folder=f"results", useLLm=useLLm, llmname=llm)
else:
    # Run evaluation
    results = evaluation.run(model, output_folder="results", useLLm=useLLm, llmname='')
    

print(results)