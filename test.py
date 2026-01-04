import coir
from coir.data_loader import get_tasks
from coir.evaluation import COIR
from coir.models import YourCustomDEModel
import ollama

model_name = "intfloat/e5-base-v2"

# Load the model
model = YourCustomDEModel(model_name=model_name)

# Get tasks
#all task ["codetrans-dl","stackoverflow-qa","apps","codefeedback-mt","codefeedback-st","codetrans-contest","synthetic-
# text2sql","cosqa","codesearchnet","codesearchnet-ccr"]
tasks = get_tasks(tasks=["codetrans-dl"])

# Initialize evaluation
evaluation = COIR(tasks=tasks,batch_size=128)

# Run evaluation
results = evaluation.run(model, output_folder=f"results/{model_name}")
print(results)