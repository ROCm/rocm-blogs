import os
import json
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

# Define base directory for eval results
results_dir = "eval_results"

# Define tasks from openllm.yaml
openllm_tasks = [
    "arc_challenge", "hellaswag", "mmlu", "truthfulqa_mc2", "winogrande", "gsm8k"]


model_names = ['Mixtral 8x7B', 'Zephyr-8x7B-SFT', 'Zephyr-8x7B-DPO']
# Initialize scores: task -> model -> score
scores = {task: {} for task in openllm_tasks}

json_files = sorted(glob('old_eval_results/**/*.json', recursive=True))
json_files = [json_files[0], json_files[2], json_files[1]]
print(json_files)
for ii in range(len(json_files)):
    json_file = json_files[ii]
    model_name = model_names[ii]
    with open(json_file, "r") as f:
        data = json.load(f)['results']
        for task in openllm_tasks:
            if task in data:
                score = data[task].get('acc_norm,none', None)

                if not score:
                    score = data[task].get('acc,none', None)
                if not score:
                    score = data[task].get('exact_match,flexible-extract', None)
            #if task == "gsm8k": print(data[task]['exact_match,flexible-extract'], score)
            print(model_name, task, score)
            # Use "acc" or "accuracy" field
            if score:
                scores[task][model_name] = score

# Get sorted list of models
all_models = sorted({model for task_scores in scores.values() for model in task_scores})

# Plotting
x = np.arange(len(openllm_tasks))
width = 0.8 / max(len(all_models), 1)  # dynamic bar width

fig, ax = plt.subplots(figsize=(14, 6))

for i in [0,2,1]:
    model = all_models[i]
    print(model)
    model_scores = [scores[task].get(model, 0.0) for task in openllm_tasks]
    print(model_scores)
    bars = ax.bar(x + i * width, model_scores, width, label=model)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,  # small offset above bar
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )
# Formatting
ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison on OpenLLM Tasks")
ax.set_xticks(x + width * (len(all_models) - 1) / 2)
ax.set_xticklabels(openllm_tasks, rotation=45, ha="right")
ax.legend(title="Model")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('eval_results.png')
