---
blogpost: true
blog_title: "Chain-of-Thought Guided Visual Reasoning Using Llama 3.2 on a Single AMD Instinct MI300X GPU"
date: 21 Jul 2025
author: 'Matthias Reso'
thumbnail: 't.png'
tags: Fine-Tuning
category: Software tools & optimizations
target_audience: AI Enthusiast and Developers
key_value_propositions: Fine-tune Llama 3.2 Vision Instruct models on a single AMD MI300X GPU using Torchtune, achieving 2.3× better accuracy with 11B vs 90B models.
language: English
myst:
    html_meta:
        "author": "Matthias Reso"
        "description lang=en": "Fine-tune Llama 3.2 Vision models on AMD MI300X GPU using Torchtune, achieving 2.3× better accuracy with 11B vs 90B model on chart-based tasks."
        "keywords": "Fine-tune Llama 3.2"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Matthias Reso"
---

# Chain-of-Thought Guided Visual Reasoning Using Llama 3.2 on a Single AMD Instinct MI300X GPU
  
In this post, we will show you how to fine-tune the Llama 3.2 Vision Instruct models, specifically the 11B and 90B parameter variants, on a synthetic multi-modal dataset using torchtune. This blog focuses on chain-of-thought (CoT) guided visual reasoning, a technique where the model is encouraged to articulate intermediate reasoning steps before arriving at a final answer. By incorporating the CoT approach, we aim to improve the model’s interpretability and accuracy in tasks that require multi-step understanding of visual inputs. By utilizing the high-bandwidth memory (HBM) of the AMD Instinct™ MI300X GPU, we aim to enhance the model's vision understanding, particularly for interpreting charts, all on a single GPU provided by TensorWave. Our evaluation shows that we can train an 11B parameter model to perform with 2.3x better accuracy than a 90B parameter model. The blog will walk you through our dataset preparation, model configuration, training recipes, and evaluation—all optimized to run on a single GPU.

## Key Takeaway

- Our fine-tuned 11B parameter vision language model, trained on a synthetic reasoning dataset, outperforms a 90B parameter model by 2.3x.
- Through torchtune, we can fine-tune both the 11B and 90B models on a single AMD Instinct MI300X GPU, thanks to LoRA and QLoRA.
- A simple configuration file can be used to create a new synthetic dataset, thanks to the integration of the visual Q&A task into the synthetic dataset toolkit.

## What is torchtune

torchtune is a powerful PyTorch library designed to streamline the process of authoring, post-training, and experimenting with LLMs. It offers a variety of hackable training recipes and simple implementations of popular LLMs like Llama, Gemma, Mistral, Phi, and Qwen, among others.
The library offers best-in-class memory efficiency and performance improvements. This is particularly advantageous when fine-tuning models on advanced hardware like AMD Instinct MI300X GPUs, which boast a lot of high-bandwidth memory (HBM) on a single GPU.

By leveraging torchtune's capabilities, users can efficiently fine-tune models with various post-training methods, such as Supervised Fine-tuning (SFT), Quantization-Aware Training (QAT), knowledge distillation, and reinforcement learning techniques like direct preference optimization (DPO), proximal policy optimization (PPO), and group relative policy optimization (GRPO), all while benefiting from the latest PyTorch APIs and YAML-configured training recipes. The configurable recipes make it easy to set up and customize training, evaluation, quantization, or inference processes. This flexibility, combined with its best-in-class memory efficiency and performance improvements, allows users to fully leverage the capabilities of advanced hardware like AMD Instinct MI300X GPUs.

## Prepare the training environment

To prepare the training environment, we will be using a virtual environment for setup. Optionally, we can install the Jupyter notebook server and then continue the setup from within a notebook.

In a terminal, create a new virtual environment and install the dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate venv
source venv/bin/activate

# Install torchtune, torchvision, torchao nightlies
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.3
pip install -U git+https://github.com/pytorch/torchtune@20bdf10
pip install git+https://github.com/pytorch/ao@v0.11.0 \
    --no-build-isolation

#Install lm-evaluation-harness 
pip install git+https://github.com/EleutherAI/lm-evaluation-
harness.git@2cfdd0a294214fe156caa1b8a69da17a29c39e63
```

## Preparing a Synthetic Visual Reasoning Dataset

Modern LLMs have been shown to perform more accurately when they are asked and trained to reason about a question. In previous works, it has been demonstrated that this kind of enhancement can significantly improve the tool calling capabilities of a Llama 3.1 model.

Here, we are going to create a synthetic multi-modal dataset based on the ChartQA dataset, which features a diverse array of image and question/answer pairs. Our goal is to enhance the original ground truth training data by adding a chain of thought (CoT), which is designed to teach the model to reason more effectively about the task at hand, thereby increasing the accuracy of its answers.

To automate the process, we can leverage the synthetic-data-kit. The data is generated by providing a special system prompt to a Llama 3.2 90B Vision Instruct model, instructing it to append a chain of thought to a given sample. To host the 90B parameter model, we are using the rocm/vllm Docker container provided by AMD:

```bash
docker pull rocm/vllm:latest
docker run \

  --device /dev/dri \
  --device /dev/kfd \
  --network host \
  --ipc host \
  --group-add video \
  --security-opt seccomp=unconfined \
  --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
  --shm-size 64G rocm/vllm:latest \
  vllm serve meta-llama/Llama-3.2-90B-Vision-Instruct \
  --max-model-len 8129 --max-num-seqs 16
```

**Note:** Be sure to set your Hugging Face Hub token and apply for access to the 11B and 90B models used for this blog.

Then in another terminal on the same host we install the synthetic-data-kit, create the config file and start generating the dataset:

```bash
pip install git+https://github.com/meta-llama/synthetic-data-kit

cat << EOF >config.yaml
# config.yaml
llm:
  provider: api-endpoint
api-endpoint:
  api_base: "http://localhost:8000/v1"
  api_key: "NO_KEY_NEEDED"
  model: "meta-llama/Llama-3.2-11B-Vision-Instruct"
  max_retries: 3
  retry_delay: 1.0

generation:
  temperature: 0.2   # Lower temperature for more consistent reasoning
  top_p: 0.95
  max_tokens: 4092   # Allow for longer outputs to accommodate CoT reasoning

input_split: "train"       # Split to use for input
output_split: "train"       # Split to use for output
# The most important part - our custom Chain of Thought prompt
prompts:
  xqa_add_reasoning: |
    You are an AI with an IQ of 170 tasked with enhancing dataset examples by adding a Chain of Thought (CoT) before the final answer.
    Given an image and a query-answer pair in the format "{query}. Final answer: {answer}",
    analyze the image and think out loud to create a detailed CoT. Return only the CoT and "Final answer: {answer}" without the query.
    Maximize tokens in the CoT.
    For example,
    transform:
    "Is the value of Favorable 38 in 2015? Final answer: Yes"
    into:
    "There are two graphs in the diagram. A green showing the Favorable trend, and an orange showing the Unfavorable trend. In 2015 the value of the green graph is 38. Final answer: Yes".

    REMEMBER: Follow the format "{CoT} Final answer: {answer}" for your answer AND DO NOT CHANGE THE ORIGINAL FINAL ANSWER. Begin now.
EOF

synthetic-data-kit -c config.yaml \
  create "HuggingFaceM4/ChartQA" \
  --type xqa_add_reasoning \
  -o cot_chartqa/
```

The dataset will be saved into the cot_chartqa/ folder where we can load it back with the load_dataset method from Hugging Face datasets package. In the next two sections, we will use the created dataset to fine-tune the 11B and 90B variants of the Llama 3.2 Vision Instruct model.

## LoRA Fine-tuning Llama 3.2 11B Vision Instruct with torchtune

To maximize memory and compute efficiency for our training, we are utilizing the Low-Rank Adaptation (LoRA) method. LoRA reduces the number of trainable parameters by decomposing them into lower-rank matrices. torchtune provides a training recipe for LoRA on a single device named lora_finetune_single_device. To customize the default configuration of the recipes we first create a local copy:

```bash
tune cp llama3_2_vision/11B_lora_single_device \

    llama3_2_vision_11B_lora_single_device.yaml
```

Subsequently, we make the following changes to
llama3_2_vision_11B_lora_single_device.yaml:

```bash
# Model parameter
model:
  _component_: torchtune.models.llama3_2_vision.lora_llama3_2_vision_11b
  # ...
  lora_rank: 16  # higher increases accuracy and memory
  lora_alpha: 32  # usually alpha=2*rank
# ...
# Dataset parameter
dataset:
  _component_: torchtune.datasets.multimodal.vqa_dataset
  source: "cot_chartqa/"
  packed: False
  # This maps torchtune expected column names to the one we saw above when exploring the dataset
  column_map: {"input":"query", "image":"image", "output":"label"}
# ...
epochs: 10
batch_size: 16 # Maximize memory usage on AMD Instinct MI300X GPU
gradient_accumulation_steps: 1
# ...
# Optionally, don't forget to log into wandb using `wandb login` first
metric_logger:
  _component_: torchtune.training.metric_logging.WandBLogger
  project: vision_llama_reasoning
```

Next, we can go ahead, download the model weights and start the fine-tuning:

```bash
# Download model weights
tune download meta-llama/Llama-3.2-11B-Vision-Instruct \
--output-dir /tmp/Llama-3.2-11B-Vision-Instruct \
--ignore-patterns "original/consolidated*.pth"

# Start fine-tuning with torchtune
tune run lora_finetune_single_device \
--config configs/llama3_2_vision_11B_lora_single_device.yaml
```

On a single AMD Instinct MI300X GPU, the fine-tuning takes about 25 hours to complete, which we can observe as shown in Figure 1 on the WandB dashboard if the WandBLogger is enabled.

```{figure} ./images/Picture1.png
:align: center
:alt: Scaling performance
Figure 1. The cross-entropy loss over the global steps for the Llama 3.2 11B visual language model.
```

## QLoRA Fine-tuning Llama 3.2 90B Vision Instruct with torchtune

In this section, we are going to train the 90B variant of the model. For this, we will employ the QLoRA method, which quantizes the base model and strategically offloads the optimizer state to reduce peak memory usage. Thanks to the large high bandwidth memory (HBM) of the AMD Instinct MI300X GPU, we are still able to run the model on a single GPU. We start by creating a local configuration llama3_2_vision_90B_qlora.yaml from the default llama3_2_vision/90B_qlora with the following changes:

```bash
# Dataset
dataset:
  _component_: torchtune.datasets.multimodal.vqa_dataset
  source: "cot_chartqa/"
  packed: False
  column_map: {"input":"query", "image":"image", "output":"label"}
seed: 42
shuffle: True
collate_fn: torchtune.data.padded_collate_tiled_images_and_mask
# ...
# Fine-tuning arguments
epochs: 3
max_steps_per_epoch: null
batch_size: 2
gradient_accumulation_steps: 8  # Use to increase effective batch size
# Optionally add WandBLogger
```

Then we can go ahead and download the model before starting the fine-tuning:

```bash
# Download model weights
tune download meta-llama/Llama-3.2-90B-Vision-Instruct \
--output-dir /tmp/Llama-3.2-90B-Vision-Instruct \
--ignore-patterns "original/consolidated*.pth"

# Start fine-tuning
tune run lora_finetune_single_device \
--config configs/llama3_2_vision_90B_qlora.yaml
```

The duration of the three epochs is about 13 hours on a single MI300X GPU.

## Generation Recipes

After the fine-tuning succeeds, we can leverage the experimental generation_v2 recipe of torchtune to generate some outputs. We only need to make slight adjustments to the default configuration [llama3_2_vision/11B_generation_v2](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3_2_vision/11B_generation_v2.yaml) by editing the following parts:

```bash
# Generation arguments
prompt:
  system: |
      <image>{{query}}
      Analyze the image and question carefully, using step-by-step reasoning.
      First, describe any image provided in detail. Then, present your reasoning.
      And finally your final answer in this format:
      Final Answer: <answer>
      where <answer> follows the following instructions:
      - <answer> should be a single phrase or number.
      - <answer> should not paraphrase or reformat the text in the image.
      - If <answer> is a ratio, it should be a decimal value like 0.25 instead of 1:4.
      - If the question is a Yes/No question, <answer> should be Yes/No.
      - If <answer> is a number, it should not contain any units.
      - If <answer> is a percentage, it should include a % sign.
      - If <answer> is an entity, it should include the full label from the graph.
      IMPORTANT: Remember, to end your answer with Final Answer: <answer>.
  user:
    image: example_chart.png
    text: How many monthly sessions per user did Netflix have?
max_new_tokens: 200
temperature: 0.6 # 0.8 and 0.6 are popular values to try
top_k: 300
```

The system prompt is taken from the [Eleuther AI ChartQA task](https://github.com/EleutherAI/lm-evaluation-harness/blob/147e9d616aedab6334d109141dce7492d57088bd/lm_eval/tasks/chartqa/chartqa.yaml#L9). The image corresponding to the query can be created with the following command:

```bash
python -c "from datasets import load_dataset;load_dataset('HuggingFaceM4/ChartQA', split='train')[27901]['image'].save('example_chart.png')"
```

For the 90B parameter model we need to adjust further sections of the 11B configuration:

```bash
# Model arguments
model:
  _component_: torchtune.models.llama3_2_vision.llama3_2_vision_90b
# ...
tokenizer:
  path: /tmp/Llama-3.2-90B-Vision-Instruct/original/tokenizer.model
# ...
checkpointer:
  checkpoint_dir: /tmp/Llama-3.2-90B-Vision-Instruct/
  checkpoint_files:
    max_filename: "00037"
# ...
```

Then we can kick off the generation with one of the commands below:

```bash
# 11B Baseline model
tune run generate_v2.py --config 11B_generation_v2.yaml

# 90B Baseline model
tune run generate_v2.py --config 90B_generation_v2.yaml

# 11B Fine-tuned model
tune run generate_v2.py --config 11B_generation_v2.yaml \
checkpointer.checkpoint_dir=/tmp/torchtune/llama3_2_vision_11B/lora/epoch_9

# 90B Fine-tuned model
tune run generate_v2.py --config 90B_generation_v2.yaml \
checkpointer.checkpoint_dir=/tmp/torchtune/llama3_2_vision_90B/qlora/epoch_2
```

The output of the 11B parameter model looks as follows:

```bash
# Baseline result:

Netflix had 19 monthly sessions per user.

# Fine-tuned result after 10 epochs:

To find the number of monthly sessions per user for Netflix, we need to look at the bar graph. The graph shows the number of monthly sessions per user for various streaming services. We can see that Netflix has a bar that corresponds to 19 monthly sessions per user.

Therefore, the number of monthly sessions per user for Netflix is 19.

Final answer: 19
```

As we can see, the model reasons about what is visible in the given chart and then produces the final answer. While this is a great result, it is based on a sample from the training set, so in the next section, we want to run a complete evaluation of the test split, which was not used during the training process.

## Evaluation with the Eleuther AI lm-harness Recipe

For the evaluation, we are going to use the eleuther_eval recipes which we slightly modify to make it compatible with the multi-modal ChartQA task and run it for both models:

```bash
# Copy the lm_harness recipes
tune cp eleuther_eval eleuther_eval.py

#Apply a patch to support the ChartQA eval task
patch copy_original_eleuther_eval.py <<EOF
--- original_eleuther_eval.py   2025-06-06 04:07:24.905294553 +0000
+++ eleuther_eval.py    2025-06-05 16:28:24.583934752 +0000
@@ -65,0 +66,3 @@
+        image_width = 560,
+        image_height = 560,
+        image_max_side = 560,
@@ -77,0 +81,5 @@
+
+        self.image_width = image_width
+        self.image_height = image_height
+        self.image_max_side = image_max_side
+
@@ -255 +263 @@
-            generated_tokens = []
+            generated_tokens = batch["input_pos"].size(1)*[0,]
@@ -276 +283,0 @@
-
EOF

tune run eleuther_eval.py --config llama3_2_vision/11B_evaluation \
    tasks="['chartqa']" \
    checkpointer.checkpoint_dir=/tmp/torchtune/llama3_2_vision_11B/lora/epoch_9/

tune run eleuther_eval.py --config configs/11B_evaluation \
    tasks="['chartqa']" \
    checkpointer.checkpoint_dir=/tmp/torchtune/llama3_2_vision_90B/qlora/epoch_2/ \
    model._component_=torchtune.models.llama3_2_vision.llama3_2_vision_90b \
    tokenizer.path="/tmp/Llama-3.2-90B-Vision-Instruct/original/tokenizer.model" \
    checkpoint_files.max_filename="00037"
```

## Evaluation Results

The evaluation results for the four models are visualized in Figure 2. We provide the exact_match accuracy of the ChartQA task from lm_harness where the ground truth label is directly compared to the given answer by the model. We can see that both fine-tuned models (90B and 11B parameters) perform better than their base model counterparts, and the fine-tune 11B model now provides 2.3x better accuracy than the 90B base model.

```{figure} ./images/Picture2.png
:align: center
:alt: Scaling performance
Figure 2. Exact match accuracy of the ChartQA task as determined by the Eleuther lm_eval harness for the two base and fine-tuned models. The figure shows a significant improvement for the fine-tuned models over the base (Instruct) models.
```

## Summary

This blog post demonstrates how to fine-tune Llama 3.2 Vision Instruct models (11B and 90B parameters) on a single AMD Instinct MI300X GPU. The hardware used for creating this blog post was provided by TensorWave. It highlights the use of torchtune for efficient fine-tuning, leveraging the GPU's high-bandwidth memory. A key aspect is the creation of a synthetic multi-modal reasoning dataset, enhanced with a chain of thought (CoT) to improve model reasoning. The fine-tuning, utilizing LoRA for the 11B model and QLoRA for the 90B model, significantly improves accuracy. Evaluation results show that the fine-tuned 11B model achieves 2.3x better accuracy than the 90B base model, showcasing substantial performance improvements through this fine-tuning approach. This work illustrates how memory-efficient training techniques and targeted dataset augmentation can unlock high performance even with smaller models on a single GPU.

## Additional Resources

- Aaron Grattafiori et al. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- Torchtune maintainers and contributors. [torchtune: PyTorch's finetuning library](http://https//github.com/pytorch/torchtune)
- Sanyam Bhutani et al. [Unlocking Reasoning in Llama 3](https://github.com/meta-llama/synthetic-data-kit/tree/main/use-cases/adding_reasoning_to_llama_3)
- Ahmed Masry et al. [ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning](https://arxiv.org/abs/2203.10244)
- Leo Gao et al. [The Language Model Evaluation Harness](https://zenodo.org/records/12608602?)
- Torchtune on AMD GPUs [Torchtune AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/torchtune/README.html)
- Fine-tune Llama-3.1 8B with torchtune [Fine tune Llama 3.1 8B](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/torchtune_llama3.html)

## Endnotes

Configuration Details

On average, a system configured with a single AMD Instinct™ MI300X GPU running the LoRA fine-tuning will take about 23 hours for the 11B parameter model and 13 hours for the 90B parameter QLoRA fine-tuning. Testing done on hardware provided by TensorWave by authors on 05/30/2025, results may vary based on configuration, usage, software version, and optimizations.

SYSTEM CONFIGURATION:

AMD Instinct ™ MI300X platform
CPU: 2x AMD EPYC 9654 96-Core Processor
Memory: 2,434 GiB
Disk: 13.97 TiB (4x SAMSUNG MZQL23T8HCLS-00A07 3576 GiB, 2x Micron_7450_MTFDKCB960TFR 960 GiB)
GPU: 1x AMD Instinct MI300X GPU 192GB HBM3 750W
Host OS: Ubuntu 22.04.4
System BIOS: 3.5.0
System Bios Vendor: American Megatrends International, LLC.
Host GPU Driver: (amdgpu version): ROCm 6.4
Pytorch: 2.8.0.dev20250520+rocm6.4
Torchtune: commit 2af4db8
lm_eval: commit 2cfdd0a

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->
