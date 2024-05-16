---
blogpost: true
date: 1 Feb 2024
author: Sean Song
tags: LLM, AI/ML, Generative AI, Tuning
category: Applications & models
language: English

myst:
  html_meta:
    "description lang=en": "Fine-tune Llama 2 with LoRA: Customizing a large language
  model for question-answering"
    "keywords": "LoRA, Low-rank Adaptation, fine-tuning, large language model,
  Generative AI"
    "property=og:locale": "en_US"
---

# Fine-tune Llama 2 with LoRA: Customizing a large language model for question-answering

In this blog, we show you how to fine-tune Llama 2 on an AMD GPU with ROCm. We use Low-Rank
Adaptation of Large Language Models (LoRA) to overcome memory and computing limitations and
make open-source large language models (LLMs) more accessible. We also show you how to
fine-tune and upload models to Hugging Face.

## Introduction

In the dynamic realm of Generative AI (GenAI), fine-tuning LLMs (such as Llama 2) poses distinctive
challenges related to substantial computational and memory requirements. LoRA introduces a
compelling solution, allowing rapid and cost-effective fine-tuning of state-of-the-art LLMs. This
breakthrough capability not only expedites the tuning process, but also lowers associated costs.

To explore the benefits of LoRA, we provide a comprehensive walkthrough of the fine-tuning process
for Llama 2 using LoRA specifically tailored for question-answering (QA) tasks on an AMD GPU.

Before jumping in, let's take a moment to briefly review the three pivotal components that form the
foundation of our discussion:

* Llama 2: Meta's advanced language model with variants that scale up to 70 billion parameters.
* Fine-tuning: A crucial process that refines LLMs for specialized tasks, optimizing its performance.
* LoRA: The algorithm employed for fine-tuning Llama 2, ensuring effective adaptation to specialized
  tasks.

### Llama 2

[Llama 2](https://arxiv.org/abs/2307.09288) is a collection of second-generation, open-source LLMs
from Meta; it comes with a commercial license. Llama 2 is designed to handle a wide range of natural
language processing (NLP) tasks, with models ranging in scale from 7 billion to 70 billion parameters.

Llama 2 Chat, which is optimized for dialogue, has shown similar performance to popular
closed-source models like ChatGPT and PaLM. You can improve the performance of this model by
fine-tuning it with a high-quality conversational data set. In this blog post, we delve into the process of
refining a Llama 2 Chat model using a QA data set.

### Fine-tuning a model

Fine-tuning in machine learning is the process of adjusting the weights and parameters of a
pre-trained model using new data in order to improve its performance on a specific task. It involves
using a new data set--one that is specific to the current task--to update the model's weights. It's
typically not possible to fine-tune LLMs on consumer hardware due to inadequate memory and
computing power. However, in this tutorial, we use LoRA to overcome these challenges.

### LoRA

[LoRA](https://arxiv.org/abs/2106.09685) is an innovative technique-- developed by researchers at
Microsoft--designed to address the challenges of fine-tuning LLMs. This results in a significant
reduction in the number of parameters (by a factor of up to 10,000) that need to be fine-tuned, which
significantly reduces GPU memory requirements. To learn more about the fundamental principles of LoRA, refer to [Using LoRA for efficient fine-tuning: Fundamental principles](../lora-fundamentals/README.md).

## Step-by-step Llama 2 fine-tuning

Standard (full-parameter) fine-tuning involves considering all parameters. It requires significant
computational power to manage optimizer states and gradient check-pointing. The resulting memory
footprint is typically about four times larger than the model itself. For example, loading a 7 billion
parameter model (e.g. Llama 2) in FP32 (4 bytes per parameter) requires approximately 28 GB of GPU
memory, while fine-tuning demands around 28*4=112 GB of GPU memory. Note that the 112 GB
figure is derived empirically, and various factors like batch size, data precision, and gradient
accumulation contribute to overall memory usage.

To overcome this memory limitation, you can use a parameter-efficient fine-tuning (PEFT) technique,
such as LoRA.

This example leverages two GCDs (Graphics Compute Dies) of a AMD MI250 GPU and each GCD are equipped with 64 GB of VRAM. Using this setup allows us to explore different settings for fine-tuning the Llama 2–7b weights with and without LoRA.

Our setup:

* Hardware & OS: See [this link](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for a list of supported hardware and OS with ROCm.
* Software:
  * [ROCm 6.1.0+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
  * [Pytorch 2.0.1+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html#installing-pytorch-for-rocm)
* Libraries: `transformers`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `scipy`

In this blog, we conducted our experiment using a single MI250GPU with the Docker image [rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.1.2](https://hub.docker.com/layers/rocm/pytorch/rocm6.1_ubuntu22.04_py3.10_pytorch_2.1.2/images/sha256-f6ea7cee8aae299c7f6368187df7beed29928850c3929c81e6f24b34271d652b?context=explore).

### Step 1: Getting started

First, let's confirm the availability of the GPU.

```bash
!rocm-smi --showproductname
```

Your output should look like this:

```bash
========================= ROCm System Management Interface =========================
=================================== Product Info ===================================
GPU[0]      : Card series:      AMD INSTINCT MI250 (MCM) OAM AC MBA
GPU[0]      : Card model:      0x0b0c
GPU[0]      : Card vendor:      Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0]      : Card SKU:      D65209
GPU[1]      : Card series:      AMD INSTINCT MI250 (MCM) OAM AC MBA
GPU[1]      : Card model:      0x0b0c
GPU[1]      : Card vendor:      Advanced Micro Devices, Inc. [AMD/ATI]
GPU[1]      : Card SKU:      D65209
====================================================================================
=============================== End of ROCm SMI Log ================================
```

Next, install the required libraries.

```bash
!pip install -q pandas peft==0.9.0 transformers==4.31.0 trl==0.4.7 accelerate scipy
```

#### Install bitsandbytes

1. Install bitsandbytes using the following code.

    ```bash
    git clone --recurse https://github.com/ROCm/bitsandbytes
    cd bitsandbytes
    git checkout rocm_enabled
    pip install -r requirements-dev.txt
    cmake -DCOMPUTE_BACKEND=hip -S . #Use -DBNB_ROCM_ARCH="gfx90a;gfx942" to target specific gpu arch
    make
    pip install .
    ```

2. Check the bitsandbytes version.

    At the time of writing this blog, the version is 0.43.0.

    ```bash
    %%bash
    pip list | grep bitsandbytes
    ```

#### Import the required packages

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer
```

### Step 2: Configuring the model and data

You can access Meta's official Llama-2 model from Hugging Face after making a request, which can
take a couple of days. Instead of waiting, we'll use NousResearch’s Llama-2-7b-chat-hf as our base
model (it's the same as the original, but quicker to access).

```python
# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model_name = "llama-2-7b-enhanced" #You can give your own name for fine tuned model

# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto"
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
```

After you have the base model, you can start fine-tuning. We fine-tune our base model for a
question-and-answer task using a small data set called
[mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k), which
is a subset (1,000 samples) of the
[timdettmers/openassistant-guanaco](https://huggingface.co/datasets/OpenAssistant/oasst1) data set.
This data set is a human-generated, human-annotated, assistant-style conversation corpus that
contains 161,443 messages in 35 different languages, annotated with 461,292 quality ratings. This
results in over 10,000 fully annotated conversation trees.

```python
# Data set
data_name = "mlabonne/guanaco-llama2-1k"
training_data = load_dataset(data_name, split="train")
# check the data
print(training_data.shape)
# #11 is a QA sample in English
print(training_data[11])
```

```python
(1000, 1)
{'text': '<s>[INST] write me a 1000 words essay about deez nuts. [/INST] The Deez Nuts meme first gained popularity in 2015 on the social media platform Vine. The video featured a young man named Rodney Bullard, who recorded himself asking people if they had heard of a particular rapper. When they responded that they had not, he would respond with the phrase "Deez Nuts" and film their reactions. The video quickly went viral, and the phrase became a popular meme. \n\nSince then, Deez Nuts has been used in a variety of contexts to interrupt conversations, derail discussions, or simply add humor to a situation. It has been used in internet memes, in popular music, and even in politics. In the 2016 US presidential election, a 15-year-old boy named Brady Olson registered as an independent candidate under the name Deez Nuts...</s>'}
```

```python
## There is a dependency during training
!pip install tensorboardX
```

### Step 3: Start fine-tuning

To set your training parameters, use the following code:

```python
# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=50,
    learning_rate=4e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)
```

#### Training with LoRA configuration

Now you can integrate LoRA into the base model and assess its additional parameters. LoRA essentially
adds pairs of rank-decomposition weight matrices (called update matrices) to existing weights, and
only trains the newly added weights.

```python
from peft import get_peft_model
# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=8,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, peft_parameters)
model.print_trainable_parameters()
```

The output looks like this:

```python
trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06220594176090199
```

Note that there are only 0.062% parameters added by LoRA, which is a tiny portion of the original
model. This is the percentage we'll update through fine-tuning, as follows.

```python
# Trainer with LoRA configuration
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)

# Training
fine_tuning.train()
```

The output looks like this:

```python
[250/250 07:59, Epoch 1/1]\
Step     Training Loss \
50       1.976400 \
100      1.613500\
150      1.409100\
200      1.391500\
250      1.377300

TrainOutput(global_step=250, training_loss=1.5535581665039062, metrics={'train_runtime': 484.7942, 'train_samples_per_second': 2.063, 'train_steps_per_second': 0.516, 'total_flos': 1.701064079130624e+16, 'train_loss': 1.5535581665039062, 'epoch': 1.0})
```

To save your model, run this code:

```python
# Save Model
fine_tuning.model.save_pretrained(new_model_name)
```

##### Checking memory usage during training with LoRA

During training, you can check the memory usage by running the `rocm-smi` command in a terminal.
This command produces the following output:

```python
======================= ROCm System Management Interface ====================
=============================== Concise Info ================================
GPU  Temp (DieEdge)  AvgPwr  SCLK     MCLK     Fan  Perf  PwrCap  VRAM%  GPU%
0    52.0c           179.0W  1700Mhz  1600Mhz  0%   auto  300.0W   65%   100%
1    52.0c           171.0W  1650Mhz  1600Mhz  0%   auto  300.0W   66%   100%
=============================================================================
============================ End of ROCm SMI Log ============================
```

To facilitate a comparison between fine-tuning with and without LoRA, our subsequent phase involves
running a thorough fine-tuning process on the base model. This involves updating all parameters
within the base model. We then analyze differences in memory usage, training speed, training loss, and
other relevant metrics.

#### Training without LoRA configuration

*For this section, you must restart the kernel and skip the 'Training with LoRA configuration' section.*

For a direct comparison between models using the same criteria, we maintain consistent settings
(without any alterations) for `train_params` during the full-parameter fine-tuning process.

To check the trainable parameters in your base model, use the following code.

```python
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

print_trainable_parameters(base_model)
```

The output looks like this:

```python
trainable params: 6738415616 || all params: 6738415616 || trainable%: 100.00
```

Continue the process using the following code:

```python
# Set a lower learning rate for fine-tuning
train_params.learning_rate = 4e-7
print(train_params.learning_rate)
```

```python
# Trainer without LoRA configuration

fine_tuning_full = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params
)

# Training
fine_tuning_full.train()
```

The output looks like this:

```python
[250/250 3:02:12, Epoch 1/1]\
Step     Training Loss\
50       1.712300\
100      1.487000\
150      1.363800\
200      1.371100\
250      1.368300

TrainOutput(global_step=250, training_loss=1.4604909362792968, metrics={'train_runtime': 10993.7995, 'train_samples_per_second': 0.091, 'train_steps_per_second': 0.023, 'total_flos': 1.6999849383985152e+16, 'train_loss': 1.4604909362792968, 'epoch': 1.0})
```

##### Checking memory usage during training without LoRA

During training, you can check the memory usage by running the `rocm-smi` command in a terminal.
This command produces the following output:

```python
======================= ROCm System Management Interface ====================
=============================== Concise Info ================================
GPU  Temp (DieEdge)  AvgPwr  SCLK     MCLK     Fan  Perf  PwrCap  VRAM%  GPU%
0    40.0c           44.0W   800Mhz   1600Mhz  0%   auto  300.0W   100%  89%
1    39.0c           50.0W   1700Mhz  1600Mhz  0%   auto  300.0W   100%  85%
=============================================================================
============================ End of ROCm SMI Log ============================
```

### Step 4: Comparison between fine-tuning with LoRA and full-parameter fine-tuning

Comparing the results from the *Training with LoRA configuration* and
*Training without LoRA configuration* sections, note the following:

* Memory usage:
  * In the case of full-parameter fine-tuning, there are **6,738,415,616** trainable parameters, leading
        to significant memory consumption during the training back propagation stage.
  * LoRA only introduces **4,194,304** trainable parameters, accounting for **0.062%** of the total
        trainable parameters in full-parameter fine-tuning.
  * Monitoring memory usage during training with and without LoRA reveals that fine-tuning with LoRA
        uses only **65%** of the memory consumed by full-parameter fine-tuning. This presents an
        opportunity to increase batch size and max sequence length, and train on larger data sets using
        limited hardware resources.

* Training speed:
  * The results demonstrate that full-parameter fine-tuning takes **hours** to complete, while
        fine-tuning with LoRA finishes in less than **9 minutes**. Several factors contribute to this
        acceleration:
    * Fewer trainable parameters in LoRA translate to fewer derivative calculations and less memory
        required to store and update weights.
    * Full-parameter fine-tuning is more prone to being memory-bound, where data movement
        becomes a bottleneck for training. This is reflected in lower GPU utilization. Although adjusting
        training settings can alleviate this, it may require more resources (additional GPUs) and a smaller
        batch size.

* Accuracy:
  * In both training sessions, a notable reduction in training loss was observed. We achieved a closely
        aligned training loss for two both approaches: **1.368** for full-parameter fine-tuning and
        **1.377** for fine-tuning with LoRA. If you're interested in understanding the impact of LoRA on
        fine-tuning performance, refer to
        [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

### Step 5: Test the fine-tuned model with LoRA

To test your model, run the following code:

```python
# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
from peft import LoraConfig, PeftModel
model = PeftModel.from_pretrained(base_model, new_model_name)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

The output looks like this:

```python
    Loading checkpoint shards: 100%|██████████| 2/2 [00:04<00:00,  2.34s/it]
```

Uploading the model to Hugging Face let's you conduct subsequent tests or share your model with
others (to proceed with this step, you'll need an active Hugging Face account).

```python
from huggingface_hub import login
# You need to use your Hugging Face Access Tokens
login("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
# Push the model to Hugging Face. This takes minutes and time depends the model size and your
# network speed.
model.push_to_hub(new_model_name, use_temp_dir=False)
tokenizer.push_to_hub(new_model_name, use_temp_dir=False)
```

Now you can test with the base model (original) and your fine-tuned model.

* Base model:

    ```python
    # Generate text using base model
    query = "What do you think is the most important part of building an AI chatbot?"
    text_gen = pipeline(task="text-generation", model=base_model_name, tokenizer=llama_tokenizer, max_length=200)
    output = text_gen(f"<s>[INST] {query} [/INST]")
    print(output[0]['generated_text'])
    ```

    ```python
    # Outputs:
    <s>[INST] What do you think is the most important part of building an AI chatbot? [/INST]  There are several important aspects to consider when building an AI chatbot, but here are some of the most critical elements:

    1. Natural Language Processing (NLP): A chatbot's ability to understand and interpret human language is crucial for effective communication. NLP is the foundation of any chatbot, and it involves training the AI model to recognize patterns in language, interpret meaning, and generate responses.
    2. Conversational Flow: A chatbot's conversational flow refers to the way it interacts with users. A well-designed conversational flow should be intuitive, easy to follow, and adaptable to different user scenarios. This involves creating a dialogue flowchart that guides the conversation and ensures the chatbot responds appropriately to user inputs.
    3. Domain Knowledge: A chat
    ```

* Fine-tuned model:

    ```python
    # Generate text using fine-tuned model
    query = "What do you think is the most important part of building an AI chatbot?"
    text_gen = pipeline(task="text-generation", model=new_model_name, tokenizer=llama_tokenizer, max_length=200)
    output = text_gen(f"<s>[INST] {query} [/INST]")
    print(output[0]['generated_text'])
    ```

    ```python
    # Outputs:
    <s>[INST] What do you think is the most important part of building an AI chatbot? [/INST] The most important part of building an AI chatbot is to ensure that it is able to understand and respond to user input in a way that is both accurate and natural-sounding. This requires a combination of natural language processing (NLP) capabilities and a well-designed conversational flow.

    Here are some key factors to consider when building an AI chatbot:

    1. Natural Language Processing (NLP): The chatbot must be able to understand and interpret user input, including both text and voice commands. This requires a robust NLP engine that can handle a wide range of language and dialects.
    2. Conversational Flow: The chatbot must be able to respond to user input in a way that is both natural and intuitive. This requires a well-designed conversational flow that can handle a wide range
    ```

You can observe the outputs of the two models based on a given query. These outputs exhibit slight
differences due to the fine-tuning process altering the model weights.
