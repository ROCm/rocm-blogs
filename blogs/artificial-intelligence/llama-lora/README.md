---
blogpost: true
date: 1 Feb 2024
author: Sean Song
blog_title: Fine-tune Llama model with LoRA\: Customizing a large language model for question-answering
tags: LLM, AI/ML, GenAI, Fine-Tuning
category: Applications & models
language: English
thumbnail: './images/image.jpg'
myst:
  html_meta:
    "description lang=en": "This blog demonstrate how to use Lora to efficiently fine-tune Llama model on a single AMD GPU with ROCm.
  model for question-answering"
    "keywords": "LoRA, Low-rank Adaptation, fine-tuning, large language model,
  Generative AI"
    "property=og:locale": "en_US"
---

# Fine-tune Llama model with LoRA: Customizing a large language model for question-answering

<span style="font-size:0.7em;">1, Feb 2024 by {hoverxref}`Sean Song<seansong>`. </span>

In this blog, we show you how to fine-tune a Llama model on an AMD GPU with ROCm. We use Low-Rank
Adaptation of Large Language Models (LoRA) to overcome memory and computing limitations and
make open-source large language models (LLMs) more accessible. We also show you how to
fine-tune and upload models to Hugging Face.

## Introduction

In the dynamic realm of Generative AI (GenAI), fine-tuning LLMs (such as Llama models) poses distinctive
challenges related to substantial computational and memory requirements. LoRA introduces a
compelling solution, allowing rapid and cost-effective fine-tuning of state-of-the-art LLMs. This
breakthrough capability not only expedites the tuning process, but also lowers associated costs.

To explore the benefits of LoRA, we provide a comprehensive walkthrough of the fine-tuning process
for Llama 2 using LoRA specifically tailored for question-answering (QA) tasks on an AMD GPU.

Before jumping in, let's take a moment to briefly review the three pivotal components that form the
foundation of our discussion:

* Llama model: Meta's advanced language model with variants that scale up to 405 billion parameters.
* Fine-tuning: A crucial process that refines LLMs for specialized tasks, optimizing its performance.
* LoRA: The algorithm employed for fine-tuning Llama model, ensuring effective adaptation to specialized
  tasks.

### Llama models

<!-- [Llama](https://arxiv.org/abs/2407.21783) is a collection of second-generation, open-source LLMs
from Meta; it comes with a commercial license. Llama 2 is designed to handle a wide range of natural
language processing (NLP) tasks, with models ranging in scale from 7 billion to 70 billion parameters. -->

The [Meta Llama collection](https://arxiv.org/abs/2407.21783) consists of multilingual large language models (LLMs) in three sizes: 7B, 70B, and 405B parameters. These pretrained and instruction-tuned generative models support text input and output. Built on an optimized transformer architecture, the models are auto-regressive by design. The instruction-tuned versions are further refined using supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF), ensuring alignment with human preferences for helpfulness and safety.

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

## Step-by-step Llama fine-tuning

Standard (full-parameter) fine-tuning involves considering all parameters. It requires significant
computational power to manage optimizer states and gradient check-pointing. The resulting memory
footprint is typically about four times larger than the model itself. For example, loading a 7 billion
parameter model (e.g. Llama 2) in FP32 (4 bytes per parameter) requires approximately 28 GB of GPU
memory, while fine-tuning demands around 28*4=112 GB of GPU memory. Note that the 112 GB
figure is derived empirically, and various factors like batch size, data precision, and gradient
accumulation contribute to overall memory usage.

To overcome this memory limitation, you can use a parameter-efficient fine-tuning (PEFT) technique,
such as LoRA.

This example leverages one AMD [MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) GPU equipped with 192 GB of VRAM. Using this setup allows us to explore different settings for fine-tuning the Llama 2 weights with and without LoRA.

>**Note:**
Based on the configurations outlined in the blog, you will need at least 84 GB of VRAM for LoRA fine-tuning and 144 GB for full parameter fine-tuning. If your GPU has less VRAM, consider using a smaller model, reducing the batch size, or lowering the precision to accommodate the limitations.

Our setup:

* Hardware & OS: See [this link](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for a list of supported hardware and OS with ROCm.
* Software:
  * [ROCm 6.1.0+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
  * [Pytorch 2.0.1+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html#installing-pytorch-for-rocm)
* Libraries: `transformers`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `scipy`

In this blog, we conducted our experiment using a single MI300X GPU with the Docker image [rocm/pytorch:rocm6.1.2_ubuntu22.04_py3.10_pytorch_release-2.1.2](https://hub.docker.com/layers/rocm/pytorch/rocm6.1.2_ubuntu22.04_py3.10_pytorch_release-2.1.2/images/sha256-c8b4e8dfcc64e9bf68bf1b38a16fbc5d65b653ec600f98d3290f66e16c8b6078?context=explore).

### Step 1: Getting started

First, let's confirm the availability of the GPU.

```bash
!rocm-smi --showproductname
```

Your output should look like this:

```bash
============================ ROCm System Management Interface ============================
====================================== Product Info ======================================
GPU[0]		: Card Series: 		AMD Instinct MI300X OAM
GPU[0]		: Card Model: 		0x74a1
GPU[0]		: Card Vendor: 		Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0]		: Card SKU: 		MI3SRIOV
GPU[0]		: Subsystem ID: 	0x74a1
GPU[0]		: Device Rev: 		0x00
GPU[0]		: Node ID: 		2
GPU[0]		: GUID: 		47056
GPU[0]		: GFX Version: 		gfx942
==========================================================================================
================================== End of ROCm SMI Log ===================================
```

Next, install the required libraries.

```bash
!pip install -q pandas peft transformers trl accelerate scipy
```

#### Import the required packages

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer
```

### Step 2: Configuring the model and data

You can access Meta's official Llama model from Hugging Face after making a request, which can
take a couple of days. Instead of waiting, we'll use NousResearch’s Llama-2-7b-chat-hf as our base
model (it's the same as the original, but quicker to access).

```python
# Model and tokenizer names
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model_name = "Llama-2-7b-chat-hf-enhanced" #You can give your own name for fine tuned model

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

### Step 3: Start fine-tuning

To set your training parameters, use the following code:

```python
# Training Params
train_params = TrainingArguments(
    output_dir="./results_modified",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
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
    lr_scheduler_type="constant"
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

Note that there are only 0.0622% parameters added by LoRA, which is a tiny portion of the original
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
50	     1.978400 \
100	     1.613600 \
150	     1.411400 \
200	     1.397700 \
250	     1.378800 \

TrainOutput(global_step=250, training_loss=1.555977081298828, metrics={'train_runtime': 196.2034, 'train_samples_per_second': 5.097, 'train_steps_per_second': 1.274, 'total_flos': 1.701064079130624e+16, 'train_loss': 1.555977081298828, 'epoch': 1.0})
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
============================================= ROCm System Management Interface =============================================
======================================================= Concise Info =======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK     MCLK     Fan  Perf  PwrCap  VRAM%  GPU%  
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)                                                    
============================================================================================================================
0       2     0x74a1,   47056  71.0°C      682.0W    NPS1, SPX, 0        1653Mhz  1300Mhz  0%   auto  750.0W  43%    100%   
============================================================================================================================
=================================================== End of ROCm SMI Log ====================================================
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
50	     1.721100 \
100	     1.491000 \
150	     1.366400 \
200	     1.374200 \
250	     1.369800 \

TrainOutput(global_step=250, training_loss=1.4645071716308593, metrics={'train_runtime': 683.314, 'train_samples_per_second': 1.463, 'train_steps_per_second': 0.366, 'total_flos': 1.6999849383985152e+16, 'train_loss': 1.4645071716308593, 'epoch': 1.0})
```

##### Checking memory usage during training without LoRA

During training, you can check the memory usage by running the `rocm-smi` command in a terminal.
This command produces the following output:

```python
============================================= ROCm System Management Interface =============================================
======================================================= Concise Info =======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK     MCLK     Fan  Perf  PwrCap  VRAM%  GPU%  
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)                                                    
============================================================================================================================
0       2     0x74a1,   47056  79.0°C      728.0W    NPS1, SPX, 0        1618Mhz  1300Mhz  0%   auto  750.0W  75%    100%   
============================================================================================================================
=================================================== End of ROCm SMI Log ====================================================
```

### Step 4: Comparison between fine-tuning with LoRA and full-parameter fine-tuning

Comparing the results from the *Training with LoRA configuration* and
*Training without LoRA configuration* sections, note the following:

* Memory usage:
  * In the case of full-parameter fine-tuning, there are **6,742,609,920** trainable parameters, leading
        to significant memory consumption during the training back propagation stage.
  * LoRA only introduces **4,194,304** trainable parameters, accounting for **0.06%** of the total
        trainable parameters in full-parameter fine-tuning.
  * Monitoring memory usage during training with and without LoRA reveals that fine-tuning with LoRA
        uses only **57%** of the memory consumed by full-parameter fine-tuning. This presents an
        opportunity to increase batch size and max sequence length, and train on larger data sets using
        limited hardware resources.

* Training speed:
  * The results demonstrate that full-parameter fine-tuning takes **12 minutes** to complete, while
        fine-tuning with LoRA finishes in around **3 minutes**. Fewer trainable parameters in LoRA translate to fewer derivative calculations and less memory required to store and update weights.

* Accuracy:
  * In both training sessions, a notable reduction in training loss was observed. A closely
        aligned training loss is achieved for two both approaches: **1.369** for full-parameter fine-tuning and
        **1.379** for fine-tuning with LoRA. If you're interested in understanding the impact of LoRA on
        fine-tuning performance, refer to [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

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

```python
from transformers import GenerationConfig
def inference(llm_model, tokenizer, user_input):
  def formatted_prompt(question)-> str:
    return f"<|start|>user\n{question}<|end|>\n<|start|>assistant:"
  prompt = formatted_prompt(user_input)
  inputs = tokenizer([prompt], return_tensors="pt")
  generation_config = GenerationConfig(penalty_alpha=0.6, do_sample = True,top_k=3, temperature=0.5,
                                       repetition_penalty=1.2, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id
                                      )
  inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
  outputs = llm_model.generate(**inputs, generation_config=generation_config)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Now you can test with the base model (original) and your fine-tuned model.

* Base model:

    ```python
    # Generate text using base model
    inference(base_model, tokenizer, user_input='What do you think is the most important part of building an AI chatbot?'))
    ```

    ```python
    # Outputs:
    <|start|>user
    What do you think is the most important part of building an AI chatbot?<|end|>
    <|start|>assistant: The most critical aspect in developing a conversational agent, or ChatBot as it's commonly known these days, lies in creating a natural language understanding model that can accurately interpret and respond to user inputs. This requires extensive training data, robust linguistic processing techniques, and continuous refinement over time.
    The other key component for any successful chatbot implementation would be its ability to provide relevant information quickly while maintaining high levels of accuracy throughout all interactions with users regardless of their background knowledge base. Additionally, having strong integration capabilities into existing systems such as CRM platforms will allow businesses better insights about customer behavior which could help them improve overall service quality.<|end|>
    ```

* Fine-tuned model:

    ```python
    # Generate text using fine-tuned model
    inference(model_enhanced, tokenizer, user_input='What do you think is the most important part of building an AI chatbot?')
    ```

    ```python
    # Outputs:
    <|start|>user
    What do you think is the most important part of building an AI chatbot?<|end|>
    <|start|>assistant: 
    The most important parts of a good conversational agent are:
    1. Natural language processing (NLP) capabilities, which allow it to understand and respond appropriately to human input.
    2. A large corpus of training data that can be used for machine learning algorithms to improve its performance over time.
    3. An interface or platform through which users interact with the bot, such as voice recognition software or text messaging applications.

    4. Regular maintenance and updates by developers who ensure that the system remains up-to-date on current trends in NLP research while also keeping track of any potential security vulnerabilities.<|end|>
    ```

You can observe the outputs of the two models based on a given query. These outputs exhibit slight
differences due to the fine-tuning process altering the model weights.
