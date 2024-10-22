---
blogpost: true
date: 15 April 2024
author: Sean Song
blog_title: Enhancing LLM Accessibility\: A Deep Dive into QLoRA Through Fine-tuning Llama Model on a single AMD GPU
tags: LLM, AI/ML, GenAI, Fine-Tuning
category: Applications & models
language: English
thumbnail: './images/image.jpg'
myst:
  html_meta:
    "description lang=en": "This blog demonstrate how to use QLora to efficiently fine-tune Llama model on a single AMD GPU with ROCm."
    "keywords": "LoRA, Low-rank Adaptation, QLoRA, peft, Quantization, Fine-tuning, Large Language Model, MI210, MI250, MI300, ROCm  Generative AI"
    "property=og:locale": "en_US"
---

# Enhancing LLM Accessibility: A Deep Dive into QLoRA Through Fine-tuning Llama Model on a single AMD GPU

<span style="font-size:0.7em;">15, Apr 2024 by {hoverxref}`Sean Song<seansong>`. </span>

Building on the previous blog [Fine-tune Llama model with LoRA: Customizing a large language model for question-answering](https://rocm.blogs.amd.com/artificial-intelligence/llama-lora/README.html), we delve into another Parameter Efficient Fine-Tuning (PEFT) approach known as Quantized Low Rank Adaptation (QLoRA). The focus will be on leveraging QLoRA for the fine-tuning of Llama-2 7B model using a single AMD GPU with ROCm. This task, made possible through the use of QLoRA, addresses challenges related to memory and computing limitations. The exploration aims to showcase how QLoRA can be employed to enhance accessibility to open-source large language models.

## QLoRA Fine-tuning <a class="anchor" id="Introduction"></a>

[QLoRA](https://arxiv.org/abs/2305.14314) is a fine-tuning technique that combines a high-precision computing technique with a low-precision storage method. This helps keep the model size small while making sure the model is still highly performant and accurate.

### How does QLoRA work?

In few words, QLoRA optimizes the memory usage of LLM fine-tuning without compromising performance, in contrast to standard 16-bit model fine-tuning. Specifically, QLoRA employs 4-bit quantization to compress a pretrained language model. The language model parameters are then frozen, and a modest number of trainable parameters are introduced in the form of Low-Rank Adapters. During fine-tuning, QLoRA backpropagates gradients through the frozen 4-bit quantized pretrained language model into the Low-Rank Adapters. Notably, only the LoRA layers undergo updates during training. For a more in-depth exploration of LoRA, refer to the original [LoRA](https://arxiv.org/abs/2106.09685) paper.

### QLoRA vs LoRA

QLoRA and LoRA represent two parameter-efficient fine-tuning techniques. LoRA operates as a standalone fine-tuning method, while QLoRA incorporates LoRA as an auxiliary mechanism to address errors introduced during the quantization process and to additionally minimize the resource requirements during fine-tuning.

## Step-by-step Llama model fine-tuning with QLoRA<a class="anchor" id="Step-By-Step-Guide"></a>

This section will guide you through the steps to fine-tune the Llama 2 model, which has 8 billion parameters, on a single AMD GPU. The key to this accomplishment lies in the crucial support of QLoRA, which plays an indispensable role in efficiently reducing memory requirements.

For that, we will use the following setup:

* Hardware & OS: See [this link](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for a list of supported hardware and OS with ROCm.
* Software:
  * [ROCm 6.1.0+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
  * [Pytorch for ROCm 2.0+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)
* Libraries: `transformers`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `scipy`

In this blog, we conducted our experiment using a single MI300X GPU with the Docker image [rocm/pytorch:rocm6.1.2_ubuntu22.04_py3.10_pytorch_release-2.1.2](https://hub.docker.com/layers/rocm/pytorch/rocm6.1.2_ubuntu22.04_py3.10_pytorch_release-2.1.2/images/sha256-c8b4e8dfcc64e9bf68bf1b38a16fbc5d65b653ec600f98d3290f66e16c8b6078?context=explore).

You can find the complete code used in this blog from the [Github repo](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/llama-Qlora).

### 1: Getting started

Our first step is to confirm the availability of GPU.

```python
!rocm-smi --showproductname
```

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

We will start by installing the required libraries.

```python
pip install -q pandas peft transformers trl accelerate scipy
```

#### Installing bitsandbytes

ROCm needs a special version of bitsandbytes (`bitsandbytes-rocm`).

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

3. Import the required packages.

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

### 2. Configuring the model and data

#### Model configuration

You can access Meta's official Llama-2 model from Hugging Face after making a request, which can
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
```

#### QLoRA 4-bit quantization configuration

As outlined in the paper, QLoRA stores weights in 4-bits, allowing computation to occur in 16 or 32-bit precision. This means whenever a QLoRA weight tensor is used, we dequantize the tensor to 16 or 32-bit precision, and then perform a matrix multiplication. Various combinations, such as float16, bfloat16, float32, etc., can be chosen. Experimentation with different 4-bit quantization variants, including normalized float 4 (NF4), or pure float4 quantization, is possible. However, guided by theoretical considerations and empirical findings from the paper, the recommendation is to opt for NF4 quantization, as it tends to deliver better performance.

In our case, we chose the following configuration:

* 4-bit quantization with NF4 type
* 16-bit (float16) for computation
* Double quantization, which uses a second quantization after the first one to save an additional 0.3 bits per parameters

Quantization parameters are controlled from the BitsandbytesConfig (see [Hugging Face documentation](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)) as follows:

* Loading in 4 bits is activated through load_in_4bit
* The datatype used for quantization is specified with bnb_4bit_quant_type. Note that there are two supported quantization datatypes fp4 (four-bit float) and nf4 (normal four-bit float). The latter is theoretically optimal for normally distributed weights, so we recommend using nf4.
* The datatype used for the linear layer computations with bnb_4bit_compute_dtype
* Nested quantization is activated through bnb_4bit_use_double_quant

```python
# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
```

Load the model and set the quantization configuration.

```python
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="auto"
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
```

#### Dataset configuration

We fine-tune our base model for a
question-and-answer task using a small data set called
[mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k), which
is a subset (1,000 samples) of the
[timdettmers/openassistant-guanaco](https://huggingface.co/datasets/OpenAssistant/oasst1) data set.
This data set is a human-generated, human-annotated, assistant-style conversation corpus that
contains 161,443 messages in 35 different languages, annotated with 461,292 quality ratings. This
results in over 10,000 fully annotated conversation trees.

```python
# Dataset
data_name = "mlabonne/guanaco-llama2-1k"
training_data = load_dataset(data_name, split="train")
# check the data
print(training_data.shape)
# #11 is a QA sample in English
print(training_data[11])
```

```bash
    (1000, 1)
    {'text': '<s>[INST] write me a 1000 words essay about deez nuts. [/INST] The Deez Nuts meme first gained popularity in 2015 on the social media platform Vine. The video featured a young man named Rodney Bullard, who recorded himself asking people if they had heard of a particular rapper. When they responded that they had not, he would respond with the phrase "Deez Nuts" and film their reactions. The video quickly went viral, and the phrase became a popular meme. \n\nSince then, Deez Nuts has been used in a variety of contexts to interrupt conversations, derail discussions, or simply add humor to a situation. It has been used in internet memes, in popular music, and even in politics. In the 2016 US presidential election, a 15-year-old boy named Brady Olson registered as an independent candidate under the name Deez Nuts. He gained some traction in the polls and even made appearances on national news programs.\n\nThe Deez Nuts meme has had a significant impact on popular culture. It has become a recognizable catchphrase that people use to add humor to everyday conversations. The meme has also been used to satirize politics and other serious issues. For example, in 2016, a group of activists in the UK used the phrase "Deez Nuts for President" as part of a campaign to encourage young people to vote in the EU referendum. </s><s>[INST] Rewrite the essay in a more casual way. Instead of sounding proffesional, sound like a college student who is forced to write the essay but refuses to do so in the propper way. Use casual words and slang when possible. [/INST] Yo, so you want me to write a 1000-word essay about Deez Nuts? Alright, fine. So, this whole thing started on Vine back in 2015. Some dude named Rodney Bullard made a video where he would ask people if they knew a rapper, and when they said no, he would hit them with the classic line: "Deez Nuts!" People loved it, and it became a viral meme.\n\nNowadays, Deez Nuts is used for all kinds of stuff. You can throw it out there to interrupt someone or just to be funny. It\'s all over the internet, in music, and even in politics. In fact, during the 2016 US presidential election, a kid named Brady Olson registered as an independent candidate under the name Deez Nuts. He actually got some attention from the media and made appearances on TV and everything.\n\nThe impact of Deez Nuts on our culture is pretty huge. It\'s become a thing that everyone knows and uses to add some humor to their everyday conversations. Plus, people have used it to make fun of politics and serious issues too. Like, in the UK, some groups of activists used the phrase "Deez Nuts for President" to encourage young people to vote in the EU referendum.\n\nThere you have it, a thousand words about Deez Nuts in a more casual tone. Can I go back to playing video games now? </s>'}

```

```python
## There is a dependency during training
!pip install tensorboardX
```

### 3. Start fine tuning

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
    learning_rate=2e-4,
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

### Training with QLoRA configuration<a class="anchor" id="Training_with_LoRA_configuration"></a>

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

``` bash
    trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06220594176090199
```

Note that there are only 0.062% parameters added by LoRA, which is a tiny portion of the original
model. This is the percentage we'll update through fine-tuning, as follows.

```python
# Trainer with QLoRA configuration
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

``` python
[250/250 05:31, Epoch 1/1]\
Step     Training Loss \
50	     1.564200 \
100	     1.348400 \
150	     1.277900 \
200	     1.323500 \
250	     1.345700

TrainOutput(global_step=250, training_loss=1.3719460601806641, metrics={'train_runtime': 312.3577, 'train_samples_per_second': 3.201, 'train_steps_per_second': 0.8, 'total_flos': 8679674339426304.0, 'train_loss': 1.3719460601806641, 'epoch': 1.0})
```

```python
# Save Model
fine_tuning.model.save_pretrained(new_model_name)
```

#### Checking memory usage during training with QLoRA<a class="anchor" id="Checking_memory_usage_during_training_with_LoRA"></a>

During the training you could check the memory usage by using "rocm-smi" command in a terminal. The command will produce the following output, which tells the usage of memory and GPU.

``` python
============================================= ROCm System Management Interface =============================================
======================================================= Concise Info =======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK     MCLK     Fan  Perf  PwrCap  VRAM%  GPU%  
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)                                                    
============================================================================================================================
0       2     0x74a1,   47056  79.0°C      748.0W    NPS1, SPX, 0        1539Mhz  1300Mhz  0%   auto  750.0W  6%     99%     
============================================================================================================================
=================================================== End of ROCm SMI Log ====================================================
```

To enhance comprehension of QLoRA's impact on training, we will conduct a quantitative analysis comparing QLoRA, LoRA, and full-parameter fine-tuning. This analysis will encompass memory usage, training speed, training loss, and other pertinent metrics, providing a comprehensive evaluation of their respective effects.

## 4. Comparison between QLoRA, LoRA, and full-parameter fine tuning <a class="anchor" id="Comparison"></a>

Building upon our earlier blog titled [Fine-tune Llama with LoRA: Customizing a large language model for question-answering](https://github.com/ROCm/rocm-blogs/tree/release/blogs/artificial-intelligence/llama-lora), which demonstrated the fine-tuning of the Llama 2 model using both LoRA and full-parameter methods, we will now integrate the results obtained with QLoRA. This  aims to provide a comprehensive overview that incorporates insights from all three fine-tuning approaches.

| Metric | Full-parameter | LoRA | **QLoRA**|
|--------------------|-----------|-----------|-----------|
|Trainable parameters| 6,738,415,616 | 4,194,304 | 4,194,304 |
|Mem usage/GB |144 | 82.56 | 11.52|
|Training Speed|12 minutes| 3 minutes |5 minutes|
|Training Loss|1.369|1.379|1.3457|

* Memory usage:
  * In the case of full-parameter fine-tuning, there are **6,738,415,616** trainable parameters, leading to significant memory consumption during the training back propagation stage.
  * In contrast, LoRA and QLoRA introduces only **4,194,304** trainable parameters, accounting for a mere **0.062%** of the total trainable parameters in full-parameter fine-tuning.
  * When monitoring memory usage during training, it becomes evident that fine-tuning with LoRA utilizes only **57%** memory consumed by full-parameter fine-tuning. Impressively, QLoRA goes even further by significantly reducing memory consumption to just **8%**.
  * This presents an opportunity to increase batch size, max sequence length, and train on larger datasets within the constraints of limited hardware resources.

* Training speed:
  * The results demonstrate that full-parameter fine-tuning takes **12 minutes** to complete, while fine-tuning with LoRA and QLoRA takes **3 minutes** and **5 minutes**, respectively.
  * Several factors contribute to this acceleration in training speed:
    * The fewer trainable parameters in LoRA and QLora translates to fewer derivative calculations and less memory needed to store and updates the weights.
    * Fine-tuning with LoRA is more straightforward and faster than QLoRA, as the quantization and de-quantization processes in QLoRA introduce additional overhead during training.

* Accuracy:
  * In both training sessions, a notable reduction in training loss was observed. We achieved a closely aligned training loss for three fine-tuning approaches.
  * In the original work on QLoRA, the author mentioned the performance lost due to the imprecise quantization can be fully recovered through adapter fine-tuning after quantization. In alignment with this insight, our experiments validate and resonate with this observation, emphasizing the effectiveness of adapter fine-tuning in restoring performance after the quantization process.

## 5. Test the fine-tuned model with QLoRA<a class="anchor" id="Test_the_model"></a>

```python
# Reload model in FP16 and merge it with fine-tuned weights
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

Now we can test with the base model (original) and the fine-tuned model.

### Test the base model

```python
inference(base_model, tokenizer, user_input='What do you think is the most important part of building an AI chatbot?')
```

```text
<|start|>user
What do you think is the most important part of building an AI chatbot?<|end|>
<|start|>assistant: The most important thing when creating an AI chatbot is to make sure that it can understand and respond appropriately. This requires developing a sophisticated natural language processing (NLP) system that can accurately interpret user input, identify relevant information in its knowledge base, and generate appropriate responses based on this analysis. Additionally, ensuring that your NLP model has access to high-quality training data sets will be critical for achieving accurate results. Finally, testing with real users should always follow development so as to ensure performance under actual usage conditions before deployment!<|end|>
```

### Test the fine-tuned model

```python
inference(model_enhanced, tokenizer, user_input='What do you think is the most important part of building an AI chatbot?')
```

```bash
<|start|>user
What do you think is the most important part of building an AI chatbot?<|end|>
<|start|>assistant: The most important thing to consider when creating a chatbot is its purpose. What problem does it solve, and what kind of experience are users expecting from their interactions with your bot? Once you have identified these factors, you can start thinking about how best to implement them using natural language processing (NLP) techniques such as sentiment analysis or intent detection. It’s also essential to ensure that your model works well on different devices like mobile phones or tablets so they don't break down too quickly due to limited resources available at times when people need help urgently! Lastly but not least – make sure there aren’t any security vulnerabilities left unattended after deployment by testing everything thoroughly before launching anything publically accessible online without proper authorization first because hackers might exploit those weaknesses if given access inside systems where sensitive data resides within networks owned privately by individuals who trust only specific parties responsible for protecting confidential information shared between friends across social media platforms

```

You can now observe the outputs of the two models based on the given query. As anticipated, the two outputs exhibit slight differences due to the fine-tuning process altering the model weights.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
