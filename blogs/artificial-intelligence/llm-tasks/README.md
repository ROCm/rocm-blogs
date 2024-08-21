---
blogpost: true
date: 21 August 2024
author: Eliot Li
tags: AI/ML, LLM
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Performing natural language processing tasks with LLMs on ROCm running on AMD GPUs"
    "keywords": "GenAI, Question Answering, Summarization, Information Retrieval, Sentiment Analysis, AMD, GPU, MI300, MI250, LLM, C4AI Command-R, Qwen, OPT, MPT, DistilBERT, Longformer, DistilRoBERTa, FinBERT, BART, Pegasus, Contriever, Phi-3"
    "property=og:locale": "en_US"
---

# Performing natural language processing tasks with LLMs on ROCm running on AMD GPUs

In this blog you will learn how to use ROCm, running on AMD’s Instinct GPUs, for a range of popular and useful natural language processing (NLP) tasks, using different large language models (LLMs). The blog includes a simple to follow hands-on guide that shows you how to implement LLMs for core NLP applications ranging from text generation and sentiment analysis to extractive question answering (QA), and solving a math problem.

General purpose LLMs such as GPT and Llama can perform many different tasks with reasonable performance. However, certain tasks require either fine tuning or a different model architecture to support the use cases. The ML community developed a number of models that are designed or fine-tuned for specific tasks to complement the general-purpose models. In this blog we touch on both general-purpose and special-purpose LLMs, and show you how to use them on ROCm running on AMD GPUs for several common tasks.

## Introduction

Ever since the launch of ChatGPT by OpenAI in late 2022, millions of people have experienced the power of generative AI. While general purpose LLMs can deliver reasonably good performance on many tasks, such as answering quick questions and problem solving, they often fall short when the prompt is highly domain specific or requires certain skills that they are not specifically trained for. Prompt engineering can help mitigate the problem by providing specific instructions or examples on how the LLM should respond in the prompt. However, the skill required to create the prompt and the limit on context length often prevents LLMs from reaching their full potential.

To address these problems, general-purpose LLMs have gotten bigger (with some models like Grok-1 reaching a few hundred billion parameters) and more powerful. At the same time, the ML community has developed many special-purpose models that can perform certain tasks really well, at the expense of lower performance on other tasks.

HuggingFace lists about a dozen different NLP tasks that LLMs can perform, including text generation, question answering, translation, and many others. This blog demonstrates how to use a number of general-purpose and special-purpose LLMs on ROCm running on AMD GPUs for these NLP tasks:

* Text generation
* Extractive question answering
* Solving a math problem
* Sentiment analysis
* Summarization
* Information retrieval

## Prerequisites

To run this blog, you will need the following:

* **AMD GPUs**: [AMD Instinct GPU](https://www.amd.com/en/products/accelerators/instinct.html).
* **Linux**: see the [supported Linux distributions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems).
* **ROCm 6.0+**: see the [installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html).
* Some of the models used in this blog are gated. You must request access on Hugging Face and use your Hugging Face token to download the model weights. You must also agree to share your contact information on Hugging Face.

## Getting started

First check if you can detect the GPUs on the server.

```bash
rocm-smi
```

```text
========================================= ROCm System Management Interface =========================================
=================================================== Concise Info ===================================================
Device  [Model : Revision]    Temp        Power     Partitions      SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%
        Name (20 chars)       (Junction)  (Socket)  (Mem, Compute)
====================================================================================================================
0       [0x74a1 : 0x00]       35.0°C      140.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
1       [0x74a1 : 0x00]       37.0°C      138.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
2       [0x74a1 : 0x00]       40.0°C      141.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
3       [0x74a1 : 0x00]       36.0°C      139.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
4       [0x74a1 : 0x00]       38.0°C      143.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
5       [0x74a1 : 0x00]       35.0°C      139.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
6       [0x74a1 : 0x00]       39.0°C      142.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
7       [0x74a1 : 0x00]       37.0°C      137.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W    0%   0%
        AMD Instinct MI300X
====================================================================================================================
=============================================== End of ROCm SMI Log ================================================
```

All 8 GPUs on the MI300X system are available. Start the docker container with ROCm 6.0 and PyTorch support and install the required packages.

```bash
docker run -it --ipc=host --network=host --device=/dev/kfd  --device=/dev/dri -v $HOME/dockerx:/dockerx --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name=llm-tasks rocm/pytorch:rocm6.1.3_ubuntu22.04_py3.10_pytorch_release-2.1.2 /bin/bash
```

```bash
pip install --upgrade pip
pip install transformers accelerate einops
```

The following sections demonstrate how to run the LLMs on ROCm to perform various NLP tasks.

## Text generation

Text generation is probably the first task that most people associate with an LLM. Given a text prompt, the LLM generates a text response that addresses the prompt. There are several ROCm blogs discussing how popular models perform this task, including [Llama2](https://rocm.blogs.amd.com/artificial-intelligence/llm-inference-optimize/README.html), [GPT-3](https://rocm.blogs.amd.com/artificial-intelligence/megatron-deepspeed-pretrain/README.html), [OLMo](https://rocm.blogs.amd.com/artificial-intelligence/ai2-olmo/README.html), and [Mixtral](https://rocm.blogs.amd.com/artificial-intelligence/moe/README.html). This blog covers four more high-end models.

### C4AI Command-R

After publishing the ground-breaking paper "Attention is all you need" with his team at Google Brain, Aidan Gomez left Google and co-founded [Cohere](https://cohere.com/). Cohere has developed several state-of-the-art LLMs, including the [`C4AI Command-R`](https://huggingface.co/collections/CohereForAI/c4ai-command-r-6604150f4c8ac7bea92963ed) and [`C4AI Command-R Plus`](https://huggingface.co/collections/CohereForAI/c4ai-command-r-plus-660ec4c34f7a69c50ce7f7b9) model families, and made them available on [HuggingFace](https://huggingface.co/CohereForAI/c4ai-command-r-v01).

This test involves a medium-sized model [`c4ai-command-r-v01`](https://huggingface.co/CohereForAI/c4ai-command-r-v01) with 35B parameters for text generation on ROCm.

```{note}
The `c4ai-command-r-v01` model is gated. This means you must request access on HuggingFace before you can use it. Use your HuggingFace token to download the model by replacing the variable `token` with your token in the code block below.
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

token = "your HuggingFace user access token here"
model_name = "CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, token=token)

prompt = "Write a poem about artificial intelligence in Shakespeare style."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt")

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

Here is the generated response to the prompt:

```text
In days of yore, when mortals' minds did roam,
A wondrous birth, a thought-borne gem,
From human intellect, a progeny did bloom,
AI, a brain-child, bright and new.

From bits and bytes, a creature formed, so keen,
To serve and aid, a helpful hand,
With algorithms, it thinks, and learns, and sees,
A clever clone, a mental clone.

It parses speech, solves problems hard,
With speed beyond compare,
It understands, assists, and guides,
A thoughtful, digital friend.
```

Here is another example showing how to use C4AI Command-R for text generation, in this case to answer a question:

```python
prompt = "Which countries are the biggest rare earth metal producer?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt")

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

C4AI Command-R is able to give a detailed answer to the question.

```text
As of 2022, the top three countries that are the biggest producers of rare earth metals are:
1. China: China is the world's largest producer of rare earth metals, accounting for over 58% of the global production. China's production share is even larger when it comes to the more valuable and technologically important rare earth oxides. The country has a strong hold on the supply chain, from mining to processing and manufacturing of rare earth metals and products.

2. Australia: Australia is the second-largest producer of rare earth metals. It has significant reserves and several operational mines producing rare earth elements. Lyn
```

### Qwen

While models developed by US and European companies, such as Llama, GPT, and Mistral, drew the most attention from the media, there are some notable contenders from Chinese companies as well. The best-known one is the Qwen series from Alibaba Cloud. The Qwen models are general-purpose Transformer-based LLM AI assistants trained on a diverse selection of web texts, books, code samples, and other materials.

The latest releases in the Qwen series are the [Qwen2 family models](https://qwenlm.github.io/blog/qwen2/). All the models in the Qwen2 family have adopted Group Query Attention (GQA) to benefit from lower latency and less memory usage in model inference. In terms of context length, the Qwen2-7B and Qwen2-72B models can support up to 128k tokens. The first Qwen series models were trained on English and Chinese texts only. Qwen2 has 27 additional languages from different regions of the world in its training data, resulting in much better performance in multilingual tasks.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model_name = "Qwen/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

After you have the Qwen2 model and tokenizer ready, ask it a question.

```python
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

Here is the response from Qwen2:

```text
A Large Language Model (LLM) is a type of artificial intelligence model that has been trained on vast amounts of text data to understand and generate human-like language. These models are capable of performing various natural language processing tasks such as text translation, summarization, question answering, text generation, etc. 

LLMs typically use deep learning techniques, often involving transformer architectures, which allow the model to understand context and relationships between words in sentences. This makes them very powerful tools for generating coherent and contextually relevant responses, even when given complex or nuanced prompts.

One of the most famous examples of an LLM is the GPT series created by OpenAI, including GPT-2 and GPT-3. However, it's worth noting that these models can also be used for potentially harmful purposes if not handled responsibly due to their ability to create realistic but false information. Therefore, they need to be used ethically and with appropriate safeguards in place.
```

### OPT

OPT (Open Pre-trained Transformer Language Models) from Meta was introduced in the paper [Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/2205.01068). It is a suite of pre-trained transformer models ranging from 125M to 175B parameters. The goals of OPT are to provide the research community with access to a set of highly capable pre-trained LLMs that it can use for further development and to reproduce results produced by the community.

This example tests the 125M parameter version of OPT ['opt-125m'](https://huggingface.co/facebook/opt-125m), which is one of the most popular versions due to its small size, on ROCm. It leverages the `text-generation` pipeline from HuggingFace to use the model to generate text from a prompt. It also sets `do_sample=True` to enable top-k sampling, which makes the generated text more interesting.

```python
from transformers import pipeline, set_seed

set_seed(32)
text_generator = pipeline('text-generation', model="facebook/opt-125m", max_new_tokens=256, do_sample=True, device='cuda')

output = text_generator("Provide a few suggestions for family activities this weekend.")
print(output[0]['generated_text'])
```

```text
Provide a few suggestions for family activities this weekend.

The summer schedule is a great opportunity to spend some time enjoying the summer with those who might otherwise be working from home or working from a remote location. You will discover new and interesting places to eat out and spend some time together. There are things you’ll do in different weathers (in particular you’ll learn what it’s like to enjoy a hot summer summer outside. For example you may see rainbows, waves crashing against a cliff, an iceberg exploding out of the sky, and a meteor shower rolling through the sky.

I’ve tried to share some ideas on how to spend all summer on our own rather than with a larger family. In addition to family activities, here are several ways to stay warm for the holidays during a time of national emergency.

...
```

OPT tends to ramble rather than providing a concise and relevant answer. There are quite a few fine-tuned versions of OPT available on HuggingFace. You are encouraged to explore those models or fine tune your own model.

### MPT

Generating instructions on how to accomplish a task, such as cooking a recipe, is another common use case for LLMs. While prompt engineering can be used with a general-purpose LLM to guide the model to generate the instructions, the prompt must be carefully curated to achieve the desired outputs.

The MPT collection from [Mosaic Research](https://www.databricks.com/research/mosaic) (now part of Databricks) is a series of decoder-style transformer models that includes two base models, the MPT-7B and MPT-30B. The [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct) model in this collection is a LLM that is fine-tuned from the MPT-7B model using a dataset derived from the Databricks Dolly-15k and Anthropic Helpful and Harmless (HH-RLHF) datasets. The model is supported by the `text-generation` pipeline from HuggingFace and is easy to use on ROCm.

```python
import torch
import transformers
from transformers import pipeline

model_name = 'mosaicml/mpt-7b-instruct'

config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.max_seq_len = 4096

model = transformers.AutoModelForCausalLM.from_pretrained(
  model_name,
  config=config,
  trust_remote_code=True
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')

prompt = "Here is the instruction to change the oil filter in your car:\n"
with torch.autocast('cuda', dtype=torch.bfloat16):
    instruction = text_generator(prompt,
                                 max_new_tokens=512,
                                 do_sample=True,
                                 use_cache=True)

print(instruction[0]['generated_text'])
```

Here is the generated text for the prompt "Here is the instruction to change the oil filter in your car:" by MPT-7B-Instruct:

```text
Here is the instruction to change the oil filter in your car:
1. Open the hood. 2. Find the oil filter. 3. Look to the right underneath the cap to find the oil filter. 4. Screw the oil filter cap off from the bottom.5. Pull oil filter out from the bottom of the engine.
What is the oil filter? The oil filter is a part that catches particles from your engine oil as it travels through your engine. It traps most of the particles and keeps them from passing straight into your engine. This keeps your engine from getting damaged because of those particles. How many oil filters are there?
There is one oil filter for the entire vehicle. However different types of vehicles have different requirements that can change the oil more often than others.
When should you change the oil filter? It is recommended to change oil filters between 30,000 to 60,000 miles. However some engine types are harder on filters and may require changing every 15,000 miles instead of 30,000.
What can you get at your local automotive store before changing your oil filter: 5-10 quarts 5-10 oil filter, a drain pan, and oil filter wrench.
Step 1. Drain the oil. 2. Check the oil filter to be sure that it is still in good shape. 3. Install the new oil filter. 4. Fill the reservoir with the proper amount of oil.
```

## Extractive question answering

When people think about how an LLM answers questions, they usually think of a chatbot that is like an oracle and can answer any question they have in mind, as illustrated in the text generation examples earlier. On the other hand, there are many LLMs trained specifically to perform what is known as "extractive question answering". The idea is that the input of the LLM consists of the question as well as the context for the answer. Furthermore, the model's answer to the question must incorporate a portion of the context. The main use cases for extractive question answering involve scenarios where the user knows the answer is within some known context, such as identifying the preferences of a particular customer from their purchase history. Extracting answers from the context alone can limit the likelihood that an LLM hallucinates and makes up false answers, even if the context is in its training data.

Here's a test of two popular LLMs that have been fine-tuned to perform extractive question answering.

### DistilBERT

One of the challenges in deploying LLMs is their large size results in high computation power requirements, latency, and power consumption. An active area of research is to train smaller models using the outputs of larger trained models and retain most of the performance, a process known as knowledge distillation. A notable example of such models is the DistilBERT model, which was proposed in the blog post [Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5). DistilBERT is a small, fast, cheap, and light Transformer model trained by distilling the BERT base. This means it was pre-trained with generated inputs and labels from the BERT base model only. It is 40% smaller than the `bert-base-uncased` model in terms of the number of parameters and runs 60% faster, while preserving over 95% of BERT's performance as measured on the GLUE language understanding benchmark.

This example tests a version of the DistilBERT model ['distilbert-base-cased-distilled-squad'](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad), which is a fine-tuned checkpoint of [`DistilBERT-base-cased`](https://huggingface.co/distilbert-base-cased), using knowledge distillation on the [SQuAD v1.1 dataset](https://huggingface.co/datasets/rajpurkar/squad). The task is to find the birthplace of Marie Curie's doctoral advisor from a context that include four facts, only one of which has the answer to the question.

```python
from transformers import pipeline
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

context = """Gabriel Lippmann, who supervised Marie Curie's doctoral research, was born in Bonnevoie, Luxembourg. 
        Marie Curie was born in Warsaw, Poland in what was then the Kingdom of Poland, part of the Russian Empire.
        Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867. 
        Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."""
question = "Where was Marie Curie's doctoral advisor Gabriel Lippmann born?"

result = question_answerer(question=question, context=context)
print(f"Answer: '{result['answer']}'\n Score: {round(result['score'], 4)},\n start token: {result['start']}, end token: {result['end']}")
```

DistilBERT is able to find the right answer with a high level of confidence.

```text
Answer: 'Bonnevoie, Luxembourg'
 Score: 0.9714,
 start token: 78, end token: 99
```

### Longformer

One of the main limitations of transformer models is the self-attention operation scales quadratically with the input sequence length, making it hard to scale them to handle long input sequences. The Longformer model from Allen AI, proposed in [Longformer: The Long-Document Transformer by Iz Beltagy, Matthew E. Peters, Arman Cohan](https://arxiv.org/pdf/2004.05150) attempts to mitigate this problem by replacing the self-attention operation with a local windowed attention combined with a task-motivated global attention.

Allen AI has trained [a number of models](https://huggingface.co/docs/transformers/model_doc/longformer) based on the Longformer architecture for various tasks. This example demonstrates the ability of the [LongformerForQuestionAnswering model](https://huggingface.co/docs/transformers/v4.41.3/en/model_doc/longformer#transformers.LongformerForQuestionAnswering) to extract the answer to a question from a context.

This model takes the context and question as input, and outputs the span start logits and span end logits of each token in the encoded input. The best answer to the question can then be extracted based on the span logits.

```python
from transformers import AutoTokenizer, LongformerForQuestionAnswering
import torch

# setup the tokenizer and the model
model_name = "allenai/longformer-large-4096-finetuned-triviaqa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LongformerForQuestionAnswering.from_pretrained(model_name)

# context and question
context = """Gabriel Lippmann, who supervised Marie Curie's doctoral research, was born in Bonnevoie, Luxembourg. 
        Marie Curie was born in Warsaw, Poland in what was then the Kingdom of Poland, part of the Russian Empire.
        Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867. 
        Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."""
question = "Where was Marie Curie's doctoral advisor Gabriel Lippmann born?"

# encode the question and the context
encoded_input = tokenizer(question, context, return_tensors="pt")
input_ids = encoded_input["input_ids"]

# Generate the output masks
outputs = model(input_ids)
# find the beginning and end index of the answer in the encoded input
start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits)

# Convert the input ids to tokens
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

# extract the answer tokens and decode it
answer_tokens = all_tokens[start_idx : end_idx + 1]
answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

print(answer)
```

Longformer gives the city "Bonnevoie" as the answer, which is correct.

```text
 Bonnevoie
 ```

## Solving a math problem

The ability to understand a problem and provide an answer through logical reasoning has always been one of the main goals of artificial intelligence. A prime example of such a use case is to solve mathematical problems. Even general-purpose LLMs like GPT4 have demonstrated remarkable performance in simple math problems. This section explores the use of a fine-tuned version of the Phi-3 model on AMD GPUs to solve math problems.

### Phi-3

The [Phi-3 collection](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) is the next generation of the popular [Phi-2](https://rocm.blogs.amd.com/artificial-intelligence/Phi2/README.html) model from Microsoft. This example uses the fine-tuned version [`Phi-3-Mini-4K-Instruct'](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), which is a 3.8B parameter model trained with carefully curated high-quality educational data and code, as well as synthetic data that resembles textbook material in subjects like math, coding, and common-sense reasoning.

First, setup the Phi-3 model using the `text-generation` pipeline.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model_name = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 1024,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
```

Then ask Phi-3 to find the Taylor series of the sum of two simple functions `sin(x) + ln(x)`.

```python
messages = [
    {"role": "user", "content": "What is the Taylor series expansion of sin(x) + ln(x)? about a point x=a"},
]

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
```

```text
 The Taylor series expansion of a function f(x) about a point x=a is given by:

f(x) = f(a) + f'(a)(x-a) + f''(a)(x-a)^2/2! + f'''(a)(x-a)^3/3! +...

For the function sin(x) + ln(x), we need to find the derivatives and evaluate them at x=a.

First, let's find the derivatives of sin(x) and ln(x):

1. sin(x):
   f(x) = sin(x)
   f'(x) = cos(x)
   f''(x) = -sin(x)
   f'''(x) = -cos(x)
  ...

2. ln(x):
   f(x) = ln(x)
   f'(x) = 1/x
   f''(x) = -1/x^2
   f'''(x) = 2/x^3
  ...

Now, let's evaluate these derivatives at x=a:

1. sin(a):
   f(a) = sin(a)
   f'(a) = cos(a)
   f''(a) = -sin(a)
   f'''(a) = -cos(a)
  ...

2. ln(a):
   f(a) = ln(a)
   f'(a) = 1/a
   f''(a) = -1/a^2
   f'''(a) = 2/a^3
  ...

Now, we can write the Taylor series expansion of sin(x) + ln(x) about x=a:

sin(x) + ln(x) = (sin(a) + ln(a)) + (cos(a)(x-a) + (1/a)(x-a)) + (-sin(a)(x-a)^2/2! + (-1/a^2)(x-a)^2/2!) + (-cos(a)(x-a)^3/3! + (2/a^3)(x-a)^3/3!) +...

This is the Taylor series expansion of sin(x) + ln(x) about x=a.
```

Not bad. Next ask Phi-3 to do the same for a slightly more complicated function, `sin(x) + 1/cos(x)`.

```python
messages = [
    {"role": "user", "content": "What is the Taylor series expansion of sin(x) + 1/cos(x) about a point x=a?"},
]

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
```

```text
 The Taylor series expansion of a function f(x) about a point x=a is given by:

f(x) = f(a) + f'(a)(x-a) + f''(a)(x-a)^2/2! + f'''(a)(x-a)^3/3! +...

First, let's find the Taylor series expansion of sin(x) and 1/cos(x) separately about x=a.

For sin(x), the derivatives are:
sin'(x) = cos(x)
sin''(x) = -sin(x)
sin'''(x) = -cos(x)
sin''''(x) = sin(x)
...

The Taylor series expansion of sin(x) about x=a is:
sin(x) = sin(a) + cos(a)(x-a) - sin(a)(x-a)^2/2! - cos(a)(x-a)^3/3! + sin(a)(x-a)^4/4! +...

For 1/cos(x), the derivatives are:
(1/cos(x))' = sin(x)/cos^2(x)
(1/cos(x))'' = (cos(x) + sin^2(x))/cos^3(x)
(1/cos(x))''' = (-2cos(x)sin(x) + 3sin^2(x))/cos^4(x)
...

The Taylor series expansion of 1/cos(x) about x=a is:
1/cos(x) = 1/cos(a) + (sin(a)/cos^2(a))(x-a) + (cos(a)(sin^2(a) - 1)/cos^3(a))(x-a)^2/2! + (2cos(a)(sin^3(a) - 3sin(a))/cos^4(a))(x-a)^3/3! +...

Now, we can find the Taylor series expansion of sin(x) + 1/cos(x) by adding the two series:

sin(x) + 1/cos(x) = (sin(a) + 1/cos(a)) + (cos(a) + sin(a)/cos^2(a))(x-a) - (sin(a)(x-a)^2/2! + 1/cos^3(a)(x-a)^2/2!) +...

This is the Taylor series expansion of sin(x) + 1/cos(x) about x=a.
```

Although Phi-3 is able to follow the standard procedure of finding the derivatives of each term and then summing the Taylor series of each term, it fails to correctly find the higher derivatives of `1/cos(x)` and add the correct derivatives together in the last step. For instance, the second derivative of `1/cos(x)` should be `(1 + sin^2(x))/cos^3(x)` rather than `(cos(x) + sin^2(x))/cos^3(x)`. This shows the limit of LLMs, which are inherently token predictors rather than reasoning machines, in problem solving.

## Sentiment analysis

Sentiment analysis has been an active research topic among the ML community for many years due to its wide number of applications. Transformer-based LLM models open new opportunities to improve the performance of sentiment analysis models by taking into account the context in a large corpus of relevant text data. In particular, there is strong interest in using LLMs to understand the sentiment of financial news, which can be extremely valuable in making investment decisions. The example below tests two notable models that are fine-tuned for sentiment analysis. In both cases, it leverages the `sentiment-analysis` pipeline in HuggingFace.

### DistilRoBERTa

The [DistilRoberta-financial-sentiment model](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis) is a lightweight, distilled version of the RoBERTa-base model with only 82M parameters. Due to its smaller size, this model runs twice as fast as the RoBERTa-base model. The model was trained on a polar sentiment dataset of sentences from financial news, annotated by between five to eight human annotators.

Set up the model and use it to determine the sentiment of four financial news communications.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, device_map="cuda")
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentences = ["there is a shortage of capital, and we need extra financing",  
             "growth is strong and we have plenty of liquidity", 
             "there are doubts about our finances", 
             "profits are flat"]

for sentence in sentences:
    result = sentiment_analyzer(sentence)
    print(f"Input sentence: \"{sentence}\"")
    print(f"Sentiment: '{result[0]['label']}'\n Score: {round(result[0]['score'], 4)}\n")
```

```text
Input sentence: "there is a shortage of capital, and we need extra financing"
Sentiment: 'negative'
 Score: 0.666

Input sentence: "growth is strong and we have plenty of liquidity"
Sentiment: 'positive'
 Score: 0.9996

Input sentence: "there are doubts about our finances"
Sentiment: 'neutral'
 Score: 0.6857

Input sentence: "profits are flat"
Sentiment: 'neutral'
 Score: 0.9999
```

The sentiments determined by the model seem reasonable. One can argue the third statement "there are doubts about our finances" should be considered negative. On the other hand, the "neutral" rating from the model comes with a relatively low confidence score of 0.6857, which shows the rating could tip to "negative" with a slightly different threshold.

### FinBERT

FinBERT was proposed in the paper [FinBERT: A Pretrained Language Model for Financial Communications](https://arxiv.org/pdf/2006.08097) by researchers at Hong Kong University of Science and Technology. It's a BERT-based model pre-trained on financial communication text. The training data includes three financial communication corpus with a total size of 4.9B tokens.

The [finbert-tone model](https://huggingface.co/yiyanghkust/finbert-tone) used here is a FinBERT model fine-tuned on 10,000 manually annotated (positive, negative, neutral) sentences from analyst reports.

This example uses FinBERT to determine the sentiment for the same financial communications analyzed by DistilRoBERTa above.

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

model_name = "yiyanghkust/finbert-tone"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3, device_map="cuda")
tokenizer = BertTokenizer.from_pretrained(model_name)

sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentences = ["there is a shortage of capital, and we need extra financing",  
             "growth is strong and we have plenty of liquidity", 
             "there are doubts about our finances", 
             "profits are flat"]

for sentence in sentences:
    result = sentiment_analyzer(sentence)
    print(f"Input sentence: \"{sentence}\"")
    print(f"Sentiment: '{result[0]['label']}'\n Score: {round(result[0]['score'], 4)}\n")
```

```text
Input sentence: "there is a shortage of capital, and we need extra financing"
Sentiment: 'Negative'
 Score: 0.9966

Input sentence: "growth is strong and we have plenty of liquidity"
Sentiment: 'Positive'
 Score: 1.0

Input sentence: "there are doubts about our finances"
Sentiment: 'Negative'
 Score: 1.0

Input sentence: "profits are flat"
Sentiment: 'Neutral'
 Score: 0.9889
```

The only difference between the outputs of DistilRoBERTa and FinBERT models is the third case where FinBERT considered it negative rather than neutral.

## Summarization

Early approaches in text summarization focused on extracting keywords or key phrases from the text to be summarized and assembling them into a summary using human-defined rules. LLM changes summarization due to its ability to capture relationships between tokens in a long sequence of text. There are many notable LLMs that are trained specifically for such tasks. This section demonstrates two of them.

### BART

BART, from Facebook, was introduced in the paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461). BART adopts a transformer-based neural network architecture, with a denoising bidirectional autoencoder and a sequence-to-sequence GPT-like autoregressive decoder model. The pre-training of BART involves two steps. It first corrupts the training text data with arbitrary noise. Then it trains the model to reconstruct the original text from the corrupted text. This method provides enormous flexibility in generating the training data, including changing the text length and word order.

The BART base model can be used for text infilling but is not suitable for most tasks of interest. BART really shines when it's fine-tuned for a particular task, such as summarization. This example uses a [version of BART](https://huggingface.co/facebook/bart-large-cnn) that has been fine-tuned with CNN Daily Mail, a collection of document-summary pairs data, for summarization tasks.

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cuda")

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""

print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False)[0]['summary_text'])
```

```text
Liana Barrientos, 39, is charged with two counts of "offering a false instrument for filing in the first degree" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is believed to still be married to four men.
```

### Pegasus

Another LLM that is well known for summarization is Pegasus from Google. It was introduced in the paper [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/pdf/1912.08777). Pegasus masks key sentences from the training documents and trains the model to generate the gap sentences. According to the authors, this approach is particularly well-suited for abstract summarization because it forces the model to understand the context of the entire document.

This example uses the Pegasus model to summarize the same text `ARTICLE` that the BART model handled earlier.

```python
from transformers import AutoTokenizer, PegasusForConditionalGeneration

model_name = "google/pegasus-xsum"
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
 
inputs = tokenizer(ARTICLE, max_length=1024, return_tensors="pt")
summary_ids = model.generate(inputs["input_ids"])

print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
```

The summary generated is shorter but still retains the key points of the text.

```text
A New York woman who has been married 10 times has been charged with marriage fraud.
```

## Information Retrieval

The birth of generative AI could spell the death of information retrieval because many people don't care about the original source if the model gives them what they are looking for. However, there are still use cases, such as fact checking and legal approvals, where specific documents from the corpus are required. The most prominent model that leverages recent advances in machine learning is the [Contriever model from Meta](https://huggingface.co/facebook/contriever).

### Contriever

There are many attempts to train deep neural network models using supervised learning for information retrieval applications. However, these approaches suffer from the lack of training samples in most real-life applications. They require lots of human-generated labels indicating which document is most relevant to each query in the training dataset. The main idea behind Contriever is to train the model without labeled data by using an auxiliary task that approximates retrieval. Specifically, for a given document in the training corpus, it generates a synthetic query where the document is the perfect answer to the query. These pairs are then used to train the model. In addition, contrastive learning enhances the discriminative power of the model between relevant and irrelevant results. Details of the approach adopted by Contriever can be found in the paper [Unsupervised Dense Information Retrieval with Contrastive Learning](https://arxiv.org/pdf/2112.09118).

You can use the same example from the extractive question answering section to illustrate how Contriever can retrieve the most relevant document from a corpus. First, use the model's output to score the documents.

```python
import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "facebook/contriever"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

query = ["Where was Marie Curie born?"]

docs = [
    "Gabriel Lippmann, who supervised Marie Curie's doctoral research, was born in Bonnevoie, Luxembourg.",
    "Marie Curie was born in Warsaw, in what was then the Kingdom of Poland, part of the Russian Empire",
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

corpus = query + docs

# Apply tokenizer
inputs = tokenizer(corpus, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
outputs = model(**inputs)

# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings
embeddings = mean_pooling(outputs[0], inputs['attention_mask'])

score = [0]*len(docs)
for i in range(len(docs)):
    score[i] = (embeddings[0] @ embeddings[i+1]).item()

print(score) 
```

```text
[0.9390654563903809, 1.1304867267608643, 1.0473244190216064, 1.0094892978668213]
```

Then print the query and the best matching document to see if Contriever got it right.

```python
print("Most relevant document to the query \"", query[0], "\" is")
docs[score.index(max(score))]
```

```text
Most relevant document to the query " Where was Marie Curie born? " is
'Marie Curie was born in Warsaw, in what was then the Kingdom of Poland, part of the Russian Empire'
```

Contriever was able to pick out the correct document, despite the fact that the other three documents look very similar.

## Summary

In this blog you learned how to implement several popular LLMs using ROCm running on AMD GPUs, to easily perform various NLP tasks such as text generation, summarization, and solving math problems. If you are interested in improving the performance of these models, check out the ROCm blogs on fine-tuning [Llama2](https://rocm.blogs.amd.com/artificial-intelligence/llama2-lora/README.html) and [Starcoder](https://rocm.blogs.amd.com/artificial-intelligence/starcoder-fine-tune/README.html).

## Disclaimer

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
