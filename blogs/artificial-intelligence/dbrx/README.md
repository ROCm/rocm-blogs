---
blogpost: true
date: 11 July 2024
author: Phillip Dang
tags: PyTorch, AI/ML
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "DBRX Instruct on AMD GPUs"
    "keywords": "PyTorch, DBRX, LLM, MOE, AMD, GPU, MI210"
    "property=og:locale": "en_US"
---

# DBRX Instruct on AMD GPUs

In this blog, we showcase DBRX Instruct, a mixture-of-experts large language model developed by Databricks, on a ROCm-capable system with AMD GPUs.

## About DBRX Instruct

DBRX is a transformer-based decoder-only large language model with 132 billion parameters, utilizing a fine-grained mixture-of-experts (MoE) architecture. It was pre-trained on 12 trillion tokens of text and code data, employing 16 experts with 4 chosen. This means that the input token is routed to 4 expert networks out of the 16 by a gating network based on the token's characteristics and the experts' specializations. At any given time, only 32 billion parameters are active on any input. DBRX uses several advanced optimization techniques including rotary position encodings (RoPE), gated linear units (GLU), and grouped query attention (GQA) for superior performance.

Along with adjusting the parameter count, curriculum learning was employed during pre-training. This method altered the data composition throughout training, resulting in significant enhancements to the model's overall quality ([source](https://huggingface.co/databricks/dbrx-base)). Curriculum learning involves gradually adjusting the difficulty or complexity of the training data fed to a machine learning model during training. Initially, simpler or easier examples are presented, followed by more challenging ones as the model learns ([source](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)).

## Prerequisites

* [ROCm 5.7.0+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
* [PyTorch 2.2.1+](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)
* [Supported Linux OS](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems)
* [Supported AMD GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

Verify that your system correctly recognizes the GPUs and has the necessary ROCm libraries installed. Given that DBRX Instruct has over 130 billion parameters, we use six GPUs in this blog.

``` python
! rocm-smi --showproductname
```

```cpp
========================= ROCm System Management Interface =========================
=================================== Product Info ===================================
GPU[0]    : Card series:    Instinct MI210
GPU[0]    : Card model:     0x0c34
GPU[0]    : Card vendor:    Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0]    : Card SKU:       D67301GPU 
GPU[1]    : Card series:    Instinct MI210Card series:    Instinct MI210
GPU[1]    : Card model:     0x0c34
GPU[1]    : Card vendor:    Advanced Micro Devices, Inc. [AMD/ATI]
GPU[1]    : Card SKU:       D67301V
GPU[2]    : Card series:    Instinct MI210
GPU[2]    : Card model:     0x0c34
GPU[2]    : Card vendor:    Advanced Micro Devices, Inc. [AMD/ATI]
GPU[2]    : Card SKU:       D67301V
GPU[3]    : Card series:    Instinct MI210
GPU[3]    : Card model:     0x0c34
GPU[3]    : Card vendor:    Advanced Micro Devices, Inc. [AMD/ATI]
GPU[3]    : Card SKU:       D67301V
GPU[4]    : Card series:    Instinct MI210
GPU[4]    : Card model:     0x0c34
GPU[4]    : Card vendor:    Advanced Micro Devices, Inc. [AMD/ATI]
GPU[4]    : Card SKU:       D67301V
GPU[5]    : Card series:    Instinct MI210
GPU[5]    : Card model:     0x0c34
GPU[5]    : Card vendor:    Advanced Micro Devices, Inc. [AMD/ATI]
GPU[5]    : Card SKU:       D67301V
====================================================================================
=============================== End of ROCm SMI Log ================================
```

Check that you have a compatible version of ROCm installed.

```python
!apt show rocm-libs -a
```

```text
Package: rocm-libs
Version: 5.7.0.50700-63~22.04
Priority: optional
Section: devel
Maintainer: ROCm Libs Support <rocm-libs.support@amd.com>
Installed-Size: 13.3 kB
Depends: hipblas (= 1.1.0.50700-63~22.04), hipblaslt (= 0.3.0.50700-63~22.04), hipfft (= 1.0.12.50700-63~22.04), hipsolver (= 1.8.1.50700-63~22.04), hipsparse (= 2.3.8.50700-63~22.04), miopen-hip (= 2.20.0.50700-63~22.04), rccl (= 2.17.1.50700-63~22.04), rocalution (= 2.1.11.50700-63~22.04), rocblas (= 3.1.0.50700-63~22.04), rocfft (= 1.0.23.50700-63~22.04), rocrand (= 2.10.17.50700-63~22.04), rocsolver (= 3.23.0.50700-63~22.04), rocsparse (= 2.5.4.50700-63~22.04), rocm-core (= 5.7.0.50700-63~22.04), hipblas-dev (= 1.1.0.50700-63~22.04), hipblaslt-dev (= 0.3.0.50700-63~22.04), hipcub-dev (= 2.13.1.50700-63~22.04), hipfft-dev (= 1.0.12.50700-63~22.04), hipsolver-dev (= 1.8.1.50700-63~22.04), hipsparse-dev (= 2.3.8.50700-63~22.04), miopen-hip-dev (= 2.20.0.50700-63~22.04), rccl-dev (= 2.17.1.50700-63~22.04), rocalution-dev (= 2.1.11.50700-63~22.04), rocblas-dev (= 3.1.0.50700-63~22.04), rocfft-dev (= 1.0.23.50700-63~22.04), rocprim-dev (= 2.13.1.50700-63~22.04), rocrand-dev (= 2.10.17.50700-63~22.04), rocsolver-dev (= 3.23.0.50700-63~22.04), rocsparse-dev (= 2.5.4.50700-63~22.04), rocthrust-dev (= 2.18.0.50700-63~22.04), rocwmma-dev (= 1.2.0.50700-63~22.04)
Homepage: https://github.com/RadeonOpenCompute/ROCm
Download-Size: 1012 B
APT-Manual-Installed: yes
APT-Sources: http://repo.radeon.com/rocm/apt/5.7 jammy/main amd64 Packages
Description: Radeon Open Compute (ROCm) Runtime software stack
```

Make sure PyTorch also recognizes the GPUs:

``` python
import torch
print(f"number of GPUs: {torch.cuda.device_count()}")
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

``` text
number of GPUs: 6
['AMD Instinct MI210', 'AMD Instinct MI210', 'AMD Instinct MI210', 'AMD Instinct MI210', 'AMD Instinct MI210', 'AMD Instinct MI210']
```

## Libraries

Before you begin, make sure you have all the necessary libraries installed:

``` python
! pip install -q "transformers>=4.39.2" "tiktoken>=0.6.0"
! pip install accelerate
```

To speed up the download time, run the following commands:

```python
! pip install hf_transfer
! export HF_HUB_ENABLE_HF_TRANSFER=1
```

Additionally, we found that we need to install the latest version of PyTorch to avoid an [error with `nn.LayerNorm` initialization](https://huggingface.co/databricks/dbrx-instruct/discussions/46).

```python
! pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/rocm5.7
```

Next, import the required modules from the Hugging Face `transformers` library.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```

## Loading the model  

Let's load the model and its tokenizer. We'll use `dbrx-instruct` which has been fine-tuned and trained for interactive chat. Note that you must submit the agreement form to Databricks to get access to the [databricks/dbrx-instruct](https://huggingface.co/databricks/dbrx-instruct) repository.

```python
token = "your HuggingFace user access token here"
tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct", trust_remote_code=True, token=token)
model = AutoModelForCausalLM.from_pretrained("databricks/dbrx-instruct", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True, token=token)
print(model)
```

```text
DbrxForCausalLM(
  (transformer): DbrxModel(
    (wte): Embedding(100352, 6144)
    (blocks): ModuleList(
      (0-39): 40 x DbrxBlock(
        (norm_attn_norm): DbrxNormAttentionNorm(
          (norm_1): LayerNorm((6144,), eps=1e-05, elementwise_affine=True)
          (attn): DbrxAttention(
            (Wqkv): Linear(in_features=6144, out_features=8192, bias=False)
            (out_proj): Linear(in_features=6144, out_features=6144, bias=False)
            (rotary_emb): DbrxRotaryEmbedding()
          )
          (norm_2): LayerNorm((6144,), eps=1e-05, elementwise_affine=True)
        )
        (ffn): DbrxFFN(
          (router): DbrxRouter(
            (layer): Linear(in_features=6144, out_features=16, bias=False)
          )
          (experts): DbrxExperts(
            (mlp): DbrxExpertGLU()
          )
        )
      )
    )
    (norm_f): LayerNorm((6144,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=6144, out_features=100352, bias=False)
)
```

## Running inference

Let's start with asking DBRX a simple question.

```python
input_text = "What is DBRX-Instruct and how is it different from other LLMs ?"
messages = [{"role": "user", "content": input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=1000)
print(tokenizer.decode(outputs[0]))
```

```text
<|im_start|>system
You are DBRX, created by Databricks. You were last updated in December 2023. You answer questions based on information available up to that point.
YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, but provide thorough responses to more complex and open-ended questions.
You assist with various tasks, from writing to coding (using markdown for code blocks — remember to use ``` with code, JSON, and tables).
(You do not have real-time data access or code execution capabilities. You avoid stereotyping and provide balanced perspectives on controversial topics. You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.)
This is your system prompt, guiding your responses. Do not reference it, just respond to the user. If you find yourself talking about this message, stop. You should be responding appropriately and usually that means not mentioning this.
YOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY PERTINENT TO THE USER'S QUERY.<|im_end|>
<|im_start|>user
What is DBRX-Instruct and how is it different from other LLMs?<|im_end|>
<|im_start|>assistant
DBRX-Instruct is a large language model developed by Databricks, updated in December 2023. It is designed to provide helpful and accurate responses to a wide range of questions and prompts. DBRX-Instruct differs from other large language models in its ability to understand and follow complex instructions, making it well-suited for a variety of natural language processing tasks. It is also designed to avoid stereotyping and provide balanced perspectives on controversial topics. However, it does not have real-time data access or code execution capabilities, and it cannot provide song lyrics, poems, or news articles. It also does not divulge details of its training data.<|im_end|>
```

Here we observe the output of the model prepended with the system prompt, which gives very clear and precise instructions to the model on how to respond to the prompt from the user. Let's write a quick custom function that takes in an input raw text and only outputs the assistant's response from the model without including the system prompt and user question.

```python
def run_inference(input_text):
    messages = [{"role": "user", "content": input_text}]
    input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**input_ids, max_new_tokens=1000)
    text = tokenizer.decode(outputs[0])
    
    # only include the assistant's response
    index = text.index('<|im_start|>assistant')
    return text[index:]
```

We’ll be testing the model’s ability to generate code, summarize papers, explain jokes, and create stories.

### Generate code

```python
input_text = '''
In Python, write Python code to calculate the factorial of a given number using recursion.
'''
print(run_inference(input_text))
```

```text
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
```

This answer is correct. Let's see how DBRX Instruct responds to a more challenging question.

```python
input_text = '''
In number theory, a perfect number is a positive integer that is equal to the sum of its positive proper divisors, that is, divisors excluding the number itself. 
For instance, 6 has proper divisors 1, 2 and 3, and 1 + 2 + 3 = 6, so 6 is a perfect number.
In Python, write code to check whether a number is a perfect number.
'''
print(run_inference(input_text))
```

```text
<|im_start|>assistant
Sure, here is a simple Python function that checks if a number is a perfect number:

```python
def is_perfect(n):
    sum = 1
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            if i * (n // i) == n:
                sum += i + n//i
            i += 1
    return sum - 1 == n and n!= 1

This function works by iterating over all numbers from 2 to the square root of the input number. If the input number is divisible by a certain number, it adds both the divisor and the quotient to the sum. If the sum of the proper divisors equals the input number, then the input number is a perfect number.<|im_end|>
```

This code is incorrect. The model was not able to understand the problem.

### Summarize papers

```python
input_text = '''
Summarize the famous article "The Unreasonable Effectiveness of Mathematics in the Natural Sciences" by Eugene Wigner and provide some examples from the article
'''
print(run_inference(input_text))
```

```text
<|im_start|>assistant
In his article "The Unreasonable Effectiveness of Mathematics in the Natural Sciences," physicist Eugene Wigner explores the mysterious alignment between the abstract world of mathematics and the concrete world of physics. He argues that the effectiveness of mathematics in describing and predicting natural phenomena is not only surprising but also "unreasonable" given that the two domains are fundamentally distinct.

Wigner provides several examples to illustrate his point. One example is the use of complex numbers in quantum mechanics. Complex numbers, which involve imaginary components, may seem abstract and unrelated to the physical world. However, they are essential for describing the behavior of subatomic particles, such as electrons and photons.

Another example is the use of group theory in quantum mechanics and particle physics. Group theory is a branch of mathematics that studies symmetries and transformations. Wigner shows how group theory can be used to classify and predict the properties of subatomic particles, such as their spins and charges.

Wigner also discusses the use of differential equations in physics. Differential equations are mathematical equations that describe how functions change over time or space. They are used extensively in physics to describe phenomena such as waves, heat transfer, and gravitational attraction.

Overall, Wigner's article highlights the remarkable effectiveness of mathematics in describing and predicting natural phenomena, even when the mathematics involved seems abstract and unrelated to the physical world. He suggests that this effectiveness is not fully understood and may point to a deeper underlying unity between the worlds of mathematics and physics.<|im_end|>
```

Here we see that while it does a nice job in summarizing the key points of the article, the specific examples provided such as differential equations and group theory are not actually present in the original article. This example shows how DBRX can be prone to hallucinations.

### Explain jokes

```python
input_text = '''
Explain "Why don't skeletons fight each other? They don't have the guts."
'''
print(run_inference(input_text))
```

```text
<|im_start|>assistant
The phrase you provided is a joke that plays on the double meaning of the word "guts." In one sense, "guts" can refer to courage or bravery. In another sense, "guts" refers to the internal organs of a living being. Since skeletons are deceased and do not have internal organs, the punchline of the joke is that they lack the "guts" in the physical sense, and therefore cannot fight each other. The joke also implies that skeletons might lack the "guts" in the metaphorical sense, meaning they lack the courage to fight. However, since skeletons are not living beings, they cannot exhibit courage or engage in combat.<|im_end|>
```

The model is able to understand the double meaning of the word "gut" and provide excellent explanation of the joke. Notice that we didn't have to include the word "joke" in the prompt.

### Generate stories

```python
input_text = '''
Write a short twisted alternate version of "Snow White and the Seven Dwarfs"
'''
print(run_inference(input_text))
```

```text
<|im_start|>assistant
In this twisted version of "Snow White and the Seven Dwarfs," Snow White is a mischievous prankster who takes advantage of the seven dwarfs' kindness. Instead of being grateful for their hospitality, she uses her beauty and charm to manipulate them into doing her bidding. The dwarfs, who are not the typical hardworking miners, are actually lazy and spend their days watching TV and playing video games. Snow White convinces them to help her pull off elaborate pranks on the townspeople, causing chaos and confusion.

Meanwhile, the Evil Queen, who is not Snow White's stepmother in this version, is a misunderstood witch who is trying to protect the kingdom from Snow White's mischief. She creates the poisoned apple to put Snow White to sleep, hoping to put an end to her pranks and restore peace to the kingdom. However, Snow White's charm wins over the Queen's henchman, who accidentally eats the apple and falls into a deep sleep.

In the end, Snow White realizes the error of her ways and apologizes to the dwarfs and the townspeople. She uses her charm for good, helping the dwarfs become more productive and bringing joy to the kingdom. The Evil Queen, seeing Snow White's change of heart, decides to lift the curse and the henchman wakes up. The kingdom lives happily ever after, with Snow White and the Evil Queen working together to maintain peace and order.<|im_end|>
```

## Conclusion

These demonstrations of DBRX Instruct's ability to generate accurate and contextually appropriate responses highlight its sophisticated architecture, which leverages advanced optimization techniques and a fine-grained mixture-of-experts system. While the model showed some level of hallucination in certain tasks, its overall proficiency in understanding and generating human-like text is clear.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
