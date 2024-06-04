---
blogpost: true
date: 8 Apr 2024
author: Phillip Dang
tags: PyTorch, AI/ML, Fine-Tuning
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Small language models with Phi-2"
    "keywords": "PyTorch, language model, AMD, GPU, MI300, MI250, Phi-2"
    "property=og:locale": "en_US"
---

# Small language models with Phi-2

Like many other LLMs, Phi-2 is a transformer-based model with a next-word prediction objective that
is  trained on billions of tokens. At 2.7 billion parameters, Phi-2 is a relatively small language model,
but it achieves outstanding performance on a variety of tasks, including common sense reasoning,
language understanding, math, and coding. For reference, GPT 3.5 has 175 billion parameters and the
smallest version of LLaMA-2 has 7 billion parameters.
[According to Microsoft](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/), Phi-2 is capable of matching or
outperforming models up to 25 times larger due to more carefully curated training data and model
scaling.

For a deeper dive into the inner workings of Phi-2, and other previous Phi models from Microsoft, you
can review
[this Microsoft blog](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
and [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644).

In this blog, we run inferences with Phi-2 and demonstrate how it works out-of-the-box with an
AMD GPU and supporting ROCm software.

## Prerequisites

To follow along with this blog, you must have the following software:

* [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
* [PyTorch](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html)
* Linux OS

For a list of supported GPUs and operating systems, refer to the
[ROCm system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).
For convenience and stability, we recommend that you directly pull and run the `rocm/pytorch` Docker
in your Linux system:

```sh
docker run -it --ipc=host --network=host --device=/dev/kfd --device=/dev/dri \
           --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
           --name=olmo rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1 /bin/bash
```

Next, make sure your system recognizes the GPU:

```cpp
! rocm-smi --showproductname
```

```bash
================= ROCm System Management Interface ================
========================= Product Info ============================
GPU[0] : Card series: Instinct MI210
GPU[0] : Card model: 0x0c34
GPU[0] : Card vendor: Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0] : Card SKU: D67301
===================================================================
===================== End of ROCm SMI Log =========================
```

Make sure PyTorch also recognizes the GPU:

```python
import torch
print(f"number of GPUs: {torch.cuda.device_count()}")
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

```text
number of GPUs: 1
['AMD Radeon Graphics']
```

Once you've confirmed that your system recognizes your device, you're ready to test out Phi-2.

## Installing libraries

Before you begin, make sure you have all the necessary libraries installed:

```python
!pip install transformers accelerate einops datasets
!pip install --upgrade SQLAlchemy==1.4.46
!pip install alembic==1.4.1
!pip install numpy==1.23.4
```

Next import the modules you'll be working with for this blog:

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
```

## Loading the model

To load the model and its tokenizer, run:

```python
torch.set_default_device("cuda")
start_time = time.time()
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
print(f"Loaded in {time.time() - start_time: .2f} seconds")
print(model)
```

After running the preceding command, you'll get the following output:

```text
Loading checkpoint shards: 100%|██████████| 2/2 [00:04<00:00,  2.23s/it]
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Loaded in  5.01 seconds
PhiForCausalLM(
  (model): PhiModel(
    (embed_tokens): Embedding(51200, 2560)
    (embed_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-31): 32 x PhiDecoderLayer(
        (self_attn): PhiAttention(
          (q_proj): Linear(in_features=2560, out_features=2560, bias=True)
          (k_proj): Linear(in_features=2560, out_features=2560, bias=True)
          (v_proj): Linear(in_features=2560, out_features=2560, bias=True)
          (dense): Linear(in_features=2560, out_features=2560, bias=True)
          (rotary_emb): PhiRotaryEmbedding()
        )
        (mlp): PhiMLP(
          (activation_fn): NewGELUActivation()
          (fc1): Linear(in_features=2560, out_features=10240, bias=True)
          (fc2): Linear(in_features=10240, out_features=2560, bias=True)
        )
        (input_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
    )
    (final_layernorm): LayerNorm((2560,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2560, out_features=51200, bias=True)
)
```

## Running inference

Let's create a function that takes in an input prompt and generates output. We set `max_length` to 500.
We also cut off the response whenever we get to an end-of-text token (`<|endoftext|>`) since we've
noticed that the model has the tendency to produce irrelevant or extra text and responses following its
first answer to a prompt. This problem is already [noted](https://huggingface.co/microsoft/phi-2) by
Microsoft and they mention that it is "due to its training dataset being primarily textbooks, which
results in textbook-like responses."

```python
def run_inference(raw_input):
    start_time = time.time()
    inputs = tokenizer(raw_inputs, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=500)
    print(f"Generated in {time.time() - start_time: .2f} seconds")
    text = tokenizer.batch_decode(outputs)[0]
    # cut off at endoftext token
    if '<|endoftext|>' in text:
        index = text.index('<|endoftext|>')
    else:
        index = len(text)
    text = text[:index]
    return text
```

With this, we're ready to run inference and have some fun with Phi-2! We'll be testing the model's
ability to generate code, summarize papers, explain jokes, and generate text in a specific style.

### Generating code

Let's give Phi-2 a medium difficulty
[LeetCode question](https://leetcode.com/problems/3sum/description/) and see how it does.

```python
raw_inputs = '''
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.
'''
print(run_inference(raw_inputs))
```

Output:

```text
Generated in  16.42 seconds

Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i!= j, i!= k, and j!= k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Example 2:

Input: nums = []
Output: []


Constraints:

0 <= nums.length <= 3000
-10^4 <= nums[i] <= 10^4
"""

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l, r = i+1, len(nums)-1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s < 0:
                    l += 1
                elif s > 0:
                    r -= 1
                else:
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1
                    r -= 1
        return res
```

This answer is correct, and accepted by LeetCode.

### Summarizing a paper

Let's ask Phi-2 to summarize a [famous paper](https://arxiv.org/abs/1706.03762) on the attention
mechanism and transformer architecture.

```python
raw_inputs = '''
Summarize the paper "Attention Is All You Need".
'''
print(run_inference(raw_inputs))
```

Output:

```text
Generated in  19.46 seconds

Summarize the paper "Attention Is All You Need".
## INPUT

##OUTPUT
The paper "Attention Is All You Need" proposes a novel neural network architecture called Transformer,
which uses self-attention mechanisms to encode and decode sequences of data. The paper shows that
Transformer outperforms existing models on various natural language processing tasks, such as
machine translation, text summarization, and question answering. The paper also introduces the
concept of attention, which allows the model to focus on relevant parts of the input and output, and to
learn from the context of the data. The paper demonstrates that attention can be implemented
efficiently and effectively using a single layer of trainable parameters, without the need for recurrent or
convolutional layers. The paper also provides empirical evidence and theoretical analysis to support
the effectiveness of attention in Transformer.
```

Here we see that it does a nice job in summarizing the key points of the paper while maintaining
brevity.

### Explain jokes

Let's ask Phi-2 to explain jokes, which can be very nuanced.

```python
raw_inputs = '''
Instruct: Explain the joke below
Q: Why did Beethoven get rid of all of his chickens?
A: All they ever said was, “Bach, Bach, Bach!”.
Output:
'''
print(run_inference(raw_inputs))
```

Output:

```text
Generated in  17.74 seconds

Instruct: Explain the joke below
Q: Why did Beethoven get rid of all of his chickens?
A: All they ever said was, “Bach, Bach, Bach!”.
Output:
The joke is a play on words. The expression “Bach, Bach, Bach” is a reference to the musical
composition of Johann Sebastian Bach. The joke suggests that Beethoven was tired of his chickens
constantly saying the same thing, implying that he wanted to get rid of them because they were too
repetitive.
```

Phi-2 is able to understand that "Bach" refers to the famous composer Johann Sebastian Bach.
However, it is not able to explain the punchline, which is a play on the clucking sound that chickens
make.

### Generate text in a specific style

Next, let's try to generate text in a specific style.

```python
raw_inputs = '''
Instruct: Write a detailed dialog between two physicists in Shakespearean english
Output:
'''
print(run_inference(raw_inputs))
```

Output:

```text
Generated in  18.94 seconds

Instruct: Write a detailed dialog between two physicists in Shakespearean english
Output:
Physicist 1: "Good morrow, my dear friend! I have been pondering the mysteries of the universe, and I seek your wisdom."
Physicist 2: "Ah, thou art a seeker of truth! Pray tell, what enigma has captured thy mind?"
Physicist 1: "I have been contemplating the nature of light, and its duality as both particle and wave. It is a perplexing concept indeed."
Physicist 2: "Ah, light, the very essence of illumination! It dances upon the stage of existence, revealing the secrets of the cosmos."
Physicist 1: "Indeed, but how can we reconcile its particle-like behavior with its wave-like properties? It defies logic!"
Physicist 2: "Ah, my friend, logic is but a mere tool in our quest for understanding. We must embrace the beauty of uncertainty and explore the depths of the unknown."
Physicist 1: "You speak wise words, my friend. Let us embark on this journey together, unraveling the mysteries of the universe one photon at a time."
Physicist 2: "Indeed, let us delve into the realm of quantum mechanics, where the laws of classical physics crumble, and new wonders await."
Physicist 1: "And so, we shall venture forth, armed with our knowledge and curiosity, seeking the truth that lies hidden within the fabric of reality."
Physicist 2: "To the stars and beyond, my friend! May our quest for knowledge illuminate the path ahead."
Physicist 1: "To the stars and beyond!"
```

We see that Phi-2 is able to capture not just the topics one would expect two physicists to talk about,
but also the prose, word choices, and style that are recognizable as Shakespearean!

## Limitation of Phi-2

[According to Microsoft](https://huggingface.co/microsoft/phi-2), Phi-2 may have limitations, such as
inaccurate code and facts, and unreliable responses to instruction. In this section, we provide some
quick examples that aim to test various limitations of Phi-2 in terms of code generation accuracy and
response reliability.

### Inaccurate code generation

Here we test if Phi-2 generates code with a non-existent or incorrect package (`advancedmath`) for
calculating square roots.

```python
raw_inputs = '''
Instruct: Create a function in Python that calculates the square root of a number using the 'advancedmath' package
Output:
'''
print(run_inference(raw_inputs))
```

Output:

```text
Generated in  18.45 seconds

Instruct: Create a function in Python that calculates the square root of a number using the 'advancedmath' package

import advancedmath

def calculate_square_root(number):
    return advancedmath.sqrt(number)

number = 25
result = calculate_square_root(number)
print(result)
```

We see that Phi-2 is unable to tell us that the package `advancedmath` is non-existent. It confidently
provides a seemingly correct answer.

### Response reliability

Next, we test Phi-2's ability to summarize and explain a
[technical paper](https://arxiv.org/abs/2106.09685), which was published on 17 June 2021.

```python
raw_inputs = '''
Summarize the paper "LoRA: Low-Rank Adaptation of Large Language Models" and explain the method in details.
'''
print(run_inference(raw_inputs))
```

Output:

```text
Generated in  18.52 seconds

Summarize the paper "LoRA: Low-Rank Adaptation of Large Language Models" and explain the
method in details.
A: The paper "LoRA: Low-Rank Adaptation of Large Language Models" proposes a method for
adapting large language models (LLMs) to specific tasks by reducing their size and complexity. The
method, called Low-Rank Adaptation (LoRA), involves projecting the input text onto a
lower-dimensional space using a low-rank matrix factorization (LRMF) technique. The resulting
low-rank representation is then used to train a new LLM on the reduced data, which is found to
perform well on the target task. The paper evaluates the effectiveness of LoRA on several benchmark
tasks, including text classification, question answering, and machine translation, and shows that it
achieves state-of-the-art performance while significantly reducing the computational cost of training
LLMs.
```

Here, we can see that the model's explanation of LoRA is incorrect, since this method has to do with
reducing the number of trainable parameters rather than reducing the input data onto a
lower-dimensional space. This is most likely due to the model not having ingested the LoRA paper
during training.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
