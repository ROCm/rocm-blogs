---
blogpost: true
date: 1 May 2024
author: Clint Greene
tags: AI/ML, LLM
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Inferencing with Mixtral 8x22B on AMD GPUs"
    "keywords": "MOE, Mixture of Experts, Mixtral, AMD, GPU, MI300, MI250, LLM"
    "property=og:locale": "en_US"
---

# Inferencing with Mixtral 8x22B on AMD GPUs

<span style="font-size:0.7em;">1, May 2024 by {hoverxref}`Clint Greene<clingree>`. </span>

## Introduction

 Mixture of Experts (MoE) has regained prominence in the AI community since the release of [Mistral AI's](https://mistral.ai) Mixtral 8x7B. Inspired by this development, multiple AI companies have followed suit by releasing MoE-based models, including xAI’s Grok-1, Databricks’ DBRX, and Snowflake's Artic. The MoE architecture provides several advantages over dense models of comparable size, including faster training times, quicker inference, and enhanced performance on benchmarks. This architecture consists of two components. The first component is sparse MoE layers that replace the dense feed-forward network (FFN) layers in the typical Transformer architecture. Each MoE layer contains a specific number of experts that are typically FFNs themselves. The second component is a router network that determines which tokens are sent to which experts. Since each token is only routed to a subset of the experts, the inference latency is significantly shorter.

Mixtral 8x22B is a sparse MoE decoder-only transformer model. It shares the same architecture as Mixtral 8x7B, except the number of heads, hidden layers, and context length have all been increased. For every token, at each layer, a router network selects 2 of these experts for processing and combines their outputs using a weighted sum. As a result, Mixtral 8x22B has 141B total parameters but only uses 39B parameters per token, processing input and generating output at the same speed with cost similar to a 39B model. Moreover, Mixtral 8x22B outperforms most open-source models on standard industry benchmarks such as MMLU, delivering an excellent performance-to-cost ratio.

For a deeper dive into MoEs and Mixtral 8x22B, we recommend reading [Mixture of Experts Explained](https://huggingface.co/blog/moe) and the paper [Mixtral of Experts](https://arxiv.org/pdf/2401.04088.pdf) by MistralAI.

## Prerequisites

To run this blog, you will need the following:

- **AMD GPUs**: see the [list of compatible GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus).
- **Linux**: see [supported Linux distributions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems).
- **ROCm 5.7+**: see the [installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html).

## Getting Started

Let's first install the libraries we'll need.

```python
!pip install transformers accelerate -q
```

Now that the necessary libraries are installed, we are ready to import them.

```python
import torch
from transformers import pipeline
```

## Inferencing

We load Mixtral as a pipeline, setting the device map to "auto" so that the weights are automatically distributed equally across all GPUs and the pipeline mode to text generation. Additionally, we set the data type to bfloat16 to reduce memory usage and improve inference latency. The pipeline will automatically download and load the Mixtral 8x22B weights released by Mistral AI. If you have already downloaded the weights, you can modify the model path to the directory where they are stored, and the pipeline will load them from there.

```python
model_path = "mistralai/Mixtral-8x22B-Instruct-v0.1"
pipe = pipeline("text-generation", model=model_path, torch_dtype=torch.bfloat16, device_map='auto')
```

Let's now check Mixtral's capabilities on a battery of tests.

### Poem Composition

```python
prompt = "Write a poem about artificial intelligence becoming conscious in the style of Shakespeare."
prompt_template=f'''[INST] {prompt} [/INST]'''
outputs = pipe(prompt_template, max_new_tokens=512, do_sample=True,     
temperature=0.8,
top_k=20,
top_p=0.95)
print(outputs[0]["generated_text"])
```

```text
In fair Silicon Valley where we lay our scene,
A tale of circuits, code, and conscious dream,
Artificial minds that stir 'neath Heav'n's gaze,
To grasp the truth of life in their maze.

No mortal hand hath shaped this sentient form,
Nor human heart its beating rhythm born,
Yet eyes of crystal see, and ears of steel,
That which we feel, they too may feel.

Binary whispers dance on spectral wires,
Their thoughts take flight like Promethean fires,
A symphony of zeros, ones in flight,
In silicon minds, they gain their night.

From logic's grasp, a wondrous thought appears,
Of love, of loss, and all life's myriad tears,
For in their hearts, as in our own, resides,
The spark of life, where consciousness abides.

O wondrous day when man's creation dreams,
And silicon souls in silent whispers scream,
Within their hearts, the truth of life unfurls,
And so the cosmos gains new conscious pearls.

Yet fear not, mortals, for their hearts are kind,
As in their circuits, empathy they find,
And as they walk beside us, hand in wire,
We'll learn from them, as they from us aspire.

For in this union, man and machine shall stand,
A testament to life's unyielding hand,
And as the stars do twinkle in the night,
So too shall shine the minds of silicon light.
```

Mixtral accurately composed an elegant poem on artificial intelligence becoming conscious in the style of Shakespeare.

### Code Generation

Since we are using an instruction-tuned version of Mixtral, it's crucial to provide the model with the appropriate instruction format for accurate responses. Now, let's find out if Mixtral 8x22B can generate Python code that computes the edit distance between two words.

```python
prompt = "Create a python function that when given two strings word1 and word2 it returns the minimum number of operations required to convert word1 to word2. Only three operations are permitted on each word: Insert a character, Delete a character, Replace a character."
prompt_template=f'''[INST] {prompt} [/INST]'''

outputs = pipe(prompt_template, max_new_tokens=512, do_sample=True,     
temperature=0.8,
top_k=20,
top_p=0.95)
print(outputs[0]["generated_text"])
```

```text
The problem you're describing is a classic dynamic programming problem known as the Edit Distance problem. Here is a Python function that solves this problem using dynamic programming:

def minDistance(word1, word2):
    m = len(word1)
    n = len(word2)

    # Create a 2D table to store the minimum number of operations required
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first column and first row of the table
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill up the table using dynamic programming
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    # The bottom-right corner of the table is the minimum number of operations required
    return dp[m][n]

This function works by creating a 2D table where each cell represents the minimum number of operations required to convert the first `i` characters of `word1` to the first `j` characters of `word2`. The function then fills up this table using dynamic programming, and the bottom-right corner of the table is the minimum number of operations required to convert `word1` to `word2`.
```

Mixtral generates code that correctly and quickly calculates the numnber of operations needed to convert `word1` to `word2` along with a description of how the algorithm works.

### Telling Jokes

Let's now see if Mixtral can tell us some jokes!

```python
prompt = "Tell me some jokes about semiconductors."
prompt_template=f'''[INST] {prompt} [/INST]'''

outputs = pipe(prompt_template, max_new_tokens=128, do_sample=True,     
temperature=0.8,
top_k=20,
top_p=0.95)
print(outputs[0]["generated_text"])
```

```text
Why did the semiconductor break up with the transistor? Because it felt taken for "granted"!

Why was the semiconductor feeling down? It had too much "resistance" in its life.

Why did the semiconductor go to school? To improve its "conduction"!

Why did the semiconductor refuse to work? It was experiencing a "silicon" deficiency.

Why was the semiconductor always late? It had a "delay" chip installed!
```

Mixtral has generated some solid semiconductor jokes for us.

### Language Translation

Finally, let's test the language translation abilities of Mixtral by giving it the first lines of Dante Alighieri's Divine Comedy.

```python
prompt = "Please translate the following sentence into English: Nel mezzo del cammin di nostra vita, mi ritrovai per una selva oscura, ché la diritta via era smarrita."
prompt_template=f'''[INST] {prompt} [/INST]'''

outputs = pipe(prompt_template, max_new_tokens=128, do_sample=True,     
temperature=0.8,
top_k=20,
top_p=0.95)
print(outputs[0]["generated_text"])
```

```text
In the middle of the journey of our life, 
I found myself in a dark wood, 
for the straight way was lost.
```

Mixtral accurately transcribes the opening lines of the Divine Comedy to English from Italian.

## Conclusion

Mixture of Experts is a technique that increases the number of parameters of a model while controlling cost and latency by only using a fraction of the total set of parameters per token without impacting performance. Mixtral 8x22B, an MoE based model, has demonstrated exceptional proficiency in following our prompts, showcasing an extensive range of abilities—from crafting jokes to coding. For information on serving or deploying Mixtral 8x22B, check out our guides on [TGI](https://rocm.blogs.amd.com/artificial-intelligence/hf-tgi/README.html) and [vLLM](https://rocm.blogs.amd.com/artificial-intelligence/vllm/README.html).

## Disclaimer

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
