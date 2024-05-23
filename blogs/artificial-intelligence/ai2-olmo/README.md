---
blogpost: true
date: 17 Apr 2024
author: Douglas Jia
tags: AI/ML, GenAI, PyTorch, Diffusion Model, LLM
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Inferencing with AI2's OLMo model on AMD GPU"
    "keywords": "OLMo, AI2, Allen Institute for AI, GPU,
  AMD, MI300, MI250, LLM, ROCm, Open Source"
    "property=og:locale": "en_US"
---

# Inferencing with AI2's OLMo model on AMD GPU

In this blog, we will show you how to generate text using AI2's OLMo model on AMD GPU.

## Introduction

The OLMo (Open Language Model) developed by the Allen Institute for AI is of significant importance to the generative AI field. It is a truly open Large Language Model (LLM) and framework, designed to provide full access to its pre-training data, training code, model weights, and evaluation suite. This commitment to openness sets a new precedent in the LLM landscape, empowering academics and researchers to collectively study and advance the field of language models. This open approach is expected to drive a burst of innovation and development around generative AI.

OLMo follows the classical decoder-only transformer architecture that is used by many GPT-style models. Its performance on major benchmarks matches or exceeds that of other popular models of similar size. For more details about its architecture and performance evaluation, refer to [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838).

One notable aspect is that the OLMo team conducted a performance comparison by concurrently pre-training their model on both AMD MI250X GPU and Nvidia A100 GPU. Their study, coupled with two separate investigations carried out by the Databricks team: [Training LLMs with AMD MI250 GPUs and MosaicML](https://www.databricks.com/blog/amd-mi250) and [Training LLMs at Scale with AMD MI250 GPUs](https://www.databricks.com/blog/training-llms-scale-amd-mi250-gpus), offers comprehensive third-party comparisons between AMD and Nvidia GPU performance.

The opinions expressed here are not endorsed by AMD and do not represent their official views.

## Implementation

The code examples used in this blog were tested with ROCm 6.0, Ubuntu 20.04, Python 3.9, and PyTorch 2.1.1. For a list of supported GPUs and OS, please refer to [this page](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html). For convenience and stability, we recommend you to directly pull and run the `rocm/pytorch` Docker in your Linux system with the following code:

```sh
docker run -it --ipc=host --network=host --device=/dev/kfd --device=/dev/dri \
           --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
           --name=olmo rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1 /bin/bash
```

After entering the docker container, we need to install the required packages:

```sh
pip install transformers ai2-olmo
```

Then we will run the following code in a Python console. First, we will need to check if PyTorch can detect the GPUs on your system. The following code block will show you the number of GPU devices on your system.

```python
import torch
torch.cuda.device_count()
```

```sh
8
```

In the code block below, we will instantiate the inference pipeline with OLMo-7B model. Please note, OLMo models have different sizes: 1B, 7B and 65B.

```python
import hf_olmo
from transformers import pipeline
# Default device is CPU; device>=0 is setting the device to a GPU.
olmo_pipe = pipeline("text-generation", model="allenai/OLMo-7B", device=0)
```

Next, supply the text prompt, generate and print out the output from the model.

```python
output = olmo_pipe("Language modeling is ", max_new_tokens=100)
print(output[0]['generated_text'])
```

```text
Language modeling is 
a branch of natural language processing that aims to 
understand the meaning of words and sentences. 
It is a subfield of computational linguistics. 
The goal of natural language modeling is to 
build a model of language that can be used 
to predict the next word in a sentence. 
This can be used to improve the accuracy 
of machine translation, to improve the 
performance of speech recognition systems, 
and to improve the performance of 
```

You can also input multiple prompts to generate responses in one run.

```python
input = ["Deep learning is the subject that", "There are a lot of attractions in New York", "Why the sky is blue"]
output = olmo_pipe(input, max_new_tokens=100)
print(*[i[0]['generated_text'] for i in output], sep='\n\n************************\n\n')
```

```text
Deep learning is the subject that is being studied by the researchers. It is a branch of machine learning that is used to create artificial neural networks. It is a subset of deep learning that is used to create artificial neural networks. It is a subset of deep learning that is used to create artificial neural networks. It is a subset of deep learning that is used to create artificial neural networks. It is a subset of deep learning that is used to create artificial neural networks. It is a subset of deep learning that is used to create artificial

************************

There are a lot of attractions in New York City, but the most popular ones are the Statue of Liberty, the Empire State Building, and the Brooklyn Bridge.
The Statue of Liberty is a symbol of freedom and democracy. It was a gift from France to the United States in 1886. The statue is made of copper and stands on Liberty Island in New York Harbor.
The Empire State Building is the tallest building in the world. It was built in 1931 and stands 1,454 feet tall. The building has 102 floors and

************************

Why the sky is blue?
Why the grass is green?
Why the sun shines?
Why the moon shines?
Why the stars shine?
Why the birds sing?
Why the flowers bloom?
Why the trees grow?
Why the rivers flow?
Why the mountains stand?
Why the seas are blue?
Why the oceans are blue?
Why the stars are blue?
Why the stars are white?
Why the stars are red?
Why the stars are yellow?
```

But you may have noticed that the above generated text can be highly repetitive. For example, the first response repeated the sentence "It is a subset of deep learning that is used to create artificial neural networks." several times; the third response repeated the pattern "Why the xxx is xxx?" multiple times. Why is that? The pipeline's default decoding strategy is greedy search, selecting the token with the highest probability as the next token. While effective for many tasks and small output sizes, it can lead to repetitive results when generating longer outputs. Next, we will employ other decoding strategies to mitigate this problem. If you are interested to know more about this topic, you can refer to [this tutorial](https://huggingface.co/docs/transformers/en/generation_strategies) from Hugging Face.

In the following code block, we'll demonstrate how to optimize text generation using a combination of Top-K and Top-P token sampling strategies. The typical approach is to use Top-K sampling to narrow down potential tokens to the K most likely options, then apply Top-P sampling within this subset to select tokens that cumulatively reach the probability threshold P. This process balances selecting high-probability tokens (Top-K) with ensuring diversity within a confidence level (Top-P). You can also use these two strategies separately.

```python
output = olmo_pipe(input, max_new_tokens=100, do_sample=True, top_k=40, top_p=0.95)
print(*[i[0]['generated_text'] for i in output], sep='\n\n************************\n\n')
```

```text
Deep learning is the subject that deals with Artificial intelligence and machine learning. In the context of artificial intelligence, Deep learning is an emerging technology that is based on artificial neural networks. It is used in almost all fields of AI such as robotics, language translation, computer vision, and others. This technology is used in computer vision for automatic image processing and recognition tasks. It is also used for image classification, speech recognition, and text translation.
With the increasing demand for artificial intelligence, the use of deep learning has also been

************************

There are a lot of attractions in New York, such as Central Park and the Brooklyn Bridge. Visiting all of these places would be quite overwhelming, so we recommend starting with the ones that you find the most interesting.
The best attractions for teens are Times Square, the Statue of Liberty, The Empire State Building, Central Park, and the Brooklyn Bridge.
New York City is a very busy city, so it can be challenging for a teenager to get from one place to another. This is why we recommend using public transportation, which

************************

Why the sky is blue" - it is a question that has been puzzling philosophers and scientists since time began.
But the world's top physicist has unveiled the secret to the colour and says he "loves" being asked about it as it has fascinated him throughout his career.
Prof Stephen Hawking, 74, of Cambridge University, said blue appears in the sky because it takes the longest wavelength of sunlight, blue, to reach the earth after it passes through the atmosphere.
He added that sunlight in the sky
```

As evident from the generated output, the repetitive issue has been addressed, resulting in more natural-sounding text. However, please note that these responses may not be factually accurate as they are generated solely based on the trained model and lack fact-checking capability. We will explore ways to improve the factual accuracy of the responses in our future blogs. Stay tuned!

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
