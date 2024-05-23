---
blogpost: true
date: 4 Apr 2024
author: Clint Greene
tags: AI/ML, LLM, Serving
category: Applications & models
language: English
html_meta:
  "description lang=en": "Inferencing and Serving with vLLM on AMD GPUs"
  "keywords": "LLM, vLLM, Inference, Serving, ROCm, AMD GPUs, MI250, MI210, MI300"
  "property=og:locale": "en_US"
---

# Inferencing and serving with vLLM on AMD GPUs

## Introduction

vLLM is a high-performance, memory-efficient serving engine for large language models (LLMs). It leverages PagedAttention and continuous batching techniques to rapidly process LLM requests. PagedAttention optimizes memory utilization by partitioning the Key-Value (KV) cache into manageable blocks. The KV cache stores previously computed keys and values, enabling the model to focus on calculating attention solely for the current token. These blocks are subsequently managed through a lookup table, akin to memory page handling in operating systems.

Continuous batching dynamically accumulates incoming requests into batches, eliminating the need to wait for a fixed batch size to be reached. This strategy enables vLLM to begin processing requests promptly upon arrival, thereby reducing latency and enhancing overall throughput.

In this blog, you will learn how to inference offline and deploy LLMs as a service using state of the art LLMs such as: Mistral-7B, Yi-34B, and Falcon-40B with vLLM.

## Prerequisites

To run this blog, you'll need:

* **Linux**: see [supported Linux distributions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems)
* **ROCm**: see the [installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
* **AMD GPUs**: see the [list of compatible GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

## Installation

We recommend using the vLLM ROCm docker container as a quick start because it's not trivial to install and build vLLM and it's dependencies from source. To get started, let's pull the vLLM ROCm docker container.

```bash
docker pull embeddedllminfo/vllm-rocm:vllm-v0.2.4
```

And then run it, replacing <path/to/model> with the appropriate path if you have a folder of LLMs you would like to mount and access in the container.

```bash
docker run -it --network=host --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd --device /dev/dri -v <path/to/model>:/app/model embeddedllminfo/vllm-rocm:vllm-v0.2.4
```

If you need to build it from source with newer versions, we recommend following the official [vLLM ROCm installation guide](https://docs.vllm.ai/en/latest/getting_started/amd-installation.html#quick-start-docker-rocm).

## Inferencing

To perform offline inferencing on a batch of prompts, you first need to import the `LLM` and
`SamplingParams` classes.

```python
from vllm import LLM, SamplingParams
```

Next, define your batch of prompts and the sampling parameters for generation. The sampling
parameters are:

* temperature = 0.8
* top k tokens to consider = 20
* nucleus sampling probability = 0.95
* maximum number of tokens generated = 128

For more information about the sampling parameters, refer to the
[class definition](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py).

```python
prompts = ["Write a haiku about machine learning"]
sampling_params = SamplingParams(max_tokens=128,
    skip_special_tokens=True,
    temperature=0.8,
    top_k=20,
    top_p=0.95,)
```

You're now ready to load an LLM. We'll demonstrate how to load the smaller Mistral-7B model, as well
as the larger models, Yi-34B and Falcon-40B.

### Mistral-7B

Since Mistral easily fits into the VRAM on an MI210, we can simply call the LLM class with the model's name which will load Mistral-7B from the Hugging Face cache folder. If you have the model weights elsewhere, you can also directly specify the path like this: `model="/app/model/mistral-7b/"` assuming you specified the appropriate folder to mount in the `docker run` command. If you haven't predownloaded the weights yet, we recommend doing it before this step to speedup the loading time.

```python
llm = LLM(model="mistralai/Mistral-7B-v0.1")
```

To generate text using the preceding prompt, we simply call `generate` to print the output

```python
outputs = llm.generate(prompts, sampling_params)

prompt = prompts[0]
generated_text = outputs[0].outputs[0].text
print(prompt + ': ' + generated_text)
```

```text
Silent algorithm
Whispers patterns to the world,
Intelligence grows.
```

To run much larger (30 B+) parameter language models, we must utilize tensor parallelism to distribute
the model across multiple GPUs. This works by splitting the model weight matrices column-wise into N
parts, with each of the N GPUs receiving a different part. After each GPU finishes computing, results are
joined with an `allreduce` operation. vLLM utilizes Megatron-LM's tensor parallelism algorithm and
`torch.distributed` to manage the distributed runtime on single nodes.

To enable tensor parallelism with vLLM, simply add it as a parameter to LLM, specifying the number of
GPUs you want to split across

```python
llm = LLM(model="tiiuae/falcon-40b-instruct", tensor_parallel_size=4)
```

Using the same prompt and sampling parameters, Falcon-40B outputs:

```text
Artificial intelligence
Takes in data, learns the patterns
Predicts the future
```

Now let's try another top-performing LLM: Yi-34B.

```python
llm = LLM(model="01-ai/Yi-34B", tensor_parallel_size=4)
```

This outputs:

```text
In the realm of data, where patterns dwell,
Machine learning finds its way to tell,
A story without words, just numbers strong.
```

## Serving

You can deploy your LLM as a service with vLLM by calling `vllm.entrypoints.api_server` in the terminal.

```bash
python -m vllm.entrypoints.api_server --model="mistralai/Mistral-7B-v0.1"
```

You can then query the vLLM service using a curl command in another terminal window.

```bash
curl http://localhost:8000/generate \
    -d '{
        "prompt": "Write a haiku about artificial intelligence",
        "max_tokens": 128,
        "top_p": 0.95,
        "top_k": 20,
        "temperature": 0.8
    }'
```

This generates the following JSON output:

```text
{
  "text": [
    "Silent thoughts awaken,
     Dance with logic and emotion,
     Life beyond the human."
  ]
}
```

If you need to serve an LLM that is too large to fit onto a single GPU, you can run multi-GPU serving by
adding `--tensor-parallel-size <number-of-gpus>` when starting the `api_server`.

```bash
python -m vllm.entrypoints.api_server --model="tiiuae/falcon-40b-instruct" --tensor-parallel-size 4
```

This generates the following output:

```text
{
  "text": [
    "A creation of man
     It can think like a human
     But without feelings"
  ]
}
```

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
