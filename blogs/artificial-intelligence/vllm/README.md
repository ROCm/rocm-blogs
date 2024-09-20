---
blogpost: true
blog_title: 'Inferencing and serving with vLLM on AMD GPUs'
date: Sept 19 2024
author: Clint Greene
tags: AI/ML, LLM, Serving
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "Inferencing and Serving with vLLM on AMD GPUs"
    "keywords": "LLM, vLLM, Inference, Serving, ROCm, AMD GPUs, MI250, MI210, MI300"
    "property=og:locale": "en_US"
---

# Inferencing and serving with vLLM on AMD GPUs

<span style="font-size:0.7em;">19 September, 2024 by {hoverxref}`Clint Greene<clingree>`. </span>

## Introduction

In the rapidly evolving field of artificial intelligence, Large Language Models (LLMs) have emerged as powerful tools for understanding and generating human-like text. However, deploying these models efficiently at scale presents significant challenges. This is where vLLM comes into play. vLLM is an innovative open-source library designed to optimize the serving of LLMs using advanced techniques. Central to vLLM is PagedAttention, a novel algorithm that enhances the efficiency of the model's attention mechanism by managing it as virtual memory. This approach optimizes GPU memory utilization, facilitating the processing of longer sequences and enabling more efficient handling of large models within existing hardware constraints. Additionally, vLLM incorporates continuous batching to maximize throughput and minimize latency. By leveraging these cutting-edge techniques, vLLM significantly improves the performance and scalability of LLM deployment, allowing organizations to harness the power of state-of-the-art AI models more effectively and economically.

Diving deeper into vLLM’s advanced features, PagedAttention optimizes memory usage by partitioning the Key-Value (KV) cache into manageable blocks of non-contiguous memory, similar to how operating systems manage memory pages. This structure ensures optimal use of memory resources. The KV cache enables the model to focus attention calculations solely on the current token by storing previously computed keys and values. This approach speeds up processing and reduces memory overhead, as it eliminates the need to recompute attention scores for past tokens.

In parallel, continuous batching improves throughput and minimizes latency by dynamically grouping incoming requests into batches, eliminating the need to wait for a fixed batch size. This allows vLLM to process requests immediately as they arrive, ensuring faster response times and greater efficiency in handling high volumes of requests.

In this blog, we’ll guide you through the basics of using vLLM to serve large language models, from setting up your environment to performing basic inference with state of the art LLMs such as Qwen2-7B, Yi-34B, and Llama3-70B with vLLM on AMD GPUs.

## Prerequisites

To run this blog, you'll need:

* **Linux**: see the [supported Linux distributions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems)
* **ROCm**: see the [installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
* **AMD GPUs**: see the [list of compatible GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)

## Installation

To access the latest vLLM features on ROCm, clone the vLLM repository and build the Docker image using the commands below. Depending on your system, the build process might take a significant amount of time.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t vllm-rocm .
```

Once you've successfully built the vLLM ROCm Docker image, you can run it using the following command. If you have a folder containing multiple LLMs that you'd like to access within the container, simply replace <path/to/model> with the actual path to that folder to mount and utilize your LLMs seamlessly within the container; if not, just remove `-v <path/to/model>:/app/models` from the command below.

```bash
docker run -it --network=host --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd --device /dev/dri -v <path/to/model>:/app/models vllm-rocm
```

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

You're now ready to load an LLM. We'll demonstrate how to load the smaller Qwen2-7B model, as well
as the larger models, Yi-34B and Llama3-70B.

### Qwen2-7B

Since Qwen2 easily fits into the VRAM on an MI210, we can simply call the LLM class with the model's name which will load Qwen2-7B from the Hugging Face cache folder. If you have the model weights elsewhere, you can also directly specify the path like this: `model="/app/model/qwen-7b/"` assuming you specified the appropriate folder to mount in the `docker run` command. If you haven't predownloaded the weights yet, we recommend doing it before this step to speedup the loading time.

```python
llm = LLM(model="Qwen/Qwen2-7B-Instruct")
```

To generate text using the preceding prompt, we simply call `generate` to print the output

```python
outputs = llm.generate(prompts, sampling_params)

prompt = prompts[0]
generated_text = outputs[0].outputs[0].text
print(prompt + ': ' + generated_text)
```

which outputs:

```text
Data flows in streams,
Algorithms sift and learn,
Predictions emerge.
```

To run much larger (30 B+) parameter language models, we might need to utilize tensor parallelism to distribute
the model across multiple GPUs. This works by splitting the model weight matrices column-wise into N
parts, with each of the N GPUs receiving a different part. After each GPU finishes computing, results are
joined with an `allreduce` operation. vLLM utilizes Megatron-LM's tensor parallelism algorithm and python's `multiprocessing` to manage the distributed runtime on single nodes.

To enable tensor parallelism with vLLM, simply add it as a parameter to LLM, specifying the number of GPUs you want to split across. We also recommend using multiprocessing `mp` as the backend for distributing rather than `ray` because it's faster.

```python
llm = LLM(model="meta-llama/Meta-Llama-3-70B-Instruct", tensor_parallel_size=4, distributed_executor_backend="mp")
```

Using the same prompt and sampling parameters, Llama3-70B outputs:

```text
Algorithms dance
Data whispers secrets sweet
Intelligence born
```

Now let's try another top-performing LLM: Yi-34B.

```python
llm = LLM(model="01-ai/Yi-34B-Chat", tensor_parallel_size=4, distributed_executor_backend="mp")
```

This outputs:

```text
In the digital realm,
Algorithms learn and evolve,
Predicting the future.
```

## Serving

You can deploy your LLM as a service with vLLM by calling `vllm serve <model-name>` in the terminal.

```bash
vllm serve Qwen/Qwen2-7B-Instruct
```

You can then query the vLLM service using a curl command in another terminal window.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/workspace/Meta-Llama-3-70B-Instruct/",
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
  "id": "cmpl-622896e563984235a6f83633c46db7cf",
  "object": "text_completion",
  "created": 1724445396,
  "model": "Qwen/Qwen2-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "text": ". Machines learn and grow,\nBinary thoughts never falter,\nIntelligence artificial. \n\nThis haiku captures the idea of machines learning and growing through artificial intelligence, while their thoughts are never subject to the same limitations as human emotions or physical constraints. The use of binary suggests the reliance on a system of ones and zeros, which is a fundamental aspect of how computers process information. Overall, the haiku highlights the potential and possibilities of artificial intelligence while also acknowledging the limitations of machines as they lack the same complexity and depth of human intelligence.",
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null,
      "prompt_logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "total_tokens": 115,
    "completion_tokens": 108
  }
}
```

If you need to serve an LLM that is too large to fit onto a single GPU, you can run multi-GPU serving by adding `--tensor-parallel-size <number-of-gpus>` when starting `vllm serve`. We also specify multiprocessing `mp` as the backend for distributing.

```bash
vllm serve --model="meta-llama/Meta-Llama-3-70B-Instruct" --tensor-parallel-size 4 --distributed-executor-backend=mp
```

This generates the following output:

```text
{
  "id": "cmpl-bed1534b639a4ab7b65775f75cdeed33",
  "object": "text_completion",
  "created": 1724446207,
  "model": "meta-llama/Meta-Llama-3-70B-Instruct",
  "choices": [
    {
      "index": 0,
      "text": "\nHere is a haiku about artificial intelligence:\n\nMetal minds awake\nIntelligence born of code\nFuture's uncertain",
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": 128009,
      "prompt_logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 32,
    "completion_tokens": 24
  }
}
```

```{Note}
This blog was originally uploaded on April 4, 2024
```

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
