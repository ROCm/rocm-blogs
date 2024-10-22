---
blogpost: true
blog_title: 'Enhancing vLLM Inference on AMD GPUs'
date: 11 October 2024
thumbnail: './images/2024-07-29-roberta.jpg'
author: Clint Greene
tags: AI/ML, LLM, Serving
category: Applications & models
language: English
myst:
  html_meta:
    "description lang=en": "In this blog, we’ll demonstrate the latest performance enhancements in vLLM inference on AMD Instinct accelerators using ROCm. In a nutshell, vLLM optimizes GPU memory utilization, allowing more efficient handling of large language models (LLMs) within existing hardware constraints, maximizing throughput and minimizing latency."
    "keywords": "LLM, vLLM, Inference, Serving, ROCm, AMD GPUs, MI250, MI210, MI300"
    "property=og:locale": "en_US"
---

# Enhancing vLLM Inference on AMD GPUs

<span style="font-size:0.7em;">11 October, 2024 by {hoverxref}`Clint Greene<clingree>`. </span>

In this blog, we’ll demonstrate the latest performance enhancements in vLLM inference on AMD Instinct accelerators using ROCm. In a nutshell, vLLM optimizes GPU memory utilization, allowing more efficient handling of large language models (LLMs) within existing hardware constraints, maximizing throughput and minimizing latency. We start the blog by briefly explaining how causal language models like Llama 3 and ChatGPT generate text, motivating the need to enhance throughput and reduce latency. If you’re new to vLLM, we also recommend reading our introduction to [Inferencing and serving with vLLM on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/vllm/README.html).
ROCm 6.2 introduces support for the following vLLM features which we will use in this blog post.

- [**FP8 KV Cache**](#fp8-kv-cache): Store Key-Value (KV) pair data in FP8 (8-bit floating point) to enhance efficiency.

- [**GEMM Tuning**](#gemm-tuning): Achieve significant performance boosts by tuning matrix multiplications.

- [**FP8 Quantization**](#fp8-quantization): Enable inference on models quantized to FP8, improving speed without compromising accuracy.

## Understanding text generation

To fully appreciate the benefits of these features, let’s first walk through how causal language models like Llama 3 and ChatGPT generate text. These LLMs are pretrained to predict the next token in a sequence based on the preceding tokens. For example, if the model is given the sequence "The color of the sky varies," it learns to predict that the next token might be "depending" or another suitable word. When LLMs are given a prompt, they generate text iteratively by predicting the most probable next token in the sequence. The predicted token is then appended to the input, and the generation process continues until a maximum number of tokens is reached, or an End of Sentence (EOS) token is output, conveying to the model that the sequence is concluded. For instance, if an LLM is prompted with "The color of the sky varies," it might iteratively generate "depending," "on," "the," "time," "of," "day," and then "\<EOS>," ending the generation process. Typically, the generation process (inference) is divided into two key phases: prefill and decoding.
  
The prefill phase encodes, embeds, and computes the keys and values for the tokenized input prompt. In the transformer architecture, the self-attention mechanism relies on key-value pairs, where each input token is associated with a key vector and a value vector. The keys are used to compute attention scores by taking the dot product with the query vectors, allowing the model to determine the relevance of each token in the context of the current input. The values, in turn, provide the contextual information that is weighted by these attention scores during the generation process. These KV pairs are essential for inference, as they need to be computed or retrieved to inform the model's predictions. For the prefill phase, computations are mostly already optimized and can be executed in a single pass on the GPU.

In contrast, the decoding phase, responsible for generating the output, is significantly slower than the prefill phase due to the sequential nature of generating one token at a time. The model generates the first token based on the input prompt and the KV values computed from it. This token is appended to the existing input, and the updated sequence is used to compute a new set of KV values. This process is repeated to generate subsequent tokens until a stopping criterion is met. Due to the challenges in parallelizing the decoding phase, GPU throughput is limited, leading to reduced cost performance. To enhance throughput and latency, several optimizations can be applied. Notable optimizations supported by ROCm 6.2 with vLLM are the FP8 KV cache, GEMM tuning, and FP8 model quantization.

## Prerequisites

To follow along with this blog, you'll need:

- **Linux**: see the [supported Linux distributions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems).
- **ROCm**: see the [installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html).
- **MI300+ GPUs**: FP8 support is only available on MI300 series.

## Installation

To access the latest vLLM features in ROCm 6.2, clone the vLLM repository, modify the `BASE_IMAGE` variable in Dockerfile.rocm to `rocm/pytorch:rocm6.2_ubuntu20.04_py3.9_pytorch_release_2.3.0`, and build the Docker image using the commands below. Depending on your system, the build process might take a significant amount of time.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t vllm-rocm .
```

After building the vLLM ROCm Docker image, you can run it using the following command. To use a folder of LLMs in the container, replace `<path/to/model>` with the actual folder path. If you don't have any models to mount, remove the `-v <path/to/model>:/app/models` option.

```bash
docker run -it --network=host --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd --device /dev/dri -v <path/to/model>:/app/models vllm-rocm
```

## FP8 KV cache

The KV cache is key to speeding up LLM inference by avoiding redundant calculations of KV values for the same input tokens. When a prompt is processed, KV values are computed and a token is generated. This new token is appended to the input sequence, requiring new KV values to be computed. However, the KV values for the original tokens don't change, so recalculating them is inefficient. The KV cache solves this by storing the KV values of previous tokens in the GPU’s high-bandwidth memory (HBM) to prevent redundant computations. This boosts both throughput and efficiency.

However, using the KV cache does come with some limitations. It is typically stored in float16, and its size increases linearly with the sequence length, batch size, number of attention layers, and embedding dimension. As a result, larger model sizes and longer sequences require more memory, which can limit the feasible batch size. Additionally, GPU memory constraints impose practical limits on the sequence length that can be effectively managed with caching. With ROCm 6.2, the KV cache can now be stored in FP8 format in vLLM, significantly reducing the memory footprint. This enhancement effectively enables you to double the sequence length or batch size while keeping other parameters unchanged.

Let's now explore how to access this new feature in vLLM. To store the KV values in FP8, you simply include the `--kv-cache-dtype fp8` in the `vllm serve` command.

```bash
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --kv-cache-dtype fp8
```

If you watch the terminal output, you will see the following output for a single GPU in an MI300X system:

```text
INFO 08-29 19:06:51 gpu_executor.py:121] # GPU blocks: 157794, # CPU blocks: 4096
```

If you now run `vllm serve` without specifying `--kv-cache-dtype fp8`, you will see it output:

```text
INFO 08-29 19:05:21 gpu_executor.py:121] # GPU blocks: 78897, # CPU blocks: 2048
```

To estimate the maximum number of tokens that can be served, multiply the number of GPU blocks by the default block size of 16. With the KV cache in FP8 format, the maximum capacity is approximately 2,524,704 tokens. In comparison, with the KV cache in FP16 format, the capacity is about 1,262,352 tokens. This doubling of token capacity with FP8 demonstrates the significant reduction in the KV cache memory footprint.

## GEMM tuning

General Matrix Multiply (GEMM) operations underpin many neural network computations, such as convolutions and fully connected layers, and represent a large portion of the computational load in both training and inference. Optimizing GEMM can significantly improve throughput and reduce latency in generative AI applications. GEMM tuning adjusts factors like tile sizes, memory access patterns, and thread block configurations to maximize the GPU's parallel processing power.

With ROCm 6.2, optimizing GEMMs in vLLM is now much easier thanks to PyTorch's TunableOp integration. For more details, visit our blog on [GEMM Tuning with TunableOps](https://rocm.blogs.amd.com/artificial-intelligence/pytorch-tunableop/README.html). To optimize your vLLM workload, set the environment variable `PYTORCH_TUNABLEOP_ENABLED=1` and run your workload as usual. Once tuning is complete, a CSV file called `tunableop_results0.csv` will be generated with the results. Future runs will automatically load and apply these tunings. If the input or output lengths change, a new tuning run will be triggered.

For example, to GEMM tune Llama3-8B with an input and output length of 512 tokens and benchmark its latency, run the following command in the terminal:

```bash
export PYTORCH_TUNABLEOP_ENABLED=1
python3 benchmarks/benchmark_latency.py --input-len 512 --output-len 512 --num-iters 10 --model meta-llama/Meta-Llama-3-8B-Instruct
```

When the benchmarking finishes, you'll see it output the average latency.

```text
Avg latency: 4.3067230121872855 seconds
```

Next, let's benchmark and compare the latency without GEMM tuning.

```bash
PYTORCH_TUNABLEOP_ENABLED=0 python3 benchmarks/benchmark_latency.py --input-len 512 --output-len 512 --num-iters 10 --model meta-llama/Meta-Llama-3-8B-Instruct
```

Without GEMM tuning, the average latency increases to over 4.60 seconds! GEMM tuning decreases the latency by approximately 6.5%.

## FP8 quantization

As the size of LLMs grows to hundreds of billions of parameters, deploying them efficiently becomes both increasingly crucial and challenging. Typically, LLM components—such as weights, activations, and the KV cache—are represented using 16-bit floating point numbers, as this provides a good balance between output quality and speed. One widely used technique to further enhance inference speed in production is quantization, which reduces the numerical precision of some or all components of the LLM. Traditionally, this involves converting the data type of the model parameters from FP16 to INT8. However, INT8 quantization can significantly degrade model output quality, particularly for smaller LLMs with fewer than 7 billion parameters.

A promising alternative is the FP8 format, which offers similar performance benefits to 8-bit integer quantization without compromising output quality. FP8 provides greater precision and dynamic range than INT8, making it well-suited for quantizing performance-critical components of the LLM, including weights, activations, and the KV cache.

To understand FP8, it's useful to recall that a floating point number consists of three parts:

- **Sign**: A single bit indicating if the number is positive or negative
- **Exponent (Range)**: The power of the number.
- **Mantissa (Precision)**: The significant digits of the number.

FP8 comes in two variants with distinct use cases: E4M3 and E5M2:

- **E4M3**: Featuring 1 sign bit, 4 exponent bits, and 3 mantissa bits, it can represent values up to ±448 and NaN (Not a Number).
- **E5M2**: Featuring 1 sign bit, 5 exponent bits, and 2 mantissa bits, it can store values up to ±57,344, ±Infinity, and NaN.

Typically, E4M3 is used for the forward pass or inference because activations and weights require higher precision. In contrast, E5M2 is used during the backward pass, as gradients are less sensitive to precision loss but benefit from a higher dynamic range. In comparison, INT8 primarily focuses on mantissa (precision) and may or may not include a sign bit, but lacks an exponent, meaning it does not support a wide range.

With ROCm 6.2, you can now deploy models in E4M3 FP8 format using vLLM by simply running the command `vllm serve <model-name>` in the terminal. Here, `<model-name>` refers to either the local path of an E4M3 FP8 quantized model or the name of one of NeuralMagic's FP8 quantized models available on Hugging Face. For a complete list of FP8 NeuralMagic models supported by vLLM, [click here](https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127).

For instance to serve Meta-Llama-3-8B in FP8, run the following command:

```bash
vllm serve neuralmagic/Meta-Llama-3-8B-Instruct-FP8
```

You can then query it using a curl command in another terminal window.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
        "prompt": "Write a haiku about artificial intelligence",
        "max_tokens": 128,
        "top_p": 0.95,
        "top_k": 20,
        "temperature": 0.8
      }'
```

Let's now benchmark the latency Llama3-8B using the same parameters as before but in FP8 quantized format.

```bash
python3 benchmarks/benchmark_latency.py --input-len 512 --output-len 512 --num-iters 10 --model neuralmagic/Meta-Llama-3-8B-Instruct-FP8
```

This results in an average latency of 4.13 seconds. FP8 quantization, out of the box, reduces latency by approximately 10% compared to the previous FP16 benchmark. Next, we will benchmark the latency with GEMM tuning.

```bash
export PYTORCH_TUNABLEOP_ENABLED=1
python3 benchmarks/benchmark_latency.py --input-len 512 --output-len 512 --num-iters 10 --model neuralmagic/Meta-Llama-3-8B-Instruct-FP8
```

With GEMM tuning, the average latency now drops to under 3.40 seconds, representing a decrease of approximately 26% compared to the previous FP16 benchmark.

## Summary

In this blog post, we briefly discussed how LLMs like Llama 3 and ChatGPT generate text, motivating the role vLLM plays in enhancing throughput and reducing latency. We covered how to store values in FP8 format in the KV cache, optimize matrix multiplies for even faster computations, and perform full inference in FP8. With these latest enhancements, we showed how ROCm 6.2 can significantly accelerate your vLLM workloads.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
