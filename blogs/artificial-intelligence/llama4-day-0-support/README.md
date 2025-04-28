---
blogpost: true
blog_title: "Power Up Llama 4 with AMD Instinct: A Developer’s Day 0 Quickstart"
date: 06 Apr 2025
author: 'Liz Li, Seungrok Jung, Andy Luo'
thumbnail: 'llama.jpg'
tags: AI/ML
category: Applications & models
target_audience: ai enthusiasts and developers
key_value_propositions: day 0 support for Llama4
language: English
myst:
    html_meta:
        "author": "Liz Li, Seungrok Jung, Andy Luo"
        "description lang=en": "Explore the power of Meta’s Llama 4 multimodal models on AMD Instinct™ MI300X and MI325X GPUs - available from Day 0 with seamless vLLM integration"
        "keywords": "Llama4, Day0,  AMD Instinct GPUs, vLLM"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Applications and Models"
        "amd_developer_type": "ML/AI Developer"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_product_type": "Accelerators"
        "amd_developer_tool": "ROCm Software, Open-Source Tools"
        "amd_applications": "Large Language Model (LLM)"
        "amd_industries": "Data Center"
        "amd_blog_releasedate": Sun Apr 06, 12:00:00 PST 2025
---
<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->

# Power Up Llama 4 with AMD Instinct: A Developer’s Day 0 Quickstart

AMD is excited to announce Day 0 support for Meta’s latest leading multimodal intelligence Models — the [Llama 4 Maverick and Scout models](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) — on our AMD Instinct™ MI300X and MI325X GPU accelerators using vLLM. In this blog we will walk you through a step-by-step guide on deploying Meta’s Llama4 model using vLLM, docker setup, dependencies, and inference testing.

## Brief Introduction to Llama 4 Multimodal Models

Llama 4 introduces a leap in AI model architecture with a mixture-of-experts (MoE) design — featuring 17 billion active parameters and scaling up to 400 billion total parameters. Built for multimodal capabilities, it supports extended context lengths of 10M for deep, nuanced understanding across text and image inputs. Llama4 Maverick can achieve comparable results to DeepSeek v3 on reasoning and coding with less than half active parameters.

It uses alternating dense and MoE layers (128 routed experts and 1 shared expert) for inference efficiency. The multimodality function incorporates early fusion to integrate text and vision tokens into a unified model backbone, which enables pre-training the model with large amount of unlabeled text, image and video data jointly. A new training technique called MetaP is introduced to reliably set critical model hyper-parameters such as per-layer learning rates and initialization scales in FP8 precision, without sacrificing quality and ensuring high model FLOPs utilization.

## AMD and Meta Collaboration: Day 0 Support and Beyond

AMD has longstanding collaborations with Meta, vLLM, and Hugging Face and together we continue to push the boundaries of AI performance. Thanks to this close partnership, Llama 4 is able to run seamlessly on AMD Instinct GPUs from Day 0, using PyTorch and vLLM. Day 0 support means developers can start experimenting and deploying immediately, without waiting for post-launch optimization. Our partnership paves the way for even more groundbreaking innovations, ensuring that AI performance continues to evolve and meet the increasing demands of modern computing.

## Accelerating AI with AMD: Key Benefits for Developers

The AMD Instinct GPUs accelerators are purpose-built to handle the demands of next-gen models like Llama 4:

- **MI300X and MI325X** can run the full **400B-parameter Llama 4 Maverick** model in **BF16** on a **single node**, **reducing infrastructure complexity**.
- Massive **HBM memory capacity** enables support for **extended context lengths**, delivering smooth and efficient performance.
- Using Optimized **Triton** and **AITER** kernels provide best-in-class performance and TCO for production deployment.

## How to Run Llama4 On-line Inference mode with vLLM on AMD Instinct GPUs

With a few simple steps, you can run Llama4 on MI300x smoothly and efficiently on AMD Instinct GPUs.

**Prerequisites:**

Before you start, ensure:

- You have **AMD Instinct GPUs** and the **ROCm** drivers set up.
- Pull the Prebuilt [**Docker**](https://hub.docker.com/layers/rocm/vllm-dev/llama4-20250405/images/sha256-6da44b24235e47480f5ee1f7f9aff630c4233d6620047b3eefadb2097be686e3).

```bash
docker pull rocm/vllm-dev:llama4-20250407
```

- Download Llama models through Hugging face: [Llama 4 - a meta-llama Collection](https://huggingface.co/collections/meta-llama/llama-4-67f0c30d9fe03840bc9d0164)
  
```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-4-Scout-17B-16E --local-dir $LLAMA_DIR
```

### Step 1. Launch docker container

To run Llama4 efficiently on MI300x, launch docker container as below

```bash
docker run -it \
  --device /dev/dri \
  --device /dev/kfd \
  --network host \
  --ipc host \
  --group-add video \
  --security-opt seccomp=unconfined \
  -v /home:/workspace \
  --shm-size 64G rocm/vllm-dev:llama4-20250407 /bin/bash
```

### Step 2. Start vLLM online server

Once the container is launched , start the vllm online server

```shell
export VLLM_WORKER_MULTIPROC_METHOD=spawn 
export VLLM_USE_MODELSCOPE=False 
export VLLM_USE_TRITON_FLASH_ATTN=0 
vllm serve $LLAMA_DIR \
  --disable-log-requests -tp 8 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --max_num_batched_tokens=320000 \
  --max_model_len 32000
```

```{note}
`$LLAMA_DIR` is your model location. By default, it is vLLM v0 mode. To enable V1 mode, add `export VLLM_USE_V1=1` before running `vllm serve`. More information on vLLM v1 is available in this blog post: [https://blog.vllm.ai/2025/01/27/v1-alpha-release.html](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)
```

### Step3. Runing Inference using benchmark script

Once the vLLM server is up and running, open a new terminal, start the same docker, and execute the benchmark script as shown below.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn 
export VLLM_USE_TRITON_FLASH_ATTN=0 
python /app/vllm/benchmarks/benchmark_serving.py \
  --backend vllm \
  --model $LLAMA_DIR \
  --dataset-name random \
  --random-input-len $ISL \
  --random-output-len $OSL \
  --num-prompts 320 \
  --ignore-eos \
  --max-concurrency $concurrency \
  --percentile-metrics ttft,tpot,itl,e2el
```

```{note}
`$ISL` is input sequence length, `$OSL` is output sequence length, `$concurrency` is number of users
```

### Step 4. Running Inference with real world use case

The following command sends two images and text, and determines whether the images are similar or different.

```shell
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/models/Llama-4-Maverick-17B-128E-Instruct",
        "prompt": "<image>https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg</image><image>https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png</image> Can you describe how these two images are similar, and how they differ?",
        "max_tokens": 256,
        "temperature": 0
    }'
```

## Summary

This blog provided a step-by-step Day 0 guide allowing you to explore the power of Meta’s Llama 4 multimodal models on AMD Instinct MI300X and MI325X GPUs. With Llama 4 running seamlessly on AMD Instinct GPUs, developers can immediately build and scale next-gen multimodal AI applications. This milestone is part of AMD’s broader mission to support open, high-performance AI tooling. This collaboration drives innovation, providing the AI community with high-performance, open-source tools. Stay tuned for more technical insights and updates on how AMD and our partners are advancing AI and optimizing Llama4 performance

## Acknowledgements

AMD team members who contributed to this effort: Peng Sun, Hongxia Yang, James Jiang, Carlus Huang, Divakar Verma, Aleksandr Malyshev, Shengnan Xu, Joe Shajrawi, Shekhar Pandey, Niles Burbank, and Guruprasad MP.
This work would not have been possible without the strong collaboration and support of the Meta, vLLM, and Hugging Face teams.

```{update} Apr 8, 2025
Updated the blog with the latest vllm upstream on MI300X and a basic multi-modality example
```

## Additional Resources

- The Llama Models :[The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- Docker Link: [Image Layer Details - rocm/vllm-dev:llama4-20250405 | Docker Hub](https://hub.docker.com/layers/rocm/vllm-dev/llama4-20250405/images/sha256-ff5948a7cc5c9e78789748179dcfdd72775bf8ff6b4f0d2ccdac36982c98e9f2)
- Visit the ⁠[ROCm AI Developer Hub](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai.html?utm_source=web&utm_medium=amd&utm_campaign=deepseek_blog) for additional tutorials, blogs, open-source projects, and other resources for AI development on AMD GPUs.
- Explore AMD ROCm Software <https://www.amd.com/en/products/software/rocm.html>
- AMD Instinct Accelerators : <https://www.amd.com/en/products/accelerators/instinct.html>

### Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
