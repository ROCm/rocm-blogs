---
blogpost: true
blog_title: "Step-3 Deployment Simplified: A Developer’s Guide on AMD Instinct™ GPUs"
date: 4 Sep 2025
author: 'George Wang, Ning Zhang'
thumbnail: 'tb21.jpg'
tags: AI/ML, Serving
category: Applications & models
target_audience: AI DEVELOPERS AND ENTHUSIAST
key_value_propositions: Step-3 enables efficient long-context reasoning with up to 321B params, optimized for AMD Instinct™ GPUs to cut decoding costs and boost throughput
language: English
myst:
    html_meta:
        "author": "George Wang, Ning Zhang"
        "description lang=en": "Learn how to deploy Step-3, a 321B-parameter VLM with MFA & AFD, on AMD Instinct™ GPUs to cut decoding costs and boost long-context reasoning"
        "keywords": "Step3, vLLM, SGLang, Inference"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
        "amd_blog_authors": "George Wang, Ning Zhang"
---

# Step-3 Deployment Simplified: A Day 0 Developer’s Guide on AMD Instinct™ GPUs

Today’s large language models (LLMs) still face high decoding costs for long-context reasoning tasks. [Step-3](https://github.com/stepfun-ai/Step3) is a 321B-parameter open-source vision-language model (VLM) designed with hardware-aware model–system co-design to minimize decoding costs. With strong support from the open-source community—especially SGLang and Triton—AMD is excited to bring Step-3 to our Instinct™ GPU accelerators.

In this blog, we will walk you through a step-by-step guide to deploying Step-3 models using SGLang, docker setup, dependencies, and inference testing.

## Introduction to Step-3 Models

Step-3 prioritized optimizing decoding for several key reasons:

- Decoding is the most expensive both in terms of latency, as each token is generated sequentially, and in terms of compute cost, as it is heavily dependent on the KV cache.
- As reasoning models grow more capable with longer chains of thought, optimizing decoding allows the same compute budget to support deeper reasoning and greater intelligence.

Step-3 was designed to tackle this challenge. Built with model–system co-design, Step-3 aligns algorithmic innovation with hardware efficiency. Two key advances stand out:

- **Multi-Matrix Factorization Attention (MFA):** Shrinks the KV cache and reduces computation to 22% of DeepSeek-V3's per token attention cost, without sacrificing expressiveness.
- **Attention-FFN Disaggregation (AFD):** Separates attention and feed-forward layers into dedicated subsystems, enabling more efficient distributed inference.

Step-3 is a multimodal reasoning model, built on a Mixture-of-Experts (MoE) architecture with 321B total parameters and 38B active. Its configuration includes 61 Transformer layers with a hidden dimension of 7168. In each Transformer layer, MFA has 64 query heads which share a Key and a Value head, each with a dimension of 256. The query dimension is down-projected from 7168 to a lower-rank of 2048, followed by a normalization, and then up-projected to 64*256. MoE layers are applied to all FFNs except the first four and the last layer.

## Collaboration with the Open-Source Community

The success of Step-3 on AMD GPUs is rooted in deep collaboration with the open-source community, particularly SGLang and Triton. SGLang released the model-related code alongside Step-3’s launch, enabling developers to adapt it for AMD GPUs. For operators tied to NVIDIA-specific platform features, Triton provides alternative implementations that run smoothly on AMD hardware. The community has contributed numerous Triton operator kernels and integrated them into popular LLM inference engines like SGLang and vLLM, helping overcome vendor-specific library limitations.

Thanks to this collaboration, AMD has paved the way for supporting new AI models efficiently, ensuring performance continues to improve on AMD platforms while fostering growth across the broader AI open-source ecosystem.

## Run Step-3 on AMD Instinct GPUs

AMD Instinct GPUs offer three key advantages to running the Step-3 model:

1. **Scale efficiency at lower cost** - The full 321B- parameter Step-3 model can run in BF16 on a single CDNA3 GPU node, significantly reducing overall system cost.
2. **Memory capacity and bandwidth** - With large HBM memory and high bandwidth AMD Instinct GPUs are well-suited for Step-3's MFA/AFD solution, delivering higher throughput.
3. **Optimized software stack** - The optimized operator libraries of AMD ROCm™ software stack, and the optimization passes implemented in Triton AMD backend, which were used to accelerate other models’ inference, can also be beneficial to Step-3 support.

The detailed steps of running Step-3 are introduced in the sections below.

### Prerequisites

**Operating system:**

- Ubuntu 22.04/24.04: Ensure your system is running Ubuntu version 22.04/24.04.

**Hardware:**

- X86 server with AMD Instinct GPUs: we have run Step-3 support on an AMD Instinct GPU node with 8 MI300x. You can also try other CDNA3 GPUs.

**Software:**

- ROCm 6.3 or later version driver: Install and verify ROCm by following [the ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using the rocm-smi command.
- Docker engine: we will use the pre-built ROCm SGlang docker container in our Step-3 test, so docker engine is also required. You can follow the [instructions](https://docs.docker.com/engine/install/ubuntu/) to install docker engine on your server.
- Pre-built ROCm SGLang Docker image: Step-3 requires an SGLang 0.4.10 or later Docker image, such as lmsysorg/sglang:v0.4.10-rocm630-mi30x

### Step 1. Launch docker container

In this test, we use lmsysorg/sglang:v0.4.10-rocm630-mi30x image as an example. You can also try the latest ROCm SGLang Docker image for validation.

```bash
docker run -it \
  --device /dev/dri \
  --device /dev/kfd \
  --network host \
  --ipc host \
  --group-add video \
  --security-opt seccomp=unconfined \
  -v {The path to store Step-3 model file in host }:/models \
  --shm-size 64G lmsysorg/sglang:v0.4.10-rocm630-mi30x    /bin/bash
```

### Step 2. Launch SGLang serving service for Step-3

In this step, if you encounter issues with FA3 vision attention, you can refer to this [PR](https://github.com/sgl-project/sglang/pull/8656) and modify your local SGLang source code. The relevant file is located at /sgl-workspace/sglang/python/sglang/srt/layers/attention/vision.py inside the SGLang Docker container.

FA3 vision attention is developed for Nvidia platform, which needs cutlass and TMA hardware feature support. Step-3 used it as the default vision attention, which is hard to run directly on AMD Instinct GPUs. Instead, we can use the Triton version of vision attention as a replacement.

The ROCm AITER library is used by SGLang by default, boosting LLM performance on AMD Instinct GPUs. For Step-3, however, some new kernels and features are still being developed in AITER. If you encounter AITER-related issues during support, you can try switching to Triton kernels to run the new models successfully.

The SGLang serving command to run Step-3 on an 8xGPUs node is listed below:

```bash
python -m sglang.launch_server     --model-path /models/step3     --trust-remote-code     --tool-call-parser step3     --reasoning-parser step3     --tp 8 --mem-fraction-static 0.9 --attention-backend triton --port 50000
```

### Step 3. Running inference using the benchmark script

SGLang has provided a benchmark tool to test model performance, including throughput, TTFT, ITL these metrics. We can use it to test Step-3 inference on AMD Instinct GPUs.

The detailed command is listed below. You can change the settings of input/output length, number of prompts, and max concurrency value as you need in the test.

```bash
python3 -m sglang.bench_serving \
        --backend sglang --host 127.0.0.1 --port 50000 \
        --dataset-name random \
        --random-input-len $ISL \
        --random-output-len $OSL \
        --num-prompt $PROMPTS \
        --random-range-ratio 1.0 \
        --max-concurrency $cono
```

## Summary

In this blog we showed you how to deploy Step-3 on AMD Instinct™ GPUs using SGLang and Triton, simplifying setup while enabling efficient large-scale inference. Step-3 on AMD Instinct GPUs isn’t just about proving that one model runs well — it’s about establishing a repeatable path for developers to bring frontier AI models to AMD hardware. By leveraging open-source frameworks and AMD ROCm’s expanding ecosystem, you gain a faster route to deployment, with less friction and more freedom to innovate. Explore the workflow shared here, try Step-3 yourself, and start shaping how future models get supported from the very beginning.

## Additional Resources

- [Step-3 Model](https://huggingface.co/stepfun-ai/step3)
- [Step-3 technical report](https://arxiv.org/abs/2507.19427)
- [SGLang ROCm docker images](https://hub.docker.com/r/lmsysorg/sglang/tags?name=rocm)
- [Explore AMD ROCm Software](https://www.amd.com/en/products/software/rocm.html)
- [AMD Instinct Accelerators](https://www.amd.com/en/products/accelerators/instinct.html)

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

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
