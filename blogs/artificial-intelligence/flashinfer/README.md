---
blogpost: true
blog_title: "Enabling FlashInfer on ROCm for Accelerated LLM Serving"
date: 1 Oct 2025
author: 'Rishi Madduri, Diptorup Deb, Debasis Mandal, Clint Greene, Mukhil Azhagan Mallaiyan Sathiaseelan, Yao Liu, Phani Vaddadi, Vish Vadlamani'
thumbnail: 'flashinfer-thumbnail.jpg'
tags: AI/ML, GenAI, Optimization, Serving
category: Applications & models
target_audience: AI practitioners
key_value_propositions: Demonstrate that FlashInfer now runs on ROCm for accelerated LLM serving.
language: English
myst:
    html_meta:
        "author": "Rishi Madduri, Diptorup Deb, Debasis Mandal, Clint Greene, Mukhil Azhagan Mallaiyan Sathiaseelan, Yao Liu, Phani Vaddadi, Vish Vadlamani"
        "description lang=en": "FlashInfer is an open-source library for accelerating LLM serving that is now supported by ROCm."
        "keywords": "Flashinfer, transformer, inference, prefill, decode, flash attention, LLM, serving"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI, Data Science"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Rishi Madduri, Diptorup Deb, Debasis Mandal, Clint Greene, Mukhil Azhagan Mallaiyan Sathiaseelan, Yao Liu, Phani Vaddadi, Vish Vadlamani"
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

# Enabling FlashInfer on ROCm for Accelerated LLM Serving

[FlashInfer](https://arxiv.org/pdf/2501.01005) is an innovative framework designed to accelerate inference of large language models (LLMs). Given the explosive growth and adoption of models like DeepSeek R1, Llama 3, and Qwen 3, efficient inference is critical to meet the demands of real-world deployment. However, challenges such as GPU memory bottlenecks, throughput limitations, and latency remain significant hurdles for deploying these models at scale.

Originally developed for NVIDIA GPUs using CUDA, FlashInfer leverages advanced techniques like efficient key-value (KV) cache management and optimized attention mechanisms to minimize latency and memory usage. Now we are excited to announce an early release of [FlashInfer on ROCm](https://github.com/ROCm/flashinfer), enabling users with AMD GPUs to achieve improved inference performance. This release extends FlashInfer's capabilities to AMD hardware, laying the foundation for reduced inference latency, optimized memory usage, and lower operational costs.

This blog provides an overview of FlashInfer, its core concepts, and a simple example of how to use FlashInfer on ROCm for decoding. Note that this is an early release, with some features still in development.

## What is FlashInfer?

FlashInfer is a library that accelerates LLM inference by optimizing critical components of the transformer architecture, including:

- **Efficient KV-Cache Management**: Reduces memory overhead by compressing and managing key-value caches dynamically.
- **Optimized Attention Mechanisms**: Implements high-performance attention kernels to minimize computation latency.
- **Memory-Efficient Decoding**: Streamlines decoding processes to maximize throughput and reduce memory footprint.

AMD GPU users can now begin leveraging these optimizations.

## Prerequisites

To run FlashInfer on ROCm, ensure the following requirements are met:

- **Linux**: see the [supported Linux distributions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems).
- **ROCm 6.4+**: see the [installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html).
- **MI300X or MI325X**

Once your system is ready, follow the steps below to set up the environment.

## Getting Started

To simplify the setup process, we'll clone the FlashInfer repository from ROCm's GitHub and build a Docker container tailored for ROCm.

```bash
git clone https://github.com/ROCm/flashinfer.git
cd flashinfer
```

Using the provided Dockerfile, create a container with all necessary dependencies for FlashInfer on ROCm:

```bash
docker build -f docker/Dockerfile.rocm_ci --target flashinfer_base -t flashinfer-rocm . 2>&1 | tee docker_build.log
```

Run the container with necessary privileges and GPU access:

```bash
docker run -it --network=host --group-add=video \
           --privileged --ipc=host --cap-add=SYS_PTRACE \
           --security-opt seccomp=unconfined --device /dev/kfd \
           --device /dev/dri flashinfer-rocm
```

The following section explains how to run FlashInfer on ROCm using a basic example.

## Running FlashInfer on ROCm

To illustrate FlashInfer's capabilities, here's a simple example showcasing a single-request decode attention kernel:

```python
import torch
import flashinfer

kv_len = 2048
num_kv_heads = 32
head_dim = 128

k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)

# decode attention

num_qo_heads = 32
q = torch.randn(num_qo_heads, head_dim).half().to(0)

o = flashinfer.single_decode_with_kv_cache(q, k, v) # decode attention without RoPE on-the-fly
```

## Summary

The early release of FlashInfer for ROCm marks an important milestone in making high-performance LLM inference accessible on AMD GPUs. While this is just the beginning, ongoing development will expand features and improve support, helping the community run large models more efficiently and cost-effectively on AMD hardware.

Stay tuned for updates, and try FlashInfer on ROCm today to start experiencing accelerated LLM serving on AMD GPUs!

## Acknowledgements

The authors wish to acknowledge the AMD teams that supported this work, whose contributions were instrumental in enabling FlashInfer: Aditya Bhattacharji, Pankaj Gupta, Radha Srimanthula, Anisha Sankar, Amit Kumar, Ram Seenivasan, Eliot Li, Ian Dass, Kiran Thumma, Aakash Sudhanwa, Ehud Sharlin, Saad Rahim.

## Additional Resources

Ye, Z., Chen, L., Lai, R., Lin, W., Zhang, Y., Wang, S., Chen, T., Kasikci, B., Grover, V., Krishnamurthy, A., & Ceze, L. (2025). FlashInfer: Efficient and customizable attention engine for LLM inference serving. *arXiv preprint*. [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
