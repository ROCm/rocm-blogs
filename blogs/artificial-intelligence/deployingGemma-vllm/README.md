---
blogpost: true
blog_title: "Deploying Google’s Gemma 3 Model with vLLM on AMD Instinct MI300X GPUs: A Step-by-Step Guide"
date: 14 Mar 2025
author: 'Shekhar Pandey, Anshul Gupta'
thumbnail: 'gemma_blog.png'
tags: AI/ML, LLM, Serving
category: Applications & models
target_audience: AI Enthusiast and AI Developers
key_value_propositions: day 0 support for Gemma3 Model
language: English
myst:
    html_meta:
        "author": "Shekhar Pandey, Anshul Gupta"
        "description lang=en": "AMD is excited to announce the integration of Google’s Gemma 3 models with AMD Instinct™ MI300X GPUs"
        "keywords": "Gemma3, Gemini Modes, LLM, AMD, MI300X, Instinct"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Applications and models"
        "amd_developer_type": "ML/AI Developer, Software Developer, Data & Research Scientists"
        "amd_deployment": "Servers"
        "amd_product_type": "Accelerators"
        "amd_developer_tool": "ROCm Software, Open-Source Tools"
        "amd_applications": "Large Language Model (LLM)"
        "amd_industries": "Data Center"
        "amd_blog_releasedate": Fri Mar 14, 12:00:00 PST 2025
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
# Deploying Google’s Gemma 3 Model with vLLM on AMD Instinct&trade; MI300X GPUs: A Step-by-Step Guide

AMD is excited to announce the integration of Google’s Gemma 3 models with AMD Instinct MI300X GPUs, optimized for high-performance inference using the vLLM framework. This collaboration empowers developers to harness advanced AMD AI hardware for scalable, efficient deployment of state-of-the-art language models. In this blog we will walk you through a step-by-step guide on deploying Google’s Gemma 3 model using [vLLM](https://github.com/vllm-project/vllm) on AMD Instinct GPUs, covering Docker setup, dependencies, authentication, and inference testing. Remember, the Gemma 3 model is gated—ensure you request access before beginning deployment.

## What is Gemma 3 Model?

Gemma 3 is a family of lightweight yet powerful open models, built from the same research behind Google’s Gemini models. With multimodal capabilities, Gemma 3 processes both text and image inputs to generate high-quality text output. It supports a large 128K context window, over 140 languages, and comes in multiple sizes, making it highly versatile across different deployment environments. Leveraging Google’s powerful Gemma 3 multimodal model on AMD Instinct™ MI300 GPUs can significantly enhance inference workloads.  

## Prerequisites

Before you start, ensure:

- **Docker** is installed and configured correctly.
- You have **AMD Instinct GPUs** and the **ROCm** drivers set up.
- You have **requested access** to Gemma 3 at the gated Hugging Face repository:

   👉 [https://huggingface.co/google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it)

## Step 1: Create a Dockerfile

Create a file named `Dockerfile` with the following content:

```dockerfile
# Base image with ROCm support
FROM rocm/vllm-dev:base_main_20250312

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y git curl && apt-get clean

# Clone vLLM repository
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /workspace/vllm

# Upgrade pip and install AMD SMI utility
RUN pip install --upgrade pip && \
    pip install /opt/rocm/share/amd_smi

# Install Python dependencies
RUN pip install --upgrade numba scipy huggingface-hub[cli,hf_transfer] setuptools_scm && \
    pip install "numpy<2" && \
    pip install -r requirements/rocm.txt

# Install specific Transformers version for Gemma 3 support
RUN pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# Set up GPU architecture and install vLLM
ENV PYTORCH_ROCM_ARCH="gfx942"
RUN python3 setup.py develop

# Set working directory for when container starts
WORKDIR /workspace

# Default command when container starts
CMD ["/bin/bash"]
```

This Dockerfile encapsulates all the setup steps in a single file, making your deployment process reproducible and consistent. Let's break down what it does:

- Uses the official ROCm vLLM development image as a base
- Installs required system dependencies
- Clones the vLLM repository
- Sets up the Python environment with all necessary packages
- Installs the specific Transformers version that supports Gemma 3
- Configures GPU architecture and builds vLLM from source

## Step 2: Build and Run Your Docker Container

Build your custom Docker image:

```bash
docker build -t gemma3-vllm:latest .
```

Run the container with proper GPU access:

```bash
docker run -it --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri \
    -v $HOME:/work-gemma \
    --security-opt seccomp=unconfined \
    --group-add video \
    -w /workspace gemma3-vllm:latest
```

## Step 3: Authenticate with Hugging Face (Gemma 3 Gated Repo)

Once inside the container, authenticate with your Hugging Face account:

```bash
huggingface-cli login
```

**Reminder**:

If you haven't requested access yet, you must first request permission here:

👉 [https://huggingface.co/google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it)

## Step 4: Serve Gemma 3 via vLLM Server

Start the vLLM inference server for Gemma 3:

```bash
vllm serve "google/gemma-3-27b-it"
```

Wait until the server fully initializes (logs indicate readiness).

## Step 5: Test Your Gemma 3 Deployment via REST API

Verify the successful deployment by sending a test request using `curl`:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "google/gemma-3-27b-it",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe this image in one sentence."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
            }
          }
        ]
      }
    ]
  }' | jq
```

You should receive a concise image description similar to below mentioned output, confirming Gemma 3 is operational.

```bash
{
  "id": "chatcmpl-971f9680f0d1422c940c25eab1db2978",
  "object": "chat.completion",
  "created": 1741892856,
  "model": "google/gemma-3-27b-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": null,
        "content": "The Statue of Liberty stands proudly on Liberty Island with the Manhattan skyline rising in the background under a clear blue sky.",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": 106
    }
  ],
  "usage": {
    "prompt_tokens": 276,
    "total_tokens": 300,
    "completion_tokens": 24,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null
}
```

## Summary

AMD is committed to Day 0 support for important new AI models on AMD hardware. With Day 0 support for Gemma 3, we are excited to see how the community will make use of these innovative new models with AMD Instinct GPUs.

## Additional Resources

- [**Deploying Llama-3.1 8B using vLLM**](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/3_inference_ver3_HF_vllm.html) — Tutorials for AI Developers 2.0
- [**OCR with Vision-Language Models using vLLM**](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/ocr_vllm.html) — Tutorials for AI Developers 2.0
- [**ROCm vLLM Docker Images**](https://hub.docker.com/r/rocm/vllm/tags) — Docker Hub

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
