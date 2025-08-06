---
blogpost: true
blog_title: "Day 0 Developer Guide: Running the First Open GPT Models from OpenAI on AMD AI Hardware"
date: 05 Aug 2025
author: 'Andy Luo, Shekhar Pandey, Hongxia Yang, Mahdi Ghodsi, Charles Yang, Niles Burbank, George Wang, Kailash Gogineni,Xun Wang, Zhenyu Gu, Yao Fu, Yanyuan Qin, Anshul Gupta'
thumbnail: 'OpenAI12.jpg'
tags: AI/ML
category: Ecosystems and Partners
target_audience: ai developers and enthusiasts
key_value_propositions: Day 0 enablement of AMD HW on Open AI Models
language: English
myst:
    html_meta:
        "author": "Andy Luo, Shekhar Pandey, Hongxia Yang, Mahdi Ghodsi, Charles Yang, Niles Burbank, George Wang, Kailash Gogineni, Hongxia Yang, Xun Wang, Zhenyu Gu, Yao Fu, Yanyuan Qin, Ali Zaidy, Lukasz Burzawa, Mehmet Cagri, Shao-Chun Lee, Anshul Gupta"
        "description lang=en": "Day 0 support across our AI hardware ecosystem from our flagship AMD InstinctTM MI355X and MI300X GPUs, AMD Radeon™ AI PRO R700 GPUs and AMD Ryzen™ AI Processors"
        "keywords": "OpenAI, Day0, AMD Instinct, AMD Radeon, AMD Ryzen, ROCm"
        "vertical": "Developers, AI, Data Science"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Radeon Graphics, Ryzen Processors, Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Ryzen AI Software, Open-Source Tools"
        "amd_blog_applications": "AI Inference, AI Training"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
        "amd_blog_authors": "Andy Luo, Shekhar Pandey, Hongxia Yang, Mahdi Ghodsi, Charles Yang, Niles Burbank, George, Wang, Kailash Gogineni, Xun Wang, Zhenyu Gu, Yao Fu, Yanyuan Qin,Anshul Gupta"
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

# Day 0 Developer Guide: Running the Latest Open Models from OpenAI on AMD AI Hardware

OpenAI has officially released its open models: gpt-oss-120b and gpt-oss-20b. AMD now provides out-of-the-box, day 0 support for the latest open models from OpenAI, enabling developers to easily fine-tune and deploy across cloud to client environments using AMD hardware, the AMD ROCm™  and AMD Ryzen™ AI software stack, and seamless open source integrations. At AMD, we’re excited to announce day 0 support across our AI hardware, including our flagship AMD Instinct™ MI355X and MI300X GPUs, AMD Radeon™ AI PRO R9700  GPUs, and AMD Ryzen™ AI processors.  

We are also thrilled to introduce a Hugging Face Space featuring a chatbot powered by OpenAI models running on AMD Instinct GPUs. This interactive platform allows users to engage with our models effortlessly, showcasing the capabilities of AMD hardware in real-world applications.

In this blog, we’ll walk developers through the process of fine-tuning models and running fast, efficient inference on AMD hardware. Built in collaboration with the open source community, our workflow leverages  vLLM for inference and PyTorch for fine-tuning OpenAI models. We’ll cover everything from a simple Docker setup to installing dependencies and running tests—ensuring the entire process is smooth and developer-friendly.

## Uniqueness of OpenAI Models

The new open models from OpenAI gpt-oss-120b and gpt-oss-20b are
designed with flexibility and real-world applicability in mind. We're
excited to bring the best of these capabilities to AMD platforms from
day zero.

<!-- markdownlint-disable -->

-   **Reasoning-Optimized Architectures**: Both models support
    chain-of-thought reasoning with adjustable effort levels, enabling
    dynamic control over compute and latency trade-offs.

-   **Instruction Following & Tool Use**: With built-in support for
    instruction following and tool calling, these models are ideal for
    agent-based systems and workflows.

-   **Responses API Compatibility**: The models aim for compatibility
    with the Responses API reference spec. Open community support makes
    integration with modern APIs and tools straightforward for
    developers looking to build on top of these models.

-   **Text-Only Simplicity, High Efficiency**: These models are designed
    for text-in/text-out tasks, ideal for workloads where GPUs
    shine---delivering massive parallelism for token generation while
    minimizing system memory overhead.

-   **Advanced Attention Design**: the OpenAI model uses Rotary Positional
    Embeddings (RoPE) with support for up to 128k context tokens, an
    alternating attention pattern switching between full context and a
    sliding window of 128 tokens

-   **Token-Choice MoE with SwiGLU**: The architecture uses a
    Mixture-of-Experts (MoE) design featuring SwiGLU activations and
    softmax-after-topk for expert weight calculation.

## Accelerating OpenAI with AMD: Key Benefits for Developers at Day 0

Day 0 support from AMD goes beyond technical readiness---it's about
accelerating time-to-insight, reducing time-to-market, and putting
developers first at every step. Developers are able to train, fine-tune,
or run inference with the latest open models on a variety of AMD
devices the day they are released. Our collaborative efforts with the
open source community enable optimized workflows for cloud-scale
training or local prototyping, without roadblocks, putting the latest generative AI models right at your fingertips.


### Inference and Fine-tuning on AMD Instinct GPUs

-   We provide a reference example for fine-tuning the [OpenAI](https://openai.com/) gpt-oss-20b model using PyTorch and Hugging Face-PEFT LoRA on AMD Instinct MI300X and MI355X GPUs leveraging the HuggingFaceH4/ultrachat_200k dataset with minimal  computational overhead.

-   For inference, we use vLLM and ROCm software support and the
    underlying kernels that are optimized, ensuring lightning-fast speed
    and delivering a smooth, low-latency experience.

-   The OpenAI gpt-oss-120b model fits well within the memory footprint
    of AMD Instinct™ MI300X, MI325X, and MI355X GPUs---making it ideal
    for inference.

-   Token-based MoE with SwiGLU and softmax-after-topk is well-optimized
    for AMD Instinct GPUs, leveraging Triton and ROCm for high
    throughput during inference.

### Inference on Radeon™ AI PRO R9700 GPUs

-   The R9700 GPUs' 32GB GDDR6 and multi-GPU support run OpenAI gpt-oss-20b
    entirely in GPU memory while 640 GB/s bandwidth ensures fast VRAM
    access, boosting attention-heavy inference with higher throughput
    and lower latency.

-   With dual-slot cooling and PCIe 5.0, a 4×R9700 configuration offers
    128 GB VRAM and 128 AI accelerators, providing substantial compute
    capacity to boost vLLM inference speed, parallelism, and deployment
    efficiency.

### Inference on AMD Ryzen™ AI MAX+ and AMD Ryzen™ AI 300 Series Processors

-   The AMD Ryzen™ AI Max+ 395 (128GB) is the world’s first consumer AI PC processor to run OpenAI GPT-OSS 120B.
-   The AMD Ryzen™ AI 300 processors with 32GB memory or more can run the OpenAI GPT-OSS 20B models using LM Studio.
-   See how to run it today and learn more: [How To Run OpenAI’s GPT-OSS 20B and 120B Models on AMD Ryzen™ AI Processors and Radeon™ Graphics Cards](https://statics.teams.cdn.office.net/evergreen-assets/safelinks/2/atp-safelinks.html)

## Day-0 Access on Hugging Face Spaces

We've made the newly released gpt-oss-120b model publicly accessible via
a [Gradio-based chatbot hosted on Hugging Face Spaces](https://huggingface.co/spaces/amd/gpt-oss-120b-chatbot). This demo runs on
AMD MI300X GPUs and provides a simple streaming chat interface for
immediate interaction with the model, with no setup or credentials
required.

While the current demo offers basic access, the underlying model
supports a range of advanced capabilities:

-   Controllable reasoning effort: Developers can configure how
    deeply the model reasons, selecting from low, medium, or high effort
    levels ([Learn
    more here.](https://platform.openai.com/docs/guides/reasoning?api-mode=responses))

-   Agentic Capabilities:

    -   Built-in tools: The model includes native support for tools
        such as web browsing and a Python code interpreter.

    -   Custom tool integration: Developers can connect their own
        tools via function calling, enabling integration with
        domain-specific APIs and workflows.

These capabilities are exposed through the OpenAI-compatible responses
API, which supports structured input and output, streaming, tool use,
and reasoning control. Full documentation is available
[here](https://platform.openai.com/docs/api-reference/responses).

The Hugging Face Space provides a simple starting point to explore the
model, with more advanced capabilities available through direct API
integration and custom deployments on AMD GPUs.


## How to Run Inference with vLLM using AMD GPUs

AMD has enabled vLLM to support OpenAI gpt-oss-120b and gpt-oss-20b
models on AMD GPUs on Day 0, including

-   AMD Instinct MI355X (gfx950),

-   AMD Instinct MI300X and MI325X (gfx942)

-   AMD Radeon AI PRO R9700 (gfx1201).

### Step 1: Select the vLLM Docker Image Based on Your GPU

Pick the vLLM Docker image according to your choice of GPU (Instinct MI355X, MI325X, MI300X, or Radeon AI PRO R9700).

Follow these steps to get started.

```bash

VLLM_IMAGE=$(arch=$(/opt/rocm/llvm/bin/amdgpu-arch | tail -1); [[ $arch == gfx942 ]] && echo "rocm/vllm-dev:open-mi300-08052025" || ([[ $arch == gfx950 ]] && echo "rocm/vllm-dev:open-mi355-08052025") || ([[ $arch == gfx1201 ]] && echo "rocm/vllm-dev:open-r9700-08052025"))

```


### Step 2: Launch the ROCm vLLM Docker Container

Start a container with the necessary ROCm software, device, network
privileges and AMD GPU specific containers:

```bash
docker run -it \
  --ipc=host \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --shm-size 32G \
  -w /workspace $VLLM_IMAGE
```

### Step 3: Authenticate with Hugging Face

Request access via the model page on Hugging Face and then log in using:

`huggingface-cli login`

Ensure you have access to the models below.

```shell
# gpt-oss-120b
huggingface-cli download openai/gpt-oss-120b

# gpt-oss-20b
huggingface-cli download openai/gpt-oss-20b 
```


### Step 4 : Launch the vLLM Server

Both gpt-oss models can fit on a single GPU when using AMD Instinct
MI355X, MI325X and MI300X GPUs.

You can serve the model using tensor parallelism with configurations
like 1, 2, 4, or 8. The example script below uses tensor parallelism
(\$TP) set to 1, but feel free to adjust it based on your hardware and
performance requirements.

The example below is for the MI300X, MI325X and Radeon GPUs, for performance, AITER is enabled
(AITER_ROCM_USE_AITER) and AITER Unified attention
(VLLM_USE_AITER_UNIFIED_ATTENTION) is used as the attention
implementation, and the "full_cuda_graph" mode is turned on.

```shell
#!/bin/bash
TP=1
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
vllm serve openai/gpt-oss-120b \
  --tensor-parallel  $TP \
  --no-enable-prefix-caching \
  --disable-log-requests \
  --compilation-config '{"full_cuda_graph": true}'
```
For **MI355X** we have some extra flags which provides an additional performance boost.
```shell
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_USE_AITER_TRITON_FUSED_SPLIT_QKV_ROPE=1
export VLLM_USE_AITER_TRITON_FUSED_ADD_RMSNORM_PAD=1
export TRITON_HIP_PRESHUFFLE_SCALES=1
export VLLM_USE_AITER_TRITON_GEMM=1
 
vllm serve openai/gpt-oss-120b \
--tensor-parallel 1 \
--no-enable-prefix-caching --disable-log-requests \
--compilation-config '{"compile_sizes": [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 4096, 8192], "full_cuda_graph": true}' \
--block-size 64
```


### Step 5 : Send an Inference Request to Server

In a new terminal (attached to the same container or host network), send
a test prompt:

```shell
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openai/gpt-oss-120b",
        "prompt": "The future of AI is",
        "max_tokens": 100,
        "temperature": 0
}'
```

## How to Run Fine-tuning on gpt-oss using AMD Instinct MI300X and MI355X GPUs

With a few simple steps, you can efficiently fine tune the gpt-oss model on AMD Instinct MI300X and MI355X GPUs.


### Step 1. Pull the PyTorch Docker Container

**For MI300X GPU**

Use the Docker command below to pull the docker image.

```bash
docker pull rocm/pytorch-training:v25.6
``` 


**For MI355X GPU**

Use the docker command below to pull the docker image.

```bash

docker pull rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha
```

### Step 2. Launch the Docker Container

**For MI300X GPU**

To launch the docker container use the command below.

```bash
docker run -it \
  --device /dev/dri \
  --device /dev/kfd \
  --network host \
  --ipc host \
  --group-add video \
  --security-opt seccomp=unconfined \
  -v /home/USERNAME/:/workspace/ \
  --name YOUR_DOCKER_NAME \
  rocm/pytorch-training:v25.6
```

**For MI355X GPU**

To launch the docker container use the command below.

```bash
docker run -it \
  --device /dev/dri \
  --device /dev/kfd \
  --network host \
  --ipc host \
  --group-add video \
  --security-opt seccomp=unconfined \
  -v /home/USERNAME/:/workspace/ \
  --name YOUR_DOCKER_NAME \
  rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35X_alpha
```
**Note (for MI300X & MI355X):**

1.   If you wish to download the model locally and mount it into the container, first complete Step 3, then in Step 2 replace `/home/USERNAME/` with the path to your downloaded model directory so that it mounts at `/workspace/`; if you do not require a local mount, simply omit the `-v /home/USERNAME/:/workspace/` flag from your Docker command.

2.   Replace `YOUR_DOCKER_NAME` with a custom name of your choice.


### Step 3. Download the 20B MoE Model Checkpoint

Download OpenAI model through Hugging Face platform.

```bash
# Log in to Hugging Face
huggingface-cli login
# Enter your HF token when prompted

# Download the OpenAI official checkpoint
huggingface-cli download HUGGING_FACE_OPENAI_MODEL_LINK --local-dir ./models/MODEL_NAME # MODEL_NAME can be changed, and checkpoint can be downloaded from Hugging Face
```

**Note:**

1.  For step 3 all the commands are the same for MI300X and MI355X.

2.  `MODEL_NAME` with `model_path` needs to be set in the LoRA fine-tuning
    script (`run_peft_lora_openai.sh` - Step 4)

### Step 4. Clone PEFT Setup, Upgrade Transformers/Accelerate and Run LoRA Script

To execute the OpenAI MoE model using the Hugging Face PEFT LoRA framework use the below command.

```bash
cd /workspace/

# Clone repository
git clone https://github.com/AMD-AIG-AIMA/HF_PEFT_GPT_OSS.git
cd HF_PEFT_GPT_OSS
# For MI300, upgrade libraries using the following command
bash requirements_MI300.sh 
# For MI355, upgrade libraries using the following command
bash requirements_MI355.sh

# Run the LoRA fine-tuning script
bash run_peft_lora_openai.sh # Please make sure to set the model_name_or_path (MODEL_NAME) in the script file.
```

By default, this runs on a single node with eight GPUs. Modify
parameters in the run_peft_lora_openai.sh script as needed.

Please refer to the
[HF_PEFT_GPT_OSS](https://github.com/AMD-AIG-AIMA/HF_PEFT_GPT_OSS.git)
for more details and troubleshooting.
## Summary

AMD open ecosystem with cloud to client hardware, ROCm/Ryzen AI software
stack together with contributions from the open source community (vLLM,
Hugging face PEFT, and PyTorch) enables developers to deploy gpt-oss 20b
and 120b models from OpenAI seamlessly on Day 0. Day 0 support allows
developers to easily deploy models out of the box and start
experimenting without delay in post launch optimizations. This milestone
empowers developers to unlock the full potential of the AMD ecosystem
across cloud to client, reinforcing our commitment to providing
everything developers need to succeed, including robust open source
community collaboration. In this blog, we walked you through the steps for fine-tuning and running inference with OpenAI’s gpt-oss models on AMD hardware using ROCm, vLLM, and Hugging Face tools. With day 0 support for OpenAI models, we are
excited to see how the community will leverage these innovative capabilities.

## Acknowledgements

We extend our sincere thanks to the kernel team — Lukasz Burzawa, Mehmet Cagri Kaymak, Shao-Chun Lee, Ali Zaidy, and Vinayak Gokhale — and the compiler team — Kyle Wang, Jungwook Park, and Lei Zhang — for their outstanding contributions in enabling OpenAI models on AMD hardware. Their hard work, expertise, and dedication were instrumental to publishing this blog.
 
## Additional Resources 

To get started, explore the following resources:

For AMD GPUs,

-   [AMD Developer
    Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html)
-   [ROCm™ AI Developer
    Hub](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai.html)
-   [AMD ROCm™ Blogs](https://rocm.blogs.amd.com/)
-   [Tutorials for AI developers --- Tutorials for AI developers
    5.0](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/)
-   [AMD Radeon™ Developer Panel - AMD
    GPUOpen](https://gpuopen.com/rdp/)

For AMD Ryzen AI processors,

-   [AMD Ryzen™ AI
    Software](https://www.amd.com/en/developer/resources/ryzen-ai-software.html)

<!-- markdownlint-restore -->
## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
