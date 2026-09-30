---
blogpost: true
blog_title: "AIM: Unified User Experience from Profile Discovery to Deployment"
date: "30 Sep 2026"
author: "Rasmus Larsson, Emelie Wahlstrom"
thumbnail: 'aims-deployment-exp-thumbnail.png'
tags: "AI/ML"
category: "Applications & models"
target_audience: "AI Practitioners, Data Scientists, AI Engineers, ML Engineers, AI Scientists"
key_value_propositions: "Learn how to utilize AIMs across multiple accelerators"
language: English
myst:
    html_meta:
        "author": "Rasmus Larsson, Emelie Wahlstrom"
        "description lang=en": "Learn how AIMs offer a unified user experience as you explore and deploy them across AMD Instinct, Radeon PRO, and EPYC."
        "keywords": "AIM, AIMs, AMD Inference Microservice, AI"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs, Radeon Graphics, EPYC Server Processors"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Industry Applications & Use Cases"
        "amd_blog_authors": "Rasmus Larsson, Emelie Wahlstrom"
---

<!---
Copyright (c) 2026 Advanced Micro Devices, Inc. (AMD)

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

# AIM: Unified User Experience from Profile Discovery to Deployment

This post is a continuation of the [Multi-Accelerator Support for AIMs and AMD Solution Blueprints](https://rocm.blogs.amd.com/software-tools-optimization/eai-hw-support/README.html) blog, which introduced (i) multi-accelerator coverage across AMD Instinct™ GPUs, Radeon™ PRO GPUs, and EPYC™ CPUs, and (ii) walked through the Document Summarization Solution Blueprint. **Here we focus on [AIMs](https://enterprise-ai.docs.amd.com/en/latest/aims/overview.html) specifically** and show how they deliver a unified user experience from Radeon to Instinct, i.e., similar profile discovery commands, container workflow, and OpenAI-compatible API regardless of the accelerator.

We will demonstrate this by running AIMs on a Radeon GPU, EPYC CPUs, a single Instinct GPU, and an 8-GPU Instinct node. This walkthrough covers three steps:

1. **AIM containers**: How AIM profile selection works
2. **AIM exploration**: `dry-run` and `list-profiles`
3. **Docker deployment**: Start the server and call the OpenAI-compatible API

## AMD Inference Microservices - AIMs

**AIMs** are standardized inference microservices for serving AI models on AMD hardware. They are distributed as Docker images, which makes them easy to deploy and manage. Each AIM ships with predefined [**profiles**](https://github.com/amd-enterprise-ai/aim-build/blob/main/docs/aim_architecture.md#35-profile-example): inference engine configurations for specific accelerators, precisions, tensor-parallel layouts, and latency or throughput targets.

As shown in Figure 1, AIMs abstract away the complexities involved in configuring and serving AI models by providing a [mechanism](https://github.com/amd-enterprise-ai/aim-build/blob/main/docs/aim_architecture.md#4-aim-runtime--command-execution) that automatically selects runtime parameters based on the user’s input, hardware, and model specifications.

```{image} ./images/AIM-DEPLOYMENT.png
:label: aim-deployment-sequence
:alt: AIM automated deployment sequence
:width: 40%
:align: center
:class: dark-light
```

<p style="text-align:center">
<em>Figure 1: AIM automated deployment sequence.</em>
</p>

In practice, the high-level steps look like this:

1. **Initialize runtime** and **detect accelerators**
   - Load and validate configurations.
   - Example: AIM detects one AMD Instinct MI300X.
2. **Select a profile** and **generate the launch command**
   - Profiles are predefined configurations for specific models and hardware.
   - Selection is automatic, based on inputs such as:
     - Model (for example `meta-llama/Llama-3.1-8B-Instruct`)
     - Precision (for example `auto`, `fp16`, ...)
     - Engine (for example `vllm`)
     - Metric (`latency` or `throughput`)
     - Detected accelerator count (for example `1`, `2`, `4`, `8` GPUs)
     - Detected accelerator model (for example `MI300X`, `R9700`, ...)
   - It is possible to bypass automatic selection and specify a particular profile.
   - See the [selection algorithm](https://github.com/amd-enterprise-ai/aim-build/blob/main/docs/aim_architecture.md#34-selection-algorithm-overview) for more information.
3. **Launch the inference engine**
   - AIM produces the command and environment variables using the selected profile to start the server.
4. **Load model weights**
   - AIM supports e.g. flexible [model caching strategies](https://github.com/amd-enterprise-ai/aim-build/blob/main/docs/aim_architecture.md#6-model-caching-support).
5. **Deployment completed**
   - AIM exposes an [OpenAI-compatible API](https://platform.openai.com/docs/api-reference/introduction) for LLMs.

To illustrate this sequence, the rest of this post explores the AIM container with Docker. While we explore the container, be aware that there are multiple ways to deploy an AIM depending on your use case. See the following resources for more information:

- [AIM Container Technical Architecture](https://github.com/amd-enterprise-ai/aim-build/blob/main/docs/aim_architecture.md)
- [AIM Deployment](https://enterprise-ai.docs.amd.com/en/latest/aims/deployment_overview.html)
- [AIM Engine](https://enterprise-ai.docs.amd.com/en/latest/aim-engine/README.html)
- [AMD AI Workbench](https://enterprise-ai.docs.amd.com/en/latest/workbench/overview.html)

## Prerequisites

This post was validated separately on an AMD Instinct MI300X cluster, an AMD EPYC 9965 cluster, and an AMD Radeon PRO R9700 cluster. Also ensure that your system meets the following requirements:

- AMD GPU with ROCm support, required for GPU deployment. See the [accelerator support](https://enterprise-ai.docs.amd.com/en/latest/aims/accelerator_support.html) page for the host ROCm version for your accelerator.
- Docker installed, with device access on GPU hosts (`/dev/kfd` and `/dev/dri`).

## Choosing the AIM

This guide uses the following models:

| Hardware | Model | Container image |
| --- | --- | --- |
| Radeon | `Qwen/Qwen3.5-9B` | `amdenterpriseai/aim-radeon-qwen-qwen3-5-9b:0.12.0-preview` |
| EPYC | `Qwen/Qwen3.5-9B` | `amdenterpriseai/aim-epyc-qwen-qwen3-5-9b:0.13.0` |
| Instinct | `openai/gpt-oss-120b` | `amdenterpriseai/aim-openai-gpt-oss-120b:0.11.1` |

*Version tags (for example `0.11.1`) change with new AIM releases.*

For a high-level view of which AIMs are available for which hardware, see the [accelerator support](https://enterprise-ai.docs.amd.com/en/latest/aims/accelerator_support.html) table.

To use a different model:

- Open the [AIM catalog](https://enterprise-ai.docs.amd.com/en/latest/aims/catalog/models.html).
- Locate the model for your hardware.
- Open the Docker Hub link ([example](https://hub.docker.com/r/amdenterpriseai/aim-openai-gpt-oss-120b/tags)) and copy the image name.
- Open the **Technical specification** link in the catalog. That page lists available profiles for the specific model ([example](https://enterprise-ai.docs.amd.com/en/latest/aims/docs-aim/instinct/openai/gpt-oss-120b/README.html#model-specific-aim)). **To follow this guide**, confirm the model has an "optimized" or "preview" profile for your hardware (see Figure 2 for an example). If it does not, pick another model from the catalog.

```{image} ./images/AIM-PROFILES.png
:label: aim-profiles-spec
:alt: Subset of AIM profiles for gpt-oss-120b
:width: 80%
:align: center
:class: dark-light
```

<p style="text-align:center">
<em>Figure 2: Subset of AIM profiles for gpt-oss-120b.</em>
</p>

## Docker Profile Discovery

To illustrate the deployment sequence in action, we will use two AIM inspection commands:

- [`dry-run`](https://github.com/amd-enterprise-ai/aim-build/blob/main/docs/cli.md#dry-run-dry-run): shows which profile AIM would select and the exact engine command it would run
- [`list-profiles`](https://github.com/amd-enterprise-ai/aim-build/blob/main/docs/cli.md#list-profiles-list-profiles): lists and categorizes all available profiles by their compatibility with the current configuration.

This helps you understand which profiles are available and why certain profiles may or may not be selected. **Neither starts a server**, so both are safe to run before you commit to a deployment.

### Explore the Profile Selection

Let's begin with the `dry-run` command. The command shape is similar on every platform. What changes is the container image (and, for AMD Instinct 1 vs 8 GPUs, how many accelerators AIM detects). Pay attention to the detected hardware, accelerator count (e.g. detected GPUs), engine arguments, and environment variables — AIM fills those in for you.

Command:

::::{tab-set}

:::{tab-item} Radeon

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  amdenterpriseai/aim-radeon-qwen-qwen3-5-9b:0.12.0-preview \
  dry-run
```

Truncated output:

```yaml
profile:
  aim_id: Qwen/Qwen3.5-9B
  metadata:
    accelerator_count: 1
    accelerator_model: R9700
    accelerator_type: gpu
    metric: latency
    precision: bf16
    ...
  engine_args:
    tensor-parallel-size: 1
    ...
  env_vars:
    FLASH_ATTENTION_TRITON_AMD_ENABLE: 'TRUE'
    ...

...

exec python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-9B ....

```

:::

:::{tab-item} Instinct (1x)

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  amdenterpriseai/aim-openai-gpt-oss-120b:0.11.1 \
  dry-run
```

Truncated output:

```yaml
profile:
  aim_id: openai/gpt-oss-120b
  metadata:
    accelerator_count: 1
    accelerator_model: MI300X
    accelerator_type: gpu
    metric: latency
    precision: fp4
    ...
  engine_args:
    tensor-parallel-size: 1
    ...
  env_vars:
    VLLM_ROCM_USE_AITER: '1'
    ...

...

exec python -m vllm.entrypoints.openai.api_server --model openai/gpt-oss-120b ....

```

:::

:::{tab-item} Instinct (8x)

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  amdenterpriseai/aim-openai-gpt-oss-120b:0.11.1 \
  dry-run
```

Truncated output:

```yaml
profile:
  aim_id: openai/gpt-oss-120b
  metadata:
    accelerator_count: 8
    accelerator_model: MI300X
    accelerator_type: gpu
    metric: latency
    precision: fp4
    ...
  engine_args:
    tensor-parallel-size: 8
    ...
  env_vars:
    VLLM_ROCM_USE_AITER: '1'
    ...

...

exec python -m vllm.entrypoints.openai.api_server --model openai/gpt-oss-120b ....

```

:::

:::{tab-item} EPYC

```bash
docker run --rm \
  amdenterpriseai/aim-epyc-qwen-qwen3-5-9b:0.13.0 \
  dry-run
```

Truncated output:

```yaml
profile:
  aim_id: Qwen/Qwen3.5-9B
  metadata:
    accelerator_count: 188
    accelerator_model: EPYC_9965
    accelerator_type: cpu
    metric: latency
    precision: bf16
    ...
  engine_args:
    enable-chunked-prefill: true
    ...
  env_vars:
    VLLM_CPU_OMP_THREADS_BIND: auto
    ...

...

exec python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-9B ....

```

:::

::::

As you can see, the profile configuration changes depending on the model and the hardware. Latency is picked as the performance metric by default. For a technical guide to the **selection algorithm**, see [algorithm overview](https://github.com/amd-enterprise-ai/aim-build/blob/main/docs/aim_architecture.md#34-selection-algorithm-overview).

AIM also supports customizing the deployment with [environment variables](https://enterprise-ai.docs.amd.com/en/latest/aims/docker_deployment.html#customizing-deployment-with-environment-variables) and [custom profile configurations](https://enterprise-ai.docs.amd.com/en/latest/aims/custom_profiles.html) that extend beyond the built-in predefined profiles.

### List Profiles

Use `list-profiles` to see every profile available for the model. Here is an example for a single AMD Instinct MI300X GPU. Truncated output is shown in Figure 3:

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  amdenterpriseai/aim-openai-gpt-oss-120b:0.11.1 \
  list-profiles
```

```{image} ./images/AIM-LIST-PROFILES.png
:label: aim-list-profiles
:alt: list-profiles output for gpt-oss-120b
:width: 80%
:align: center
:class: dark-light
```

<p style="text-align:center">
<em>Figure 3: Subset of AIM profiles for gpt-oss-120b.</em>
</p>

The output lists and categorizes all available profiles by their compatibility with the current configuration. Each row shows the target accelerator, precision, inference engine, tensor parallelism (TP), performance metric, and profile type (`optimized`, `preview`, ..). The **Compatibility** column shows whether a profile matches your current configuration; mismatches (for example `accelerator_mismatch` or `metric_mismatch`) explain why others are excluded.

## Deployment

The following is a small example. See [Docker deployment guide](https://enterprise-ai.docs.amd.com/en/latest/aims/docker_deployment.html#docker-deployment) for more detail, including customization.

Note: If you are running larger models in multi-GPU environments, read the following [guide](https://enterprise-ai.docs.amd.com/en/latest/aims/docker_deployment.html#running-larger-models-in-multi-gpu-environments) first.

Once you have validated the profile selection, start the inference server. The following command is an example for one AMD Instinct MI300X GPU. It runs the container with the profile that the container automatically selects on the hardware it detects:

```bash
docker run \
  --device=/dev/kfd --device=/dev/dri \
  -p 8000:8000 \
  amdenterpriseai/aim-openai-gpt-oss-120b:0.11.1
```

Watch the container logs for profile selection, then for the inference server to start. Truncated output:

```text
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

On first start:

1. AIM detects your GPU and selects a profile automatically.
2. Model weights are downloaded from Hugging Face.
3. The inference engine starts and listens on port `8000`.

Send a completion request (if you use another model, change the `model` field):

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "prompt": "Once upon a time,",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

Example output (truncated):

```json
{
  "text": " in a small village near a dense forest, there lived a wise ..."
}
```

## Summary

In this blog, we demonstrated the flexibility and unified user experience of AIMs. You saw profile discovery across platforms, and a Docker deployment example on Instinct. AIM detects your accelerator, selects a profile automatically, and adjusts engine arguments and environment variables for each platform without manual tuning. To explore further, browse the [AIM catalog](https://enterprise-ai.docs.amd.com/en/latest/aims/catalog/models.html) for models and technical specifications to find the model for your use case. Read the [Docker deployment guide](https://enterprise-ai.docs.amd.com/en/latest/aims/docker_deployment.html) for customization options beyond the defaults shown here.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, AMD Instinct, AMD Radeon, AMD EPYC, ROCm and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
