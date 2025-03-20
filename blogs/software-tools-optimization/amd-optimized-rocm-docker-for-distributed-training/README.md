---
blogpost: true
blog_title: "Optimized ROCm Docker for Distributed AI Training"
date: 13 Mar 2025
author: 'Yao Fu, Anshul Gupta'
thumbnail: 'docker_img.png'
tags: PyTorch, LLM, GenAI, AI/ML, Fine-Tuning, Optimization, Performance
category: Software tools & optimizations
target_audience: ai enthusiast and ai developers
key_value_propositions: Updated docker images for Pytorch and MegatronLM with finetuning capability ,FP8 datatype support, single node  performance boost, bug fixes and updated benchmarking scripts for training workflows.
language: English
myst:
    html_meta:
        "author": "Yao Fu, Anshul Gupta"
        "description lang=en": "AMD updated Docker images incorporate torchtune finetuning, FP8 support, single node  performance boost, bug fixes & updated benchmarking for stable, efficient distributed training"
        "keywords": "Docker, PyTorch , Megatron, Training Docker, AMD GPU"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Applications and models"
        "amd_developer_type": "ML/AI Developer"
        "amd_deployment": "Servers"
        "amd_product_type": "Development Tools"
        "amd_developer_tool": "ROCm Software, Open-Source Tools"
        "amd_applications": "Large Language Model (LLM)"
        "amd_industries": "Data Center"
        "amd_blog_releasedate": Fri Mar 07, 12:00:00 PST 2025
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

# Optimized ROCm Docker for Distributed AI Training

This blog will introduce you to the updated AMD Docker image, specifically built and optimized for distributed training. As you will see, the optimized AMD ROCm Docker image makes training large AI models faster and more efficient. It includes updates such as better fine-tuning tools, improved performance for multi-GPU setups, and support for FP8 precision, which helps speed up training while using less memory, and can provide you with an overall smoother and more efficient training experience on popular models such as Flux and Llama 3.1 running on AMD GPUs.

The blog will provide you with an in-depth overview of the recent updates focusing on enhanced scalability and efficiency within training pipelines. We will discuss the updated Docker images incorporating torchtune finetuning capability ,FP8 datatype support, single node performance boost, bug fixes and updated benchmarking scripts, ensuring a stable, consistent and reliable environment for managing complex distributed training workflows.

## The PyTorch and Megatron-LM Training Dockers

The ROCm™ PyTorch [training docker release (v25.3)](https://hub.docker.com/r/rocm/pytorch-training) contains two updated Docker images for distributed training: PyTorch and Megatron-LM. Let's briefly discuss their major features:

### PyTorch Training Docker

The ROCm [Pytorch Training docker container](https://hub.docker.com/r/rocm/pytorch-training) provides a prebuilt, optimized environment for fine tuning, pre-training a model on AMD Instinct™ MI300X and MI325X GPUs.

**<u>Key Highlights</u>**

- Updated benchmarking scripts for pre-training popular models such as Flux, Llama 3.1 8B, and Llama 3.1 70B.

- Support for torchtune finetuning capability to enable full weight Low Ranking Adaption( LoRA) finetuning.

- Added Huggingface Accelerate and torchtitan libraries to optimize the training with FSDP.

- Integrated with MAD system for regression testing, documentation and example publication.

### Megatron-LM Training Docker

The ROCm [Megatron-LM training docker container](https://hub.docker.com/r/rocm/megatron-lm) is designed to enable efficient training of large-scale language models on AMD Instinct MI300X and MI325X GPUs.
AMD Megatron-LM delivers enhanced scalability, improved performance and resource utilization for AI workloads. It is purpose-built to support models like Meta’s Llama 2, Llama 3, and Llama 3.1, enabling developers to train next-generation AI models with greater efficiency.

 **<u>Key Highlights</u>**

- The Megatron-LM focused Docker image is built on top of the Pytorch 6.1 release training Docker.

- Supports multimode training, DeepseekV2-Lite and FP8 datatypes.
  
## Finetuning with Torchtune

Using an updated Docker equipped with optimized torchtune, users can have a streamlined out-of-box experience of finetuning, both full weights and LoRA, from the AMD Pytorch training docker release. The docker image is available in the [torchtune dockerhub repository.](https://github.com/AMD-AIG-AIMA/torchtune) Below is an example of using the PyTorch training Docker release v25.3 to run finetuning.

- Launch PyTorch Training Docker
  
```shell
docker pull rocm/pytorch-training:v25.3
docker run -it --device /dev/dri --device /dev/kfd --network host --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v $HOME:$HOME -v $HOME/.ssh:/root/.ssh --shm-size 64G --name training_env rocm/pytorch-training:v25.3
```

- Clone ROCm MAD benchmarking repo

```shell
git clone https://github.com/ROCm/MAD
cd MAD/scripts/pytorch-train
```

- Download models and dataset
  
```shell
# pass your HF_TOKEN
export HF_TOKEN=$your_personal_hf_token
./pytorch_benchmark_setup.sh
```

- Lama 3.1 70B Full weight finetuning with Wikitext database

```shell
./pytorch_benchmark_report.sh -t finetune_fw -p BF16 -m Llama-3.1-70B
```

With larger HBM memory, MI300X GPUs can execute finetuning with larger batch sizes and do not rely on CPU offload, both benefit the training throughputs. This resulted in about **1.09x** better unmasked token/sec/GPU in comparing to an equivalent H100 solution as shown in Figure 1 below. We observed that H100 runs out of memory in scenarios when CPU offload is turned off or for batch size 64 or larger.

```{figure} ./images/FIGURE1.png
:align: center
:alt: Scaling performance
Figure 1: Full Fine-tuning Unmasked Tokens/sec/gpu w/ variable sample length from WikiText [^1]
```

## Low Precision FP8 Multi-node Scaling Performance

FP8 support is now integrated into the Megatron-LM docker container. Using the latest optimizations and configurations, found in the Megatron-LM docker container, our testing demonstrates 97% near-linear multi-node scaling, as illustrated in Figure 2. In other words, these benchmarks demonstrate minimal impacts of multi-node distributed training on GPU performance.

```{figure} ./images/FIGURE2.png
:align: center
:alt: Scaling performance
Figure 2. Llama 3.1 8B FP8 Training TFLOPS[^2]
```

## MoE support with Megatron-LM training

AMD also offers support for Mixture of Experts (MoE) models—a class of deep learning architectures that leverage multiple specialized sub-models, or "experts." MoE models dynamically route input data to the most relevant experts, enabling the scaling of model capacity while maintaining computational efficiency by activating only a subset of experts per input.

We observe that a single node (8x MI300X) delivers about 1.29x better performance vs. single node(8x H100) on DeepSeekV2-Lite training. We also now have an added advantage of being able to train the full model without checkpoint recompute to support larger micro batch size as shown in Figure 3.

```{figure} ./images/FIGURE3.png
:align: center
:alt: Scaling performance
Figure 3: DeepSeekV2-Lite Training Tokens/s/GPU[^3]
```

## Summary

AMD is committed to releasing an improved performance dockers at a regular cadence. This blog lays the foundation for an improved optimized docker for training with AMD GPUs. We are excited to continue and provide members of the open-source community with an opportunity to try our updated docker image for their training workloads.
Stay tuned for more updates in our next blog as we’re working on implementing performance improvements, additional features and enhancing the user experience for Docker deployments moving forward.

## Additional Resources

Torchtune (<https://github.com/AMD-AIG-AIMA/torchtune>)

PyTorch training benchmark (<https://github.com/ROCm/MAD/blob/develop/benchmark/pytorch_train/README.md>)

MoE support with Megatron-LM docker using DeepSeekV2-Lite (<https://github.com/ROCm/MAD/blob/develop/benchmark/megatron_lm/README.md#deepseek-v2-lite-training-procedure>)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

<!-- markdownlint-disable -->
[^1]: MI300-078 On average, a system configured with 8x AMD Instinct MI300X GPUs shows up to 1.09x better performance comparing to a similarly configured system with 8x NVIDIA H100 GPUs, in full fine-tuning workloads with variable input sequence lengths. Testing done by AMD on 02/014/2025, results may vary. \
    SYSTEM CONFIGURATION: \
    AMD Instinct™ MI300X platform: System Model: Supermicro GPU A+ Server AS - 8125GS-TNMR2 \
    CPU: 2x AMD EPYC 9654 96-core Processor (2 sockets, 96 cores per socket, 2 threads per core) \
    NUMA Config: 2 NUMA node per socket \
    Memory: 2.3 TiB (24 DIMMs, 4800 mts, 96 GiB/DIMM) \
    Disk: Root drive + Data drive combined: 2x 960GB Samsung MZ1L2960HCJR-00A07 4x 3.84TB Samsung MZQL23T8HCLS-00A07 \
    GPU: 8x AMD MI300X 192GB HBM3 750W \
    Host OS: Ubuntu 22.04.5 LTS with Linux kernel 5.15.0-122-generic. \
    System BIOS: 5.27 \
    Host GPU Driver: 6.3.0 ROCm 6.3 (Pre-release) \
    Docker version: rocm/pytorch-training:v25.3 \
    \
    NVIDIA H100 platform: System Model: Supermicro AS -8125GS-TNHR \
    CPU: 2x AMD EPYC 9654 96-Core Processor (2 Sockets, 96 cores per socket, 2 Threads per core) NUMA Config: 1 NUMA node per socket Memory: 2304 GB (24 DIMMS, 4800 mts, 96 GB/DIMM) \
    Disk: Data drives: 8x 7 TiB INTELSSDPF2KX076T1NVMe SSDs \
    Root drive: 1.75 TiB Micron MTFDDAK1T9TDS-1AW1ZA \
    GPU: 8x NVIDIA H100 80GB HBM3 700W \
    Host OS: Ubuntu 22.04.5 LTD with Linux kernel titan 6.8.0-51-generic \
    Host GPU Driver:535.183.01 \
    Firmware System: BIOS 2.1 \
    Docker version:nvcr.io/nvidia/pytorch:25.01-py3
[^2]: MI300-079 On average, a system configured with an AMD Instinct MI300X GPU shows linear scalability of performance on Llama 3.1 8B model in FP8 precision, with multi-node training. Testing done by AMD on 02/014/2025, results may vary. \
    SYSTEM CONFIGURATION: \
    AMD Instinct™ MI300X platform System Model: Supermicro GPU A+ Server AS - 8125GS-TNMR2 \
    CPU: 2x AMD EPYC 9654 96-core Processor (2 sockets, 96 cores per socket, 2 threads per core) \
    NUMA Config: 2 NUMA node per socket \
    Memory: 2.3 TiB (24 DIMMs, 4800 mts, 96 GiB/DIMM) \
    Disk: Root drive + Data drive combined:2x 960GB Samsung MZ1L2960HCJR-00A07 4x 3.84TB Samsung MZQL23T8HCLS-00A07 \
    GPU: 8x AMD MI300X 192GB HBM3 750W \
    Host OS: Ubuntu 22.04.5 LTS with Linux kernel 5.15.0-122-generic. \
    System BIOS: 5.27 \
    Host GPU Driver: 6.3.0 ROCm 6.3 (Pre-release) \
    Docker version: rocm/megatron-lm:v25.3
[^3]: MI300-080 On average, a system with 8x AMD Instinct MI300X GPUs shows up to  1.29x better performance comparing to a similarly configured system with 8x NVIDIA H100 GPUs, in the DeepSeekV2-Lite model training benchmark. Testing done by AMD on 02/014/2025, results may vary. \
    SYSTEM CONFIGURATION: \
    AMD Instinct™ MI300X platform System Model: Supermicro GPU A+ Server AS - 8125GS-TNMR2 \
    CPU: 2x AMD EPYC 9654 96-core Processor (2 sockets, 96 cores per socket, 2 threads per core) \
    NUMA Config: 2 NUMA node per socket \
    Memory: 2.3 TiB (24 DIMMs, 4800 mts, 96 GiB/DIMM) \
    Disk: Root drive + Data drive combined:2x 960GB Samsung MZ1L2960HCJR-00A07 4x 3.84TB Samsung MZQL23T8HCLS-00A07 \
    GPU: 8x AMD MI300X 192GB HBM3 750W \
    Host OS: Ubuntu 22.04.5 LTS with Linux kernel 5.15.0-122-generic. \
    System BIOS: 5.27 \
    Host GPU Driver: 6.3.0 ROCm 6.3 (Pre-release) \
    Docker version: rocm/megatron-lm:v25.3 \
    \
    NVIDIA H100 platform: \
    System Model: Supermicro AS -8125GS-TNHR \
    CPU: 2x AMD EPYC 9654 96-Core Processor (2 Sockets, 96 cores per pocket, 2 Threads per core) \
    NUMA Config: 1 NUMA node per socket \
    Memory: 2304 GB (24 DIMMS, 4800 mts, 96 GB/DIMM) \
    Disk: Data drives: 8x 7 TiB INTEL SSDPF2KX076T1 NVMe SSDs \
    Root drive: 1.75 TiB Micron MTFDDAK1T9TDS-1AW1ZA \
    GPU: 8x NVIDIA H100 80GB HBM3 700W \
    Host OS: Ubuntu 22.04.5 LTS with Linux kernel titan 6.8.0-51-generic \
    Host GPU Driver:535.183.01 \
    Firmware System: BIOS 2.1 \
    Docker version: nvcr.io/nvidia/pytorch:24.10-py3  
<!-- markdownlint-enable -->
