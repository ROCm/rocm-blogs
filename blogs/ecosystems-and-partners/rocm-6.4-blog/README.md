---
blogpost: true
blog_title: "ROCm 6.4: Breaking Barriers in AI, HPC, and Modular GPU Software"
date: 11 Apr 2025
author: 'Jayacharan Kolla, Aditya Bhattacharji, Farshad Ghodsian, Saad Rahim, Marco Grond, Ronnie Chatterjee'
thumbnail: 'rocm-6.4-blog.png'
tags: AI/ML, HPC, Computer Vision, Profiling, Kubernetes
category: Ecosystems and Partners
target_audience: AI / HPC Developers, System Administrators, Researchers
key_value_propositions: The blog talks about the highlights of our latest ROCm 6.4 release.
language: English
myst:
    html_meta:
        "author": "Jayacharan Kolla, Aditya Bhattacharji, Ronnie Chatterjee, Saad Rahim, Farshad Ghodsian"
        "description lang=en": "Explore ROCm 6.4's key advancements: AI/HPC performance boosts, enhanced profiling tools, better Kubernetes support and modular drivers, accelerating AI and HPC workloads on AMD GPUs."
        "keywords": "ROCm, Model Optimizations, Containers, Kubernetes, Profilers, Software Packaging"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_developer_type": "ML/AI Developer, Application Developer, Software Developer, HPC Developer, Algorithm Developer, Data & Research Scientists"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_product_type": "Software & Applications"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "Computer Vision"
        "amd_blog_topic_categories": 'Software & Ecosystem'
        "amd_blog_releasedate": Thu Apr 10, 12:00:00 PST 2025
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

# ROCm 6.4: Breaking Barriers in AI, HPC, and Modular GPU Software

In the rapidly evolving landscape of high-performance computing and artificial intelligence, innovation is the currency of progress. AMD's ROCm 6.4 isn't just another software update—it's a leap forward that redefines the boundaries of what is possible for AI, developers, researchers, and enterprise innovators.

Imagine a GPU software stack that doesn't just incrementally improve performance, but dramatically transforms how we approach complex computational challenges. ROCm 6.4 does exactly that, delivering a suite of breakthrough optimizations that touch every critical aspect of AI and high-performance computing (HPC). From lightning-fast model training to revolutionary developer tools, this release is a testament to AMD's relentless pursuit of computational excellence.

Whether you're training massive AI models that push the limits of machine learning, optimizing intricate HPC workloads, or deploying cutting-edge Kubernetes clusters, ROCm 6.4 provides the technological fuel to accelerate your most ambitious projects.
In this blog you will learn how AMD ROCm 6.4 can help unlock next-gen performance for your AI, HPC, and Kubernetes workloads. The blog will show you all the highlights of the ROCm 6.4 release, covering its key advancements: AI/HPC performance boosts, enhanced profiling tools, better Kubernetes support and modular drivers, and
accelerating AI and HPC workloads on AMD GPUs.

## AI Frameworks & Ecosystem Enablement

With ROCm 6.4 AI developers can look forward to substantial performance improvements across cutting-edge machine learning frameworks and optimization tools.
The latest ecosystem updates introduce prebuilt, optimized Docker containers that streamline AI training and inference workflows, making advanced computational resources more accessible and efficient.

### AI Frameworks

#### PyTorch

- **Flex Attention Performance Gains**: ROCm 6.4 introduces **optimized Flex Attention**, delivering competitive performance with CUDA.
- **TopK Enhancements**: Achieves **3x faster inference**, leading to quicker responding LLMs while maintaining coherent, high-quality outputs.
- **SDPA Optimization**: Improves efficiency in scaled dot-product attention (SDPA), enabling **faster, memory-efficient LLM inference** in **long-context scenarios**.
- The PyTorch for ROCm training Docker image provides a prebuilt optimized environment for fine-tuning and pre-training a model on AMD Instinct MI300X accelerators supporting optimized Llama 3.1 (8B, 70B), Llama 2 (70B), and FLUX.1-dev.  
  - Access ROCm's Pytorch Training Docker and training resources: [Docker Container](https://hub.docker.com/r/rocm/pytorch-training), [User Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html), [Performance Numbers](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html#tabs-a8deaeb413-item-21cea50186-tab), [Performance Validation](https://github.com/ROCm/MAD/blob/develop/benchmark/pytorch_train/README.md).
  - To learn more about Pytorch for ROCm Training, read the [blog](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html).
- The Megatron-LM framework for ROCm is a specialized fork of the robust Megatron-LM, designed to enable efficient training of large-scale language models on AMD GPUs supporting optimized Llama 3.1 (8B, 70B), Llama 3 (8B, 70B), Llama 2 (7B, 70B), and DeepSeek-V2-Lite.
  - Access Megatron-LM Docker and training resources: [Docker Container](https://hub.docker.com/r/rocm/megatron-lm), [User Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html), [Performance Numbers](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html#tabs-a8deaeb413-item-21cea50186-tab), [Performance Validation](https://github.com/ROCm/MAD/blob/develop/benchmark/megatron_lm/README.md).
  - To learn more about training a model with Megatron-LM for ROCm refer to this [blog](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html).

#### JAX

- **RNN Support**: Enables efficient processing of sequential data, improving applications like speech recognition and time-series forecasting.

#### OpenXLA

- **MLIR Fusion Path**: Reduces **kernel execution overhead** and optimizes GEMM operations, enabling lower latency inference for improved AI model performance.
- **Cross Target GEMM Fusion**: Combines appropriate GEMM operations to optimize matrix multiplication on AMD GPUs.
- **Triton Autotuning**: Automatically identifies optimal configurations of Triton kernels leading to maximized utilization of Instinct hardware, resulting in improved out-of-box performance on AI workloads.

### Inference Solutions

#### vLLM

- **Day 0 Support for Google's Gemma 3**: Ensures seamless inference deployment for Google’s latest open-source model. Learn more about blog: [Deploying GEMMA3](https://rocm.blogs.amd.com/artificial-intelligence/deployingGemma-vllm/README.html)
- **INT4 Attention**: Enables **memory-efficient attention calculations**, making inference faster and more resource-efficient.
- Plug-and-play **LLM inference** with **stable (bi-weekly) & dev (weekly) containers** for Llama, Mistral, Mixtral, Qwen, Gemma, Cohere, and DeepSeek.
- **Stable Container Relevant Links**: [Docker Container](https://hub.docker.com/r/rocm/vllm), [User Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/vllm-benchmark.html?model=pyt_vllm_mixtral-8x7b), [Performance Numbers](https://github.com/ROCm/MAD/blob/develop/benchmark/vllm/README.md).
- **Development Build Container Relevant Links**: [Docker Container](https://hub.docker.com/r/rocm/vllm-dev), [User Guide](https://github.com/ROCm/vllm/blob/main/docs/dev-docker/README.md).
- Learn more about [Best practices for competitive inference optimization on AMD Instinct™ MI300X GPUs](https://rocm.blogs.amd.com/artificial-intelligence/LLM_Inference/README.html)

#### SGLang

- **Optimized DeepSeek R1 Support in SGLang**: Delivers state-of-the-art DeepSeek R1 performance on MI300X accelerators.
- **MLA Support**: **Parallel multi-head attention** enhances the model’s ability to capture complex patterns, critical for DeepSeek.
- **DeepGEMM Support**: **Optimized FP8 GEMM operations** deliver fast, lightweight compute performance across both dense and Mixture-of-Experts implementations.
- Optimized for DeepSeek R1, structured generation & agentic workflows, supporting Grok 1, Mixtral, and Llama 3.1 (8B, 70B, 405B).
- To simplify benchmarking and performance validation SGLang Resources: [Docker Container](https://hub.docker.com/r/lmsysorg/sglang/tags?name=rocm), [User Guide](https://github.com/sgl-project/sglang/blob/main/docs/references/amd.md)
- Learn more about SGLang: [SGLang](https://rocm.blogs.amd.com/artificial-intelligence/sglang/README.html), [Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)

### AITER

AITER is designed to accelerate critical AI operations—such as GEMM, Attention, and Mixture-of-Experts—by providing developers with drop-in, pre-optimized kernels, reducing the need for the time-consuming and complex process of manual kernel tuning. It delivers up to 17× speedups in decoder execution, 14× in Multi Headed Attention, and over 2× throughput in LLM inference (DeepSeek). Read more about [AITER](https://rocm.blogs.amd.com/software-tools-optimization/aiter:-ai-tensor-engine-for-rocm%E2%84%A2/README.html).

These enhancements directly translate to higher throughput, reduced latency, and lower compute costs for AI workloads running on Instinct GPUs.

## Math Libraries: Accelerating AI and HPC Computation

ROCm 6.4 brings major performance improvements in mixed precision, GEMM, sparse computation, and FFT optimizations, enabling faster training, reduced memory footprint, and better numerical accuracy.

### WMMA Enhancements

ROCm 6.4 brings major performance improvements to rocWMMA, enhancing GEMM efficiency for AI and HPC workloads. **Interleaved GEMM optimization** has been implemented to improve data locality, ensuring better memory access patterns and higher computational throughput. Additionally, we've introduced the **Stream-K algorithm** in [rocWMMA](https://github.com/ROCm/rocWMMA) which allows better utilization of compute resources, providing more consistent performance across various GEMM problems.

### Sparse Computation Enhancements

The **rocSPARSE library** sees key enhancements for sparse computation, enabling better performance for HPC applications that rely on sparse data structures. We've **Optimized Compressed Sparse Row (CSR) Matrix-Vector Multiplication** which accelerates sparse workloads by reducing memory overhead, benefiting large HPC simulations. Additionally, **expanded support for Sparse Matrix-Matrix Multiplication (SpMM)** improves efficiency for applications to process sparse matrices more effectively. These upgrades reduce computation time and enhance scalability, ensuring optimal performance for workloads requiring sparse data handling. Access the rocSPARSE repository [here](https://github.com/ROCm/rocSPARSE).

## Communication Libraries

ROCm 6.4 enhances GPU communication much more by enabling efficient data transfer mechanisms for AI and HPC workloads with enhanced intra-node and inter-node optimizations.

### rocSHMEM Intra-Node GPU Communication

- ROCm 6.4 introduces an intra-node communication feature within [rocSHMEM](https://github.com/ROCm/rocSHMEM), facilitating GPU-initiated data transfers. This implementation is designed for single-node jobs, eliminating network dependencies while ensuring seamless interoperability across different ROCm versions. This rocSHMEM conduit can provide a way for users to initiate communication operations from the GPU Kernel. This helps avoid kernel launch overhead and synchronization.

### Topology Optimization in RCCL

- RCCL is now optimized for a single-layer switch topology network, eliminating the need for Tier-2 spine switches and enabling efficient rail topology communication with only Tier-1 switches. This approach optimizes intra-node and inter-node communication, reducing network congestion, lowering latency, and improving scalability for large-scale AI and HPC workloads by leveraging the Infinity Fabric ( xGMI ) for inter-rail traffic. To learn more about XGMI, refer to this [blog](https://rocm.blogs.amd.com/software-tools-optimization/mi300x-rccl-xgmi/README.html).

## Developer Tools: Upgrades in ROCm Profiler Suite

ROCm 6.4 expands profiling capabilities, helping developers analyze bottlenecks and optimize performance.

- **Network Performance Profiling**: ROCm 6.4 introduces network performance profiling within the ROCm Systems Profiler, enabling users to analyze data transfer efficiency and bottlenecks in network traffic. Customers can now track **bytes sent and received** for network-related operations
- **OpenMP Offload Kernel Activity Monitoring with C++ Device Tracing Support:** This feature provides deeper insights into kernel execution, memory movement, and compute efficiency, allowing developers to analyze interactions between the host and device more effectively. By capturing fine-grained execution timelines for OpenMP offloaded kernels, users can now pinpoint bottlenecks, optimize data transfers, and improve overall workload efficiency.
- **VCN Engine Activity Profiling:** ROCm Systems Profiler introduces VCN (Video Core Next) engine activity monitoring, offering deeper profiling for video encoding and decoding workloads on AMD GPUs. The example below explains how Developers can now correlate rocDecode API Trace to the VCN Engine Activity.

```{figure} ./images/vcn.png
:align: center
:alt: VCN Tracing
Figure 1: Example of rocDecode API Trace & VCN Engine Activity
```

All ROCm profiler tools also come with support for MI325 GPUs, ensuring full compatibility with optimized profiling capabilities.

To understand more about ROCprofiler SDK, refer to this [blog](https://rocm.blogs.amd.com/software-tools-optimization/rocprofiler-sdk/README.html).

## Computer Vision: Enabling Advanced Media Processing

ROCm 6.4 strengthens **computer vision capabilities** with **new video decoding and audio processing features**.

- **VP9 Decode Support**: In addition to the existing support to decode the **HEVC**, **AVC**, and **AV1** video codecs, rocDecode has been extended to support the widely used, high compression **VP9** codec.
- **New Bitstream Reader**: rocDecode includes new functionality to parse **AVC/HEVC/AV1** elementary streams and **IVF** container files without the need for FFMPEG, accelerating the parsing and decoding of these formats. For other formats, FFMPEG is still used.
- **Audio Processing Support**: We've added support for **Spectrogram & Mel Filter Bank operators** to provide audio processing capabilities related to extracting different frequencies, which in turn can be used for speech-related models.

## Kubernetes (K8s): Seamless GPU Deployment

ROCm 6.4 enables new Kubernetes-based GPU management features that make it easier for enterprises to deploy, monitor, and maintain AMD GPUs in cloud and on-prem environments.

### AMD GPU Operator: Simplified Cluster Management

- With **automatic GPU scheduling and driver management**, workload deployment across Kubernetes clusters becomes seamless by eliminating manual configuration efforts. The **GPU Operator now supports Red Hat OpenShift (v4.16-v4.17) & Ubuntu 22.04/24.04**, ensuring broad compatibility across cloud and enterprise environments, making it easier for organizations to integrate AMD GPUs into their infrastructure.
- **Automatic driver upgrades & workload orchestration**, minimizing downtime.
- ROCm 6.4 comes with automatic GPU driver upgrades & workload orchestration, preventing potential performance issues and vulnerabilities. Features like **node cordon, drain, and reboot automation** allow smooth rolling updates with minimal downtime, ensuring uninterrupted workloads. This automation improves cluster reliability, making large-scale deployments easier to maintain.
- For organizations operating in air-gapped network environments, ROCm 6.4 enables **air-gapped and proxy deployments**, allowing GPU clusters to function efficiently. This is particularly valuable for government, enterprise, and cloud service providers that need to comply with strict security and data protection regulations.
- Learn more about [GPU Operator](https://github.com/ROCm/gpu-operator) from the [blog](https://rocm.blogs.amd.com/software-tools-optimization/gpu-operator-exporter/README.html).

### Device Metrics Exporter: Real-Time GPU Health Monitoring

- With **Prometheus-based telemetry**, customers can track critical GPU metrics such as **ECC errors, power consumption, and GPU/Memory utilization** in real time. This allows AI/HPC administrators and DevOps teams to detect anomalies early, ensuring system stability and minimizing unexpected failures.
- The ability to set custom health thresholds ensures that GPUs are automatically flagged or removed before they cause system instability. Customers can configure thresholds via **Kubernetes ConfigMaps**, tailoring monitoring to their specific workloads. This helps predict failures, improve uptime, and reduce costly downtimes in mission-critical AI applications.
- Now available as a standalone Debian package for broader adoption. Learn more about [Device Exporter](https://github.com/ROCm/device-metrics-exporter) from the [documentation](https://instinct.docs.amd.com/projects/device-metrics-exporter/en/latest/#amd-device-metrics-exporter).

## Runtime & Drivers & Packaging

### Software Modularity with the Instinct GPU Driver

ROCm 6.4 introduces the **Instinct GPU Driver**, decoupling the driver from the ROCm userspace for modular software management.

Key Benefits:

- **Independent updates**: Upgrade driver & ROCm separately for better stability & compatibility.
- **Flexible deployment**: Use Instinct driver with multiple ROCm toolkits, containers, or ISV applications.
- **Clearer versioning**: Clarifies **backward & forward compatibility** across software releases.

🔗 [Instinct Driver Docs](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/) | [ROCm Kernel Driver Source](https://github.com/ROCm/ROCK-Kernel-Driver)

Ensures seamless integration between different ROCm versions and driver updates, minimizing disruptions for developers and enterprises.

- Stable long-term support for **forward-backward compatibility** between userspace and driver for ±12 months, increased from the current ±6 months.
- No breaking changes, enabling smooth software transitions.

Read more about it in our [blog](https://rocm.blogs.amd.com/ecosystems-and-partners/instinct-gpu-driver/README.html).

## Summary

ROCm 6.4 marks a significant milestone in AMD’s journey toward democratizing high-performance computing and AI development. As we outline in this blog ROCm 6.4 release delivers sweeping performance improvements across major AI frameworks like PyTorch, Megatron-LM, JAX, and SGLang—enabling faster, memory-efficient LLM training and inference on AMD Instinct™ GPUs. Whether you're building next-gen AI applications, optimizing LLM inference, or managing containerized GPU clusters, ROCm 6.4 is built to accelerate your goals—efficiently, reliably, and at scale. Stay tuned for upcoming blogs on how to leverage some of these recently added features!
