---
blogpost: true
blog_title: "ROCm 7.14: TheRock Goes Production and Expands AMD's AI Software Platform"
date: "15 Jul 2026"
author: "Amy Wiebe, Mohammed Faraaz Mustafa, Anshul Gupta, Saad Rahim, Layla Frischman, Liam Berry"
thumbnail: 'ROCm-7.14-thumbnail.png'
tags: "Hardware, HPC, Installation, Optimization, Performance, AI/ML, Profiling, Fine-Tuning"
category: "Ecosystems and Partners"
target_audience: "All / Any"
key_value_propositions: "This blog is being used to highlight the important new features, changes, and general improvements implemented in ROCm 7.14"
language: English
myst:
    html_meta:
        "author": "Amy Wiebe, Mohammed Faraaz Mustafa, Anshul Gupta, Saad Rahim, Layla Frischman, Liam Berry"
        "description lang=en": "Explore what's new in ROCm 7.14: TheRock goes production, expanded hardware support, stronger AI frameworks, and enhanced profiling tools."
        "keywords": "Release, ROCm, Software, Version, AI/ML, HPC, Data Science, Libraries, Compilers, Toolchains, Computer vision, MIOpen, Profiling, Developer tools, Debugging, Communication, Math"
        "vertical": "Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Ecosystem and Partners"
        "amd_blog_hardware_platforms": "Instinct GPUs, Radeon Graphics"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training, Computer Vision, Data Science, Design, Simulation & Modeling, Edge Computing, Generative AI, Deploying AI at Scale"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Amy Wiebe, Mohammed Faraaz Mustafa, Anshul Gupta, Saad Rahim, Layla Frischman, Liam Berry"
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

# ROCm 7.14: TheRock Goes Production and Expands AMD's AI Software Platform

In this blog, you will explore the key updates and improvements that make the [ROCm 7.14](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html) release a major milestone in the evolution of AMD's open source AI software ecosystem. At the heart of this release is the production debut of TheRock, AMD's new automated, open-source build and release system that modernizes how ROCm is built, packaged, validated, and delivered.

Alongside this foundational change, ROCm 7.14 expands support for AMD's latest AI hardware, advances AI frameworks and inference software, strengthens communication and math libraries, enhances developer tooling, and simplifies deployment across client, enterprise, cloud, and data center environments.

## TheRock Reaches Production

This release marks a significant milestone with the first official release of TheRock, AMD's new automated, open-source build and release system for the ROCm software stack. The preview release train spanning versions 7.9 through 7.13 is complete, and ROCm 7.14 marks the start of our future production releases. Users on 7.2 and older are encouraged to migrate and follow the transition guide on [ROCm docs](https://rocm.docs.amd.com/en/docs-7.14.0/about/transition-guide-TheRock.html). TheRock packages the foundational components needed to run high-performance workloads on AMD GPUs, streamlining how ROCm is built and delivered going forward. TheRock introduces ROCm Core SDK, a base installation containing the essential components most users need, with optional expansion SDKs available for specialized domains including HPC, computer vision (ROCm-CV), data science (ROCm-DS), and life sciences (ROCm-LS). Built on a unified, pure-CMake build system, TheRock provides stable nightly releases with support for multiple Linux distributions and Windows. The open-source nature of TheRock accelerates adoption through public PRs, transparent CI pipelines, and code developed directly in public GitHub repositories. This results in a faster, continuous release cycle with improved quality and stability, reliable validation, and a smooth out-of-box experience.

## Expanded Hardware Support Across AMD's AI Portfolio

ROCm 7.14 continues AMD's investment in expanding support across its growing AI portfolio, enabling production support of [AMD Instinct™ MI350 PCIe® GPUs](https://www.amd.com/en/blogs/2026/amd-instinct-mi350p-pcie-gpus-run-enterprise-ai-on-your.html). OS coverage broadens to match MI350X and MI355X, with new support added for RHEL 9.8, SLES 15 SP7, SLES 16, and Debian 13 across both bare-metal and passthrough virtualization environments. MI350P is also now enabled for Vanilla Kubernetes on Ubuntu and Red Hat OpenShift v4.21, providing a validated path for deploying MI350P in containerized and orchestrated production environments. This provides a consistent development experience across the AMD Instinct accelerator family, allowing developers to use the same open-source ROCm frameworks, libraries, compilers, and profiling tools to build, optimize, and deploy AI inference applications on PCIe-based enterprise servers.

The release brings full ROCm enablement to the AMD Ryzen AI MAX+ PRO 495 and Ryzen AI MAX PRO 490 and 485 processors, enabling developers to leverage the complete ROCm software stack for AI, HPC, and content creation workloads. Optimized libraries such as hipBLASLt and GPU-accelerated Blender rendering extend GPU compute capabilities to AMD's latest AI PCs. Support is also expanded to Ryzen AI 5 435 and 430 across Ubuntu and Windows platforms.

Multi-GPU scalability continues to mature with validated support for systems containing two, four, and eight Radeon AI PRO GPUs, providing developers with a reliable foundation for larger AI training and inference deployments.

The developer ecosystem expands alongside the hardware. ROCm Systems Profiler and ROCm Compute Profiler are now enabled on Ryzen AI MAX+ PRO platforms, while Compute Profiler adds support for Strix Halo and Strix Point. AMD SMI also introduces expanded command-line telemetry for Ryzen AI systems, providing visibility into GPU and memory utilization, power, thermal, and fan metrics. ROCgdb, rocprofiler-SDK, and rocprofv3 are now validated across Ryzen AI platforms, while the complete ROCm developer toolchain reaches production Linux support on the AMD Ryzen AI Halo developer workstation.

## Stronger AI Framework and Inference Support

ROCm 7.14 strengthens AMD's AI software ecosystem with updated frameworks, optimized inference software, and expanded enterprise tooling. The release adds support for PyTorch 2.12 and JAX 0.10.0, enabling developers to leverage the latest framework capabilities while benefiting from ROCm's optimized runtime.

TensorFlow support has been expanded to cover versions 2.21, 2.20, and 2.19.1, validated across all supported operating systems including Debian 12.13. Transformer Engine v2.12 is also validated for MI300A, MI300X, MI350X, and MI355X, with PyTorch 2.10 and JAX 0.8.2 as the tested framework versions.

Performance-tuned configurations are now available for popular open-source models including Llama 3.1 8B Instruct, Whisper Large v3, and Qwen3.6-35B-A3B running with vLLM on Radeon Linux, providing faster out-of-the-box inference with minimal manual tuning. Inference framework support extends to the broader Ryzen product family as well, with vLLM, llama.cpp, and Ollama now available on Ryzen Linux, while llama.cpp and Ollama bring that same coverage to Windows.

ComfyUI support now spans Radeon RX 7000 and 9000 series, Radeon PRO, Ryzen AI MAX, and Instinct GPUs across Linux, with Windows support available for Radeon and Ryzen AI MAX hardware. Stable Diffusion 3.5 and Chroma are validated across all supported devices, while Radeon RX 7000 and 9000 series GPUs and Ryzen AI MAX gain new support for Wan2.2 video generation models.

## Expanding Enterprise AI Software Support

The [AMD Enterprise AI](https://enterprise-ai.docs.amd.com/en/latest/index.html) Reference Stack continues to expand its support for production AI deployments on AMD platforms. AMD Solution Blueprints will also expand with industry-focused solutions for healthcare, telecommunications, and financial services, providing validated starting points for deploying enterprise AI workloads tailored to vertical-specific use cases.

AMD Enterprise AI Reference Stack now includes support for AMD Radeon™ GPUs as well as additional capabilities for AMD Inference Microservices (AIMs) and AMD AI Workbench. RDNA 4 (Radeon AI PRO R9600 & R9700) and RDNA 3 (Radeon PRO W7900 & W7800) GPUs are now supported. AIMs now expose an OpenAI-compatible API for LLM serving, with prebuilt inference containers that automatically configure based on the detected hardware profile, built-in model download and caching, and native integration with Kubernetes and Red Hat OpenShift for enterprise deployment. AMD AI Workbench complements this with prebuilt training and fine-tuning pipelines, automatically configured developer workspaces, GPU-as-a-Service support, and built-in ROCm profiling integration.

ROCm 7.14 introduces the first-ever SGLang support on AMD Radeon™ GPUs, bringing one of the industry's most popular LLM serving frameworks beyond Instinct™ accelerators. Optimized for multi-user, multi-request inference, SGLang is widely adopted for chatbots, AI assistants, and agentic AI workloads, and can deliver better performance in multi-turn and long-context scenarios compared to general-purpose serving frameworks. With SGLang now available on Radeon, customers gain greater flexibility to choose the framework best suited to their workload while leveraging AMD's unified ROCm software experience across Radeon™ and Instinct™ platforms.

## Improved Collective Communication for Distributed AI

ROCm 7.14 introduces significant enhancements to [ROCm Communication Collectives Library (RCCL)](https://rocm.docs.amd.com/projects/rccl/en/latest/) that improve communication efficiency for both large-scale distributed training and latency-sensitive workloads. Three new collective communication algorithms reduce communication overhead, improve scalability, and better utilize modern AMD Instinct accelerators.

A new direct ReduceScatter algorithm replaces the previous ring-based implementation, reducing latency for small- and medium-sized messages and accelerating distributed AI training across multi-node AMD Instinct deployments.

For large GPU clusters, a new hierarchical AllGather algorithm separates communication into intra-node and inter-node phases, reducing communication contention and maintaining higher throughput as deployments scale.

ROCm 7.14 also enables Copy Engine (CE) Collectives in RCCL, offloading collective communication to dedicated GPU copy engines. This improves FP8 communication performance on AMD Instinct MI355X accelerators by allowing communication to progress independently of compute, keeping GPU compute resources focused on application workloads.

## Math Libraries

ROCm 7.14 brings a broad set of updates across the math libraries, expanding platform coverage, adding new operations, and closing parity gaps with CUDA equivalents.

libhipcxx, the C++ standard library for HIP, joins the ROCm Core SDK in this release, available on Linux across Instinct, Radeon, and Ryzen, and on Windows for Radeon and Ryzen. rocBLAS also gains build and runtime support for SPIR-V, an open standard that bridges GPU compute code and hardware drivers for greater flexibility in compilation and integration.

The sparse math libraries see several additions: [hipSPARSE](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/index.html) adds Block Sparse Row (BSR) format support to Sparse Matrix Vector Product (SpMV) and Sparse Matrix-Matrix Multiplication (SpMM), closing a gap with cuSPARSE for developers working with sparse matrix workloads. [rocSPARSE](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html) gains ILDLT factorization and deprecates uint16 indexing. rocFFT improves work-buffer management for parallel FFT workloads.

On the dense linear algebra side, [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html) adds per-batch bias stride support to match cuBLASLt behavior, and rocBLAS extends its batched BLAS coverage with GEMV and GER/GERU routines that accept per-batch scalar coefficients stored on device. hipBLAS exposes and routes these new APIs to rocBLAS. [hipTensor](https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html) expands with Windows platform support, RDNA 3 and RDNA 4 based GPUs, FP16 and BF16 contraction, unary element-wise operators, and tensor trinary contraction.

## A Richer HIP Developer Experience

ROCm 7.14 introduces HIP Execution Context support, enabling developers to partition GPU compute resources across concurrent workloads. By assigning dedicated resources to individual streams rather than having workloads compete for the entire device, Execution Contexts improve parallelism, resource utilization, and performance when multiple jobs run simultaneously.

In addition, ROCm 7.14 introduces new asynchronous memory management APIs for memory prefetching and discard operations, along with library management APIs. These capabilities give developers greater control over memory behavior and runtime resources, helping optimize application performance while reducing the need for platform-specific workarounds.

## More Powerful Profiling and Debugging Tools

ROCm 7.14 significantly enhances the developer experience through improvements across its profiling ecosystem.

[ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/) now supports re-attaching to running applications, selective profiling of MPI ranks, profiling scoped to ROCTx regions, and periodic sampling of hardware performance counters. MPI rank filtering lets developers focus profiling on specific processes within a distributed parallel job, while ROCTx region scoping targets named sections of code that developers have annotated in their application. Unified Memory Profiling provides detailed insight into page migrations, faults, and transfer behavior, helping developers identify memory bottlenecks more efficiently.

[ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) expands platform support while simplifying installation through pip distribution. Underlying infrastructure improvements in rocprofiler-SDK and rocprofv3 reduce profiling overhead and introduce beta support for Streaming Performance Monitors (SPM), enabling continuous hardware counter sampling during application execution.

Beginning with PyTorch 2.12, the PyTorch Profiler also adopts rocprofiler-SDK as its ROCm backend, creating a stronger foundation for future profiling capabilities.

## Introducing hipFile as Part of AMD Infinity Storage

ROCm 7.14 brings [hipFile](https://rocm.docs.amd.com/projects/hipFile/en/latest/), AMD's direct storage library, to production readiness, enabling applications to move data directly between storage and GPU memory without passing through the CPU memory, reducing latency and overhead for storage-intensive workloads. The amdgpu-file driver now supports NVMe-over-Fabrics (NVMeoF) and NFS-over-RDMA, which enables high-performance access to network file storage without buffering data in CPU memory, expanding the range of storage environments hipFile can connect to. hipFile now works across a wider range of storage hardware configurations, both local and remote, and operators can query the AMD Infinity Storage initialization status directly to confirm the setup before running workloads.

## Expanded GPU Virtualization

ROCm 7.14 expands virtualization support for AMD Instinct GPUs with VMware ESXi 9.1 support for the AMD Instinct MI350 Series. MI350 Series also introduces additional SR-IOV partitioning configurations supporting up to 64 virtual functions per node, enabling more flexible GPU resource allocation for cloud and virtualized deployments.

For Radeon platforms, Hyper-V virtualization support now extends to Radeon RX 7900 and Radeon RX 6900 Series GPUs on Windows Server 2025. ROCm 7.14 also adds support for Radeon RX 9700, Radeon RX 9600D, and Radeon PRO V710 GPUs, while expanding operating system support to include Debian 13 and Ubuntu 24.04.

## Simplified Packaging and Deployment

ROCm 7.14 introduces a new Runfile Installer, a self-contained installation package that works without an OS package manager. It supports user-selectable install directories, multiple ROCm versions installed side-by-side, local GPU autodetection, and optional driver installation across Ubuntu, RHEL, CentOS, and SLES.

MIGraphX and ONNX Runtime are now delivered as multi-architecture artifacts — including signed RPMs, an MIGraphX Debian package, and Python wheels — making it easier to deploy across different system configurations. Multi-architecture enablement for MIGraphX was introduced to support Red Hat Inference Server 3.5. rocprofiler-SDK also adds Address Sanitizer (ASAN) build support, helping developers catch memory errors in GPU-accelerated applications earlier in the development cycle.

The [ROCm Validation Suite (RVS)](https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/) is now published as deb and rpm packages through a formal release mechanism on [repo.amd.com](https://repo.amd.com/), making it easier to install and stay current. [TransferBench](https://rocm.docs.amd.com/projects/TransferBench/en/latest/) is now bundled directly inside the RVS packages, so users no longer need a separate TransferBench installation. [ROCm Bandwidth Test (RBT)](https://rocm.docs.amd.com/projects/rocm_bandwidth_test/en/latest/) reaches end of life in this release, so users are encouraged to migrate to RVS or TransferBench for bandwidth testing going forward.

For HPC users, ROCm 7.14 adds pre-built hipTensor and rocALUTION packages on repo.amd.com, so users no longer need to build these libraries from source. Both are now available for Linux.

## Summary

In this blog, you explored the key updates and improvements that make ROCm 7.14 a major milestone in AMD's open AI software ecosystem. This includes the production debut of TheRock, expanded hardware support, stronger AI frameworks, enhanced profiling tools, faster communication libraries, and streamlined enterprise deployment.

ROCm 7.14 represents more than a feature release—it establishes a new foundation for the future of AMD's AI software platform. With TheRock now serving as ROCm's production build and release infrastructure, ROCm is now a modular software platform capable of delivering innovation more rapidly while improving software quality and simplifying deployment.

Whether you're developing on Ryzen AI workstations, scaling distributed training across AMD Instinct clusters, or deploying enterprise AI services in production, ROCm 7.14 provides the tools, performance, and flexibility needed to build the next generation of AI applications. Stay tuned for more updates from Advancing AI Day.

For complete [release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html), installation guidance, and migration documentation, visit the [ROCm documentation](https://rocm.docs.amd.com/en/docs-7.14.0/index.html).

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED 'AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
