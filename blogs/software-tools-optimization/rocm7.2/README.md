---
blogpost: true
blog_title: "ROCm 7.2: Smarter, Faster, and More Scalable for Modern AI Workloads"
date: 22 Jan 2026
author: 'Anshul Gupta'
thumbnail: 'rocm72-thumbnail.jpg'
tags: AI/ML, GenAI, HPC
category: Software tools & optimizations
target_audience: AI Developers and enthusiast
key_value_propositions: ROCm7.2 software launch
language: English
myst:
    html_meta:
        "author": "Anshul Gupta"
        "description lang=en": "we highlight the latest ROCm 7.2 enhancements for AMD Instinct GPUs, designed to boost AI and HPC performance"
        "keywords": "AMD ROCm7 Software Update"
        "vertical": "AI, HPC, Developers, Data Science, Systems"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, AI Inference, Computer Vision, Generative AI, Deploying AI at Scale"
        "amd_blog_topic_categories": "Adaptive & Embedded Computing"
        "amd_blog_authors": "Anshul Gupta"
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

# ROCm 7.2: Smarter, Faster, and More Scalable for Modern AI Workloads

Modern AI workloads demand more than raw compute—they require a tightly integrated software stack that can extract maximum performance, scale efficiently across systems, and operate reliably in production environments. With the latest ROCm 7.2 release, we’re delivering a broad set of optimizations and software enhancements designed to improve developer productivity, runtime performance, and enterprise readiness.

In this blog, we highlight the latest ROCm 7.2 enhancements for AMD Instinct GPUs, designed to boost AI and HPC performance. Learn how hipBLASLt and GEMM optimizations, FP8/FP4 support in rocMLIR and MIGraphX, and topology-aware communication with GDA and RCCL deliver higher throughput and lower latency. We also cover AI model tuning for AMD Instinct™ MI300X and MI350 GPUs and Node Power Management (NPM) for efficient multi-GPU operation—together enabling faster, more scalable, and reliable AI workloads.

## hipBLASLt Optimization and Enhancements

We introduced a set of targeted enhancements to hipBLASLt focused on improving developer productivity and runtime performance for GEMM-heavy workloads. These updates include expanded tuning capabilities, new features such as restore-from-log for reproducible performance, and swizzle A/B to optimize memory access patterns. In addition, benchmarking and reporting improvements provide deeper visibility into kernel selection and performance behavior. Together, these optimizations deliver measurable performance gains on AMD Instinct™ MI200 and MI300 GPUs compared to ROCm 7.1, directly supporting our end-customer workloads while strengthening internal benchmarking and performance analysis pipelines.

## Single Root IO Virtualization (SR-IOV)/RAS Features for MI350X/MI355X

With ROCm 7.2, we enabled SR-IOV and RAS enhancements for MI350X and MI355X GPUs to support secure, reliable, and scalable multi-tenant deployments. Key updates include bad page avoidance to improve GPU availability under memory fault conditions, security hardening such as volatile memory clearing and MMIO fuzzing protections, and feature parity with competing platforms. These capabilities are essential for cloud and enterprise environments, directly addressing hyperscale requirements while improving GPU reliability, isolation, and security for virtualized workloads.

## Performance Boost with GEMM Tuning

With this release, we performed extensive tuning of GEMM (general matrix multiplication) kernels across FP8, BF16, and FP16 data types on AMD MI300X, MI350, and MI355  platforms, with a strong focus on real customer use-case models such as GLM-4.6 and Llama 2. This work involved optimizing kernel selection, tiling strategies, memory layouts, and data movement to better match model shapes and execution patterns. The result is a direct, measurable performance uplift for AI training and inference workloads, improving throughput, latency, and overall customer experience.

## rocMLIR  and MIGraphX New Data Types (FP8/FP4) Enablement

We enabled next-generation low-precision data types—FP8 and FP4—across the ROCm compiler and graph stack,  rocMLIR and MIGraphX, to support emerging AI training and inference workloads. This work provides the foundational compiler, lowering, and execution support required for efficient use of these data types, and is a key enabler for MI350 new product introduction (NPI) bring-up. By expanding ROCm’s low-precision capabilities, these enhancements allow developers and customers to unlock higher performance and efficiency for advanced models while future-proofing the platform for upcoming AI architectures and workloads.

## End-to-End Communication Enhancements in ROCm: From GDA to Smart Collectives

With the latest ROCm release, AMD has significantly advanced GPU-to-GPU communication across the stack. rocSHMEM now supports a GPUDirect Async (GDA) backend for both intra-node and inter-node communication. This enables GPUs to directly exchange data with peer GPUs or communicate across nodes via an RNIC (RDMA NIC) using device-initiated kernels, completely removing the CPU from the critical communication path and reducing latency.

Building on this foundation, RCCL has become more topology-aware and intelligent in how it utilizes modern multi-rail networks. With native support for 4-NIC network topologies, RCCL can distribute collective communication across all available network interfaces rather than treating them as a single logical link. Communication patterns are optimized for rail alignment, minimizing cross-rail contention and improving aggregate bandwidth utilization.

Additionally, by backporting features from NCCL 2.28, RCCL incorporates more advanced collective algorithms, resulting in faster, more stable distributed training and improved scalability for large-scale AI workloads on AMD GPUs.

## Compiler Enhancements with ThinLTO Support

With ROCm 7.2, upgrades to the compiler infrastructure enable ThinLTO for AMD GPUs. Normally, compilers optimize each source file in isolation, limiting cross-file optimizations. ThinLTO allows the compiler to analyze and optimize across multiple object files, making better decisions around function inlining, specialization, and dead-code removal—without the long build times of full LTO. The result is global optimization with near-local build speed, which is especially important for existing AI frameworks such as PyTorch, Triton, XLA, and custom kernel stacks.

## Optimizing AI Models for AMD Instinct MI300X and MI350 GPUs

AMD has made significant progress in optimizing leading AI models on AMD Instinct MI300X and MI350 series GPUs, delivering higher throughput, lower latency, and more efficient large-scale inference performance.

On the MI355X and MI350X GPUs, the Llama 3.1 405B model has been tuned with kernel-level enhancements and memory bandwidth optimizations, while the Llama 3 70B and Llama 2 70B models have also been optimized to fully leverage the advanced architecture of these GPUs.

For the MI300X Series GPUs, AMD focused on GEMM-level optimization for the GLM-4.6 model and implemented performance improvements in DeepEP, enabling faster execution and more efficient GPU utilization. These optimizations highlight AMD’s commitment to maximizing AI model performance.

## Node Power Management for Multi-GPU Nodes

Node Power Management (NPM) dynamically manages power distribution and GPU frequencies across multiple GPUs within a node by leveraging built-in telemetry and advanced control algorithms. It automatically adjusts GPU frequencies to ensure that the total node power remains within defined limits. You can use AMD SMI to verify NPM status and monitor the node’s power allocation. This feature is supported on AMD Instinct MI355X and MI350X GPUs in both bare-metal and KVM SR-IOV virtualized environments when used with Platform Level Data Model (PLDM) bundle 01.25.17.07.

## Summary

With this ROCm 7.2 release, the platform continues to mature as a high-performance, production-ready ecosystem for AI and HPC. These updates strengthen performance, scalability, and reliability across the ROCm software stack for real-world deployments.

We invite you to explore ROCm 7.2 and join the growing community by pushing the frontiers of AI on an open platform.

For more information, visit [AMD>ROCm](https://www.amd.com/en/products/software/rocm.html) and join the open AI/HPC community driving ROCm forward.

## Additional Resources

[ROCm 7.2 Release Notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
