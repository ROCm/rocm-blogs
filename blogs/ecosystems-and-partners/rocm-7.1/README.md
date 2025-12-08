---
blogpost: true
blog_title: "Continuing the Momentum: Refining ROCm for the Next Wave of AI and HPC"
date: 5 Nov 2025
author: 'Anshul Gupta, Liam Berry, Saad Rahim'
thumbnail: 'blog7.jpg'
tags: AI/ML
category: Ecosystems and Partners
target_audience: AI DEVELOPERS AND ENTHUSIAST
key_value_propositions: ROCm 7.1 release for AI and HPC
language: English
myst:
    html_meta:
        "author": "Anshul Gupta, Liam Berry, Saad Rahim"
        "description lang=en": "ROCm 7.1 builds on 7.0’s AI and HPC advances with faster performance, stronger reliability, and streamlined tools for developers and system builders."
        "keywords": "ROCm 7.1, AI, HPC, Developers"
        "vertical": "Developers, AI, HPC"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Ecosystem and Partners"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training, Data Science, Computer Vision, Content Creation / Rendering, Edge Computing, Deploying AI at Scale"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Anshul Gupta, Liam Berry, Saad Rahim"
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

# Continuing the Momentum: Refining ROCm For The Next Wave Of AI and HPC

Earlier this year, we introduced [ROCm 7.0](https://rocm.docs.amd.com/en/docs-7.0.0/about/release-notes.html), a major milestone that supercharged AI and HPC infrastructure with improved performance, expanded datatype support—including FP4 and FP6—and deeper integration across frameworks like PyTorch, TensorFlow, and Triton. 

Building on this foundation, [ROCm 7.1](https://rocm.docs.amd.com/en/docs-7.1.0/about/release-notes.html) delivers targeted enhancements designed to make the platform faster, more reliable, and easier for developers and system builders to work with. In this blog, you’ll see how ROCm 7.1 continues to elevate GPU performance, scalability, and developer productivity across the AMD ecosystem. This release introduces smarter HIP runtime features for faster data movement, unified memory optimization, and better multi-GPU control. It also expands system support to new OS versions and enables flexible GPU partitioning on AMD Instinct MI350 and MI355X accelerators for scalable, efficient deployments. Enhanced RCCL communication delivers faster multi-GPU training and improved throughput, while upgraded profiling tools give developers deeper insights to fine-tune performance across AI and HPC workloads.

## Elevating HIP Runtime for Faster, Smarter GPU Performance:  

ROCm 7.1 expands HIP with new CUDA-parity memory and stream controls plus runtime optimizations that reduce launch overheads and accelerate multi-GPU workflows. On the memory side, HIP adds 2D memset variants including `hipMemsetD2D8/16/32` and their async forms along with `hipMemcpyBatchAsync` and `hipMemcpy3DBatchAsync` to cut setup time for large tensors and batched transfers, and new 3D peer copies including `hipMemcpy3DPeer` and `hipMemcpy3DPeerAsync` for device-to-device pipelines. Managed memory gets smarter through `hipMemPrefetchAsync_v2` and `hipMemAdvise_v2` so applications can prefetch or hint placement based on observed access patterns, enabling more predictable performance under unified memory for developers.

For deployment and orchestration, module utilities like `hipModuleLoadFatBinary` and `hipModuleGetFunctionCount` streamline fat binary loading and introspection, while new stream attributes (`hipStreamSetAttribute`, plus `hipStreamGetAttribute`/`hipStreamGetId`) let you tune synchronization policy and query priority/IDs to better schedule concurrent kernels and copies enabling high-throughput data transfers across devices for faster multi-GPU workflows. This will also allow developers to better orchestrate concurrent GPU tasks for maximum utilization. Developers migrating from CUDA will also notice expanded cooperative groups capabilities-now with nested tile partitioning-to match CUDA semantics in fine-grained thread collaboration. API references and "how-to" guides for these areas are available in the HIP docs ([Stream Management](https://rocm.docs.amd.com/projects/HIP/en/develop/reference/hip_runtime_api/modules/stream_management.html?utm_source=chatgpt.com), [Module Management](https://rocm.docs.amd.com/projects/HIP/en/develop/reference/hip_runtime_api/modules/module_management.html#module-management-reference), [Managed/Unified Memory](https://rocm.docs.amd.com/projects/HIP/en/develop/reference/hip_runtime_api/modules/memory_management/unified_memory_reference.html#unified-memory-reference), and [Cooperative Groups](https://rocm.docs.amd.com/projects/HIP/en/develop/reference/hip_runtime_api/modules/cooperative_groups_reference.html#cooperative-groups-reference)).

Under the hood, HIP shortens time-to-first-kernel with lower module-load latency and faster kernel-metadata retrieval and improves the doorbell path to batch packets more efficiently for graph launches, raising throughput for short-lived or graph-heavy workloads. The [ROCm 7.1 release notes](https://rocm.docs.amd.com/en/docs-7.1.0/about/release-notes.html#id4) also document behavior fixes during stream capture (e.g., aligning legacy-stream enqueue semantics with CUDA and resolving cross-stream capture faults), which helps portability for complex graph capture/replay flows. For detailed enhancements and updates refer to the [HIP Changelog](https://rocm.docs.amd.com/en/docs-7.1.0/about/release-notes.html#hip-7-1-0).

## Accelerating Matrix Computation with hipBLAS and hipBLASLt

ROCm 7.1 delivers major updates to the **hipBLAS** and **hipBLASLt** libraries, extending both precision coverage and model compatibility for AMD Instinct™ accelerators while improving developer workflows.

The **hipBLASLt 1.1.0** update brings a suite of performance and functionality enhancements tuned for training and inference workloads on next-generation Instinct GPUs. New fused-epilogue paths-Clamp and Clamp + Bias-allow activation and bias fusion directly within GEMM operations (`HIPBLASLT_EPILOGUE_CLAMP_EXT`, `HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT`), reducing kernel dispatch overhead and improving operator efficiency. Developers can now also capture auxiliary outputs for ReLU and Clamp activation in FP16 and BF16 precision, useful for frameworks that reuse activation masks in backpropagation.

From a performance perspective, hipBLASLt integrates TF32-optimized kernels for the MI355X and FP32/FP16/BF16 kernels for the MI350X, delivering measurable throughput gains in both training and inference workloads. Compatibility improvements for large-scale LLMs include fixes for Llama 2 70B stability and tuned heuristics for Mixtral-8x7B on MI325X. Developers can inspect these specialized kernel paths using the [hipBLASLt API Reference](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/).

The **hipBLAS 3.1.0** release complements these updates by modernizing build and deployment. A new `--clients-only` build mode lets teams compile only client applications against prebuilt libraries; ideal for CI/CD environments. The new build mode also adds Fortran enablement for Windows, extended GPU support, and shorter client build times. These changes align hipBLAS with other ROCm math libraries' modular build philosophy described in the [ROCm Core SDK and TheRock Build System Blog](https://rocm.blogs.amd.com/software-tools-optimization/therock/README.html).
For more information you can see the [hipBLAS](https://rocm.docs.amd.com/en/docs-7.1.0/about/release-notes.html#hipblas-3-1-0) and [hipBLASLt](https://rocm.docs.amd.com/en/docs-7.1.0/about/release-notes.html#hipblaslt-1-1-0) Changelogs.

## Expanded System Enablement:  

ROCm 7.1 broadens system and virtualization support across the AMD Instinct™ accelerator family, giving developers more flexibility in deploying AI and HPC workloads on diverse infrastructures.

This release extends official support to **Debian 13, Ubuntu 24.04.4 LTS, and RHEL 10.1**, as well as **SLES 15 SP7** and **Oracle Linux 9/10**, enabling seamless installation and maintenance across enterprise and open-source environments. These additions simplify integration in modern data centers that combine multiple Linux distributions under a unified ROCm stack. For the full compatibility matrix, refer to the [Supported Operating Systems](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.0/reference/system-requirements.html#supported-operating-systems) and [Supported GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.0/reference/system-requirements.html#supported-gpus) tables in the documentation.

Additionally, the ROCm 7.1 Kernel Fusion Driver (KFD) now efficiently handles **NPS2 + CPX** partitioning modes, improving GPU resource management across multi-socket servers and heterogeneous compute environments. On AMD Instinct MI350 and MI355 accelerators, users can now configure 2- or 8-GPU partitions for both bare-metal and virtualized deployments. This granular partitioning lets operators run concurrent AI, HPC, or mixed workloads with minimal interference, maximizing GPU utilization while reducing total cost of ownership (TCO).

ROCm 7.1 also adds **KVM SR-IOV** Guest OS Support for RHEL 10.0 on AMD Instinct MI355X and MI350X GPUs. This feature allows secure GPU pass-through and partitioned compute access inside virtual machines, ideal for multi-tenant clouds and AI research clusters. Developers can deploy containerized or virtualized workloads with near-native performance, fully managed by AMD's GPU Virtualization Host Driver. For more information on virtualization and GPU partitioning, see the [Virtualization Updates](https://rocm.docs.amd.com/en/docs-7.1.0/about/release-notes.html#virtualization-update-for-amd-instinct-mi350-series-gpus) and [Virtualization Support](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.1.0/reference/system-requirements.html#virtualization-support).

## Enhanced Multi-GPU Communication and Performance with RCCL:

ROCm 7.1 delivers significant enhancements to the [AMD ROCm Communication Collectives Library (RCCL)](https://github.com/ROCm/rccl), boosting performance and scalability for multi-GPU workloads. This release delivers higher throughput, reduced latency, and improved scaling across multi-GPU and multi-node configurations.

The RCCL 2.27.7 update adds new direct all-gather and tuned collective algorithms designed to reduce startup latency for small- and medium-sized transfers: critical for large-batch AI training and gradient-synchronization workloads. Channel-count selection and inlining optimizations further lower latency for key collectives, including AllReduce, AllGather, and ReduceScatter, while dynamic channel balancing boosts efficiency across GPUs connected via XGMI and InfiniBand.

To give developers more control over communication behavior, ROCm 7.1 introduces new environment flags for peer-to-peer (P2P) batching:

- `RCCL_P2P_BATCH_ENABLE=1` - enables batched P2P operations for small messages (up to 4 MB).
- `RCCL_P2P_BATCH_THRESHOLD=<bytes>` - sets message-size limit for batching.

When enabled, these variables significantly improve AllToAll and AllToAllv performance at scale, particularly in transformer-based models that issue many small collective calls.

Extensive tuning for AMD Instinct™ MI350 series yields measurable throughput gains for both single-node and multi-node (up to 16 nodes) configurations. Higher-bandwidth XGMI links and driver-level scheduling improvements translate into smoother overlap between compute and communication phases, resulting in improved end-to-end training efficiency on large clusters.

Finally, RCCL 2.27.7 maintains API-level compatibility with NCCL 2.27.7, integrating **Parallel Aggregated Tree (PAT)** algorithms for enhanced hierarchical reductions. This ensures that frameworks built on top of NCCL continue to operate seamlessly with RCCL as the backend. Symmetric-memory kernels remain disabled in this release while AMD finalizes unified memory enablement within the HIP runtime; users can track progress via the [ROCm Release Notes](https://rocm.docs.amd.com/en/docs-7.1.0/about/release-notes.html#rccl-amd-instinct-mi350-series-enhancements).

## Enhanced Profiling and Debugging Tools:     

ROCm 7.1 significantly advances AMD's GPU profiling and debugging ecosystem, giving deeper visibility into performance across both runtime and framework levels. These improvements span **ROCprofiler-SDK, ROCm Compute Profiler, ROCgdb**, and the **ROCm Systems Profiler**, creating a unified toolchain for kernel-level analysis, AI workload tracing, and cross-process performance correlation.

Developers can now attach profilers to live workloads without relaunching applications. Both [ROCm Compute Profiler 3.3.0](https://rocm.docs.amd.com/en/docs-7.1.0/about/release-notes.html#rocm-compute-profiler-updates) and the [ROCprofiler-SDK 1.0.0](https://rocm.docs.amd.com/en/docs-7.1.0/about/release-notes.html#rocprofiler-sdk-updates) introduce dynamic process attachment, allowing runtime profiling via process ID (PID). This capability is particularly valuable for long running or containerized AI workloads, enabling real-time performance sampling and trace capture without interrupting execution.

The Compute Profiler adds single-pass counter collection, reducing overhead when capturing complex metric sets. Users can now profile kernels in one pass using predefined metrics subsets through the `--set` and `--list-sets` options, while the new **TUI (Text-Based User Interface)** provides interactive metric descriptions, high-level panels for compute and memory throughput, and real-time roofline visualization. Results are stored in the new **ROCm Profiling Data** (`rocpd`) format; an SQLite3-based database that enables advanced post-analysis and visualization in the ROCprof Compute Viewer. For more information, see [ROCm Profiling Data (rocpd) output](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/docs-7.1.0/how-to/understanding-rocprof-sys-output.html#rocm-profiling-data-rocpd-output).

ROCprofiler-SDK introduces experimental **Streaming Performance Monitor (SPM)** support and richer program counter (PC) sampling, improving event-level resolution for fine-grained latency diagnostics. Developers targeting AMD's latest architectures benefit from new host trap-based PC sampling and MultiKernelDispatch thread trace support, enabling analysis of concurrent dispatches across multiple shader engines. Thread-trace data now also includes real-time clock alignment for more accurate timeline correlation across CPUs and GPUs.

[ROCgdb](https://rocm.docs.amd.com/projects/ROCgdb/en/docs-7.1.0/index.html) now achieves up to 80% code coverage, offering more robust break point handling and call-stack introspection for HIP kernels, while maintaining full compatibility with modern ROCm runtimes. The [ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/index.html) (formerly Omnitrace) now validates JAX and PyTorch frameworks, defaults to AMD SMI for telemetry, and integrates directly with `rocpd` for multi-process summary generation, ideal for profiling large, distributed training runs.

## Optimizing Efficiency with AMD SMI Power Cap:

The AMD SMI “Set Power Cap” improvement introduces finer power management control for AMD Instinct MI300X GPUs in virtualized environments. With this update, a virtual machine (VM) can now directly set a GPU power cap when running in a 1VF (single virtual function) configuration. The system automatically enforces the lowest power limit among those defined by the host, VM, and APML (Advanced Platform Management Link), ensuring safe and efficient operation. This allows users to tailor GPU power usage to their specific workload needs—balancing performance, thermal limits, and energy efficiency.

## Summary  

With [ROCm 7.1](https://rocm.docs.amd.com/en/latest/about/release-notes.html), the platform continues to evolve as a robust, production-ready ecosystem for AI and HPC. Each update strengthens the ROCm software stack, ensuring developers have the tools and performance they need to deploy next-generation workloads efficiently. We invite you to explore ROCm 7.1 and join the growing community pushing the frontiers of AI on an open platform. 

For more information, visit [amd.com/ROCm](https://www.amd.com/en/products/software/rocm.html) and join the open AI/HPC community driving ROCm forward. 

## Additional Resources

- [ROCm 7.1 Release Notes](https://rocm.docs.amd.com/en/docs-7.1.0/about/release-notes.html)
- [ROCm Documentation](https://rocm.docs.amd.com/en/latest/)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
