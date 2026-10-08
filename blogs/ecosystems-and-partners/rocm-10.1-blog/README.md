---
blogpost: true
blog_title: "ROCm 10.1: Breaking the Data-Movement Bottleneck"
date: "05 Oct 2026"
author: "Adam Chan, Amy Wiebe, Liam Berry, Saad Rahim, Danny Guan, Anshul Gupta"
thumbnail: 'rocm-10.1-thumbnail.jpg'
tags: "AI/ML, Fine-Tuning, Linear Algebra, Compiler, Hardware, Computer Vision, Installation, JAX, LLM, Memory, Multimodal, Optimization, Performance"
category: "Ecosystems and Partners"
target_audience: "All / Anyone"
key_value_propositions: "This blog discusses the highlights of what is included in ROCm 10.1"
language: English
myst:
    html_meta:
        "author": "Adam Chan, Amy Wiebe, Liam Berry, Saad Rahim, Danny Guan, Anshul Gupta"
        "description lang=en": "ROCm 10.1 targets the data-movement bottleneck with AMD Infinity Storage, NUMA-aware memory, and a modernized stack."
        "keywords": "Release, ROCm, Software, Version, AI/ML, HPC, Data Science, Libraries, Compilers, Toolchains, Computer vision, MIOpen, Profiling, Developer tools, Debugging, Communication, Math"
        "vertical": "AI, Developers, HPC, Systems"
        "amd_category": "Software tools & optimizations"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Ecosystem and Partners"
        "amd_blog_hardware_platforms": "Instinct GPUs, Radeon Graphics"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training, Computer Vision, Data Science, Design, Simulation & Modeling, Deploying AI at Scale, Edge Computing"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Adam Chan, Amy Wiebe, Liam Berry, Saad Rahim, Danny Guan, Anshul Gupta"
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

# ROCm 10.1: Breaking the Data-Movement Bottleneck

What limits your training run today: how fast the GPU computes, or how fast you can get data to it? For a growing number of workloads, the answer is the latter. ROCm 10.1 leads with that path, and carries a full slate of compiler, library, and tooling work behind it.

Raw accelerator throughput is still a limiting factor, but as AI models and HPC datasets scale, an emerging bottleneck is feeding those accelerators data. Checkpoints, key-value caches, and model parameters now outgrow the fast memory nearest to the GPU, and the data path from storage to the device becomes the largest bottleneck. ROCm 10.1 addresses this shift with substantive improvements in its AMD Infinity Storage technology and the HIP runtime. hipFile gains additional capabilities to smoothly move data directly between storage and the GPU. The HIP runtime adds NUMA-aware host memory allocation, which places memory close to the compute that uses it.

Beyond data paths, ROCm 10.1 advances the developer workflow. AMD's growing catalog of [AMD Skills](https://github.com/amd/skills), complemented by the ROCm CLI, gives coding agents standardized integrations to set up local AI, diagnose ROCm issues, and optimize LLM inference on AMD hardware.

## Next-Generation Storage and I/O with AMD Infinity Storage

As model parameters, checkpoints, and key-value (KV) caches grow, storage increasingly becomes the bottleneck that starves the GPU rather than feeds it. **AMD Infinity Storage (AIS)** debuted in ROCm 7.14 as a library that moves data directly between storage and memory without routing it through CPU memory.

```{figure} images/hipFile.png
:alt: Diagram comparing two storage data paths: the traditional path moving NVMe storage through system RAM as a bounce buffer with the CPU in the data path before reaching the GPU, versus the hipFILE path connecting NVMe storage directly to the GPU via the PCIe bus with the CPU only in the control plane

Comparison of the traditional storage path through host RAM versus the new hipFILE path that moves storage directly to the GPU
```

AMD builds on this foundation with three [`hipFILE`](https://hipfile.readthedocs.io/en/latest/) improvements that reduce unnecessary data movement and increase accelerator utilization:

- **Asynchronous Fast-Path Backend.** Read and write requests now run directly on a HIP stream, skipping the host-memory staging step through which data traditionally passes. This reduces latency and increases sustained throughput for checkpoint engines and KV-cache offloading systems, improving end-to-end responsiveness.
- **Batch I/O API.** Submits multiple file requests at once and dispatches them across an internal pool of worker threads, improving utilization across storage devices for large training datasets and data-intensive inference pipelines.
- **Multi-Tier I/O Statistics.** Progressively richer `hipFILE` telemetry flows to ROCm profiling tools, making it easier to determine whether a workload is limited by compute, memory, or data delivery.

## NUMA-Aware Host Memory Allocation with HIP

Adding more CPUs and GPUs does not automatically improve performance if data must travel between sockets to reach the compute using it. ROCm 10.1 helps reduce this overhead by extending HIP virtual memory management to let applications allocate host memory on a specific CPU NUMA node instead of falling back to generic memory pools. Libraries like RCCL can use this to build NUMA-local, zero-copy communication pipelines that keep data movement close to the compute that needs it, reducing unnecessary traffic between sockets.

## ROCm CLI & AMD Skills

ROCm 10.1 also brings [ROCm CLI](https://github.com/ROCm/rocm-cli) v1.0.0, which provides a single-binary tool for installing, configuring, and running local AI workloads on AMD GPUs. AMD has added first-class support for both ROCm 10.0 and ROCm 10.1, including automatic detection of compatible `flash-attn` and `amd-aiter` wheels for vLLM. The `rocm install` command handles the new ROCm 10 “next” package layout out of the box and provides a flag to enable fully non-interactive installs for scripted or CI environments. A full-screen TUI dashboard provides live GPU telemetry alongside model serving and chat. ROCm CLI is available on Linux, Windows, and WSL2.

Where the CLI is the tool a developer runs directly, [AMD Skills](https://github.com/amd/skills) brings that same ROCm expertise to AI coding agents. Each skill packages verified configurations into a standardized integration that agents like Claude Code and Codex can apply consistently. AMD is highlighting two new skills that were added in this release:

- **quark-install** installs or verifies AMD Quark with a PyTorch build that matches your accelerator (PyPI, wheel index, local wheel, or source), then checks imports, kernels, and quark-cli.
- **quark-torch-llm-ptq** runs post-training quantization on PyTorch / Hugging Face LLMs. It inspects the model, helps you pick a scheme (FP8, INT4 and more), and produces a verified quantized model.

Each skill is owned and versioned by the team behind the product it describes, and the list is expected to grow as more skills land.

## Developer Tools

The ROCprofiler-SDK introduces [kernel replay](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) (currently in beta), which collects a full counter set in one application run instead of rerunning the application once per counter group. Because GPUs can collect only a limited number of hardware performance counters per kernel dispatch, kernel replay re-executes each dispatch in place and restores tracked device memory between passes. ROCprofiler-SDK also lets AI framework profilers, such as PyTorch's Kineto and Triton's Proton, start and stop at any point during a run. A new [PC sampling agent skill](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk/skills/pc-sampling) in ROCprofiler-SDK guides AI coding assistants through running `rocprofv3` PC sampling on your application and reporting hotspots and stall reasons.

ROCm Compute Profiler's [standalone roofline report](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/analyze/cli.html#roofline-html-generation) is now an interactive HTML page where you can isolate a single kernel or roof and switch the arithmetic-intensity axis between cache levels. Its [PC sampling](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/pc_sampling.html) analysis now reports results per kernel, down to sample and stall counts on individual instructions. For inference workloads, a new guide covers [profiling vLLM](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/how-to/profile/mode.html#profile-vllm-workloads) with Compute Profiler.

ROCm Systems Profiler now supports Gorgon Point 1, 2, and 3 APUs (gfx1150, gfx1152, and gfx1153) on Linux, so you can trace CPU and GPU activity together for workloads running on these integrated GPUs. ROCm Optiq extends its unified visualization and analysis capabilities to support rocprofv3, enabling developers to visualize and analyze the latest profiling data within Optiq's unified interface alongside ROCm Systems Profiler traces and ROCm Compute Profiler analysis.

## LLVM 24 and Faster Rebuilds

ROCm 10.1 moves compiler development onto LLVM 24, bringing a host of new features and optimizations from the LLVM community, including reduced rebuild times for large projects. The `amdclang++` compiler advances from LLVM 23 to LLVM 24, delivering newer C++ language support and community optimization benefits.

ROCm 10.1 also speeds up incremental builds by caching the partitions that link-time optimization (LTO) creates when it splits HIP device code. As a result, a small change can reuse unchanged partitions instead of rebuilding them from scratch.

> **Upgrade note:** If you’re upgrading from ROCm 10.0, this is a major compiler version change. The `clang_major` version macro now reports `24`. Review any code or build scripts keyed to a specific compiler version, along with any tools that link directly against `libLLVM`.

## Advancing AI and Math Libraries

ROCm 10.1 also updates its AI and math libraries with expanded kernel support, compiler improvements, and more efficient sparse operations:

- **Composable Kernel:** AMD's library of optimized AI kernels speeds up attention on Radeon GPUs and brings more quantized matrix-multiply types to its kernel dispatcher.
- **MIGraphX:** AMD's graph compiler for AI inference replaces rocMLIR with rocMLIRTriton as its backend compiler, delivering strong inference performance improvements and often faster model compile times. The switch also enables MIGraphX to benefit from ongoing improvements to the Triton compiler.
- **rocSPARSE:** AMD's sparse linear algebra library can now run triangular solves directly on matrices stored in the ELLPACK (ELL) sparse format, without converting them first.

## ROCm SMI Transitions to AMD SMI

With the deprecation period for **ROCm SMI** (`rocm-smi`) now complete, **[AMD SMI](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)** (`amd-smi`) is its successor. AMD SMI provides a unified monitoring and management experience across AMD GPUs, CPUs, and APUs on Linux and Windows in both bare-metal and virtualized environments. Users are encouraged to migrate to `amd-smi`, which offers:

- Modern subcommand-driven CLI
- Native C library (amd_smi_lib)
- First-class bindings for Python, Rust, and Go

AMD SMI also extends visibility into containerized environments, identifying GPU processes running in containerd, CRI-O, Podman, LXC, and LXD containers, including Kubernetes pods. By reporting the full container IDs used by Docker and Kubernetes tooling, `amd-smi` lets operators match GPU processes directly to the containers running them, allowing for straightforward resource attribution across orchestrated and multi-tenant deployments.

## Explore ROCm with WSL2

Windows Subsystem for Linux (WSL2) enters tech preview in ROCm 10.1, with standard Linux packages installing directly in the guest, leading to no manual patching and no guest kernel builds. ROCm communicates with the Windows host driver through GPU paravirtualization, so PyTorch, HIP, and AI workloads run against the GPU from inside WSL2. Install the Windows driver, then install ROCm in the distribution as you normally would.

## GPU Virtualization on Ubuntu 26.04

ROCm now extends KVM SR-IOV support to Ubuntu 26.04 LTS as both the host and guest operating system, allowing teams to standardize their entire virtualized stack on the newest long-term support release. This pairs with version 9.3.0.K of AMD’s GPU virtualization driver (GIM) on the host, bringing the latest Ubuntu LTS into fully supported SR-IOV deployments without falling behind on driver currency.

## hipThreads: Incremental GPU Acceleration for Threaded Code

Writing high-performance GPU software has traditionally meant expressing concurrency through low-level execution and synchronization primitives. ROCm 10.1 introduces hipThreads to the ROCm Core SDK under the ROCm Threading libraries, bringing standard C++ threading concepts into heterogeneous device programming. hipThreads provides a path for accelerating existing CPU-threaded code on AMD GPUs without a full HIP rewrite.

The library maps those CPU concepts onto GPU execution on both Linux and Windows, letting developers move incrementally rather than all at once. It targets applications that have parallel CPU workloads, where the effort of a full GPU programming model migration isn't justified by the return.

For more information, see the [hipThreads documentation](https://rocm.docs.amd.com/projects/hipThreads/en/docs-10.1.0/) and the [hipThreads repository](https://github.com/ROCm/hipThreads).

## Summary

ROCm 10.1 reflects a clear priority: as workloads scale, moving and placing data efficiently matters as much as raw compute. **AMD Infinity Storage** and the `hipFILE` fast path shorten the route from storage to the GPU, and NUMA-aware allocation keeps host memory close to the compute that needs it. Combined with richer I/O telemetry that reveals whether a workload is bound by compute, memory or data delivery, ROCm now offers a more intelligent way to target the data-movement bottlenecks that increasingly gate large-model training and inference.

Around the memory and I/O foundation, the release modernizes the broader platform with LLVM 24 and faster rebuilds, updated AI and math libraries, hipThreads acceleration, expanded virtualization, first-class WSL2 and improvements to ROCm CLI.

Whether you're building on Ryzen AI systems, deploying inference across Radeon and AMD Instinct platforms, or scaling distributed training and scientific computing, ROCm 10.1 provides the performance, flexibility, and production-ready tools needed to develop the next generation of accelerated applications.

- For the full release notes, package lists, and installation instructions, see the [ROCm documentation](https://rocm.docs.amd.com/en/latest/).
- For more information on the previous ROCm version please visit [ROCm 10.0 Blog](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-x-blog/README.html).

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED “AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved.
