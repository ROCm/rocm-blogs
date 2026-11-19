---
blogpost: true
blog_title: "ROCm 10.2.0: Advancing Performance, Stability, and Developer Experience"
date: "18 Nov 2026"
author: "Amy Wiebe, Adam Chan, Onil Gunawardana, John Tyler, Saad Rahim"
thumbnail: 'ROCm10.2-thumbnail.png'
tags: "Reinforcement Learning, AI/ML, Scientific Computing, Fine-Tuning, System-Tuning, C++, Linear Algebra, Compiler, Hardware, HPC, Installation, JAX, LLM, Memory, Multimodal, Optimization, Performance, Profiling, PyTorch, Kubernetes"
category: "Ecosystems and Partners"
target_audience: "All / Anyone"
key_value_propositions: "This blog discusses the highlights of the ROCm 10.2.0 release."
language: English
myst:
    html_meta:
        "author": "Amy Wiebe, Adam Chan, Onil Gunawardana, John Tyler, Saad Rahim"
        "description lang=en": "Explore ROCm 10.2.0, featuring refreshed frameworks, enhanced memory and library capabilities, and more as AMD advances the developer experience"
        "keywords": "Release, ROCm, Software, Version, AI/ML, HPC, Data Science, Libraries, Compilers, Toolchains, Computer vision, MIOpen, Profiling, Developer tools, Debugging, Communication, Math, ROCm.AI, ROCm CLI, AMD Skills, Hyperloom, ROCm Core SDK, Agentic AI, Long Term Support"
        "vertical": "AI, Developers, Systems"
        "amd_category": "Developers"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Ecosystem and Partners"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training, Computer Vision, Data Science, Design, Simulation & Modeling, Deploying AI at Scale, Edge Computing"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Amy Wiebe, Adam Chan, Onil Gunawardana, John Tyler, Saad Rahim"
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

# ROCm 10.2.0: Advancing Performance, Stability, and Developer Experience

[ROCm 10.2.0](https://rocm.docs.amd.com/en/latest/about/release-notes.html) is the latest in our six-week release cadence with a set of focused, incremental improvements across the stack. This release prioritizes stability, refreshing the frameworks, libraries, and tools developers already rely on so they can upgrade with confidence.

Highlights include updated serving builds moving to vLLM 0.32.0 and SGLang 0.5.21; PyTorch 2.15 wheels arriving, and multi-GPU communication in RCCL receiving additional tuning. hipFile and AMD Infinity Storage adds filesystem support for logical volumes (LVMs) and software raid (mdraid), and the Systems Profiler tool consolidates multiple binaries into a single command.

## Frameworks

Developers serve large language models on AMD GPUs, ROCm 10.2.0 gives them newer engines to do it with. Our serving builds move up to vLLM 0.32.0 and SGLang 0.5.21. vLLM ships in prebuilt Docker images and Python wheels, SGLang also ships in Docker images, and both frameworks now run on Python 3.14.

The official vLLM Docker images, also make the AI Tensor Engine for ROCm (AITER) easier to use. With vLLM packaged for Instinct™ GPUs, AMD Radeon™ GPUs, and Ryzen™ processors, AITER now operates on Radeon GPUs and Ryzen AI processors with no separate manual installation. Instinct MI300X, MI325X, MI350X and MI355X GPUs get prebuilt AITER kernels. On Radeon and Ryzen AI, the kernels compile automatically the first time they are used.

### Training: Strong Results Across the Board

AMD had solid results with our latest MLPerf 6.1 submission. On the DLRM v4 benchmark the Instinct MI355X trained in 38.5% less time than NVIDIA B300 and delivered competitive performance against NVIDIA GB300 at an 8-node scale. AMD submitted a total of 5 optimized recipes in total, including a brand-new submission for GPTOSS-20B. Read more about AMD's MLPerf Training in the [companion blog](https://www.amd.com/en/blogs/2026/amd-advances-ai-training-across-more-workloads-in-mlperf-training.html).

Primus is also making Mixture-of-Experts (MoE) training faster and more efficient. With optimized FP8/MXFP8 kernels and a new fused MoE Megakernel, it keeps compute and communication running side by side. On DeepSeek-V3, the team observed better-than-linear weak scaling from 256 to 1,024 Instinct MI355X GPUs with per-scale tuning. To find out more visit AMD's [MoE Training Optimization with Primus blog](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-optimization/README.html).

This release also adds PyTorch 2.15 wheels along with JAX 0.11.2, MIGraphX 2.19 and ONNX Runtime 1.29. PyTorch picks up some new additions with: FlyDSL kernels for MI350, Tensor Data Mover (TDM) for MI450, and ROCm 10.0 wheels are now featured on the get-started page. One change to be aware of: PyTorch packages built by TheRock now install ROCm as a separate dependency, instead of bundling it inside the PyTorch package itself.

## Planning and Tuning Inference Deployments

Once a model runs on 10.2.0, the next step is deciding how to deploy and then optimizing that deployment for the most efficient runs. Two tools in this release cover these steps.

### Configuring the Deployment

[ROCm InferaSim](https://rocm.docs.amd.com/projects/infera/en/latest/simulation/overview.html) projects how a serving configuration will perform from a few short, measured GPU runs. It leverages the CPU to estimate whether a configuration fits in GPU memory and how many replicas a latency target requires. It also helps decide how to split prefill (prompt processing) from decode (token generation).

### Tuning the Deployment

[Hyperloom](https://github.com/AMD-AGI/Hyperloom), AMD's autonomous agentic system designed to optimize end-to-end inference workloads on AMD GPUs, continues to expand. New utilities let you set power budgets, so optimization targets efficiency rather than raw throughput alone. Early access support for single-node Radeon AI Pro R9700 GPUs extends Hyperloom beyond Instinct.

## ROCm CLI and AMD Skills

Getting from a fresh machine to a running model on AMD GPUs keeps getting faster, and two recent releases, [ROCm CLI v0.3.0](https://github.com/ROCm/rocm-cli/releases/tag/v0.3.0) and [AMD Skills](https://github.com/amd/skills), continue this trend. ROCm CLI makes `rocm install sdk` install the latest stable ROCm (10.2.0) by default, so new installs start on the current release. `rocm serve` adds an AIM container engine for AMD Enterprise AI alongside vLLM and Lemonade. It also adds a new `/v1/messages` endpoint with tool use and streaming that lets local models work with tools.

AMD Skills extends that same speed to AI-assisted development through enabling AI agents to work effectively with AMD's software stack. The collection now covers 12 AMD projects, from serving with Lemonade and quantizing with Quark to accelerating inference with ZenDNN, allowing AI coding assistants to work with them directly. This release adds three new skills for kernel work and porting. Two cover FlyDSL, one for writing GPU kernels ([flydsl-kernel-authoring](https://github.com/amd/skills/tree/main/skills/flydsl-kernel-authoring)) and one for analyzing their traces ([flydsl-kernel-trace-analysis](https://github.com/amd/skills/tree/main/skills/flydsl-kernel-trace-analysis)). A third skill allows Moat to port a CUDA project to run on AMD GPUs with ROCm and HIP ([moat-cuda-to-rocm](https://github.com/amd/skills/tree/main/skills/moat-cuda-to-rocm)). Developers can add them all with `npx skills add amd/skills`.

## RCCL Optimization for PCIe based GPUs

ROCm 10.2.0 tunes the ROCm Communication Collectives Library (RCCL) for single-node Radeon PCIe servers with eight GPUs. Earlier releases optimized RCCL on these servers, and this release builds on that with additional tuning.
The tuning includes RCCL collectives for single-node configurations. It targets tensor-parallel serving, where RCCL handles the collective communication between the GPUs that share one model.
AMD has validated the tuning with four-way tensor-parallel vLLM serving on eight-GPU Radeon servers.

## Expanded Ryzen Support

ROCm broadens local AI support on Ryzen laptops with support for 18 new AMD Ryzen processors, built on the same gfx1103 architecture as existing Radeon 700M Series processors. Developers get compatibility across Windows and Linux, with no custom builds or special configurations required. This is backed by validated gfx1103 support across HIP, PyTorch, and vLLM.

## Next-Gen Storage and I/O (hipFile/AIS)

Building off the latest [ROCm 10.1](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-10.1-blog/README.html) release, AMD has expanded the platform compatibility of AMD Infinity Storage (AIS) and [hipFile](https://rocm.docs.amd.com/projects/hipFile/en/latest/index.html), as well as their integration into the AI and storage benchmarking ecosystem, by introducing support for [PyTorch](https://github.com/pytorch/pytorch/pull/191069), [fio](https://github.com/axboe/fio/pull/2113), and [ior](https://github.com/hpc/ior/pull/540).

### AI/Storage Software Ecosystem Integration

PyTorch AI developers can leverage the `load_storage` and `save_storage` APIs to accelerate data transfer of large tensors between the GPU and storage devices. Both are exposed through PyTorch's native Python API, delivering hipFile's improved data transfer bandwidth when loading model weights, checkpointing training jobs, or in other input/output heavy workloads.

hipFile support was recently added in the latest release of [fio](https://github.com/axboe/fio/releases/tag/fio-3.43) (v3.43), and in the main branch of [ior](https://github.com/hpc/ior/pull/540), two industry standard, open-source storage benchmarking tools for evaluating storage performance.

### Platform Compatibility Expansion

The portability of hipFile has been improved by introducing support for storage devices that utilize Logical Volume Management (LVM) and RAID through the Linux Kernel's mdraid subsystem. LVM and RAID allow multiple storage devices to be abstracted as a single device for improved reliability and performance. This growth in platform support is a step towards hipFile being compatible with any modern, production-grade, storage infrastructure built around AMD GPUs.

## Libraries

### Linear Algebra: Complex GEMMs in hipBLASLt & Reduced Library Footprint

ROCm 10.2.0 expands [hipBLASLt's](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/) mathematical domain by introducing host and client infrastructure for single- and double-precision complex matrix multiplication (CGEMM and ZGEMM). This initial release delivers functional parity across AMD Instinct MI300 Series and next-generation MI350 Series architectures (gfx942 and gfx950), with rocBLAS routing complex GEMM calls to hipBLASLt.

Alongside architectural expansion, ROCm 10.2.0 shrinks hipBLASLt's deployment footprint. In Debian and RPM packages, hipBLASLt now installs its solution selection catalogs as zlib-compressed .dat.zlib files. This change shrinks the catalog disk footprint from ~3.8 GB to ~194 MB. Catalogs are decompressed on first use, which can add initial overhead, but subsequent calls remain unaffected.

### Sparse Linear Algebra: rocSPARSE & hipSPARSELt

Working with sparse data in simulations and graph analytics often means dealing with launch overhead and memory bottlenecks. In ROCm 10.2.0, rocSPARSE introduces strided batched triangular solvers (`rocsparse_sptrsm`), letting developers dispatch thousands of independent matrix solves in a single GPU call instead of slow, sequential host loops. For simulation workflows like algebraic multigrid setup, a new memory accumulation path speeds up sparse matrix multiplication on wide rows. Matrix sorting is also cleaner thanks to rocsparse_spsort, which consolidates multiple format-specific routines into one simple API. Finally, alongside hipBLASLt, hipSPARSELt adopts compressed solution catalogs (`.dat.zlib`), cutting disk footprint by more than 90%.

### Fast Fourier Transforms (FFT): Scaling Across GPUs and Large Datasets in rocFFT

Large-scale signal processing and physical simulations depend on running FFTs across multiple GPUs and massive datasets without memory limits getting in the way. rocFFT brings just-in-time (JIT) callbacks to multi-GPU transforms. By using the newer plan-description APIs (`rocfft_plan_description_set_load_callback` and `rocfft_plan_description_set_store_callback`), developers can fuse custom data preparation and post-processing directly into multi-GPU workflows, cutting out redundant kernel launches.

Additionally, rocFFT now automatically switches to 64-bit index arithmetic whenever problem sizes exceed 32-bit addressing limits. This includes transforms with more than 2^32 elements, large strides, and high batch counts. This lets very large FFTs compute correctly up to the limits of device memory, where previously the offset calculations could overflow. Transforms that fit within 32-bit limits continue to use 32-bit indexing, avoiding the extra register and instruction cost of 64-bit math.

## Developer Tools and Profiling

The profiler now simplifies how it is invoked and connects GPU activity to storage I/O. As AI and HPC applications grow more complex, profiling tools need to deliver deeper insight. Counter data has historically only been visible while kernels are running, so anything the GPU does between dispatches goes unmeasured. This release adds a device counting mode to the Streaming Performance Monitor (SPM) that samples counters across the whole device for as long as profiling is active, whether a kernel is in flight or not. SPM supports gfx1200, gfx1201, and gfx1250. On those targets, the sample interval is configurable in reference-clock cycles.

The ROCm Systems Profiler now introduces `rocsys`, an [experimental unified command-line entry](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/develop/how-to/using-rocsys.html) point that simplifies profiling workflows by consolidating multiple binaries under a single, intuitive interface. Developers can profile any application with a single command, getting a full trace out of the box without needing to remember which underlying binary to invoke. Beyond basic profiling, `rocsys` exposes dedicated subcommands for binary re-writing, call-stack sampling, instrumentation, attaching to live processes, and causal profiling, all with consistent argument forwarding.

Container [workflows](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/how-to/attaching-to-running-process.html) also get easier to profile. With the ROCm Systems Profiler, developers can now attach directly from the host into a running container without manually syncing runtime libraries between host and container versions.

For storage-heavy pipelines, AMD Infinity Storage telemetry is now exposed in AMD-SMI and ROCm Systems Profiler, so I/O activity can be correlated against GPU compute timelines in Perfetto and ROCm-Optiq.

## Physical AI

The [AMD Simulation Toolkit](https://rocm.docs.amd.com/projects/simulation/en/latest/index.html) is an open-source software kit built for ROCm to enable high-performance physical simulations and advanced computational graphics on AMD GPUs. In this release four open-source simulation libraries have been added to the kit, bringing their capabilities to Instinct GPUs:

- **Quadrants**: A multi-platform compiler for physics simulation.
- **Genesis**: A Physical AI simulation platform.
- **Warp**: A Python framework for writing high-performance simulation and graphics code.
- **MuJoCo-Warp**: A general-purpose physics engine.

Together they give developers a foundation for building complex physics simulators on AMD hardware and lay the groundwork for AMD's continued expansion into Physical AI. Stay tuned for the upcoming companion blog on the individual libraries.

## Summary

[ROCm 10.2.0](https://rocm.docs.amd.com/en/latest/about/release-notes.html) is a stability-focused step forward on our six-week cadence. It brings updated vLLM and SGLang serving builds, PyTorch 2.15 wheels, additional RCCL tuning for multi-GPU systems, and broader hipFile and AMD Infinity Storage support. It also adds complex GEMM support in hipBLASLt, new sparse and FFT capabilities, and a single `rocsys` command for systems profiling. Each is an incremental improvement to tools developers already use, so teams can upgrade with confidence.

We look forward to sharing ROCm 10.3.0 in the new year. With the holiday season approaching, expect that release to arrive slightly later than our usual six-week cadence.

- For the full [release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html), package lists, and installation instructions, see the [ROCm documentation](https://rocm.docs.amd.com/en/latest/).
- For more information on the previous ROCm version please visit the [ROCm 10.1 Blog](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-10.1-blog/README.html).

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED “AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF SUCH THIRD-PARTY CONTENT.

AMD, the AMD Arrow logo, AMD Instinct, ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved.
