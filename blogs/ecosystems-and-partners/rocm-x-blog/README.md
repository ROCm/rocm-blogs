---
blogpost: true
blog_title: "ROCm 10.0: A Decade of Open Compute, Built for the Age of Agentic AI"
date: "27 Aug 2026"
author: "Liam Berry, Saad Rahim, Anshul Gupta, Evan Groenke"
thumbnail: 'ROCm-X-Blog-Thumbnail.png'
tags: "AI/ML, Linear Algebra, Compiler, Hardware, Computer Vision, HPC, Installation, JAX, LLM, Memory, OpenMP, Performance, Profiling, PyTorch, Agentic AI, ROCm.AI, ROCm CLI, AMD Skills, Hyperloom"
category: "Ecosystems and Partners"
target_audience: "All / Anyone"
key_value_propositions: "This blog highlights the important new features, changes, and general improvements implemented in ROCm 10.0, with a focus on ROCm.AI, the ROCm CLI, AMD Skills, Hyperloom, and the ROCm Core SDK"
language: English
myst:
    html_meta:
        "author": "Liam Berry, Saad Rahim, Anshul Gupta, Evan Groenke"
        "description lang=en": "Explore what's new in ROCm 10.0, headlined by ROCm.AI, the ROCm CLI, AMD Skills, and Hyperloom, alongside the ROCm Core SDK and platform-wide improvements"
        "keywords": "Release, ROCm, Software, Version, AI/ML, HPC, Data Science, Libraries, Compilers, Toolchains, Computer vision, MIOpen, Profiling, Developer tools, Debugging, Communication, Math, ROCm.AI, ROCm CLI, AMD Skills, Hyperloom, ROCm Core SDK, Agentic AI, Long Term Support"
        "vertical": "Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Ecosystem and Partners"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training, Computer Vision, Data Science, Deploying AI at Scale, Edge Computing, Generative AI"
        "amd_blog_topic_categories": "Software & Ecosystem, AI & Intelligent Systems"
        "amd_blog_authors": "Liam Berry, Saad Rahim, Anshul Gupta, Evan Groenke"
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

# ROCm 10.0: A Decade of Open Compute, Built for the Age of Agentic AI

AMD shipped ROCm 1.0 in April 2016: an open-source GPU compute stack built around a C++ compiler and a GPU programming language called HIP, aimed at high-performance computing. A decade later, the same platform trains and serves frontier AI models across industries and enables nearly every compute application benefiting from GPUs.

ROCm 10.0 marks that decade with a major version bump, and the jump is deliberate. TheRock, the automated open-source build and release system that reached production in ROCm 7.14, is now the foundation the entire release ships on. And for the first time, ROCm ships with a native agentic AI developer experience: [ROCm.AI](http://rocm.ai), built from the new ROCm CLI, AMD Skills, and Hyperloom, reshaping how the platform itself is packaged and supported.

This post walks through the highlights: what changed, what it enables, and why it matters — starting with the foundations, moving through [ROCm.AI](http://rocm.ai) and the release's other focal points, and closing with what's still ahead.

## A New Major Version, Built on TheRock

ROCm Core SDK 10.0 is the first major version bump since the 7.x series, and it is built end to end on [TheRock](https://github.com/ROCm/TheRock) — the automated, open-source build and release system that reached production in ROCm 7.14. Primitives, libraries, and framework wheels all come out of one pipeline, staged and validated across the whole stack before they ship. ROCm 10 and its minor releases will continue to be released approximately every six weeks.

That single pipeline is what puts ROCm on AMD GPUs across the entire portfolio — Instinct accelerators, Radeon graphics cards, and Ryzen integrated graphics — on Windows and Linux alike. One stack, one source tree, one release. A capability that lands for one family or one operating system is built and validated for the others in the same pass, not queued up as a separate porting project. The platform you prototype on a laptop is the same platform that serves frontier models in a datacenter rack, and the work you do on one carries straight over to the other.

## One Place to Get AMD GPU Software

Alongside the release itself, AMD is consolidating GPU software distribution onto a redesigned [repo.amd.com](https://repo.amd.com) with future releases. Historically, getting a full GPU stack running meant collecting pieces from different places: the ROCm packages from one host, the graphics and compute driver from another, and the datacenter tools from somewhere else again. Each had its own layout, its own naming, and its own instructions to follow.

The redesign brings AMD GPU software under one roof. ROCm packages, the amdgpu driver, and AMD's public GPU tools publish to a single repository with a consistent structure — one host to trust, one repository to configure, one set of instructions that stays the same regardless of which part of the stack you need. The new layout also carries multiple ROCm versions and multiple architectures side by side, so a machine can hold more than one ROCm version at once and teams can pin, stage, and roll forward on their own schedule rather than the release train's. Nightly, release candidate, and stable GA releases all publish artifacts in one standard format.

For anyone automating installs, this is the part that compounds. A container build, a provisioning script, or an air-gapped mirror now points at one place and stays pointed there. Adding the driver to an existing ROCm deployment stops being a second repository and a second set of keys and becomes another package from a repository already configured.

## ROCm.AI: An AI-Native Developer Experience

The headline of ROCm 10.0 is **[ROCm.AI](http://rocm.ai)**, AMD's new AI-native developer experience, publicly introduced at [Advancing AI 2026](https://www.amd.com/en/corporate/events/advancing-ai.html) and rolling out starting this release. Where past ROCm releases mainly gave you better primitives and libraries to build with, [ROCm.AI](http://rocm.ai) focuses on the workflow around those primitives: installing, validating, serving, and optimizing AI workloads on AMD hardware with far less manual toil. It's built from three pieces that ship together in ROCm 10.0 — the ROCm CLI, [AMD Skills](https://github.com/amd/skills), and [Hyperloom](https://github.com/AMD-AGI/Hyperloom).

### ROCm CLI

The **ROCm CLI** is a single, unified command-line tool for installing, validating, serving, updating, and troubleshooting AI workloads on AMD platforms, replacing what used to be a collection of separate scripts and manual steps. It ships as a tech preview in ROCm 10.0 — treat it as early access and expect the interface to keep evolving. `rocm serve <model>` spins up a model for inference on top of PyTorch, `rocm examine` diagnoses environment and driver problems, and the CLI supports air-gapped environments by letting its dependencies be downloaded as a self-contained bundle alongside the binary.

### AMD Skills

**AMD Skills** bring AMD-validated ROCm knowledge directly into the AI coding assistants developers already use — Claude, Cursor, Codex — through the Agent Skills format, as shown in Figure 1. The official catalog lives in the [`amd/skills`](https://github.com/amd/skills) GitHub repository as a federated collection of skills, compatible with the standard skill directories those tools already look in (`~/.claude/skills/`, `~/.cursor/skills/`, `.claude/skills/`), with a companion marketplace that packages skills built by AMD teams for one-command installation.

The catalog spans both GPU and CPU workflows. On the GPU side, `rocm-doctor` drives the CLI's `rocm examine` diagnostics, and `serving-llms-on-instinct` walks an agent through standing up a vLLM OpenAI-compatible endpoint on MI300X, MI325X, MI350X, or MI355X. On the CPU side, the catalog extends the same approach to EPYC processors through ZenDNN and zentorch: `serving-llms-on-epyc` and `quantize-for-zentorch` give an agent everything it needs to serve and quantize models for CPU-only inference, building on the public AMD PACE vLLM plugin. It's the same skill-driven workflow whether the target is an Instinct GPU or an EPYC CPU — an assistant with AMD Skills installed reaches for the right one automatically.

```{figure} ./images/AMD-Skills.gif
:align: center
:alt: AMD Skills
Figure 1: AMD Skills with ROCm CLI
```

### Hyperloom: Closing the Optimization Loop

**Hyperloom** is the third piece of [ROCm.AI](http://rocm.ai), and the one that goes furthest toward taking the human out of the loop entirely. It's a new open-source, agentic system that automates end-to-end inference workload optimization — profiling a workload, analyzing the results, planning changes, applying them, and validating correctness, on repeat, without an engineer manually driving each cycle. AMD reports that Hyperloom cuts what used to be weeks of manual optimization work down to hours, while exploring far more of the solution space than a human would attempt under time pressure.

Under the hood, Hyperloom orchestrates five components into a single Profile → Analyze → Plan → Optimize → Validate loop, shown in Figure 2: TraceLens-Agent for automated bottleneck identification, Magpie for kernel evaluation and benchmarking, IntelliKit's conversational profiling tools, GEAK's autonomous multi-agent kernel optimization (spanning Triton, HIP, CK, FlyDSL, and TileLang backends), and Arbor's self-evolving search over the optimization space. A separate tool, AgentKernelArena, lets teams A/B test and score different optimization agents against each other on standardized tasks. Hyperloom is public today, runs on MI300X, MI325X, and MI355X, and installs with a single `pip` command — point it at a model and configuration and tell your agent something like "optimize Minimax M3 MXFP8 with Hyperloom" to kick off a session. For the full architecture, component breakdown, and setup walkthrough, see the dedicated [Hyperloom blog](https://rocm.blogs.amd.com/software-tools-optimization/hyperloom/README.html).

```{figure} ./images/Hyperloom-ROCm-10.png
:align: center
:alt: Hyperloon
Figure 2: The Hyperloom Workflow
```

## Frameworks and Model Performance

### Turnkey Serving Containers

ROCm 10.0 delivers production-ready support for **[vLLM](https://github.com/vllm-project/vllm)** and **[SGLang](https://github.com/sgl-project/sglang)**, enabling teams to run LLM inference on AMD Instinct, Radeon, and Ryzen hardware without building the stack from source. Both frameworks undergo the same release validation as the rest of ROCm, with upstream contributions flowing back to their respective communities. Validated containers and Python wheels are available on Docker Hub, built through TheRock's multi-architecture CI pipeline.

### Local LLM Fine-Tuning on Ryzen™ AI MAX

ROCm 10.0 adds support for [Unsloth](https://github.com/unslothai/unsloth) on Ryzen™ AI MAX platforms, enabling fast, memory-efficient local fine-tuning of large language models. Through support for techniques such as LoRA and QLoRA, developers can fine-tune models with lower memory requirements while taking advantage of the large unified memory available on Ryzen™ AI MAX. This enables a more accessible and private AI development experience, allowing users to fine-tune models locally without depending on cloud infrastructure.

### Optimized Generative AI Workloads with ComfyUI

ROCm 10.0 enhances generative AI experiences with expanded ComfyUI optimizations for leading image and video generation models. AMD has performance-tuned popular models, including Wan2.2, FLUX.2 KLEIN, Stable Diffusion 3.5 Medium, Stable Diffusion 2.1, and Stable Diffusion XL Base, helping users achieve strong out-of-the-box performance on supported Radeon and Ryzen platforms.

Alongside model validation, ROCm provides practical guidance for selecting optimal attention algorithms and backends for ComfyUI image and video generation workloads ([Understanding Attention Algorithms and Their Backends for Image and Video Generation](https://rocm.blogs.amd.com/software-tools-optimization/comfyui-fa-backends/README.html)). These recommendations help users achieve optimal performance on AMD GPUs with minimal tuning, making it easier to go from installation to content creation.

## Communication Libraries: RCCL and rocSHMEM

Communication libraries get the largest single investment in this release — a concerted push to close parity gaps with NVIDIA's NCCL and NVSHMEM, and add capabilities that matter once you're running at real scale.

[RCCL](https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl), the ROCm Collective Communications Library, advances its upstream NCCL merge from 2.28.3 to 2.30.4 and adds symmetric memory support for tightly-coupled multi-GPU peer-to-peer communication, a GPU-initiated networking (GIN) device API that lets the GPU kick off network transfers directly over GDA and SDMA paths instead of bouncing through the CPU, one-sided host APIs for data movement beyond collectives, and a new set of Pythonic APIs for using RCCL directly from Python. Large-scale bootstrap improvements cut startup overhead across hundreds of GPUs, fault-tolerance enhancements improve resilience for long-running jobs, and copy-engine collectives, hierarchical AllGather, and direct collectives push more throughput out of both multi-node and single-node topologies.

**[rocSHMEM](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/)** continues closing the API gap with NVSHMEM 3.6.5: this release adds host AMO and context APIs with non-MPI IPC runtime support, `reduce_on_stream` variants and a native `reduce_scatter` implementation, wave-level collective operations, `team_split_2d` for finer-grained team management, and additional buffer and heap management functions — bringing GPU-initiated communication in ROCm closer to functional parity with its CUDA counterpart.

## Libraries

Two libraries stand out in this release, and both are aimed at giving AMD platforms a sharper edge over the competition.

**[hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/)** ships a new GEMM kernel optimizer that lets teams tune kernel selection for their own workloads locally, without ever exposing model weights or proprietary data to AMD or any third party — benchmarking runs on-premises and produces a configuration profile biased toward the best-performing kernels for a given problem shape. It turns GEMM-bound transformer inference and training, the workloads most teams actually run, into a tunable advantage rather than a fixed cost. **[rocSPARSE](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse)** adds a matching set of sparse linear algebra improvements built to put AMD ahead on sparse workloads: a smart SPMM selector that automatically chooses the best sparse matrix-matrix multiplication algorithm for a given matrix structure, CSC triangular solves, and Blocked-ELL DenseToSparse conversion.

The rest of ROCm's math and media libraries carry routine version bumps this release, including HIP, hipBLAS, rocBLAS, rocPRIM, rocRAND, rocThrust, hipFFT, rocJPEG, rocDecode, and rocSHMEM.

## Developer Tools and Profiling

ROCm Compute Profiler's **[Roofline analysis](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html#standalone-roofline)** extends to GFX11xx (RDNA 3) architectures, bringing performance-ceiling analysis — a fundamental tool for telling whether a kernel is compute-bound or memory-bandwidth-bound — to Radeon hardware for the first time. **rocSHMEM API tracing** is now supported both in the ROCm Systems Profiler and the ROCprofiler-SDK, giving visibility into communication activity in the same timeline as GPU compute to understand its contribution to overall application performance. ROCprofiler-SDK now allows attaching from a host to a containerized process, improving the user experience and removing the need for manually copying .so files.

ROCm 10.0 marks the general availability of [ROCm Optiq 1.0](https://rocm.docs.amd.com/projects/roc-optiq/en/latest/), AMD’s unified visualization and analysis environment for ROCm profiling data. Optiq combines ROCm Systems Profiler traces and ROCm Compute Profiler analysis in a single application, enabling developers to move seamlessly from system-level timeline exploration to kernel-level performance analysis. Designed for large-scale AI and HPC workloads, Optiq delivers responsive navigation of large traces, system topology correlation, roofline analysis, and baseline comparison across Windows, Linux, and macOS.

## Developer Experience

ROCm 10.0 ships **[ASAN](https://github.com/google/sanitizers/wiki/AddressSanitizer)**-instrumented packages alongside the standard ones. Address Sanitizer builds used to be something you produced yourself if you needed to chase a memory-safety bug through the stack; now they install like any other ROCm package, so catching a use-after-free or an out-of-bounds access in library code is a package swap rather than a rebuild. ASAN build support also arrives for **[ROCgdb](https://rocm.docs.amd.com/projects/ROCgdb/en/latest/)** and the **[ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/)**, improving the reliability of the profiling infrastructure itself.

Packaging also changes how ROCm binaries find their libraries. Installed packages — RPM, DEB, and the runfile installer — now embed `RPATH` rather than `RUNPATH`. The practical difference is precedence: `RPATH` is consulted before `LD_LIBRARY_PATH`, so a ROCm binary resolves to the ROCm libraries it was built and validated against rather than whatever happens to be earlier on the library path. Fewer surprises on machines carrying more than one ROCm version.

Tarball distributions keep `RUNPATH`, which is the right default for that audience — `LD_LIBRARY_PATH` still takes precedence, so teams composing their own environments or layering ROCm into an existing toolchain retain the override they rely on. Either way, this affects only how ROCm's own tree resolves internally; nothing outside it changes.

## Windows: One SDK, One Release Cadence

On Windows, the [HIP SDK](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/) is retired. Its role passes to the ROCm Core SDK, so Windows and Linux now draw from the same SDK definition and the same release cadence rather than a Windows-specific package on its own schedule. One version number, one set of release notes, one answer to "what's in this release" regardless of which OS you're targeting.

ROCm on Windows ships as a tarball today — extract it where you want it and point your build at it. That keeps the first release of the unified SDK simple and predictable, and it composes cleanly with the toolchains Windows developers already have. Native Windows installers follow later in 2026, bringing the same managed install and update experience Linux users get from the package repositories.

## Summary

ROCm 10.0 marks ten years of AMD's open GPU software platform, and the version number is earned in more than one way. TheRock reaching full production, the ROCm Core SDK's new packaging and release model, and — most of all — [ROCm.AI's](http://rocm.ai) arrival with the ROCm CLI, AMD Skills, and Hyperloom mark a real shift in how developers and their agents build on AMD hardware. Around that core, the release reaches broadly: expanded virtualization support, validated inference containers across Instinct, Radeon, and Ryzen, and the largest single investment RCCL has ever received.

For the full release notes, package lists, and installation instructions, see the [ROCm documentation](https://rocm.docs.amd.com/en/latest/).

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
