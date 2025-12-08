---
blogpost: true
blog_title: "Empowering Developers to Build a Robust PyTorch Ecosystem on AMD ROCm™ with Better Insights and Monitoring"
date: 21 Oct 2025
author: 'Hongxia Yang, Peng Sun, Nick Romero, Jeff Daily, Jithun Nair, Pruthvi Madugundu, Jagadish Krishnamoorthy, Srinivasan Subramanian, Eli Uriegas'
thumbnail: 'pytorch_thumbnail.png'
tags: AI/ML, PyTorch, Performance
category: Applications & models
target_audience: AI DEVELOPERS AND ENTHUSIASTS
key_value_propositions: Pytorch increased unit test coverage and monitoring
language: English
myst:
    html_meta:
        "author": "Hongxia Yang, Peng Sun, Nick Romero, Jeff Daily, Jithun Nair, Pruthvi Madugundu, Jagadish Krishnamoorthy, Srinivasan Subramanian, Eli Uriegas"
        "description lang=en": "Production ROCm support for N-1 to N+1 PyTorch releases is in progress. The AI Software Head-Up Dashboard shows status of PyTorch on ROCm."
        "keywords": "Pytorch, ecosystem, AI, Unit test coverage, performance, quality, monitoring, AI tool"
        "vertical": "Developers, AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs, Radeon Graphics"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Training, AI Inference, Generative AI"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Hongxia Yang, Peng Sun, Nick Romero, Jeff Daily, Jithun Nair, Pruthvi Madugundu, Jagadish Krishnamoorthy, Srinivasan Subramanian, Eli Uriegas"
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

# Empowering Developers to Build a Robust PyTorch Ecosystem on AMD ROCm™ with Better Insights and Monitoring

The PyTorch ecosystem is a vibrant and expansive collection of tools, libraries, and community-driven projects that enhance and extend the core PyTorch framework. It empowers researchers and developers to build, train, and deploy deep learning models across a wide range of domains with flexibility and efficiency.

At AMD, the PyTorch and ML Framework team is dedicated to delivering an exceptional out-of-the-box experience for developers. Over the past year, the team has made significant strides in several key areas:

- **Broader PyTorch ecosystem support**
- **Improved PyTorch CI test coverage across a wider range of GPU architectures**
- **Expanded training and inference capabilities**
- **Streamlined developer experience**
- **New functionality and performance enhancements**
- **Enhanced quality monitoring**

This blog highlights our ongoing efforts to build a robust PyTorch ecosystem on AMD ROCm™ software, productizing PyTorch for N-1, N, N+1 releases across ROCm N-1, N, N+1 versions. We’re also excited to introduce [AI SoftWare Heads-Up Dashboard (AISWHUD)](https://aiswhud.amd.com/), a comprehensive dashboard designed to provide deep insights into the health and performance of the PyTorch ecosystem on ROCm software.

## Broader PyTorch Ecosystem Support

The PyTorch ecosystem continues to expand with growing support for AMD hardware across multiple projects. Here are some examples of key developments:

### TorchInductor Performance Dashboard

PyTorch 2.0 introduced torch.compile, a compiler-based approach that boosts performance while keeping the familiar Pythonic workflow. The [TorchInductor Performance Dashboard](https://hud.pytorch.org/benchmark/compilers) tracks these improvements across benchmarks like TorchBench, HuggingFace, and TIMM, offering insights into end-to-end workload performance under various Inductor configurations (e.g., precision modes, HIPGraphs, dynamic shapes).

- **AMD contributions to community Inductor Dashboard**
  
  AMD supports the TorchInductor Dashboard with AMD Instinct™ MI300 series GPUs, enabling regular performance monitoring and faster regression detection. Leveraging this dashboard, AMD has been actively analyzing performance data to ensure rapid turnaround in addressing regressions on AMD GPUs.

- **AMD managed Inductor dashboard**
  
  In addition to the community dashboard, AMD maintains an internal version that tests pre-release ROCm builds, updated Triton compilers, and hardware like the AMD Instinct™ MI350 series as part of our broader **[AISWHUD](https://aiswhud.amd.com/)** initiative to ensure strong PyTorch performance across AMD platforms.

### TorchAO (Advanced Optimization)

TorchAO focuses on quantization techniques to improve inference efficiency. Recent advancements include support for lower-precision formats such as INT4 and FP4, which are crucial for reducing memory footprint and accelerating performance. Notably, AMD Instinct™ MI300 CI integration has been added, marking a step forward in supporting AMD latest hardware.

A few notable features supported on ROCm are listed below:

- [Fused Matmul + Dropout support for Sparse kernel](https://github.com/pytorch/ao/pull/1868)
- Block Sparse MatMul support through hipSparseLt
- INT8, INT4, MX-FP8 Inference and QAT (Quantization-Aware Training) support on AMD Instinct MI300.
- Training support for FP8 QAT + sparse optimizer with ZeRO.

### Enable PyTorch Native Windows Support

PyTorch now offers [native support for Windows](https://www.amd.com/en/blogs/2025/the-road-to-rocm-on-radeon-for-windows-and-linux.html) as a public preview. This feature enables developers to run AI inference workloads directly on AMD Radeon™ 7000/9000 Series GPUs, as well as select Ryzen™ AI APUs, on both Windows and Linux. This milestone significantly broadens the accessibility and test matrix of PyTorch for developers working in diverse environments.

### TorchTitan (Work in Progress)

TorchTitan is a PyTorch native platform for training generative AI models. TorchTitan on ROCm is an emerging initiative aimed at enhancing PyTorch's compatibility with AMD GPUs. While optimization is still underway, significant progress has been made in enabling broader support. Continuous Integration (CI) pipelines are being set up to ensure stability and performance, with a pending pull request that will further streamline development and testing workflows.

## Improved PyTorch CI test coverage on wider GPU architecture

AMD has made a significant investment to expand continuous integration (CI) coverage across a broader range of hardware platforms, including multiple generations of Instinct GPUs from the **MI200 Series** (MI210X, MI250X)  **MI300 Series** (MI300X, MI325X), to the **MI350 Series** (MI350X, MI355X). This effort ensures that PyTorch remains robust, performant, and well-tested on AMD latest GPU architectures.

These additions enable more comprehensive testing and optimization for high-performance workloads, especially in AI and scientific computing.

Looking ahead, AMD plans to further expand CI coverage by integrating **Navi 3 (AMD Radeon™ 7900X)** into the PyTorch CI pipeline. This will help ensure broader support for consumer-grade GPUs and enhance PyTorch’s accessibility for developers working across diverse hardware environments.

## Expanded training and inference capabilities

With the enablement of the AMD latest architectures **(MI300X and MI355X)**, PyTorch on ROCm has significantly expanded its training and inference capabilities.

### Unlocking Larger Models and Optimized Inference Performance

Powerful GPUs like MI300X, MI325X, and MI355X allow developers to scale larger models efficiently, especially when using frameworks like **vLLM** and **SGLang**, which are built on top of PyTorch and optimized for high-throughput inference.

### Scalable Training with Advanced Distributed Features

PyTorch now supports robust distributed training on ROCm with:

- **Fully Sharded Data Parallel (FSDP)** for memory-efficient training of large models.
- **One-shot and two-shot AllReduce** strategies for optimized communication.
- **Symmetric memory enablement,** improving performance, and reducing bottlenecks in multi-GPU setups.

These features make it easier to train massive models across multiple GPUs without compromising speed or stability.

### Expanding Attention Mechanism Support

To further enhance model efficiency and flexibility, PyTorch on ROCm is expanding support for various attention mechanisms:

- **Sliding Window Attention** – ideal for long-context models.
- **FlashAttention** – a fast, memory-efficient attention implementation.
- **Memory-Efficient Attention** – reduces memory usage while maintaining performance.

## Streamlining Developer Experience

AMD continues to invest in improving the developer experience for PyTorch on ROCm, making it easier to build, test, and deploy AI workloads across a wide range of hardware. Here’s a look at the key components of this streamlined ecosystem:

**Nightly Builds for Rapid Development**

- **Nightly Wheels**:
PyTorch ROCm nightly wheels are built and published regularly via upstream CI, enabling developers to test the latest features and fixes without waiting for formal releases.

- **Nightly Docker Images**:
AMD hosts [nightly PyTorch ROCm Docker images](https://hub.docker.com/r/rocm/pytorch-nightly) to support fast iteration of development and testing. 

**Official Releases and Flexible Deployment Options**

- **Stable Docker Images**:
The [rocm/pytorch Docker repository](https://hub.docker.com/r/rocm/pytorch/tags) provides official release images for PyTorch on ROCm, supporting a wide range of ROCm versions and Ubuntu configurations. These images are ideal for developers looking for prebuilt environments tailored to AMD GPUs.

- **Matching Python Wheels**:
AMD also publishes PyTorch ROCm wheels on repo.radeon.com that align with the Docker images, offering flexibility for users who prefer Python wheel package installations over containers.

- **Docker Advantage**:
The ROCm Docker images often include combinations of PyTorch, ROCm, and Python versions that are not available in upstream CI, making them a valuable resource for testing and deployment.

**Support PyTorch for N-1, N, N+1 releases across ROCm N-1, N, N+1**

- AMD ROCm follows a rolling compatibility model for PyTorch, supporting:

1. The latest PyTorch release **(N)**
2. One prior release **(N-1)**
3. One future release **(N+1)** (in preview or limited testing)

- This applies similarly to ROCm versions, meaning PyTorch releases are tested against ROCm **N-1, N,** and **N+1** versions. This includes minor version. For example, today’s version (N) is ROCm 7.0, N-1 would be 6.4, N+1 would be 7.1.

**Community Engagement**

- AMD maintains an active cadence of reviewing issues reported by the PyTorch community, helping ensure that bugs are addressed and feedback is incorporated into future releases.

These efforts reflect commitment from AMD to build a robust and developer-friendly PyTorch ROCm ecosystem, empowering researchers and engineers to innovate faster across diverse hardware platforms.

## New functionality and performance enhancements

The latest PyTorch updates (2.8, 2.9) bring powerful new features and optimizations that improve both training and inference across AMD GPUs. These enhancements are designed to support larger models, more efficient execution, and broader hardware compatibility.

**Low Precision & Format Enablement**

- **OCP Micro-Scaling Format Support (MXFP8/MXFP4)**:
PyTorch on ROCm now supports OCP’s micro-scaling formats on **gfx950,** enabling ultra-low precision training and inference. These formats are ideal for memory-constrained environments and high-throughput workloads.

**Compiler & Backend Improvements**

- **AOT Inductor with CK Backend on MI350X and MI355X (gfx950)**:
The Inductor compiler now supports **Ahead-of-Time (AOT)** compilation using the **Composable Kernel (CK)** backend for **gfx950,** enabling maximum autotuning and performance optimization.
- **CK Backend for SDPA and GEMM Operators**:
SDPA and GEMM operations now leverage the CK backend, improving performance and compatibility with AMD’s latest GPU architectures.
- **CK Backend Enabled in CI**:
The CK backend is now integrated into PyTorch’s CI pipeline, ensuring consistent testing and validation across supported platforms.

**Operator-Level Performance Enhancements**

- **SDPA with AOTriton 0.11b**:
  Scaled Dot Product Attention now uses AOTriton 0.11b, delivering faster and more memory-efficient attention computation.
- **hipBLASLt Default on MI100X (gfx908) (ROCm ≥ 6.3)**:
  Matrix operations on gfx908 now default to **hipBLASLt,** offering improved performance for GEMM workloads.
- **MIOpen Enhancements**:
  - Enabled `channels_last_3d` format for convolution and batch normalization.
  - Removed redundant transpose in NHWC convolutions, reducing overhead.
- **FP8 Rowwise Support in hipBLASLt**:
  Added support for FP8 rowwise operations in _scaled_grouped_mm, complementing existing _scaled_mm functionality.

**Memory & Distributed Training Optimizations**

- **Symmetric Memory Enablement**:
  Improves memory efficiency and performance in distributed training setups, especially for two-shot AllReduce strategies.

These updates reflect the continued commitment to building a high-performance, scalable, and developer-friendly PyTorch ROCm ecosystem. Whether you're training massive LLMs or deploying efficient inference pipelines, the latest enhancements make AMD GPUs a compelling choice.

## Enhanced PyTorch ecosystem quality monitoring

The PyTorch team at AMD has continually monitored PyTorch unit test coverage using our unit test parity automation tool to track and monitor progress and regression of PyTorch on ROCm unit test coverage. During the recent weeks, PyTorch team has improved the PyTorch unit test coverage by enabling 1823 out of 2137 tests based on *skipIfRocm* decorator and *DISABLED* github issues in subcomponents like *sdpa, fsdp, inductor, symmetric memory, scaled mm, and linalg.*

In addition, AISWHUD is developed for monitoring the PyTorch ecosystem unit test parity, which will be described in detail in **“Introducing AISWHUD”** section.

## Introducing AISWHUD

[AISWHUD](https://aiswhud.amd.com/) is a centralized portal designed to monitor and publish the health and performance metrics of AI software components running on ROCm. Its goal is to provide developers with actionable insights into the PyTorch ecosystem, helping ensure stability, performance, and reliability.

In this blog, we feature two dashboards: the newly public PyTorch Unit Test Coverage and the upcoming externally available TorchInductor performance dashboard.

**PyTorch Unit Tests Coverage Monitoring**

[AISWHUD](https://aiswhud.amd.com/) tracks the status of PyTorch unit tests across key workloads like Default, Distributed, and Inductor on PyTorch Unit Test coverage. It provides detailed statistics on passed, skipped, and failed tests. Figure 1 shows the screenshot of PyTorch test status on MI350X on ROCm 7.0.

```{figure} ./images/Picture1.png
:align: center
:alt: Scaling performance
Figure 1: AISWHUD PyTorch Unit Test Summary – MI350X
```

**TorchInductor Performance Dashboard**

AISWHUD’s TorchInductor Performance Dashboard shows the benchmarks (Huggingface, Torchbench, TIMM) for both MI300X and MI350X (as shown in Figure 2). Note that this dashboard is **not** externally accessible at the moment. We are working on making this internal dashboard externally available.

```{figure} ./images/Picture2.png
:align: center
:alt: Scaling performance
Figure 2: AISWHUD TorchInductor Performance Dashboard
```

**Work in progress**

The AISWHUD team is working on expanding more OSS repositories (including TorchAO, vLLM, etc) for monitoring and making the system accessible to the public audience. 

## Summary and Next Steps
In summary, the AMD's PyTorch ecosystem team continues to make impactful strides in delivering a seamless and powerful experience for developers. Through expanded ecosystem support, improved CI coverage, enhanced training and inference capabilities, and a streamlined developer journey, we are committed to building a robust and production-ready PyTorch experience on AMD ROCm™ Software. The introduction of the AISWHUD further strengthens our efforts by offering deep visibility into the health and performance of the PyTorch ecosystem, empowering developers with actionable insights and greater confidence in deploying on ROCm.

Looking ahead, we plan to expand our CI coverage across more OSS repositories in the PyTorch ecosystem, expand [AISWHUD’s](https://aiswhud.amd.com/) capabilities to integrate broader ecosystem metrics, and collaborate closely with the community to drive further improvements across the PyTorch ecosystem software stack.

## Acknowledgements
We would like to thank the PyTorch engineering team at **Meta Platforms** for their close collaboration and technical support. Their insights and responsiveness were instrumental to the success of this work, and we greatly appreciate the opportunity to work together.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
