---
blogpost: true
blog_title: "Plug-and-Play CuPy on ROCm: Data Analytic Acceleration Made Simple"
date: 14 Nov 2025
author: 'Grant Pickert, Eliot Li'
thumbnail: 'cuPy-ROCm7-thumbnail.png'
tags: Scientific Computing, AI/ML, Time Series, Optimization
category: Applications & models
target_audience: Data scientists
key_value_propositions: Inform the AI community that AMD has released a new CuPy update with major performance improvement.
language: English
myst:
    html_meta:
        "author": "Grant Pickert, Eliot Li"
        "description lang=en": "Learn about how to enhance your analytics project with the latest AMD CuPy release."
        "keywords": "CuPy, NumPy, ROCm 7, Per-Thread Default Stream, CUDA Array Interface"
        "vertical": "Data Science, AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Data Science, Design, Simulation & Modeling, Predictive Analytics, Video Analytics"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Grant Pickert, Eliot Li"
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

# Plug-and-Play CuPy on ROCm: Data Analytics Acceleration Made Simple

AMD is committed to ensuring that CuPy works seamlessly on AMD Instinct GPUs through ROCm and has worked to support the latest features in upstream CuPy on ROCm. In this blog, you will learn about the enhancements in the current and upcoming AMD CuPy releases that will supercharge your analytics and data science projects.
In an earlier [blog on CuPy and hipDF](https://rocm.blogs.amd.com/artificial-intelligence/cupy_hipdf_portfolio_opt/README.html), it was demonstrated that CuPy and hipDF can be applied to complex analytics tasks with large datasets on ROCm using AMD GPUs. That blog used a PyPI wheel forked from earlier versions of CuPy and cuDF, and both CuPy and ROCm have advanced since then. In the latest AMD CuPy release, you will find many exciting improvements from the upstream CuPy library as well as ROCm 7.

[CuPy](https://docs.cupy.dev/en/stable/index.html) is a Python library for GPU-accelerated computing that offers a NumPy-like API for easy code migration, high-performance array operations, and efficient memory management with features like memory pools to reduce allocation overhead. It integrates with frameworks such as TensorFlow and PyTorch and allows custom kernels for fine-grained optimization, with multi-GPU and distributed computing capabilities for large-scale workloads. CuPy features a robust memory management subsystem and direct access to GPU compute, which can significantly accelerate array operations compared to CPU-based NumPy. It aims to be a drop-in replacement for NumPy with a compatible subset of its API and established interoperability protocols. For these reasons, CuPy has become a cornerstone of GPU-accelerated computing in Python, providing a familiar NumPy-like interface while leveraging the power of modern GPUs.

[ROCm 7.0.0](https://www.amd.com/en/blogs/2025/rocm7-supercharging-ai-and-hpc-infrastructure.html) is a major release of the ROCm software platform optimized for generative AI, large-scale training and inference, and accelerated discovery. It delivers end-to-end performance gains, especially on AMD Instinct MI350-series GPUs, while boosting portability with HIP 7.0 and simplifying deployment with enterprise-grade tools. The release also introduces production-ready MXFP4 and FP8 models via AMD Quark for faster, more efficient model delivery at scale. In addition, it includes enhanced tools, libraries, and infrastructure updates to help you build, train, and deploy next-generation AI applications.

To help you leverage the latest CuPy and ROCm features in your data analytics projects, AMD will continue to release PyPI wheels regularly while pushing changes to the upstream CuPy repository. This blog describes key enhancements in the PyPI wheel based on the [CuPy v13](https://docs.cupy.dev/en/stable/upgrade.html#cupy-v13) release and ROCm 7.0.0. It also outlines the plan to upstream those enhancements to the next CuPy version (v14).

## The CuPy v13 fork from AMD

The latest published [PyPI wheel](https://pypi.amd.com/simple/amd-cupy/) from AMD is based on a fork of CuPy v13 that introduces expanded functionality and hardware support, including:

- **ROCm 7.0.0:** The AMD CuPy v13 fork is compatible with [ROCm 7.0.0](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.0-blog/README.html), enabling out-of-the-box acceleration on a wide range of AMD Instinct GPUs, including MI350X and MI355X.
- **Per-Thread Default Stream (PTDS):** PTDS gives each host thread its own default stream. This per-thread stream is non-blocking and does not synchronize with others, enabling concurrency. PTDS is helpful for applications that run tasks in parallel across multiple threads while avoiding synchronization overhead. With PTDS enabled, GPU kernels and memory operations launched from different host threads are isolated, reducing unintended synchronization and improving performance.
- **CUDA Array Interface (CAI):** CAI is a Python-side protocol for zero-copy interoperability between different implementations of CUDA array-like objects in various projects. The CuPy v13 fork from AMD allows consumers implementing CAI (for example, Numba, PyTorch, RAPIDS libraries) to carry the producing stream rather than forcing a global device synchronization. This ensures safe producer-consumer ordering while avoiding unnecessary synchronization overhead, providing efficient and consistent cross-library interoperability.  With full PTDS support, data shared through CAI can safely use per-thread streams for synchronization, preserving concurrency across libraries.
- **Default C++14 backend:** Unlike upstream CuPy v13, which defaults to C++11, AMD’s fork adopts C++14 as the default backend standard. This change aligns with ROCm compiler toolchains, modernizes the build environment, and provides access to newer C++ language features.
- **Additional Bug Fixes:** The AMD CuPy v13 wheel also includes several stability and performance patches not yet present in the upstream [CuPy v13](https://docs.cupy.dev/en/stable/upgrade.html#cupy-v13) repository.

Together, these enhancements make the CuPy v13 fork from AMD a strong choice for developers targeting ROCm 7.0.0.

## Get started

### Requirements

To follow the steps in this blog, you need:

- AMD MI300X, MI325X, MI350X, or MI355X: See the [ROCm system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for supported operating systems.
- ROCm 7.0.0: See the [ROCm installation for Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html) for installation instructions.
- Docker: See [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) for installation instructions.

### Install CuPy

To install the CuPy wheel provided by AMD:

```bash
# (Optional) Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate
# Install from AMD’s PyPI index
pip install amd-cupy --extra-index-url https://pypi.amd.com/rocm-7.0.2/simple
```

To verify your installation, start a Python console and run:

```python
import cupy as cp
print("CuPy version:", cp.__version__)
```

Expected output:

```python
CuPy version: 13.5.1
```

Print the CuPy configuration by running:

```python
cp.show_config()
```

<details>
<summary>The config should look similar to this:</summary>

```python
OS                        : Linux-5.15.0-70-generic-x86_64-with-glibc2.35
Python Version            : 3.10.12
CuPy Version              : 13.5.1
CuPy Platform             : AMD ROCm
NumPy Version             : 2.2.6
SciPy Version             : None
Cython Build Version      : 3.0.12
Cython Runtime Version    : None
CUDA Root                 : /opt/rocm
hipcc PATH                : /opt/rocm/bin/hipcc
CUDA Build Version        : 60443484
CUDA Driver Version       : 60550421
CUDA Runtime Version      : 60550421 (linked to CuPy) / 60550421 (locally installed)
CUDA Extra Include Dirs   : None
cuBLAS Version            : (available)
cuFFT Version             : 10034
cuRAND Version            : 300400
cuSOLVER Version          : ImportError('libhipblas.so.2: cannot open shared object file: No such file or directory')
cuSPARSE Version          : (available)
NVRTC Version             : (9, 0)
Thrust Version            : 0
CUB Build Version         : 300400
Jitify Build Version      : None
cuDNN Build Version       : None
cuDNN Version             : None
NCCL Build Version        : 22203
NCCL Runtime Version      : 22501
cuTENSOR Version          : None
cuSPARSELt Build Version  : None
Device 0 Name             : AMD Instinct MI355X
Device 0 Arch             : gfx950:sramecc+:xnack-
Device 0 PCI Bus ID       : 0000:75:00.0
Device 1 Name             : AMD Instinct MI355X
Device 1 Arch             : gfx950:sramecc+:xnack-
Device 1 PCI Bus ID       : 0000:05:00.0
Device 2 Name             : AMD Instinct MI355X
Device 2 Arch             : gfx950:sramecc+:xnack-
Device 2 PCI Bus ID       : 0000:65:00.0
Device 3 Name             : AMD Instinct MI355X
Device 3 Arch             : gfx950:sramecc+:xnack-
Device 3 PCI Bus ID       : 0000:15:00.0
Device 4 Name             : AMD Instinct MI355X
Device 4 Arch             : gfx950:sramecc+:xnack-
Device 4 PCI Bus ID       : 0000:f5:00.0
Device 5 Name             : AMD Instinct MI355X
Device 5 Arch             : gfx950:sramecc+:xnack-
Device 5 PCI Bus ID       : 0000:85:00.0
Device 6 Name             : AMD Instinct MI355X
Device 6 Arch             : gfx950:sramecc+:xnack-
Device 6 PCI Bus ID       : 0000:e5:00.0
Device 7 Name             : AMD Instinct MI355X
Device 7 Arch             : gfx950:sramecc+:xnack-
Device 7 PCI Bus ID       : 0000:95:00.0
```
</details>

You can also verify that CuPy can access the GPU device(s) on your system. Test GPU operations by creating a simple array. This code initializes a CuPy array and checks its device location. The `.device` attribute confirms GPU usage if it returns a device ID.

```python
x = cp.array([1, 2, 3])
print("Array device:", x.device)
```

Expected output:

```bash
Array device: <CUDA Device 0>  
```

A non-GPU result (for example, "CPU") indicates an improper setup. Errors such as RuntimeError suggest missing drivers or incompatible hardware.

## A simple “Hello, CuPy” example

To confirm your setup and see CuPy in action, try this quick example that performs some simple matrix algebra and trigonometric operations:

```python
import cupy as cp
# Allocate an array on the GPU
x = cp.arange(12, dtype=cp.float32).reshape(3, 4)

# Apply some element-wise operations
y = cp.sin(x) + cp.cos(x)

# Define a simple custom kernel
kernel = cp.ElementwiseKernel(
    'float32 a, float32 b',
    'float32 c',
    'c = a * b + 1.0f;',
    'axpyish'
)
z = kernel(x, y)

# Synchronize before printing
cp.cuda.get_current_stream().synchronize()
print("z = a*b + 1:\n", z)
```

The output should be as shown below. If the output prints as expected, your environment is ready for GPU-accelerated Python on AMD GPUs.

```python
z = a*b + 1:
 [[  1.          2.3817732   1.9863012  -1.5466175]
 [ -4.6417847  -2.3763103   5.084529   10.876223 ]
 [  7.750866   -3.4911058 -12.830927   -9.95121  ]]
```

The CuPy GitHub repository has many examples to help you test-drive CuPy on AMD Instinct GPUs. This [script](https://github.com/cupy/cupy/blob/main/examples/finance/black_scholes.py) computes prices of European-style stock options using the [Black–Scholes model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model). This [script](https://github.com/cupy/cupy/blob/main/examples/kmeans/kmeans.py) partitions a large dataset of high-dimensional vectors into a given number of clusters using the well-known [k-means clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering). These examples show that you can implement GPU algorithms with CuPy without significantly modifying existing NumPy code.

## Looking ahead: CuPy v14

AMD is merging the following changes into the next official CuPy release (v14):

- ROCm 7 support.
- CUDA Array Interface compatibility
- PTDS support
- Adoption of C++14 as the default backend standard.
- Bug fixes contributed by AMD.

This means that the CuPy v14 upstream version will reach feature parity with AMD’s v13 fork described in this blog. After that, you can take advantage of all the improvements in the AMD CuPy v13 fork by following the installation instructions on the [official CuPy page](https://docs.cupy.dev/en/stable/install.html).

While AMD is working with CuPy to upstream the improvements described here, AMD will continue to support the CuPy v13 wheel. This ensures that developers who depend on a stable v13 environment can remain on that release while upstreaming of changes from the AMD CuPy v13 fork to v14 proceeds.

## Summary

From this blog, you have learned about several key improvements in the AMD CuPy v13 fork, including ROCm 7.0.0 support, PTDS, CAI with proper stream semantics, and a modernized C++14 backend.
You can immediately take advantage of these improvements by installing the latest CuPy wheel. Looking forward, the official CuPy v14 release will bring these capabilities to the upstream repository, while AMD continues to maintain its v13 wheel for those who prefer a stable environment.

## Acknowledgements

The authors would like to acknowledge the AMD teams that supported this work on CuPy: Philipp Samfass, Dominic Etienne Charrier, Michael Obersteiner, Mohammad NorouziArab, Lior Galanti, Matthew Cordery, Jason Riedy, Marco Grond, Bhavesh Lad, Pankaj Gupta, Bhanu Kiran Atturu, Ritesh Hiremath, Radha Srimanthula, Randy Hartgrove, Amit Kumar, Ram Seenivasan, Saad Rahim, Ramesh Mantha.
 
Special thanks to Kenichi Maehashi and his team for their help in adding ROCm support to CuPy.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
