---
blogpost: true
blog_title: "Introducing the ROCm Handbooks"
date: "10 Nov 2026"
author: "Jan Stephan, Adel Johar, Istvan Kiss"
thumbnail: 'rocm-handbooks-intro-thumbnail.webp'
tags: "C++, Optimization, Performance, Compiler, Profiling, HPC, Fine-Tuning"
category: "Software tools & optimizations"
target_audience: "Developers who program AMD GPUs with HIP, from newcomers with C++ experience to experienced HIP programmers and AI developers who optimize low-level kernels."
key_value_propositions: "The ROCm handbooks turn the ROCm documentation into book-style guides, in HTML and PDF, that lead developers from their first HIP kernel to architecture-specific GPU optimization."
language: English
myst:
    html_meta:
        "author": "Jan Stephan, Adel Johar, Istvan Kiss"
        "description lang=en": "Discover the ROCm handbooks: book-style HTML and PDF guides that take you from your first HIP kernel to optimizing code for AMD GPUs."
        "keywords": "ROCm, HIP, ROCm handbooks, AMD ROCm Programming Guide, AMD ROCm Optimization Guide, GPU programming, GPU optimization, HIP tutorial, HIP kernel, AMD GPU, Instinct, Radeon, CDNA, RDNA, MFMA, WMMA, compiler builtins, performance optimization, PDF documentation"
        "vertical": "Developers, AI, HPC"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs, Radeon Graphics"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, AI Inference"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Jan Stephan, Adel Johar, Istvan Kiss"
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

# Introducing the ROCm Handbooks

Documentation is organized by topic. Learning follows a sequence.

The documentation for the
[AMD ROCm™ platform](https://rocm.docs.amd.com/en/latest/) is organized by
component. Each component documents itself: the
[HIP documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/),
for example, explains the basics of GPU programming with clear concepts and
small tutorials, right next to its API reference. From the
ROCm documentation landing page, every component is only a few clicks away.

The **ROCm handbooks** bring a book-style format to ROCm: they draw on the
documentation of several components, arrange the material along one learning
path, and complement the official documentation. In this blog, you'll meet the
two ROCm handbooks, see what each volume covers, and find out which one to open
first, whether you're writing your first HIP kernel or tuning one for a specific
AMD GPU architecture.

## What the ROCm Handbooks Are

The ROCm handbooks target developers who require a unified, book-style
reference for ROCm and HIP. They aggregate documentation from the ROCm portal
and organize it into a structured format optimized for in-depth study and
offline access, available in both PDF and HTML.

Each handbook collects the relevant pages from the documentation
portal, adds chapters written specifically for the book, and arranges
everything along a learning path.

The handbooks release together with ROCm. When a new ROCm version ships, the
matching handbook versions ship with it, so the content you read matches the
software you install.

The series currently has two volumes:

- The **AMD ROCm Programming Guide** teaches HIP from the basics. It's the
  first entry in the series and has been available since November 2025.
- The **AMD ROCm Optimization Guide** teaches you how to make HIP code fast on
  AMD GPUs. It's the second volume in the series, published in October 2026.

## AMD ROCm Programming Guide

The [AMD ROCm Programming Guide](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/latest/)
is for developers who are new to ROCm and HIP. You should be comfortable with
programming in general and with C++ in particular.

The guide introduces the core concepts, APIs, and recommended practices of
[HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/). Its examples run on
any AMD GPU that the ROCm platform supports, from AMD Radeon™ GPUs in workstations to
AMD Instinct™ GPUs in data centers. The guide has four parts:

- **[Getting started](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/latest/conceptual/introduction.html)** explains what ROCm and HIP are, introduces the parallel
  programming model, and shows how HIP maps onto the GPU hardware. It ends with
  a tutorial that walks you through your first complete HIP program.
- **[Core HIP programming](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/latest/how-to/core-introduction.html)** covers the HIP C++ language extensions and the HIP
  runtime API: initialization, memory management, error handling, streams and
  asynchronous execution, cooperative groups, HIP graphs, and interoperability
  with graphics APIs. The memory management chapter alone spans host, device,
  unified, and virtual memory as well as the stream-ordered allocator. Five
  tutorials then apply this knowledge to common programming patterns: matrix
  multiplication, histograms with atomic operations, k-means clustering across
  CPU and GPU, stencil operations, and a breadth-first search that chains
  multiple kernels. The part closes with an overview of the available HIP
  compilers.
- **[Performance optimization](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/latest/tutorial/hip-performance-optimization.html)** introduces you to tuning. Three
  tutorials take an image gamma correction kernel and a matrix transpose and
  improve them step by step, using grid-stride loops and tiling.
- **[ROCm platform](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/latest/install/rocm.html)** covers installing ROCm, the ROCm Runfile Installer,
  building ROCm from source, and third-party tools.

## AMD ROCm Optimization Guide

The [AMD ROCm Optimization Guide](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/)
continues where the Programming Guide ends. It's for experienced ROCm and HIP
programmers, for AI developers who work on low-level implementations and
optimizations, and for anyone with existing HIP code that needs to be optimized
for AMD GPUs. You should know how to launch kernels, index threads, and manage
device memory.

The guide starts from one principle: profile first. You measure where your
application spends its time, and only then do you apply targeted
optimizations. The guide has three parts:

- **[Getting started](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/conceptual/introduction.html)** explains the
  fundamentals of GPU performance on AMD hardware and collects general
  performance guidelines for memory access and thread organization. A chapter
  on multi-GPU programming covers device selection, synchronization, and memory
  handling across several GPUs.
- **[Performance optimization patterns](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/patterns/introduction.html)** form
  the core of the guide. Each pattern is a complete optimization journey for
  one fundamental algorithm. You start with a naive implementation that works,
  then apply one optimization at a time. For most steps, the guide lists the
  profiler counters to watch, so you can measure the effect on your own GPU.
  The three patterns build on each other:

  - **Reduction** introduces the fundamentals: avoiding thread divergence,
    resolving bank conflicts in the Local Data Share (LDS), performing
    reductions within a wavefront, and vectorizing memory loads.
  - **Histogram** adds concurrent updates. You move from atomic operations in
    global memory to atomics in LDS and finally to per-block partial histograms
    that a second kernel merges.
  - **Matrix multiplication** combines everything in seven steps on 4096 x 4096
    single-precision matrices. After a naive baseline, six optimizations follow:
    tiling in LDS, register tiling, double
    buffering, vectorized loads, tuning register pressure with
    `__launch_bounds__`, and finally a generic, policy-based kernel template
    that dispatches by GPU architecture.

  The patterns use standard HIP only, so they run on any GPU architecture that
  ROCm supports, unless a step shows architecture-specific functionality.
- **[Compiler builtins](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/introduction.html)** give you
  direct access to GPU instructions that the compiler doesn't always generate on its
  own. This part is both a reference and a how-to guide, organized by
  architecture:

  - **Cross-architecture builtins** cover arithmetic (sum of absolute
    differences, integer and floating-point dot products, conversion and
    packing), direct loads from global memory into LDS, and wavefront
    operations: shuffles, Data Parallel Primitives (DPP), lane permutations,
    reductions, and votes.
  - **Matrix Fused Multiply-Add (MFMA) builtins** cover the matrix
    instructions of the CDNA™ architectures, from AMD Instinct MI100 GPUs (CDNA)
    to the AMD Instinct MI350 Series (CDNA 4), in dense and sparse variants.
  - **Wave-Matrix Multiply-Accumulate (WMMA) builtins** cover the matrix
    instructions of the RDNA™ 3 and RDNA 4 architectures: dense variants on
    both, plus sparse variants and transpose loads on RDNA 4.

  Each matrix chapter for a specific architecture includes a worked example: it
  plugs that architecture's MFMA or WMMA instructions into the generic matrix
  multiplication kernel from the patterns part. The introduction to the
  builtins also discusses their cost.

## Where to Start

Your starting point depends on your experience:

- **You're new to HIP.** Start with the
  [AMD ROCm Programming Guide](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/latest/).
  Read *Getting started* and the *SAXPY tutorial*, then work through
  *Core HIP programming*. The programming-pattern tutorials show you how the
  pieces fit together.
- **You're already writing HIP code.** Skim the parts of the
  Programming Guide that are new to you, then move to the
  [AMD ROCm Optimization Guide](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/).
  Work through the patterns in order: reduction, histogram, matrix
  multiplication.
- **You're tuning compute-bound kernels on AMD Instinct or AMD Radeon GPUs.**
  Go to the compiler builtins in the Optimization Guide and pick the chapter
  for your architecture.

The two volumes complement each other. The Programming Guide points you to the
Optimization Guide for profiling-driven optimization patterns and multi-GPU
programming. The Optimization Guide assumes familiarity with the HIP basics
covered in the Programming Guide: kernel launches, thread indexing, and memory
management.

## Get the Handbooks

Both handbooks are available in HTML and PDF formats:

| Handbook | HTML | PDF |
| --- | --- | --- |
| AMD ROCm Programming Guide | [Read online](https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/latest/) | [Download PDF](https://rocm-handbook.amd.com/_/downloads/amd-rocm-programming-guide/en/latest/pdf/) |
| AMD ROCm Optimization Guide | [Read online](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/) | [Download PDF](https://rocm-handbook.amd.com/_/downloads/amd-rocm-optimization-guide/en/latest/pdf/) |

The sources live on GitHub in the
[AMD ROCm Programming Guide repository](https://github.com/ROCm/amd-rocm-programming-guide)
and the
[AMD ROCm Optimization Guide repository](https://github.com/ROCm/amd-rocm-optimization-guide).
If you find a mistake or a missing topic, open an issue in the respective
repository. The example code is in the [ROCm examples repository](https://github.com/ROCm/rocm-examples).

## Summary

In this blog, you learned that the ROCm handbooks turn the ROCm documentation
into books that you can read from cover to cover. You saw that the AMD ROCm
Programming Guide takes you from your first kernel to the HIP runtime API, and
that the AMD ROCm Optimization Guide teaches you to profile, to optimize three
fundamental algorithms step by step, and to use compiler builtins and matrix
instructions on AMD Instinct and AMD Radeon GPUs. Both come as HTML and PDF, and
both are released alongside ROCm.

Pick the volume that matches your experience, download the PDF, and start
reading. Once you know your way around, the documentation of each component is
there for the details. Because the handbooks ship with every ROCm release, come
back with each new version to see what has changed in the books and in the
software they describe.

---

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED “AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, AMD Instinct, AMD Radeon, ROCm, CDNA, RDNA, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. GITHUB is a trademark of GitHub, Inc.
© 2026 Advanced Micro Devices, Inc. All rights reserved
