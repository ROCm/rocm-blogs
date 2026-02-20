---
blogpost: true
blog_title: "Blog: FlyDSL - Expert GPU Kernel Development with the Ease of MLIR Python Native DSL on AMD GPUs"
date: 20 Feb 2026
author: 'Felix Li, Shijie Feng, Carlus Huang, Dewei Wang, Hongxia Yang, Peng Sun, Emad Barsoum'
thumbnail: 'flydsl_thumbnail_happyNewYear.png'
tags: Compiler, Optimization, Performance, AI/ML, HPC
category: Software tools & optimizations
target_audience: Developers and AI enthusiasts
key_value_propositions: Python-first, MLIR-native DSL for expert-level AMD GPU kernel development.
language: English
myst:
    html_meta:
        "author": "Felix Li, Shijie Feng, Carlus Huang, Dewei Wang, Hongxia Yang, Peng Sun, Emad Barsoum"
        "description lang=en": "FlyDSL is a Python-first, MLIR-native DSL for expert GPU kernel development and tuning on AMD GPUs."
        "keywords": "FlyDSL, AMD, ROCm, GPU kernels, kernel development, Python DSL, MLIR, FLIR, ROCDL, HSACO, layout algebra, tiling, thread-level control, performance optimization, compiler, Triton, CUTLASS, cuteDSL, GEMM, softmax, layernorm, RMSNorm, quantization, mixture of experts, FlashAttention, FlashInfer, TorchInductor"
        "vertical": "Developers, HPC, AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs, EPYC Server Processors"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Training, Deploying AI at Scale, Generative AI"
        "amd_blog_topic_categories": "HPC & Scientific Computing"
        "amd_blog_authors": "Felix Li, Shijie Feng, Carlus Huang, Dewei Wang, Hongxia Yang, Peng Sun, Emad Barsoum"
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

# FlyDSL: Expert GPU Kernel Development with the Ease of MLIR Python Native DSL on AMD GPUs

The AMD ROCm™ software ecosystem continues to grow rapidly as developers build new kernels, compilers, and AI frameworks optimized for AMD GPUs. As workloads become more complex and the demand for both performance and agility increases, a clear need has emerged for a modern, flexible, and open GPU kernel authoring framework.

Today, we’re excited to introduce [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL): a Python first, MLIR native DSL that aims to make expert level GPU kernel development faster, more intuitive, and more powerful on AMD architectures.

In this blog, we explain what FlyDSL is, why we built it, how it complements existing tools like Triton, and what it enables the ROCm and broader developer community. You’ll also learn about current capabilities, the ecosystem impact, and what’s coming next. If you're eager to dive in, you can start building and installing right away and get hands-on with a quick start.

## What Is FlyDSL and Why It Matters?

FlyDSL (Flexible Layout Python DSL) is a Python DSL and an MLIR stack for authoring high-performance GPU kernels with explicit layouts and tiling.

FlyDSL is powered by FLIR (Flexible Layout Intermediate Representation), an end-to-end, MLIR-native compiler stack for GPU kernels. Its core is the flir dialect, a first-class layout IR with explicit algebra and coordinate mapping, plus a composable lowering pipeline to GPU/ROCDL.

FlyDSL was created to meet several long-standing needs expressed by the open-source and ROCm communities:

### 1. A Familiar Pathway for Developers Coming from Cutlass and CuTe DSL

Many community and customer workloads rely on Cutlass or CuTe DSL. FlyDSL preserves the essential tile based and layout algebra design patterns, allowing developers to:

- Migrate existing kernels with minimal redesign
- Reuse familiar abstractions on AMD hardware
- Maintain predictable performance behavior

This dramatically reduces friction when bringing projects such as FlashAttention, FlashInfer, or custom GEMM/attention kernels into the ROCm ecosystem.

### 2. A Modern, Python-based Alternative to Template-heavy HIP C++

Template based kernel frameworks like CK (Composable Kernel) are powerful but come with known challenges: long build times, slow iteration cycles, brittle compiler interactions, and steep onboarding requirements.

FlyDSL addresses these issues by providing the following (see also Figure 1):

- A native Python DSL for expressing kernels
- AST transforms to convert Pythonic control flow into MLIR
- JIT friendly compilation, dramatically reducing iteration time
- Clear MLIR → ROCDL → HSACO lowering pipeline designed for AI workloads

This results in faster kernel development and more predictable experimentation.

```{figure} ./images/flydsl-figure1-compilation-flow.svg
:align: center
:alt: compilation flow
Figure 1: The FlyDSL Compilation flow
```

### 3. A Complementary Tool, not a Replacement for Triton

AMD and OpenAI actively collaborate on Triton as the primary block level kernel DSL for most developers. Triton excels in productivity and high-level operator development.

FlyDSL intentionally targets a different layer:

- **Triton**: block-level programming for mainstream developers
- **FlyDSL**: thread-level and IR-level control for expert developers seeking roofline performance or working on compiler infrastructure

By focusing on explicit lane control, register usage, custom layouts, and ISA level hints, FlyDSL enables performance tuning that lies outside Triton’s abstraction boundary.

## Built on CuTe Layout Algebra

FlyDSL incorporates the formally validated CuTe layout algebra [[1]](#references), giving developers a unified mathematical foundation for expressing tensor layouts. This ensures:

- Consistent representation across kernel families
- Predictable optimization behavior
- Portability across GPU architectures

CuTe layout algebra provides the structural rigor needed for advanced kernel tuning.

## Current Status

FlyDSL already supports several essential AI operators with performance competitive with, or exceeding, CK-based implementations. These include:

- Softmax
- LayerNorm / RMSNorm
- Quantization
- GEMM
- Mixture of Experts (MOE) kernels

The underlying thread level IR is nearly complete, and early demos of transpose, elementwise, and quant kernels using layout-based transformations are fully functional.

MLIR-based tracing, lowering, and code generation through ROCDL are also working end-to-end with continuous CI integration.

## Ecosystem Impact

FlyDSL opens a smoother path for AMD enablement across many open-source projects already based on CuTe DSL or Cutlass like abstractions. These include:

- FlashInfer kernels (GEMM, fused reduce)
- FlashAttention
- Dao Lab's Quack/Quark kernels
- TorchInductor's CuTe DSL backend
- TileLang's new CuTe DSL backend

It also accelerates Cutlass-derived ODM workloads such as DeepGEMM, FlashMLA, and XFormers.

Whether you're maintaining a high-performance LLM operator library or exploring new fused kernels, FlyDSL provides a clear on-ramp into the ROCm ecosystem.

## Getting Started with FlyDSL

In this section, you can jump straight into building and installing the components, giving you a quick, hands-on start on FlyDSL.

### Install from a Wheel

To install FlyDSL, run the following pip command [[2]](#references):

```bash
pip install flydsl
```

Now you are ready to try a simple example below:

```python
"""Simple example demonstrating fused add + relu operation in FlyDSL."""

from flydsl.compiler.context import RAIIMLIRContextModule

from flydsl.dialects.ext import flir, arith

from _mlir.ir import InsertionPoint

import _mlir.extras.types as T

import os

# Set up MLIR context

ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)

# Define the fused add + relu function

with InsertionPoint(ctx.module.body):

    @flir.jit(T.f32(), T.f32())

    def fused_add_relu(x, y):

        # Add the two inputs

        sum_val = x + y

        # ReLU: max(x, 0) using arith.maximum

        zero = arith.f32(0.0)

        result = arith.maximum(sum_val, zero)

        return result

# Print the generated MLIR

print("Generated MLIR:")

print(ctx.module)

print("\n" + "="*50 + "\n")

# Verify the module

try:

    ctx.module.operation.verify()

    print("✓ Module verification passed!")

except Exception as e:

    print(f"✗ Module verification failed: {e}")

os._exit(0)
```

You will see the output below:

```mlir
Generated MLIR:

module {

  func.func @fused_add_relu(%arg0: f32, %arg1: f32) -> f32 {

    %cst = arith.constant 0.000000e+00 : f32

    %0 = arith.addf %arg0, %arg1 : f32

    %1 = arith.maximumf %0, %cst : f32

    return %1 : f32

  }

}

==================================================

✓ Module verification passed!
```

In this example, you learned how to use FlyDSL to define a GPU-optimized function (fused add + ReLU) with MLIR operations and verify the generated IR.

### Building from Source

The pip package (flydsl) supports Python 3.10 or 3.12, glibc >= 2.35. Alternatively, you can build from source.

Figure 2 below outlines the full building from source workflow, including building MLIR, building FLIR, installing the Python package, and running tests.

```{figure} ./images/flydsl-figure2-getting-started-flow.svg
:align: center
:alt: build from source
Figure 2: Getting started with FlyDSL by building from source
```

Please check [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) for details.

## Summary

FlyDSL marks an important step forward in AMD’s mission to deliver an open, modern, and high-performance GPU programming experience.

In this blog, you learned how FlyDSL brings together Python’s ease of use, the mathematical rigor of CuTe‑style layout algebra, an explicit thread‑level IR for fine‑grained tuning, and a clean MLIR‑native compilation pipeline, providing a powerful and familiar workflow for developers coming from the Cutlass and CuTe DSL ecosystems. These foundations enable FlyDSL to simplify kernel development and prepare for future extensibility, including layout‑agnostic designs and support for diverse workload‑optimized strategies.

Our roadmap and future work include but are not limited to:

## Language and Compiler

- MFMA, Atom, and additional intrinsic support
- Expanded AST-transform coverage for Python syntax
- Separation of platform-agnostic vs. platform-specific components
- Exploring a layout agnostic design to support multiple layout strategies

## Kernel Projects

- Finalize GEMM and MOE kernels
- Upcoming support for attention, AR+GEMM, and more complex fusions
- Integration with AITER, vLLM, sglang, ATOM
- Performance breakthroughs for MLA and ASM-only kernels
- Ongoing LLVM and ROCDL codegen improvements

Whether you’re developing cutting edge research kernels, optimizing operators for large-scale LLM workloads, or contributing to compiler infrastructure, FlyDSL opens new possibilities and streamlines the developer experience across the ROCm ecosystem.

We’re excited for what comes next, and even more excited to see what you build with FlyDSL. Contributions, feedback, and community engagement will shape the next stages of this project, and we look forward to growing it together.

## Acknowledgements

FLIR's design is inspired by ideas from several projects:

- Categorical Foundations for CuTe Layouts [1] – mathematical framework for layout algebra (companion code)
- NVIDIA CUTLASS – CuTe layout algebra concepts (BSD-3-Clause parts only; no EULA-licensed code was referenced)
- Triton – Python DSL for GPU kernel authoring
- ROCm Composable Kernel – tile-based kernel design patterns for AMD GPUs

## References

1. [https://arxiv.org/abs/2601.05972](https://arxiv.org/abs/2601.05972)

2. [https://pypi.org/project/flydsl](https://pypi.org/project/flydsl)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

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
