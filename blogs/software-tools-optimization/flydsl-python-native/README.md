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

FlyDSL is powered by the Fly dialect: an end-to-end, MLIR-native compiler stack for GPU kernels. Its core is the `fly` dialect, a first-class layout IR with explicit algebra and coordinate mapping, plus a composable lowering pipeline to GPU/ROCDL.

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
- Clear MLIR → Fly → ROCDL → HSACO lowering pipeline designed for AI workloads

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

FlyDSL-based high-performance operators have entered early production adoption for **large-scale** inference workloads. These deployments operate at production hyperscale across MI GPU clusters, demonstrating scalability and production readiness.

## Ecosystem Impact

**Firstly**, FlyDSL opens a smoother path for AMD enablement across many open-source projects already based on CuTe DSL or Cutlass like abstractions. These include:

- FlashInfer kernels (GEMM, fused reduce)
- FlashAttention
- Dao Lab's Quack/Quark kernels
- TorchInductor's CuTe DSL backend
- TileLang's new CuTe DSL backend

It also accelerates Cutlass-derived ODM workloads such as DeepGEMM, FlashMLA, and XFormers.

**Secondly**, the FlyDSL project is also collaborating with our industry partners to actively design and incorporate more high performance layout variants in the future. These efforts are complementary to Triton Linear layouts and aim to extend their flexibility and performance coverage. This helps pave the way for continuous evolution and drives open source collaboration at both the kernel development and DSL solution levels.

**In summary**, whether you're maintaining a high-performance LLM operator library or exploring new fused kernels, FlyDSL provides a clear on-ramp into the **ROCm ecosystem**.

## Getting Started with FlyDSL

In this section, you can jump straight into building and installing the components, giving you a quick, hands-on start on FlyDSL.

### Install from a Wheel

To install FlyDSL, run the following pip command [[2]](#references):

```bash
pip install flydsl
```

Now you are ready to try a simple example below:

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def vectorAddKernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    A = fx.rocdl.make_buffer_tensor(A)

    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
    tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))

    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))
    tC = fx.logical_divide(tC, fx.make_layout(1, 1))

    RABMemRefTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)

    copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    copyAtomBuffer = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

    rA = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))
    rB = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))
    rC = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))

    fx.copy_atom_call(copyAtomBuffer, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)

    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))


@flyc.jit
def vectorAdd(
    A: fx.Tensor,
    B: fx.Tensor,
    C,  # omitted for auto induction
    n: fx.Int32,  # dynamic int32
    const_n: fx.Constexpr[int],  # static int32, it has an effect on function cache-key
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 64
    grid_x = (n + block_dim - 1) // block_dim

    vectorAddKernel(A, B, C, block_dim).launch(grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream)

# Usage
n = 128
A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
C = torch.zeros(n, dtype=torch.float32).cuda()
tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)
vectorAdd(tA, B, C, n, n + 1, stream=torch.cuda.Stream())
torch.cuda.synchronize()
print('Result correct:', torch.allclose(C, A + B))
```

In this example, you can see how FlyDSL expresses a vectorized GPU kernel using CuTe-style layout algebra: tensors are partitioned hierarchically by block and thread, data is moved through typed register buffers via copy atoms, and arithmetic is performed on register-resident vectors, all within a clean Python DSL that compiles through the Fly dialect to HSACO.

### Building from Source

The pip package (flydsl) supports Python 3.10+. Alternatively, you can build from source.

Figure 2 below outlines the full building from source workflow, including building MLIR, building FlyDSL, installing the Python package, and running tests.

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

### Language and Compiler

- MFMA, Atom, and additional intrinsic support
- Expanded AST-transform coverage for Python syntax
- Separation of platform-agnostic vs. platform-specific components
- Exploring a layout agnostic design to support multiple layout strategies

### Kernel Projects

- Finalize GEMM and MOE kernels
- Upcoming support for attention, AR+GEMM, and more complex fusions
- Integration with AITER, vLLM, sglang, ATOM
- Performance breakthroughs for MLA and ASM-only kernels
- Ongoing LLVM and ROCDL codegen improvements

Whether you’re developing cutting edge research kernels, optimizing operators for large-scale LLM workloads, or contributing to compiler infrastructure, FlyDSL opens new possibilities and streamlines the developer experience across the ROCm ecosystem.

We’re excited for what comes next, and even more excited to see what you build with FlyDSL. Contributions, feedback, and community engagement will shape the next stages of this project, and we look forward to growing it together.

## Acknowledgements

FlyDSL's design is inspired by ideas from several projects:

- Categorical Foundations for CuTe Layouts [1] – mathematical framework for layout algebra (companion code)
- NVIDIA CUTLASS – CuTe layout algebra concepts (BSD-3-Clause parts only; no EULA-licensed code was referenced)
- Triton – Python DSL for GPU kernel authoring
- ROCm Composable Kernel – tile-based kernel design patterns for AMD GPUs
- ROCm Aiter – test infrastructure and performance comparison baselines

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

```{update} Mar 30, 2026
Updated the blog to match FlyDSL's current API and terminology
```
