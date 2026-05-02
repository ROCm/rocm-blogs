---
blogpost: true
blog_title: "From Naive to Near-Peak: Building High-Performance GEMM Kernels with Gluon"
date: 28 Apr 2026
author: 'Lixun Zhang'
thumbnail: 'gluon-gemm-performance-progression.png'
tags: AI/ML, Linear Algebra, Performance, Profiling, Optimization, Hardware, Compiler
category: Software tools & optimizations
target_audience: GPU kernel developers, ML compiler engineers, and performance engineers
key_value_propositions: Learn how a hands-on Gluon tutorial teaches profiling-driven GEMM optimization on AMD GPUs.
language: English
myst:
    html_meta:
        "author": "Lixun Zhang"
        "description lang=en": "Learn how a Gluon GEMM tutorial teaches profiling-driven AMD GPU optimization from FP16 baseline to BF8 and MXFP4 kernels."
        "keywords": "Gluon, GEMM, ROCm, Triton, AMD GPUs, MFMA, LDS, profiling, optimization"
        "vertical": "HPC"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Inference, AI Training"
        "amd_blog_topic_categories": "Software & Ecosystem, AI & Intelligent Systems"
        "amd_blog_authors": "Lixun Zhang"
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

# From Naive to Near-Peak: Building High-Performance GEMM Kernels with Gluon

High-performance GPU kernels are built through measurement, not guesswork. The
[`gfx950-gluon-tutorials`](https://github.com/ROCm/gfx950-gluon-tutorials) repository
is a hands-on ROCm tutorial that shows how to write, profile, and optimize Gluon
GEMM kernels on AMD Instinct GPUs. Starting from a correctness-first FP16 GEMM,
the tutorial walks through memory movement, LDS layout design, software
pipelining, register pressure, MFMA efficiency, and XCD-aware L2 locality. The
same design ideas then extend to BF8 and MXFP4 kernels for low-precision AI
workloads.

This blog is the map. The repository is the full tutorial. Here we explain what
the tutorial covers, why Gluon is useful for this style of kernel development,
and how profiling turns each optimization step into an engineering decision.

## Why another GEMM tutorial?

ROCm already provides optimized linear algebra libraries and tuning workflows
for production workloads. Those tools should be the first stop for most users.
Kernel developers, compiler engineers, and performance specialists often need a
different kind of resource: they need to understand how a high-performance
kernel is constructed, what the hardware bottleneck is at each step, and why a
specific code change improves the result.

The Gluon GEMM tutorial is designed for that second audience. It does not start
with the final kernel. It starts with a simple FP16 GEMM that is correct but far
from optimal, then builds performance one measured bottleneck at a time. Each
version isolates one idea so that readers can connect code structure to
hardware behavior.

The tutorial focuses on AMD MI350 and MI355 GPUs, using gfx950 as the target
architecture. The main kernels are:

| Kernel | Data type | Shape used in the summary | Documented result |
| --- | --- | --- | --- |
| `a16w16` | FP16 | `4096x4096x8192` | `1619 TFLOPS`, `98%` MFMA efficiency |
| `a8w8` | BF8 | `4096x4096x16384` | `3456 TFLOPS`, `99%` MFMA efficiency |
| `a4w4` | MXFP4 | `4096x4096x32768` | `5728 TFLOPS`, `92%` MFMA efficiency |

These numbers are measurements from the tutorial configuration and should be
read with the repository's documented ROCm, Triton, and profiling setup. The
main value of the tutorial is not only the final number, but the path from
baseline to that number.

```{figure} ./images/gluon-gemm-performance-progression.png
:align: center
:alt: FP16 GEMM performance progression from the naive baseline to the optimized tutorial kernel

FP16 GEMM performance progression across the tutorial versions.
```

## What Gluon makes explicit

Gluon is a block-level programming model in Triton. In a conventional
thread-level GPU kernel, the compiler receives code that describes what each
thread should compute and then has to recover scheduling, register allocation,
and memory movement opportunities from that lower-level representation. That
works well for many kernels, but high-performance GEMM often needs very precise
control over layouts, pipeline stages, and live ranges.

Gluon raises the authoring level to tiles and block-level operations. The
kernel author describes how data moves between global memory, LDS, registers,
and MFMA instructions. Layouts are explicit. Pipeline stages are explicit.
Register budgeting becomes part of the kernel design instead of something that
is discovered only after the backend compiler has lowered the code.

That explicit control is the central teaching point of the repository. The
tutorial repeatedly asks:

* Which instruction should move this data?
* Which layout avoids LDS bank conflicts?
* Which pipeline stage hides global memory or LDS latency?
* How many registers are live at the MFMA boundary?
* What does the trace show after the change?

The result is a workflow where hardware reasoning, source code, generated code,
and profiler data all stay connected.

## The FP16 path: one bottleneck at a time

The `a16w16` tutorial is the recommended starting point. It is organized as a
versioned optimization journey from `v0_naive` to `v9_beyond_hotloop`.

The early versions focus on getting data movement right. `v0_naive` establishes
a correct baseline with explicit layouts. `v1_buffer_load` switches masked
loads to AMD buffer operations so out-of-bounds handling can be done by
hardware instead of control-flow branches. `v2_async_copy` moves data directly
from global memory to LDS, avoiding register staging and eliminating unnecessary
`ds_write` instructions. `v3_lds` then studies LDS layout choices and bank
conflicts.

The middle versions focus on overlap. `v4_global_prefetch` introduces a
two-stage pipeline so the kernel can prefetch data for the next K iteration
while computing on the current one. `v5_local_prefetch` adds a third stage so
MFMA compute, LDS reads, and global memory movement can overlap. At that point,
instruction ordering becomes a first-class performance problem, so the tutorial
uses the LLIR scheduler described in the repository to interleave MFMA and
memory operations according to the hardware throughput model.

The later versions focus on register pressure and locality. `v6_loop_unroll`
removes copy overhead at iteration boundaries. `v7_sliceN` and `v8_sliceMN`
reduce register pressure by slicing the tile structure. `v9_beyond_hotloop`
looks outside the hot loop and improves L2 locality through XCD-aware workgroup
remapping.

```{figure} ./images/gluon-gemm-slicemn-design.png
:align: center
:alt: M and N slicing design used by the optimized FP16 Gluon GEMM kernel

M and N slicing reduce register pressure and structure the pipeline around
smaller operand regions.
```

## Profiling drives the tutorial

The tutorial is intentionally measurement-heavy. Instead of treating TFLOPS as
the only signal, it tracks the evidence needed to explain a result:

* MFMA efficiency from thread traces
* VGPR usage and spills
* generated LLVM IR, AMDGCN, and assembly
* rocprof kernel timing
* hardware counters for cache and memory behavior
* ATT screenshots for instruction-level bottleneck analysis

This matters because the same end-to-end runtime can hide very different
problems. A kernel can be limited by LDS bank conflicts, global memory latency,
register copies, spilled values, missing interleaving, or L2 locality. The fix
depends on identifying the real bottleneck.

The repository includes helper scripts for this workflow. For example,
`scripts/run_perf_table.py` runs selected kernel versions under different
scheduler configurations and reports TFLOPS, VGPRs, spills, and MFMA
efficiency. `scripts/process_json.py` analyzes ATT output and computes loop
timing breakdowns. The goal is to make the optimization process reproducible,
not only the final kernel.

## Applying the design to BF8 and MXFP4

After the FP16 path, the repository shows how the same design transfers to
lower precision formats.

The BF8 kernel keeps the same high-level structure but changes the tile shape,
MFMA instruction, K width, and LDS padding. This part of the tutorial is a
checklist proof: if you understand the FP16 design, the BF8 design follows from
the changed instruction shape and data type.

The MXFP4 kernel adds a genuinely new problem: scales. MXFP4 stores two 4-bit
values per byte and uses per-group 8-bit scale factors. The data path now needs
a scale pipeline in addition to the tile pipeline. Scales are loaded from global
memory into registers, written to LDS, and read back in the layout required by
the scaled MFMA instruction. The tutorial refers to this as a GR -> LW -> LR
round trip.

```{figure} ./images/gluon-gemm-mxfp4-pipeline.png
:align: center
:alt: MXFP4 scale pipeline used by the Gluon GEMM tutorial

The MXFP4 tutorial adds a scale pipeline on top of the inherited tile pipeline.
```

This is a useful example because it shows how the same method extends beyond a
single clean FP16 GEMM. The author still reasons about layout, pipeline stage,
register lifetime, instruction throughput, and profiler evidence, but the
kernel now has an additional dataflow to schedule.

## Try the tutorial

To start, clone the tutorial repository and follow the setup instructions in
the README. The peak numbers use the ROCm Triton branch and pinned tag
documented in the repository.

```bash
git clone https://github.com/ROCm/gfx950-gluon-tutorials.git
cd gfx950-gluon-tutorials
```

The FP16 tutorial is the best first stop:

```bash
cd kernels/gemm/a16w16
python bench.py --version 9 --K 8192 --dtype fp16 --use-rocprof
```

For a broader performance table, run from the repository root:

```bash
python scripts/run_perf_table.py \
  --kernel a16w16 \
  --versions 5 6 7 8 9 \
  --configs base llir llir+amdgcnas \
  --K 8192 \
  --dtype fp16 \
  --use-rocprof
```

The recommended reading order is:

1. Start with `kernels/gemm/a16w16/README.md`.
2. Read each version README from `v0_naive` through `v9_beyond_hotloop`.
3. Compare the code changes with the profiler evidence.
4. Move to `a8w8` for BF8 and `a4w4` for MXFP4.
5. Use the `docs/` directory when you want a deeper model for MFMA efficiency,
   LDS throughput, or memory bandwidth.

## Summary

The `gfx950-gluon-tutorials` repository is a practical guide to AMD GPU kernel
optimization with Gluon. It teaches the full path from a simple FP16 GEMM to
high-performance FP16, BF8, and MXFP4 kernels by connecting each optimization
to profiler evidence and hardware behavior.

The main takeaway is that near-peak GEMM performance is not one trick. It is a
sequence of design decisions: use the right data movement instruction, choose
the right LDS layout, overlap memory and compute, control register pressure,
measure MFMA efficiency, and look beyond the hot loop when locality matters.
Gluon makes those decisions explicit, which makes the optimization process
teachable and reproducible.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
