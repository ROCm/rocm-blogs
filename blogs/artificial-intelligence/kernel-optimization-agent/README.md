---
blogpost: true
blog_title: "GEAK V3: Agent-Driven, Repository-Level GPU Kernel Optimization across HIP, Triton, and FlyDSL on AMD GPUs"
date: "20 Jul 2026"
author: "Saptarshi Majumder, Umang Pandey, Subrahmanya Pavankumar Dubagunta, Aditya Kumar Singh, Yue Liu, Chao Xu, Ethan Yang, Dan Li, Jianghui Wang, HongTao Meng, Puyuan Yang, Fan Wang, Bin Ding, Zheng Yao, Arthur Huang, Fuwei Yang, Manem Chaitanya, Johanna Xuyue Yang, Kristoffer Peyron, Balazs Toth, Yashvardhan Agarwal, Baiqiang Xia, Mehdi Saeedi, Ziqiong Liu, Arsalan Farooq, Pratik Prabhanjan Brahma, Dong Li, Zicheng Liu, Emad Barsoum"
thumbnail: 'geak-v3-teaser.png'
tags: "AI/ML, GenAI, LLM"
category: "Applications & models"
target_audience: "AI developers, Kernel Engineers, Infra and Efficiency folks, and enthusiasts"
key_value_propositions: "This blog introduces GEAK-v3 that expands GEAK-v2, a kernel-generation assistant, into a fully agent-driven framework for end-to-end GPU kernel optimization in real codebases, producing reviewable patches backed by profiling, testing, and LLM-guided iteration.   Now supports three kernel languages - HIP, Triton, and FlyDSL - within a single unified optimization workflow."
language: English
myst:
    html_meta:
        "author": "Saptarshi Majumder, Umang Pandey, Subrahmanya Pavankumar Dubagunta, Aditya Kumar Singh, Yue Liu, Chao Xu, Ethan Yang, Dan Li, Jianghui Wang, HongTao Meng, Puyuan Yang, Fan Wang, Bin Ding, Zheng Yao, Arthur Huang, Fuwei Yang, Manem Chaitanya, Johanna Xuyue Yang, Kristoffer Peyron, Balazs Toth, Yashvardhan Agarwal, Baiqiang Xia, Mehdi Saeedi, Ziqiong Liu, Arsalan Farooq, Pratik Prabhanjan Brahma, Dong Li, Zicheng Liu, Emad Barsoum"
        "description lang=en": "Explore GEAK v3: agent-driven, repository-level GPU kernel optimization across HIP, Triton, and FlyDSL on AMD Instinct™ GPUs."
        "keywords": "Kernel Optimization, HIP, Triton, FlyDSL, GEAK, LLM, Inference, Efficiency"
        "vertical": "AI, Developers, Data Science"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "Open-Source Tools"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Saptarshi Majumder, Umang Pandey, Subrahmanya Pavankumar Dubagunta, Aditya Kumar Singh, Yue Liu, Chao Xu, Ethan Yang, Dan Li, Jianghui Wang, HongTao Meng, Puyuan Yang, Fan Wang, Bin Ding, Zheng Yao, Arthur Huang, Fuwei Yang, Manem Chaitanya, Johanna Xuyue Yang, Kristoffer Peyron, Balazs Toth, Yashvardhan Agarwal, Baiqiang Xia, Mehdi Saeedi, Ziqiong Liu, Arsalan Farooq, Pratik Prabhanjan Brahma, Dong Li, Zicheng Liu, Emad Barsoum"
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

# GEAK V3: Agent-Driven, Repository-Level GPU Kernel Optimization across HIP, Triton, and FlyDSL on AMD GPUs

In the ever-evolving world of GPU computing, optimizing kernels for
performance and efficiency is a critical challenge. Hand-tuning kernels
demands deep technical expertise and manual iteration. In this blog, you will read about how [GEAK v3](https://github.com/AMD-AGI/GEAK/tree/v3.2.2),
the latest iteration of the agent-driven framework, tackles this problem
using enhanced features such as task planning, test-harness discovery,
patch-based handling of multi-file kernels, dynamic memory system and
expert knowledge database. Our results show improvements across three
kernel languages (HIP, Triton, and FlyDSL) and both CDNA and RDNA GPUs.

## Key Takeaways

- [GEAK v3](https://github.com/AMD-AGI/GEAK/tree/v3.2.2) evolves from a
  kernel-generation assistant into a highly automated agent-driven
  framework for end-to-end GPU kernel optimization in real codebases at
  the repository levels, producing reviewable patches backed by profiling,
  testing, and LLM-guided iteration.
- Now supports three kernel languages — HIP, Triton, and FlyDSL — as well
  as both [CDNA](https://www.amd.com/en/technologies/cdna.html) and
  [RDNA](https://www.amd.com/en/technologies/rdna.html) GPUs, within a
  single unified optimization workflow.
- GEAK v3 achieves 3.02× and 2.22× speedups on HIP and Triton benchmarks,
  respectively, which include kernels spanning a wide range of complexity
  levels, from individual kernels to full repository-level workloads.
- GEAK is already supporting various production optimization use cases, including:
  1. [GPT-OSS 120B](https://huggingface.co/openai/gpt-oss-120b) workload
     in the MLPerf inference submission, resulting in [8-10% uplift due to attention kernel optimization](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html).
  2. Optimized the [DeepSeek V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
     workload on MI355 (ISL/OSL = 8k/1k, TP = 8, concurrency = 32) — [the optimized MLA kernel improved end-to-end throughput by 2.10× and reduced TTFT by 3.71×](https://rocm.blogs.amd.com/software-tools-optimization/geak-mla-optimization/README.html).
  3. Optimized a rendering kernel, delivering a [36% speedup while maintaining numerical accuracy for a 3D world model use case](https://rocm.blogs.amd.com/artificial-intelligence/dworld-m3d/README.html).
- The codebase is fully open source, enabling the community to accelerate its further development.

## Quick Look Back at GEAK v1 & v2

- **[GEAK v1](https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-ai/README.html) introduced an agentic framework for Instruction-to-Triton generation**, producing AMD GPU kernels from brief natural-language instructions.
- **The v1 system was built around generator, evaluator, reflector, and optimizer modules**, enabling iterative code generation, correctness checks, debugging, and performance tuning.
- **[GEAK v1](https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-ai/README.html) combined 1-shot prompting, knowledge injection, Reflexion, LLM-as-optimizer, debugging-trap avoidance, and parallel scaling** to improve kernel correctness and efficiency.
- **On TritonBench-modified, GEAK v1 reached up to 54.89% execution accuracy and 2.59× average speedup**, showing strong gains over direct LLM prompting.
- **[GEAK v2](https://rocm.blogs.amd.com/artificial-intelligence/geak-agents-family/README.html) expanded GEAK into a family of specialized agents**, introducing OptimAgentV2 for Instruction-to-Triton generation and [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) for Triton-to-Triton optimization.
- **These agents added multi-offspring evolution, LLM-based candidate evaluation, and hardware-aware profiler (+ analyzer) feedback**, allowing kernels to evolve using both semantic and profiler-guided signals.
- **[GEAK v2](https://rocm.blogs.amd.com/artificial-intelligence/geak-agents-family/README.html) further improved results**, delivering up to a +9.76% execution-accuracy gain over v1, 3.32× average speedup with profiler guidance, and up to 7.02× speedup with [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) on ROCm benchmarks.
- **[GEAK-HIP](https://rocm.blogs.amd.com/software-tools-optimization/geak-hip-optimizations/README.html) adapted [GEAK v2](https://rocm.blogs.amd.com/artificial-intelligence/geak-agents-family/README.html) for HIP kernel optimization**, an extension of the GEAK family that supports HIP-to-HIP optimization and shows strong gains.

## Architecture & Core Features

```{figure} ./images/geak-agent-loop.svg
:align: center
:alt: GEAK v3 agent loop
Figure 1: GEAK v3 overall system workflow
```

As shown in Figure 1 above, GEAK v3 moves from single file-level tuning to
repository-level optimization, meaning it can analyze an entire codebase
for files associated with a kernel, discover (or if needed generate) the
right tests/harnesses, and then run an iterative agent-driven optimization
loop. The remainder of this section walks through the core components that
make up this workflow.

### 1. Optimization Pipeline

GEAK v3 (based on the [mini-swe](https://github.com/SWE-agent/mini-swe-agent) agent) runs a closed, multi-round optimization pipeline: preprocess the
repository, build a fixed evaluation contract, plan optimization tasks,
dispatch them to multiple agents that create code changes in the form of
patches, evaluate candidate patches, and carry the best verified result
into the next round.

#### Preprocessing, Harness Setup, and Baseline

GEAK first identifies the target kernel and related files and gathers context about their dependencies and available tests. If no test command is provided, it
discovers or generates a harness. For existing test harnesses, GEAK modifies the shapes to match those required for the given problem. GEAK then creates a fixed evaluation
contract for correctness, benchmarking, and optimization metrics. GEAK
then records baseline performance and profiling results as the reference
point.

#### Task Planning

Based on the baseline and profiling signals, GEAK plans concrete
optimization strategies for the next round, such as memory-access
optimization, launch-configuration tuning, algorithmic restructuring, or
backend-specific rewrites.

#### Multi-Agent Execution

The planned tasks are dispatched to multiple agents running in isolated
git workspaces. Each agent explores one optimization path, produces
sequential concrete patches, and evaluates them under the same contract,
making results reproducible and comparable.

#### Multi-Round Optimization

GEAK can run multiple rounds. After each round, the best verified patch
becomes the starting point for the next round, allowing improvements to
compound over time.

#### Patch Saving and Selection

GEAK saves patches, logs, benchmark results, and evaluation summaries for
each round. It then selects the best valid patch and outputs the winning
diff, benchmark evidence, profiling logs, and a final report.

### 2. MCP Tools

GEAK v3 uses MCP tools to connect agents with external optimization
capabilities. The profiler tool provides kernel-level bottleneck
analysis, while the RAG tool retrieves relevant GPU optimization
knowledge during the run. This gives the agent both measurement feedback
and technical context instead of relying only on LLM reasoning.

### 3. Knowledge Layer

The GEAK knowledge layer is built around RAG. It indexes curated GPU
optimization documents and can be extended with custom materials. It can
also extract reverse knowledge from previous baseline-vs-optimized code,
turning past optimization experience into reusable guidance.

### 4. Memory

GEAK uses a dual memory system to avoid repeated exploration, leading to
better optimizations.

- Within a session, it tracks profiling results, attempted strategies,
  failed patches, successful patches, and benchmark outcomes.
- Across sessions, it can reuse prior optimization insights and
  repository-specific lessons to converge faster in future runs.

### 5. Skills and Subagents

GEAK v3 uses skills and subagents to specialize in the workflow. Skills
provide domain knowledge for Triton, HIP, FlyDSL, PyTorch-to-FlyDSL
translation, and GEMM tuning. Subagents handle focused tasks such as
codebase exploration, harness generation, kernel optimization, speedup
verification, reverse-knowledge extraction, and GEMM tuning.

### 6. GEMM Tuning

GEMM tuning is a specialized GEAK workflow. GEMM is the dominant cost in
transformer serving, requiring *the selection and configuration of the right prebuilt vendor kernel for each problem shape*. Instead of relying on static heuristics,
GEAK captures the exact `(M, N, K)` shapes a workload issues, runs the vendor
autotuner over them, and wires the resulting kernel table back into the dispatch
path, then re-runs the end-to-end serving benchmark. Because
tuning selects among numerically-equivalent vendor kernels, correctness is preserved by construction.

### 7. RDNA Support

GEAK v3 extends beyond AMD Instinct™ ([CDNA](https://www.amd.com/en/technologies/cdna.html)) data-center GPUs to RDNA
consumer/gaming architectures, validated end-to-end on [RDNA4](https://www.amd.com/en/technologies/rdna.html) (gfx1201,
Navi 48). GEAK detects the GPU architecture at runtime and adapts its
pipeline accordingly, retargeting the profiler and optimization guidance
to RDNA hardware and steering the agent towards WMMA (Wave Matrix
Multiply-Accumulate) matrix instructions, wave32 execution and the GDDR memory hierarchy.

## Overall Results[^1]

We evaluate GEAK v3 on two kernel optimization tasks: [HIP](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/hip2hip) kernel
optimization and [Triton](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton) kernel optimization, using the [AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena)
benchmark[^2]. All experiments in this section were conducted on MI355[^3]. Both benchmarks report speedup over the original
baseline kernel. Candidate patches are only counted after passing
correctness validation and benchmark evaluation.

Kernels are grouped into three levels based on the optimization scope:

- **L1 — single-kernel-function optimization:** the target is mainly one
  kernel function, and the optimization is mostly local to that function.
- **L2 — multi-function optimization:** the target involves multiple
  related functions, wrappers, helper routines, or launch logic, so the
  agent needs to reason across more than one function.
- **L3 — repository-level optimization:** the target requires broader
  repository understanding, including build context, runtime integration,
  benchmark harnesses, kernel dispatch paths, or multiple files across
  the codebase.

### Task 1: HIP Kernel Optimization

The HIP benchmark contains 16
HIP kernels covering point-cloud operators, 3D perception kernels, tensor
manipulation kernels, and low-level device primitives.

| Level | Kernel count | Kernel names |
|---|---:|---|
| L1 | 5 | [`ball_query`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/hip2hip/others/ball_query), [`knn`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/hip2hip/others/knn), [`matrix_multiplication`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/hip2hip/others/matrix_multiplication), `silu`, [`three_nn`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/hip2hip/others/three_nn) |
| L2 | 7 | [`assign_score_withk`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/hip2hip/others/assign_score_withk), [`furthest_point_sample`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/hip2hip/others/furthest_point_sample), [`gather_points`](https://github.com/AMD-AGI/AgentKernelArena/tree/geak_benchmark_legacy/tasks/hip2hip/others/gather_points), [`points_in_boxes`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/hip2hip/others/points_in_boxes), [`roiaware_pool3d`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/hip2hip/others/roiaware_pool3d), [`roipoint_pool3d`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/hip2hip/others/roipoint_pool3d), `three_interpolate` |
| L3 | 4 | [`block_radix_rank`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/repository/rocprim/block_radix_rank), [`device_merge_sort`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/repository/rocprim/device_merge_sort), [`device_binary_search`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/repository/rocprim/device_binary_search), [`device_search_n`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/repository/rocprim/device_search_n) |

GEAK v3 achieves a 3.02× final geomean speedup, compared with 2.11× for
mini-swe. The advantage is clearest on L1 and L2 kernels,
where GEAK v3 reaches 4.14× and 4.20× geomean speedup respectively.

| | L1 geomean | L2 geomean | L3 geomean | Final geomean |
|---|---|---|---|---|
| GEAK v3 | 4.14× | 4.20× | 1.14× | 3.02× |
| mini-swe | 2.58× | 2.74× | 1.04× | 2.11× |

### Task 2: Triton Kernel Optimization

The Triton benchmark contains
17 Triton kernels covering AI runtime workloads such as MoE, routing,
RoPE, RMSNorm, FP8 / FP4 quantization, GEMM, and attention-related
kernels.

| Level | Kernel count | Kernel names |
|---|---:|---|
| L1 | 6 | [`fused_append_shared_experts`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L1/fused_append_shared_experts), [`llama_ff_triton`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L1/llama_ff_triton), [`mla_decode`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L1/mla_decode), [`moe_routing_sigmoid_top1`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L1/moe_routing_sigmoid_top1), [`refk_fp8_blockwise_mm`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L1/refk_fp8_blockwise_mm), [`refk_identity`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L1/refk_identity) |
| L2 | 3 | [`fast_rms_layernorm`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L2/fast_rms_layernorm), [`lean_atten_paged`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L2/lean_atten_paged), [`topk`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L2/topk) |
| L3 | 8 | [`fused_moe_mxfp4`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L3/fused_moe_mxfp4), [`fused_mxfp4_quant_moe_sort`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L3/fused_mxfp4_quant_moe_sort), [`fused_qk_rope_cache_mla`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L3/fused_qk_rope_cache_mla), [`fused_qkv_rope`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L3/fused_qkv_rope), [`fused_rms_fp8`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L3/fused_rms_fp8), [`gemm`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L3/gemm), [`gemm_a16w16_atomic`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L3/gemm_a16w16_atomic), [`gemm_a16wfp4`](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval/L3/gemm_a16wfp4) |

GEAK v3 achieves a 2.22× final geomean speedup, compared with 1.25× for
mini-swe. The strongest improvement comes from L3 kernels, where GEAK v3
reaches 2.78× geomean speedup, showing its ability to optimize harder
model-runtime kernels.

| | L1 geomean | L2 geomean | L3 geomean | Final geomean |
|---|---|---|---|---|
| GEAK v3 | 1.75× | 2.08× | 2.78× | 2.22× |
| mini-swe | 1.39× | 1.50× | 1.05× | 1.25× |

### Comparison with GEAK v2

We additionally compare GEAK v3 against GEAK v2 on the subset of Triton
kernels supported by both systems. GEAK v2 is designed for single-file
kernel optimization, whereas GEAK v3 extends the optimization workflow to
repository-level codebases with multi-file reasoning and agent-driven
optimization. On the shared kernels, GEAK v3 achieves a 1.95× geomean
speedup compared to 1.03× for GEAK v2, demonstrating the effectiveness of
the improved optimization workflow.

| Kernel | GEAK v3 | GEAK v2 |
|---|---:|---:|
| `fused_qkv_rope` | 1.36× | 1.06× |
| `fused_rms_fp8` | 1.29× | 1.05× |
| `fast_rms_layernorm` | 3.95× | 0.92× |
| `topk` | 1.78× | 1.11× |
| **Geomean** | **1.95×** | **1.03×** |

### Task 3: FlyDSL Kernel Optimization

Similarly, on a [benchmark of 12 FlyDSL kernels](https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/flydsl2flydsl), GEAK v3 achieves a 1.06× geomean speedup.

| | L1 geomean | L2 geomean | L3 geomean | Final geomean |
|---|---|---|---|---|
| GEAK v3 | 1.02× | 1.13× | 1.02× | 1.06× |

## Getting Started

Typical kernel optimization using natural language input:

```bash
geak -t "Optimize the kernel from /path/to/aiter, specifically aiter/ops/triton/topk.py. Use the harness at /path/to/test_topk_harness.py. Use four GPUs with IDs 0-3 simultaneously."
```

For further instructions, please check out the README.md of [GEAK v3](https://github.com/AMD-AGI/GEAK/blob/v3.2.2/README.md).

## Case Studies

### Case Study #1: `gemm_a16wfp4` (Triton, AMD Instinct™ MI300X)

BF16 × MXFP4 block-scaled GEMM for LLM inference. GEAK v3 replaced
`log2`/`exp2` scale reconstruction with direct IEEE-754 exponent
bit-manipulation, fused split-K into the output via `atomic_add`, and
added shape-adaptive autotuning. Staying entirely in Triton, it reached a
**1.95× geomean speedup** across 55 configs (0.268 → 0.137 ms).

The winning optimization — bitwise MXFP4 scale reconstruction:

<div style="overflow-x: auto;">
<table style="min-width:900px; table-layout:fixed;">
<tr>
<th style="width:50%; text-align:left;">Baseline (transcendental ops)</th>
<th style="width:50%; text-align:left;">GEAK v3 (direct exponent bit extraction)</th>
</tr>
<tr>
<td style="vertical-align:top;">

```python
scale = tl.log2(amax).floor() - 2
quant_scale = tl.exp2(-scale)
qx = x * quant_scale
```

</td>
<td style="vertical-align:top;">

```python
amax_exp = (amax_int >> 23) & 0xFF
scale = amax_exp - 129
quant_scale = ((127 - scale) << 23).to(tl.float32, bitcast=True)
qx = x * quant_scale
```

</td>
</tr>
</table>
</div>

### Case Study #2: `roiaware_pool3d` (HIP, gfx942/gfx950)

3D ROI-aware LiDAR point pooling from PointPillars/OpenPCDet. GEAK v3 replaced serial per-box scanning with cooperative
thread-parallel scanning + `atomicAdd`, swapped separate sin/cos for the
`__sincosf` intrinsic, and moved to stream-ordered async allocation
(`hipMallocAsync`). Result: **14.46× geomean speedup** across 5 configs
(4.199 → 0.290 ms).

The winning optimization — cooperative per-box point scan:

<div style="overflow-x: auto;">
<table style="min-width:900px; table-layout:fixed;">
<tr>
<th style="width:50%; text-align:left;">Baseline (one thread scans all points serially)</th>
<th style="width:50%; text-align:left;">GEAK v3 (threads scan cooperatively with atomic reservation)</th>
</tr>
<tr>
<td style="vertical-align:top;">

```cpp
for (int k = 0; k < pts_num; k++) {
  if (pts_mask[box_idx * pts_num + k] != -1) {
    unsigned int cnt = pts_idx_of_voxels[base_offset];
    if (cnt < max_num_pts) {
      pts_idx_of_voxels[base_offset + cnt + 1] = k;
      pts_idx_of_voxels[base_offset]++;
    }
  }
}
```

</td>
<td style="vertical-align:top;">

```cpp
for (int k = threadIdx.x; k < pts_num; k += blockDim.x) {
  if (mask[k] != -1) {
    unsigned int cnt = atomicAdd(&voxels[base_offset], 1);
    if (cnt < max_num_pts)
      voxels[base_offset + cnt + 1] = k;
  }
}
```

</td>
</tr>
</table>
</div>

### Case Study #3: `layernorm_kernel` (FlyDSL, AMD Instinct™ MI300X)

Transformer LayerNorm with per-row reduction + affine transform. GEAK v3
generalized the vectorized 128-bit buffer-load fast path to all
tile-aligned hidden sizes, prefetched all gamma/beta tiles into registers
so loads overlap the reduction, and simplified the affine loop. Staying
entirely in FlyDSL, it reached a **1.43× geomean speedup** across 10
configs (0.00709 → 0.00494 ms).

The winning optimization — bulk gamma/beta register prefetch overlapping
the reduction:

```python
# GEAK v3: prefetch all affine tiles into registers BEFORE reduction
g_local, b_local = [], []
for tile_i in range_constexpr(num_tiles_py):
    gb_idx = tid + tile_i * BLOCK_THREADS
    g_local.append(_load_vec(gamma_div, gb_idx).to(fx.Float32))
    b_local.append(_load_vec(beta_div, gb_idx).to(fx.Float32))

# reduction now overlaps with the loads above
sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
mean, rstd = compute_mean_rstd(sum_val, sumsq_val)
```

## Summary

In this blog, you explored how GEAK v3 turns GPU kernel optimization from one-shot code editing into a
multi-file kernel editing problem, combining repository understanding,
task planning, multi-agent execution, profiling, knowledge retrieval,
memory, and patch-based validation. The closed optimization loop grounds
every reported speedup through its trustworthy and verifiable harness
framework. Across HIP and Triton benchmarks, GEAK v3 showed strong
speedups.

GEAK is an active area of development for our team, with deeper workload-level optimization, broader language and hardware coverage, and richer cross-run learning in the pipeline. The codebase is fully open-sourced, and we invite developers to try
[GEAK v3](https://github.com/AMD-AGI/GEAK/tree/v3.2.2) on their custom
kernels, follow along as the framework evolves, and watch this space for
future updates.

## Additional Resources

1. [GEAK: Introducing Triton Kernel AI Agent & Evaluation Benchmarks — ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-ai/README.html)

2. [GEAK-Triton v2 Family of AI Agents: Kernel Optimization for AMD Instinct GPUs — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/geak-agents-family/README.html)

3. [GEAK-HIP: Expanding GEAK for HIP Code Optimization — ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/geak-hip-optimizations/README.html)

4. [Triton: Open-source GPU programming language](https://triton-lang.org/)

5. [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

6. [OpenEvolve: Open-source implementation of AlphaEvolve](https://github.com/algorithmicsuperintelligence/openevolve)

7. [AMD-AGI/GEAK-eval (TritonBench-modified and evaluation suite)](https://github.com/AMD-AGI/GEAK-eval)

8. [MLPerf Inference v6.0 on AMD Instinct GPUs — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html)

9. [3D World Model use case — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/dworld-m3d/README.html)

10. [OpenPCDet: point-cloud 3D object detection toolbox (PointPillars)](https://github.com/open-mmlab/OpenPCDet)

11. [AgentKernelArena (Triton and HIP kernel benchmarks)](https://github.com/AMD-AGI/AgentKernelArena)

## Bias, Risks & Limitations

- The agent code is being released for research purposes only. It is not intended for applications that require high levels of factual accuracy, including safety-critical, healthcare, or medical applications, or for generating false information or facilitating toxic conversations.
- The agent code is made accessible without any assurances of safety. Users must conduct comprehensive evaluations and implement safety filtering mechanisms as per their respective use cases.
- It may be possible to prompt the agent to generate content that may be factually inaccurate, harmful, violent, toxic, biased, or otherwise objectionable. Such content may also be generated by prompts that were not intended to produce output as such. Users are therefore requested to be aware of this and exercise caution and responsible thinking when using it.
- Multilingual abilities have not been tested; therefore, the agent may misunderstand and generate erroneous responses when prompted using different languages.

## License

Apache 2.0

## Acknowledgements

We would like to acknowledge the following folks for their constructive discussions and feedback during this work — Sina Rafati, Muhammad Awad, Sharareh Younesian, Rita Brugarolas Brufau, Chaojun Hou, Xiaofei Zheng, Zhen Gong, Shuoshuo Li, Bao Yunkai, Chenyi Yang, Zhang Lei, Tharun Adithya Srikrishnan, Deval Shah, Steve Reinhardt, Mohammad Abdul Basit, Kyle Hoffmeyer, Ahmed Hasssan, Zhenyu Gu, Xi Zhao, Vikram Appia, Mehdi Rezagholizadeh and Sarunas Kalade.

## System Configuration

All performance benchmarks were conducted using the following hardware
and software configuration:

| Component | Specification |
|---|---|
| **GPU** | AMD Instinct™ MI300X |
|  | AMD Instinct™ MI355X |
|  | AMD Radeon™ RDNA4 (gfx1201, Navi 48) |
| **ROCm** | 7.1.1 (HIP runs), 7.2.0 (Triton runs) |
| **Host OS** | Ubuntu 24.04 |
| **Python** | 3.12 |
| **HIP-run Docker image** | `rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| **Triton-run Docker image** | `lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi35x-20260610` |

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, ROCm, Instinct, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved

[^1]: Results are based on AMD testing using the benchmark configurations described herein. Performance results may vary based on hardware, software, kernel implementation, workload characteristics, and optimization settings.

[^2]: AgentKernelArena: <https://github.com/AMD-AGI/AgentKernelArena>. Triton-to-Triton subset: <https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/triton2triton/geak_eval>. HIP-to-HIP subset: <https://github.com/AMD-AGI/AgentKernelArena/tree/main/tasks/hip2hip>.

[^3]: Refer to the System Configuration section for more details.
