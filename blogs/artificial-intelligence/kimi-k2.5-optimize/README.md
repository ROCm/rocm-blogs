---
blogpost: true
blog_title: "Accelerating Kimi-K2.5 on AMD Instinct™ MI300X: Optimizing Fused MoE with FlyDSL"
date: 24 Mar 2026
author: 'Bobo Fang, Chunhung Wang, Clement Lin, Dai Yan, Eveline Chen, Felix Li, Menghsuan Yang, Peng Sun'
thumbnail: 'kimi-k2.5-opt-thumbnail.jpg'
tags: AI/ML
category: Applications & models
target_audience: AI DEVELOPERS AND ENTHUSIAST
key_value_propositions: With the recent surge in popularity of OpenClaw, its officially recommended model, Kimi-K2.5, has taken the AI community by storm. In this blog, we profile Kimi-K2.5 on AMD Instinct™ MI300X, identify fused MoE as the key bottleneck, and leverage FlyDSL to rapidly build an optimized kernel, resulting in significant end-to-end performance gains.
language: English
myst:
    html_meta:
        "author": "Clement Lin, Chunhung Wang, Menghsuan Yang, Bobo Fang, Eveline Chen"
        "description lang=en": "Optimize Kimi-K2.5 on AMD MI300X using FlyDSL for fused MoE kernel acceleration. Achieve faster TTFT, TPOT, and throughput with our step-by-step optimization guide."
        "keywords": "Kimi-K2.5, sglang, MI300X, FlyDSL, AITER, fused MoE, W4A16"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
        "amd_blog_authors": "Bobo Fang, Chunhung Wang, Clement Lin, Dai Yan, Eveline Chen, Felix Li, Menghsuan Yang, Peng Sun"
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

# Accelerating Kimi-K2.5 on AMD Instinct™ MI300X: Optimizing Fused MoE with FlyDSL

With the recent surge in popularity of OpenClaw [[1]](#references), its officially recommended model, Kimi-K2.5 [[2]](#references), has taken the AI community by storm. As developers and researchers flock to this powerful Mixture-of-Experts (MoE) LLM, the need for high-performance inference on cutting-edge hardware has never been more critical.

In this blog, we walk through our optimization journey for Kimi-K2.5 on AMD Instinct™ MI300X GPUs — starting from an out-of-the-box performance baseline, profiling to identify `fused_moe` as the key bottleneck, and then leveraging [FlyDSL](https://github.com/ROCm/FlyDSL) [[3]](#references) to rapidly build an optimized mixed-precision (W4A16 + BF16) fused MoE kernel. Thanks to FlyDSL's Python-native workflow, we were able to achieve strong kernel performance in a very short development cycle. When combined with framework-level optimizations in SGLang and AITER, the end-to-end result delivers up to **65% lower TTFT**, **69% lower TPOT**, and **162% higher throughput** — all with no accuracy degradation.

---

## Out-of-the-Box Performance Baseline

We begin by evaluating the out-of-the-box performance of Kimi-K2.5 served through SGLang [[4]](#references) on AMD Instinct™ MI300X GPUs. All tests in this blog — both the baseline and the optimized configuration — use the following Docker image:

```bash
docker pull clementlincf/amdafde:v0.5.8-rocm720-mi30x-kimi-k2.5-opt-20260224
```

This image contains the full software stack (ROCm 7.2.0, SGLang, AITER, FlyDSL) with all the optimizations pre-installed. We first establish a baseline without the FlyDSL kernel enabled.

### Launching the SGLang Server

```bash
export SGLANG_USE_AITER=1
python -m sglang.launch_server \
    --model /data/cf/Kimi-K2.5/ \
    --tp 8 \
    --attention-backend aiter \
    --host 0.0.0.0 \
    --port 9527 \
    --mem-fraction-static 0.9 \
    --trust-remote-code \
    --disable-custom-all-reduce
```

### Running the Benchmark

We benchmark under two different concurrency settings to stress different phases of the inference pipeline:

- Concurrency = 2 (Prefill-dominated): With only 2 concurrent requests and long input sequences, the GPU spends a large portion of time in the prefill phase. In this setting, TTFT (Time To First Token) is the key metric, as it is largely determined by how quickly the model finishes prefill and starts producing the first output token.

- Concurrency = 40 (Decode-dominated): With 40 concurrent requests, many sequences are simultaneously in the decode phase, generating tokens in parallel. In this setting, TPOT (Time Per Output Token) and overall throughput are the key metrics, since they reflect how efficiently the system handles parallel decoding under heavy load.

### Profiling: Where Is the Time Spent?

To understand where the bottleneck lies, we profiled the model execution and ranked the top 5 most time-consuming kernels (prefill + decode combined) under both concurrency settings.

**Top 5 Kernels — Concurrency = 2:**

| Rank | Kernel | Percentage |
| :---: | :--- | :---: |
| 1 | `fused_moe_kernel_gptq_awq` | **87.8%** |
| 2 | `nccl AllReduce` | 2.2% |
| 3 | `GEMM (Cijk_Alik MT256x256x32)` | 2.2% |
| 4 | `AITER fmha_fwd (flash attention)` | 1.9% |
| 5 | `GEMM (Cijk_Alik MT192x256x64)` | 1.7% |

**Top 5 Kernels — Concurrency = 40:**

| Rank | Kernel | Percentage |
| :---: | :--- | :---: |
| 1 | `fused_moe_kernel_gptq_awq` | **89.7%** |
| 2 | `nccl AllReduce` | 2.3% |
| 3 | `AITER fmha_fwd (flash attention)` | 1.9% |
| 4 | `GEMM (Cijk_Alik MT256x256x32)` | 1.3% |
| 5 | `GEMM (Cijk_Alik MT192x256x64)` | 1.3% |

The profiling results clearly show that `fused_moe` dominates GPU execution time under both concurrency settings — accounting for 87.8% at low concurrency and 89.7% at high concurrency. This makes it the primary target for optimization.

---

## FlyDSL: Rapid Kernel Development for Fused MoE Optimization

### What Is FlyDSL?

FlyDSL (Flexible layout python DSL) is a Python DSL backed by a custom MLIR stack for authoring high-performance GPU kernels on AMD hardware. At its core is FLIR (Flexible Layout Intermediate Representation) — a layout algebra system inspired by CuTe [[5]](#references) that expresses complex data mapping patterns such as tiling, swizzling, and vectorization through composable `(Shape, Stride)` abstractions.

FlyDSL stands out for two key reasons that made it our tool of choice:

- Python-native development: Kernel authors write everything in Python using the `flydsl` package. The familiar syntax and rapid iteration cycle means new kernels can be prototyped, tested, and benchmarked in hours rather than days — no need to drop into raw HIP C++ or assembly.
- Hierarchical, instruction-level control: Despite the high-level interface, FlyDSL keeps the full tiling hierarchy explicit — from block to warp to thread, down to individual MFMA (Matrix Fused Multiply-Add) instructions. Developers can precisely control memory access patterns (global → LDS → register), register allocation, and instruction scheduling. This level of control is typically only achievable with hand-written assembly, yet FlyDSL delivers it through a composable Python API.

Under the hood, FlyDSL compiles through a pipeline of MLIR passes — canonicalization, CSE, GPU-to-ROCDL lowering — and emits optimized binaries targeting `gfx942` (MI300X) and `gfx950` (MI350) architectures.

### Implementing Mixed-Precision Fused MoE with FlyDSL

Since `fused_moe` was identified as the bottleneck, we needed to quickly produce an optimized kernel for this operator. The out-of-the-box SGLang stack uses a Triton-based fused MoE kernel, which would require significant additional tuning effort to reach peak performance on MI300X. Alternatively, Composable Kernel (CK) relies on hand-written assembly code, making rapid iteration impractical for new model shapes. FlyDSL, on the other hand, lets us write kernels in Python while still controlling instruction-level details — offering the best time-to-performance tradeoff.

We used FlyDSL to implement a mixed-precision fused MoE kernel that supports both BF16 (A16W16) and W4A16 data paths. This implementation has been merged into the [FlyDSL upstream repository](https://github.com/ROCm/FlyDSL). By allowing different MoE stages to use different precisions, we can balance numerical accuracy and compute throughput.

We benchmarked FlyDSL against PyTorch (torch), Triton, and Composable Kernel (CK) [[6]](#references) across four representative MoE shapes using the following software stack:

- ROCm: 7.2.0
- PyTorch: 2.9.1
- Triton: 3.5.1
- CK: via AITER 0.1.5.post5.dev409+g6b157bbb2

We include Torch, Triton, and CK results as reference points to illustrate the performance landscape across different implementation approaches.

#### Shape: large (tokens=16384, model_dim=7168, inter_dim=512, E=384, topk=8)

This shape accounts for over half of the fused MoE invocations in Kimi-K2.5 inference, making it the most performance-critical configuration.

| dtype | Torch (ms) | Triton (ms) | CK (ms) | FlyDSL (ms) |
| :--- | :---: | :---: | :---: | :---: |
| bf16 (A16W16) | 119.82 | 12.09 | gpu_fault | **8.68** |
| w4a16 (A16W4) | 131.33 | 31.43 | unsupported | **9.77** |

#### Shape: small (tokens=512, model_dim=2048, inter_dim=512, E=64, topk=4)

| dtype | Torch (ms) | Triton (ms) | CK (ms) | FlyDSL (ms) |
| :--- | :---: | :---: | :---: | :---: |
| bf16 (A16W16) | 16.56 | 0.30 | 0.32 | **0.13** |
| w4a16 (A16W4) | 17.12 | 0.29 | unsupported | **0.11** |

#### Shape: medium (tokens=2048, model_dim=4096, inter_dim=512, E=64, topk=8)

| dtype | Torch (ms) | Triton (ms) | CK (ms) | FlyDSL (ms) |
| :--- | :---: | :---: | :---: | :---: |
| bf16 (A16W16) | 16.87 | 0.96 | 0.83 | **0.60** |
| w4a16 (A16W4) | 18.14 | 1.90 | unsupported | **0.69** |

#### Shape: large (tokens=4096, model_dim=7168, inter_dim=512, E=128, topk=8)

| dtype | Torch (ms) | Triton (ms) | CK (ms) | FlyDSL (ms) |
| :--- | :---: | :---: | :---: | :---: |
| bf16 (A16W16) | 36.68 | 3.18 | 2.47 | **2.25** |
| w4a16 (A16W4) | 39.98 | 6.81 | unsupported | **2.42** |

> **Note:** The Triton and CK numbers shown above may not reflect their fully tuned peak performance — achieving that would typically require hours of additional tuning per shape and data type. In contrast, once the FlyDSL fused MoE kernel was implemented, it achieved competitive performance out-of-the-box across the tested shapes with minimal per-configuration tuning effort.

Across all tested shapes, FlyDSL delivers strong performance for both BF16 and W4A16 data types. Notably, CK does not yet support the W4A16 fused MoE path and encounters issues on the Kimi-K2.5 shapes (E=384), while Triton would require additional tuning effort to reach its full potential on these configurations. FlyDSL's key advantage is time-to-performance: written entirely in Python, it enabled us to develop and optimize a production-quality fused MoE kernel in a fraction of the time that would be needed with assembly-level (CK) or extensive manual tuning (Triton) approaches.

Given these results, we adopted the FlyDSL-based mixed-precision fused MoE kernel as our default for Kimi-K2.5 inference.

---

## End-to-End Optimization Results

In addition to the kernel-level optimization, we also made several framework-level modifications to SGLang and AITER to fully integrate the FlyDSL kernel and other improvements. To reproduce the results below, install from the following branches:

- FlyDSL: [`main`](https://github.com/ROCm/FlyDSL)
- AITER [[7]](#references): [`dev/kimi-K2.5`](https://github.com/ROCm/aiter/tree/dev/kimi-K2.5)
- SGLang: [`kimi-K2.5-dev`](https://github.com/AFDEAPAC/sglang/tree/kimi-K2.5-dev)

> **Note:** If any of the branches above are no longer available, it means the changes have already been merged into the upstream repositories.

### Launching the Optimized Server

Set the required environment variables and launch the SGLang server with the FlyDSL-optimized fused MoE kernel enabled:

```bash
export TRITON_MAX_CACHE_SIZE=2147483648
export AITER_FLYDSL_MOE_COMPARE_STAGE2=0
export AITER_FLYDSL_MOE_COMPARE=0
export AITER_FLYDSL_DEBUG=0
export AITER_ENFORCE_DSL=1
export DSL2_ROOT=/opt/FlyDSL
export AITER_USE_FLYDSL_MOE=1
export AITER_USE_FLYDSL_MOE_STAGE1=1
export AITER_USE_FLYDSL_MOE_STAGE2=1
export MLIR_PATH=/opt/mlir_install
export CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3

export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_USE_AITER=1

export FLYDSL_W4A16_HYBRID=w2_bf16

python -m sglang.launch_server \
    --model /mnt/md0/models/Kimi-K2.5 \
    --tp 8 \
    --attention-backend aiter \
    --host 0.0.0.0 \
    --port 9527 \
    --mem-fraction-static 0.9 \
    --trust-remote-code \
    --disable-radix-cache \
    --enable-torch-compile \
    --disable-custom-all-reduce
```

> **Note on `FLYDSL_W4A16_HYBRID`:** In a standard MoE block, each expert contains two stages of computation — Stage 1 (gate/up projection) and Stage 2 (down projection). This environment variable enables mixed-precision across the two stages: setting it to `w2_bf16` keeps Stage 1 in W4A16 (4-bit quantized) while running Stage 2 in BF16 (full precision). This trades a modest increase in memory for better compute throughput and numerical stability on Stage 2. When unset, both stages default to W4A16.

<!-- -->
> **Note on `--disable-custom-all-reduce`:** This flag disables the AITER custom all-reduce implementation within SGLang. We include it here to avoid potential numerical instability observed with the current custom all-reduce path, ensuring consistent accuracy during evaluation.

<!-- -->
> **Note on `--disable-radix-cache` and `--enable-torch-compile`:** Beyond the fused MoE kernel optimization, these two flags provide additional end-to-end performance gains:
>
> - `--disable-radix-cache` disables SGLang's radix tree-based prefix cache. Since our benchmark uses random inputs with no shared prefixes, the cache lookup overhead provides no benefit and can be safely turned off to free memory for KV cache.
> - `--enable-torch-compile` enables `torch.compile` to trace and optimize the model graph, significantly reducing CPU-side kernel launch overhead. This is especially effective during the decode phase, where many small kernels are dispatched in rapid succession.
>
> Together with the FlyDSL fused MoE kernel, these framework-level optimizations contribute to the overall end-to-end performance gains reported below.

### Running the E2E Benchmark

We use the same input/output length settings to ensure an apples-to-apples comparison with the out-of-the-box baseline.

**Prefill-dominated, max request concurrency = 2:**

```bash
python3 -m sglang.bench_serving \
    --model /mnt/md0/models/Kimi-K2.5/ \
    --dataset-name random \
    --random-input 10240 \
    --random-output 512 \
    --num-prompts 10 \
    --max-concurrency 2 \
    --request-rate inf \
    --port 9527 \
    --random-range-ratio 1.0
```

**Decode-dominated, max request concurrency = 40:**

```bash
python3 -m sglang.bench_serving \
    --model /mnt/md0/models/Kimi-K2.5/ \
    --dataset-name random \
    --random-input 10240 \
    --random-output 512 \
    --num-prompts 160 \
    --max-concurrency 40 \
    --request-rate inf \
    --port 9527 \
    --random-range-ratio 1.0
```

### Performance Comparison: Out-of-the-Box vs. Optimized

**Concurrency = 2:**

| Metric | Statistic | Out-of-the-Box | Optimized | Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **TTFT (ms)** ↓ | Mean | 2918.16 | 1014.13 | **-65.3%** |
| | Median | 2875.91 | 1005.01 | **-65.1%** |
| | P99 | 4041.28 | 1262.39 | **-68.8%** |
| **TPOT (ms/token)** ↓ | Mean | 38.77 | 28.26 | **-27.1%** |
| | Median | 38.07 | 28.27 | **-25.7%** |
| | P99 | 46.58 | 29.62 | **-36.4%** |
| **Output Throughput (tok/s)** ↑ | | 45.04 | 66.24 | **+47.1%** |

**Concurrency = 40:**

| Metric | Statistic | Out-of-the-Box | Optimized | Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **TTFT (ms)** ↓ | Mean | 33478.68 | 17730.03 | **-47.0%** |
| | Median | 32840.94 | 13353.81 | **-59.3%** |
| | P99 | 71887.16 | 52652.53 | **-26.8%** |
| **TPOT (ms/token)** ↓ | Mean | 230.37 | 70.86 | **-69.2%** |
| | Median | 231.43 | 69.59 | **-69.9%** |
| | P99 | 298.46 | 92.62 | **-69.0%** |
| **Output Throughput (tok/s)** ↑ | | 135.39 | 355.35 | **+162.4%** |

> ↓ = indicates lower is better, ↑ = indicates higher is better

### Accuracy Validation

To verify that the optimized kernel does not degrade model quality, we run an accuracy evaluation using lm-eval-harness [[8]](#references) on the GSM8K benchmark:

```bash
lm_eval \
    --model local-completions \
    --model_args model=/mnt/md0/models/Kimi-K2.5/,base_url=http://localhost:9527/v1/completions,num_concurrent=1,tokenized_requests=False,trust_remote_code=True \
    --tasks gsm8k \
    --num_fewshot 10 \
    --limit 100
```

| Benchmark | Metric | Out-of-the-Box | Optimized |
| :--- | :---: | :---: | :---: |
| **GSM8K** (10-shot, 100 samples) | exact_match (flexible) | 0.96 ± 0.0197 | 0.96 ± 0.0197 |
| | exact_match (strict) | 0.96 ± 0.0197 | 0.96 ± 0.0197 |

The results confirm that our optimization maintains full model accuracy — the optimized mixed-precision fused MoE kernel produces identical scores to the out-of-the-box baseline on the GSM8K benchmark, with no accuracy degradation.

---

## Summary

In this blog, we demonstrated how to accelerate Kimi-K2.5 inference on AMD Instinct™ MI300X GPUs through a systematic optimization approach:

1. Profiling first: We established an out-of-the-box baseline and identified `fused_moe` as the dominant bottleneck, accounting for 88–90% of total GPU time across both concurrency settings.
2. Kernel optimization with FlyDSL (primary optimization): We used FlyDSL to rapidly implement a mixed-precision (W4A16 + BF16) fused MoE kernel that replaces the default Triton-based implementation in SGLang. FlyDSL's Python-native workflow and instruction-level control enabled us to achieve strong kernel performance in a very short development cycle — a task that would have required significant additional tuning with Triton or impractical effort with assembly-level approaches like CK.
3. Framework-level optimizations: In addition to the kernel replacement, we enabled `--enable-torch-compile` to reduce CPU-side kernel launch overhead (especially effective during decode), and `--disable-radix-cache` to eliminate unnecessary prefix cache lookups in random-input benchmarks and free memory for KV cache. These framework-level flags further complement the FlyDSL kernel optimization.

Combined, these optimizations deliver substantial end-to-end improvements: up to **65% lower TTFT**, **69% lower TPOT**, and **162% higher throughput** at high concurrency — all with no accuracy degradation on the GSM8K benchmark.

FlyDSL proved to be a powerful tool for rapidly optimizing new models as they emerge. Instead of spending weeks on hand-tuned HIP C++ or assembly kernels, our team was able to iterate quickly in Python — leveraging FlyDSL's composable layout abstractions and MLIR compilation pipeline — while still achieving top-tier performance on MI300X.

Looking ahead, the optimizations demonstrated in this blog — including the FlyDSL-based fused MoE kernel and the associated framework changes — will be progressively merged into the upstream SGLang and AITER repositories, making these performance gains available to the broader community.

## Acknowledgements

The authors wish to thank the [FlyDSL](https://github.com/ROCm/FlyDSL) community developers for their invaluable guidance and suggestions on the FlyDSL-related optimizations presented in this blog.

## References

[1] [OpenClaw](https://openclaw.ai/) - An open-source, local-first autonomous AI agent

[2] [Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) — Moonshot AI's Mixture-of-Experts LLM

[3] [FlyDSL](https://github.com/ROCm/FlyDSL) — Flexible Layout Python DSL for GPU kernel development

[4] [SGLang](https://github.com/sgl-project/sglang) — Fast serving framework for large language models

[5] [Cute](https://github.com/NVIDIA/cutlass) — CuTe layout algebra concepts (BSD-3-Clause parts only; no EULA-licensed code was referenced)

[6] [CK](https://github.com/ROCm/composable_kernel) — Tile-based kernel design patterns for AMD GPUs

[7] [AITER](https://github.com/ROCm/aiter) — AI Tensor Engine for ROCm

[8] [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — Framework for evaluating language models

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
