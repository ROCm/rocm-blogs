---
blogpost: true
blog_title: "DFlash Speculative Decoding on AMD Instinct MI355X: Up to 5× Faster Qwen3.5 Inference"
date: 16 Sep 2026
author: Hang Yang, Wei Luo, Xinjun Niu, Spandan Tiwari, Ashish Sirasao
thumbnail: 'dflash_thumbnail.png'
tags: AI/ML, GenAI, Serving, Performance
category: Applications & models
target_audience: AI developers and practitioners
key_value_propositions: Learn how DFlash block-diffusion speculative decoding delivers up to 5× faster Qwen3.5 inference on AMD Instinct MI355X with vLLM on ROCm, and how it stacks with mxfp4 target quantization.
language: English
myst:
    html_meta:
        "author": "Hang Yang, Wei Luo, Xinjun Niu, Spandan Tiwari, Ashish Sirasao"
        "description lang=en": "Explore how DFlash speculative decoding delivers up to 5× faster Qwen3.5 inference on AMD Instinct MI355X with vLLM on ROCm."
        "keywords": "DFlash, speculative decoding, block diffusion, MTP, vLLM, ROCm, MI355X, Qwen3.5, mxfp4, AMD Instinct"
        "property=og:locale": "en_US"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI, Deploying AI at Scale"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Hang Yang, Wei Luo, Xinjun Niu, Spandan Tiwari, Ashish Sirasao"
---

# DFlash Speculative Decoding on AMD Instinct MI355X: Up to 5× Faster Qwen3.5 Inference

Autoregressive decode generates one token per model forward pass, so single-request latency is bounded by how fast you can stream weights through the GPU — not by compute. **Speculative decoding** attacks that bottleneck by letting a small *draft* model propose several tokens that the large *target* model then verifies in a single parallel pass. In this post we bring **DFlash** [[3](#references)] — a *block-diffusion* drafter — to **AMD Instinct MI355X** through **vLLM** [[4](#references)] on **ROCm** [[5](#references)], benchmark it head-to-head against Qwen3.5's [[6](#references)] built-in **MTP** drafter, and then quantize the target model to **mxfp4** to show that speculation and quantization stack.

The result: **up to 5.02× single-request throughput** on Qwen3.5-27B and consistent wins over MTP, with an mxfp4 target adding another ~10–20% on top for free.

## Why DFlash?

### Speculative Decoding: Amortize One Verify Pass Over Many Tokens

The win in speculative decoding is amortization: one expensive target verify pass commits *multiple* tokens. The more tokens the drafter can propose that survive verification (the **acceptance length**), the more the verify cost is amortized. So the ideal drafter proposes *deep* blocks *cheaply* — and that is exactly where autoregressive drafters struggle, because depth and cost are coupled.

### Autoregressive Drafting Is Sequential and Drafting-Bound

Speculative decoding pairs a small **draft** model with the large **target** model. The drafter proposes several tokens, the target verifies them in one parallel forward pass, and every token that matches what the target would have produced is committed for free.

The catch is *how* the drafter generates its guesses. EAGLE-style drafters and the native **multi-token-prediction (MTP)** modules shipped with recent models are **autoregressive**: they emit draft tokens one at a time, so drafting cost grows with the number of tokens you want to speculate. That forces the drafter to stay shallow — capping draft depth and, ultimately, speedup.

### DFlash in Particular: Block Diffusion + KV Injection

**DFlash** [[1](#references)] replaces the autoregressive drafter with a lightweight **block-diffusion** model, refer to *Figure 1*. Also refer to **DFlash paper**[[3](#references)]. Two ideas make it work:

- **Block-diffusion drafting.** Instead of predicting tokens one-by-one, the drafter starts from a masked block of length *k* and denoises all *k* positions together in a single forward pass. The number of draft tokens is decoupled from the number of draft *passes* — proposing a longer block is nearly free.
- **KV injection.** Rather than re-encoding the context, DFlash writes the target model's hidden representations directly into the draft model's KV cache, so the drafter conditions on the target's rich features without recomputing them.

```{image} ./images/dflash-arch-diagram.png
:label: dflash-arch-diagram
:alt: dflash-arch-diagram
:width: 50%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="margin-left: 10%; width: 80%; font-size: 0.9em; color: #555;"><strong>Figure 1.</strong> <em>The DFlash architecture[<a href="#references">1</a>][<a href="#references">3</a>]. A lightweight block-diffusion drafter denoises a whole masked block in a single parallel pass, while KV injection writes the target model's hidden states directly into the drafter's KV cache — so the drafter conditions on the target's rich features without re-encoding the context, and drafting cost stays roughly flat as the block grows.</em></p>

### How Much Does DFlash Alone Buy?

The clearest signal is acceptance length as a function of draft budget. As the budget grows (block size / MTP steps of 4 → 8 → 16), MTP climbs but flattens, while DFlash keeps pulling ahead — because a bigger block is nearly free to draft. Please refer to *Figure 2*

```{image} ./images/throughput_curve.png
:label: throughput_curve
:alt: throughput_curve
:width: 50%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="margin-left: 10%; width: 80%; font-size: 0.9em; color: #555;"><strong>Figure 2.</strong> <em>Qwen3.5-27B · HumanEval · concurrency 1. MTP's throughput plateaus around 245 tok/s past budget 8 — each extra draft token costs another sequential pass — while DFlash keeps climbing to 396 tok/s at block=16.</em></p>

## Why Quantize the Target Model on Top?

DFlash accelerates the *decode loop*; weight quantization shrinks the *target's memory footprint*. They act on different bottlenecks, so the natural question is whether they compose — or whether a quantized target degrades the draft acceptance that speculative decoding depends on.

We re-ran both models with the **target weights quantized to mxfp4** (the draft model is untouched). A quantized target is cheaper to verify (fp4 weights → ~¼ the HBM traffic in memory-bound batch-1 decode), and as long as it still agrees with the drafter about as often, DFlash keeps committing the same long blocks over a faster verify pass. As the [results](#results-quantizing-the-target-model) show, that is exactly what happens.

## End-to-End Performance Results

### Test Environment

| Component | Version / Setting |
| --- | --- |
| Hardware | AMD Instinct **MI355X** (single GPU, `gfx950`), TP=1 |
| vLLM | `0.22.1rc1.dev43+g8c3cc98cf` (ROCm build) |
| ROCm | 7.2.3 |
| PyTorch | 2.10.0 |
| AITER | 0.1.13 (bf16) · ≥ 0.1.16.post2 (mxfp4 MoE native path) |
| Precision | `bfloat16`, `mxfp4` (target weights, quantized by AMD quantization tool **Quark**) |
| Target models | `Qwen/Qwen3.5-27B` (dense), `Qwen/Qwen3.5-35B-A3B` (MoE) |
| Draft models | `z-lab/Qwen3.5-27B-DFlash`, `z-lab/Qwen3.5-35B-A3B-DFlash` |
| Workloads | GSM8K, MATH500, HumanEval, MBPP, MT-Bench |
| Concurrency | 1 and 32; continuous batching; max output 4096 tokens |

Config mapping (per the DFlash model card): `t3` → MTP steps=3 / DFlash block=4, `t7` → steps=7 / block=8, `t15` → steps=15 / block=16.

### Launching the Server

**Common environment** (shared across all runs):

```bash
# MoE mxfp4 only: enable the native AITER fp4 MoE kernel (needs AITER >= 0.1.16.post2).
# Without it, the MoE path falls back to a dequant-to-bf16 emulation that erases the gain.
export VLLM_ROCM_USE_AITER=1
```

**Common launch command** (autoregressive baseline):

```bash
vllm serve Qwen/Qwen3.5-27B \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.9
```

**DFlash** — add a `--speculative-config` pointing at the block-diffusion draft model
(`num_speculative_tokens` = 3 / 7 / 15 for block = 4 / 8 / 16):

```bash
vllm serve Qwen/Qwen3.5-27B \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.9 \
    --speculative-config '{"method": "dflash",
                           "model": "z-lab/Qwen3.5-27B-DFlash",
                           "num_speculative_tokens": 15,
                           "draft_tensor_parallel_size": 1}'
```

**MTP** — the native Qwen3.5 multi-token-prediction path, no separate draft model:

```bash
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 15}'
```

**mxfp4 target** — point at the mxfp4 checkpoint and keep everything else identical (the draft model is unchanged). For the dense 27B this resolves to a fused fp4 GEMM automatically; for the MoE 35B-A3B, `VLLM_ROCM_USE_AITER=1` selects the native fp4 MoE kernel:

```bash
vllm serve Qwen/Qwen3.5-27B-MXFP4 \
    ... same flags ... \
    --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash",
                           "num_speculative_tokens": 15, "draft_tensor_parallel_size": 1}'
```

### Benchmark Commands

We drive the server with the [z-lab/dflash](https://github.com/z-lab/dflash) [[2](#references)] benchmark client, sweeping every `(dataset, concurrency)` pair:

```bash
python -m dflash.benchmark \
    --backend vllm \
    --base-url http://127.0.0.1:8000 \
    --model Qwen/Qwen3.5-27B \
    --dataset gsm8k \
    --num-prompts 128 \
    --concurrency 1        # repeated for concurrency 32
```

Throughput = generated output tokens / wall-clock time. Speedup = config / autoregressive baseline at the same workload, concurrency, **and precision**. In throughput tables **bold** marks the fastest speculative config per row; in accept-length tables **bold** marks the higher value within each matched MTP/DFlash pair.

### Results: Throughput and Speedup

For latency-bound single-request serving — where speculative decoding matters most — **DFlash is the better config on every 27B workload (shown in Figure 3) and on workloads for 35B-A3B(shown in Figure 4).** Speculative decoding has the least speedup on MT-Bench which is consistent with the result from DFlash paper [[3](#references)]. And we can also conclude that DFlash or more generally, speculative decoding is more beneficial for low-concurrency scenarios. That's because the arithmetic intensity at high concurrency is high enough which means the hardware is much more saturated. In this case, verification time of target model will increase by a large margin damaging the overall speedup.

```{image} ./images/speedup_c1.png
:label: speedup_c1
:alt: speedup_c1
:width: 50%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="margin-left: 10%; width: 80%; font-size: 0.9em; color: #555;"><strong>Figure 3.</strong> <em>Single-request (concurrency 1) speedup vs the autoregressive baseline. DFlash block=16 (dark orange) leads on every 27B workload and on the reasoning/coding workloads for the MoE 35B-A3B.</em></p>

```{image} ./images/speedup_c32.png
:label: speedup_c32
:alt: speedup_c32
:width: 50%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="margin-left: 10%; width: 80%; font-size: 0.9em; color: #555;"><strong>Figure 4.</strong> <em>Batched (concurrency 32) speedup. DFlash block=16 still leads the high-acceptance workloads (GSM8K, HumanEval, MATH500), but at a full batch the verify step is already compute-bound, so on shorter-acceptance workloads (MBPP, MT-Bench) the smaller <strong>block=8</strong> — or even MTP steps=3 — is the safer pick.</em></p>

### Results: Acceptance Length

Acceptance length is where block diffusion pulls away: going from budget 4 → 8 → 16, MTP climbs but flattens, while DFlash keeps rising. On 27B HumanEval, DFlash reaches **10.38** accepted tokens per target step versus MTP's 8.02.

| Workload | MTP steps=3 | DFlash block=4 | MTP steps=7 | DFlash block=8 | MTP steps=15 | DFlash block=16 |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 3.675 | **3.690** | 5.869 | **6.177** | 7.371 | **8.495** |
| humaneval | 3.684 | **3.755** | 6.027 | **6.625** | 8.022 | **10.377** |
| math500 | 3.664 | **3.698** | 5.833 | **6.285** | 7.143 | **8.829** |
| mbpp | 3.424 | **3.510** | 4.909 | **5.453** | 5.519 | **6.994** |
| MT-Bench | **3.133** | 3.087 | 4.257 | **4.316** | 4.820 | **5.242** |

<p style="font-size: 0.9em; color: #555;"><strong>Table 1.</strong> <em>Qwen3.5-27B — mean accepted tokens / target step (concurrency 1).</em></p>

<br/>

| Workload | MTP steps=3 | DFlash block=4 | MTP steps=7 | DFlash block=8 | MTP steps=15 | DFlash block=16 |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k | **3.582** | 3.581 | 5.589 | **5.714** | 6.861 | **7.109** |
| humaneval | 3.658 | **3.712** | 6.021 | **6.440** | 8.088 | **9.660** |
| math500 | **3.615** | 3.615 | 5.645 | **5.874** | 6.886 | **7.634** |
| mbpp | 3.407 | **3.434** | 4.836 | **5.163** | 5.435 | **6.133** |
| MT-Bench | **3.065** | 2.943 | **4.111** | 3.960 | **4.683** | 4.631 |

<p style="font-size: 0.9em; color: #555;"><strong>Table 2.</strong> <em>Qwen3.5-35B-A3B — mean accepted tokens / target step (concurrency 1).</em></p>

### Results: Quantizing the Target Model

This is the most interesting result: **quantizing the target to mxfp4 stacks cleanly with DFlash.** Acceptance is preserved and the cheaper verify pass lifts absolute throughput.

> **Note.** On the dense 27B, mxfp4 resolves to a fused fp4 GEMM out of the box. On the **MoE 35B-A3B** the native fp4 MoE kernel must be enabled explicitly (`VLLM_ROCM_USE_AITER=1`, AITER ≥ 0.1.16.post2); otherwise vLLM falls back to a dequant-to-bf16 path that re-materializes the full expert stack every step and erases the gain. All 35B-A3B mxfp4 numbers below use the native fp4 path.

**Acceptance length vs the bf16 target.** Quantizing the target barely moves acceptance — every difference is within run-to-run noise. DFlash's draft quality does not depend on the target being full precision.

<p style="font-size: 0.9em; color: #555;"><em>DFlash block=16, concurrency 1 — mean accepted tokens / target step</em></p>

| Workload | 27B bf16 | 27B mxfp4 | Δ | 35B-A3B bf16 | 35B-A3B mxfp4 | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 8.495 | 8.445 | −0.05 | 7.109 | 7.177 | +0.07 |
| humaneval | 10.377 | 10.329 | −0.05 | 9.660 | 9.395 | −0.27 |
| math500 | 8.829 | 8.784 | −0.05 | 7.634 | 7.690 | +0.06 |
| mbpp | 6.994 | 6.967 | −0.03 | 6.133 | 5.972 | −0.16 |
| MT-Bench | 5.242 | 5.094 | −0.15 | 4.631 | 4.509 | −0.12 |

**Per-position acceptance rate (marginal).** Zooming into the *shape* of acceptance confirms it. The marginal accept rate at position *i* is `P(accept ≥ i+1 draft tokens)` — the survival curve vLLM prints as "Per-position acceptance rate". Overlaying bf16 (solid) and mxfp4 (dashed), the dashed curves land right on the solid ones at every draft depth, as shown in *Figure 5*

```{image} ./images/perpos_quant.png
:label: perpos_quant
:alt: perpos_quant
:width: 50%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="margin-left: 10%; width: 80%; font-size: 0.9em; color: #555;"><strong>Figure 5.</strong> <em>Per-position marginal acceptance for DFlash block=16 at concurrency 1. Each color is a dataset; solid = bf16 target, dashed = mxfp4 target. The mean per-position gap is well under 0.02 — quantization does not systematically shift acceptance at any depth.</em></p>

**Throughput vs the bf16 target.** Because acceptance holds while the verify pass gets cheaper, the mxfp4 target lifts absolute single-request throughput by ~10–20% across the board — the two optimizations act on different bottlenecks and simply add up, as shown in *Figure 6*. (Note that the mxfp4 models are not fully optimized on vLLM.)

```{image} ./images/quant_throughput.png
:label: quant_throughput
:alt: quant_throughput
:width: 50%
:align: center
:max-width: 700px
:class: dark-light
```

<p style="margin-left: 10%; width: 80%; font-size: 0.9em; color: #555;"><strong>Figure 6.</strong> <em>DFlash block=16, concurrency 1. mxfp4 (orange) raises absolute tok/s on every workload; labels show the mxfp4 throughput and its lift over the bf16 target.</em></p>

The best single-request configuration on each model is **mxfp4 target + DFlash block=16**, reaching 460 tok/s on 27B and 600 tok/s on 35B-A3B.

### Output Quality

DFlash and MTP both use lossless rejection sampling: speculative decoding does not change the target model's output distribution — it only changes *how fast* those tokens are produced. The mxfp4 results reflect the target's own quantization (orthogonal to speculation), and mxfp4 leaves the draft acceptance untouched, as the per-position curves
above show.

## Summary

In this blog you learned how speculative decoding accelerates latency-bound LLM inference, why autoregressive drafters cap the speedup, and how DFlash's block-diffusion drafting lifts that cap on AMD Instinct MI355X. You walked through the full recipe — launching vLLM on ROCm with a `--speculative-config`, sweeping the DFlash, MTP, and autoregressive baselines across five workloads, and reading the results through both throughput and acceptance length. Speculative decoding has two moving parts on Instinct hardware, and you saw both land cleanly on MI355X through vLLM on ROCm:

- **DFlash block-diffusion drafting** [[1](#references)] delivers **up to ~5×** single-request throughput on Qwen3.5-27B (5.02× on MATH500) and **up to 3.27×** on the MoE 35B-A3B, beating native MTP on nearly every matched setting — driven by acceptance lengths that keep growing with block size (10.38 accepted tokens/step on 27B HumanEval) because a diffusion block is nearly free to draft.
- **mxfp4 target quantization stacks on top**: acceptance is preserved (per-position gap < 0.02) while the cheaper fp4 verify pass adds another ~10–20% absolute throughput — best config on each model is mxfp4 + DFlash block=16.

The takeaway: block diffusion is a portable way to push acceptance depth without paying an autoregressive drafting tax, and it composes with quantization — no vendor-specific trick required. You can reproduce every number here with the [z-lab/dflash](https://github.com/z-lab/dflash) [[2](#references)] benchmark client and the launch commands above.

We are continuing to tune the mxfp4 path on vLLM (the MoE numbers are not yet fully optimized) and to extend this work to larger models, longer draft blocks, and higher-concurrency serving. Follow the [ROCm Blogs](https://rocm.blogs.amd.com) for upcoming posts from the AMD team on speculative decoding, quantization with **Quark**, and end-to-end inference optimization on AMD Instinct GPUs.

## References

1. [DFlash project page — Z Lab.](https://z-lab.ai/projects/dflash/)
2. [DFlash source and benchmark client — z-lab/dflash.](https://github.com/z-lab/dflash)
3. [DFlash paper.](https://arxiv.org/abs/2602.06036)
4. [vLLM.](https://github.com/vllm-project/vllm)
5. [ROCm.](https://rocm.docs.amd.com)
6. [Qwen3.5 models.](https://huggingface.co/Qwen)

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

AMD, the AMD Arrow logo, AMD Instinct, ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
