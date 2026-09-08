---
blogpost: true
blog_title: "Efficiently Serving NVFP4 Models on AMD Instinct™ MI350X/MI355X Accelerators via Online NVFP4 to Quark MXFP4 Requantization"
date: "08 Sep 2026"
author: "Kelin Zeng, Bowen Bao, Spandan Tiwari, Ashish Sirasao, Hai Xiao"
thumbnail: 'nvfp4-to-mxfp4-thumbnail.png'
tags: "AI/ML, HPC, LLM, Optimization, Performance, Serving"
category: "Software tools & optimizations"
target_audience: "AI Model Optimization Developers, AI Model Inference Service Providers, Open-Source Project Contributors"
key_value_propositions: "Efficiently Serving NVFP4 Models on AMD Hardware without NVFP4 Compute Capabilities"
language: English
myst:
    html_meta:
        "author": "Kelin Zeng, Bowen Bao, Spandan Tiwari, Ashish Sirasao, Hai Xiao"
        "description lang=en": "Serve NVFP4 models on MI350X/MI355X via SGLang's online NVFP4 to MXFP4 requantization: no preprocessing, minimal accuracy impact, native throughput."
        "keywords": "NVFP4, MXFP4, Requantization, SGLang, AMD Instinct, MI350X, MI355X, FP4, Quantization, Inference, Serving"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "Open-Source Tools"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Kelin Zeng, Bowen Bao, Spandan Tiwari, Ashish Sirasao, Hai Xiao"
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

# Efficiently Serving NVFP4 Models on AMD Instinct™ MI350X/MI355X Accelerators via Online NVFP4 to Quark MXFP4 Requantization

A growing share of frontier open-weight models are available in NVFP4 format, but on some AMD
GPUs (MI350X / MI355X), FP4 compute support is limited to MXFP4. This means NVFP4
checkpoints couldn't run on the native 4-bit acceleration path these GPUs provide.

To close that gap, we present an online NVFP4 to MXFP4 requantization pipeline in SGLang that
converts NVFP4 weights to MXFP4 once at weight loading time and serves them through the AMD GPU's
native MXFP4 path. In this blog you will explore how the pipeline is designed, see how its serving
throughput and accuracy stack up against native MXFP4 checkpoints and an emulated NVFP4 reference,
and learn how to serve your own NVFP4 checkpoints directly on MI350X / MI355X GPUs,
with three key benefits:

- **No offline preprocessing** is needed, with SGLang server launched directly using an NVFP4
  checkpoint, requantization is done automatically and efficiently at load time;
- **Accuracy on par with native NVFP4**, evaluated against an emulated NVFP4 reference, the requantized
  path shows no meaningful accuracy loss;
- **Production-ready, high-performance MXFP4 inference pipeline** is used, enabling serving throughput
  on par with that of the native MXFP4 checkpoints on MI350X / MI355X GPUs.

## Background

NVFP4 is a popular 4-bit floating-point format (packed E2M1 with an FP8 per-block
scale and an FP32 per-tensor scale). Checkpoints like
`nvidia/Qwen3.5-397B-A17B-NVFP4` and `nvidia/Kimi-K2.6-NVFP4` fit very large models into a
fraction of the memory their bf16 versions would need.

However, there is a catch for AMD GPU users: GPUs like MI350X and MI355X implement their
native 4-bit compute pipeline in MXFP4, not NVFP4. The two formats both pack two FP4 (E2M1)
values per byte, but they differ in the following important aspects:

| Property | NVFP4 | MXFP4 |
| --- | --- | --- |
| Block size | 16 | 32 |
| Block scale | FP8 E4M3 | E8M0 (power-of-two) |
| Per-tensor scale | FP32 | None |
| Native compute support on MI350X / MI355X | No | **Yes** |

Because the block sizes and scale encodings differ, an NVFP4
tensor can't be fed directly into an MXFP4 serving backend. Historically,
this left users with two options: run an offline conversion tool and manage a
second copy of every NVFP4 checkpoint, or emulate NVFP4 by dequantizing the weights
to bf16 and running standard bf16 GEMM (slow and impractical for large scale inference
with the largest models).

In this blog, we present a third option that avoids both downsides: an online NVFP4
to MXFP4 requantization pipeline integrated in [SGLang](https://github.com/sgl-project/sglang) that
enables efficient NVFP4 checkpoint serving on AMD GPUs like the MI350X and MI355X.
The rest of this post walks through the pipeline's design, performance and accuracy comparison, and some usage guidance.

## How it Works

When serving in SGLang using the `--quantization quark_mxfp4` flag, SGLang will automatically
detect the NVFP4 source format, stream each weight tensor through a requantization
step, and hand the resulting MXFP4 weights to the native MXFP4 compute kernels. Layers the
checkpoint producer chose to keep in higher precision, like FP8 layers in mixed-precision
checkpoints, will branch off to their own load-as-is paths (Figure 1).

<!-- markdownlint-disable -->
```{figure} ./images/pipeline.png
:align: center
:width: 900px
:alt: Online NVFP4 to MXFP4 requantization pipeline
Figure 1. The online requantization pipeline.
```

Note that the requantization step only runs once during model load on a per-layer basis, so there is
no per-request conversion overhead, and at steady state the serving throughput of this pipeline is
indistinguishable from that of a natively MXFP4-quantized one.

### Efficient Memory Usage

A naive requantization approach that dequantizes the whole model to bf16 at once before
requantizing would need hundreds of gigabytes of disk scratch space and memory for a frontier
model. Instead, with the online NVFP4 to MXFP4 requantization feature in SGLang, we requantize
one layer at a time as its weights load: materializing just that layer's dequantized intermediate,
quantizing it to MXFP4, writing the result straight into a preallocated destination, and freeing
the intermediate before moving on. Peak extra memory in the pipeline is bounded by a single layer,
not the entire model.

### Support for Different Source Formats and Configs

NVFP4 checkpoints can come in a variety of formats, so the pipeline handles the common
variants out of the box:

- **Both ModelOpt and AMD Quark NVFP4 exports are recognized:** Source checkpoint metadata is read from either `config.json`
  or a standalone `hf_quant_config.json`.
- **Producer excludes honored:** Modules deliberately left in higher precision (listed under
 `ignore` / `exclude_modules` / `exclude`) are automatically kept out of the requantization
 step, so a layer meant to stay in high precision never gets silently quantized.
- **Mixed-precision support:** Checkpoints like `nvidia/Qwen3.5-397B-A17B-NVFP4-V2`
  (`"quant_algo": "MIXED_PRECISION"`) utilize different precision formats per layer. This is also
  automatically recognized and supported.

### Activation MXFP4 Quantization

At serve time, activations are quantized to MXFP4 dynamically for
each batch rather than from stored scales. This means no calibration pass and no pre-computed
activation scales are needed, and both operands of every GEMM are in MXFP4, so the matmul runs
directly on the GPU's native 4-bit hardware acceleration path.

## Performance Comparison of Online NVFP4 to MXFP4 Requant vs. Native AMD MXFP4 Checkpoints

Performance is always one of the most important questions when it comes to online
requantization. Does the online requantization path give up any performance compared to a checkpoint that was
quantized to MXFP4 offline and shipped as-is?

AMD publishes native MXFP4 checkpoints for many popular models, so we perform a direct
serving throughput comparison: nvidia NVFP4 checkpoints with `--quantization quark_mxfp4`
(online NVFP4 to MXFP4 requant) vs. an AMD offline quantized MXFP4 checkpoint with the same
quantization configuration on an identical serving stack and workload. All results presented
below are from a steady state (warmed-up, averaged across 10 runs). As shown in Figure 2 below,
the two paths track each other closely across all five model checkpoints tested.

<!-- markdownlint-disable -->
```{figure} ./images/native_vs_online.png
:align: center
:width: 900px
:alt: Steady-state throughput of online requant vs. native AMD MXFP4
Figure 2. Steady-state output-token throughput of the online NVFP4 to MXFP4 requant path vs. native AMD MXFP4 checkpoints across five models.
```

| Model | TP | Online NVFP4 to MXFP4 (tok/s) | Native AMD MXFP4 (tok/s) | Δ native | med ITL (online / native) |
| --- | --- | --- | --- | --- | --- |
| MiniMax-M2.7 | 4 | **3516.0** | 3491.2 | -0.7% | 14.28 / 14.18 ms |
| Qwen3.5-397B-A17B | 4 | 2963.2 | **2974.4** | +0.4% | 17.23 / 17.19 ms |
| Qwen3.5-397B-A17B-V2 | 4 | 3007.5 | **3014.7** | +0.2% | 17.30 / 17.66 ms |
| DeepSeek-R1 | 8 | **2799.0** | 2773.2 | -0.9% | 18.90 / 19.14 ms |
| Kimi-K2.6 | 8 | 2148.8 | **2170.4** | +1.0% | 25.59 / 25.09 ms |

The benchmarked online NVFP4 to MXFP4 requant path throughput is very similar to
that of the Native AMD MXFP4 path, with small deltas within
run-to-run noise. The online path shows no significant steady-state throughput
difference for converting at load instead of shipping a pre-quantized checkpoint.
Commands and configuration to reproduce these throughput results are available in
[Appendix C](#appendix-c-reproducing-the-results-in-this-blog).

### The MXFP4 Serving Path

The high serving throughput that the online NVFP4 to MXFP4 requantization feature
enables is all thanks to the underlying MXFP4 serving path that's designed
specifically for the CDNA4 hardware's MXFP4 compute capabilities. Several optimizations in SGLang's AMD MXFP4 path work together to improve FP4 serving throughput
on AMD GPUs:

**Native FP4 matmul:** Both the dense linear layers and the MoE experts dispatch to
AMD's AITER MXFP4 GEMM and fused-MoE kernels, which target the CDNA4 scaled matrix-core
(MFMA) instructions directly. These units consume the OCP MXFP4 microscaling format
natively, so the 4-bit operands feed straight into the tensor cores with no separate
intermediate dequantize step.

**Fusing activation quantization + GEMM:** A W4A4 layer needs its activations in
MXFP4 as well as its weights. For small-batch decode GEMMs, where kernel launch and
memory-traffic costs dominate, a kernel fusing activation quantization and GEMM is available
in the AMD MXFP4 acceleration path. The fused kernel loads activations directly in high
precision and quantizes them to MXFP4 right before feeding them into the MXFP4 matrix-core
instruction, eliminating an extra kernel launch and the additional HBM round-trip for the
activations. One downside, however, is that the kernel will need to quantize the same input
tiles for multiple output tiles, resulting in some redundant work. For large input sizes where
this redundant work outweighs the savings, a standalone quant kernel path is also available to
eliminate the redundant quantization. This path quantizes each element exactly once and sends packed
MXFP4 activations (instead of BF16) into a standard MXFP4 GEMM kernel separately. With the right
path selected for each case, activations are quantized and fed into MXFP4 compute with minimal
overhead.

**Pre-shuffled weights:** MXFP4 weights and their block scales are
reshuffled a single time when the model loads into the exact tile layout the MFMA
units expect, and the expert weight dimensions are padded up to the kernel's tile
alignment. As a result, weight reads in every forward pass are coalesced with no
need for in-kernel repacking or bounds handling.

Together, these optimizations (and many more) are what enable the online NVFP4 to
MXFP4 requantization feature to unlock the full potential of the AMD native 4-bit acceleration path, going beyond simply running NVFP4 checkpoints on AMD
GPUs.

## Accuracy Comparison and Analysis

Across every accuracy benchmark we ran, the online NVFP4 to MXFP4 requantization path shows
no meaningful accuracy loss. From a basic GSM8K sanity check that shows it recovers over 99%
of each source checkpoint's score (details in [Appendix A](#appendix-a-gsm8k-accuracy)),
to harder reasoning benchmarks like GPQA-Diamond and AIME25 that show it stays within
run-to-run seed noise of an emulated NVFP4 reference, we can conclude the feature causes no
systematic accuracy degradation. With a weight-level analysis, we will also discuss the intuition
behind these results and reinforce our conclusions.

### A Custom NVFP4 Emulation Path for Accuracy Reference

To measure the exact accuracy cost of the NVFP4 to MXFP4 conversion step in isolation,
we need a baseline that runs the same NVFP4 checkpoint but skips the MXFP4 step entirely.
This way, any significant difference is attributable to the format conversion. The
problem is that MI350X / MI355X have no native NVFP4 compute capabilities to run that baseline on.
Therefore, we built a small software NVFP4 emulation backend for SGLang
(based on the design and implementation described in [this previous blog](https://rocm.blogs.amd.com/software-tools-optimization/nvfp4-mi355/README.html)) to
overcome this gap.

The emulation backend's design is deliberately simple, prioritizing numerical fidelity
over speed. At each linear/MoE layer it dequantizes the NVFP4 weights back to BF16
(unpacking the E2M1 values and applying the FP8 block scale and FP32 per-tensor scale),
passes the activations through an NVFP4 quantize-dequantize step with the same
rounding the real format would impose, and then runs an ordinary BF16 matmul. This
reproduces NVFP4's exact numerics on hardware that can't execute NVFP4 natively, at
the cost of running the GEMMs in BF16.

### Advanced Reasoning Benchmarks

To isolate the exact accuracy loss via *format conversion* alone (from quantizing from
the base model's higher precision to a 4-bit block scale format), we compare the online requant
serving path to the emulated NVFP4 path head-to-head on two advanced reasoning tasks:

- **GPQA-Diamond-CoT**: graduate-level, "Google-proof" multiple-choice
  questions in biology, physics, and chemistry, answered with chain-of-thought.
  It probes deep domain knowledge and careful step-by-step reasoning, where small
  numerical perturbations from quantization are most likely to change an answer.
- **AIME25**: competition math problems (American Invitational Mathematics Examination)
  with exact integer answers. It's a demanding test of long-form symbolic reasoning;
  the small item count makes it high-variance, so we average results across different
  runs with unique seeds.

<!-- markdownlint-disable -->
```{figure} ./images/reasoning_parity.png
:align: center
:width: 900px
:alt: Reasoning accuracy of online requant vs. NVFP4 emulation
Figure 3. Averaged reasoning accuracy on AIME25 (left) and GPQA-Diamond-CoT (right), comparing online NVFP4 to MXFP4 requant against the NVFP4 emulation reference.
```

Results in Figure 3 are averaged scores across multiple evaluation runs, each with unique seeds. Online NVFP4 to MXFP4 requant and NVFP4 emulation agree within seed noise across all task questions,
with final averaged accuracy score within ±0.05 of each other (illustrated in Figure 4). Re-running the same configuration
under different random seeds moves the score by roughly 5–10% on these small and high variance benchmarks,
so any gap below that band is indistinguishable from the model's own run-to-run variance. It should be
noted that the difference between the two paths is also unsystematic (NVFP4 emulation wins some, loses
others), which is the signature of sampling noise rather than a real precision gap.

<!-- markdownlint-disable -->
```{figure} ./images/accuracy_delta.png
:align: center
:width: 750px
:alt: Per-task accuracy delta between NVFP4 emulation and online requant
Figure 4. Per-task accuracy difference (NVFP4 emulation minus online requant) across all model/benchmark pairs.
```

### Identical Answers from Divergent Traces

To further understand the difference in behavior of the NVFP4 to MXFP4 requantized
runs, we studied the responses of each model under greedy decoding with byte-identical
prompts. Although 100% of generations diverge between the two paths, the
final answer is preserved in the large majority of cases (as shown in left plot of Figure 5). Additionally, when outcomes differ,
it's near-random which path lands on the correct answer, with both paths generating
coherent reasoning text.

<!-- markdownlint-disable -->
```{figure} ./images/divergence.png
:align: center
:width: 900px
:alt: Trace divergence vs. preserved correctness outcomes
Figure 5. Left: although 100% of greedy-decoded traces diverge between the two paths, the correctness outcome is preserved for 90% (AIME25) and 92% (GPQA-Diamond-CoT) of questions. Right: per-question outcomes on GPQA-Diamond-CoT, where disagreements split between the two paths with no evidence of online requant accuracy degradation.
```

This shows that online NVFP4 to MXFP4 requantization does not measurably degrade accuracy
relative to the native NVFP4 quantized checkpoint.

### Why doesn't a Second 4-bit Rounding Step Compound the Error?

The benchmark parity above raises an obvious question: how does going from one format of
pre-quantized 4-bit weights to a different format of 4-bit weights *not* compound quantization
error? It's a fair concern, so we measured it directly on Qwen3.5-397B-A17B. We pulled the same
MoE expert weights in three forms: the bf16 source release (`Qwen/Qwen3.5-397B-A17B`), the NVFP4
version (`nvidia/Qwen3.5-397B-A17B-NVFP4`), and the online NVFP4 to MXFP4 reconstruction.

After comparing them tensor by tensor, we saw that MXFP4’s coarser 32-wide E8M0 blocks do round more
aggressively than NVFP4’s 16-wide FP8-scaled blocks. However, we also make three key observations:

1. **The extra rounding is zero-mean noise, not a systematic shift.** The error
   the MXFP4 step adds on top of NVFP4 is symmetric and centered on zero (mean/σ ≈
   10⁻⁴), so it perturbs individual weights without biasing the entire tensor in any
   direction.
2. **The two quantization errors are independent, so they add in quadrature.** NVFP4's
   rounding error and MXFP4's rounding error are nearly uncorrelated (mean pairwise
   correlation ≈ −0.04). Independent errors combine as `√(e₁² + e₂²)`, which is far
   smaller than the `e₁ + e₂` you'd get if the second step amplified the first.
3. **Matmuls average out the noise.** What the model actually computes is
   `y = W·x`, a sum over thousands of weights. Zero-mean, uncorrelated per-weight
   errors partially cancel in that reduction, so the *output* signal-to-quantization-noise
   ratio stays high, around ~17 dB for NVFP4 to MXFP4, only ~3.6 dB below NVFP4 itself.

<!-- markdownlint-disable -->
```{figure} ./images/weight_precision.png
:align: center
:width: 1000px
:alt: Weight-level analysis of why requant does not compound error
Figure 6. Weight-level analysis on Qwen3.5-397B-A17B MoE experts.
```

The three plots in Figure 6 visualize each of these observations. The data used in the plots is drawn from
a representative sample of MoE expert weight tensors of Qwen3.5-397B-A17B, compared across all three versions (the bf16
source, the NVFP4 release, and the online NVFP4 to MXFP4 reconstruction).

Let `e₁` be the error NVFP4 already introduced versus the bf16 source, and `e₂` the
*additional* error the MXFP4 requant step layers on top:

- **Left (zero-mean noise):** the two histograms are the distributions of `e₁` and
  of `e₂` across a tensor's weights. Both are symmetric and centered around zero. `e₂`
  (the MXFP4 step) is a bit wider than `e₁`, but it has no directional bias.
- **Middle (quadrature, not compounding):** *correlation* measures whether two error
  signals move together. +1 means they reinforce, −1 means they cancel, and 0 means
  no relationship. The measured correlation between `e₁` and `e₂` averages to around
  −0.04, i.e. essentially independent, so their magnitudes combine in quadrature
  (`√(e₁² + e₂²)`) rather than adding linearly (`e₁ + e₂`). This shows that the second
  4-bit rounding costs far less than what one might expect to be a compounding error.
- **Right (matmuls average out the noise):** individual weight errors of around 14% sound
  alarming out of context, but the model never uses a weight alone. Every output is a
  dot product over thousands of weights. Summing `n` independent, zero-mean
  errors grows the true signal with `n` but the error only with `√n`, so the signal-to-noise
  ratio (SNR) *improves* by roughly `√n` (e.g. 64× for a 4096-wide projection). In the right
  plot, SQNR (signal-to-quantization-noise ratio) is shown in decibels (higher is cleaner,
  and every −3 dB means about double the noise). NVFP4 to MXFP4 Requant lands at around 17 dB,
  only about 3.6 dB under NVFP4 alone. Although the extra rounding roughly doubles the output noise,
  from a high enough baseline, the result stays firmly signal-dominated.

The above observations and analysis serve as helpful intuition for why we don’t see a significant drop
in model generation accuracy after applying online NVFP4 to MXFP4 requantization. It’s clear that the
weights were not requantized without loss, but the second rounding injects independent and zero-mean
noise rather than systematic error that compounds the first. It’s likely due to these reasons that the
final quantization error stays well within the network’s own tolerance, resulting in the reasoning
benchmark parity as shown above.

## Expected Applications and Potential Drawbacks

The online NVFP4 to MXFP4 requantization feature can especially help in the following cases:

- You're serving a checkpoint that's only available in NVFP4 precision on MI350X / MI355X GPUs,
  and want to fully utilize the available hardware's native 4-bit acceleration path.
- You do not want to requantize an NVFP4 model locally, either because you lack the disk space to
  store a second copy of the weights or because you simply prefer the convenience of running NVFP4
  checkpoints on AMD hardware that only supports native MXFP4 compute.
- You want to quickly test a checkpoint that has a specific quantization configuration,
  and no publicly available MXFP4 checkpoints offer that same quantization configuration.

However, there are some drawbacks to the online NVFP4 to MXFP4 requantization feature.
In the following cases, you may want to consider another alternative:

- You're on hardware without native MXFP4 compute or your checkpoint uses a source format not yet supported (please refer to [relevant
  documentation](https://docs.sglang.io/docs/advanced_features/quantization#quark_mxfp4-online-quantization-method)
  of the online NVFP4 to MXFP4 requantization feature in SGLang for all supported formats).
- You need results that are bit-exact with the source NVFP4 checkpoint:
  requantization to MXFP4's coarser 32-element E8M0 blocks is not lossless in principle,
  so if you would like to perform any accuracy-sensitive tests specific to your source
  NVFP4 checkpoints, the online NVFP4 to MXFP4 requantization feature may not be the best
  fit.
- Your applications are extremely sensitive to model load time: although steady-state
  serving is unaffected, requantization does add a one-time dequantize + requantize pass
  at load time that can skew any load-time related performance benchmarks/profiling you
  might perform (see [Appendix B: Load-Time Cost of Requantization](#appendix-b-load-time-cost-of-requantization)
  for more details).

## Summary

In this blog you explored how the online NVFP4 to MXFP4 requantization pipeline in SGLang lets you
serve NVFP4 checkpoints directly on AMD Instinct MI350X / MI355X GPUs with serving throughput matching native MXFP4
checkpoints.

Whether you're serving a model that ships only in NVFP4, or simply want to avoid managing a second
copy of your weights, this feature lets you get the most out of AMD's native 4-bit acceleration path
today. We plan to continue expanding online quantization format coverage and FP4 serving optimizations in SGLang,
so stay tuned for future blogs on quantization and inference on AMD Instinct hardware.

## Acknowledgements

We would like to express our thanks to Zhao Lin, Wei Luo, and the AMD Quark Team, for their insightful feedback and technical guidance, which helped inform parts of this work. We also thank Felix Marty and Yi-Chih Cheng (Jacky) for valuable technical discussions, ideation, and implementation of the NVFP4 to MXFP4 requantization feature in SGLang.

## Appendix A: GSM8K Accuracy

GSM8K is a set of grade-school math word problems that tests multi-step arithmetic
reasoning. It's a fast and stable sanity check for whether quantization has broken a model's
basic reasoning. Figure 7 below and the table that follows give the per-model recovery of the
online NVFP4 to MXFP4 path versus each model's original source checkpoint (the higher-precision
checkpoint each NVFP4 model was quantized from, e.g. `Qwen/Qwen3.5-397B-A17B`, `moonshotai/Kimi-K2.6`).

<!-- markdownlint-disable -->
```{figure} ./images/gsm8k_recovery.png
:align: center
:width: 900px
:alt: GSM8K accuracy of online requant vs. original release
Figure 7. GSM8K accuracy of the online NVFP4 to MXFP4 path vs. each model's original higher-precision release.
```

| Model | Orig release | NVFP4 to MXFP4 | % recovery |
| --- | --- | --- | --- |
| MiniMax-M2.7 | 0.918 | 0.924 | 100.7% |
| Qwen3.5-397B-A17B | 0.954 | 0.945 | 99.1% |
| Qwen3.5-397B-A17B-V2 | 0.954 | 0.941 | 98.7% |
| Kimi-K2.6 | 0.939 | 0.930 | 99.0% |
| DeepSeek-R1 | 0.958 | 0.950 | 99.2% |

*% recovery is NVFP4 to MXFP4 accuracy / original-release accuracy; values above 100%
are within run-to-run noise of the source.*

## Appendix B: Load-Time Cost of Requantization

The online path adds a one-time NVFP4 to MXFP4 requantization step during weight loading. It
has no effect on steady-state serving throughput, but it does increase server startup time.
To quantify it, we measured the total end-to-end server startup time on both the requant path
and the native MXFP4 checkpoint path (which does no online requant). All experiments are done
on AMD MI350X GPUs.

| Model | TP | Native MXFP4 Startup Time (Seconds) | NVFP4 to MXFP4 Requant Startup Time (Seconds) | Requant - Native MXFP4 Time Delta |
| --- | --- | --- | --- | --- |
| DeepSeek-R1 | 8 | 78.16 / 78.09 | 112.27 / 112.12 | +34 s |
| Kimi-K2.6 | 8 | 88.24 / 86.09 | 134.17 / 151.80 | +55 s |
| MiniMax-M2.7 | 4 | 66.06 / 66.07 | 78.35 / 78.16 | +12 s |
| Qwen3.5-397B-A17B | 4 | 82.13 / 80.07 | 108.10 / 106.20 | +26 s |
| Qwen3.5-397B-A17B-V2  | 4 | 72.13 / 74.07 | 116.19 / 118.12 | +44 s |

Across all models we tested, requantization adds a one-time server startup overhead ranging
from 10 to 55 seconds (dependent on model size, TP size, and checkpoint quantization configuration).
In practice, this is generally an acceptable cost for the efficiency and convenience of serving an
NVFP4 checkpoint directly on hardware that only supports MXFP4 compute.

## Appendix C: Reproducing the Results in this Blog

Serving a model in SGLang with online NVFP4 to MXFP4 requantization at load time using the `--quantization quark_mxfp4` flag:
```bash
sglang serve --model-path <ckpt> \
    --tensor-parallel-size <TP> \
    --quantization quark_mxfp4
```

Accuracy benchmarking with lm-eval-harness:

GPQA-Diamond-CoT and/or AIME25:
```bash
lm_eval --model sglang \
  --model_args "pretrained=<ckpt>,tp_size=<TP>,quantization=quark_mxfp4,trust_remote_code=True" \
  --tasks <gpqa_diamond_cot_zeroshot|aime25> --apply_chat_template --seed <seed> \
  --gen_kwargs do_sample=True,temperature=<T>,top_p=0.95,top_k=40,max_gen_toks=64000 \
  --batch_size auto
```

GSM8K:
```bash
lm_eval --model sglang \
    --model_args "pretrained=<ckpt>,tp_size=<TP>,quantization=quark_mxfp4,trust_remote_code=True" \
    --tasks gsm8k \
    --gen_kwargs do_sample=False,temperature=0.0,max_gen_toks=32000 \
    --batch_size auto
```

Throughput benchmarking with `sglang.benchmark.serving`:
```bash
python3 -m sglang.benchmark.serving --backend sglang \
      --host 127.0.0.1 --port 30000 \
      --model <ckpt> --tokenizer <ckpt> \
      --dataset-name random --random-input-len 1024 --random-output-len 1024 \
      --random-range-ratio 1.0 --num-prompts 200 \
      --max-concurrency 64 --request-rate 20
```

**Note:** For a fair comparison, ensure the quantization configs of the checkpoints used match exactly, or utilize server launch flags (like `--disable-shared-experts-fusion`) to manually disable optimizations that are only routed to AMD/Nvidia released checkpoints.

### Environment Setup

| Env | Version |
| --- | --- |
| Hardware | 8× AMD Instinct MI350X GPUs |
| ROCm |  7.0.0 |
| Triton | 3.4.0 |
| lm-eval-harness | 0.4.12 |
| Checkpoints Used | `nvidia/{Qwen3.5-397B-A17B, Qwen3.5-397B-A17B-V2, MiniMax-M2.7, Kimi-K2.6, DeepSeek-R1}-NVFP4` |

- Sampling held constant for each checkpoint across different paths: `top_p=0.95, top_k=40, max_gen_toks=64000`; temperature 0.6 (Qwen/Kimi/DeepSeek-R1), 1.0 (MiniMax).
- Metric: GPQA `flexible-extract`, AIME `exact_match`, GSM8K `flexible-extract`.
- Throughput comparison vs. native AMD MXFP4 checkpoints (with `--quantization quark` flag) on the identical stack.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.

THIS INFORMATION IS PROVIDED “AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Results shown are from specific test configurations and may vary based on workload, model, and system configuration.

AMD, the AMD Arrow logo, AMD Instinct, AMD CDNA, ROCm, AMD Quark, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2026 Advanced Micro Devices, Inc. All rights reserved.
