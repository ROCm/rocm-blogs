---
blogpost: true
blog_title: "Accelerating Large-Scale LLM Inference on AMD Instinct™ MI350X/MI355X with Eagle3 and AMD Quark"
date: 3 Jul 2026
author: "Larry Li, Chun Fang, Chao Li, Xikai Meng, Andy Luo, Haichen Zhang, Bowen Bao, Spandan Tiwari, Ashish Sirasao"
thumbnail: 'eagle3-thumbnail.png'
tags: LLM, AI/ML, Serving, Performance, Benchmarking
category: Applications & models
language: English
target_audience: AI developers, AI practitioners, inference engineers
key_value_propositions: Eagle3 speculative decoding acceleration on AMD Instinct MI355X
myst:
    html_meta:
        "author": "Larry Li, Chun Fang, Chao Li, Xikai Meng, Andy Luo, Haichen Zhang, Bowen Bao, Spandan Tiwari, Ashish Sirasao"
        "description lang=en": "Learn how the AMD Quark team enables Eagle3 speculative decoding for Kimi-K2.5 and MiniMax-M2.5 on AMD Instinct MI355X GPUs with ROCm, vLLM, and InferenceX."
        "keywords": "LLM, Speculative Decoding, Eagle3, AMD Quark, vLLM, ROCm, AITER MLA, InferenceX, FP8, MI355X"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "Software & Ecosystem, AI & Intelligent Systems"
---

# Accelerating Large-Scale LLM Inference on AMD Instinct MI350X/MI355X with Eagle3 and AMD Quark

Large language model (LLM) inference is increasingly constrained by autoregressive decoding. Even when prefill is highly optimized, the decode phase still generates tokens one step at a time, and each step typically requires running the full target model. For large mixture-of-experts and attention-heavy models such as Kimi-K2.5 and MiniMax-M2.5, this sequential pattern limits serving throughput and increases latency for real-time applications.

Speculative decoding is one of the most practical ways to address this bottleneck. It is a lossless LLM inference acceleration technique that preserves the exact output distribution of the target model while improving decoding efficiency. It uses a smaller or lighter-weight draft model to propose multiple future tokens, then asks the original target model to verify those tokens in a single forward pass. When the draft model predicts tokens that the target model would also produce, those tokens can be accepted together, reducing the number of expensive target-model decode iterations.

Common speculative decoding approaches include small draft models, multi-token prediction (MTP), Medusa-style multi-head prediction, and feature-level drafting methods such as Eagle3 and DFlash. Among existing speculative decoding methods, EAGLE3 is particularly attractive due to its strong draft quality, high acceptance rate, and consistently competitive inference speedups.

In this blog, we introduce speculative decoding with Eagle3, then describe the recent work from the AMD Quark team that enables Eagle3-based acceleration for Kimi-K2.5 and MiniMax-M2.5 on AMD Instinct™ MI355X GPUs. The work spans AMD Quark FP8 draft-model quantization, ROCm/vLLM backend enablement, and InferenceX benchmark integration for Kimi-K2.5 BF16/FP8 Eagle3 and MiniMax-M2.5 BF16 Eagle3 configurations.

## Why Speculative Decoding and Eagle3 Matter

Standard autoregressive decoding emits one token per target-model step. If a model needs to generate 1,000 output tokens, the serving engine typically performs roughly 1,000 target-model decode iterations after prefill. This is expensive because each decode iteration touches the model weights, attention state, scheduler, and KV cache machinery.

Speculative decoding changes this process:

1. A draft model proposes several candidate next tokens.
2. The target model verifies those candidates in one pass.
3. Tokens that match the target model distribution are accepted.
4. Generation continues from the first rejected token, or from the end of the accepted block.

The key metric is the acceptance rate: how many draft tokens the target model can accept. Imagine the draft model proposes the next four tokens: if the target model agrees with all four, they can be accepted in a single verification step; if it disagrees on the third token, only the first two are accepted, and the remaining tokens must be regenerated. The more tokens that are accepted at a time, the fewer expensive decoding steps the target model needs to perform, resulting in higher throughput. Importantly, every draft token is checked by the target model before being emitted, so speculative decoding speeds up inference without changing the final output quality.

Eagle has been continuously improving over the past few years. It started with feature-level speculative decoding in Eagle, improved draft quality and acceptance rates in Eagle2, and further increased accuracy and speedups in Eagle3 by leveraging multi-layer features from the target model. Instead of relying on an unrelated small language model, it trains a draft module that is closely aligned with the target model. It uses training-time testing techniques and combines low-, mid-, and high-level semantic features from the target model, helping the draft model propose candidates that the verifier is more likely to accept.

For production inference, the important point is simple: Eagle3 can improve generation throughput while preserving the target model output behavior through verification.

## AMD Quark Team Contributions

The AMD Quark team worked on three complementary pieces of Eagle3 enablement:

- Eagle3 benchmark support in InferenceX for Kimi-K2.5 with a BF16 draft model.
- AMD Quark FP8 quantization of the Kimi-K2.5 Eagle3 draft model, enabling a lower-precision draft path while keeping its LM head unquantized and storing Quark FP8 metadata in the model config.
- Eagle3 benchmark support in the SemiAnalysis' InferenceX benchmark for MiniMax-M2.5 with a BF16 draft model.

The InferenceX work adds benchmark recipes, matrix validation, and launcher routing so Eagle3 configurations can be selected and swept in the same way as other single-node MI355X benchmark entries. The proposed InferenceX configurations cover common fixed sequence-length scenarios, including 1K/1K, 1K/8K, and 8K/1K prompt/output patterns, with tensor-parallel and expert-parallel search points such as TP/EP = 4/4 and 8/8.

The Kimi-K2.5 FP8 Eagle3 path uses [`amd/kimi-k2.5-eagle3-fp8`](https://huggingface.co/amd/kimi-k2.5-eagle3-fp8), an AMD Quark FP8 quantization of [`lightseekorg/kimi-k2.5-eagle3`](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3). This is especially useful because speculative decoding draft models are performance-sensitive: the draft needs to be accurate enough to achieve good acceptance rates, but also efficient enough that its overhead does not erase the speedup from accepting multiple tokens.

## ROCm and vLLM Backend Enablement

A key part of this work was enabling the fast ROCm AITER MLA attention backend to work with Eagle3 speculative decoding in vLLM. This contribution was implemented in [vLLM PR #39616](https://github.com/vllm-project/vllm/pull/39616). This PR resolves a KV-cache block-size incompatibility that previously forced a trade-off between fast AITER MLA decoding and Eagle3 speculative decoding support. By decoupling vLLM's block-size constraint from the kernel's internal token indexing and simplifying metadata handling during multi-token verification, it makes AITER MLA fully compatible with Eagle3. As a result, Kimi-K2.5 Eagle3 can now run with the fast AITER MLA backend on AMD MI350X/MI355X without falling back to slower attention implementations.

## InferenceX Benchmark Integration

The InferenceX work is represented by three PRs, alongside the upstream vLLM backend enablement PR:

| Area | PR | Main Contribution |
|------|----|-------------------|
| Kimi-K2.5 BF16 Eagle3 | [SemiAnalysisAI/InferenceX #1116](https://github.com/SemiAnalysisAI/InferenceX/pull/1116) | Adds Kimi-K2.5 MI355X Eagle3 benchmark support using the BF16 Eagle3 draft model, with `spec-decoding: eagle3` validation and launcher routing. |
| MiniMax-M2.5 BF16 Eagle3 | [SemiAnalysisAI/InferenceX #1234](https://github.com/SemiAnalysisAI/InferenceX/pull/1234) | Adds MiniMax-M2.5 MI355X Eagle3 benchmark support using a BF16 Eagle3 draft model, with sweep coverage for 1K/1K, 1K/8K, and 8K/1K. |
| Kimi-K2.5 FP8 Eagle3 | [SemiAnalysisAI/InferenceX #1515](https://github.com/SemiAnalysisAI/InferenceX/pull/1515) | Adds a Kimi-K2.5 MI355X Eagle3 FP8 draft path using the AMD Quark quantized draft model [`amd/kimi-k2.5-eagle3-fp8`](https://huggingface.co/amd/kimi-k2.5-eagle3-fp8). |
| ROCm/vLLM backend enablement | [vLLM #39616](https://github.com/vllm-project/vllm/pull/39616) | Enables the AITER MLA backend to work with Eagle3 speculative decoding on ROCm and is merged upstream. |

For the benchmark scripts, we also ensure chat-template handling is aligned with Eagle3 speculative decoding. Eagle3 draft models are trained on chat-formatted data, so benchmarking against raw prompts can distort acceptance rate and throughput. Using chat templates keeps the benchmark closer to realistic chat-serving traffic.

## Acceleration Results

The following draft results section lists only the 1K/1K workload, with ISL=1024 and OSL=1024. Speedup is computed as Eagle3 throughput divided by the corresponding no-speculative-decoding baseline throughput. Kimi-K2.5 results use AMD Instinct MI355X, TP=4, random prompts, `num_prompts=10 x concurrency`, `num_warmups=2 x concurrency`, and 10 seeds per cell. The merged Kimi table lists BF16 and FP8 draft paths side by side; the BF16 vLLM v0.19.0 sweep uses MML=2248, while the FP8 sweep uses MML=2304. Here, MML (max-model-len) is the maximum context length — the total number of tokens (prompt + generated output) that a vLLM model can process in a single request.

### Kimi K2.5 Eagle3: BF16 and AMD Quark FP8 Drafts

Docker images: BF16 sweep uses `vllm/vllm-openai-rocm:v0.19.0` (MML=2248); FP8 sweep uses `vllm/vllm-openai-rocm:nightly-fb1ac806c55a6dc96fe92261b80c8550e9c39d2f` (MML=2304).

Target model: [`amd/Kimi-K2.5-MXFP4`](https://huggingface.co/amd/Kimi-K2.5-MXFP4). BF16 draft model: [`lightseekorg/kimi-k2.5-eagle3`](https://huggingface.co/lightseekorg/kimi-k2.5-eagle3). FP8 draft model: [`amd/kimi-k2.5-eagle3-fp8`](https://huggingface.co/amd/kimi-k2.5-eagle3-fp8), quantized with AMD Quark FP8 metadata and sharing the BF16 target LM head. In this setup, the FP8 draft path dispatches through vLLM `RowWiseTorchFP8ScaledMMLinearKernel`, i.e. `torch._scaled_mm` over hipBLASLt row-wise scaled FP8 GEMM, rather than the AITER preshuffled FP8 path. The target MXFP4 model uses the ROCm FP4 ASM path via `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1`.

| Concurrency | No-spec (tok/s/GPU) | BF16 Eagle3 (tok/s/GPU) | FP8 Eagle3 (tok/s/GPU) |
|-------------|---------------------|-------------------------|------------------------|
| 4           | 82.7                | 157.0 (1.90x)           | 165.2 (2.00x)          |
| 8           | 142.2               | 269.1 (1.89x)           | 270.1 (1.90x)          |
| 16          | 220.5               | 399.6 (1.81x)           | 412.7 (1.87x)          |
| 32          | 342.2               | 627.6 (1.83x)           | 633.8 (1.85x)          |
| 64          | 533.3               | 901.6 (1.69x)           | 936.6 (1.76x)          |

### MiniMax M2.5 BF16 Eagle3

Docker image: `vllm/vllm-openai-rocm:nightly-4eafc729285e459a5fc96efd6f7b313b155cad48`

Target model: [`MiniMaxAI/MiniMax-M2.5`](https://huggingface.co/MiniMaxAI/MiniMax-M2.5). Draft model: [`MiniMax-M2.5-Eagle3`](https://huggingface.co/thoughtworks/MiniMax-M2.5-Eagle3/tree/main), BF16 draft path with `num_speculative_tokens=3` and `draft_tensor_parallel_size=1`. The numbers below use 1K/1K random prompts, TP=4/EP enabled, and 5 seeds per concurrency, with the same baseline-vs-Eagle3 setup used for the MiniMax reproduction.

| Concurrency | No-Spec (tok/s/GPU) | BF16 Eagle3 (tok/s/GPU) |
|-------------|---------------------|-------------------------|
| 64          | 822.0               | 1137.3 (1.38x)          |
| 32          | 484.8               | 759.0 (1.57x)           |
| 16          | 282.7               | 436.9 (1.55x)           |
| 8           | 162.6               | 277.9 (1.71x)           |
| 4           | 90.4                | 161.7 (1.79x)           |

Across the Kimi-K2.5 1K/1K sweep, the BF16 Eagle3 draft path delivers 1.69x to 1.90x throughput over the no-speculative-decoding baseline, while the AMD Quark FP8 Eagle3 draft path delivers 1.76x to 2.00x throughput. For MiniMax-M2.5, the BF16 Eagle3 draft path delivers 1.38x to 1.79x throughput over baseline across concurrency 64 to 4. The FP8 draft path for Kimi uses the currently integrated RowWise hipBLASLt FP8 draft-kernel path.

## Why 8K/1K Speedup Degrades

Eagle3 speculative decoding speedup degrades when the workload moves from 1K/1K to 8K/1K mainly because target-model verification becomes much more expensive on long KV caches. Acceptance rate stays roughly stable and the Eagle draft cost is nearly unchanged, but the target verification cost per accepted token grows from about 0.30 ms at 1K input to about 2.56 ms at 8K input, an about 8.5x increase that tracks the KV length growth.

This changes the balance of the spec loop: at 1K, target verification is only about 14% of draft-plus-verify model time, while at 8K it rises to about 59%. In short, the extra long-KV attention and TP/MoE communication cost in target verification eats most of the speculative decoding benefit at 8K/1K, even though draft-model quality does not regress. Increasing concurrency from 4 to 64 further worsens this degradation.

## Output Quality

For output quality, we refer to the GSM8K evaluation from a successful InferenceX Kimi-K2.5 Eagle3 run. That run completed successfully with the eval matrix and success-rate calculation, providing a lightweight check that Eagle3 acceleration did not introduce a visible quality regression on this math-reasoning task.

We keep quality validation brief in this blog because speculative decoding still verifies accepted tokens with the target model. The main takeaway is that the measured throughput gains come with a GSM8K sanity check showing no observed accuracy loss in the referenced InferenceX run.

## What This Enables

The combined work creates a practical path for Eagle3 acceleration on AMD Instinct MI355X systems:

- vLLM on ROCm can use AITER MLA together with Eagle3 speculative decoding for Kimi-K2.5-style MLA models.
- InferenceX can represent Eagle3 benchmark configurations explicitly through `spec-decoding: eagle3` and `spec-decoding: eagle3_fp8`.
- AMD Quark can reduce the draft-model cost through FP8 quantization, enabling a Kimi-K2.5 Eagle3 draft model that is smaller and more efficient while preserving the metadata needed by the serving stack.
- Benchmark recipes can compare BF16 and FP8 Eagle3 draft paths under consistent sequence-length, concurrency, TP/EP, and quality-validation settings.

This is the type of end-to-end enablement needed for production inference: model optimization, backend compatibility, serving integration, and benchmark coverage all have to work together.

## Summary

Eagle3 speculative decoding is a promising way to accelerate LLM generation by proposing multiple candidate tokens with a draft model and verifying them with the target model. On AMD Instinct MI355X GPUs, the AMD Quark team's work brings this approach closer to production use for Kimi-K2.5 and MiniMax-M2.5.

The merged vLLM ROCm contribution enables AITER MLA to work with Eagle3 speculative decoding, removing a backend compatibility limitation. In parallel, the InferenceX work adds benchmark support for Kimi-K2.5 BF16 Eagle3, Kimi-K2.5 AMD Quark FP8 Eagle3, and MiniMax-M2.5 BF16 Eagle3. The 1K/1K results show up to 1.90x speedup for Kimi-K2.5 BF16 Eagle3, up to 2.00x speedup for Kimi-K2.5 FP8 Eagle3, and up to 1.79x speedup for MiniMax-M2.5 BF16 Eagle3. The Quark FP8 draft model path demonstrates how quantization can reduce draft-model overhead while preserving the structure needed by the serving stack.

Together, these contributions demonstrate a full-stack path for high-throughput speculative decoding on AMD GPUs: AMD Quark for draft-model optimization, ROCm and vLLM for serving backend support, and InferenceX for reproducible benchmark integration.

## Acknowledgements

We would like to thank the AMD Quark team, the AMD ROCm and vLLM contributors, the InferenceX maintainers and reviewers, and the Eagle3 research community for their work and feedback. Special thanks to Andy Luo's team members including Chun Fang, Haichen Zhang, and Chang Liu for their contributions to the InferenceX Eagle3 benchmark integration.

## Additional Resources

- Eagle3 project: <https://github.com/SafeAILab/EAGLE>
- InferenceX Kimi-K2.5 Eagle3 GSM8K/eval quality reference: <https://github.com/SemiAnalysisAI/InferenceX/actions/runs/24980905592>

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, ROCm, Instinct, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
