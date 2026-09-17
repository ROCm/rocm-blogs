---
blogpost: true
blog_title: "Technical Dive into AMD MLPerf Inference v6.1 Submission"
date: "17 Sep 2026"
author: "Meena Arunachalam, Miro Hodak, Poovaiah Palangappa, Uma Kannikanti, Rita Brugarolas, Hemanth Acharya, Karan Verma, Yamini Preethi Kamisetty, Rebecca Lee, Neha Matthews, Rajesh Poornachandran, Karan Verma,  Jiawei Chen, Huasha Zhao, Mikko Lauri, Jesus Carabano Bravo, Nico Holmberg, Eliot Li"
thumbnail: 'mlperf_inf_v61_thumbnail.png'
tags: "AI/ML, GenAI, Performance, Optimization, LLM, MLPerf, MLPerf Inference"
category: "Applications & models"
target_audience: "AI developers, AI practitioners"
key_value_propositions: "Share the technical details of how we accomplish the results in our MLPerf Inference v6.1 submission"
language: English
myst:
    html_meta:
        "author": "Meena Arunachalam, Miro Hodak, Poovaiah Palangappa, Uma Kannikanti, Rita Brugarolas, Hemanth Acharya, Karan Verma, Yamini Preethi Kamisetty, Rebecca Lee, Neha Mathews, Rajesh Poornachandran, Jiawei Chen, Huasha Zhao, Mikko Lauri, Jesus Carabano Bravo, Nico Holmberg, Eliot Li"
        "description lang=en": "Learn about the ROCm optimizations powering dlrm-v3, llama2-70b, and gpt-oss-120b performance."
        "keywords": "MLPerf Inference v6.1, AMD Instinct MI355X, ROCm, llama2-70b, gpt-oss-120b, dlrm-v3, quantization, vLLM, AITER, MLCommons"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Meena Arunachalam, Miro Hodak, Uma Kannikanti, Poovaiah Palangappa, Yamini Preethi Kamisetty, Rebecca Lee, Neha Mathews, Rajesh Poornachandran, Karan Verma, Rita Brugarolas, Hemanth Acharya, Jiawei Chen, Huasha Zhao, Mikko Lauri, Jesus Carabano Bravo, Nico Holmberg, Eliot Li"
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

# Technical Dive into AMD MLPerf Inference v6.1 Submission

MLPerf Inference v6.1 [results](https://mlcommons.org/benchmarks/inference-datacenter/) were released on September 16, 2026. For AMD, this was a highly successful round in which we achieved several leadership scores, expanded benchmark coverage, introduced a new GPU with validated performance, and enabled record-setting results from our partners. In this round, AMD and its partners provided validated results across a diverse set of workloads, including **dlrm-v3** (Deep Learning Recommendation Model v3), **llama2-70b** (Meta's 70-billion-parameter language model), **gpt-oss-120b** (an open-weight 120-billion-parameter Mixture-of-Experts model), **deepseek-r1** (DeepseekAI's 671-billion-parameter Mixture-of-Experts model), and **Wan 2.2-t2v** (a 14-billion-parameter text-to-video generation model). Single-node results used 8 GPUs per node; gpt-oss-120b was additionally submitted at multi-node scale across 72 AMD Instinct<sup>™</sup> MI355X GPUs. All results were validated through MLPerf peer review.

This blog provides an overview of the optimizations used to achieve the results and walks readers through the results we achieved. The results are fully reproducible and readers can do their own measurements by following step-by-step instructions in our [reproduction blog](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inf_v6.1-repro/README.html).

## Key takeaways

- **Leadership performance:** AMD Instinct™ MI355X and MI350P GPUs delivered leading results across multiple benchmarks, establishing AMD Instinct as a top-tier platform for today's most demanding AI workloads.
- **A powerful new accelerator:** This round introduces the AMD Instinct™ MI350P, a PCIe-based GPU for enterprise inference that posted the top result among all PCIe-based submissions.
- **Broader benchmark coverage:** AMD expanded its footprint over prior rounds, spanning recommendation, multimodal, reasoning, and large-language-model inference - giving customers validated performance across a wider range of workloads than ever before.
- **Record-setting scale:** In collaboration with our partners, AMD powered the largest-scale submission in MLPerf Inference history, a 512-GPU configuration, showcasing how AMD Instinct GPUs scale in real-world deployments.
- **Record-breaking throughput:** The same collaboration delivered the highest total LLM throughput ever recorded in an MLPerf submission, underscoring the raw performance of AMD Instinct GPUs at scale.
- **A thriving partner ecosystem:** Seven partners submitted results across multiple AMD Instinct GPUs that closely tracked results by AMD - proving these results are reproducible from a single server to the largest clusters.
- **Fifth MLPerf appearance:** Fifth consecutive MLPerf Inference for AMD round reflects our sustained investment in transparent, standards-based benchmarking.

## Hardware: AMD Instinct&#8482; MI350P PCIe&#174; GPU

This round introduces the AMD Instinct&#8482; MI350P PCIe&#174; GPU, a new addition to the AMD Instinct portfolio built on the **CDNA<sup>™</sup> 4** architecture. PCIe-based and purpose-built for enterprise generative and agentic AI, its standard full-height, full-length dual-slot form factor drops into mainstream air-cooled servers — with no specialized cooling or rack redesigns — and supports up to eight cards per server. Each card offers 144 GB of HBM3E, 4 TB/s of memory bandwidth, and a configurable 450–600W TDP, and delivers a peak of 4600 TFLOPS at the MXFP4 precision format well-suited to modern AI inference[^1]. Because it shares the CDNA 4 architecture with the rest of the AMD Instinct MI350 series, the MI350P GPU readily benefits from the computational optimizations developed across those GPUs. Backed by the open-source AMD Enterprise AI stack, it brings datacenter-class AI performance into existing infrastructure and power envelopes with favorable total cost of ownership.

[^1]: [AMD Instinct MI3550P Product Brochure](https://www.amd.com/content/dam/amd/en/documents/epyc-business-docs/other/amd-instinct-mi350p-product-brochure.pdf)

```{figure} images/4711051-instinct-accelerator-mi350p-pcle-prod-front-delidded_crop.png
:alt: MI350P architectural diagram
:width: 80%
:align: center

Figure 1:  AMD Instinct MI350P PCIe card
```

## Hardware: AMD Instinct&#8482; MI355X GPU

To achieve the highest inference performance, we used the AMD Instinct&#8482; MI355X GPU. Designed specifically for AI workloads, high-performance computing (HPC), and large-scale cloud deployments, AMD Instinct MI355X GPU leverages AMD advanced CDNA4 architecture to deliver exceptional compute power and efficiency. The AMD Instinct MI350 Series GPU specifications are shown below.

```{figure} images/MI350-specification.png
:alt: MI355X architectural diagram
:width: 80%
:align: center

Figure 2: AMD Instinct MI350 Series specifications.
```

## Workloads

### Overview of MLPerf Inference v6.1 Benchmark Tasks

AMD participated in the **closed division, datacenter category** and submitted results for five models across multiple benchmark configurations. The closed division enforces a strict accuracy floor — 99% of the reference score for most workloads, and 99.9% for dlrm-v3 — and requires documented, reproducible system configurations. The table below lists each submitted workload:

| Model | Task | Dataset | Accuracy Target | Scenarios | Platform |
| --- | --- | --- | --- | --- | --- |
| Llama2-70B | LLM Chat | OpenOrca | 99% of FP32 reference | Offline, Server, Interactive | MI355X, MI350P |
| GPT-OSS 120B | MoE LLM Generation | AIME25, GPQA Diamond, LCb v6 | 99% of reference | Offline, Server | MI355X, MI350P |
| DLRM-v3 | Recommendation | MLCommons DLRMv3 synthetic streaming dataset | 99.9% of FP32 reference | Offline, Server | MI355X, MI350X, MI350P |
| Wan 2.2-t2v | Text-to-Video Generation | VBench prompts | 99% of reference | Offline, SingleStream | MI355X, MI350X, MI350P |
| Llama3.1-8b | Summarization | CNN-DailyMail | 99% of FP32 reference | Offline, Server, Interactive | MI350P |
| DeepSeek R1 | Reasoning | mlperf_deepseek_r1 | 99% of FP8 reference | Offline, Server | MI355X |

## Performance Optimizations

### GPT-OSS-120B Optimizations

#### MoE Kernel Tuning

MoE layers account for most of the compute in gpt-oss-120b. AMD optimized the [FlyDSL](https://github.com/rocm/flydsl) MoE a8w4 kernels — 8-bit activations against 4-bit weights — and tuned them for the token distributions each scenario actually produces, yielding an 8% end-to-end improvement in Offline and a 5% improvement in the Server scenario.

#### Attention Kernel Optimizations

AMD replaced the single unified attention kernel with separate kernels for the prefill and decode phases, improving attention performance by 25–28%. Prefill is a compute-bound problem over a contiguous query block, while decode is memory-bound over a paged KV cache, so one kernel serving both could only ever be a compromise for each. Prefills are now served by [AITER](https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html)'s variable-length flash attention, which handles causal masking, the sliding window, and attention sinks inside the kernel. Decode is served by a new paged-attention kernel written in Gluon.

Rewriting the decode kernel in [Gluon](https://triton-lang.org/main/gluon/index.html) gave explicit control over MFMA and shared-memory layouts and over instruction scheduling, which is what made the memory pipeline changes possible: 16-byte coalesced loads over a shuffled KV cache layout, two K buffers kept live so each block's load is issued underneath the previous block's P·V MFMA, and V streamed underneath the QK MFMA. With an FP8 KV cache, both matrix multiplications run in FP8 on the native 16x16x32 MFMA instructions of the CDNA 4 architecture. Sliding-window layers collapse to a single context partition with attention sinks folded inline into the softmax denominator, removing the reduction kernel and its scratch traffic.

#### vLLM params tuning & scheduler optimizations

To improve performance, tuning experiments were conducted on vLLM parameters and used to refine the configurations. AMD additionally developed a new scheduler policy that enforces the serving latency targets directly — a 3-second TTFT and an 80-millisecond TPOT — improving Server throughput by 8%.

In addition to these optimizations, AMD tuned GEMMs, fused operations around attention such as RoPE into the KV cache write, and extended HIP graph capture and kernel warmup to cover the decode batch sizes.

### Llama2-70b Optimizations

Llama2-70B inference is dominated by MXFP4 GEMMs. AMD tuned the assembly MXFP4 GEMM kernels and optimized vLLM scheduling config to admit queued requests based on the amount of work already in flight. In addition, AMD explored kernel-fusion opportunities, including SiLU–Quant fusion. This optimization introduces a fused kernel, `rocm_aiter_act_mul_and_mxfp4_group_quant`. The kernel computes SiLU and the gated multiply, then emits MXFP4 activations that the following GEMM consumes directly, removing a separate quantization pass. These optimizations improved throughput across both the Llama2-70B Offline and Server scenarios while continuing to meet the closed-division accuracy target.

Round-to-round improvements on 8× AMD Instinct MI355X GPU are 2.9% for Offline and 3% for the Server scenario.

### DeepSeek R1 Optimizations

DeepSeek-R1 is a 671-billion-parameter Mixture-of-Experts model combining Multi-head Latent Attention (MLA) with 256 routed experts and one always-on shared expert. AMD serves it with [SGLang](https://github.com/sgl-project/sglang) on 8 AMD Instinct MI355X GPUs — attention under data parallelism (DP8), MoE under expert parallelism (EP8) — using the AMD-published [DeepSeek-S3 MXFP4 checkpoint](https://huggingface.co/amd/Deepseek-S3_sq_a05_v2_mlperf6_1) (MXFP4 experts and attention output projection, BF16 for the remaining MLA projections, FP8 KV cache) with a [MoRI](https://github.com/ROCm/mori)-EP all-to-all. Unless noted, uplifts below are Offline output-token throughput.

#### Model Quantization from FP8 to MXFP4

The original weights for the DeepSeek-R1 model is in FP8 precision — this occupies about 671GB of HBM memory. We use [Quark](https://github.com/amd/Quark) to quantize the model from FP8 to mostly MXFP4 — we do maintain some of the attention projection GEMMs in BF16 for accuracy gains. First, serving the model in AMD Instinct MI355x GPU in its original FP8 precision limits the KV cache due to the high model footprint. However, by quantizing the model to MXFP4, we are able to use an additional ~300GB of KV cache. Second, AMD Instinct MI355x GPU has much higher MXFP4 TFLOPs in comparison to the FP8 TFLOPs. Thus, MXFP4 enables higher computational speeds, faster weight-fetch from the memory and larger batch sizes due to higher KV cache.

#### Shared-Expert Fusion and MoRI Dispatch/Combine (+15%)

MoRI-EP replaces SGLang's dense expert-parallel collectives with GPU-initiated dispatch/combine kernels that route each token point-to-point — over Infinity Fabric™ (xGMI) — only to the ranks serving the selected experts, carrying routing indices, weights, and quantization scales with the payload. Dispatch quantizes tokens to MXFP4 before transfer; combine stays in FP8 for accuracy.

Shared-expert fusion folds the always-on shared expert into the grouped MoE GEMM as one more local expert (32→33 experts, top-8→top-9), removing a separate dense FFN GEMM. The intra-node combine was extended from eight sources to nine, and AITER ships tuned two-stage MoE kernels for the 33-expert, top-9 shape.

#### MXFP4 Attention Projections (+6%)

The attention output projection (already MXFP4) runs through AITER's ASM a4w4 kernel, with weights pre-shuffled at load and activations quantized by the CDNA 4 `cvt_scalef32_pk_fp4` hardware instruction — bit-identical to the software path at 1.2–1.8× the speed. The BF16 query up-projection is dynamically quantized to MXFP4 and routed through the same kernel. Absorbed MLA projections stay at BF16 (quantizing them costs accuracy for little gain), which also lets the constant dequant scale fold into the weights at load.

#### GEMM and MoE Tuning (+5%)

Dense GEMM shapes are tuned offline via [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/) and AITER GradLib across backends, with winning selections loaded at startup. The shared-expert-fused MoE shape received its own sweep so it doesn't fall back to a generic CK kernel; AITER now ships tuned stage-1/stage-2 selections for the 33-expert, top-9 configuration.

#### Kernel Fusions (+4–5%)

Several fusions remove launch-bound elementwise kernels from the MoE and MLA paths, each bit-identical to the unfused version: routing post-processing collapsed into one Triton kernel; the MoE stage-1 SiLU + MXFP4 quantization folded into the GEMM epilogue; prefill K/V materialization fused via `fused_gemm_a16w16_split_cat`; the decode latent-cache write fused via `concat_and_cache_mla`; and transposed-view writes that turn downstream transpose/flatten into views instead of copies.

#### SGLang Scheduler and Serving Optimizations (+13%)

Under DP attention each rank schedules independently, but every decode step ends in a MoRI-EP all-to-all — so any imbalance leaves faster ranks idle at the barrier. With output lengths ranging to 20,000 tokens (avg. 3,886), three complementary controls, tuned in order, keep all eight ranks full and in phase:

1. **Attention-DP load balancing** steers each request to the rank with the lightest *token* load (not just request count), shrinking the straggler gap at every MoE all-to-all. Local control broadcast distributes routing decisions instead of funneling them through a single dispatcher.
2. **Scheduler conservativeness** is tuned to carefully balance the advantages of aggressive (high-batch size) prefill scheduling with the risks (request preemtions and OOM). The conservativeness is tuned such that the prefill rate and the request completion rate are aligned to produce a steady state of near constant batch sizes. Specifically, we tuned such that the steady-state decode batch size holds in the 800–1024/rank band rather than draining between admissions.
3. **Prefill delayer** (fixed receive-skip for Offline; a token-usage watermark for Server) plus chunked prefill (2048 tokens/rank) keep ranks in the same phase, so decoding ranks aren't stalled behind unbounded prefills.

In addition to these hyperparameters, we tuned several other SGLang hyperparameters to maximize the throughput and limit the latencies.

### Wan2.2-T2V-A14B Optimizations

Wan2.2-T2V-A14B is a 14‑billion‑parameter text‑to‑video generative model. This is a mixture-of-experts (MoE) model that consists of two experts that are activated sequentially during the denoising process. The first expert is known as the High Noise Expert and is active during the early stages of denoising. The model then switches to the Low Noise Expert to complete the denoising process.

A Closed category MLPerf Inference submission requires both Offline and Single Stream scores. Accuracy evaluation uses the VBench dataset and 99% of reference accuracy is required to satisfy accuracy constraints.

AMD Wan 2.2 submission uses [xDiT](https://github.com/xdit-project/xDiT) as the underlying inference engine to parallelize the video generation task over multiple GPU ranks. The key technical improvements made for MLPerf Inference v6.1 are already available in the latest ROCm and AITER releases and are part of [AMD Docker images optimized for diffusion model inference](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/xdit.html).

#### Quantized Attention

Up to 80% of compute time in Wan 2.2 is consumed by self-attention operators with a long context length. AMD submission uses a new high-performance attention kernel using MXFP4 quantization. To maintain output video quality, a hybridization strategy was applied where the last denoising steps used self-attention with higher-precision FP8 quantization. The quantized attention kernels used are available through the [MHAv4 entrypoint in AITER](https://github.com/ROCm/aiter/blob/eec768a47c1da52e175f9089c640e695f8b572a3/aiter/ops/mha_v4.md).

#### Quantized GEMMs and Communication

A significant amount of compute time can be saved by quantizing GEMMs from their default accuracy down to MXFP4 accuracy. In this submission, MXFP4 GEMMs delivered via AITER were used in the High Noise Expert, and FP8 GEMMs were used in the Low Noise Expert.

In the Single Stream scenario, all-to-all communication is used to share the query, key, and value tensors between ranks. The communication can be done in FP8 accuracy after quantizing the full-precision tensors, saving time required for communication. To retain quantization accuracy, a calibration process prior to inference records the maxima of activations across model layers. The maxima are used at runtime to select appropriate scaling factors for quantization.

#### Parallel VAE

The final step of video generation with diffusion models involves a variational autoencoder (VAE) step, where the denoised latent tensor is decoded into the final output video frames. In the Single Stream scenario this step can be parallelized across all the GPU ranks, further boosting performance. The latent tensor is split among ranks to distribute the computation load using the open source implementation from [DistVAE](https://github.com/xdit-project/DistVAE).

### DLRM v3 Optimizations

DLRM-v3 reframes recommendation as a sequence problem: each user history is a token sequence processed by stacked causal-attention **HSTU** layers, scoring 2,048 candidate items per request. It is expensive to serve because histories can run to thousands of tokens (attention cost grows with the square of sequence length) and the single `item_id` embedding table is roughly **1 TB** — far larger than a single GPU can hold. The Server scenario enforces an **80 ms P99** latency bound under Poisson arrival.

Because this is a closed-division submission, every optimization preserves the reference model's mathematics — no retraining, no architecture change, and full-causal attention preserved exactly. All of the gain comes from how efficiently that same computation compiles and runs on ROCm/CDNA 4, taking the submission from roughly 22 queries/s with the unoptimized reference to the submitted **12,199 queries/s at 60.43 ms P99**.

Key optimizations:

- **Triton kernel enablement:** Reworked the jagged concatenate/split kernels to use single-base-pointer, mask-based implementations so they compile for the `gfx950` target. This turned an unrunnable model into a running one, dropping the HSTU forward pass from ~1,170 ms to ~68.5 ms per call.

- **Terabyte-scale sharded embeddings:** Row-sharded the ~1 TB item table across eight GPUs with cooperative peer reads over AMD Infinity Fabric™ (xGMI). Narrowing over-broad peer-memory grants unwedged the node at full scale on stock ROCm 7.2.3, and parallelized, double-buffered checkpoint loading cut startup from 545 s to ~13 s with byte-identical weights.

- **Fused FP8 dataflow:** Fused FP8 quantization into surrounding kernel epilogues (LayerNorm, UVQK projection, attention, output), avoiding memory-bound standalone cast kernels and unlocking CDNA 4's 2× FP8 throughput. Per-batch time fell ~16% with accuracy at 99.9997% of the FP16 reference.

- **Full-causal attention tuning:** Split attention into interior/boundary regions to skip mask arithmetic on unmasked blocks, and widened autotuning over tile shapes, warp count, and occupancy. These bit-for-bit-verified choices made batch-64 full-causal runs feasible within the latency bound.

- **Last-layer target-only compute:** The final layer computes outputs only for candidate rows (still forming keys/values over the full history), eliminating provably-unused work. Final-layer attention dropped from 4,427 to 1,601 μs per call and P99 latency from 72.5 ms to 43.0 ms, enabling larger batches.

- **Polynomial attention gate:** Replaced the transcendental SiLU gate with a validated degree-5 polynomial approximation, letting the Server scenario clear its P99 target at the top of the throughput envelope while passing the accuracy audit.

- **Serving and host path:** Adopted a load-generator plus eight independent GPU worker processes, pinned each GPU before framework init, and raised the in-flight query cap (defaulting to one) to keep all eight GPUs busy. Combined with batch tuning, a larger on-GPU embedding cache, and removal of host-side allocation and serialization overhead, this unlocked an order of magnitude of throughput outside the attention kernel.

The submission passes the **99.9%-of-reference GAUC** accuracy bar and the **TEST08** compliance audit on both systems, using the same FP8 configuration for accuracy and performance measurements as MLPerf requires.

## Performance Results

The following table gives a summary of AMD MLPerf submissions in this round:

| Submission ID | Benchmark | Scenario | GPU | GPUs | Score | Units |
| --- | --- | --- | --- | --- | --- | --- |
| 6.1-0003 | dlrm-v3 | Server | MI355X | 8 | 12,198.80 | Queries/s |
| 6.1-0003 | dlrm-v3 | Offline | MI355X | 8 | 13,027.70 | Samples/s |
| 6.1-0003 | llama2-70b-99.9 | Server | MI355X | 8 | 103,275 | Tokens/s |
| 6.1-0003 | llama2-70b-99.9 | Offline | MI355X | 8 | 106,517 | Tokens/s |
| 6.1-0003 | llama2-70b-99.9 | Interactive | MI355X | 8 | 73,191.5 | Tokens/s |
| 6.1-0003 | text_to_video | SingleStream | MI355X | 8 | 16.124 | Latency (s) |
| 6.1-0003 | text_to_video | Offline | MI355X | 8 | 0.073 | Samples/s |
| 6.1-0003 | gpt-oss-120b | Server | MI355X | 8 | 113,241 | Tokens/s |
| 6.1-0003 | gpt-oss-120b | Offline | MI355X | 8 | 121,818 | Tokens/s |
| 6.1-0004 | gpt-oss-120b | Server | MI355X | 72 | 964,468 | Tokens/s |
| 6.1-0004 | gpt-oss-120b | Offline | MI355X | 72 | 1,039,900 | Tokens/s |
| 6.1-0109 | dlrm-v3 | Server | MI355X | 8 | 16,194.40 | Queries/s |
| 6.1-0109 | dlrm-v3 | Offline | MI355X | 8 | 16,363.00 | Samples/s |
| 6.1-0040[^2] | dlrm-v3 | Server | MI350P | 8 | 4,503.53 | Queries/s |
| 6.1-0040[^2] | dlrm-v3 | Offline | MI350P | 8 | 4,837.63 | Samples/s |
| 6.1-0040[^2] | llama2-70b-99.9 | Server | MI350P | 8 | 41,830.00 | Tokens/s |
| 6.1-0040[^2] | llama2-70b-99.9 | Offline | MI350P | 8 | 42,286.20 | Tokens/s |
| 6.1-0040[^2] | llama2-70b-99.9 | Interactive | MI350P | 8 | 22,860.50 | Tokens/s |
| 6.1-0040[^2] | llama3.1-8b | Server | MI350P | 8 | 72,305.10 | Tokens/s |
| 6.1-0040[^2] | llama3.1-8b | Offline | MI350P | 8 | 73,643 | Tokens/s |
| 6.1-0040[^2] | llama3.1-8b | Interactive | MI350P | 8 | 62,723.70 | Tokens/s |
| 6.1-0040 | text_to_video | SingleStream | MI350P | 8 | 41.838 | Latency (s) |
| 6.1-0040 | text_to_video | Offline | MI350P | 8 | 0.035 | Samples/s |
| 6.1-0040 | gpt-oss-120b | Server | MI350P | 8 | 40,867.90 | Tokens/s |
| 6.1-0040 | gpt-oss-120b | Offline | MI350P | 8 | 48,971.10 | Tokens/s |
| 6.1-0001 | dlrm-v3 | Server | MI350X | 8 | 9,297.78 | Queries/s |
| 6.1-0001 | dlrm-v3 | Offline | MI350X | 8 | 10,891.10 | Samples/s |
| 6.1-0002 | text_to_video | SingleStream | MI350X | 8 | 20.603 | Latency (s) |
| 6.1-0002 | text_to_video | Offline | MI350X | 8 | 0.057 | Samples/s |
| 6.1-0026[^3] | DeepSeek R1 | Server | MI355X | 512 | 2,405,310 | Tokens/s |
| 6.1-0026[^3] | DeepSeek R1 | Offline | MI355X | 512 | 2,901,950 | Tokens/s |
| 6.1-0027[^3] | gpt-oss-120b | Server | MI355X | 512 | 5,392,930 | Tokens/s |
| 6.1-0027[^3] | gpt-oss-120b | Offline | MI355X | 512 | 5,749,440 | Tokens/s |

[^2]: Joint Dell and AMD submission
[^3]: Crusoe submission based on AMD optimizations

### Generational Improvement on AMD Instinct MI355X GPU

AMD submitted llama2-70b, gpt-oss-120b, and WAN scores in both v6.0 and v6.1, enabling direct round-over-round comparison on identical workload definitions on the same hardware. On AMD Instinct MI355X GPU systems, the performance improvements are as follows:

| Model | Scenario | v6.0 Score (MI355X) | v6.0 ID | v6.1 Score (MI355X) | v6.1 ID | Improvement |
| --- | --- | --- | --- | --- | --- | --- |
| llama2-70b | Offline | 103,480.00 tokens/s | 6.0-0003 | 106,517 tokens/s | 6.1-0004 | 2.9% |
| llama2-70b | Server | 100,282.36 tokens/s | 6.0-0003 | 103,275 tokens/s | 6.1-0004 | 3% |
| gpt-oss-120b | Offline | 95,004.00 tokens/s | 6.0-0003 | 121,818 tokens/s | 6.1-0004 | 28% |
| gpt-oss-120b | Server | 82,136.10 tokens/s | 6.0-0003 | 113,241 tokens/s | 6.1-0004 | 38% |
| wan2.2-t2v | Single Stream | 27.38 s (Open) | 6.0-0102 | 16.124 s | 6.1-0004 | 70% |

### Distributed and Multi-Node Inference

AMD v6.1 submission includes **GPT-OSS 120B** results at cluster scale using **9 nodes × 8 AMD Instinct MI355X GPUs = 72 total GPUs**. Scale-out communication uses **RCCL** (ROCm Collective Communications Library) over AMD Pensando Pollara 400 AI NIC. Near-linear scaling was observed indicating that communication overhead is well managed by the ROCm network stack.

| Workload | Scenario | 8 GPUs (6.1-0003) | 72 GPUs (6.1-0004) | Ideal 72-GPU (8-GPU × 9) | Scaling Efficiency |
| --- | --- | --- | --- | --- | --- |
| GPT-OSS 120B | Offline | 121,818 | 1,039,900 | 1,096,362 | 94.9% |
| GPT-OSS 120B | Server | 113,241 | 964,468 | 1,019,169 | 94.6% |

### AMD Instinct MI350X GPU: Balance Between Performance and Power

The AMD Instinct MI350X GPU delivers highly efficient performance, striking a balance between throughput and power. With results on llama2-70b, gpt-oss-120b, WAN text-to-video, and DLRM benchmarks (submissions by AMD and MiTAC), it delivers about 80% of the performance of AMD Instinct MI355X GPU:

```{figure} images/Inference6.1_350x_1.png
:alt: 350X results
:width: 80%
:align: center

Figure 3: Performance of MI350X
```

## Competitive Performance: AMD Instinct MI355X GPU

In the following we will compare AMD results vs those on Nvidia B200 and B300 GPUs. When making comparisons, we use Nvidia submissions wherever available; if none are available, we use the best Nvidia partner submissions.

### WAN Text-to-Video

For the WAN benchmark, the only comparable data comes from B300. The comparison is shown below and shows AMD Instinct MI355X GPU leading Nvidia B300 by 18% and 11%, respectively, for Offline and SingleStream scenarios.

```{figure} images/Inference6.1_wan_1.png
:alt: WAN Comparison
:width: 80%
:align: center

Figure 4: Performance comparison for WAN Text-to-Video
```

### GPT-OSS-120B

In the GPT-OSS-120B test, AMD Instinct MI355X GPU is leading both Nvidia B200 and B300 across Offline and Server scenarios as shown in the figure below:

```{figure} images/Inference6.1_gptoss_1.png
:alt: GPT-OSS-120B Comparison
:width: 80%
:align: center

Figure 5: Performance comparison for GPT-OSS-120b benchmark
```

 As mentioned above, we have also submitted a scale-out submission for this benchmark on 72 GPUs, across 9 servers. The same size submission is also available from Nvidia on their GB200 NVL72 system. Because of highly efficient scaling, AMD maintains its lead and is ahead by 18% in Offline and 7% in Server scenarios. The comparison is displayed below:

```{figure} images/Inference6.1_gptoss_72gpus_1.png
:alt: GPT-OSS-120B Scale Out Comparison
:width: 80%
:align: center

Figure 6: Performance comparison for GPT-OSS-120b benchmark on 72 GPUs
```

### Llama2-70b

Next figure shows competitive performance in Llama2-70b benchmark. Here AMD Instinct MI355X GPU achieves leadership in Offline and Interactive categories against B200 and in Interactive vs B300. All other performance tests are tied (we define a tie as being within 3%).

```{figure} images/Inference6.1_llama2_1.png
:alt: Llama2-70b Comparison
:width: 80%
:align: center

Figure 7: Performance comparison for Llama2-70b benchmark
```

### DLRM v3

In this benchmark, AMD is the only submitter. To facilitate a comparison, we use the previous round, 6.0, in which Nvidia submitted DLRM v3 scores on 8xB200 GPUs. The comparison shows AMD Instinct MI355X GPU leadership by over 20%.

```{figure} images/Inference6.1_dlrmv3_1.png
:alt: DLRM v3 Comparison
:width: 80%
:align: center

Figure 8: Performance comparison for DLRM v3 benchmark
```

## Competitive Performance: AMD Instinct MI350P GPU

Our performance evaluation of AMD Instinct MI350P AMD Instinct focused on the following benchmarks: Llama2-70b, GPT-OSS-120B, DLRM v3, WAN Text-to-video, and Llama3.1-8b. Compared to RTX PRO 6000 Blackwell Server Edition, the AMD Instinct MI350P GPU achieves clear leadership in every test where a direct comparison exists. Most notable of these is the GPT-OSS-120B test, where AMD Instinct MI350P GPU leads the competition by a remarkable 211% and 176% as shown in the following figures:

```{figure} images/Inference6.1_gptoss_350p_1.png
:alt: GPT-OSS-120b Performance on MI350P
:width: 80%
:align: center

Figure 9: Performance comparison for GPT-OSS-120B on AMD Instinct MI350P GPU
```

```{figure} images/Inference6.1_llama3_350p_1.png
:alt: Llama3.1-8b Performance on Llama3.1-8b
:width: 80%
:align: center

Figure 10: Performance comparison for Llama3.1-8b on AMD Instinct MI350P GPU
```

```{figure} images/Inference6.1_llama2_350p_1.png
:alt: Llama2-70b Performance on Llama2-0b
:width: 80%
:align: center

Figure 11: Performance comparison for Llama2-70b on AMD Instinct MI350P GPU
```

## Partner Submissions

For this round of submissions, AMD collaborated with 7 partners: Crusoe, Dell, HPE, MangoBoost, MiTAC, Oracle, and Supermicro. These submissions cover all 4 AMD Instinct GPUs: MI355X, MI350X, MI300X, and MI350P. The results are within a few percent of AMD own scores. The collaboration demonstrates the strength of AMD ecosystem and gives customers confidence that results are reproducible across a range of Instinct-based platforms.  

Partner submissions are summarized in the table below

| Submission ID | Submitter | GPU | GPUs | Benchmarks |
| --- | --- | --- | --- | --- |
| 6.1-0037 | Dell | MI355X | 8 | llama3.1-8b, llama2-70b-99.9, deepseek-r1, gpt-oss-120b |
| 6.1-0040 | Dell + AMD | MI350P | 8 | dlrm-v3, llama3.1-8b, llama2-70b-99.9, deepseek-r1, text_to_video, gpt-oss-120b |
| 6.1-0041 | Dell + MangoBoost | MI300X + MI355X | 32 | gpt-oss-120b |
| 6.1-0042 | Dell + MangoBoost | MI355X | 8 | gpt-oss-120b |
| 6.1-0043 | Dell + MangoBoost | MI355X | 8 | gpt-oss-120b |
| 6.1-0026 | Crusoe | MI355X | 512 | deepseek-r1 |
| 6.1-0027 | Crusoe | MI355X | 512 | gpt-oss-120b |
| 6.1-0050 | HPE | MI355X | 8 | llama3.1-8b, llama2-70b-99.9, deepseek-r1, text_to_video, gpt-oss-120b |
| 6.1-0066 | MangoBoost | MI355X | 16 | gpt-oss-120b |
| 6.1-0067 | MangoBoost | MI300X | 8 | gpt-oss-120b |
| 6.1-0068 | MiTAC | MI350X | 8 | llama3.1-8b, llama2-70b-99.9, gpt-oss-120b |
| 6.1-0069 | MiTAC | MI355X | 8 | llama3.1-8b, llama2-70b-99.9, gpt-oss-120b |
| 6.1-0084 | Oracle | MI355X | 8 | llama3.1-8b, llama2-70b-99.9, deepseek-r1, text_to_video, gpt-oss-120b |
| 6.1-0099 | Supermicro | MI355X | 8 | llama3.1-8b, llama2-70b-99.9, deepseek-r1, gpt-oss-120b |

Of these submissions, 6.1-0026 and 6.1-0027 by Crusoe are particularly noticeable. They use 512 GPUs, which is the highest number ever used in MLPerf Inference submissions. The large scale also led to additional MLPerf records:

- **Largest number of tokens per second generated in MLPerf Inference:** GPT-OSS-120b Offline score of 5.75M/s tokens is the absolute highest token throughput ever achieved in an MLPerf submission.
- **Highest DeepSeek-R1 throughput:** The DeepSeek R1 model has been very popular for multi-node submissions achieving high token throughput. The current submission by Crusoe, first-ever DeepSeek R1 on AMD Instinct GPUs, tops them all with throughput of 2.90M and 2.41M for Offline and Server scenarios, respectively.

```{figure} images/Inference6.1_crusoe_1.png
:alt: Crusoe 512 GPU submission
:width: 80%
:align: center

Figure 12: Record-breaking partner submission by Crusoe
```

Additionally, a joint MangoBoost/Dell submission (6.1-0041) employs two different types of AMD Instinct GPUs: MI355X and MI300X that were geographically distributed on two different continents. With the AMD Instinct MI355X GPU servers located in the US and the MI300X ones in the USA and South Korea, they achieve a remarkably high scaling efficiency of 97% for Offline and 94% for Server in GPT-OSS-120B as shown below:

```{figure} images/Inference6.1_mangoboost_1.png
:alt: MangoBoost submission
:width: 80%
:align: center

Figure 13: Partner submission by MangoBoost and Dell utilizing servers on 2 continents
```

## Summary

MLPerf Inference v6.1 was a strong round for AMD. Across our own submissions and those of our partners, AMD Instinct™ GPUs delivered leadership performance across a diverse set of workloads — from recommendation (DLRM-v3) and large language models (Llama 2-70B) to Mixture-of-Experts reasoning (GPT-OSS-120B) and text-to-video generation (WAN 2.2) — and set multiple MLPerf records along the way. The results are fully reproducible by following step-by-step [instructions](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inf_v6.1-repro/README.html).

This round stands out on several fronts:

- **Leadership performance across the board.** AMD Instinct MI355X GPU delivered leading results across GPT-OSS-120B, WAN text-to-video, and DLRM-v3, and remained competitive on Llama 2-70B — establishing AMD Instinct as a top-tier platform for today's most demanding generative and reasoning workloads.

- **A powerful new accelerator.** AMD introduced the PCIe-based AMD Instinct **MI350P** GPU, which posted the top result among all PCIe-based submissions in the round — bringing datacenter-class AI performance into mainstream, air-cooled enterprise servers.

- **Record-setting scale.** In collaboration with Crusoe, AMD powered the largest-scale submission in MLPerf Inference history — **512 GPUs** — delivering the **highest total LLM throughput ever recorded** in an MLPerf submission at **5.75M tokens/s** on GPT-OSS-120B, along with the first-ever DeepSeek-R1 results on AMD Instinct GPUs at 2.90M tokens/s.

- **Generational gains from software alone.** On identical AMD Instinct MI355X GPU hardware, AMD posted double-digit round-over-round improvements — including sizeable jumps on GPT-OSS-120B and WAN — proving that the maturing ROCm™ software stack keeps unlocking more performance from the same silicon.

- **A thriving partner ecosystem.** Seven partners — Crusoe, Dell, HPE, MangoBoost, MiTAC, Oracle, and Supermicro — submitted results across multiple AMD Instinct GPUs that closely tracked our own numbers, proving these results are reproducible at scale. Partner innovation also delivered a multi-continent, geographically distributed submission with 97% scaling efficiency.

Together, these results showcase the strength of the complete AMD inference platform — cutting-edge silicon, the open-source ROCm software foundation, and a fast-growing partner ecosystem — and further establish AMD Instinct as a leadership choice for production AI inference from a single server to the largest clusters. We're just getting started, and we look forward to raising the bar again in future rounds.

## Additional Resources

- [MLPerf Inference v6.1 results](https://mlcommons.org/benchmarks/inference-datacenter/) - MLCommons results publication
- [Reproducing AMD MLPerf Inference v6.1 Submission Results](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inf_v6.1-repro/README.html) — step-by-step guide to run the v6.1 benchmarks yourself
- [AMD Instinct MI355X GPUs MLPerf Inference v6.0 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v6.0/README.html) — the previous round's results and optimization techniques
- [Reproducing AMD MLPerf Inference v6.0 Submission Results](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inf_v6.0-repro/README.html) — v6.0 reproduction guide
- [AMD Instinct MI325X GPUs MLPerf Inference v5.1 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v5.1/README.html) — v5.1 results on the prior hardware generation
- [MLCommons Results Visualizer](https://mlcommons.org/visualizer) — official interactive comparison of all v6.1 results across vendors
- [PLACEHOLDER: AMD.com v6.1 announcement blog — add URL when published]
- [ROCm Documentation](https://rocm.docs.amd.com) — ROCm software stack installation and API reference

## Disclaimers

[AMD Cautionary Statement](https://www.amd.com/en/legal/copyright.html)

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
©2026 Advanced Micro Devices, Inc. All rights reserved
