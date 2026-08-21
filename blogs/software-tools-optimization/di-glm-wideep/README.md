---
blogpost: true
blog_title: "DI Series: Scaling GLM-5.1-FP8 to 64 MI300X GPUs"
date: "21 Aug 2026"
author: "Ravi Gupta, Shiksha Patel, Chaitanya Lolla, Aswin Mathews, Onil Gunawardana, Vincent Cave, Mir Ali, Janet Tseng, Mohit Deopujari, Peng Sun, Emad Barsoum"
thumbnail: 'di-glm-thumbnail.png'
tags: "AI/ML, LLM, Performance, Serving, Optimization"
category: "Software tools & optimizations"
target_audience: "AI infra, ML, and performance engineers serving large MoE LLMs at long context on AMD Instinct"
key_value_propositions: "Enable and scale GLM-5.1-FP8 long-context prefill-decode disaggregated WideEP serving from EP8 to EP32 on MI300X with the AMD MoRI stack in vLLM"
language: English
myst:
    html_meta:
        "author": "Ravi Gupta, Shiksha Patel, Chaitanya Lolla, Aswin Mathews, Onil Gunawardana, Vincent Cave, Mir Ali, Janet Tseng, Mohit Deopujari, Peng Sun, Emad Barsoum"
        "description lang=en": "Scale GLM-5.1-FP8 long-context serving beyond a single node on AMD Instinct MI300X with WideEP prefill-decode disaggregation and MoRI."
        "keywords": "GLM-5.1, MoE, prefill-decode disaggregation, WideEP, expert parallelism, MoRI, MoRI-EP, MoRI-IO, vLLM, MI300X, DeepSeek Sparse Attention, long context"
        "vertical": "AI, Systems"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Inference, Deploying AI at Scale"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Ravi Gupta, Shiksha Patel, Chaitanya Lolla, Aswin Mathews, Onil Gunawardana, Vincent Cave, Mir Ali, Janet Tseng, Mohit Deopujari, Peng Sun, Emad Barsoum"
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

# DI Series: Scaling GLM-5.1-FP8 to 64 MI300X GPUs

Serving a frontier Mixture-of-Experts (MoE) model well is a systems problem, and it gets harder the moment one node is not enough. GLM-5.1 is a good example: it is a large, sparse MoE that users want to run at **long context**, and it ships a **new attention family** that breaks assumptions older serving stacks quietly relied on. Fitting it on eight GPUs is only the start. The real question is how to keep it correct and fast as you spread it across several nodes.

This post walks through how to bring up **GLM-5.1-FP8** for **prefill-decode (PD) disaggregated serving** with **wide expert parallelism (WideEP)** on AMD Instinct™ MI300X, using AMD's **MoRI** communication stack inside vLLM. Along the way, it covers two production-blocking defects that only appear once you disaggregate and scale out, and the fixes for both defects. The result is correct long-context serving and clean throughput scaling across four topologies, from a single prefill and decode pair (EP8) up to EP32 on both the prefill and decode all-to-all paths.

*This post is the first in the AMD Disaggregated Inference (DI) Blogs Series, which takes frontier open models from a single-node demo to production, multi-node serving on AMD Instinct™ GPUs, with more posts to follow.*

## At a Glance

The following bullets summarize the main results and fixes in this post:

- **GLM-5.1-FP8** (`GlmMoeDsaForCausalLM`, Multi-head Latent Attention plus **DeepSeek Sparse Attention**) is enabled for PD-disaggregated WideEP serving on MI300X, with **MoRI-EP** carrying the MoE all-to-all and **MoRI-IO** carrying the RDMA KV-cache transfer.
- Two production-blocking defects are fixed: a **long-context accuracy collapse** past ~30k tokens, and an **8k-prefill disaggregation crash** caused by a second, easy-to-miss KV cache that GLM-5.1's sparse attention introduces.
- **Long-context retrieval passes on every topology tested**: 1P/1D (EP8), 2P/2D (EP16), 1P/4D and 2P/4D (decode EP32), and 4P/4D (EP32 on both sides), across the full 2k–35k range, with DeepSeek-V3 non-regressed on the shared stack.
- A **complete 72-cell benchmark matrix** (4 topologies × 3 shapes × 6 concurrencies). Peak **16,525 tokens/s** (2P/4D, 28k input, concurrency 256).
- The defining finding is a **capacity wall you can predict from first principles**: at 28k input and concurrency 256, a single decode instance collapses to **1,405 tokens/s, 11.8× slower** than the multi-decode topologies, because its KV working set exceeds its pool. A closed-form KV-capacity model reproduces where the wall appears and turns "how many nodes do I need?" into arithmetic.

Figure 1 shows a representative 2P/2D (EP16) deployment that this post builds on.

```{figure} ./images/ep16-topology.png
:align: center
:alt: Detailed 2P2D EP16 topology: prefill head and child nodes and decode head and child nodes, each with 8 GPUs and 8 backend NICs, connected over an InfiniBand and RoCE fabric with MoRI-EP dispatch and combine and MoRI-IO KV transfer

Figure 1. A 2P/2D EP16 deployment. Each role (prefill, decode) spans two 8-GPU MI300X nodes that together form a 16-rank expert-parallel group: experts are sharded across GPUs (each `E` is an expert-shard slice), attention is replicated per rank, and every GPU owns a dedicated backend NIC. MoRI-EP carries the MoE dispatch (blue) and combine (orange) all-to-all across the InfiniBand and RoCE fabric, while MoRI-IO carries the RDMA KV-cache transfer between the prefill and decode roles. The rest of this post explains how each piece works and how it scales.
```

## Why Disaggregate, and Why WideEP

Inference has two phases with opposite personalities. **Prefill** reads the whole prompt in one compute-heavy pass. **Decode** then emits tokens one at a time, limited by memory bandwidth. Run both on the same GPUs and they interfere: a burst of long prompts blocks token generation for everyone (head-of-line blocking), and inter-token latency spikes under load.

**PD disaggregation** puts the two phases on separate instances, so each can be sized, parallelized, and scheduled on its own. Prefill-heavy traffic no longer stalls the steady decode stream.

That split raises a second question: how do you distribute a 256-expert MoE across all those GPUs? A tensor-parallel-only layout shards and synchronizes every layer, which is wasteful for a model where each token activates only a few experts. **WideEP** instead spreads the experts across the ranks. Each GPU holds a slice, and routes each token only to the ranks that own its experts. As you add nodes, the expert-parallel (EP) width grows, and the all-to-all that shuffles tokens between experts has to stay correct and fast across the network. That transport is exactly what MoRI provides.

> For the single-node parallelism background (TP, DP, and EP), DP attention, and when `--enable-expert-parallel` helps, see the [vLLM MoE Playbook](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html). This post picks up where that leaves off and scales across nodes.

## The Architecture: PD Disaggregation + WideEP on MoRI

Topologies use `xPyD` notation: `x` prefill instances and `y` decode instances, each instance an 8-GPU MI300X node. The EP width is `instances × 8`, so 1P/1D is EP8, 2P/2D is EP16, and adding decode instances (2P/4D) pushes the decode side to EP32.

The two schematics below summarize the deployment: the serving and request path (Figure 2, left), and the extra KV cache that GLM-5.1's sparse attention introduces (Figure 3, right), which the engineering section covers later.

````{grid} 2
:gutter: 3

```{figure} ./images/di-topology.png
:align: center
:alt: PD-disaggregated WideEP serving request flow on MoRI with router, prefill and decode instances

Figure 2. PD-disaggregated WideEP serving on MoRI (request flow). The router pins each request's prefill and decode legs to a stable DP rank; MoRI-EP carries the MoE all-to-all within each role; MoRI-IO moves the KV cache from prefill to decode over RDMA.
```

```{figure} ./images/di-dsa-kvcache.png
:align: center
:alt: GLM-5.1 DeepSeek Sparse Attention dual KV cache: the disaggregation defect and the pair-and-ship fix

Figure 3. GLM-5.1's DSA adds a second (indexer) KV cache per layer. Because it is written in a fused op, it never went through `save_kv_layer`, so MoRI-IO never shipped it and decode stalled into a crash. The fix pairs main and indexer caches by layer index and transfers all 156.
```
````

Two MoRI libraries carry the two traffic patterns:

- **MoRI-EP:** The MoE dispatch and combine all-to-all across EP ranks. vLLM selects a high-throughput kernel for prefill (large token batches) and a low-latency kernel for decode.
- **MoRI-IO:** Remote direct memory access (RDMA) GPU-Direct KV-cache transfer from prefill to decode. The traffic lands on the eight GPU-local backend NICs, evenly balanced, with the two front-end NICs carrying none. The KV path uses the fabric it should.

## The Model: GLM-5.1-FP8 = MLA + DeepSeek Sparse Attention

GLM-5.1-FP8 is a large sparse MoE: 78 layers (3 dense, 75 MoE), 256 routed experts (top-8) plus one shared expert, quantized to FP8 with block-128 scaling. Its architecture id, `GlmMoeDsaForCausalLM`, is the key detail: it combines **Multi-head Latent Attention (MLA)** with **DeepSeek Sparse Attention (DSA)**.

DSA is what makes GLM-5.1 interesting to serve, and what broke the existing stack. To decide which tokens each query attends to, DSA maintains a **second KV cache per layer**: a sparse "indexer" k-cache that DeepSeek-V3 does not have. That extra cache is invisible in the usual code paths, and it turned out to be the root of the disaggregation crash below.

## Two Engineering Problems to Solve

Bringing up a new attention family under disaggregation and WideEP is an engineering problem, not a config change. Two defects blocked production; both are now fixed.

### Problem 1: Long-Context Accuracy Collapses Past ~30k Tokens

**Symptom.** Generation was correct up to roughly 30k tokens, then produced incorrect output ([vLLM issue #47042](https://github.com/vllm-project/vllm/issues/47042)).

**Root cause.** The persistent, work-stealing sparse-MLA kernel keyed its attention metadata on a value that **collided under chunked prefill**. Two in-flight requests could map to the same key, so one request could reuse another's context. This is exactly the kind of bug that only shows up at long context under real batching.

**Fix.** [vLLM #47766](https://github.com/vllm-project/vllm/pull/47766) keys the metadata cache on a per-request `(context_len, query_len)` pair. This keeps the fast persistent kernel on and correct, as opposed to an earlier workaround (#47567) that disabled the kernel and gave up its performance. With #47766, the AITER persistent gqa64 fold path is exercised, so **stock ROCm and AITER work with no fork**.

### Problem 2: 8k-Prefill Disaggregation Crash

**Symptom.** Disaggregated runs with 8k+ prefills crashed the prefill EngineCore.

**Root cause.** This is DSA's second KV cache coming back to bite. The indexer cache is written *inside a fused op*, so the indexer module's `forward()` is effectively a no-op. It never calls `save_kv_layer`. The MoRI-IO connector transfers whatever gets registered through `save_kv_layer`, so those 78 indexer caches were **never shipped** to decode. Decode then waited for KV that never arrived, hit an `unmap MISS`, and after the 60-second deferred-write timer expired, the prefill EngineCore crashed.

**Fix.** Pair each main cache with its indexer cache by layer index and transfer **both** (78 main + 78 indexer = **156 caches**) together, plus the DP-rank KV-notify fixes carried in the vLLM branch. Disaggregated serving at 8k and beyond is then clean. Figure 3, in the architecture section above, shows this second cache and the before-and-after states.

## Results

Validation covers four topologies: 1P/1D (EP8, 2 nodes, 16 GPUs), 2P/2D (EP16, 4 nodes, 32 GPUs), 2P/4D (EP16 prefill, EP32 decode, 6 nodes, 48 GPUs), and 4P/4D (EP32 on both sides, 8 nodes, 64 GPUs), using needle-in-a-haystack (NIAH) retrieval for accuracy and `vllm bench serve` for throughput and latency. All runs used the same from-source image (see the System configuration section).

### Accuracy: Long-Context Retrieval Holds Everywhere

The following table lists NIAH scores (needles found out of 10) by context length for each topology:

| Topology | 2k | 8k | 16k | 20k | 28k | 35k |
| --- | --- | --- | --- | --- | --- | --- |
| 1P/1D (EP8) | 10 | 10 | 9 | 9 | 9 | 10 |
| 2P/2D (EP16) | 10 | 10 | 9 | 8 | 9 | 10 |
| 2P/4D (EP32 decode) | 10 | 10 | 9 | 9 | 8 | 10 |
| 4P/4D (EP32 both) | 10 | 10 | 8 | 9 | 7 | 10 |

*Table 1. NIAH retrieval (found/10), thinking disabled, greedy decoding.*

This is the headline correctness result. With the #47766 fix and the DSA indexer transfer in place, GLM-5.1 answers long-context needle queries correctly under disaggregated serving across the whole 2k–35k range, including the 28k–35k band that previously collapsed to near zero. The occasional 7–9/10 is single-needle variance, not a length collapse; the pre-fix signature is gone entirely.

### Throughput Scales Cleanly with Concurrency

```{figure} ./images/throughput.png
:align: center
:alt: Total token throughput versus concurrency for all four topologies at 1k/1k, 8k/1k, and 28k/1k input/output shapes

Figure 4. Total token throughput (tokens/s) versus concurrency, across three input and output shapes (1k and 1k, 8k and 1k, and 28k and 1k).
```

As Figure 4 shows, at 1k and 8k input, throughput scales **near-linearly** with concurrency. A c8→c128 step gains 13.5–15.4× (ideal is 16×) on every topology, so batching is efficient and disaggregation adds no scaling penalty. Longer inputs raise total token throughput because each request carries more prefill tokens; the peak of the whole matrix is **16,525 tokens/s** (2P/4D, 28k input, concurrency 256).

The interesting shape is the long one. The full 28k/1k table follows. Watch the last column:

| Topology | c8 | c16 | c32 | c64 | c128 | c256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1P/1D | 2,219 | 4,111 | 7,613 | 12,506 | 13,715 | **1,405** |
| 2P/2D | 2,170 | 4,043 | 7,414 | 13,166 | 15,353 | 15,429 |
| 2P/4D | 2,026 | 3,737 | 6,804 | 12,027 | 15,097 | **16,525** |
| 4P/4D | 2,023 | 3,745 | 6,676 | 11,912 | 14,470 | 15,337 |

*Table 2. Total token throughput (tokens/s) at 28k input. Every topology tracks together until the last cell, where 1P/1D drops sharply.*

### The Capacity Cliff Is the Defining Result

At 28k input and concurrency 256, a **single decode instance (1P/1D) drops to 1,405 tokens/s, 11.8× slower** than the best topology, while every multi-decode topology stays at full throughput:

| Topology | 28k × c256 | vs. best |
| --- | ---: | --- |
| **1P/1D** | **1,405 tokens/s** | **11.8× slower** |
| 2P/2D | 15,429 tokens/s | 1.1× |
| 4P/4D | 15,337 tokens/s | 1.1× |
| **2P/4D** | **16,525 tokens/s** | 1.0× (best) |

This is not a gentle regression. It is a regime change. Time-to-first-token (TTFT) tells the same story: 1P/1D's 28k × c256 TTFT is **382 s** (a deep preemption queue) versus 290–340 s on the larger topologies. Crucially, time-per-output-token (TPOT) does not rise. Per-token decode is still fast; sequences are being **preempted** because they no longer fit in memory. The next section shows this wall is exactly where a simple capacity model says it must be.

### Latency: Decode Stays Flat, TTFT Is the Tunable Knob

```{figure} ./images/ttft.png
:align: center
:alt: Mean time-to-first-token versus concurrency across topologies and input shapes

Figure 5. Mean time-to-first-token (TTFT) versus concurrency.
```

```{figure} ./images/tpot.png
:align: center
:alt: Mean time-per-output-token versus concurrency across topologies and input shapes

Figure 6. Mean time-per-output-token (TPOT) versus concurrency.
```

The key latency story is in **TTFT** (Figure 5): it rises with concurrency and input length, as expected. At 28k inputs it grows steeply as the prefill queue deepens. That is the natural knob operators trade against throughput by setting concurrency limits.

**TPOT** (steady-state per-output-token latency; Figure 6) is remarkably flat with concurrency: about **89 ms** for 1P/1D, **91 ms** for 2P/2D, and **105–120 ms** for 2P/4D. That flatness is the whole point of disaggregation: decode stays stable under load because prefill bursts land on separate instances and cannot stall token generation.

### Reading the Topologies

- **1P/1D and 2P/2D are the most GPU-efficient**. They track within a few percent per GPU at short and medium context. 2P/2D is the strongest balanced operating point until you push long context at high concurrency.
- **2P/4D and 4P/4D** (four decode instances) carry a +25–26% TPOT penalty at 1k/1k (about 89 ms to ~112 ms, the cost of the cross-node decode EP32 all-to-all in the token loop). That penalty buys the decode KV headroom that matters at long context: 2P/4D posts the matrix peak (**16,525 tokens/s** at 28k × c256) precisely where 1P/1D drops sharply.

### Correctness Scales Across Expert-Parallel Width

| Topology | Prefill EP | Decode EP | NIAH |
| --- | --- | --- | --- |
| 1P/1D | 8 | 8 | Pass |
| 2P/2D | 16 | 16 | Pass |
| 1P/4D | 8 | 32 | Pass |
| 2P/4D | 16 | 32 | Pass |
| 4P/4D | 32 | 32 | Pass |

*Table 3. NIAH correctness across expert-parallel widths.*

GLM-5.1-FP8 retrieves correctly across the entire expert-parallel range tested, EP8 through EP32, on **both** the prefill and decode all-to-all paths. This is a strong scalability signal: the MoRI-EP dispatch and combine and MoRI-IO KV transfer stay correct as expert parallelism widens. 1P/4D appears in this table for accuracy coverage only; it was validated for correctness with NIAH but was not part of the throughput sweep in Table 2.

## A Capacity Model That Predicts the Wall

The drop is not a mystery. It is arithmetic. vLLM reports the **measured** per-rank KV pool (657,000 tokens/decode rank), and GLM-5.1's per-token KV cost is fixed: MLA latent (576 B) plus the DSA indexer (128 B), across 78 layers, is **53.6 KiB/token** in FP8. So the longest context a topology can keep resident for all in-flight sequences can be expressed in closed form:

```text
max_context = 657,000 x 8 x (decode instances) / concurrency      (tokens, full pool)
            = 5.26M x (decode instances) / concurrency            (capped at the model's 202k limit)
```

Capacity is **linear in decode instances** and **inverse in concurrency**. That single formula reproduces every measurement (Figure 7):

```{figure} ./images/kv-envelope.png
:align: center
:alt: Long-context KV-capacity envelope: maximum resident context versus concurrency for each topology, with the model 202k ceiling and the measured operating points overlaid

Figure 7. The KV-capacity envelope. Maximum resident context (no preemption) versus concurrency for each topology, from the measured 657k-token/rank pool. The 28k benchmark line crosses 1P/1D's ceiling between c=128 and c=256, exactly where the measured drop appears.
```

| Topology | decode ranks | KV pool | max context @ c256 |
| --- | ---: | ---: | ---: |
| 1P/1D | 8 | 5.26M tok | **20k** |
| 2P/2D | 16 | 10.5M tok | 41k |
| 2P/4D / 4P/4D | 32 | 21.0M tok | 82k |

*Table 4. Decode-side KV pool and the longest context that keeps 256 concurrent sequences resident.*

The prediction matches the runs. At 28k input with concurrency 256: 1P/1D's ceiling is **20k < 28k, so it saturates** (measured 1,405 tokens/s, TTFT 382 s); 2P/2D's is **41k > 28k, so it completes** (15,429 tokens/s); the four-decode topologies sit at ~35% KV-pool utilization (16,525 and 15,337 tokens/s). The wall lands where the pool says it must.

Turned around, the model becomes a **sizing tool**. To serve the model's full 202k context at a target concurrency:

| Concurrency | Decode instances | GPUs |
| --- | ---: | ---: |
| 16 | 1 | 8 |
| 64 | 3 | 24 |
| 128 | 5 | 40 |
| 256 | 10 | 80 |

*Table 5. Decode instances (and GPUs) needed to keep full 202k context resident, from the measured per-rank pool.*

Sustaining full 202k context at concurrency 256 needs about **10 decode instances (~80 GPUs)**, a concrete multi-node target derived purely from a measured number, not a guess. This is the quantitative case for scaling out.

## Reproducibility

- **PR:** [ROCm/MAD #176](https://github.com/ROCm/MAD/pull/176), a per-model Dockerfile (`vllm_disagg_inference.glmv5.1`), a `models.yaml` recipe, and `MODEL_NAME`-gated DSA runtime patchers. The DeepSeek-V3 image and code path are untouched (byte-identical to `develop`).
- **Image:** from-source vLLM (WideEP WRITE-mode branch, #47766) + stock AITER `e03fa6040` + MoRI post-1.2.1 `42e895472b08` + vllm-router (DP-rank round-robin + KV-notify).
- **Serve flags:** `--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice --chat-template-content-format string`, block size 1, FP8 KV, AITER MLA on.
- **Harnesses:** `niah_nothink.py` (accuracy, thinking-aware, multi-seed via `NIAH_SEEDS`), `run_topo_bench.sh` (per-topology NIAH + perf), and `kv_envelope_sim.py` (the capacity model). Exact commits, nodes, and command lines are recorded per topology.

## Summary

In this post, you brought **GLM-5.1-FP8** to correct, high-throughput serving on AMD Instinct MI300X under WideEP PD-disaggregated serving across 1P/1D, 2P/2D, 2P/4D, and 4P/4D, backed by a complete 72-cell benchmark matrix. Getting there meant fixing two defects that only surface under disaggregation and scale: a long-context accuracy collapse (resolved by keeping the persistent sparse-MLA kernel on and correct via vLLM #47766) and an 8k-prefill crash (resolved by pairing and shipping DSA's second, indexer KV cache). You saw long-context retrieval hold from EP8 to EP32 on both the prefill and decode all-to-all paths, with decode latency staying flat under load, and you saw that the hard limit at 28k context and very high concurrency on a single decode instance is not a defect but a capacity boundary: it is predicted by the measured KV pool and cleared by adding decode instances. Follow the AMD Disaggregated Inference (DI) Series for upcoming posts that extend these techniques to more models, longer context, and wider topologies.

## Related Reading

- [AMD ROCm Blog, The vLLM MoE Playbook: TP, DP, PP and Expert Parallelism](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html), single-node parallelism-strategy background.
- [AMD ROCm Blog, Practical, Fault-Robust Distributed Inference for DeepSeek on AMD MI300X](https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html), wide-EP DeepSeek serving.
- [vLLM Blog, Why Your Single-Node vLLM Setup Needs Prefill-Decode Disaggregation](https://vllm.ai/blog/2026-04-07-moriio-kv-connector), MoRI-IO KV connector, single-node.
- [ROCm/mori](https://github.com/ROCm/mori), MoRI-EP / MoRI-IO / MoRI-CCL communication libraries.

## Additional Resources

- [vLLM PR #47766](https://github.com/vllm-project/vllm/pull/47766): persistent sparse-MLA metadata cache key fix.
- [vLLM issue #47042](https://github.com/vllm-project/vllm/issues/47042): GLM-5.1 long-context accuracy.
- [ROCm/MAD PR #176](https://github.com/ROCm/MAD/pull/176): GLM-5.1 WideEP disaggregated recipe.
- [MoRI framework](https://github.com/ROCm/mori): MoRI-EP, MoRI-IO, and MoRI-CCL communication libraries.

## System Configuration

### Hardware

- GPUs: 8× AMD Instinct™ MI300X (gfx942) per node.
- Fabric: RoCEv2; eight GPU-local backend NICs carry the RDMA KV traffic (verified evenly balanced).
- Node counts (8 GPUs each): 1P/1D = 2 nodes (16 GPUs), 2P/2D = 4 nodes (32 GPUs), 2P/4D = 6 nodes (48 GPUs), 4P/4D = 8 nodes (64 GPUs).

### Software

- Model: GLM-5.1-FP8 (`GlmMoeDsaForCausalLM`), 78 layers (3 dense + 75 MoE), 256 routed experts (top-8) + 1 shared, FP8 block-128.
- Serving: from-source vLLM (WideEP WRITE-mode, PR #47766) + AITER `e03fa6040` + MoRI `42e895472b08` + vllm-router; MoRI-EP all-to-all (`high_throughput` prefill / `low_latency` decode), MoRI-IO WRITE-mode KV transfer.
- Container image: built from the per-model recipe in [ROCm/MAD #176](https://github.com/ROCm/MAD/pull/176) (`vllm_disagg_inference.glmv5.1` Dockerfile).

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED "AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, ROCm, Instinct, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved.

Third-party content is licensed to you directly by the third party that owns it and is not licensed to you by AMD. Benchmark results are preliminary and measured on AMD Instinct™ MI300X (gfx942); actual performance may vary based on configuration, software versions, drivers, and optimizations.
