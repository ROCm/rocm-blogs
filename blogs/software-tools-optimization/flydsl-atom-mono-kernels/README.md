---
blogpost: true
blog_title: "FlyDSL Mono Kernels for High-Interactivity LLM Serving in ATOM"
date: 16 Oct 2026
author: "Felix Li, Lingpeng Jin, Hattie Wu, Carlus Huang, Ziming Huang, Jun Chen, Yifeng Huang, Bin Ding, Xiaobing Zhang, Yi Gan, Fangzhou Ai, Wei Sun, Li Li, Doug Lehr, Richard Li, Peng Sun, Emad Barsoum"
thumbnail: 'atom-high-interactivity-reference.png'
tags: AI/ML, Performance
category: Software tools & optimizations
target_audience: GPU kernel developers and LLM inference engineers
key_value_propositions: Explains how FlyDSL mono kernels reduce decode latency, with ATOM interactivity results for MiniMax-M3 and DeepSeek-V4.1-Flash and adaptation to vLLM and SGLang.
language: English
myst:
    html_meta:
        "author": "Felix Li, Lingpeng Jin, Hattie Wu, Carlus Huang, Ziming Huang, Jun Chen, Yifeng Huang, Bin Ding, Xiaobing Zhang, Yi Gan, Fangzhou Ai, Wei Sun, Li Li, Doug Lehr, Richard Li, Peng Sun, Emad Barsoum"
        "property=og:locale": "en_US"
        "description lang=en": "Explore FlyDSL mono kernels in ATOM, with MiniMax-M3 and DeepSeek-V4.1-Flash interactivity results and adaptation to vLLM and SGLang."
        "keywords": "FlyDSL, mono kernel, interactivity, AgentX, InferenceX, ATOM, vLLM, SGLang, MiniMax-M3, DeepSeek-V4.1-Flash, TileRT, ROCm, AMD Instinct MI355X, LLM inference, decode"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_blog_authors": "Felix Li, Lingpeng Jin, Hattie Wu, Carlus Huang, Ziming Huang, Jun Chen, Yifeng Huang, Bin Ding, Xiaobing Zhang, Yi Gan, Fangzhou Ai, Wei Sun, Li Li, Doug Lehr, Richard Li, Peng Sun, Emad Barsoum"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
---

# FlyDSL Mono Kernels for High-Interactivity LLM Serving in ATOM

**Extremely high interactivity is critical for LLM applications with strict latency requirements.** Fast token generation helps a coding agent produce its next action sooner and move on to the next step. The goal is to improve interactivity by reducing decode latency.

[FlyDSL](https://github.com/ROCm/FlyDSL) gives developers the flexibility to write and tune GPU code at the **grid, block, wave, thread, and instruction levels** in Python. They can control how work is assigned, how data is laid out and moved, and where synchronization is needed. These controls let them optimize individual operations and connect the stages of a model layer into **mono kernels** that overlap independent work and reduce waits. [ATOM][4] uses these capabilities to build mono decode across several model families. The approach also extends to [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang).

## Why mono kernels speed up decode

Small-batch decode runs many short kernels to process a few tokens at a time. Each launch adds overhead, and some kernels have too little parallel work to keep the GPU busy. When kernels depend on one another, each must finish before the next can begin.

Mono kernels combine related operations into fewer launches and let thread blocks work across stages. Developers can load weights alongside earlier computation and let the next stage process portions of the output as they become ready. They can also overlap computation with reductions across GPUs. Together, these choices reduce idle time and help the decode step finish sooner. [3], [11]

HIP Graph replay reduces CPU submission overhead, but the GPU still executes dependent kernels in sequence. Mono also changes how stages share data and overlap work within a launch.

## What FlyDSL provides

FlyDSL gives developers four kinds of control over a GPU program: [2]

- **Work assignment.** Assign tasks to thread blocks and threads, and tune the mapping for different numbers of tokens per step.
- **Layout and tiling.** Choose how data is arranged and divided into tiles for each stage.
- **Data movement and synchronization.** Decide when to load data and how to signal that an output is ready, within a GPU or across GPUs.
- **Low-level control.** Write and tune a wide range of operations at the grid, block, wave, and thread levels, with control down to individual GPU instructions. [2], [7]

These controls connect local kernel tuning with scheduling across a layer. Developers can coordinate blocks across a grid, tune work within a block, and control how threads cooperate within a wave. They can also decide how blocks hand data to later stages. This lets them optimize both the operations themselves and the way those operations work together. [2], [7]

Mono kernels use these controls to keep **thread blocks** active across stages, passing ready results to consumers and taking on new tasks as the layer advances. [3], [7]

![Staged launches versus a resident mono grid with early loads and tile-level handoffs](images/mono-overlap.svg)

*Figure 1. Mono loads weights early, passes ready portions of the output forward, and overlaps reductions with computation. The timeline is schematic.*

**Developers design the fusion.** They manage scheduling and synchronization, including keeping the required thread blocks resident so the kernel can make progress. ATOM handles runtime buffers, coordination across tensor-parallel (TP) ranks, graph replay, correctness checks, and fallback. [3], [8]

The design is inspired by **TileRT**. [5], [6]

## Bringing mono kernels to ATOM

ATOM supports FlyDSL mono decode for **GLM-5.2, Kimi-K3, DeepSeek-V4.1, and MiniMax-M3**. Its [shared mono components](https://github.com/ROCm/ATOM/tree/53cbd6415a30e62421e76d67650e97e0090bbd95/atom/mono) let developers adapt scheduling and data handoffs to each model. Kimi-K3 uses mono for its linear-attention (KDA) layers; its MLA layers keep the existing path. [1], [8], [13]

The shared components handle recurring needs such as buffers, synchronization, and reductions. Each model supplies the schedule that connects its own stages. ATOM then connects that schedule to serving: batching and speculative decoding can change the number of tokens in a step, so runtime dispatch must select a supported path and keep participating GPUs in agreement. [8], [13]

For MiniMax-M3, the mono kernel combines attention, mixture-of-experts (MoE) computation, and their supporting operations into **one launch per sparse layer and TP rank**. The first three dense layers keep the separate path. [1], [3]

![MiniMax-M3 sparse-layer stages inside one FlyDSL launch](images/minimax-mono-flow.svg)

*Figure 2. M3's mono kernel combines the sparse-layer stages in one launch. Arrows show data dependencies.*

Three scheduling choices reduce waits:

- **Reuse thread blocks across stages.** A block can finish attention work and then take on routing or expert tasks. Its role changes as the layer progresses, allowing the schedule to use the same GPU resources for different operations. [3], [9]
- **Load weights early.** A later stage's weights can load while earlier work is still running. The later calculation still waits for its input, but its weight loads can already be under way. [3], [9]
- **Pass on ready data.** Tagged buffers identify outputs from the current decode step. Consumers and cross-GPU reductions can begin on ready portions, while other blocks are still producing the rest. Progress depends on the data each task needs. [3], [7]

### Decode latency in ATOM

These measurements use MI355X, TP4, MiniMax-M3-MXFP4, a 1M-token context, and EAGLE3 with three speculative tokens. **Separate** and **Mono** compare complete ATOM execution paths. **Mono + CP** adds indexer context parallelism, a separate long-context optimization described in the [supporting measurements](#appendix-supporting-measurements). [1]

Times are median GPU decode-step latency in milliseconds; parentheses show speedup relative to Separate.

| Batch | Separate | Mono | Mono + CP |
| ---: | ---: | ---: | ---: |
| 1 | 11.47 | 7.00 (**1.64×**) | 6.26 (**1.83×**) |
| 2 | 13.40 | 9.50 (**1.41×**) | 7.77 (**1.72×**) |
| 4 | 18.47 | 15.88 (**1.16×**) | 12.36 (**1.49×**) |

![M3 GPU decode-step latency: ATOM separate kernels 11.47 ms, FlyDSL mono 7.00 ms, FlyDSL mono plus indexer CP 6.26 ms](images/minimax-flydsl-decode-step.png)

*Figure 3. At batch one, FlyDSL mono shortens M3's GPU decode step in ATOM, and indexer CP adds a further gain at long context. [1]*

**Mono alone delivers 1.64× speedup at batch one; the 1.83× result includes CP.** Gains are smaller at larger batches. These GPU-step measurements do not directly measure per-user token rate or isolate the contribution of each scheduling choice.

A larger batch gives separate kernels more work to distribute across the GPU and more tokens over which to spread launch overhead. That can reduce the relative benefit of mono, although the balance depends on the model and schedule.

### Interactivity results in ATOM

The [InferenceX AgentX](https://inferencex.semianalysis.com/agentx) benchmark replays multi-turn, long-context agent sessions. Its **P90 interactivity is 1 / P90 inter-token latency**, with latency in seconds. Higher values mean faster token generation at the slower end of the request distribution. [14], [15]

The published ATOM results below use **MI355X** with FlyDSL mono decode enabled. The table shows concurrency one (C1); the [supporting measurements](#appendix-supporting-measurements) include the available low-concurrency curves. [16], [17]

- **MiniMax-M3-MXFP4, TP4:** EAGLE3 with three draft tokens and synthetic acceptance 2.78, FP8 KV and index caches. [18], [21]
- **DeepSeek-V4.1-Flash, TP2 and TP4:** DSpark with five draft tokens and synthetic acceptance 3.51, BF16 KV and an FP4 index cache. [19], [21]

Each point uses a **3,600-second profiling window**. These controlled replays do not establish natural acceptance or task accuracy. Both sets use GPU-resident KV caches. Exact recipes, images, acceptance settings, and source records are linked in the [ATOM serving data](data/atom-agentx.csv). [18], [19], [21]

The table reports P90 interactivity in tokens/s/user and P90 time to first token (TTFT) in seconds, at C1.

| Model / TP | Interactivity | TTFT |
| --- | ---: | ---: |
| M3 / TP4 | **469.5** | 1.13 |
| V4.1 / TP2 | **505.1** | 1.20 |
| V4.1 / TP4 | **641.0** | 1.20 |

These measurements show the complete ATOM serving configurations, including scheduling, caching, and speculative decoding. **They do not include a matched mono-off baseline**, so they cannot isolate mono's contribution. The M3 GPU-step comparison above provides that separate evidence. TTFT and tool waits also affect responsiveness; interactivity describes the pace of token generation once it starts.

## Extending FlyDSL mono decode to vLLM

ATOM already provides **FlyDSL mono decode for DeepSeek-V4.1**. The vLLM implementation takes inspiration from that work and adapts the kernels to vLLM's KV-cache layout and runtime data structures, supporting DeepSeek-V4.1-Flash on MI355X. [8], [11], [12]

FlyDSL makes this adaptation practical through explicit layout, data movement, and synchronization controls. vLLM supplies the corresponding cache views, metadata, and buffers, and manages execution-path selection, coordination across GPUs, and graph replay. [11]

The schedule uses **two FlyDSL mono launches, K1 and K2, per eligible layer and TP rank**. K1 prepares attention inputs; K2 handles attention, expert computation, and reductions. Some operations, such as compression and token selection, remain outside these launches. [11], [12]

![DeepSeek-V4.1-Flash in vLLM: Engram before K1 where present, compressor and indexer between K1 and K2 on index-source layers, attention and experts inside K2](images/deepseek-v41-mono-flow.svg)

*Figure 4. FlyDSL mono decode in vLLM. Conditional operations outside K1 and K2 depend on the layer. [12]*

In vLLM's same-node comparison at **TP2, concurrency one**, enabling K1 + K2 on 30 standard layers raises P90 interactivity from **318.1 to 429.1 tokens/s/user (1.35×)** and input + output throughput from **21,573 to 25,039 tokens/s (1.16×)**. TTFT p90 changes from **1.18 to 1.28 seconds**. Each run uses five-token DSpark with synthetic acceptance 3.51 for 3,600 seconds; this controlled replay measures performance, while task quality is evaluated separately. [11]

![DeepSeek-V4.1-Flash in vLLM on MI355X TP2: mono-off/on arrows show higher P90 interactivity and throughput at concurrency 1, 2, 4, and 8](images/vllm-dsv41-throughput-interactivity.png)

*Figure 5. Each arrow connects mono off to mono on at the same request concurrency. Moving right means higher interactivity; moving up means higher throughput. Throughput is per server and includes cached input. These four measured pairs do not establish the full curve. [11]*

## Extending FlyDSL mono decode to SGLang

SGLang's **AMD mono kernel backend** also supports FlyDSL mono decode for **GLM-5.3 and GLM-5.3-Flash** on MI355X. The implementations in `sgl-amd-mono` adapt FlyDSL kernels from AITER and vLLM to SGLang's weight layouts, KV caches, and state buffers. For supported small-batch decode steps, each covered MoE layer runs as one launch per TP rank. SGLang handles request scheduling, cache management, and sampling, while other steps use its standard execution path. [22], [23]

## Scope and costs

A fused kernel keeps more work and state in one program. Register and shared-memory use must leave enough thread blocks able to run at once. That resource budget is part of schedule design and can limit how much work to combine. [3], [7]

- **Workload fit.** Short, dependent kernels at small batch sizes offer opportunities to reduce waits. Benefits depend on the model, context length, and batch size; profile the workload you intend to serve.
- **Supported configurations.** Each schedule targets particular GPU architectures, TP sizes, and decode shapes. The engine must check eligibility and provide a fallback for other steps. [1], [11], [12]
- **Development cost.** Developers tailor and validate schedules for each model and keep them in sync with model changes. Correctness checks and task evaluations are part of that work. [11]

## Build a mono path for your model

Start with the [FlyDSL mono components](https://github.com/ROCm/FlyDSL/tree/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel) and [ATOM's shared mono framework](https://github.com/ROCm/ATOM/tree/53cbd6415a30e62421e76d67650e97e0090bbd95/atom/mono):

1. **Find the waits.** Profile the separate path at your serving batch sizes. Identify where stages wait for data or communication.
2. **Write the schedule in FlyDSL.** Use reusable compute and communication components to connect stages and give thread blocks independent work to overlap.
3. **Integrate and evaluate.** Add runtime dispatch, fallback, and correctness checks. Evaluate task quality with actual rejection sampling, then compare interactivity, TTFT, and throughput at each concurrency level.

## Appendix: Supporting measurements

**Indexer context parallelism in M3.** ATOM splits index scoring across four TP ranks, then exchanges candidates to recover the final selection. This is the additional optimization behind the Mono + CP results. [1], [3]

At 1M context, TP4, 16 token rows, and four query rows per request, the [layer measurement](data/minimax-indexer-cp-layer.csv) records **222.2 → 156.7 µs (1.42×)** with CP. Both runs use mono and graph replay. [1]

![MiniMax-M3 indexer context parallelism across four TP ranks](images/minimax-indexer-cp.svg)

*Figure 6. Each rank scores part of the cache, then exchanges its best candidates so the owning rank can make the final selection.*

**ATOM serving curves.** Figure 7 adds the low-concurrency points for the ATOM configurations above. M3's C1, C2, and C4 settings fit its 16-token-row mono limit. V4.1's plotted settings reach C8, within the measured revision's limit of 48 token rows. [18], [19], [20]

![ATOM AgentX results on MI355X with mono enabled: MiniMax-M3 TP4 and DeepSeek-V4.1-Flash TP2/TP4 across selected concurrency settings; no mono-off baseline](images/atom-throughput-interactivity.png)

*Figure 7. ATOM throughput and interactivity with mono enabled. Lines connect different concurrency settings within one model and TP configuration; they do not show mono-off/on gains. Throughput is input + output tokens/s/GPU, including cached input. [16], [17]*

M3 delivers **469.5 tokens/s/user at C1** and **374.5 at C4**, while throughput rises from **6,580 to 9,201 tokens/s/GPU**. DeepSeek-V4.1-Flash at TP2 delivers **505.1 tokens/s/user at C1** and **315.5 at C8**, with throughput rising from **13,915 to 32,005 tokens/s/GPU**. Use both axes to choose a configuration that meets your application's responsiveness needs. [16], [17]

**Additional vLLM measurements.** The [comparison CSV](data/mono-benefits.csv) records the TP2 mono-off/on results at C1, C2, C4, and C8, including TTFT; the [Figure 5 script](data/plot-vllm-agentx.py) reads those values directly. [11] The later **541.6 → 562.6 P90 tokens/s/user (+3.9%)** result at TP4/C1 compares two mono versions as coverage expands to all 40 backbone layers. Both use synthetic DSpark acceptance 3.51 and one 3,600-second run per version. [12]

**Validation and additional models.** The M3 and vLLM sources include correctness checks and task evaluations; vLLM evaluates quality separately with actual rejection sampling. [1], [11], [12] The [comparison CSV](data/mono-benefits.csv) also includes historical GLM-5 and Kimi-K3 layer tests, separate from ATOM serving results. [5] The cited Kimi-K3 serving results lack task-level accuracy and have a [confirmed DSpark auxiliary-state capture issue](https://github.com/ROCm/ATOM/pull/2535#discussion_r4266162046), so they are excluded here. [13]

**Data and reproduction.** The [M3 decode-step CSV](data/minimax-decode-step.csv) and other linked CSVs preserve reported values and configurations. The [Figure 3 script](data/plot-minimax-decode-step.py) plots the M3 latency data, and the [Figure 7 script](data/plot-atom-agentx.py) plots the ATOM AgentX data from the [archived source records](data/atom-agentx-source.json). The [M3 launch example](data/run-minimax-m3-mono.sh) adapts the source recipe for four MI355X GPUs; it starts a server and does not run the measurements. [10] All measurements in this article come from the cited sources.

## References

1. [ATOM: MiniMax-M3 mono decode and measurements](https://github.com/ROCm/ATOM/pull/2419).
2. [FlyDSL language and compiler](https://github.com/ROCm/FlyDSL).
3. [ATOM's M3 mono implementation](https://github.com/ROCm/ATOM/tree/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono).
4. [ATOM inference engine](https://github.com/ROCm/ATOM).
5. [FlyDSL: GLM-5 and Kimi-K3 mono work](https://github.com/ROCm/FlyDSL/pull/1204).
6. [TileRT source cited by FlyDSL](https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9).
7. [M3 device primitives and tagged mailboxes](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/common.py).
8. [ATOM: DeepSeek-V4.1 and shared mono components](https://github.com/ROCm/ATOM/pull/2479).
9. [M3 task scheduling](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/layout.py).
10. [MiniMax-M3 serving recipe](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/recipes/MiniMax-M3.md).
11. [vLLM: DeepSeek-V4.1-Flash mono decode and AgentX results](https://github.com/vllm-project/vllm/pull/60397).
12. [vLLM: all-layer mono coverage and evaluations](https://github.com/vllm-project/vllm/pull/61082).
13. [ATOM: GLM-5.2 and Kimi-K3 mono decode](https://github.com/ROCm/ATOM/pull/2535).
14. [InferenceX AgentX dashboard](https://inferencex.semianalysis.com/agentx).
15. [InferenceX AgentX metric definitions](https://github.com/SemiAnalysisAI/InferenceX/blob/8b25006fa0b23c7e157438996a3d2a90ff5e8cd2/inferencex-e2e/infx/results/agentic/request_metrics.py).
16. [InferenceX: ATOM MiniMax-M3 serving records](https://inferencex.semianalysis.com/api/v1/benchmarks?model=MiniMax-M3).
17. [InferenceX: ATOM DeepSeek-V4.1-Flash serving records](https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-V4.1-Flash).
18. [MiniMax-M3 AgentX recipe at the measured revision](https://github.com/SemiAnalysisAI/InferenceX/blob/4b94f096b74191b8af6b7c89e50e0b6bc69f7aba/inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/minimaxm3/atom/mi355x-fp4-mtp/agentic.yaml).
19. [DeepSeek-V4.1-Flash AgentX recipe at the measured revision](https://github.com/SemiAnalysisAI/InferenceX/blob/2597d3d1ce010ee10b82e08bd5b448fea054b5ac/inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/dsv41flash/atom/mi355x-fp4-mtp/agentic.yaml).
20. [ATOM: DeepSeek-V4.1 mono scheduling and replay updates](https://github.com/ROCm/ATOM/pull/2545).
21. [InferenceX synthetic acceptance for AgentX performance runs](https://github.com/SemiAnalysisAI/InferenceX/blob/4b94f096b74191b8af6b7c89e50e0b6bc69f7aba/inferencex-e2e/infx/srt_slurm/synthetic_acceptance.py).
22. [SGLang AMD mono: GLM-5.3 decode kernels](https://github.com/sgl-project/sgl-amd-mono/pull/7).
23. [SGLang AMD mono: GLM-5.3-Flash decode kernels](https://github.com/sgl-project/sgl-amd-mono/pull/6).

[1]: https://github.com/ROCm/ATOM/pull/2419
[2]: https://github.com/ROCm/FlyDSL
[3]: https://github.com/ROCm/ATOM/tree/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono
[4]: https://github.com/ROCm/ATOM
[5]: https://github.com/ROCm/FlyDSL/pull/1204
[6]: https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9
[7]: https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/common.py
[8]: https://github.com/ROCm/ATOM/pull/2479
[9]: https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/layout.py
[10]: https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/recipes/MiniMax-M3.md
[11]: https://github.com/vllm-project/vllm/pull/60397
[12]: https://github.com/vllm-project/vllm/pull/61082
[13]: https://github.com/ROCm/ATOM/pull/2535
[14]: https://inferencex.semianalysis.com/agentx
[15]: https://github.com/SemiAnalysisAI/InferenceX/blob/8b25006fa0b23c7e157438996a3d2a90ff5e8cd2/inferencex-e2e/infx/results/agentic/request_metrics.py
[16]: https://inferencex.semianalysis.com/api/v1/benchmarks?model=MiniMax-M3
[17]: https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-V4.1-Flash
[18]: https://github.com/SemiAnalysisAI/InferenceX/blob/4b94f096b74191b8af6b7c89e50e0b6bc69f7aba/inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/minimaxm3/atom/mi355x-fp4-mtp/agentic.yaml
[19]: https://github.com/SemiAnalysisAI/InferenceX/blob/2597d3d1ce010ee10b82e08bd5b448fea054b5ac/inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/dsv41flash/atom/mi355x-fp4-mtp/agentic.yaml
[20]: https://github.com/ROCm/ATOM/pull/2545
[21]: https://github.com/SemiAnalysisAI/InferenceX/blob/4b94f096b74191b8af6b7c89e50e0b6bc69f7aba/inferencex-e2e/infx/srt_slurm/synthetic_acceptance.py
[22]: https://github.com/sgl-project/sgl-amd-mono/pull/7
[23]: https://github.com/sgl-project/sgl-amd-mono/pull/6
