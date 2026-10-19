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
        "keywords": "FlyDSL mono kernel, interactivity, AgentX, InferenceX, ATOM, vLLM, SGLang, MiniMax-M3, DeepSeek-V4.1-Flash, TileRT, ROCm, AMD Instinct MI355X, LLM inference, decode"
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

**Extremely high interactivity is critical for LLM applications with strict latency requirements.** A coding agent needs to generate its next action quickly so it can call a tool and continue the task. Reducing decode latency helps keep this loop moving.

[FlyDSL][1] gives developers the flexibility to write GPU programs in Python and tune them at the **grid, block, wave, thread, and instruction levels**. Its **just-in-time (JIT) compilation** builds specialized kernels on demand, without compiling every variant in advance. Developers control work assignment, data layout, movement, and synchronization to combine stages of a model layer into **FlyDSL mono kernels** that overlap independent work and reduce waits. [ATOM](https://github.com/ROCm/ATOM) brings these kernels into serving across several model families. The approach also extends to [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang).

## Why FlyDSL mono kernels speed up decode

Small-batch decode runs many short kernels to process a few tokens at a time. Each launch adds overhead, and some kernels have too little parallel work to keep the GPU busy. A GPU stream executes these kernels in order, so the next stage waits for the whole kernel to finish even if part of its input is already available.

FlyDSL mono kernels combine related operations into fewer launches and let thread blocks work across stages. Developers can load weights alongside earlier computation and let the next stage process portions of the output as they become ready. They can also overlap computation with reductions across GPUs. Together, these choices reduce idle time and help the decode step finish sooner. [2], [3]

HIP Graph replay reduces CPU submission overhead, but the GPU still executes dependent kernels in sequence. FlyDSL mono also changes how stages share data and overlap work within a launch.

## What FlyDSL provides

FlyDSL provides the controls and compilation support needed to build a FlyDSL mono schedule: [1]

- **JIT specialization.** Compile specialized kernels on demand and reuse cached versions. Developers can adjust shapes and schedules without rebuilding a large library of precompiled kernels, making the kernels easier to tune and extend to new models.
- **Work assignment.** Assign tasks to thread blocks and threads, and tune the mapping for different numbers of tokens per step.
- **Layout and tiling.** Choose how data is arranged and divided into tiles for each stage.
- **Data movement and synchronization.** Decide when to load data and how to signal that an output is ready, within a GPU or across GPUs.
- **Low-level control.** Write and tune a wide range of operations at the grid, block, wave, and thread levels, with control down to individual GPU instructions. [1], [2]

Together, these controls let developers tune both the work inside each stage and the handoffs between stages. In a FlyDSL mono kernel, **thread blocks** can stay active as the layer advances: a block passes on a ready result, then takes on another task. [1], [2]

![Work inside a FlyDSL mono kernel: prepare weights early, pass ready output to reductions, and feed the next computation](images/mono-overlap.svg)

*Figure 1. FlyDSL mono loads weights early, passes ready portions of the output forward, and overlaps reductions with computation. The timeline is schematic.*

**Developers design the fusion.** They choose the schedule and synchronization scheme, and must keep enough cooperating thread blocks resident on the GPU for the kernel to make progress. ATOM provides the serving integration: runtime buffers, coordination across tensor-parallel (TP) ranks, graph replay, correctness checks, and fallback. [2], [4]

The design is inspired by **TileRT**. [5]

## Bringing FlyDSL mono kernels to ATOM

ATOM supports FlyDSL mono decode for **GLM-5.2, Kimi-K3, DeepSeek-V4.1, and MiniMax-M3**. Its [shared FlyDSL mono components](https://github.com/ROCm/ATOM/tree/53e4fb52403707ab07b69ee909501ea3cc25ca5c/atom/mono) provide buffers, synchronization, and reductions that model-specific schedules can reuse. [2], [4], [6]

Each model supplies the schedule that connects its own stages. ATOM connects that schedule to serving, where batching and speculative decoding can change the number of tokens in a step. The runtime must select a supported path and make the same choice on every TP rank. [4], [6]

M3 and V4.1 show how these shared controls support different model designs.

### MiniMax-M3: connecting attention and experts

In MiniMax-M3's sparse layers, the FlyDSL mono kernel connects input normalization, attention, and mixture-of-experts (MoE) computation through to the final TP reduction. The fused path includes the main matrix multiplications: Q/K/V and indexer projections, the attention output projection, routing, and the shared and routed experts' up/gate and down projections. It also handles head normalization, rotary position embeddings (RoPE), and cache writes. [2]

![MiniMax-M3 FlyDSL mono computation: input projections, query and cache preparation, sparse attention, output projection and reduction, routing, and expert projections and reduction](images/minimax-mono-flow.svg)

*Figure 2. Computations fused in M3's sparse-layer FlyDSL mono path. Arrows summarize data flow; independent tasks can overlap. [2]*

Three scheduling choices reduce waits: [2]

- **Reuse thread blocks across stages.** A block that finishes attention work can take on routing or expert tasks, letting the schedule reuse GPU resources as the layer progresses.
- **Load weights early.** Blocks can load a later stage's weights while earlier stages compute. The calculation waits for its input, but the weight loads can already be in progress.
- **Pass on ready data.** Tagged buffers signal when an output is ready. Later stages and cross-GPU reductions can start on the portions they need while other blocks produce the rest.

**Decode latency.** The [M3 measurements](data/minimax-decode-step.csv) use MI355X, TP4, MiniMax-M3-MXFP4, a 1M-token context, and EAGLE3 with three draft tokens. **Separate** and **FlyDSL Mono** compare complete ATOM execution paths at the measured revision. **FlyDSL Mono + CP** adds indexer context parallelism, a separate long-context optimization described in the [supporting measurements](#appendix-supporting-measurements). [7]

Times are median GPU decode-step latency in milliseconds; parentheses show speedup relative to Separate.

| Batch | Separate | FlyDSL Mono | FlyDSL Mono + CP |
| ---: | ---: | ---: | ---: |
| 1 | 11.47 | 7.00 (**1.64×**) | 6.26 (**1.83×**) |
| 2 | 13.40 | 9.50 (**1.41×**) | 7.77 (**1.72×**) |
| 4 | 18.47 | 15.88 (**1.16×**) | 12.36 (**1.49×**) |

![M3 GPU decode-step latency in ATOM: separate-kernel baseline, FlyDSL Mono, and FlyDSL Mono plus indexer CP compared at each batch size of 1, 2, and 4](images/minimax-flydsl-decode-step.png)

*Figure 3. Each batch size has its own separate-kernel baseline. Labels show latency and speedup against that baseline. Indexer CP is an additional long-context optimization. [7]*

**FlyDSL Mono alone delivers a 1.64× speedup at batch one; adding CP brings the speedup to 1.83×.** Gains shrink as batch size grows. A larger batch gives separate kernels more parallel work and spreads launch overhead across more tokens, which can reduce the relative advantage of FlyDSL Mono.

These measurements cover GPU decode steps; they do not measure per-user token rate or isolate the contribution of each scheduling choice. A [launch example](data/run-minimax-m3-mono.sh) starts an ATOM server using the measured revision's recipe.

### DeepSeek-V4.1 in ATOM

DeepSeek-V4.1 applies FlyDSL mono kernels to attention preparation and the path from attention to MoE: [4]

- **Prepare attention inputs.** Combine residual updates and normalization with query and KV projections, RoPE, and window-cache writes, adding indexer projections where needed.
- **Connect attention to MoE.** Score and select keys where needed, compute attention and output projections, then reduce the results across TP ranks and normalize the input to MoE. Routing, shared and routed expert projections, and the expert reduction are fused into this path.

The schedule makes each normalized row available to the router and expert stages as it becomes ready. Output-projection weight loads start before the kernel waits for attention output, allowing memory access to overlap earlier computation. For eligible steps, GPUs also divide token rows among themselves for some stages, reducing work repeated on every GPU. FlyDSL gives developers control over these choices: when to load, when to wait, and how to assign work. [4]

![DeepSeek-V4.1 FlyDSL mono computations: prepare attention inputs, select keys where needed, compute attention, project and reduce its output, then route tokens and compute and reduce expert outputs](images/atom-deepseek-v41-mono-flow.svg)

*Figure 4. Data flow through DeepSeek-V4.1's FlyDSL mono computations. Arrows show dependencies between the fused computations. [4]*

The DSpark draft runner also reuses these FlyDSL kernels, adapting their attention and expert settings for draft computation. [4]

### Interactivity results in ATOM

The [InferenceX AgentX](https://inferencex.semianalysis.com/agentx) benchmark replays multi-turn, long-context agent sessions. It collects inter-token latency for each request and reports **P90 interactivity as 1 / P90 inter-token latency**, with latency in seconds. Higher values mean shorter delays between tokens at the slower end of the request distribution. [8]

The [published ATOM results](data/atom-agentx-source.json) below use **MI355X** with FlyDSL mono decode enabled at concurrency one (C1). **The source provides no matched baseline with FlyDSL Mono off**, so these figures describe complete serving configurations rather than the gain from enabling FlyDSL Mono.

- **MiniMax-M3-MXFP4, TP4:** EAGLE3 with three draft tokens and synthetic acceptance length 2.78, FP8 KV and index caches.
- **DeepSeek-V4.1-Flash, TP2 and TP4:** DSpark with five draft tokens and synthetic acceptance length 3.51, BF16 KV and an FP4 index cache.

Acceptance length is the average number of output tokens per verification step, including the target token. These runs set acceptance synthetically, so acceptance during normal serving may differ. Both model setups use GPU-resident KV caches and a configured **3,600-second profiling window**. The [ATOM serving data](data/atom-agentx.csv) links to the exact recipes, container images, acceptance settings, and source records.

The table reports P90 interactivity in tokens/s/user, **output throughput** in generated tokens/s/GPU, and P90 time to first token (TTFT) in seconds. Output throughput is averaged over the full profiling window; interactivity describes the pace between tokens once a request starts generating.

| Model / TP | Interactivity | Output rate | TTFT |
| --- | ---: | ---: | ---: |
| M3 / TP4 | **469.5** | 42.5 | 1.13 |
| V4.1 / TP2 | **505.1** | 86.5 | 1.20 |
| V4.1 / TP4 | **641.0** | 45.5 | 1.20 |

For M3 at C1, input + output throughput is **6,580 tokens/s/GPU**, but generated output contributes only **42.5**. The larger total is dominated by input tokens, including cached input, so it does not represent generation speed. The M3 GPU-step comparison above measures the effect of enabling FlyDSL Mono; serving results also reflect scheduling, caching, and speculative decoding. TTFT and tool waits also affect user responsiveness.

## Extending to vLLM and SGLang

**vLLM** integrates FlyDSL mono kernels into its decode path, adapting them to its KV-cache layout and runtime. For DeepSeek-V4.1-Flash on MI355X, a published same-node **FlyDSL Mono off/on** comparison uses **TP2**, with fusion enabled on 30 standard layers in the on run. At **concurrency one**, P90 interactivity rises from **318.1 to 429.1 tokens/s/user (1.35×)**. Each concurrency setting has **one 3,600-second run per mode**, using DSpark with five draft tokens and synthetic acceptance length 3.51. [3]

![DeepSeek-V4.1-Flash in vLLM on MI355X TP2: paired FlyDSL Mono off/on bars show 1.35, 1.28, 1.15, and 1.14 times P90 interactivity at concurrency 1, 2, 4, and 8](images/vllm-dsv41-interactivity.png)

*Figure 5. Each group compares FlyDSL Mono off and on at the same request concurrency. Longer bars mean faster token generation. Labels show the measured values and their ratio. [3]*

TTFT is higher at C1/C8 and lower at C2/C4 in these runs. More runs would be needed to establish a repeatable change; the [supporting measurements](#vllm-evaluation-details) include all four pairs. As with the ATOM benchmarks, acceptance is set synthetically, so these runs do not measure the full user experience under natural acceptance.

**SGLang's [framework RFC][9]** provides an extension point for FlyDSL mono kernels through GPU-vendor plugins, including the [AMD plugin](https://github.com/sgl-project/sgl-amd-mono). The plugins supply model kernels, while SGLang handles step selection and runtime integration.

## Scope and costs

Fusion can increase register and shared-memory use, leaving room for fewer thread blocks to run at once. Developers must account for that limit when choosing which operations to combine. [2]

**Where gains shrink.** M3's GPU-step speedup falls from 1.64× at batch one to 1.16× at batch four; vLLM's interactivity gain falls from 1.35× at C1 to 1.14× at C8. A separate vLLM FFN-only layer test is already near parity at TP2/C8: **345 → 342 µs (1.01×)** at 48 token rows. The available full-model measurements do not show where FlyDSL Mono and the separate path reach parity. [3], [7]

**When FlyDSL Mono takes over.** The runtime selects FlyDSL Mono based on the model, GPU, TP size, layouts, and step metadata. In the cited revisions, M3 supports up to **16 token rows** per step; ATOM and vLLM V4.1 support up to **48**. Speculative tokens and graph padding count toward these limits: one request with five drafts can use six rows before padding. In a supported deployment, larger or unsupported steps use the standard path, with every TP rank making the same choice. These row limits define supported shapes; they do not identify where FlyDSL Mono stops improving performance. [2], [3], [4]

**Development cost.** Each model needs a schedule that developers must validate and maintain as the model changes. Correctness checks and task evaluations are part of that work. [3]

## Build a FlyDSL mono path for your model

Start with the [FlyDSL mono components](https://github.com/ROCm/FlyDSL/tree/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel) and [ATOM's shared FlyDSL mono framework](https://github.com/ROCm/ATOM/tree/53e4fb52403707ab07b69ee909501ea3cc25ca5c/atom/mono):

1. **Find the waits.** Profile the separate path at your serving batch sizes. Identify where stages wait for data or communication.
2. **Write the schedule in FlyDSL.** Use reusable compute and communication components to connect stages. Assign independent tasks to thread blocks so their work can overlap.
3. **Integrate and evaluate.** Add runtime dispatch, fallback, and correctness checks. Evaluate task quality with actual rejection sampling, then compare interactivity, TTFT, and throughput at the same concurrency levels.

## Appendix: Supporting measurements

**Indexer context parallelism in M3.** ATOM splits index scoring across four TP ranks, then exchanges candidates to make the final selection. This is the additional optimization behind the FlyDSL Mono + CP results. [2], [7]

At 1M context, TP4, 16 token rows, and four query rows per request, the [layer measurement](data/minimax-indexer-cp-layer.csv) records **222.2 → 156.7 µs (1.42×)** with CP. Both runs use FlyDSL Mono and graph replay. [7]

![MiniMax-M3 indexer context parallelism across four TP ranks](images/minimax-indexer-cp.svg)

*Figure 6. Each rank scores part of the cache, then exchanges its best candidates so the owning rank can make the final selection.*

**Direct FlyDSL Mono comparisons.** Figure 7 summarizes the paired results from Figures 3 and 5. Each group includes its own FlyDSL Mono off baseline, normalized to 1.00×. M3 uses ATOM GPU-step latency; V4.1 uses vLLM interactivity. [3], [7]

![Direct FlyDSL Mono comparisons: M3 in ATOM at batch 1, 2, and 4, and V4.1 in vLLM at concurrency 1, 2, 4, and 8; each group includes a FlyDSL Mono off baseline at 1.00 times](images/mono-speedup-comparison.png)

*Figure 7. Each group pairs FlyDSL Mono off and on. Ratios are latency (off/on) for M3 and interactivity (on/off) for V4.1. The raw values appear in Figures 3 and 5. [3], [7]*

### vLLM evaluation details

**TTFT.** The table below gives P90 TTFT in seconds for the TP2 FlyDSL Mono off/on runs in Figure 5. Each mode has one 3,600-second run per concurrency; repeated measurements would be needed to establish a consistent effect. [3]

| Concurrency | FlyDSL Mono off | FlyDSL Mono on |
| ---: | ---: | ---: |
| 1 | 1.18 | 1.28 |
| 2 | 1.10 | 1.08 |
| 4 | 1.27 | 1.26 |
| 8 | 1.01 | 1.08 |

**Additional vLLM measurements.** The [comparison CSV](data/mono-benefits.csv) records the TP2 FlyDSL Mono off/on results at C1, C2, C4, and C8, including TTFT. The CSV retains input + output totals, which include cached input and are not output throughput. [3]

## References

1. [FlyDSL language and compiler](https://github.com/ROCm/FlyDSL).
2. [ATOM's M3 FlyDSL mono implementation](https://github.com/ROCm/ATOM/tree/53e4fb52403707ab07b69ee909501ea3cc25ca5c/atom/models/minimax_m3/mono).
3. [vLLM: DeepSeek-V4.1-Flash FlyDSL mono decode and AgentX results](https://github.com/vllm-project/vllm/pull/60397).
4. [ATOM's DeepSeek-V4.1 FlyDSL mono implementation](https://github.com/ROCm/ATOM/tree/53e4fb52403707ab07b69ee909501ea3cc25ca5c/atom/models/deepseek_v41/mono).
5. [TileRT](https://github.com/tile-ai/TileRT).
6. [ATOM: GLM-5.2 and Kimi-K3 FlyDSL mono decode](https://github.com/ROCm/ATOM/pull/2535).
7. [ATOM: MiniMax-M3 FlyDSL mono decode and measurements](https://github.com/ROCm/ATOM/pull/2419).
8. [InferenceX AgentX metric definitions](https://github.com/SemiAnalysisAI/InferenceX/blob/8b25006fa0b23c7e157438996a3d2a90ff5e8cd2/inferencex-e2e/infx/results/agentic/request_metrics.py).
9. [SGLang: GPU kernel framework RFC](https://github.com/sgl-project/sglang/issues/43949).

[1]: https://github.com/ROCm/FlyDSL
[2]: https://github.com/ROCm/ATOM/tree/53e4fb52403707ab07b69ee909501ea3cc25ca5c/atom/models/minimax_m3/mono
[3]: https://github.com/vllm-project/vllm/pull/60397
[4]: https://github.com/ROCm/ATOM/tree/53e4fb52403707ab07b69ee909501ea3cc25ca5c/atom/models/deepseek_v41/mono
[5]: https://github.com/tile-ai/TileRT
[6]: https://github.com/ROCm/ATOM/pull/2535
[7]: https://github.com/ROCm/ATOM/pull/2419
[8]: https://github.com/SemiAnalysisAI/InferenceX/blob/8b25006fa0b23c7e157438996a3d2a90ff5e8cd2/inferencex-e2e/infx/results/agentic/request_metrics.py
[9]: https://github.com/sgl-project/sglang/issues/43949
