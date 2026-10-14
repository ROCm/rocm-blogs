---
blogpost: true
blog_title: "Mono Kernels: Faster LLM Decode on AMD Instinct"
date: 12 Oct 2026
author: ""
thumbnail: 'mono-overlap.png'
tags: AI/ML, Performance
category: Software tools & optimizations
target_audience: GPU kernel developers and LLM inference engineers
key_value_propositions: Shows how FlyDSL mono kernels reduce launch overhead and stage-boundary stalls, with up to 1.83x faster M3 decode in ATOM and reusable components for other models and serving engines.
language: English
myst:
    html_meta:
        "property=og:locale": "en_US"
        "description lang=en": "Explore mono kernels on AMD Instinct: FlyDSL scheduling, prefetch, tagged handoffs, fused AllReduce, and up to 1.83x faster M3 decode with indexer CP."
        "keywords": "MiniMax-M3, FlyDSL, ATOM, TileRT, mono kernel, ROCm, AMD Instinct MI355X, LLM inference, decode, indexer context parallelism"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
---

# Mono Kernels: Faster LLM Decode on AMD Instinct

Interactive chat and coding assistants need fast responses when only one or a few requests are active. Each decode step sends a small number of token rows through every layer, often as many short GPU kernels. Launching those kernels and waiting between stages can take a substantial share of the step.

**Mono kernels speed up low-concurrency decode by keeping connected computations and communication in one resident GPU schedule.** [FlyDSL](https://github.com/ROCm/FlyDSL) makes that schedule programmable; [ATOM](https://github.com/ROCm/ATOM) runs it as part of its serving engine. On MI355X at batch one and a 1M context, MiniMax-M3's GPU decode step is **1.64× faster with mono**, rising to **1.83× with indexer context parallelism (CP)**: **11.47 → 7.00 → 6.26 ms** within ATOM. [1]

![M3 GPU decode-step latency: ATOM separate kernels 11.47 ms, FlyDSL mono 7.00 ms, mono plus indexer CP 6.26 ms](images/minimax-decode-step.png)

*Figure 1. M3 at batch one: mono reduces launch and stage-boundary costs; CP additionally divides the long-context index scan across GPUs. MI355X, TP4, MXFP4, 1M context, EAGLE3 with three speculative tokens. Lower is better. [1]*

## Why many small kernels slow down decode

Consider one sparse decoder layer. Normalization and projections prepare queries, keys, and values. Cache updates and index selection feed attention. Output projection and reduction feed expert routing; expert GEMMs and another reduction finish the layer. Each producer-to-consumer boundary is a place where work may wait. [3]

The launch count can be large: vLLM's DeepSeek-V4.1 PR reports **about 40 separate launches per standard layer, replaced by two persistent FlyDSL launches, K1 and K2**. Its MI355X TP4 layer harness measures **180 → 94 µs** for a standard ratio-1 layer at six token rows. Fewer launches are part of that gain; overlapping loads, computation, and reductions also matters. [14]

HIP graph replay reduces CPU submission overhead, but the GPU still executes separate kernels and their dependencies. A later GEMM could load its weights while the previous stage produces its input. A reduction could start on one completed row while other rows are still being computed. Mono lets the kernel author express both kinds of overlap inside the layer.

The related ROCm posts on [low-latency GEMMs](https://rocm.blogs.amd.com/software-tools-optimization/accelerating-llm-inference-on-amd-gpus-with-low-latency-gemms/README.html) and [ATOM's Two-Batch Overlap](https://rocm.blogs.amd.com/software-tools-optimization/atom-optimiztion/README.html) optimize individual GEMMs and overlap work across micro-batches. Mono addresses the next question: how can the small operations **within one layer** keep each other moving?

## Mono kernels: several stages, one launch

A mono kernel launches a group of thread blocks that stay active through several stages of the layer. This is the resident grid. A block can finish an attention task and later perform routing or expert work. Blocks pass intermediate results through GPU memory that other blocks can read, and start dependent work when its inputs are ready. Registers and Local Data Share (LDS), the GPU's on-chip scratchpad, hold local working data. [3], [7]

This is a form of `megakernel`: several kinds of work share one launch. The useful distinction is how they are scheduled. Blocks have different jobs, and a consumer waits for the inputs it needs while independent work proceeds elsewhere in the grid. [3]

There are three main benefits:

| What costs time | What changes inside mono |
| --- | --- |
| Separate launches and waits for whole stages to finish | Blocks run several stages in one launch and advance when their own inputs are ready, without a grid-wide barrier between every stage. |
| Waiting to start the next stage's weight loads | A block can load those weights while earlier computation is still running. |
| Waiting between computation and GPU-to-GPU reduction | Blocks produce, reduce, and consume parts of an output within the same program. |

![Staged launches versus a resident mono grid with early loads and tile-level handoffs](images/mono-overlap.svg)

*Figure 2. Mono combines stages in one launch, starts independent loads early, passes ready tiles forward without per-stage grid-wide barriers, and overlaps reductions with computation. Bars show execution order, not measured durations.*

## Why FlyDSL makes the schedule practical to write

FlyDSL is a Python domain-specific language for GPU kernels. It gives the author control over tensor layouts, matrix instructions, memory loads, and synchronization; an MLIR compiler pipeline turns the program into GPU code. For mono, this means the task assignments and the storage they use can be generated together. M3 builds a variant for each row count from 1 to 16, choosing how tokens share expert weights before the kernel runs. [2], [3]

Two small choices in the [M3 layout code](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/layout.py) show why that control matters:

- **Spread simultaneous loads across LDS banks.** At 16 rows, the unpadded layout causes 16-way bank conflicts, costing about **3 µs per up/gate task** according to the source comment. Adding 16 bytes to each row's pitch removes those conflicts. The layout change sits beside the Python task assignments.
- **Avoid making every expert block poll the same location.** The router publishes a 1 KB score payload that 240 expert blocks need. Making eight copies spreads those reads out; the source records **37.3 → 35.9 µs** for the four-GPU K4 kernel. [11]

These are changes to how work and data move through the fused layer. The GEMMs remain specialized for their shapes, and the author can improve a handoff without redesigning the whole kernel. The M3 work also added or adopted FlyDSL primitives such as `fx.rocdl.readlane`, `ballot`, and `fx.ctpop`. [1]

The design is inspired by the **TileRT MonoKernel work**, including the [InferenceMAX/InferenceX snapshot `8ac9834`](https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9) cited by FlyDSL #1204 and associated with [#3389](https://github.com/SemiAnalysisAI/InferenceX/pull/3389). FlyDSL's [kernels/monokernel/](https://github.com/ROCm/FlyDSL/tree/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel) provides gfx950 GEMM, MXFP8 linear, and tagged symmetric AllReduce components. ATOM's [atom/mono/](https://github.com/ROCm/ATOM/tree/5e8d9131739aaa53836713c79e6d547c23980f03/atom/mono) provides shared device, planning, and runtime code for model implementations. [5], [6], [10]

## MiniMax-M3: one launch for a sparse decoder layer

M3's [mono kernel](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/post_attn.py) covers the complete sparse-layer path, including attention and mixture-of-experts (MoE) AllReduce, in **one launch per tensor-parallel (TP) rank**, with one rank on each participating GPU. The first three dense layers use ATOM's separate-kernel modules. The kernel specializes for 1–16 token rows; in the EAGLE3 configuration used here, three speculative tokens produce four target-model rows per request. [1], [3]

### Which operations are fused?

![MiniMax-M3 sparse-layer stages inside one FlyDSL launch](images/minimax-mono-flow.svg)

*Figure 3. M3's sparse-layer dependency chain. The source calls the pre-attention stages K1 and the layer kernel K4; K1 is fused into K4.*

**Prepare queries, keys, and values.** The [pre-attention stages](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/pre_attn.py) normalize and quantize inputs to per-token FP8, then use FP8 matrix instructions for the Q/K/V and index projections. Head normalization, rotary position embedding (RoPE), and cache insertion consume the projection outputs inside the same program. [3]

**Select the cache blocks and run sparse attention.** M3's [index scorer](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/index_score.py) assigns one 128-key block of a request to a wave. Speculative query rows occupy matrix-instruction columns, so the wave reuses a key load across those rows. Only a block that may contain newly inserted keys waits for that insertion. Selection chooses 16 blocks of 128 tokens, and sparse attention processes them in eight partitions of 256 keys. Scoring, selection, and attention stay connected within the launch. [3]

**Project and reduce the attention output.** Partition merging and FP8 quantization prepare the output-projection operands. Projection tasks produce TP partials; the attention reduction combines them and feeds the residual update and normalization needed by the router. This connects the attention result directly to the input of the expert computation. [3]

**Route tokens through the MoE.** The router selects four routed experts alongside a shared expert. Up/gate projections, activation, down projections, and MoE reduction complete the layer inside the same kernel. Narrow steps use BF16 activations with MXFP4 expert weights. Wider steps use MXFP8 activations and group tokens by expert to reuse weights. The final reduction publishes the inputs for the next layer. [3]

All of these operations stay in the same launch. To make that useful, M3 needs to keep blocks busy, load weights early, and tell each consumer when its input is ready.

### Reuse blocks as the layer advances

M3 launches **256 workgroups of 512 threads**, targeting one active workgroup per MI355X compute unit. Each block runs its assigned tasks in order. A block waiting for data can rely on the producing blocks remaining active. This avoids stopping the entire grid at every stage boundary, although threads within a workgroup still use local barriers. [3]

Blocks that perform attention can take on routing work later in the same launch. FlyDSL assigns those jobs before execution, so the same group of blocks carries the layer forward without a new launch for each stage. [11]

At four rows, routed up/gate tasks are grouped by token so a block can reuse one staged activation. At wider shapes, tokens sharing an expert become matrix columns, reusing that expert's weights. The row count determines the layout before the kernel runs. Actual token positions and selected experts are supplied at runtime. [3], [11]

### Load weights before the next stage needs them

The weights for a later stage can be loaded before that stage receives its input. In the narrow MoE path, a block starts loading its down-projection weights while its up/gate computation runs. Shared-expert tasks are assigned according to each block's routed-work load; where that load is balanced, shared-expert computation runs while the first routed task's weights are in flight. [3], [11]

### Tell each block when its input is ready

After loading its weights, a block still needs the previous stage's output. M3 uses mailboxes in GPU memory: each value has a tag identifying its decode step and layer. A block waits until the tags show that its input is ready. Advancing the step counter prevents graph replay from mistaking an old result for a new one. Mailbox memory accesses make writes visible to the blocks or peer GPUs that need them. The [mailbox implementation](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/common.py) supplies this synchronization inside the fused kernel. [3], [7]

### Combine AllReduce with the computations around it

With tensor parallelism, each GPU computes part of the output projection. Those partial results must be summed before the next computation can use them. In M3 mono, projection tasks write their partial hidden rows to peer memory. As soon as all GPUs have supplied a row, its reduction sums the contributions in rank order and passes the result to residual update, normalization, and routing. Other blocks can still be producing other rows. MoE reduction works the same way at the end of the layer. [3]

M3 specializes both expert computation and communication as row count grows:

| Token rows | Expert computation | Attention and MoE AllReduce |
| --- | --- | --- |
| 1–4 | Tasks organized by token and expert slot | Push partial rows to peers; each rank accumulates them. |
| 5–11 | Tokens grouped by distinct expert | Same direct peer exchange. |
| 12–16 | Tokens grouped by distinct expert | Reduce-scatter to each row's owner, then all-gather the sums. |

At 12–16 rows, each row has an owner GPU. Reduce-scatter sends partials to that owner for summation; all-gather distributes the sums. This reduces bytes per link, while keeping the reductions and the computations that use them in the same kernel. [3], [11]

### Split the long-context index scan across GPUs

Fusing kernels reduces launch and synchronization overhead. The indexer still has to scan the cache to find the blocks attention should read. At a one-million-token context, this is substantial work. **Indexer context parallelism (CP)** divides that scan across GPUs. [1]

In the TP path, each rank scans the cache for its own index-query head. With CP, each rank scans **one quarter of the context for all four heads**. Each head's owning rank receives candidates and completes selection. The extra index-query projections run only when a step contains a request past **288 index blocks**, approximately 36K tokens, where CP activates. [1], [3]

![MiniMax-M3 indexer context parallelism across four TP ranks](images/minimax-indexer-cp.svg)

*Figure 4. For each head, four ranks contribute two partitions each. Each partition returns its top 16 candidates; the owning rank merges 128 candidates into the final selection of 16 blocks.*

The CP layer harness isolates the indexer change: at 1M context, 16 rows, and four query tokens per request, graph-replay latency falls from **222.2 to 156.7 µs**, a **1.42× speedup**. Short-context changes stay within the PR's reported ±0.9 µs. The [CP layer-harness CSV](data/minimax-indexer-cp-layer.csv) records this separate measurement. [1]

Mono reduces waits between stages; CP reduces the long-context scan on each GPU. The results below separate their contributions.

## Performance results

### Scope and methodology

Each result compares a serving engine's separate-kernel path with its FlyDSL mono path. Speedup is **baseline / mono for latency**, and **mono / baseline for throughput or interactivity**; 1.00× means unchanged. These ratios summarize each test, with the measured scope beside them. They are not a ranking between models. M3's CP result adds an indexer optimization; Kimi-K3 and GLM-5.2 from open ATOM PR #2435 remain **in progress**.[^benchmark-data]

### M3: decode latency with mono and indexer CP

The server measurements use MI355X, TP4, MiniMax-M3-MXFP4, a 1M context, and EAGLE3 with three speculative tokens. Batches of 1, 2, and 4 produce 4, 8, and 16 target-model rows before graph padding. The [decode-step CSV](data/minimax-decode-step.csv) retains the reported values. **Separate** and **Mono** are the two execution paths within ATOM. Latencies are in milliseconds; Gain lists mono / mono + CP speedups. [1]

| Batch | Separate | Mono | Mono + CP | Gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.47 | 7.00 | 6.26 | **1.64× / 1.83×** |
| 2 | 13.40 | 9.50 | 7.77 | **1.41× / 1.72×** |
| 4 | 18.47 | 15.88 | 12.36 | **1.16× / 1.49×** |

### One-launch layer results: GLM-5

[GLM-5's full-indexer mono kernel](https://github.com/ROCm/FlyDSL/blob/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel/glm/kernel.py) combines indexer, sparse MLA, router, MXFP4 MoE, TP reductions, and residual update in **one launch per rank**. FlyDSL #1204 reports these results on MI355X TP8 at position 3000, against ATOM's separate-kernel layer implementation: [5]

| Token rows | Mono launches per layer and rank | Layer latency, separate → mono (µs) | Speedup |
| ---: | ---: | ---: | ---: |
| 4 | **1** | 205.8–213.7 → **78.4–78.6** | **2.62–2.73×** |
| 8 | **1** | 223.4 → **118.2–118.3** | **1.89×** |

### Why the gain changes with batch size

At batch one, M3 mono saves **4.47 ms** and CP saves a further **0.74 ms**. At batch four, mono saves **2.59 ms** and CP saves another **3.52 ms**. More rows spread launch overhead across more useful work, consistent with mono's smaller speedup at batch four. Weight loads, expert computation, and index scoring account for more of the step; grouping tokens by expert and dividing the scan across GPUs target those costs.

Mono is most useful when a layer has many short, dependent operations. A resident grid must fit on the GPU, and a larger fused region does not automatically run faster. Use the actual decode widths to choose the fusion boundary and keep the separate path for steps outside that range. [3], [5]

### Results across models

All four models below use MI355X. Each ratio uses the metric in its row; the [comparison CSV](data/mono-benefits.csv) records the full test configuration.

| Model, engine, and test | Measured metric | Speedup |
| --- | --- | ---: |
| **MiniMax-M3**: sparse layer; ATOM TP4, batch 1, 1M context | GPU decode step: 11.47 → 7.00 ms (6.26 with CP) | **1.64×** (**1.83×** with CP) [1] |
| **Kimi-K3**: MoE tail; ATOM TP8, concurrency 1 | Output throughput: 169.15 → 197.92 tokens/s | **1.17×** [8] |
| **GLM-5.2**: decoder layer; ATOM TP4, concurrency 1 | Agentic throughput: +15.92% | **1.16×** [8] |
| **DeepSeek-V4.1-Flash**: standard layer; vLLM TP4, six rows, ratio-1 harness | Layer latency: 180 → 94 µs | **1.91×** [14] |

Kimi-K3 keeps production attention, KDA, and `AttnRes`; the comparison changes only the MoE tail. GLM-5.2 also reports **1.17× throughput** at concurrency two. The [comparison CSV](data/mono-benefits.csv) includes the additional throughput and time-per-output-token measurements. [8]

### DeepSeek-V4.1: adapt the fusion boundary to the model

DeepSeek-V4.1 has state updates between groups of operations. Its [runner](https://github.com/ROCm/ATOM/blob/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono/runner.py) keeps Engram before attention preparation where present, and a compressor before index scoring on selecting layers. Around those calls, it uses **two mono launches per rank: K1 and K2**. [13]

[K1](https://github.com/ROCm/ATOM/blob/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono/kernels/attn_pre.py) fuses normalization, Q/KV and index-query projections, RoPE, and window-cache insertion. [K2](https://github.com/ROCm/ATOM/blob/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono/kernels/layer_post.py) joins **K2a—index selection, sparse attention, output projections, attention reduction, and normalization—with K2b—routing, expert GEMMs, and FFN reduction—in one launch**. The normalized attention output becomes the MoE input inside that kernel. [13]

![DeepSeek-V4.1 keeps state updates around K1 and K2, with attention and experts fused in K2](images/deepseek-v41-mono-flow.svg)

*Figure 5. Two mono launches preserve the original state-update calls while keeping attention and experts together inside K2.*

M3 can fuse a whole sparse layer. V4.1 keeps the compressor between K1 and K2, preserving its cache updates while removing launches between the attention and expert stages inside K2. The model's dependencies determine which operations belong in each fused region.

The ATOM V4.1 schedule also continues to improve: #2545 reports **4% / 10% / 11% / 12% lower decode-step time** at batches 1/2/4/8, with bit-identical outputs, relative to the previous mono schedule. [9]

### Check outputs as well as speed

M3's #2419 checks report **bit-identical index selection** over 3K–1.04M contexts, **bit-identical outputs through 16 chained layers with CP off versus on** at the tested 3K, 37K, and 200K contexts, and a successfully served **1,040,186-token request**. Selection coverage includes mixed lengths across the CP threshold and speculative row counts. These checks cover cache access and intermediate results affected by the fused implementation. [1]

*Quality note: M3's GSM8K five-shot scores are 0.9447 for ATOM's separate-kernel path, 0.9371/0.9424 for mono, and 0.9378 for mono plus CP. The two mono runs span 0.53 percentage points; the 0.69-point difference between the separate-kernel path and mono plus CP is of similar magnitude. PR #2435 reports Kimi-K3 mono/separate-kernel scores of 0.959060/0.959818 and GLM-5.2 mono/separate-kernel scores of 0.9606/0.9674, a 0.68-point difference for GLM. [1], [8]*

## Run mono in ATOM and vLLM

ATOM (AiTer Optimized Model) supplies the serving side: it loads weights, manages requests and caches, selects the row-count variant, and prepares scratch buffers and peer connections before graph capture. [4] **M3's mono dispatch defaults on for eligible TP4 decode steps**, with unsupported steps falling back to its separate-kernel path. The stock M3 recipe enables the shared index cache, which disables mono; set **`use_index_cache: false`** to use mono.[^m3-support]

For development, `ATOM_MONO_CHECK` compares each layer with the original modules, `ATOM_MONO_TRACE` records step data, and `ATOM_MONO_TIMELINE` exposes work inside the fused layer. These tools let a kernel author check a new schedule before enabling it for serving. [1], [3]

**vLLM also has a merged FlyDSL mono path.** PRs [#60397](https://github.com/vllm-project/vllm/pull/60397) and [#61082](https://github.com/vllm-project/vllm/pull/61082) add DeepSeek-V4.1-Flash mono decode and extend K1/K2 coverage to all 40 backbone layers, adapting the ATOM stage and synchronization code. Enable it with **`--monokernel`** on a build containing #61082. The supported MI355X path uses TP2/TP4 and up to 48 decode rows, with full-graph or eager execution; prefill, mixed batches, and larger steps use the existing path. The [kernel documentation](https://github.com/vllm-project/vllm/blob/cc2a6eea1a9f92c1d6b47be8fe48adf127d4c67f/vllm/models/deepseek_v41/amd/mono/README.md) describes layer eligibility and the launch sequence. [14], [15]

To try the M3 configuration, use **four MI355X GPUs**, an [ATOM installation](https://github.com/ROCm/ATOM) containing #2419 with its FlyDSL/AITER dependencies, and access to the M3 MXFP4 and EAGLE3 checkpoints. Download [run-minimax-m3-mono.sh](data/run-minimax-m3-mono.sh), which adapts the pinned [M3 recipe](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/recipes/MiniMax-M3.md), then run:

```bash
ATOM_MONO_ENABLE=1 ATOM_M3_INDEXER_CP=1 bash run-minimax-m3-mono.sh
```

The script enables mono and CP, disables the shared index cache, and uses full graph replay with three speculative tokens. Confirm `MiniMax-M3 mono decode on` in the server log; a refused configuration logs the reason. For a comparison using the same launch parameters, set both environment variables to `0` for separate kernels, or only `ATOM_M3_INDEXER_CP=0` for mono without CP. [3], [12]

## Build a mono path for your model

Start with the [FlyDSL mono components](https://github.com/ROCm/FlyDSL/tree/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel) and [ATOM's shared mono framework](https://github.com/ROCm/ATOM/tree/5e8d9131739aaa53836713c79e6d547c23980f03/atom/mono). A useful workflow is:

1. **Choose a connected region.** Profile the separate path at the decode row counts you care about. Map projections, cache operations, attention, experts, and reductions; identify which inputs each stage actually waits for.
2. **Compose and schedule it.** Reuse GEMM, quantization, mailbox, and AllReduce pieces. Give blocks enough independent work to load weights early and pass completed rows to consumers. Specialize task placement and LDS layouts for each supported width.
3. **Validate it in the engine.** Compare intermediate and layer outputs with the separate modules, exercise graph replay and all participating ranks, then measure layer and serving performance. Add dispatch conditions and a fallback alongside the kernel.

There is concrete work to build on: the open Kimi-K3 and GLM-5.2 paths extend model coverage, and [vLLM #60731](https://github.com/vllm-project/vllm/pull/60731) proposes a gfx942 port for MI300X and MI325X, with measurements on MI325X. That port identifies larger decode widths and additional TP sizes as follow-ups; the merged V4.1 work identifies indexer scoring, top-k, and draft-layer fusion as further opportunities. [8], [15], [16]

Bring a model, a profile, or a measured schedule improvement to [FlyDSL](https://github.com/ROCm/FlyDSL), [ATOM](https://github.com/ROCm/ATOM), or [vLLM](https://github.com/vllm-project/vllm). The reusable pieces are available; the next step is to find which waits your model can remove.

[^benchmark-data]: No new GPU benchmarks, quality evaluations, or server runs were performed for this article. The CSV files retain source configurations and reported values. PR status was checked on October 14, 2026; refresh the in-progress paths before publication.

[^m3-support]: At ATOM #2419, M3 mono requires TP4, DP=PP=1, FP8 KV/index caches with 128-token blocks, `max_model_len` ≤ 1,048,576, eager or non-piecewise graphs, no TBO, and no ATOM vLLM/SGLang plugin mode. Eligible decode steps have 1–16 rows including graph padding and matching cache metadata. These are ATOM M3 conditions; vLLM's native V4.1 path has its own dispatch. [3], [12]

## References

1. [ATOM PR #2419: MiniMax-M3 mono decode, indexer CP, performance, and validation](https://github.com/ROCm/ATOM/pull/2419).
2. [FlyDSL: Python DSL, explicit layouts, and MLIR compiler stack](https://github.com/ROCm/FlyDSL).
3. [M3 mono implementation at the ATOM #2419 merge commit](https://github.com/ROCm/ATOM/tree/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono): configuration, layout, dispatch, runner, and kernels.
4. [ATOM: inference engine overview and supported features](https://github.com/ROCm/ATOM).
5. [FlyDSL PR #1204: GLM-5 and Kimi-K3 launch counts, performance, and TileRT design lineage](https://github.com/ROCm/FlyDSL/pull/1204).
6. [TileRT InferenceMAX/InferenceX submission snapshot cited by FlyDSL PR #1204](https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9), associated with [submission update #3389](https://github.com/SemiAnalysisAI/InferenceX/pull/3389).
7. [M3 device primitives and tagged mailboxes at the ATOM #2419 merge commit](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/common.py).
8. [ATOM PR #2435: Kimi-K3 MoE-tail and GLM-5.2 MTP4 mono performance and accuracy](https://github.com/ROCm/ATOM/pull/2435).
9. [ATOM PR #2545: DeepSeek-V4.1 mono token parallelism, handoff prefetch, and decode-step improvements](https://github.com/ROCm/ATOM/pull/2545).
10. [ATOM PR #2479: DeepSeek-V4.1 fused sublayers and shared mono framework](https://github.com/ROCm/ATOM/pull/2479).
11. [M3 task placement, LDS pitches, and row-count specialization at #2419](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/layout.py).
12. [MiniMax-M3 recipe and EAGLE3 server configuration at #2419](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/recipes/MiniMax-M3.md).
13. [DeepSeek-V4.1 mono runner, indexer, projections, experts, and token-parallel reductions at #2545](https://github.com/ROCm/ATOM/tree/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono).
14. [vLLM PR #60397: merged DeepSeek-V4.1-Flash mono decode, launch counts, and separate-path comparisons](https://github.com/vllm-project/vllm/pull/60397).
15. [vLLM PR #61082: merged coverage of all 40 backbone layers and the `--monokernel` switch](https://github.com/vllm-project/vllm/pull/61082).
16. [vLLM PR #60731: proposed gfx942 port for MI300X and MI325X, with MI325X measurements](https://github.com/vllm-project/vllm/pull/60731).

[1]: https://github.com/ROCm/ATOM/pull/2419
[2]: https://github.com/ROCm/FlyDSL
[3]: https://github.com/ROCm/ATOM/tree/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono
[4]: https://github.com/ROCm/ATOM
[5]: https://github.com/ROCm/FlyDSL/pull/1204
[6]: https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9
[7]: https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/common.py
[8]: https://github.com/ROCm/ATOM/pull/2435
[9]: https://github.com/ROCm/ATOM/pull/2545
[10]: https://github.com/ROCm/ATOM/pull/2479
[11]: https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/layout.py
[12]: https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/recipes/MiniMax-M3.md
[13]: https://github.com/ROCm/ATOM/tree/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono
[14]: https://github.com/vllm-project/vllm/pull/60397
[15]: https://github.com/vllm-project/vllm/pull/61082
[16]: https://github.com/vllm-project/vllm/pull/60731
