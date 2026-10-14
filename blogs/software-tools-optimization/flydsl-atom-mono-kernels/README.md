---
blogpost: true
blog_title: "Mono Kernels with FlyDSL: Up to 1.83× Faster LLM Decode in ATOM"
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

# Mono Kernels with FlyDSL: Up to 1.83× Faster LLM Decode in ATOM

Interactive chat and coding assistants need fast responses when only one or a few requests are active. Each decode step sends a small number of token rows through every layer, often as many short GPU kernels. Launching those kernels and waiting between stages can take a substantial share of the step.

**Mono kernels speed up low-concurrency decode by keeping connected computations and communication in one resident GPU schedule.** [FlyDSL](https://github.com/ROCm/FlyDSL) makes that schedule programmable; [ATOM](https://github.com/ROCm/ATOM) (AiTer Optimized Model) runs it as part of its serving engine. On MI355X at batch one and a 1M context, MiniMax-M3's GPU decode step is **1.64× faster with mono**, rising to **1.83× with indexer context parallelism (CP)**: **11.47 → 7.00 → 6.26 ms** within ATOM. [1]

![M3 GPU decode-step latency: ATOM separate kernels 11.47 ms, FlyDSL mono 7.00 ms, FlyDSL mono plus indexer CP 6.26 ms](images/minimax-flydsl-decode-step.png)

*Figure 1. FlyDSL mono reduces M3 decode latency in ATOM; indexer CP adds a further gain at long context. [1]*

## Why many small kernels slow down decode

Consider one sparse decoder layer. Normalization and projections prepare queries, keys, and values. Cache updates and index selection feed attention. Output projection and reduction feed expert routing; expert GEMMs and another reduction finish the layer. Each producer-to-consumer boundary is a place where work may wait. [3]

A typical DeepSeek-V4.1-Flash layer illustrates the scale: vLLM reports **about 40 separate kernels** for its standard layer path, replaced by two mono launches. [14]

HIP Graph replay reduces CPU submission overhead, but the GPU still executes separate kernels and their dependencies. A later GEMM could load its weights while the previous stage produces its input. A reduction could start on one completed row while other rows are still being computed. Mono lets the kernel author express both kinds of overlap inside the layer.

The related ROCm posts on [low-latency GEMMs](https://rocm.blogs.amd.com/software-tools-optimization/accelerating-llm-inference-on-amd-gpus-with-low-latency-gemms/README.html) and [ATOM's Two-Batch Overlap](https://rocm.blogs.amd.com/software-tools-optimization/atom-optimiztion/README.html) optimize individual GEMMs and overlap work across micro-batches. Mono addresses the next question: how can the small operations **within one layer** keep each other moving?

## Mono kernels: several stages, one launch

A mono kernel launches a grid of **workgroups (thread blocks in HIP)** designed to remain concurrently resident for the layer invocation. This is the resident grid. A workgroup can finish an attention task and later perform routing or expert work. Workgroups pass intermediate results through global-memory buffers and start dependent work when its inputs are ready. Registers and Local Data Share (LDS), the CU-local, software-managed on-chip memory, hold local working data. [3], [7]

This is a form of `megakernel`: several kinds of work share one launch. The useful distinction is how they are scheduled. Workgroups have different jobs, and a consumer waits for the inputs it needs while independent work proceeds elsewhere in the grid. [3]

There are three main benefits:

| What costs time | What changes inside mono |
| --- | --- |
| Separate launches and waits for whole stages to finish | Workgroups run several stages in one launch and advance when their own inputs are ready, without a grid-wide barrier between every stage. |
| Waiting to start the next stage's weight loads | A workgroup can issue global-memory loads while earlier computation is still running. |
| Waiting between computation and GPU-to-GPU reduction | Workgroups produce, reduce, and consume parts of an output within the same program. |

![Staged launches versus a resident mono grid with early loads and tile-level handoffs](images/mono-overlap.svg)

*Figure 2. Mono combines stages in one launch, starts independent loads early, passes ready tiles forward without per-stage grid-wide barriers, and overlaps reductions with computation. Bars show execution order, not measured durations.*

## Why FlyDSL makes the schedule practical to write

FlyDSL is a Python domain-specific language for GPU kernels. It gives the author control over tensor layouts, matrix instructions, memory loads, and synchronization; an MLIR compiler pipeline turns the program into GPU code. For mono, the author can generate task assignments and their storage layouts together. M3 builds a variant for each row count from 1 to 16, choosing how tokens share expert weights before the kernel runs. [2], [3]

Efficient operand delivery and handoffs depend on the data layout. Two small choices in the [M3 layout code](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/layout.py) show why that control matters:

- **Spread simultaneous loads across LDS banks.** For this operand layout at 16 token rows, the source comment records 16-way bank conflicts costing about **3 µs per up/gate task** in the measured implementation. Adding 16 bytes to each row's pitch removes those conflicts. The layout change sits beside the Python task assignments.
- **Spread reads of the router results.** In the source-reported configuration, **240 expert workgroups** poll one 1 KB score payload. Making eight copies spreads those reads out; the source records **37.3 → 35.9 µs** for that four-GPU K4 configuration. [11]

These are changes to how work and data move through the fused layer. The GEMMs remain specialized for their shapes, and the author can improve a handoff without redesigning the whole kernel. The M3 work also added or adopted FlyDSL primitives such as `fx.rocdl.readlane`, `ballot`, and `fx.ctpop`. [1]

The design is inspired by the **TileRT MonoKernel work**, including the [InferenceMAX/InferenceX snapshot `8ac9834`](https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9) cited by FlyDSL #1204 and associated with [#3389](https://github.com/SemiAnalysisAI/InferenceX/pull/3389). FlyDSL's [kernels/monokernel/](https://github.com/ROCm/FlyDSL/tree/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel) provides gfx950 GEMM, MXFP8 linear, and tagged symmetric AllReduce components. ATOM's [atom/mono/](https://github.com/ROCm/ATOM/tree/5e8d9131739aaa53836713c79e6d547c23980f03/atom/mono) provides shared device, planning, and runtime code for model implementations. [5], [6], [10]

## MiniMax-M3: one launch for a sparse decoder layer

M3's [mono kernel](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/post_attn.py) covers the complete sparse-layer path, including attention and mixture-of-experts (MoE) AllReduce, in **one launch per tensor-parallel (TP) rank**, with one rank on each participating GPU. The first three dense layers use ATOM's separate-kernel modules. The kernel specializes for 1–16 token rows; in the EAGLE3 configuration used here, three speculative tokens produce four target-model rows per request. [1], [3]

### Which operations are fused?

![MiniMax-M3 sparse-layer stages inside one FlyDSL launch](images/minimax-mono-flow.svg)

*Figure 3. M3's sparse-layer dependency chain. The source calls the pre-attention stages K1 and the layer kernel K4; K1 is fused into K4.*

**Prepare queries, keys, and values.** The [pre-attention stages](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/pre_attn.py) normalize and quantize inputs to per-token FP8, then use FP8 **MFMA (matrix fused multiply-add) instructions** for the Q/K/V and index projections. Head normalization, rotary position embedding (RoPE), and cache insertion consume the projection outputs inside the same program. [3]

**Select the cache blocks and run sparse attention.** M3's [index scorer](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/index_score.py) changes its mapping with the number of scoring tasks. With at most 256 non-CP tasks, one workgroup handles a 128-key block and its **64-thread wavefronts (wave64)** split the key tiles. Larger task sets and CP use one block per wavefront. The scorer converts stored FP8 keys to BF16 MFMA operands and places speculative query rows in MFMA columns, reusing the key data across those rows. Only a block that may contain newly inserted keys waits for that insertion. Selection chooses 16 blocks of 128 tokens, and sparse attention processes them in eight partitions of 256 keys. Scoring, selection, and attention stay connected within the launch. [3]

**Project and reduce the attention output.** Partition merging and FP8 quantization prepare the output-projection operands. Projection tasks produce TP partials; the attention reduction combines them and feeds the residual update and normalization needed by the router. This connects the attention result directly to the input of the expert computation. [3]

**Route tokens through the MoE.** The router selects four routed experts alongside a shared expert. Up/gate projections, activation, down projections, and MoE reduction complete the layer inside the same kernel. Both narrow and wide paths use **MXFP8 activation operands with MXFP4 weights in native scaled MFMA instructions**. Narrow steps organize routed work by token and expert slot; wider steps group tokens by expert to reuse weights. The narrow path rounds the activation result between expert projections to BF16, then converts it to MXFP8 for the down projection. The final reduction publishes the inputs for the next layer. [3]

All of these operations stay in the same launch. To make that useful, M3 needs to keep workgroups busy, load weights early, and tell each consumer when its input is ready.

### Reuse workgroups as the layer advances

M3 launches **256 workgroups of 512 threads**, or eight wavefronts per workgroup, targeting one resident workgroup per MI355X compute unit. Forward progress depends on both a resource budget that permits concurrent residency and a task order whose dependencies point to earlier stages. Grid size alone does not establish residency. Workgroups advance when their inputs are ready; threads within a workgroup still use workgroup barriers. [3]

**M3's build-time schedule assigns tasks to workgroups.** A workgroup that performs attention can take on routing later in the same launch. The model's schedule, expressed using FlyDSL, carries the layer forward without a new launch for each stage. [11]

At four rows, routed up/gate tasks are grouped by token so a workgroup can reuse one staged activation. At wider shapes, tokens sharing an expert become matrix columns, reusing that expert's weights. The row count determines the layout before the kernel runs. Actual token positions and selected experts are supplied at runtime. [3], [11]

### Load weights before the next stage needs them

The weights for a later stage can be requested before that stage receives its input. In the narrow MoE path, **after the first up/gate task round**, the workgroup issues global-memory loads for the first token's routed down-projection weights and the shared expert's down weights. Those requests remain in flight during subsequent work and intermediate-result exchange. The schedule also overlaps shared-expert computation with the first routed task's weight loads where that task placement is selected. [3], [11]

Timing matters: issuing down loads too early can compete with upstream loads for bandwidth and lengthen VGPR live ranges, keeping vector registers occupied for longer. M3's schedule chooses a prefetch point that balances overlap with these costs. [3]

### Tell each workgroup when its input is ready

A workgroup can issue its weight loads while waiting for the previous stage's output. M3's [mailbox helper](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/common.py) stores payload/tag pairs in global-memory scratch buffers or peer-accessible buffers. **The consumer checks the tag and obtains the payload in the same load**; a successful poll returns the data for its next computation. [7]

The tag combines the decode-step counter and layer number, so HIP Graph replay can reject data from an earlier invocation. Visibility is a separate requirement: this helper uses device/agent-scope accesses for handoffs within one GPU and system-scope accesses for data written by a peer GPU. The epoch identifies which invocation produced a value; the scoped accesses make that value visible to the consumer. [3], [7]

### Combine AllReduce with the computations around it

With tensor parallelism, each GPU computes partial output channels that must be summed across ranks. M3's workgroups perform this AllReduce through peer-memory accesses inside the kernel. Reduction work advances on completed **32-channel output tiles**, across the token rows, while other workgroups produce other tiles. Residual updates can use those reduced values, but **RMSNorm still needs the complete hidden row** before routing can use the normalized input. MoE reduction similarly combines output tiles at the end of the layer. [3]

M3 specializes both expert computation and communication as row count grows:

| Token rows | Expert computation | Attention and MoE AllReduce |
| --- | --- | --- |
| 1–4 | Tasks organized by token and expert slot | Push partial output tiles to peers; each rank accumulates them. |
| 5–11 | Tokens grouped by distinct expert | Same direct peer exchange. |
| 12–16 | Tokens grouped by distinct expert | Reduce-scatter to each output tile's owner, then all-gather the sums. |

At 12–16 rows, **the owner is assigned by output tile: `owner = tile_id % TP`**. Each owner sums that 32-channel tile for all tokens, then distributes the sums to the other ranks. This partitions the hidden-channel dimension. It reduces bytes per link at the cost of an extra communication hop. [3], [11]

### Split the long-context index scan across GPUs

Fusing kernels reduces launch and synchronization overhead. The indexer still has to scan the cache to find the blocks attention should read. At a one-million-token context, this is substantial work. **Indexer context parallelism (CP)** divides that scan across GPUs. [1]

In the TP path, each rank scans the cache for its own index-query head. With CP, each rank handles **roughly one quarter of the scored key blocks, interleaved across the context, for all four heads**. A wavefront loads a key block once and reuses it across heads and query rows. The head × query combinations fill the MFMA columns: four heads and four query rows use all 16 columns in one pass. The same useful scores are evaluated, with fewer repeated key loads and better use of the MFMA columns. [3]

Each head's owning rank receives candidates and completes selection. The extra index-query projections run only when a step contains a request past **288 index blocks**, approximately 36K tokens, where CP activates. This threshold accounts for the extra projections and candidate exchange. [1], [3]

![MiniMax-M3 indexer context parallelism across four TP ranks](images/minimax-indexer-cp.svg)

*Figure 4. For each head, four ranks contribute two partitions each. Each partition returns its top 16 candidates; the owning rank merges 128 candidates into the final selection of 16 blocks.*

The CP layer harness isolates the indexer change: at 1M context, 16 rows, and four query tokens per request, HIP Graph replay latency falls from **222.2 to 156.7 µs**, a **1.42× speedup**. Short-context changes stay within the PR's reported ±0.9 µs. The [CP layer-harness CSV](data/minimax-indexer-cp-layer.csv) records this separate measurement. [1]

Mono reorganizes connected stages; CP improves key-data reuse in the long-context scan. The results below compare those path configurations.

ATOM selects the row-count variant and supplies cache metadata for each eligible decode step; unsupported steps use its separate-kernel path. **M3 mono defaults on for eligible TP4 configurations.** The stock recipe enables the shared index cache, which disables mono, so using mono requires **`use_index_cache: false`**.[^m3-support] [3], [4]

## DeepSeek-V4.1: adapt the fusion boundary to the model

DeepSeek-V4.1 has state updates between groups of operations. Its [runner](https://github.com/ROCm/ATOM/blob/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono/runner.py) keeps Engram before attention preparation where present, and a compressor before index scoring on selecting layers. Around those calls, it uses **two mono launches per rank: K1 and K2**. [13]

[K1](https://github.com/ROCm/ATOM/blob/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono/kernels/attn_pre.py) fuses normalization, Q/KV and index-query projections, RoPE, and window-cache insertion. [K2](https://github.com/ROCm/ATOM/blob/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono/kernels/layer_post.py) joins **K2a—index selection, sparse attention, output projections, attention reduction, and normalization—with K2b—routing, expert GEMMs, and FFN reduction—in one launch**. The normalized attention output becomes the MoE input inside that kernel. [13]

K2 prioritizes the current layer's dependency: the MoE needs the normalized attention output, while the mHC residual-mixing gate outputs are used by the next layer. Publishing the norm result first lets this layer's MoE preparation advance before those gates run. In the token-parallel configuration, V4.1 assigns owners by token row, so each rank normalizes its own rows and exchanges them for MoE computation. [9], [13]

![DeepSeek-V4.1 keeps state updates around K1 and K2, with attention and experts fused in K2](images/deepseek-v41-mono-flow.svg)

*Figure 5. Two mono launches preserve the original state-update calls while keeping attention and experts together inside K2.*

M3 can fuse a whole sparse layer. V4.1 keeps the compressor between K1 and K2, preserving its cache updates while removing launches between the attention and expert stages inside K2. The model's dependencies determine which operations belong in each fused region.

## Performance results

### Scope and methodology

Separate-versus-mono results compare **complete implementation paths**, so they do not isolate launch reduction, prefetch, or communication overlap. Speedup is **baseline / mono for latency** and **mono / baseline for throughput or interactivity**. M3's CP result adds an indexer optimization; Kimi-K3 and GLM-5.2 from open ATOM PR #2435 remain **in progress**.[^benchmark-data]

### M3: decode latency with mono and indexer CP

The server measurements use MI355X, TP4, MiniMax-M3-MXFP4, a 1M context, and EAGLE3 with three speculative tokens. Batches of 1, 2, and 4 produce 4, 8, and 16 target-model rows before graph padding. The [decode-step CSV](data/minimax-decode-step.csv) retains the reported values. **Separate** and **Mono** are the two execution paths within ATOM. Latencies are in milliseconds; Gain lists mono / mono + CP speedups. [1]

| Batch | Separate | Mono | Mono + CP | Gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.47 | 7.00 | 6.26 | **1.64× / 1.83×** |
| 2 | 13.40 | 9.50 | 7.77 | **1.41× / 1.72×** |
| 4 | 18.47 | 15.88 | 12.36 | **1.16× / 1.49×** |

### Why the gain changes with batch size

At batch one, M3 mono saves **4.47 ms** and CP saves a further **0.74 ms**. At batch four, mono saves **2.59 ms** and CP saves another **3.52 ms**. More rows spread launch overhead across more useful work, consistent with mono's smaller speedup at batch four. At wider shapes, expert grouping reuses weights across token rows, and CP targets repeated key loads in the index scan.

Mono is most useful when a layer has many short, dependent operations. A resident grid must fit on the GPU, and a larger fused region does not automatically run faster. Use the actual decode widths to choose the fusion boundary and keep the separate path for steps outside that range. [3], [5]

### One-launch layer results: GLM-5

[GLM-5's full-indexer mono kernel](https://github.com/ROCm/FlyDSL/blob/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel/glm/kernel.py) combines indexer, sparse MLA, router, MXFP4 MoE, TP reductions, and residual update in **one launch per rank**. FlyDSL #1204 reports these results on MI355X TP8 at position 3000, against ATOM's separate-kernel layer implementation: [5]

| Token rows | Mono launches per layer and rank | Layer latency, separate → mono (µs) | Speedup |
| ---: | ---: | ---: | ---: |
| 4 | **1** | 205.8–213.7 → **78.4–78.6** | **2.62–2.73×** |
| 8 | **1** | 223.4 → **118.2–118.3** | **1.89×** |

### Results across models

All four models below use MI355X. Each ratio uses the metric in its row; the [comparison CSV](data/mono-benefits.csv) records the full test configuration.

| Model, engine, and test | Measured metric | Speedup |
| --- | --- | ---: |
| **MiniMax-M3**: sparse layer; ATOM TP4, batch 1, 1M context | GPU decode step: 11.47 → 7.00 ms (6.26 with CP) | **1.64×** (**1.83×** with CP) [1] |
| **Kimi-K3**: MoE tail; ATOM TP8, concurrency 1 | Output throughput: 169.15 → 197.92 tokens/s | **1.17×** [8] |
| **GLM-5.2**: decoder layer; ATOM TP4, concurrency 1 | Agentic throughput: +15.92% | **1.16×** [8] |
| **DeepSeek-V4.1-Flash**: standard layer; vLLM TP4, six rows, ratio-1 harness | Layer latency: 180 → 94 µs | **1.91×** [14] |

Kimi-K3 keeps production attention, KDA, and `AttnRes`; the comparison changes only the MoE tail. GLM-5.2 also reports **1.17× throughput** at concurrency two. The [comparison CSV](data/mono-benefits.csv) includes the additional throughput and time-per-output-token measurements. [8]

The approach has also landed in **vLLM**: merged PRs [#60397](https://github.com/vllm-project/vllm/pull/60397) and [#61082](https://github.com/vllm-project/vllm/pull/61082) bring ATOM-derived FlyDSL mono kernels to DeepSeek-V4.1-Flash, with the separate-path layer-latency improvement shown above. [14], [15]

The ATOM V4.1 schedule also continues to improve: #2545 reports **4% / 10% / 11% / 12% lower decode-step time** at batches 1/2/4/8, with bit-identical outputs, relative to the previous mono schedule. This is the combined result of updates including token parallelism, handoff prefetch, and mHC seam ordering; individual contributions were not isolated. [9]

## Quality results

The merged vLLM implementation also has model-level quality evidence. For **DeepSeek-V4.1-Flash**, PR #61082 compares mono on and off on the same branch, using MI355X TP2, four seeds per mode, thinking enabled, and DSpark with block rejection. The reported pass@1 scores are: [15]

| Benchmark | Mono off (%) | Mono on (%) |
| --- | ---: | ---: |
| GPQA-Diamond | 85.0 ± 2.2 | **85.0 ± 2.2** |
| AIME25 | 98.3 ± 1.2 | **100.0 ± 0.0** |

Both modes also score **100% on all nine RULER NIAH settings**: `single-1`, `multikey-2`, and `multikey-3` at 32K, 128K, and 256K context. Each setting uses 40 samples with greedy decoding, for 360 samples per mode. [15]

M3's source PR reports **bit-identical index selection** over 3K–1.04M contexts, **bit-identical outputs through 16 chained layers with CP off versus on** at the tested 3K, 37K, and 200K contexts, and a successfully served **1,040,186-token request**. [1] The ATOM serving paths' GSM8K results are recorded in the notes.[^quality]

## Build a mono path for your model

Start with the [FlyDSL mono components](https://github.com/ROCm/FlyDSL/tree/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel) and [ATOM's shared mono framework](https://github.com/ROCm/ATOM/tree/5e8d9131739aaa53836713c79e6d547c23980f03/atom/mono). A useful workflow is:

1. **Choose a connected region.** Profile the separate path at the decode row counts you care about. Map projections, cache operations, attention, experts, and reductions; identify which inputs each stage actually waits for.
2. **Compose and schedule it.** Reuse GEMM, quantization, mailbox, and AllReduce pieces. Give workgroups enough independent work to issue weight loads early and pass completed data to consumers. Specialize task placement and LDS layouts for each supported width.
3. **Validate it in the engine.** Compare intermediate and layer outputs with the separate modules, using tools such as `ATOM_MONO_CHECK`. Exercise HIP Graph replay and all participating ranks, then measure layer and serving performance. Add dispatch conditions and a fallback alongside the kernel.

There is concrete work to build on: the open Kimi-K3 and GLM-5.2 paths extend model coverage, and a [proposed gfx942 port](https://github.com/vllm-project/vllm/pull/60731) targets MI300X and MI325X, with measurements on MI325X. That port identifies larger decode widths and additional TP sizes as follow-ups. [8], [16]

Bring a model, a profile, or a measured schedule improvement to [FlyDSL](https://github.com/ROCm/FlyDSL) or [ATOM](https://github.com/ROCm/ATOM). The reusable pieces are available; the next step is to find which waits your model can remove.

[^benchmark-data]: Results are reported in the linked PRs. The [comparison CSV](data/mono-benefits.csv) and M3 CSVs retain source configurations and reported values. PR status was checked on October 14, 2026.

[^m3-support]: At ATOM #2419, M3 mono requires TP4, DP=PP=1, FP8 KV/index caches with 128-token blocks, `max_model_len` ≤ 1,048,576, eager or non-piecewise graphs, no TBO, and no ATOM vLLM/SGLang plugin mode. Eligible decode steps have 1–16 rows including graph padding and matching cache metadata. The [example launch script](data/run-minimax-m3-mono.sh) adapts the pinned recipe for four MI355X GPUs, an ATOM installation containing #2419 with its FlyDSL/AITER dependencies, and access to the M3 MXFP4 and EAGLE3 checkpoints. `ATOM_MONO_CHECK` provides per-layer comparisons; `ATOM_MONO_TRACE` and `ATOM_MONO_TIMELINE` expose step data and timing during development. [1], [3], [12]

[^quality]: M3's GSM8K five-shot scores are 0.9447 for ATOM's separate-kernel path, 0.9371/0.9424 for mono, and 0.9378 for mono plus CP. The two mono runs span 0.53 percentage points. PR #2435 reports Kimi-K3 mono/separate-kernel scores of 0.959060/0.959818 and GLM-5.2 mono/separate-kernel scores of 0.9606/0.9674. [1], [8]

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
15. [vLLM PR #61082: merged whole-layer coverage, GPQA-Diamond, AIME25, and RULER NIAH results](https://github.com/vllm-project/vllm/pull/61082).
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
