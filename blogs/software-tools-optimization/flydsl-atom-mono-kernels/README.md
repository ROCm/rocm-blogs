---
blogpost: true
blog_title: "FlyDSL Mono Kernels: More Responsive LLM Decode in ATOM and vLLM"
date: 12 Oct 2026
author: ""
thumbnail: 'mono-overlap.png'
tags: AI/ML, Performance
category: Software tools & optimizations
target_audience: GPU kernel developers and LLM inference engineers
key_value_propositions: Shows how FlyDSL lets developers write mono kernels that shorten the decode critical path, improving interactivity without giving up throughput, with a MiniMax-M3 case study in ATOM and merged, measured reuse in vLLM.
language: English
myst:
    html_meta:
        "property=og:locale": "en_US"
        "description lang=en": "Learn how FlyDSL mono kernels shorten the decode critical path on AMD Instinct MI355X, improving per-user interactivity in ATOM and vLLM while preserving throughput."
        "keywords": "FlyDSL, mono kernel, interactivity, AgentX, InferenceX, ATOM, vLLM, MiniMax-M3, DeepSeek-V4.1-Flash, TileRT, ROCm, AMD Instinct MI355X, LLM inference, decode"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
---

# FlyDSL Mono Kernels: More Responsive LLM Decode in ATOM and vLLM

For interactive LLM serving, throughput is only part of the objective. A coding agent must generate its next action quickly enough to keep the reasoning-and-tool loop moving: users notice how fast the agent produces its next step, calls a tool, and continues the task, not how many tokens the server produces per second. The practical goal is therefore to serve more work while preserving per-user responsiveness, not to maximize aggregate token throughput by making every user wait longer.

Throughput still has value as capacity. For interactive serving, the fair comparison is the throughput each system delivers **at the same interactivity requirement**, or the full throughput–interactivity curve. The [InferenceX AgentX](https://inferencex.semianalysis.com/agentx) benchmark uses this framing: it replays multi-turn, long-context agentic sessions and plots throughput against **interactivity**, the per-user token rate derived from P90 inter-token latency. [15], [16] Interactivity is not the whole user experience, because time to first token (TTFT) and tool execution also add to the wait. This post focuses on the part decode controls.

**This post shows how [FlyDSL](https://github.com/ROCm/FlyDSL) improves interactivity by letting developers write *mono kernels*: GPU programs that remove waits from the decode critical path and can be reused across serving frameworks.** A MiniMax-M3 case study in [ATOM](https://github.com/ROCm/ATOM) (AiTer Optimized Model) explains the mechanisms. [4] FlyDSL mono kernels are also merged into [vLLM](https://github.com/vllm-project/vllm), where DeepSeek-V4.1-Flash improves on both axes at every tested concurrency. At concurrency one on two MI355X GPUs, P90 interactivity rises from **318.1 to 429.1 tokens/s/user (1.35×)** and total throughput rises **1.16×**. That run fixes the speculative-decoding acceptance length, and TTFT does not improve at every point; the [vLLM results](#across-frameworks-deepseek-v41-flash-in-vllm) give the full conditions. [11]

![DeepSeek-V4.1-Flash in vLLM on MI355X TP2: at concurrency 1, 2, 4, and 8, FlyDSL mono raises both P90 interactivity and total throughput relative to mono off](images/vllm-dsv41-throughput-interactivity.png)

*Figure 1. Throughput–interactivity points for DeepSeek-V4.1-Flash in vLLM, mono off versus FlyDSL mono on 30 standard layers, measured on the same node. Each arrow connects the same concurrency. The AgentX replay fixes DSpark acceptance at 3.51 in both arms. [11]*

## Why every operator can be fast while the decode step is still slow

Consider one sparse decoder layer. Normalization and projections prepare queries, keys, and values. Cache updates and index selection feed attention. Output projection and reduction feed expert routing, and expert GEMMs and another reduction finish the layer. A typical DeepSeek-V4.1-Flash layer in vLLM issues **about 40 separate kernels** on this path. [3], [11]

At interactive concurrency, tuning each of those kernels is not enough. Three difficulties remain:

- **Small batches cannot amortize fixed costs.** Each decode step carries only a few token rows: the paths in this post handle 1–16 or at most 48. Launch overhead and moving intermediate data through memory are large relative to the useful math in each kernel.
- **Stages wait on the dependency chain.** Each stage can begin only after the previous stage produces its data. A faster GEMM finishes sooner, but the next stage still waits for that whole kernel to finish before it even starts loading its own weights.
- **Computation waits on communication, and communication waits on computation.** Tensor-parallel (TP) reductions, handoffs of results, and the next stage's weight reads add to the critical path when they cannot overlap with work that is already running.

HIP Graph replay removes most CPU submission overhead, which is why decode already uses it. The GPU still executes separate kernels, however, and each kernel boundary is a point where the whole stage must finish. Replacing about 40 kernels with two mono launches is a good illustration, but **the goal is to remove waits from the critical path, not merely to reduce the launch count.** A later GEMM could load its weights while the previous stage produces its input. A reduction could start on a completed tile while other tiles are still being computed. [11]

The related ROCm posts on [low-latency GEMMs](https://rocm.blogs.amd.com/software-tools-optimization/accelerating-llm-inference-on-amd-gpus-with-low-latency-gemms/README.html) and [ATOM's Two-Batch Overlap](https://rocm.blogs.amd.com/software-tools-optimization/atom-optimiztion/README.html) speed up individual GEMMs and overlap work across micro-batches. Mono kernels address the waits **inside one layer**.

## What FlyDSL provides, and who does what

A **mono kernel** launches a grid of **workgroups (thread blocks in HIP)** sized so that all of them stay resident for the layer invocation. Workgroups take on different tasks as the layer advances. A consumer waits only for the inputs it needs, while independent work proceeds elsewhere in the grid. Intermediate results move through global-memory buffers. Registers and Local Data Share (LDS), the software-managed on-chip memory local to each compute unit, hold working data. This is a form of *megakernel*, and its value comes from that scheduling rather than from the single launch. [3], [7]

![Staged launches versus a resident mono grid with early loads and tile-level handoffs](images/mono-overlap.svg)

*Figure 2. Mono combines stages in one launch, starts independent loads early, passes ready tiles forward without per-stage grid-wide barriers, and overlaps reductions with computation. Bars show execution order, not measured durations.*

Writing such a program requires control that most kernel libraries do not expose. FlyDSL is a Python domain-specific language for GPU kernels, compiled through an MLIR pipeline. With it, kernel developers can: [2]

- **Express the schedule in Python.** Task assignments, per-shape variants, and their storage layouts are generated together when the kernel is built. M3, for example, builds a variant for each row count from 1 to 16.
- **Control layout and tiling explicitly.** Tensor layouts, tile shapes, and matrix instructions are chosen per stage, so each GEMM stays specialized for its shape inside the fused program.
- **Control data movement and synchronization.** Developers decide when global-memory loads are issued, where intermediate data lives, and how producers signal consumers, both within a GPU and across peer GPUs.

These capabilities make a mono schedule practical to write, but **FlyDSL does not perform the fusion automatically.** It does not decide which stages to merge, and it does not guarantee residency or forward progress. The work divides into three parts:

| Layer | Responsibility |
| --- | --- |
| **FlyDSL** | Language and compiler; reusable components in [`kernels/monokernel/`](https://github.com/ROCm/FlyDSL/tree/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel), including gfx950 GEMM, MXFP8 linear, and tagged symmetric AllReduce. [5] |
| **Kernel developer** | The mono schedule: which stages to fuse, how tasks map to workgroups, when to prefetch, and how handoffs are signaled. Residency and forward progress follow from the resource budget and task order chosen here. |
| **Serving engine (ATOM, vLLM)** | Choosing the execution path for each step; runtime metadata, caches, and buffers; keeping all TP ranks on the same path; HIP Graph compatibility; correctness checks; and fallback to separate kernels. |

The design follows the **TileRT MonoKernel work** in the [InferenceMAX/InferenceX snapshot `8ac9834`](https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9), cited by FlyDSL #1204 and associated with [#3389](https://github.com/SemiAnalysisAI/InferenceX/pull/3389). ATOM contributes the M3 implementation and the shared [`atom/mono/`](https://github.com/ROCm/ATOM/tree/5e8d9131739aaa53836713c79e6d547c23980f03/atom/mono) device, planning, and runtime code; vLLM contributes the DeepSeek-V4.1-Flash integration in its own runtime. [5], [6], [8], [11]

## Case study: MiniMax-M3 in ATOM

M3's [mono kernel](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/post_attn.py) covers the complete sparse-layer path in **one launch per TP rank**. That path includes input projections, index scoring and selection, sparse attention, the attention AllReduce, routing, the shared and routed experts, and the MoE AllReduce. The first three dense layers keep ATOM's separate-kernel modules. [1], [3]

![MiniMax-M3 sparse-layer stages inside one FlyDSL launch](images/minimax-mono-flow.svg)

*Figure 3. M3's sparse-layer dependency chain inside one launch. Stage-by-stage details are in the [implementation notes](#appendix-m3-implementation-notes).*

Three mechanisms explain most of the value. Each one targets one of the waits described above.

### 1. Workgroups carry the layer across stages

M3 launches **256 workgroups of 512 threads**, targeting one resident workgroup per MI355X compute unit. A build-time schedule written in FlyDSL assigns each workgroup a sequence of tasks: a workgroup that performs attention can take on routing or expert work later in the same launch. No relaunch or grid-wide barrier is needed at each stage boundary, so launch and stage-boundary costs are paid once per layer rather than once per operator. [3], [9]

The grid size alone does not guarantee that this works. Forward progress depends on a resource budget that lets all workgroups be resident at the same time, and on a task order in which every wait points to earlier work. This is the developer's responsibility, and it is why the engine must check the hardware before choosing mono: vLLM, for example, rejects a mono request on a partitioned GPU with fewer than 256 compute units. [3], [11]

### 2. Weight loads overlap the preceding computation

A workgroup can request the weights for a later stage before that stage receives its input. In the narrow MoE path, **after the first up/gate task round**, the workgroup issues global-memory loads for the first token's routed down-projection weights and the shared expert's down weights. Those loads stay in flight during subsequent computation and data exchange. [3], [9]

The timing is a tuned choice. Issuing the loads too early competes with upstream loads for bandwidth and keeps vector registers occupied for longer. Issuing them too late leaves the dependency-chain wait in place. Because FlyDSL exposes when loads are issued, the developer can place this point deliberately and move it when the schedule changes.

### 3. Tagged handoffs and in-kernel reductions let consumers start early

M3's [mailbox helper](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/common.py) stores payload/tag pairs in global-memory or peer-accessible buffers. **The consumer reads the tag and the payload in the same load**, so a successful poll already returns the data for its next computation. The tag combines the decode-step counter and the layer number, so a replayed HIP Graph can reject stale data from an earlier invocation. Device-scope accesses cover handoffs within a GPU, and system-scope accesses cover data written by a peer GPU. [3], [7]

The TP AllReduce applies the same idea across GPUs. Workgroups exchange partial sums through peer memory inside the kernel and reduce completed **32-channel output tiles** while other workgroups are still producing other tiles. Residual updates can consume reduced tiles immediately; RMSNorm, which needs the complete hidden row, waits for that row rather than for a separate AllReduce kernel to finish. [3]

### Results in ATOM

The server measurements use MI355X, TP4, MiniMax-M3-MXFP4, a 1M-token context, and EAGLE3 with three speculative tokens, so batches of 1, 2, and 4 produce 4, 8, and 16 target-model rows before graph padding.[^m3-support] **Separate** and **Mono** are two complete execution paths in ATOM, so the comparison does not isolate the three mechanisms from each other. **Mono + CP** adds indexer context parallelism, a separate long-context optimization described in the [implementation notes](#appendix-m3-implementation-notes). Latencies are median GPU decode-step times in milliseconds; the [decode-step CSV](data/minimax-decode-step.csv) retains the reported values. [1]

| Batch | Separate | Mono | Mono + CP | Speedup (mono / mono + CP) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.47 | 7.00 | 6.26 | **1.64× / 1.83×** |
| 2 | 13.40 | 9.50 | 7.77 | **1.41× / 1.72×** |
| 4 | 18.47 | 15.88 | 12.36 | **1.16× / 1.49×** |

![M3 GPU decode-step latency: ATOM separate kernels 11.47 ms, FlyDSL mono 7.00 ms, FlyDSL mono plus indexer CP 6.26 ms](images/minimax-flydsl-decode-step.png)

*Figure 4. At batch one, FlyDSL mono shortens M3's GPU decode step in ATOM, and indexer CP adds a further gain at long context. [1]*

These are GPU decode-step times, not serving-level interactivity, but they show where the gain comes from. **Mono saves the most at batch one (4.47 ms) and less at batch four (2.59 ms)**, as more rows spread the fixed costs over more useful work; low-concurrency decode is exactly the regime interactive serving cares about. CP's contribution grows with batch size instead (0.74 → 3.52 ms), because it targets repeated key loads in the long-context index scan.

M3's source PR reports bit-identical index selection over 3K–1.04M contexts, bit-identical outputs through 16 chained layers with CP off versus on, and a successfully served 1,040,186-token request. GSM8K five-shot scores are 0.9447 for the separate path, 0.9371 and 0.9424 for two mono runs, and 0.9378 for mono + CP. [1]

## Across frameworks: DeepSeek-V4.1-Flash in vLLM

The second test of the approach is whether it carries beyond one engine and one model. vLLM runs DeepSeek-V4.1-Flash decode on MI355X with FlyDSL mono kernels, merged in two PRs. [11], [12]

### The fusion boundary follows the model

DeepSeek-V4.1 has state updates between groups of operations, so it does not fuse the whole layer the way M3 does. Engram, compressor, and indexer calls stay outside the fused kernels where the model needs them. Around those calls, each rank runs **two mono launches**. K1 fuses normalization, Q/KV and index-query projections, RoPE, and window-cache insertion. K2 joins index selection, sparse attention, output projection, attention reduction, normalization, routing, expert GEMMs, and FFN reduction. The normalized attention output becomes the MoE input inside K2. [11], [12]

![DeepSeek-V4.1 keeps state updates around K1 and K2, with attention and experts fused in K2](images/deepseek-v41-mono-flow.svg)

*Figure 5. Two mono launches preserve the original state-update calls while keeping attention and experts together inside K2.*

### What the engine provides

The kernels need engine support to run safely in production. In vLLM, that support covers: [11]

- **Path selection and fallback.** Decode steps that fit the kernels' row and metadata limits take the mono path. Prefill, mixed steps, PIECEWISE graph steps, and steps with more than 48 rows take the original path. If mono is requested on a deployment the kernels cannot run, vLLM raises an error instead of silently falling back.
- **Lockstep across ranks.** The kernels wait on each other's peer writes, so every TP rank must take the same path at every layer. The dispatch conditions read only metadata that all ranks hold alike.
- **Graph-safe buffers.** Handoffs carry epoch tags, so buffers need no per-step clearing or host-side step hook, and each decode width gets its own scratch space.
- **Correctness checks.** Stage-by-stage numerics against vLLM's own operators at TP2 and TP4, mono-on versus mono-off layer outputs on a five-layer model, and 100K-replay race stress tests in one HIP Graph.

On a standard layer at TP4 with six token rows and cold weights, the vLLM layer test reports **180 → 94 µs (1.91×)**. [11]

### Serving results, with their conditions

The AgentX comparison in [vLLM #60397](https://github.com/vllm-project/vllm/pull/60397), merged October 10, uses MI355X, TP2, a same-node mono-off baseline, the InferenceX recipe, and **3,600 seconds per point**. Mono covers K1 + K2 on the 30 standard layers. Both arms **fix DSpark acceptance at 3.51**. That keeps the comparison controlled, but it does not measure the complete experience under natural acceptance. [11]

| Concurrency | P90 interactivity (tokens/s/user) | Total throughput (tokens/s) | TTFT p90 (s) |
| ---: | --- | --- | --- |
| 1 | 318.1 → **429.1 (1.35×)** | 21,573 → **25,039 (1.16×)** | 1.18 → 1.28 |
| 2 | 299.3 → **381.7 (1.28×)** | 22,735 → **25,695 (1.13×)** | 1.10 → 1.08 |
| 4 | 288.0 → **332.4 (1.15×)** | 33,043 → **35,885 (1.09×)** | 1.27 → 1.26 |
| 8 | 214.6 → **245.3 (1.14×)** | 58,294 → **60,501 (1.04×)** | 1.01 → 1.08 |

At every concurrency, mono improves interactivity and throughput together: the curve in Figure 1 moves outward, so the extra throughput does not come from slower per-user token generation. **TTFT does not improve consistently**, however. It rises from 1.18 to 1.28 seconds at concurrency one and from 1.01 to 1.08 seconds at concurrency eight. Mono targets decode, so prefill-side metrics need to be tracked separately. The gain also narrows as concurrency increases, consistent with the M3 results above. On GSM8K, with actual rejection sampling, the same TP2 configuration scores 0.9742 with the original path and 0.9750 with K1 + K2, within the reported ±0.0044. [11]

[vLLM #61082](https://github.com/vllm-project/vllm/pull/61082), merged October 13, extends K1 + K2 to **all 40 backbone layers** and exposes the **`--monokernel`** switch. Its TP4 concurrency-one AgentX run reports **541.6 → 562.6 tokens/s/user at P90 (3.9%)**. That number compares **two mono versions**, not mono off versus mono on, and uses one 3,600-second run per arm with the same fixed acceptance. [12]

For this all-layer version, #61082 evaluates quality with actual rejection sampling. Comparing mono on and off on the same branch at TP2, with four seeds per mode, thinking enabled, and DSpark block rejection, it reports these pass@1 scores: [12]

| Benchmark | Mono off (%) | Mono on (%) |
| --- | ---: | ---: |
| GPQA-Diamond | 85.0 ± 2.2 | 85.0 ± 2.2 |
| AIME25 | 98.3 ± 1.2 | 100.0 ± 0.0 |

Both modes also score 100% on all nine RULER NIAH settings (`single-1`, `multikey-2`, and `multikey-3` at 32K, 128K, and 256K context; 40 greedy samples each). [12]

## Scope and costs

Mono kernels are a targeted tool, not a general replacement for separate kernels:

- **Where they help.** Mono pays off when a layer is a chain of many short, dependent operations at small row counts: interactive, low-concurrency decode. As batches grow, fixed costs are amortized anyway, and the gain shrinks, as both the M3 and vLLM results show.
- **What they constrain.** The grid must stay resident, so deployments are limited to specific GPUs, TP sizes, and decode widths. M3 mono in ATOM requires TP4 and 1–16 rows. The vLLM path supports gfx950, TP2 or TP4, and at most 48 rows. Steps outside those limits use the separate path. [1], [11]
- **What they cost.** Each model needs its own schedule, specialized per decode width and validated against the separate path in the engine. The separate path must be kept as a fallback, and changes to the model's layer must be reflected in the kernel; vLLM's layer tests are designed to fail when they are not. [11]
- **What is not yet shown.** Measured integration exists today in ATOM and vLLM. There is no SGLang integration or measurement; ATOM's M3 mono path explicitly excludes its vLLM and SGLang plugin modes. A [proposed gfx942 port](https://github.com/vllm-project/vllm/pull/60731) targets MI300X and MI325X but has not been merged. [1], [13]

## Build a mono path for your model

Start with the [FlyDSL mono components](https://github.com/ROCm/FlyDSL/tree/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel) and [ATOM's shared mono framework](https://github.com/ROCm/ATOM/tree/5e8d9131739aaa53836713c79e6d547c23980f03/atom/mono):

1. **Find the waits.** Profile the separate path at the decode widths you serve. Map which inputs each stage actually waits for and where communication sits on the critical path.
2. **Write the schedule in FlyDSL.** Reuse the GEMM, quantization, mailbox, and AllReduce components. Choose the fusion boundary from the model's dependencies, give workgroups independent work so that they can prefetch, and specialize task placement per width.
3. **Integrate it in the engine.** Add dispatch conditions and a fallback, keep ranks in lockstep, and verify HIP Graph replay. Compare against the separate path using tools such as `ATOM_MONO_CHECK`, then measure on a throughput–interactivity curve, not on throughput alone.

Bring a model, a profile, or a measured schedule improvement to [FlyDSL](https://github.com/ROCm/FlyDSL), [ATOM](https://github.com/ROCm/ATOM), or [vLLM](https://github.com/vllm-project/vllm).

## Appendix: M3 implementation notes

These details support the case study but are not needed to follow the main argument.

**Stages inside the launch.** The [pre-attention stages](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/pre_attn.py) normalize and quantize inputs to per-token FP8. They then use FP8 MFMA (matrix fused multiply-add) instructions for the Q/K/V and index projections, followed by head normalization, RoPE, and cache insertion. The [index scorer](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/index_score.py) selects 16 blocks of 128 tokens, and sparse attention processes them in eight partitions of 256 keys. Only a block that may contain newly inserted keys waits for that insertion. The router selects four routed experts alongside a shared expert. Both MoE paths use MXFP8 activations with MXFP4 weights in native scaled MFMA instructions. [3]

**Layout choices.** Two choices in the [M3 layout code](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/layout.py) show why explicit layout control matters: [9]

- At 16 token rows, one operand layout produced 16-way LDS bank conflicts. The source comment records them as costing about **3 µs per up/gate task**. Adding 16 bytes to each row's pitch removes those conflicts.
- In the source-reported four-GPU configuration, **240 expert workgroups** poll one 1 KB router-score payload. Writing eight copies spreads those reads, and the source records **37.3 → 35.9 µs**.

The M3 work also added or adopted FlyDSL primitives such as `fx.rocdl.readlane`, `ballot`, and `fx.ctpop`. [1]

**Row-count specialization.** M3 changes both expert computation and communication as the row count grows. Actual token positions and selected experts are supplied at runtime. [3], [9]

| Token rows | Expert computation | Attention and MoE AllReduce |
| --- | --- | --- |
| 1–4 | Tasks organized by token and expert slot; one staged activation reused per workgroup | Push partial output tiles to peers; each rank accumulates them. |
| 5–11 | Tokens grouped by distinct expert to reuse weights | Same direct peer exchange. |
| 12–16 | Tokens grouped by distinct expert | Reduce-scatter to each tile's owner (`owner = tile_id % TP`), then all-gather the sums. |

**Indexer context parallelism.** At a one-million-token context, scanning the index cache is substantial work, and fusion does not reduce it. With CP, each of four ranks scores **roughly one quarter of the key blocks, interleaved across the context, for all four index heads**. A wavefront loads a key block once and reuses it across heads and query rows, so four heads and four query rows fill all 16 MFMA columns in one pass. Each head's owning rank merges 128 candidates into the final 16 blocks. CP activates only when a step contains a request past **288 index blocks** (about 36K tokens), and the [CP layer-harness CSV](data/minimax-indexer-cp-layer.csv) records **222.2 → 156.7 µs (1.42×)** at a 1M-token context. [1], [3]

![MiniMax-M3 indexer context parallelism across four TP ranks](images/minimax-indexer-cp.svg)

*Figure 6. For each head, four ranks contribute two partitions each. Each partition returns its top 16 candidates; the owning rank merges 128 candidates into the final selection of 16 blocks.*

**Other models.** FlyDSL #1204 (merged) applies the same components to other models in layer harnesses on MI355X TP8 at position 3000, against ATOM's separate-kernel layers. A [GLM-5 full-indexer mono layer](https://github.com/ROCm/FlyDSL/blob/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel/glm/kernel.py) runs in one launch per rank: **205.8–213.7 → 78.4–78.6 µs (2.62–2.73×)** at four token rows and **223.4 → 118.2–118.3 µs (1.89×)** at eight. Kimi-K3 staged FlyDSL schedules, which use seven or nine launches per layer rather than one, report **1.55–1.99×** across MLA and KDA layers. A Kimi-K3 KDA + MoE serving path is still in review in ATOM #2535. Under mono it currently drafts speculative tokens from stale auxiliary features, and it has no task-level accuracy evaluation, so it is not used as evidence here. The [comparison CSV](data/mono-benefits.csv) records the layer-harness configurations. [5], [14]

[^m3-support]: At ATOM #2419, M3 mono requires TP4, DP=PP=1, FP8 KV/index caches with 128-token blocks, `max_model_len` ≤ 1,048,576, eager or non-piecewise graphs, no TBO, and no ATOM vLLM/SGLang plugin mode. The stock recipe enables the shared index cache, which disables mono, so using mono requires `use_index_cache: false`. Eligible decode steps have 1–16 rows including graph padding and matching cache metadata. The [example launch script](data/run-minimax-m3-mono.sh) adapts the pinned recipe for four MI355X GPUs. `ATOM_MONO_CHECK` provides per-layer comparisons; `ATOM_MONO_TRACE` and `ATOM_MONO_TIMELINE` expose step data and timing during development. [1], [3], [10]

## References

1. [ATOM PR #2419: MiniMax-M3 mono decode, indexer CP, performance, and validation](https://github.com/ROCm/ATOM/pull/2419).
2. [FlyDSL: Python DSL, explicit layouts, and MLIR compiler stack](https://github.com/ROCm/FlyDSL).
3. [M3 mono implementation at the ATOM #2419 merge commit](https://github.com/ROCm/ATOM/tree/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono): configuration, layout, dispatch, runner, and kernels.
4. [ATOM: inference engine overview and supported features](https://github.com/ROCm/ATOM).
5. [FlyDSL PR #1204: GLM-5 and Kimi-K3 launch counts, performance, and TileRT design lineage](https://github.com/ROCm/FlyDSL/pull/1204).
6. [TileRT InferenceMAX/InferenceX submission snapshot cited by FlyDSL PR #1204](https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9), associated with [submission update #3389](https://github.com/SemiAnalysisAI/InferenceX/pull/3389).
7. [M3 device primitives and tagged mailboxes at the ATOM #2419 merge commit](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/common.py).
8. [ATOM PR #2479: DeepSeek-V4.1 fused sublayers and shared mono framework](https://github.com/ROCm/ATOM/pull/2479).
9. [M3 task placement, LDS pitches, and row-count specialization at #2419](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/layout.py).
10. [MiniMax-M3 recipe and EAGLE3 server configuration at #2419](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/recipes/MiniMax-M3.md).
11. [vLLM PR #60397: merged DeepSeek-V4.1-Flash mono decode, launch counts, engine integration, GSM8K, and AgentX comparisons](https://github.com/vllm-project/vllm/pull/60397).
12. [vLLM PR #61082: merged whole-layer coverage, GPQA-Diamond, AIME25, and RULER NIAH results](https://github.com/vllm-project/vllm/pull/61082).
13. [vLLM PR #60731: proposed gfx942 port for MI300X and MI325X, with MI325X measurements](https://github.com/vllm-project/vllm/pull/60731).
14. [ATOM PR #2535: Kimi-K3 KDA + MoE controlled benchmarks and open review items](https://github.com/ROCm/ATOM/pull/2535).
15. [InferenceX AgentX dashboard: throughput versus interactivity for multi-turn agentic workloads](https://inferencex.semianalysis.com/agentx).
16. [InferenceX AgentX request metrics: interactivity derived from inter-token latency](https://github.com/SemiAnalysisAI/InferenceX/blob/8b25006fa0b23c7e157438996a3d2a90ff5e8cd2/inferencex-e2e/infx/results/agentic/request_metrics.py).

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
[13]: https://github.com/vllm-project/vllm/pull/60731
[14]: https://github.com/ROCm/ATOM/pull/2535
[15]: https://inferencex.semianalysis.com/agentx
[16]: https://github.com/SemiAnalysisAI/InferenceX/blob/8b25006fa0b23c7e157438996a3d2a90ff5e8cd2/inferencex-e2e/infx/results/agentic/request_metrics.py
