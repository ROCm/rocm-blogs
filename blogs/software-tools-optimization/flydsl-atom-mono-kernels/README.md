---
blogpost: true
blog_title: "Mono Kernels on AMD Instinct: Faster LLM Decode with FlyDSL and ATOM"
date: 12 Oct 2026
author: ""
thumbnail: 'mono-overlap.png'
tags: AI/ML, Performance
category: Software tools & optimizations
target_audience: GPU kernel developers and LLM inference engineers
key_value_propositions: Explains how mono kernels reduce launch overhead in low-concurrency decode, with M3 and DeepSeek-V4.1 examples, measured gains, and comparisons with ATOM's separate-kernel path.
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

# Mono Kernels on AMD Instinct: Faster LLM Decode with FlyDSL and ATOM

Interactive chat and coding assistants need to respond quickly even when only one or a few requests are active. During decode, the model repeatedly processes a small number of token rows through every layer. Each step can involve many small GPU kernels for projections, normalization, cache updates, attention, routing, and expert computation.

At low concurrency, many of these kernels do only a small amount of work, but each still has to be launched and coordinated with the next one. Repeating that sequence across layers and decode steps can make kernel launch overhead a substantial part of latency. **A fast decode path needs to reduce the overhead between these small operations.**

**A mono kernel puts a layer's connected operations into one GPU kernel launch.** It can also load weights before they are needed and combine GPU-to-GPU reductions with the computations around them. This article explains **[ATOM](https://github.com/ROCm/ATOM)'s mono-kernel path, implemented with [FlyDSL](https://github.com/ROCm/FlyDSL)**: how it fuses MiniMax-M3 decode on AMD Instinct MI355X, how the approach extends to DeepSeek-V4.1, and how it improves on ATOM's separate-kernel path.

## Why many small kernels slow down decode

Consider one sparse decoder layer. The input passes through normalization and Q/K/V projections, then cache updates, index selection, and attention. Attention is followed by an output projection, expert routing, and expert computation. When the layer is split across multiple GPUs, reductions must combine their partial results along the way. [3]

A staged implementation runs these operations as separate kernels. For a small decode batch, the useful work in each kernel may be short, so launch and synchronization costs become more noticeable. HIP graph replay reduces CPU submission overhead, but the GPU still executes the separate kernels and their dependencies.

Separate stages also leave opportunities for overlap. The next GEMM needs both weights and an input produced by an earlier stage. Its weight loads could start while that earlier stage is still computing. Similarly, a reduction could begin for one completed part of an output while other parts are still being produced. Mono puts these operations in one program so the kernel author can express that overlap directly.

The related ROCm posts on [low-latency GEMMs](https://rocm.blogs.amd.com/software-tools-optimization/accelerating-llm-inference-on-amd-gpus-with-low-latency-gemms/README.html) and [ATOM's Two-Batch Overlap](https://rocm.blogs.amd.com/software-tools-optimization/atom-optimiztion/README.html) improve work inside GEMMs and overlap computation with communication across micro-batches. Here, the focus is **the many small operations within one decoder layer**.

## Mono kernels: several stages, one launch

A mono kernel launches a group of thread blocks that stay active through several stages of the layer. This is the resident grid. A block can finish an attention task and later perform routing or expert work. Blocks pass intermediate results through GPU memory that other blocks can read, and start dependent work when its inputs are ready. Registers and Local Data Share (LDS), the GPU's on-chip scratchpad, hold local working data. [3], [7]

There are three main benefits:

| What costs time | What changes inside mono |
| --- | --- |
| Launching a separate kernel for each stage | The same thread blocks run several stages in one launch. |
| Waiting to start the next stage's weight loads | A block can load those weights while earlier computation is still running. |
| Waiting between computation and GPU-to-GPU reduction | Blocks produce, reduce, and consume parts of an output within the same program. |

![Staged launches versus a resident mono grid with early loads and tile-level handoffs](images/mono-overlap.svg)

*Figure 1. Mono combines stages in one launch, starts independent loads early, and overlaps reductions with computation. Bars show execution order, not measured durations.*

**ATOM uses FlyDSL to implement its mono kernels.** FlyDSL is a Python domain-specific language for GPU kernels. The Python code controls tensor layouts, matrix instructions, memory loads, and synchronization, and an MLIR compiler pipeline turns it into GPU code. ATOM (AiTer Optimized Model) loads the model, manages requests and caches, and selects the kernel variant for the current decode step. It also handles graph replay, speculative decoding, and execution across GPUs. [2], [4]

The design is inspired by the **TileRT MonoKernel work**. FlyDSL #1204 cites the TileRT InferenceMAX/InferenceX submission at [snapshot `8ac9834`](https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9), associated with [#3389](https://github.com/SemiAnalysisAI/InferenceX/pull/3389). FlyDSL's [kernels/monokernel/](https://github.com/ROCm/FlyDSL/tree/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel) supplies reusable components such as gfx950 GEMM, MXFP8 linear, and tagged symmetric AllReduce. ATOM's shared `atom/mono` framework connects these components to each model. [5], [6], [10]

## MiniMax-M3: one launch for a sparse decoder layer

M3's [mono kernel](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/post_attn.py) covers the complete sparse-layer path, including attention and mixture-of-experts (MoE) AllReduce, in **one launch per tensor-parallel (TP) rank**, with one rank on each participating GPU. The first three dense layers use ATOM's separate-kernel modules. The kernel specializes for 1–16 token rows; in the EAGLE3 configuration used here, three speculative tokens produce four target-model rows per request. [1], [3]

### Which operations are fused?

![MiniMax-M3 sparse-layer stages inside one FlyDSL launch](images/minimax-mono-flow.svg)

*Figure 2. M3's sparse-layer dependency chain. The source calls the pre-attention stages K1 and the layer kernel K4; K1 is fused into K4.*

**Prepare queries, keys, and values.** The [pre-attention stages](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/pre_attn.py) normalize and quantize inputs to per-token FP8, then use FP8 matrix instructions for the Q/K/V and index projections. Head normalization, rotary position embedding (RoPE), and cache insertion consume the projection outputs inside the same program. [3]

**Select the cache blocks and run sparse attention.** M3's [index scorer](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/index_score.py) assigns one 128-key block of a request to a wave. Speculative query rows occupy matrix-instruction columns, so the wave reuses a key load across those rows. Only a block that may contain newly inserted keys waits for that insertion. Selection chooses 16 blocks of 128 tokens, and sparse attention processes them in eight partitions of 256 keys. Scoring, selection, and attention stay connected within the launch. [3]

**Project and reduce the attention output.** Partition merging and FP8 quantization prepare the output-projection operands. Projection tasks produce TP partials; the attention reduction combines them and feeds the residual update and normalization needed by the router. This connects the attention result directly to the input of the expert computation. [3]

**Route tokens through the MoE.** The router selects four routed experts alongside a shared expert. Up/gate projections, activation, down projections, and MoE reduction complete the layer inside the same kernel. Narrow steps use BF16 activations with MXFP4 expert weights. Wider steps use MXFP8 activations and group tokens by expert to reuse weights. The final reduction publishes the inputs for the next layer. [3]

All of these operations stay in the same launch. To make that useful, M3 needs to keep blocks busy, load weights early, and tell each consumer when its input is ready.

### Reuse blocks as the layer advances

M3 launches **256 workgroups of 512 threads**, targeting one active workgroup per MI355X compute unit. Each block runs its assigned tasks in order. A block waiting for data can rely on the producing blocks remaining active. This avoids stopping the entire grid at every stage boundary, although threads within a workgroup still use local barriers. [3]

Blocks that perform attention can take on routing work later in the same launch. FlyDSL assigns those jobs before execution, so the same group of blocks carries the layer forward without a new launch for each stage. [11]

**FlyDSL generates a kernel variant for each row count from 1 to 16.** Python code assigns tasks to blocks and lays out their LDS storage. At four rows, routed up/gate tasks are grouped by token so a block can reuse one staged activation. At wider shapes, tokens sharing an expert become matrix columns, reusing that expert's weights. The row count determines the layout before the kernel runs. Actual token positions and selected experts are supplied at runtime. [3], [11]

### Load weights before the next stage needs them

The weights for a later stage can be loaded before that stage receives its input. In the narrow MoE path, a block starts loading its down-projection weights while its up/gate computation runs. Shared-expert tasks are assigned according to each block's routed-work load; where that load is balanced, shared-expert computation runs while the first routed task's weights are in flight. [3], [11]

FlyDSL also controls how activations and scales are laid out in LDS. At 16 token rows, the unpadded layout causes **16-way LDS bank conflicts costing about 3 µs per up/gate task**, according to the source comment. Adding 16 bytes of padding to each row spreads accesses across memory banks and removes that source of stalls. The [M3 layout code](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/layout.py) makes this adjustment alongside the expert task assignments. [11]

### Tell each block when its input is ready

After loading its weights, a block still needs the previous stage's output. M3 uses mailboxes in GPU memory: each value has a tag identifying its decode step and layer. A block waits until the tags show that its input is ready. Advancing the step counter prevents graph replay from mistaking an old result for a new one. Mailbox memory accesses make writes visible to the blocks or peer GPUs that need them. The [mailbox implementation](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/common.py) supplies this synchronization inside the fused kernel. [3], [7]

The mailbox layout also affects performance. M3 makes eight copies of the router's 1 KB score payload and spreads **240 expert blocks** across those copies. This reduces contention when many blocks wait for routing results at once. The source records **K4 on four GPUs improving from 37.3 to 35.9 µs** with the eight copies. [11]

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

*Figure 3. For each head, four ranks contribute two partitions each. Each partition returns its top 16 candidates; the owning rank merges 128 candidates into the final selection of 16 blocks.*

Mono reduces the cost of running the layer as many separate stages. CP reduces the long-context scanning work on each GPU. The performance section measures their contributions separately.

## DeepSeek-V4.1: fuse attention and experts in one kernel

DeepSeek-V4.1 includes extra state-update operations: Engram runs before attention preparation where present, and selecting layers run a compressor before index scoring. The [#2545 runner](https://github.com/ROCm/ATOM/blob/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono/runner.py) uses **two mono launches per rank, K1 and K2**, with the original module calls around them. [13]

[K1](https://github.com/ROCm/ATOM/blob/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono/kernels/attn_pre.py) prepares attention: mHC residual-stream mixing, normalization, FP8 Q/KV projections, index-query preparation, Q/KV normalization, RoPE, and window-cache insertion. After the compressor on selecting layers, [K2](https://github.com/ROCm/ATOM/blob/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono/kernels/layer_post.py) runs index scoring, sparse attention, output projections, expert computation, and both reductions. Its **K2a attention stages and K2b MoE stages share one launch** in normal execution. [13]

![DeepSeek-V4.1 fusion boundaries: K1, compressor, and K2 with attention and MoE](images/deepseek-v41-mono-flow.svg)

*Figure 4. V4.1 preserves the original module calls around two mono regions. K2a and K2b are parts of one K2 launch; check mode can run them separately for comparison.*

### Fuse index selection with attention and GEMMs

Selecting layers choose up to **512 compressed-cache rows**, which sparse attention combines with the token's local window. With the FP4 index cache enabled, the [index scorer](https://github.com/ROCm/ATOM/blob/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono/kernels/index_score_fp4.py) reuses each key load across up to six query rows sharing a request's cache view, with two 64-key steps in flight per wave. Scoring and top-k merging publish the rows needed by sparse attention within K2a. [13]

Attention then feeds inverse RoPE and group-wise FP8 quantization, followed by the two output projections, **`wo_a` and `wo_b`**. Their intermediate quantization remains explicit. Attention reduction and normalization prepare the next operand: a BF16 token row for the router and expert computation. [13]

### Carry the attention result into the MoE

Normalization publishes that BF16 row and a ready flag. The [K2b MoE stages](https://github.com/ROCm/ATOM/blob/5e8d9131739aaa53836713c79e6d547c23980f03/atom/models/deepseek_v41/mono/kernels/moe.py) consume it to route each token to **six of 384 experts**, plus a shared expert. Routed up/gate and down projections use MXFP4 weights with MXFP8 activations; the shared expert uses FP8 projections. Work is organized around the union of selected experts, preserving reuse across tokens. Activation, intermediate quantization, down projection, routing weights, and shared-expert combination all stay in K2. [13]

The main benefit is that **the attention result becomes the expert input inside the same kernel**. Attention reduction and FFN reduction also run in this launch. V4.1 runs FFN normalization before the mHC gate because MoE needs the normalized input immediately, while the gate prepares the next layer's mixing. Output-projection weights and residual inputs load before their mailbox waits. [13]

### Let each GPU finish work for its own token rows

For eligible target-model steps with **S ≥ 12 and S divisible by TP**, the attention and FFN reductions become reduce-scatter operations. Each rank owns a consecutive group of `S / TP` token rows; at S=16 and TP4, that is four rows per rank. The owner sums TP partials in the existing rank order and performs the corresponding residual, mixing, and normalization work. Later stages exchange the representations they need: K1 distributes quantized query latents, while K2 gathers BF16 normalized rows before the MoE. [9], [13]

In both models, an output can pass to its next computation inside the same kernel, including when it needs a GPU-to-GPU reduction first. M3 fuses the whole sparse layer. V4.1 uses K1 and K2 around its state updates, with attention and experts fused inside K2.

## Performance results

### Scope and methodology

The serving comparisons measure **ATOM's separate-kernel path versus its FlyDSL mono-kernel path**. GLM-5 is measured separately in a layer harness against ATOM's separate-kernel implementation. GPU decode-step time, layer latency, and serving throughput measure different scopes; M3's 1.83× result combines mono and indexer CP. Kimi-K3 and GLM-5.2 results from the open ATOM PR #2435 remain **in progress**.[^benchmark-data]

### M3: decode latency with mono and indexer CP

The server measurements use MI355X, TP4, MiniMax-M3-MXFP4, a 1M context, and EAGLE3 with three speculative tokens. Batches of 1, 2, and 4 produce 4, 8, and 16 target-model rows before graph padding. Latencies are median GPU decode-step times in milliseconds; gains list mono / mono + CP speedups relative to ATOM's separate-kernel path. In the table, **Separate** is ATOM with separate kernels and **Mono** is ATOM with FlyDSL mono kernels. [1]

| Batch | Separate | Mono | Mono + CP | Gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.47 | 7.00 | 6.26 | 1.64× / 1.83× |
| 2 | 13.40 | 9.50 | 7.77 | 1.41× / 1.72× |
| 4 | 18.47 | 15.88 | 12.36 | 1.16× / 1.49× |

![MiniMax-M3 decode-step latency comparison](images/minimax-decode-step.png)

*Figure 5. M3 decode-step latency within ATOM: separate kernels, FlyDSL mono kernels, and mono with indexer CP. Lower is better. The [decode-step CSV](data/minimax-decode-step.csv) retains the reported values.*

At batch one, mono saves **4.47 ms** and CP saves a further **0.74 ms**. At batch four, mono saves **2.59 ms** and CP saves another **3.52 ms**. With more rows, each launch processes more useful work, so its overhead is spread across that work. This is consistent with the smaller mono speedup at batch four. Weight loads, expert computation, and index scoring then account for more of the work; grouping tokens by expert and splitting the index scan across GPUs target those costs.

The CP layer harness isolates the indexer change: at 1M context, 16 rows, and four query tokens per request, graph-replay latency falls from **222.2 to 156.7 µs**, a **1.42× speedup**. Short-context changes stay within the PR's reported ±0.9 µs. The [CP layer-harness CSV](data/minimax-indexer-cp-layer.csv) records this separate measurement. [1]

### One-launch layer results: GLM-5

[GLM-5's full-indexer mono kernel](https://github.com/ROCm/FlyDSL/blob/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel/glm/kernel.py) provides another complete-layer example: indexer, sparse MLA, router, MXFP4 MoE, TP reductions, and residual update in **one launch per rank**. FlyDSL #1204 reports the following on MI355X TP8 at position 3000, using ATOM's separate-kernel layer implementation as the baseline: [5]

| Token rows | Mono launches per layer and rank | Layer latency, separate → mono (µs) | Speedup |
| ---: | ---: | ---: | ---: |
| 4 | **1** | 205.8–213.7 → **78.4–78.6** | **2.62–2.73×** |
| 8 | **1** | 223.4 → **118.2–118.3** | **1.89×** |

### Results across models

The reported improvements extend to several models, with different groups of operations fused in each:

| Model and mono path | Comparison and configuration | Reported benefit |
| --- | --- | --- |
| **MiniMax-M3**: fused sparse layer | ATOM: separate kernels → mono (FlyDSL); MI355X, TP4, MXFP4, 1M context, batch 1, EAGLE3 with three speculative tokens | **1.64× faster GPU decode step**, or **1.83× with indexer CP**. [1] |
| **Kimi-K3**: fused MoE tail | ATOM: separate kernels (reference) → mono (FlyDSL); MI355X, TP8, concurrency 1, `DCP1`, DSpark7, graph mode | Output throughput **169.15 → 197.92 tokens/s (+17.01%)**; time per output token (TPOT) **5.55–5.60 → 4.70 ms**. [8] |
| **GLM-5.2**: decoder-layer mono | ATOM: separate kernels (production) → mono (FlyDSL); MI355X, TP4, MXFP4, FP8 KV, MTP4, graph mode | At concurrency 1, Agentic throughput **+15.92%**, TPOT **6.897 → 5.380 ms**. At concurrency 2, throughput **+16.8%**, TPOT **26.2% lower**. [8] |
| **GLM-5**: full-indexer mono | ATOM separate kernels → FlyDSL mono layer harness; MI355X, TP8, position 3000, S=4/8; **layer harness** | **2.62–2.73× faster at S=4**; **1.89× at S=8**. [5] |

**DeepSeek-V4.1-Flash also benefits from further work inside mono.** Its updated TP4 implementation reduces decode-step time by **4% / 10% / 11% / 12%** at batches 1/2/4/8, with bit-identical outputs, relative to its previous mono schedule. Those incremental gains come from the token-parallel sublayers and earlier independent loads described above. [9]

### Check outputs as well as speed

M3's #2419 checks report **bit-identical index selection** over 3K–1.04M contexts, **bit-identical outputs through 16 chained layers with CP off versus on** at the tested 3K, 37K, and 200K contexts, and a successfully served **1,040,186-token request**. Selection coverage includes mixed lengths across the CP threshold and speculative row counts. These checks cover cache access and intermediate results affected by the fused implementation. [1]

*Quality note: M3's GSM8K five-shot scores are 0.9447 for ATOM's separate-kernel path, 0.9371/0.9424 for mono, and 0.9378 for mono plus CP. The two mono runs span 0.53 percentage points; the 0.69-point difference between the separate-kernel path and mono plus CP is of similar magnitude. PR #2435 reports Kimi-K3 mono/separate-kernel scores of 0.959060/0.959818 and GLM-5.2 mono/separate-kernel scores of 0.9606/0.9674, a 0.68-point difference for GLM. [1], [8]*

## Takeaways

Low-concurrency decode repeats many small operations, making launch and synchronization overhead worth reducing. Mono combines connected operations in one launch, loads weights early, and runs reductions alongside the computations that use them. ATOM's FlyDSL mono kernels use these techniques to accelerate the decode path within the serving engine.

> - **Run connected stages in one launch.** M3 fuses a sparse layer; V4.1 fuses attention and experts inside K2.
> - **Use waiting time for useful work.** Load weights before their inputs arrive and overlap reductions with computation.
> - **Adapt as the workload grows.** Reuse expert weights across token rows and split long-context index scans across GPUs.

[^benchmark-data]: No new GPU benchmarks were run for this article. The [comparison CSV](data/mono-benefits.csv) records the reported metrics, baselines, configurations, and PR sources. M3 mono requires `use_index_cache: false`; the pinned stock recipe enables the shared index cache, which disables mono. The recipe and dispatch implementation document the configuration. [12], [3]

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
