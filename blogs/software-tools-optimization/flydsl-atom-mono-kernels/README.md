---
blogpost: true
blog_title: "Accelerating MiniMax-M3 Decode on AMD GPUs with FlyDSL and ATOM Mono Kernels"
date: 12 Oct 2026
author: ""
thumbnail: 'minimax-decode-step.png'
tags: AI/ML, Performance
category: Software tools & optimizations
target_audience: GPU kernel developers and LLM inference engineers
key_value_propositions: Explains MiniMax-M3 mono decode and indexer context parallelism in ATOM, using FlyDSL scheduling and synchronization on AMD Instinct MI355X, with measured 1M-context improvements from mono decode and the additional effect of indexer context parallelism.
language: English
myst:
    html_meta:
        "property=og:locale": "en_US"
        "description lang=en": "Explore MiniMax-M3 mono decode in ATOM with FlyDSL, long-context indexer parallelism, and measured MI355X decode-step and layer-level speedups."
        "keywords": "MiniMax-M3, FlyDSL, ATOM, TileRT, mono kernel, ROCm, AMD Instinct MI355X, LLM inference, decode, indexer context parallelism"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
---

# Accelerating MiniMax-M3 Decode on AMD GPUs with FlyDSL and ATOM Mono Kernels

MiniMax-M3 decode at a one-million-token context puts two costs on the critical path: moving a small batch through many GPU operators, and scoring a large index cache to find the attention inputs. ATOM addresses both with **FlyDSL mono decode** and **indexer context parallelism (CP)** on AMD Instinct MI355X GPUs.

[ATOM PR #2419](https://github.com/ROCm/ATOM/pull/2419) reports a batch-one median GPU decode step of **11.47 ms on the original path, 7.00 ms with mono decode, and 6.26 ms with mono plus indexer CP**. That is **39.0% lower latency from mono alone** and **45.4% lower latency with both optimizations**, measured with MiniMax-M3-MXFP4, TP4, a 1M context, and EAGLE3 with three speculative tokens. [1]

This article follows the M3 implementation from the language and execution model through block scheduling, long-context selection, and ATOM integration. The ATOM PR reports the original path, mono decode, and mono plus indexer CP separately, so the measurements distinguish the benefit of mono from the additional effect of CP.

## MiniMax-M3: the measured decode improvement

The PR reports these median GPU decode-step times on **MI355X, TP4, MiniMax-M3-MXFP4, 1M context, and EAGLE3 K3**. [1]

| Request batch | Original path | Mono decode | Mono + indexer CP | Speedup, mono / mono + CP |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.47 ms | 7.00 ms | 6.26 ms | 1.64× / 1.83× |
| 2 | 13.40 ms | 9.50 ms | 7.77 ms | 1.41× / 1.72× |
| 4 | 18.47 ms | 15.88 ms | 12.36 ms | 1.16× / 1.49× |

![MiniMax-M3 decode-step latency comparison](images/minimax-decode-step.png)

*Figure 1. Median GPU decode-step latency reported in ATOM PR #2419. Lower is better. Indexer context parallelism is an additional optimization, shown separately from mono decode. Speedup is original latency divided by optimized latency.*

Mono decode combines the sparse layer's operators and communication into one launch per tensor-parallel rank. Indexer CP changes how the ranks divide long-context scoring and selection. Their contributions vary with batch size: mono alone saves 4.47 ms at batch one, while adding CP to mono saves another 3.52 ms at batch four.

A request batch is different from the number of rows processed by the target kernel. EAGLE3 K3 verifies four target-model rows per request, so batches of 1, 2, and 4 correspond to 4, 8, and 16 token rows before any graph padding. The M3 mono implementation supports decode steps of **1–16 rows**, including padded rows. [1], [3]

## FlyDSL, mono kernels, and ATOM

**FlyDSL is a Python domain-specific language for writing GPU kernels with explicit control over tensor layouts and tiling.** Its layout algebra describes how tensors map to storage and threads, while its MLIR compiler pipeline lowers the specialized program to GPU code. Developers can control data movement, matrix instructions, memory allocation, and synchronization from the same programming interface. [2]

**A mono kernel combines connected operator stages into one GPU program and one launch.** For M3, the boundary is a supported sparse decoder layer: input normalization, projections, cache insertion, index scoring and selection, attention, routing, expert computation, and the associated reductions. A model forward still executes multiple layers. [1], [3]

**ATOM (AiTer Optimized Model) is a lightweight LLM inference engine focused on AMD GPUs and integration with AITER kernels.** It supplies model loading, request processing, cache management, speculative decoding, graph replay, and distributed execution. FlyDSL supplies the GPU programming mechanisms; ATOM connects the specialized kernel to the loaded model and selects it for eligible decode steps. [4]

The design builds on the **TileRT MonoKernel work**, carried into FlyDSL and then into ATOM's M3 path. FlyDSL PR #1204 cites the TileRT GLM InferenceMAX/InferenceX submission at [snapshot `8ac9834`](https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9), associated with [submission update #3389](https://github.com/SemiAnalysisAI/InferenceX/pull/3389). M3's device primitives document their origin in the FlyDSL implementation: wave reductions, tagged mailboxes, and matrix-vector loops. The team carried the design from TileRT through FlyDSL into ATOM's production path within a week. [5], [6], [7]

### Why launch latency and dependencies matter

Small decode batches expose little work in each operator. **Kernel launch latency (KLL)** can therefore occupy a meaningful share of the critical path, even when individual matrix operations are efficient. GPU graph replay reduces host submission overhead, while separate device kernels still have execution boundaries.

The implementation also cannot rely on **Programmatic Dependent Launch (PDL)**. PDL allows a consumer kernel to begin independent work, such as loading weights, before its producer finishes, then synchronize before reading the producer's output. M3's mono kernel creates opportunities for that preparation inside one resident GPU program: different blocks advance through stage tasks and wait where inputs are consumed.

The performance results measure the combined effect of fusion, scheduling, memory access, and communication. They do not isolate separate speedups for launch latency or dependency overlap.

## Inside the M3 sparse-layer mono kernel

ATOM PR #2419 fuses the pre-attention work, called **K1** in the source, into the layer kernel, called **K4**. These names identify implementation stages; the supported sparse layer executes in one FlyDSL launch per rank. The first three dense layers use the original ATOM modules. [1], [3]

```text
Input normalization and Q/K/V + index projections
    → Head normalization, RoPE, and K/V/index-cache insertion
    → Index-block scoring and sparse selection
    → Split attention and output projection
    → Attention AllReduce, residual update, and normalization
    → Router, shared expert, and top-4 routed experts
    → MoE AllReduce and the next layer's inputs
```

Here, RoPE is rotary position embedding and MoE is mixture of experts. M3 selects **16 index blocks of 128 tokens**, covering up to 2,048 sparse keys. Split attention divides those selected keys into eight partitions of 256 keys each. Including the indexer in the fused path means the layer discovers its attention inputs as part of the same launch. [3]

![MiniMax-M3 sparse-layer stages inside one FlyDSL launch](images/minimax-mono-flow.svg)

*Figure 2. M3's sparse-layer fusion boundary. A resident grid executes the stages, with tagged handoffs between blocks and peer GPUs. The diagram shows dependencies, not a measured timeline; intermediate data can pass through memory as well as registers and LDS.*

### Assign tasks to a resident grid

The M3 configuration uses **256 workgroups of 512 threads**, targeting one resident workgroup per MI355X compute unit. Residency is part of the progress guarantee: producer blocks must be able to run while consumers poll for their results. [3]

For the split, merge, output, router, and down stages, task placement follows:

```text
block_id = (stage_base + task_id) % 256
```

The bases depend on the number of token rows. At four rows, split attention occupies blocks 0–31, its merge tasks occupy 32–35, and the 192 output-projection tasks occupy 36–227. The router starts at block 228 and wraps around to block 35. Blocks change roles as their assigned tasks finish; the schedule does not require every block to reach a global barrier between stages. [3]

The MoE schedule adapts to the amount of work. At four rows, routed up/gate tasks are grouped by token across the blocks so a block can reuse that token's staged activation. From **five rows onward**, M3 groups MoE work by distinct expert, placing the tokens that use that expert into matrix columns. This reuses an expert's weights across those tokens and avoids repeating the narrow schedule's per-token loads. [3]

FlyDSL makes those choices programmable: the Python layout code sets task placement, scratch offsets, and LDS row pitches; the kernel builder specializes the device program for each supported row count. ATOM prepares the weights, caches, peer buffers, and launch arguments around that specialization.

### Prefetch and reuse memory around dependencies

The schedule starts useful memory work before its results are needed. In the narrow MoE path, a block can prefetch its down-projection weights while its up/gate computation runs. Shared-expert placement is selected according to the blocks' routed-task load, and the schedule can overlap shared-expert work with preparation for routed experts. These are concrete scheduling decisions in M3's layer implementation. [3]

LDS, or Local Data Share, is AMD's on-chip workgroup scratchpad. M3 reuses an activation pool across stages and sizes it for the largest set of data that must be live together. In the wider MoE path, padded activation and scale rows also avoid LDS bank conflicts. The implementation can therefore change storage layout and task assignment together, rather than treating fusion as a fixed concatenation of operators. [3]

### Tagged handoffs across blocks and graph replays

M3's mailbox protocol stores each payload word alongside an epoch tag. A consumer reloads the payload and tags until all required tags match the current invocation. The implementation uses device-scope coherent accesses for local blocks and system-scope accesses for peer GPUs. [7]

```text
epoch = decode_step × 128 + layer_id + 1
```

The device-side step counter and layer identifier distinguish current results from data left by earlier layers or graph replays. ATOM owns the scratch and peer-buffer lifetimes, advances the step counter, and alternates layer-output buffers by layer parity. The cache-insertion handoff also makes newly written keys visible before the indexer consumes them. [3], [7]

### AllReduce is part of the layer schedule

M3 includes both attention and MoE AllReduce inside the fused launch. Small steps push partial rows to peers, then sum the contributions in rank order. From **12 token rows onward**, the implementation uses reduce-scatter followed by all-gather: a row's owner rank sums the partials and distributes the completed result. The source describes this as trading an extra hop for fewer bytes per link at the wider shapes. [3]

The output of attention reduction feeds the residual update, normalization, and router. MoE reduction produces the values consumed by the next layer. Keeping communication in that dependency chain lets the implementation coordinate each reduction with its producers and consumers, while preserving the expected accumulation and rounding boundaries.

## Indexer context parallelism for a 1M context

Fusion removes operator boundaries, but M3 still has to inspect a long index cache. Indexer CP redistributes that work across the four TP ranks. It activates for requests past **288 index blocks**, approximately **36K tokens**. The PR places this threshold near the measured break-even range of 258–297 blocks. [1]

Without CP, each rank scores the context for its own index-query head. With CP, each rank scores **its quarter of the context blocks for all four index heads**. That lets the rank reuse its assigned cache data across heads. Candidate exchange then returns each head's selection results to the rank responsible for that head. [1], [3]

![MiniMax-M3 indexer context parallelism across four TP ranks](images/minimax-indexer-cp.svg)

*Figure 3. Long-context scoring is partitioned by context across TP4. For each index head, eight split tasks each send their best 16 candidates to the head's rank, which merges 128 candidates and selects 16 blocks. Short requests retain the original head-partitioned scoring path.*

Each rank runs two selection partitions per head over its quarter of the context. The eight partitions across TP4 contribute **8 × 16 = 128 candidates per head**. The receiving rank merges those candidates to obtain the final top-16 block selection. The PR reports that this transformation preserves the TP path's selection bit for bit. [1], [3]

The extra index-query heads' projection runs only when a step contains a request above the CP threshold. Long-context selection is also placed in a separate cold stage inside the fused kernel, keeping its code out of the short-context hot path. These choices matter for serving, where one engine handles requests with different context lengths.

### The additional layer-level result

The ATOM PR also reports a graph-replay harness measurement at **1M context, S = 16 token rows, Q = 4 query tokens per request**: [1]

| M3 sparse-layer harness | Latency |
| --- | ---: |
| Mono, indexer CP off | 222.2 µs |
| Mono, indexer CP on | 156.7 µs |
| Reduction from indexer CP | 65.5 µs, or 29.5% |

That is a **1.42× layer speedup from adding CP to mono**. The PR reports short-context changes within **±0.9 µs**. This layer comparison isolates the CP change within the mono implementation; the earlier decode-step table measures its effect at the server's GPU-step scope. The [layer-harness CSV](data/minimax-indexer-cp-layer.csv) and [decode-step CSV](data/minimax-decode-step.csv) record the reported values and benchmark scope.

## Carrying M3 into ATOM serving

ATOM enables mono through `ATOM_MONO_ENABLE` and indexer CP through `ATOM_M3_INDEXER_CP`. The runtime checks the loaded model, TP4 configuration, FP8 KV and index caches, supported cache layout, and decode shape before selecting the path. Unsupported steps use the existing model implementation. [1], [3]

Speculative verification needs the right cache view for each token. For a request with `Q` query tokens, ATOM expands its metadata into `Q` token rows. Row `j` sees `seq_len - (Q - 1 - j)` keys, so later speculative rows can see the preceding tokens without earlier rows seeing future keys. The runner also preserves the auxiliary hidden states needed by EAGLE3. [3]

Dispatch must agree across TP ranks because the kernel performs peer communication internally. ATOM prepares allocations and peer connections before graph capture, then reuses the compiled row-count specializations and buffers during replay. This runtime work connects the fused GPU layer to a real checkpoint and request stream.

### Validation reported with the M3 results

The PR checks selection against the reference Triton selector on all four ranks over **3K–1.04M contexts**, row counts **1, 4, and 16**, query lengths **1 and 4**, and mixed request lengths across the CP threshold. It also reports bit-identical results across **16 chained layers with CP off versus on** at 3K, 37K, and 200K contexts, plus a server request containing **1,040,186 tokens**. [1]

The reported GSM8K chat five-shot scores are **0.9447 for the original path**, **0.9371 / 0.9424 for mono**, and **0.9378 for mono plus CP**. The mono-plus-CP result is 0.69 percentage points below the original path in those runs. Selection equivalence validates the CP transformation; these model-quality measurements have a broader scope and do not establish general accuracy equivalence. [1]

## Developing the next specialization

The M3 results illustrate an optimization loop that crosses kernel and engine boundaries. Start with a concrete ATOM decode shape, identify expensive operator transitions and memory access, then choose the fusion boundary and task schedule in FlyDSL. M3 combines one sparse-layer launch with row-dependent MoE scheduling and reductions; indexer CP addresses the additional cost of long contexts.

Validation should follow those same boundaries: selection and intermediate checks for the kernel, changed-input graph replay and speculative-state checks for the runtime, and model-quality evaluation for serving. Report the actual row count, context length, precision, parallel configuration, and measurement scope alongside each speedup.

For the measured batch-one, 1M-context M3 workload, **mono decode reduces median GPU decode-step latency from 11.47 ms to 7.00 ms**. Adding indexer CP reduces it further to **6.26 ms**. These ATOM comparisons connect the sparse-layer implementation to a measured decode-step improvement and show the additional benefit of distributing the long-context indexer across ranks. [1]

## References

1. [ATOM PR #2419: MiniMax-M3 mono decode, indexer CP, performance, and validation](https://github.com/ROCm/ATOM/pull/2419).
2. [FlyDSL: Python DSL, explicit layouts, and MLIR compiler stack](https://github.com/ROCm/FlyDSL).
3. [M3 mono implementation at the ATOM #2419 merge commit](https://github.com/ROCm/ATOM/tree/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono): configuration, layout, dispatch, runner, and kernels.
4. [ATOM: inference engine overview and supported features](https://github.com/ROCm/ATOM).
5. [FlyDSL PR #1204: MonoKernel design lineage and TileRT submission reference](https://github.com/ROCm/FlyDSL/pull/1204).
6. [TileRT InferenceMAX/InferenceX submission snapshot cited by FlyDSL PR #1204](https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9), associated with [submission update #3389](https://github.com/SemiAnalysisAI/InferenceX/pull/3389).
7. [M3 device primitives and tagged mailboxes at the ATOM #2419 merge commit](https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/common.py).

[1]: https://github.com/ROCm/ATOM/pull/2419
[2]: https://github.com/ROCm/FlyDSL
[3]: https://github.com/ROCm/ATOM/tree/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono
[4]: https://github.com/ROCm/ATOM
[5]: https://github.com/ROCm/FlyDSL/pull/1204
[6]: https://github.com/SemiAnalysisAI/InferenceX/tree/8ac98344b038a3f2da20a565fe9b974772a67ef9
[7]: https://github.com/ROCm/ATOM/blob/2b9f3412e466fd4172a4a621fe9215a1041130c1/atom/models/minimax_m3/mono/kernels/common.py
