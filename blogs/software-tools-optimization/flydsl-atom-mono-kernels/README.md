---
blogpost: true
blog_title: "FlyDSL Mono Kernels for High-Interactivity LLM Serving in ATOM"
date: 12 Oct 2026
author: "Lingpeng Jin, Felix Li, Hattie Wu, Carlus Huang, Ziming Huang, Shijie Feng, Jun Chen, Yifeng Huang, Bin Ding, Yi Gan, Fangzhou Ai, Doug Lehr, Richard Li, Peng Sun, Emad Barsoum"
thumbnail: 'mono-overlap.png'
tags: AI/ML, Performance
category: Software tools & optimizations
target_audience: GPU kernel developers and LLM inference engineers
key_value_propositions: Explains how FlyDSL mono kernels reduce decode latency in ATOM and how mono decode improves per-user interactivity in vLLM under controlled replay.
language: English
myst:
    html_meta:
        "author": "Lingpeng Jin, Felix Li, Hattie Wu, Carlus Huang, Ziming Huang, Shijie Feng, Jun Chen, Yifeng Huang, Bin Ding, Yi Gan, Fangzhou Ai, Doug Lehr, Richard Li, Peng Sun, Emad Barsoum"
        "property=og:locale": "en_US"
        "description lang=en": "Explore how ATOM uses FlyDSL mono kernels to reduce decode latency, with reuse across model families and interactivity measurements from vLLM."
        "keywords": "FlyDSL, mono kernel, interactivity, AgentX, InferenceX, ATOM, vLLM, MiniMax-M3, DeepSeek-V4.1-Flash, TileRT, ROCm, AMD Instinct MI355X, LLM inference, decode"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_blog_authors": "Lingpeng Jin, Felix Li, Hattie Wu, Carlus Huang, Ziming Huang, Shijie Feng, Jun Chen, Yifeng Huang, Bin Ding, Yi Gan, Fangzhou Ai, Doug Lehr, Richard Li, Peng Sun, Emad Barsoum"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
---

# FlyDSL Mono Kernels for High-Interactivity LLM Serving in ATOM

**Extremely high interactivity is critical for LLM applications with strict latency requirements.** Fast token generation helps a coding agent produce its next action sooner and move on to the next step. The goal is to improve interactivity by reducing decode latency.

[FlyDSL](https://github.com/ROCm/FlyDSL) gives developers the flexibility to write and tune GPU code at the **block, wavefront, thread, and instruction levels** in Python. They can control how work is assigned, how data is laid out and moved, and where synchronization is needed. These controls let them optimize individual operations and connect the stages of a model layer into **mono kernels** that overlap independent work and reduce waits. [ATOM][4] uses these capabilities to build mono decode across several model families. The approach also extends to [vLLM](https://github.com/vllm-project/vllm). [2], [3], [8], [11], [14]

## Why mono kernels speed up decode

Small-batch decode runs many short kernels over a small number of token rows. Each launch adds overhead, and some kernels have too little parallel work to keep the GPU busy. Separate dependent kernels also create waits: the next kernel cannot start until the previous one finishes.

Mono kernels combine related operations into fewer launches and let thread blocks work across stages. Developers can schedule weight loads alongside earlier computation and let consumers start as soon as the portions of data they need are ready. They can also overlap computation with reductions across GPUs. Together, these choices reduce idle time and help the decode step finish sooner. [3], [11]

HIP Graph replay reduces CPU submission overhead, but it does not automatically remove these GPU-side dependencies. Mono complements graph replay by changing how work runs within each launch.

## What FlyDSL provides

FlyDSL gives developers four kinds of control over a GPU program: [2]

- **Work assignment.** Map tasks to thread blocks and threads, and specialize them for each decode width.
- **Layout and tiling.** Choose data layouts and tile sizes for each stage.
- **Data movement and synchronization.** Decide when to load data and how to signal that an output is ready, within a GPU or across GPUs.
- **Low-level control.** Write and tune a wide range of operations at the block, wavefront, and thread levels, with control down to individual GPU instructions. [2], [7]

In ATOM, each mono launch runs a resident grid of **thread blocks**. The blocks pass ready results to consumers and take on new tasks as the layer advances. [3], [7]

![Staged launches versus a resident mono grid with early loads and tile-level handoffs](images/mono-overlap.svg)

*Figure 1. Mono loads weights early, passes ready portions of the output forward, and overlaps reductions with computation. Bars show execution order, not measured durations.*

**Developers design the fusion.** FlyDSL supplies the language, compiler, and reusable components; it does not automatically merge stages or guarantee progress. The schedule must allow all thread blocks to remain resident and avoid waits that prevent producers from running. ATOM handles runtime buffers and metadata, coordination across tensor-parallel (TP) ranks, graph replay, correctness checks, and fallback. [3], [8]

The design draws on **TileRT's MonoKernel work**, cited by FlyDSL #1204. [5], [6]

## Bringing mono kernels to ATOM

ATOM now includes merged FlyDSL mono decode implementations for **GLM-5.2, Kimi-K3, DeepSeek-V4.1, and MiniMax-M3**. They reuse [`atom/mono/`](https://github.com/ROCm/ATOM/tree/53cbd6415a30e62421e76d67650e97e0090bbd95/atom/mono) components while adapting task assignment, fusion boundaries, and data handoffs to each model. This shows how the approach extends across model architectures. Kimi-K3 currently uses mono for its linear-attention (KDA) layers; its MLA layers keep the existing path. [1], [8], [14]

For MiniMax-M3, ATOM combines attention, mixture-of-experts (MoE) computation, and their supporting operations into **one launch per sparse layer and TP rank**. The first three dense layers keep the separate path. [1], [3]

![MiniMax-M3 sparse-layer stages inside one FlyDSL launch](images/minimax-mono-flow.svg)

*Figure 2. ATOM combines M3's sparse-layer stages in one launch. Arrows show data dependencies.*

Three scheduling choices reduce waits:

- **Reuse thread blocks across stages.** A block can finish attention work and then take on routing or expert tasks. [3], [9]
- **Load weights early.** A later stage's weights can load while earlier work is still running. [3], [9]
- **Pass on ready data.** Tagged buffers identify outputs from the current step. Consumers and cross-GPU reductions can begin on ready portions of the output without waiting for a separate kernel to finish. [3], [7]

### Decode latency in ATOM

The server measurements use MI355X, TP4, MiniMax-M3-MXFP4, a 1M-token context, and EAGLE3 with three speculative tokens, so batches of 1, 2, and 4 produce 4, 8, and 16 target-model rows before graph padding.[^m3-support] **Separate** and **Mono** are two complete execution paths in ATOM, so the comparison does not isolate the three mechanisms from each other. **Mono + CP** adds indexer context parallelism, a separate long-context optimization described in the [supporting measurements](#appendix-supporting-measurements). Latencies are median GPU decode-step times in milliseconds; the [decode-step CSV](data/minimax-decode-step.csv) retains the reported values. [1]

All times are in milliseconds; parentheses give speedup relative to Separate.

| Batch | Separate | Mono | Mono + CP |
| ---: | ---: | ---: | ---: |
| 1 | 11.47 | 7.00 (**1.64×**) | 6.26 (**1.83×**) |
| 2 | 13.40 | 9.50 (**1.41×**) | 7.77 (**1.72×**) |
| 4 | 18.47 | 15.88 (**1.16×**) | 12.36 (**1.49×**) |

![M3 GPU decode-step latency: ATOM separate kernels 11.47 ms, FlyDSL mono 7.00 ms, FlyDSL mono plus indexer CP 6.26 ms](images/minimax-flydsl-decode-step.png)

*Figure 3. At batch one, FlyDSL mono shortens M3's GPU decode step in ATOM, and indexer CP adds a further gain at long context. [1]*

**Mono saves 4.47 ms at batch one and 2.59 ms at batch four** in these measurements. The smaller relative gain at batch four is consistent with fixed costs being spread over more work, but the measurements do not establish a breakdown of the bottlenecks. The two paths also differ in kernel code, data handling, and communication. Adding CP saves a further 0.74 ms at batch one and 3.52 ms at batch four. These GPU-step results help assess the implementation; they do not directly measure per-user token rate.

M3's source PR reports bit-identical index selection against the reference over 3K–1.04M contexts and bit-identical outputs through 16 chained layers with CP off versus on at 3K, 37K, and 200K contexts. It also reports serving a 1,040,186-token request. GSM8K chat five-shot scores are 0.9447 for the separate path, 0.9371 and 0.9424 for two mono runs, and 0.9378 for mono + CP. These scores provide task-level checks, but do not establish exact numerical equivalence or unchanged quality across tasks. [1]

## Extending mono decode to vLLM

FlyDSL mono decode is also integrated into vLLM for DeepSeek-V4.1-Flash on MI355X. The implementation builds on ATOM's mono work, with integration into vLLM's execution and memory management. [8], [11], [12]

The vLLM integration uses **two mono launches, K1 and K2, per eligible layer and TP rank**. K1 prepares attention inputs; K2 handles attention, expert computation, and reductions. [11], [12]

For the 30 standard layers covered by vLLM #60397, the authors report that K1 + K2 replace **about 40 separate kernel launches per layer**. [11]

Engram runs before K1 where present. On layers that select their own cached tokens, compression, index scoring, and top-k selection remain between K1 and K2. **Two mono launches do not mean two total kernels on every layer.** The all-layer version runs compression and indexing serially there. [12]

![DeepSeek-V4.1-Flash in vLLM: Engram before K1 where present, compressor and indexer between K1 and K2 on index-source layers, attention and experts inside K2](images/deepseek-v41-mono-flow.svg)

*Figure 4. The vLLM path after #61082. Engram and the compressor/indexer calls are conditional on the layer. Index scoring and top-k selection remain outside the mono launches; K2 consumes their result. [12]*

vLLM selects eligible steps, keeps all TP ranks on the same path, and falls back for other steps. The source reports operator comparisons, multi-layer tests, and 100,000 graph replays. On a standard layer at TP4 with six token rows and cold weights, the layer test reports **180 → 94 µs (1.91×)**. [11], [12]

### Interactivity results

The [InferenceX AgentX](https://inferencex.semianalysis.com/agentx) benchmark replays multi-turn, long-context agent sessions and measures per-user interactivity. Its **P90 interactivity is 1 / P90 inter-token latency**, with latency in seconds: higher is better. The P90 label refers to the latency percentile, so this metric describes the slower end of the request distribution. AgentX also reports total throughput. [15], [16]

The AgentX comparison in [vLLM #60397](https://github.com/vllm-project/vllm/pull/60397), merged October 10, 2026, uses MI355X, TP2, a same-node mono-off baseline, the InferenceX recipe, and **3,600 seconds per point**. The four concurrency settings ran concurrently on one eight-GPU node, using two GPUs per setting. The table below compares mono off with **K1 + K2 on the 30 standard layers only**; it predates the PR's additional FFN-only path on the other 10 layers. [11]

Both runs use DSpark with five draft tokens and a **synthetic acceptance setting of 3.51**. This sets an acceptance curve instead of using actual draft/target rejection sampling; 3.51 is an acceptance-length setting, not a percentage. The replay controls how much accepted output each step delivers, but it cannot show a change in natural acceptance or establish task accuracy. In the table, arrows mean mono off → mono on. Total throughput counts input plus output tokens, including cached input context. [11], [16]

**Table units:** C is request concurrency per server; interactivity is P90 tokens/s/user; throughput is input + output tokens/s; time to first token (TTFT) is reported at p90 in seconds.

| C | Interactivity | Throughput | TTFT |
| ---: | --- | --- | --- |
| 1 | 318.1 → **429.1 (1.35×)** | 21,573 → **25,039 (1.16×)** | 1.18 → 1.28 |
| 2 | 299.3 → **381.7 (1.28×)** | 22,735 → **25,695 (1.13×)** | 1.10 → 1.08 |
| 4 | 288.0 → **332.4 (1.15×)** | 33,043 → **35,885 (1.09×)** | 1.27 → 1.26 |
| 8 | 214.6 → **245.3 (1.14×)** | 58,294 → **60,501 (1.04×)** | 1.01 → 1.08 |

![DeepSeek-V4.1-Flash in vLLM on MI355X TP2: at concurrency 1, 2, 4, and 8, FlyDSL mono raises both P90 interactivity and total throughput relative to mono off](images/vllm-dsv41-throughput-interactivity.png)

*Figure 5. Four measured throughput–interactivity points for DeepSeek-V4.1-Flash in vLLM on MI355X, TP2: mono off versus K1 + K2 on 30 standard layers. Each arrow connects runs at the same concurrency on the same node; these points do not establish the full curve. The vertical axis counts input plus output tokens. Both runs use synthetic DSpark acceptance 3.51, with 3,600 seconds per point. [11], [16]*

At concurrency one, **P90 interactivity increases from 318.1 to 429.1 tokens/s/user (1.35×)**. It also improves at the other three tested concurrency settings. Total throughput rises alongside it. These four measured points do not establish the full throughput–interactivity curve. **TTFT does not improve consistently:** it rises by 0.10 seconds at concurrency one and 0.07 seconds at concurrency eight. TTFT includes queuing, prompt processing, and other work outside the decode kernels, so it must be tracked alongside interactivity. [11]

The PR evaluates quality separately with actual rejection sampling. On GSM8K five-shot at TP2, the strict-match score is **0.9742 ± 0.0044** for the original path and **0.9750 ± 0.0043** for K1 + K2. These are scores on a specific task, not an accuracy result from the fixed-acceptance replay. [11]

[vLLM #61082](https://github.com/vllm-project/vllm/pull/61082), merged October 13, 2026, extends K1 + K2 to **all 40 backbone layers** for the supported configuration and introduces **`--monokernel`** in place of the earlier environment switch. Its TP4 concurrency-one AgentX comparison reports **541.6 → 562.6 tokens/s/user at P90 (+3.9%)**. The baseline already runs K1 + K2 on 30 layers and the FFN-only mono path on the other 10. This is an incremental comparison between two mono versions. It uses the Rust front end, one 3,600-second run per version, and synthetic DSpark acceptance 3.51; it is a different experiment from the TP2 mono-off table. [12]

For this all-layer version, #61082 evaluates quality with actual rejection sampling. Comparing mono on and off on the same branch at TP2, with four seeds per mode, thinking enabled, and DSpark block rejection, it reports these pass@1 scores: [12]

| Benchmark | Mono off (%) | Mono on (%) |
| --- | ---: | ---: |
| GPQA-Diamond | 85.0 ± 2.2 | 85.0 ± 2.2 |
| AIME25 | 98.3 ± 1.2 | 100.0 ± 0.0 |

The table retains the source's reported uncertainty values. Both modes also score 100% on all nine RULER needle-in-a-haystack (NIAH) settings (`single-1`, `multikey-2`, and `multikey-3` at 32K, 128K, and 256K context; 40 greedy samples per setting). These checks support the evaluated tasks and configurations; they do not establish unchanged accuracy for every workload. [12]

## Scope and costs

The results suggest where mono kernels are useful and what deploying them requires:

- **Where they help.** A chain of short, dependent operations at small row counts offers opportunities to remove waits. Both reported comparisons show smaller relative gains at higher concurrency or batch size. For another workload, measure whether mono reduces decode latency and improves per-user interactivity.
- **What they constrain.** Each schedule supports specific GPUs, TP sizes, and decode widths. The cited M3 path requires TP4 and 1–16 rows; the vLLM path supports gfx950, TP2 or TP4, and at most 48 rows. Ineligible steps use the separate path. An unsupported vLLM deployment with mono explicitly requested raises an error. These are necessary conditions, not a complete compatibility list. [1], [11], [12]
- **What they cost.** Each model needs its own schedule, specialized per decode width and validated against the separate path in the engine. The separate path must be kept as a fallback, and changes to the model's layer must be reflected in the kernel; vLLM's layer tests are designed to fail when they are not. [11]
- **What is not yet shown.** The sources cited here demonstrate integration in ATOM and vLLM, not SGLang. ATOM's M3 path excludes its vLLM and SGLang plugin modes; the vLLM example in this article is a separate DeepSeek integration. A [proposed gfx942 port](https://github.com/vllm-project/vllm/pull/60731) targets MI300X and MI325X and remains open as of October 16, 2026. [1], [13]

## Build a mono path for your model

Start with the [FlyDSL mono components](https://github.com/ROCm/FlyDSL/tree/31f389b3838d83f82e5bd9ccf646076243e24e18/kernels/monokernel) and [ATOM's shared mono framework](https://github.com/ROCm/ATOM/tree/53cbd6415a30e62421e76d67650e97e0090bbd95/atom/mono):

1. **Find the waits.** Profile the separate path at the decode widths you serve. Map which inputs each stage actually waits for and where communication sits on the critical path.
2. **Write the schedule in FlyDSL.** Reuse the matrix multiplication (GEMM), quantization, mailbox, and AllReduce components. Choose the fusion boundary from the model's dependencies, give thread blocks independent work so that they can prefetch, and specialize task placement per width.
3. **Integrate and evaluate.** Add dispatch conditions and a fallback, keep ranks in lockstep, and verify HIP Graph replay. Check outputs against the separate path with tools such as `ATOM_MONO_CHECK`, then run task evaluations with actual rejection sampling. Compare per-user interactivity with mono on and off at each concurrency level, and record TTFT and total throughput alongside it.

Bring a model, a profile, or a measured schedule improvement to [FlyDSL](https://github.com/ROCm/FlyDSL), [ATOM](https://github.com/ROCm/ATOM), or [vLLM](https://github.com/vllm-project/vllm).

## Appendix: Supporting measurements

**Indexer context parallelism in M3.** At long context, ATOM splits index scoring across the four TP ranks. Each rank scores roughly one quarter of the cache for all index heads, then the ranks exchange candidates to recover the same final selection. This is a separate optimization from mono fusion. [1], [3]

At 1M context, TP4, 16 token rows, and four query rows per request, the [layer-harness CSV](data/minimax-indexer-cp-layer.csv) records **222.2 → 156.7 µs (1.42×)** when CP is enabled. Both runs use mono and graph replay. CP applies beyond **288 index blocks** (about 36K tokens), based on the measured break-even point; shorter requests keep the original scorer. [1], [3]

![MiniMax-M3 indexer context parallelism across four TP ranks](images/minimax-indexer-cp.svg)

*Figure 6. Each rank scores part of the cache, then exchanges its best candidates so the owning rank can make the final selection.*

**Additional model measurements.** The [comparison CSV](data/mono-benefits.csv) retains the earlier GLM-5 and Kimi-K3 layer-harness results from FlyDSL #1204, with their configurations. These are separate from the ATOM serving integrations. [5]

GLM-5.2 and Kimi-K3 were merged into ATOM in #2535 on October 16, 2026. The PR reports no Kimi-K3 task-level accuracy evaluation, so its fixed-acceptance serving results are not used as the article's main evidence. [14]

**Data and reproduction.** The three linked CSVs retain the reported values and derived ratios. The [Figure 5 plotting script](data/plot-vllm-agentx.py) reads the comparison CSV and requires Matplotlib. All measurements and quality checks above are source-reported; this article adds no new GPU benchmark or task evaluation.

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
14. [ATOM PR #2535: merged GLM-5.2 and Kimi-K3 mono decode, shared components, and validation scope](https://github.com/ROCm/ATOM/pull/2535).
15. [InferenceX AgentX dashboard: throughput versus interactivity for multi-turn agentic workloads](https://inferencex.semianalysis.com/agentx).
16. [InferenceX AgentX request metrics: reciprocal latency for interactivity, and input plus output tokens for total throughput](https://github.com/SemiAnalysisAI/InferenceX/blob/8b25006fa0b23c7e157438996a3d2a90ff5e8cd2/inferencex-e2e/infx/results/agentic/request_metrics.py).

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
