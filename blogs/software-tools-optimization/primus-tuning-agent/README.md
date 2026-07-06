---
blogpost: true
blog_title: "Primus Tuning Agent: Closing the Configuration-Search Loop"
date: 06 Jul 2026
author: 'Anshu Raina, Peyman Razaghi, Yuankai Chen, Cheng Yao, Yao Fu, Vidushi Goyal, Devang Patel, Wen Xie, Zhenyu Gu, Emad Barsoum'
thumbnail: 'Primus-tuning-agent-thumbnail.png'
tags: AI/ML, LLM, Optimization
category: Software tools & optimizations
target_audience: distributed AI System developer, Machine learning developer
key_value_propositions: Find your best training recipe before you burn cluster time
language: English
myst:
    html_meta:
        "author": "Anshu Raina, Peyman Razaghi, Yuankai Chen, Cheng Yao, Yao Fu, Vidushi Goyal, Devang Patel, Wen Xie, Zhenyu Gu, Emad Barsoum"
        "description lang=en": "Use the Primus Tuning Agent to automatically find optimal LLM training configurations on AMD Instinct GPUs."
        "vertical": "Developers, AI, Systems"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "Open-Source Tools, ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Anshu Raina, Peyman Razaghi, Yuankai Chen, Cheng Yao, Yao Fu, Vidushi Goyal, Devang Patel, Wen Xie, Zhenyu Gu, Emad Barsoum"
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

# Primus Tuning Agent: Closing the Configuration-Search Loop

Our earlier companion blog, [Primus Projection: Estimate Memory and Performance Before You Train](https://rocm.blogs.amd.com/software-tools-optimization/primus-projection/README.html), showed how to answer two questions about a *single* training configuration before launching a run: **"Will it fit?"** and **"How fast will it be?"** But planning a large-scale training job raises a harder question that projection alone doesn't answer: **out of the thousands of legal configurations, which one should I actually run?**

This blog covers the **Primus Tuning Agent** — a tool that treats the projection engine as a fast scoring oracle and uses it to *automatically search* the joint configuration space (parallelism, micro-batch size, recomputation, pipeline schedule, MoE communication backend, and precision) for a high-throughput, memory-legal recipe. In a Mixtral 8×22B case study, the agent discovered a configuration that delivers **+27% measured throughput** over AMD's published 4-node BF16 reference — in **under 30 minutes** of single-node exploration, with no hand-written Megatron configs and no full-cluster profiling pass.

> **Ready to try it?** Jump to the [Quick Start](#quick-start) section for commands you can run immediately.

## Background

Configuring a distributed LLM training run means jointly selecting five interacting parallelism dimensions — Tensor (TP), Pipeline (PP), Expert (EP), Context (CP), and Data (DP) parallelism — alongside micro-batch size, gradient accumulation, recomputation granularity, FSDP/optimizer sharding, MoE communication backend, pipeline schedule, and precision. Each combination trades memory, compute utilization, and communication overhead differently.

The space is enormous and full of cliffs. For a Mixtral 8×22B run on 64 GPUs, the partition constraint `TP × PP × EP × CP ≤ 64` alone admits roughly 160 ordered factorizations. Layer in micro-batch size (MBS ∈ {1, …, 8}), recomputation level (none / selective / full), MoE backend (All-to-All vs. DeepEP), and pipeline algorithm (1F1B / Interleaved / Zero-Bubble / ZBV / ILP), and the search space grows to **on the order of 10⁴ topologically valid points**. Only a small fraction of those are simultaneously memory-legal *and* high-throughput — and the boundary between feasible and infeasible is razor-thin. A 4-byte miscount of activation memory per element, multiplied across 56 MoE layers, is enough to push a configuration from legal to out-of-memory (OOM), voiding hours of queued cluster time.

The traditional workflow is trial-and-error on real hardware: launch a full multi-node run, observe an OOM or disappointing throughput, tweak a knob, and repeat. Each iteration consumes hours of expensive cluster time. The Primus projection tool already collapses the per-candidate cost from *tens of minutes of multi-node profiling* to *seconds of single-node analysis*. The Tuning Agent builds directly on top of that: with a fast scoring oracle in hand, the expensive multi-node search becomes a programmatic one.

## Primus Tuning Agent

Given a model card (workload YAML), a target cluster spec, and an HBM budget, the agent returns a high-throughput, memory-legal recipe end-to-end — no hand-written configs required. It operates in **two phases — [Phase 1: deterministic seed](#phase-1-the-deterministic-seed-planner) and [Phase 2: LLM investigation](#phase-2-the-llm-investigation-loop) — over a shared trial ledger**:

| Phase                       | Driver                  | What it does                                                                                          |
| --------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------- |
| **1. Deterministic seed**   | Pure Python (no LLM)    | Sweeps single axes anchored on the user's baseline, establishing a grounded ledger of trials         |
| **2. LLM investigation**    | DSPy planner + reasoning LM | Reads the ledger and proposes *cross-axis* combinations the single-axis sweep cannot reach           |

The deterministic phase guarantees coverage of the high-leverage single-axis wins (no LLM variance), while the LLM phase spends its budget on the harder cross-axis combinations a single-axis sweep cannot reach. Both phases score every candidate through the same projection oracle — seconds per candidate instead of tens of minutes — and both write to a shared ledger guarded by a legality model and a memory pre-filter that rejects infeasible candidates before any GPU time is spent. The rest of this section walks through each piece.

### Projection as a Scoring Oracle

The agent never trains anything. For each candidate configuration it calls the Primus projection tool and reads back a small set of metrics — projected per-GPU memory, projected iteration time, and projected `tokens/s/GPU`. Because projection supports both a GPU-anchored benchmark mode and a CPU-only `simulate` mode (see the [Primus Projection blog](https://rocm.blogs.amd.com/software-tools-optimization/primus-projection/README.html)), the agent inherits the same flexibility: it can score candidates entirely analytically (no GPU), or anchor scores to a sub-node benchmark when hardware is available.

The agent exposes three evaluation tools that wrap projection, each with a different cost/fidelity trade-off:

* `evaluate_memory_only(config)` — runs only `projection memory`. The **cheapest** call and the agent's pre-filter: it answers "will this OOM?" without paying for a performance estimate, using either the analytic memory model (`simulate`) or a benchmark-anchored measured footprint (when a GPU is present).
* `evaluate_simulate(config)` — runs `projection performance --profiling-mode simulate` (Origami GEMM + Flash-Attention-v3 tile model). No GPU required.
* `evaluate_with_benchmark(config)` — runs the GPU-anchored hybrid benchmark projection. **Reserved for the top-k** because it is much slower; only available when a benchmark GPU is present.

A single, uniform result record flows back from every mode, so the search logic doesn't care which oracle produced the score:

```python
EvalResult(
    legal=True,
    reason=None,
    memory_per_gpu_gb=277.09,
    param_optimizer_gb=79.26,
    activation_gb=197.83,
    iteration_ms=...,
    tokens_per_s_per_gpu=4036.0,
    source="simulate",   # or "benchmark" | "memory_only"
    derived_dp=16,
    config={...},
)
```

### Phase 1: The Deterministic Seed Planner

Before the LLM proposes anything, the agent evaluates a small, structured set of legal configurations so the search starts *informed*. The seed planner walks single-axis sweeps, each anchored on the user-supplied baseline so that any single-axis improvement is directly attributable to the knob that changed. The exact number of seed candidates varies by model architecture and configuration (e.g. whether MoE-specific knobs apply). The order follows the optimization priority from the Primus Projection skill — high-leverage wins first, qualitative parallelism reshapes last:

1. **Baseline** — the workload's existing config (a known-working anchor and sanity check).
2. **Memory levers** — sweep `recompute_granularity ∈ {none, selective, full}` at the baseline parallelism. Workload YAMLs commonly ship with `full`-recompute defaults that cost 20–30% throughput when HBM headroom actually exists.
3. **Tier-A MoE communication** — `use_turbo_deepep` (DeepEP) and `sync_free_stage`. DeepEP can be a significant throughput optimization for MoE models, and SyncFree stage 3 improves A2A compute-communication overlap.
4. **Tier-A precision** — `fp8: hybrid`, roughly a 2× compute speedup on linear-layer GEMMs.
5. **Combined-best Tier-A** — DeepEP + SyncFree=3 + FP8 + recompute=none stacked together.
6. **Pipeline-schedule sweep** — Zero-Bubble (VPP=1) and ZBV-formatted / greedy-half (VPP=2).
7. **Baseline-neighbor sweep** — the standard manual-tuning levers: VPP × MBS × recompute around the baseline shape.
8. **DP-sharding alternatives** — `use_distributed_optimizer` (ZeRO-1-like) and `use_torch_fsdp2`.
9. **Coarse parallelism grid** — a sparse sweep over (TP, PP, EP, CP) to surface qualitatively different parallelizations.

A subtle but important detail: each coarse axis is **anchored on the baseline value**, so a configuration like "PP=4 (=baseline)" is never accidentally dropped when the legal set is reduced to a few representative points. Failing to keep the workload's own value was the single most common bug observed in practice — on 8×MI355X for Mixtral 8×22B the legal set `PP = [1, 2, 4, 7, 8, 14, 28, 56]` would collapse to `[1, 8, 56]`, making the baseline PP=4 unreachable from the grid.

The seed phase establishes a ledger of `(configuration, projected throughput, projected memory, legality)` tuples that grounds the second phase.

### Phase 2: The LLM Investigation Loop

Single-axis sweeps find the obvious wins but miss the *interactions*. The biggest gains in practice come from cross-axis combinations — e.g. MBS=1 + no-recompute + DeepEP — where dropping recompute is only safe *because* a smaller micro-batch and a more efficient MoE backend freed enough memory. Enumerating every such combination is exactly the combinatorial explosion we are trying to avoid. This is where the LLM comes in.

The loop is implemented in [DSPy](https://dspy.ai) as two cooperating modules:

* A **Planner** (`dspy.Predict` over a `TuningPlanSignature`) reads the architecture, cluster, per-axis legal sets, and seed history, then emits a hypothesis paragraph plus a JSON list of 5–8 candidate configurations to evaluate next — ordered from highest expected throughput to lowest.
* A **Reasoning LM** (`dspy.RLM` over a `TuningSearchSignature`) executes those candidates one at a time: it calls the projection oracle as a tool, reads back metrics, writes notes to a durable scratchpad, and optionally consults a fresh sub-LLM for focused architectural reasoning. It runs up to ~25 trials per round across ~3 rounds, with early stopping when no improvement appears within budget.

DSPy routes all model calls through [LiteLLM](https://docs.litellm.ai/docs/providers) internally, so any provider works without a separate proxy — set `LLM_MODEL` to a provider-prefixed name such as `openai/gpt-4o` or `anthropic/claude-opus-4-5`.

#### The Tool Belt

The reasoning LM is given a focused set of tools rather than free rein. Each call is logged, and shared state (history, budget counters, scratchpad) lives in the closure so the orchestrator can inspect it afterward:

| Tool                                | Purpose                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------ |
| `evaluate_simulate(config)`         | Analytical scoring tool (Origami + SDPA projection, no GPU required)                 |
| `evaluate_memory_only(config)`      | Cheap OOM pre-filter — confirm feasibility before paying for a performance call      |
| `evaluate_with_benchmark(config)`   | GPU-anchored projection; reserved for the top-k (gated on `has_gpu`)                 |
| `get_history(k)`                    | Compact, line-per-trial table of past results                                        |
| `get_best()`                        | The current incumbent                                                                |
| `get_legal_axes()`                  | Per-axis legal value sets — configs outside these are auto-rejected                  |
| `get_architecture()` / `get_cluster()` | The resolved model card and target cluster spec                                   |
| `get_budget_status()`               | How many evaluation calls remain                                                     |
| `note_to_scratchpad()` / `read_scratchpad()` | Durable notes carried across iterations and rounds                          |
| `query_llm(prompt)`                 | A one-shot, tool-less sub-LLM for focused reasoning ("given this MoE shape, would CP help?") |

A representative workflow for each iteration: read the scratchpad and history to get grounded, pick the next candidate from the plan that is *not* already in the ledger, evaluate it (starting with `evaluate_memory_only` if OOM is a risk), and, after a handful of evaluations, write a short note summarizing what was learned ("MBS=4 OOMs at TP=1; need recompute or TP≥2"). When the simulate budget is mostly spent, the loop proposes a polish pass over the top-k and, if a GPU is available, validates the top candidate with a benchmark.

#### The Durable Scratchpad

A plain-text scratchpad survives across RLM iterations and across rounds and is included in every prompt. The LLM uses it to record its plan, its hypotheses, and — crucially — what it has *ruled out*, so it never re-proposes a pattern it already learned was a dead end (e.g. "tried EP=16 across 2 nodes — A2A dominates"). This is what lets a multi-round search accumulate knowledge rather than restarting cold each round.

### Legality and the Memory Pre-Filter

Two guardrails keep the search both correct and cheap.

**Per-axis legality is computed in code, not left to the LLM.** From divisibility constraints and the cluster size, the agent derives the legal value set for every axis (TP, PP, EP, CP, MBS, VPP, and the legal pipeline schedules per VPP) and hands these to the model. Because the LLM can see the legal sets up front, it can't waste budget proposing an obviously illegal config and being told "no." Each candidate is still validated independently: a trial is **legal** if and only if axis validation passes (`PP × DP × CP × TP` tiles the cluster exactly), projected memory is within the cap, and the projection returns a parseable throughput.

**The memory pre-filter rejects OOM-bound candidates at zero GPU cost.** Every candidate first passes through `projection memory`. Whichever way the estimate is produced, an OOM rejection costs essentially nothing compared with the ~25 minutes a real multi-node run would burn before crashing. In the Mixtral case study below, this pre-filter rejected **7 of 30 trials** before any benchmark, saving roughly 175 GPU-minutes at zero throughput cost.

The memory estimate itself comes from one of two backends. In `simulate` mode it is fully analytical (a hierarchical, module-aligned profiler that sums parameter, optimizer, and per-microbatch activation memory). When a GPU is available, the pre-filter is instead **benchmark-anchored**: it reads a measured per-GPU memory footprint from a sub-node run — support added since the projection tool's original release — and applies a small safety margin as the OOM-decision bound. Benchmark anchoring captures real allocator high-water-marks and ROCm scratch that the analytic model can miss, so the agent's feasibility verdict tracks what the cluster will actually do far more closely. This matters precisely because the agent leans on the pre-filter so heavily: at MoE scale, where activation memory can be 2× the parameter+optimizer budget and the legal/OOM boundary is razor-thin, a more faithful memory verdict directly translates into fewer wasted trials and more aggressive (higher-throughput) configurations surviving the filter.

The objective the search maximizes is simply `arg max over legal trials of tokens/s/GPU`.

## Quick Start

### Install

```bash
python3 -m venv ~/code/Primus/.venv-agent
source ~/code/Primus/.venv-agent/bin/activate
pip install -r primus/agents/tuning_agent/requirements.txt
# Optional: only needed for --profiling-mode simulate (no-GPU scoring)
pip install git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python
```

### Configure the LLM

The agent uses DSPy, which routes LLM calls through LiteLLM — no separate proxy process is required. Set credentials for whichever provider you use:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
export LLM_MODEL=openai/gpt-4o            # default if unset

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=anthropic/claude-opus-4-5
```

You can also set these in a `.env` file or directly in the target-cluster YAML under `agent.llm`.

### Run the Agent

A dry run exercises the full loop end-to-end with synthesized metrics — useful to verify the install on a CPU-only host:

```bash
python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/agents/tuning_agent/target_cluster_mi355x_2nodes.yaml \
    --out-dir tuning_runs/dry-run \
    --dry-run --seed-only
```

Seed-only with the real projection oracle (no LLM stage), useful to see the deterministic single-axis sweep:

```bash
python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/agents/tuning_agent/target_cluster_mi355x_2nodes.yaml \
    --out-dir tuning_runs/mixtral-22b-seed \
    --seed-only
```

The full agent — deterministic seeds followed by the DSPy planner and RLM rounds:

```bash
python -m primus.agents.tuning_agent \
    --workload examples/megatron/configs/MI355X/mixtral_8x22B_v0.1-BF16-pretrain.yaml \
    --target-cluster examples/agents/tuning_agent/target_cluster_mi355x_2nodes.yaml \
    --out-dir tuning_runs/mixtral-22b-full
```

### Output Artifacts

Everything lands in `--out-dir`:

```text
trials.jsonl       all attempted configs and their results
trials.png         incumbent-vs-trial-number plot
scratchpad.txt     durable LLM notes carried across rounds
summary.json       agent-summarised winner
trials/*.yaml      one re-runnable workload-overlay YAML per trial
```

The CLI also prints the winning configuration and a ready-to-paste set of `PRIMUS_*` environment-variable exports so you can launch the real training run with the discovered recipe immediately.

## Case Study: Mixtral 8×22B (4-Node MI355X)

We applied the agent to Mixtral 8×22B (FP8, 4-node MI355X, GBS=512, seq_len=8192), grounded against AMD's published Mixtral 8×22B 4-node BF16 reference of **3,475 tok/s/GPU** (measured on the `rocm/primus:v26.2` container) on the same hardware.

**Agent search results** (all numbers are projected tok/s/GPU):

| Phase                 | Trials | Projected (tok/s/GPU) | vs. measured reference |
| --------------------- | ------ | --------------------- | ---------------------- |
| Published reference   | –      | 3,475 (measured)      | –                      |
| Seed planner only     | 12     | 4,036                 | +16.1%                 |
| LLM (seed + 3 rounds) | 30     | 4,908                 | +41.2%                 |

The deterministic seed planner reaches a projected **4,036 tok/s/GPU (+16.1%)** through single-axis sweeps alone. The LLM then jumps to a projected **4,908 tok/s/GPU (+41.2%, +21.6% over the seed planner) on its very first proposal** by combining **MBS=1, no recompute, and DeepEP** — a cross-axis combination the single-axis seed planner cannot reach.

**Hardware validation** — the top two non-redundant LLM proposals were run on the real cluster:

| Config                                | Projected | Measured | Projection Error | vs. measured reference |
| ------------------------------------- | --------- | -------- | ---------------- | ---------------------- |
| MBS=1, DeepEP                         | 4,908     | 4,402    | −10.3%           | +26.7%                 |
| MBS=1, DeepEP + distributed optimizer | 4,903     | 4,377    | −10.7%           | +25.9%                 |

The best measured configuration delivers **4,402 tok/s/GPU — +27% over the published reference**. The ~10% optimistic bias between projection and measurement is a near-constant calibration offset that *preserves rank ordering* — which is the only property the agent actually requires to pick the right winner. For context, the projection tool estimates 3,532 tok/s/GPU for the reference configuration, closely tracking the measured 3,475. The memory pre-filter rejected 7 of 30 trials at zero GPU cost, and the full workflow completed in **~30 minutes of single-node exploration**.

## Why an LLM Instead of Grid Search or Bayesian Optimization?

Classical hyperparameter optimization — grid search, or surrogate-model methods like Bayesian Optimization (BO) — works well on smooth, black-box objectives. Distributed-training configuration is neither. Three structural mismatches grow severe at LLM-training scale:

1. **The space is non-convex and constraint-ridden.** A 4-byte miscount of per-element activation memory can toggle legality. BO models the objective as a stationary Gaussian process, a poor fit for a surface riddled with hard memory cliffs.
2. **Trials have wildly different costs.** Many candidates can be rejected cheaply via the memory pre-filter — whether analytical or benchmark-anchored — before paying for a full performance evaluation, whereas BO treats every trial as an identically-priced black box.
3. **There is exploitable cross-axis structure.** Switching the MoE backend, or trading recompute against micro-batch size, has predictable directional effects a knowledgeable practitioner reasons about explicitly. The LLM exploits this structure — the +21.6% jump over the seed planner came from a single reasoned cross-axis proposal, not from blind sampling.

The two-phase design plays to each method's strength: the **deterministic planner** guarantees coverage of the high-leverage single-axis wins (no LLM variance), while the **LLM loop** spends its budget on the harder, structured, cross-axis polish.

## Assumptions and Limitations

* **Oracle fidelity.** The agent is only as good as the projection it scores against. Projections carry a small (typically <10%) error vs. measured throughput. As long as that error is a roughly constant bias, it preserves rank ordering and the agent still picks the right winner — but the projected absolute numbers should be validated on hardware before being quoted.
* **Memory estimate fidelity.** The memory pre-filter is benchmark-anchored when a GPU is available (it reads a measured per-GPU footprint) and analytic in `simulate` mode. In either case it can under-count certain effects — e.g. activation-cache high-water-marks at large batch sizes — so a config near the HBM limit may still warrant a real-run check at the margin.
* **Search is heuristic, not exhaustive.** The seed plan and LLM budget are finite; the agent finds a strong configuration, not a provably global optimum.
* **LLM variance.** The investigation loop depends on the backing model; results can vary across providers and runs. The deterministic seed floor bounds the downside.

## Tips

* **Start with `--seed-only`** to see the deterministic single-axis sweep and establish a grounded baseline before spending LLM budget.
* **Use `simulate` scoring for breadth, benchmark for the final check.** Let the agent explore analytically (no GPU), then validate the top-k with `evaluate_with_benchmark` on the hardware you have.
* **Trust the rank ordering, verify the absolute number.** The projection's calibration bias is near-constant; the agent's job is to rank, but always confirm the winner's throughput on the target cluster.
* **Re-run trials directly.** Every trial writes a self-contained workload-overlay YAML to `trials/*.yaml`, so reproducing or hand-tweaking the agent's winner is a one-liner.
* **Read the scratchpad.** `scratchpad.txt` is a human-readable record of what the agent learned and ruled out — useful both for trust and for seeding the next run.

## Summary

In this blog you explored the Primus Tuning Agent and how it closes the configuration-search loop that the [Primus projection tool](https://rocm.blogs.amd.com/software-tools-optimization/primus-projection/README.html) opens. By treating projection as a fast scoring oracle, the agent converts an expensive multi-node trial-and-error process into minutes of single-node, mostly-analytical search. Here is what you learned:

* A **deterministic seed planner** sweeps high-leverage single axes (recompute, MoE backend, precision, schedule, parallelism) anchored on the user's baseline, so every win is attributable.
* An **LLM investigation loop** (DSPy planner + reasoning LM with a focused tool belt and a durable scratchpad) proposes the cross-axis combinations that single-axis sweeps cannot reach.
* A **legality model and memory pre-filter** (analytic in `simulate` mode, benchmark-anchored when a GPU is available) keep the search correct and cheap, rejecting OOM-bound candidates at zero GPU cost.
* In validation, the agent delivered **+27% measured throughput** over a published Mixtral 8×22B reference in **~30 minutes** of single-node exploration, with no hand-written configs.

A recommended workflow: run `projection memory` to confirm feasibility, run the tuning agent in `--seed-only` mode to establish a floor, then let the full agent search for cross-axis wins — and validate its top recipe on the target cluster before committing to a full run.

## Disclaimers

The estimates and projections in this blog — including memory footprints and throughput numbers generated by the Primus projection tool and the Primus Tuning Agent — are intended for capacity planning and engineering guidance only. Results depend on hardware configuration, software versions, model settings, and workload characteristics, and may change as these evolve. These numbers are **directional** and should not be treated as official performance claims or used in external publications without independent reproduction using measurements on the target system.

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
