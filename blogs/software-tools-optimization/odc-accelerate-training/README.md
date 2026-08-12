---
blogpost: true
blog_title: "Using ODC to Accelerate AMD SFT Training"
date: "12 Aug 2026"
author: "BoTao Hu, Wen Xie, XiaoMing Peng, Yao Fu, Zhen Huang, XiaoBo Chen, Gogineni Kailash"
thumbnail: 'odc-accelerate-training-thumbnail.png'
tags: "Optimization"
category: "Software tools & optimizations"
target_audience: "This is for ML systems and performance engineers working on distributed training/large model SFT fine-tuning, especially those training with FSDP2 on AMD Instinct GPUs; and developers working on training frameworks and communication libraries (rocSHMEM/RCCL)."
key_value_propositions: "This demonstrates how to port ODC's \"on-demand point-to-point communication\" (replacing FSDP's all-gather/reduce-scatter) to AMD ROCm (rocSHMEM/MORI + Primus), eliminating layer-by-layer synchronization barriers and variable-length load imbalance bubbles. It achieves a single-machine peak speedup of approximately 1.2x and a two-machine speedup of approximately 1.15x on top of the aggregate baseline, and includes a complete and reproducible tutorial."
language: English
myst:
    html_meta:
        "author": "BoTao Hu, Wen Xie, XiaoMing Peng, Yao Fu, Zhen Huang, XiaoBo Chen, Gogineni Kailash"
        "description lang=en": "Learn how we ported ODC on-demand P2P communication to AMD Instinct MI300X to cut FSDP bubbles and speed up variable-length SFT training."
        "keywords": "ODC, On-Demand Communication, FSDP2, AMD Instinct MI300X, ROCm, rocSHMEM, Primus, distributed training, SFT fine-tuning, RCCL, point-to-point communication"
        "vertical": "HPC"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "BoTao Hu, Wen Xie, XiaoMing Peng, Yao Fu, Zhen Huang, XiaoBo Chen, Gogineni Kailash"
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

# Using ODC to Accelerate AMD SFT Training

Large-scale training spends a surprising share of its wall-clock time waiting instead of computing. Under Fully Sharded Data Parallel (FSDP), every layer ends in a collective all-gather or reduce-scatter, and every collective is a barrier that the whole data-parallel group has to reach together. Feed that machinery variable-length supervised fine-tuning (SFT) data and the picture gets worse: some ranks draw long documents while others draw short ones, so the fast ranks sit idle waiting for the slow ones.

On-Demand Communication (ODC), introduced in the ICLR 2026 paper [*On-Demand Communication for FSDP*](https://openreview.net/pdf?id=iIEEgI6WsF) and released as [sail-sg/odc](https://github.com/sail-sg/odc), attacks that waste at its root. It replaces FSDP's collectives with one-sided, on-demand point-to-point transfers, so ranks no longer have to march in lockstep. The published implementation targets NVIDIA GPUs and builds on CUDA IPC and NVSHMEM. We ported it to AMD Instinct MI300X GPUs running ROCm 7.2, rebuilt its communication substrate on XGMI with HIP IPC inside a node and on rocSHMEM/MORI across nodes, and brought up both single-node and dual-node training inside the [Primus](https://github.com/AMD-AGI/Primus) framework.

In this blog you will learn why FSDP's collectives create synchronization barriers and workload-imbalance bubbles, how ODC's two one-sided primitives remove them, what that change looks like in real PyTorch Profiler traces, and how much wall-clock time it saves on MI300X: up to 1.201× on a single node and 1.154× across two nodes, measured against a load-balanced RCCL baseline. You will also get a reproduction path so you can run the same three-way comparison on your own cluster.

If you have already read the paper, here is what this blog adds on top of it. This is a real port to AMD hardware rather than a restatement of an algorithm: you get measured numbers from MI300X, the profiler traces behind those numbers, the substrate decisions the port forced on us, and an honest account of where ODC still loses. Every speedup and trace observation below comes from an actual experiment log, and we report the shortfalls exactly as we measured them.

**The three configurations we compare throughout this blog** (all three consume the *same* variable-length data, so the comparison is apples-to-apples):

- **`NCCL_pad`** — the baseline. Standard FSDP2 with collective RCCL all-gather and reduce-scatter operations, and per-rank microbatch counts **padded** to be equal (empty "padding buckets" are added to the lighter ranks) so the collectives can run in lockstep. This is the strong, load-balanced collective baseline that ODC has to beat.
- **`ODC_pad`** — ODC's on-demand point-to-point communication (replacing the collectives), but still keeping the microbatch counts **padded** and aligned across ranks.
- **`ODC_nopad`** — ODC with **no padding**: each rank runs its true variable-length microbatch count, with no empty buckets. This is the full ODC path (it requires ODC's one-sided p2p) and it adds the load-balancing gain on top of `ODC_pad`.

## Why FSDP is Slow: Per-layer Sync Barriers and Workload-imbalance Bubbles

Standard FSDP2 (PyTorch `fully_shard`) keeps each layer's parameters sharded along the DP dimension. To process one layer it must:

- **Forward**: `all_gather` the layer's parameters to reassemble the full weights, then reshard and release them right after compute.
- **Backward**: after computing the full gradient, `reduce_scatter` it back to this rank's shard.

This collective-communication scheme carries two hidden costs.

### Per-layer Sync Barrier

- Collectives require every rank in the DP group to enter each layer in lockstep, with the same number of calls.
- Any straggler stalls the whole group inside the collective.
- Communication sits on the critical path and cannot overlap compute, so GPU cycles are spent waiting rather than computing — and the cost grows with layer count.

### Workload-imbalance Bubble

- Variable-length SFT gives each rank a different token count, so loads are naturally uneven.
- Collectives force lockstep, so light ranks idle-wait for heavy ranks — exactly the bubble the ODC paper sets out to remove.
- A pure collective baseline can only "pad" light ranks with empty buckets up to the heaviest rank's microbatch count (the `pad` path compared later), trading wasted compute for lockstep — and the waste scales with the imbalance.

Both costs stem from the **FSDP communication pattern itself**, not from insufficient bandwidth. Fixing them at the root means replacing "per-layer, per-microbatch, all-aligned" collectives with a scheme that does not force lockstep.

The official repo defines ODC as *"a patch to FSDP that adapts Parameter Server (PS) into FSDP by replacing collective all-gather and reduce-scatter with on-demand point-to-point communication."* Its direct effect is to lower synchronization frequency from per-iteration to **per-minibatch**, squeezing out FSDP's workload-imbalance bubbles.

The schematic in Figure 1 below, taken from the [paper](https://openreview.net/pdf?id=iIEEgI6WsF) and the [official repo](https://github.com/sail-sg/odc), contrasts the two schemes side by side.

```{figure} ./images/fig7_paper_barriers.png
:align: center
:alt: FSDP per-layer sync barriers compared with the ODC per-minibatch settle
Figure 1: FSDP Per-layer Barriers Versus the ODC Per-minibatch Settle
```

- **Top panel**: standard FSDP forces both devices to align at every layer; a slow device stalls the group, and the gray blocks are wasted GPU idle time.
- **Bottom panel**: ODC relaxes alignment to the end of the minibatch — each device runs forward and backward at its own pace and settles once (OS = Optimizer Step), yielding the annotated **Time Saved**. This is what we reproduce on ROCm.

Worth emphasizing: ODC's baseline is not a weak one. Its opponent is a collective version with load balancing already on (Collective + LB-Micro) — align each rank's load via packing and padding, then run standard RCCL or NCCL collectives. The `NCCL_pad` baseline used throughout this blog aligns strictly with it.

## The Core Idea of ODC: Replace Collectives with On-demand P2P

ODC is a communication-replacement patch for FSDP. It does three things, matching the official repo README and the method figure in the paper.

### From Collective Primitives to Two One-sided P2P Primitives

The paper's method figure, reproduced in Figure 2 below, shows both primitives: on the left, a gather into device 0, which pulls the shards scattered across devices (Param0/Param1) on demand and assembles them locally; on the right, a scatter-accumulate out of device 0, which pushes each gradient shard (Grad0/Grad1) one-sidedly to the owner of that shard, where it lands in a gradient accumulator (Acc).

```{figure} ./images/fig8_paper_primitives.png
:align: center
:alt: ODC's two one-sided primitives, gather and scatter-accumulate
Figure 2: ODC's Two One-sided Primitives — Gather and Scatter-accumulate
```

- **gather (fetch parameters)**: when the forward or backward pass needs a layer's full weights, it pulls them on demand from the peers holding each shard (one-sided `getmem`); no group-wide co-call is required.
- **scatter-accumulate (push gradients)**: after computing a gradient, it one-sidedly pushes (`putmem`) each shard to its owning rank (a parameter server), which asynchronously accumulates it — push and go, without waiting for the peer.

Both primitives are **one-sided**: the initiator does not require the peer to call the same collective at the same time, breaking the collective's hard constraint that call counts must match. This is precisely why ODC tolerates ranks being out of lockstep.

### From Per-iteration to Per-minibatch Synchronization

Standard FSDP reduces gradients once per microbatch, per layer. ODC instead lowers cross-rank synchronization to **once per minibatch**: microbatch gradients are accumulated locally or on the parameter server, and only at the end of the minibatch is there a single settle, ensuring all gradients have landed before the optimizer reads them.

### Overlap with Backward and a Single Settle at the End

Because pushing gradients is fire-and-forget, it can in principle overlap with the subsequent backward compute, with one unified settle at the end of the minibatch. **This is how ODC saves the bubble** — it folds the per-layer communication wait into a single wait per minibatch.

The original substrate uses CUDA IPC intra-node and NVSHMEM inter-node. Porting to AMD replaces these with their ROCm equivalents: **XGMI and HIP IPC** intra-node (direct read and write of peer memory) and **rocSHMEM or MORI** inter-node (default, GPU-initiated RDMA).

## Seeing It in the Trace

We used the PyTorch Profiler to capture real traces across single-node and dual-node systems, both NCCL and ODC backends, and both pad and nopad configurations. The figures below show, from shallow to deep, what ODC changes: first the per-layer collective barriers disappear (Figure 3 and Figure 4), then the shape difference between filling empty buckets and running variable-length microbatches (Figure 5 and Figure 6).

### Single-node 1.5B with the NCCL Baseline: Per-layer Sync Barriers

The baseline run uses the DeepSeek-R1-Distill-Qwen-1.5B model on a single node with eight GPUs, running standard FSDP2 with RCCL. The resulting trace is shown in Figure 3 below.

```{figure} ./images/fig1_nccl_perlayer_barriers.png
:align: center
:alt: Profiler trace of the NCCL baseline showing per-layer collective sync barriers
Figure 3: Single-node 1.5B NCCL Baseline — Per-layer Collective Sync Barriers
```

- A dense wall of reduce-scatter collectives, one per layer, interleaved with compute.
- Each collective is a communication alignment barrier: all ranks must align before continuing, forming a regular sawtooth of sync points.
- As the red boxes highlight, the blank stretches on the `reduce_scatter` stream are pure idle time: the faster ranks sit doing nothing, waiting for the slow rank before the per-layer `reduce_scatter` collective can proceed.

### With ODC On: Per-layer Barriers Vanish and Sync Moves to the End of the Minibatch

This run uses the same model and configuration, with only the communication backend swapped to ODC, as shown in Figure 4 below.

```{figure} ./images/fig2_odc_no_barriers.png
:align: center
:alt: Profiler trace with ODC enabled showing no per-layer collective kernels
Figure 4: Single-node 1.5B with ODC — Per-layer Collectives Replaced by One Settle per Minibatch
```

- No per-layer collectives remain; backward is a continuous run of p2p pushes and local accumulation.
- Cross-rank alignment is moved wholesale to a single settle at the end of the minibatch.
- As the figure shows, each communication is now spaced out very evenly with no synchronization waiting — the per-layer stalls are gone.
- This is the trace-level view of the shift described in [The Core Idea of ODC: Replace Collectives with On-demand P2P](#the-core-idea-of-odc-replace-collectives-with-on-demand-p2p): the barrier collapses from one per layer to one per minibatch.

### odc_pad vs odc_nopad: Empty Buckets vs Variable-length Microbatches

Both traces use the same configuration: `global_batch_size=16`, `dp=8`, so each rank gets only about two samples. With so few samples per rank, sample-length jitter cannot average out — one rank may get two long documents, another two short ones — so unequal per-rank microbatch counts are at their most visible. This is exactly the single-node gbs16 peak setting reported in [Experimental Results After ODC Adaptation](#experimental-results-after-odc-adaptation).

Figure 5 below shows `odc_pad`, which forces all ranks to the same microbatch count by padding lighter ranks with empty buckets.

```{figure} ./images/fig5_odc_pad.png
:align: center
:alt: Trace of odc_pad showing two dense forward and backward bursts
Figure 5: odc_pad — Padded Empty Buckets Add an Extra Microbatch
```

- As the figure shows, this rank runs two forward-backward samples in total — the trace's two dense bursts, the extra one being the padded empty bucket.
- The padding microbatches still consume forward and backward compute but contribute no valid gradient.
- A pure collective baseline (`NCCL_pad`) has no other option: collectives must pad to level uneven loads.

Figure 6 below shows `odc_nopad`, where ranks are allowed different microbatch counts, so each runs its true variable-length load.

```{figure} ./images/fig6_odc_nopad.png
:align: center
:alt: Trace of odc_nopad showing a single dense forward and backward burst
Figure 6: odc_nopad — Each Rank Runs Its True Variable-length Load
```

- As the figure shows, this rank runs only a single forward-backward sample — its true, smaller load, with no padding — so the trace is a single dense burst, and it does not idle-wait for a slow rank or get dragged along.
- This shape only works with ODC's one-sided p2p: collectives require every rank to call the same primitive the same number of times, so a mismatched call count hangs a cross-node collective outright.

**Takeaway:** `odc_pad` and `odc_nopad` do the same useful compute; pad additionally runs a batch of empty buckets. nopad removes that batch — savings proportional to the imbalance, largest at small gbs with strongly variable-length data — and only ODC can drive it. This is the source of the variable-length-balancing portion of the gbs16 peak in [Experimental Results After ODC Adaptation](#experimental-results-after-odc-adaptation).

## Experimental Results After ODC Adaptation

Now to the numbers. We ran the same three-way comparison — `NCCL_pad`, `ODC_pad`, and `ODC_nopad` — across five global batch sizes on two very different systems: a single node of eight GPUs training a 1.5B model, and two nodes of 16 GPUs training a 14B model. If you take away one thing from this section, make it the shape of the two curves. On a single node the speedup peaks early and then eases off as compute starts to dominate; across nodes it climbs steadily with batch size and only overtakes the baseline once the batch is large enough to amortize the cross-node cost. The tables below list every measured point, including the ones where ODC loses to the baseline.

A note on how we define the numbers: speedup = `NCCL_pad's ms/step ÷ this run's ms/step` (>1 means faster than NCCL). The baseline is the **standard RCCL collective with packing and padding enabled** (`NCCL_pad`), i.e., the "armed-to-the-teeth collective baseline" in the sense of the paper, not a weak baseline. All numbers are real values taken from each run's experiment log; the loss convergence curves of all three paths align with the NCCL baseline, with zero NaN throughout.

### Single-node 1.5B (8 GPUs, Device Path, Wall-clock Basis)

The configuration uses the DeepSeek-R1-Distill-Qwen-1.5B model on a single node with eight GPUs and intra-node XGMI and HIP IPC. The table below gives the speedup of `ODC_nopad` relative to the `NCCL_pad` baseline by gbs, along with the trend (the loss convergence of all three paths aligns with the baseline, with zero NaN throughout):

| gbs | ODC_nopad speedup | Trend interpretation |
|---|---|---|
| 8 | ≈ **0.911×** (slightly slower) | minibatch too small; the fixed overhead of p2p and settle in backward dominates the ratio and does not overlap |
| 16 | ≈ **1.201× (peak)** | two dividends stack: XGMI on-demand p2p saves collectives and variable-length balancing avoids empty buckets |
| 32 | ≈ **1.142×** | still an advantage, but compute grows and the communication and balancing dividend is diluted |
| 64 | ≈ **1.083×** | compute grows, dividend diluted, roughly on par with RCCL |
| 128 | ≈ **1.051×** | compute now fully dominates; ODC's fixed dividend is amortized away, converging with RCCL |

The single-node curve is a hump: **slightly slower at gbs8 → peak at gbs16 → slowly falling back as gbs grows, but staying stably >1.**

**Slightly slower at gbs8** (nopad 0.911×, pad 0.898×) — fixed overhead is not yet amortized:

- Per-minibatch overhead (the `barrier` plus the scatter-accumulate settle) is near-fixed, and at gbs8 the per-step compute is too small to absorb it.
- Backward's p2p pushes cannot yet overlap compute (see Figure 7), so the fixed cost is exposed on the critical path. The dividend needs a big enough batch to materialize.

**Peak at gbs16** (nopad 1.201×) — where the gain comes from:

1. **Communication side (`NCCL_pad → ODC_pad`)**: XGMI on-demand p2p replaces per-layer collectives, dropping sync from per-layer to per-minibatch and removing RCCL's group-wide alignment and its ring or tree scheduling cost.
2. **Load side (`ODC_pad → ODC_nopad`)**: variable-length microbatches skip padding buckets; at gbs16 with dp8 (about two samples per rank) the imbalance is maximal, so this dividend peaks.

Measured on the same-node ladder, the two stack cleanly: `NCCL_pad → ODC_pad` ≈ +9.8% (pure sync reduction) and `ODC_pad → ODC_nopad` ≈ +9.4% (pure load balancing), so 1.098 × 1.094 ≈ 1.201×, right on the measured peak.

**Falling back at large gbs** (gbs64 ≈ 1.083×, gbs128 ≈ 1.051×) — compute dominates:

- As gbs grows, per-step GEMM compute grows faster than the near-fixed ODC dividend, diluting it toward parity with RCCL.
- The gain never fully vanishes, which is why the curve is a hump rather than a monotonic decline.

### Dual-node 14B (16 GPUs, GDA and DEFER Path, Wall-clock Basis)

The cross-node scenario uses a 14B model on 2 × 8 GPUs, for a total of 16 GPUs, with the inter-node GDA (GPU-Direct RDMA) backend and the nopad configuration using DEFER rendezvous. The table below gives the speedup of `ODC_nopad` relative to the `NCCL_pad` baseline by gbs, along with the trend:

| gbs | ODC_nopad speedup | Trend interpretation |
|---|---|---|
| 16 | ≈ **0.796×** (clearly slower) | lacking GDRW (GPU-initiated RDMA write), must manually sync reads to preserve ordering, adding one sync per backward layer |
| 32 | ≈ **0.892×** (pad 0.844×) | gap narrows, but still behind RCCL |
| 64 | ≈ **1.120× (first overtake)** | large batch amortizes cross-node fixed overhead and variable-length bubble gains materialize |
| 128 | ≈ **1.154×** | the larger the gbs, the more expensive cross-node collectives are, and the bigger ODC's amortization gain |

The dual-node curve is the opposite of single-node: **it rises monotonically with gbs**, driven by the tension between cross-node fixed overhead and amortizable batch.

**Clearly slower at small gbs** (gbs16 ≈ 0.796×, gbs32 nopad 0.892× and pad 0.844×) — cross-node fixed overhead dominates:

- The cross-node RDMA itself is expensive (measured at ~3× the equivalent single-node kernels).
- ROCm GDA currently lacks a GDRW (GPU-initiated RDMA write), so ordering must be enforced with a manual sync read (an HDP flush, the strided-touch described in [Discussion: Settle Does Not Yet Overlap with Backward](#discussion-settle-does-not-yet-overlap-with-backward)) — adding one sync per backward layer.
- These are per-step fixed costs that gbs16 is too small to amortize.

**Monotonically rising with gbs** — fixed overhead amortizes while the imbalance dividend amplifies:

- Per-step compute grows with gbs while cross-node fixed overhead barely does, so its relative share shrinks.
- Cross-node imbalance bubbles are costlier than single-node (waits cross the network), so nopad's dividend grows too; both forces push the speedup up.

**`odc_nopad` overtakes from gbs64** (1.120×), reaching 1.154× at gbs128:

- At gbs64, nopad overtakes RCCL for the first time (odc_pad still short at 0.950×).
- At gbs128, larger batches make RCCL's cross-node collectives costlier, so both overtake (pad 1.130× and nopad 1.154×).

## Discussion: Settle Does Not Yet Overlap with Backward

The port works and it wins where the paper predicts it should, but one part of ODC's promise is not yet realized on our stack: the settle currently sits mostly on the backward critical path instead of overlapping compute. Figure 7 below captures the ODC backward pass on a single node.

```{figure} ./images/fig4_odc_settle_no_overlap.png
:align: center
:alt: ODC backward trace where the post-backward reduce work clusters instead of overlapping compute
Figure 7: ODC Backward — Settle Clusters at One Moment Instead of Overlapping Compute
```

The scatter-accumulate settle (`FSDP::post_backward_reduce`) stacks up at one moment rather than spreading across compute, so overlap is very low (measured at roughly 5%). Cross-rank sync is already per-minibatch, but the settle wait is still exposed outside compute: as Figure 7 shows, FSDP's communication is not overlapped with the kernel compute at all.

Native NCCL behaves very differently on the same workload, as Figure 8 below shows.

```{figure} ./images/fig3_nccl_backward_overlap.png
:align: center
:alt: Native NCCL backward trace where reduce-scatter overlaps the compute kernels
Figure 8: Native NCCL Backward — Reduce-scatter Overlaps Compute
```

FSDP2's prefetch fires the next layer's reduce-scatter while the current layer computes, so communication is genuinely hidden behind compute. In the red box, `reduce_scatter_base` runs in parallel with the `void ck_tile` compute over the same window — the overlap that ODC has yet to match.

Ranked by cost-effectiveness, here are the steps we plan to take:

- **Overlap settle with backward (top priority)**: build a cross-iteration software pipeline (a separate stream and events) that overlaps the previous microbatch group's settle with this group's backward compute, joining once per minibatch. This matches the paper's "overlap gradient push with backward" and should close the dual-node mid-gbs gap and lift the peak.
- **Cut and merge cross-node fixed overhead**: warm-up settle is already strided (saving ~9–10%); further, bucket and merge the small per-step collectives to reduce barrier count and squeeze the per-step fixed overhead measured on the dual-node ladder.

## Summary

FSDP's collectives produce two kinds of waste: the **waiting waste** of per-layer sync barriers and the **compute waste** of empty-bucket padding. ODC removes both — it relaxes the per-layer barrier to a single per-minibatch settle, and it uses one-sided p2p to run variable-length microbatches without any padding.

In this blog you learned how we brought that idea to AMD hardware. We ported ODC's algorithm layer — gather, scatter-accumulate, per-minibatch synchronization, and LB-Mini variable-length balancing — onto AMD Instinct MI300X through the **rocSHMEM or MORI** backend inside Primus, and you saw both single-node and dual-node runs converge correctly, with loss curves aligned to the RCCL baseline and zero NaN throughout. You also saw where the wins live: ODC pays off most with **large batches, cross-node communication, and genuinely imbalanced workloads**. On a single node the speedup peaks at ~1.201× at gbs16 and still leads at ~1.051× at gbs128; across two nodes `ODC_nopad` overtakes RCCL from gbs64 and reaches ~1.154× at gbs128. That confirms the paper's claim — removing FSDP's workload-imbalance bubbles saves real end-to-end training time — on production ROCm hardware rather than in simulation.

You also saw the parts that are not finished. ODC is slightly slower than the collective baseline at small single-node batches, it trails RCCL at the small dual-node operating points, and its settle still does not overlap backward compute. Our next step is the cross-iteration settle pipeline, together with the work to bucket and merge the small per-step collectives that make cross-node runs expensive. Both should raise the peak and pull the small-batch curve up, and we will report the measured results in a follow-up blog — so check back if you want to see how far the ROCm port can go.

## Reproducing the Results

The full hands-on reproduction guide — covering the single-node 1.5B and dual-node 14B configurations, from setting up the container and building the ODC rocSHMEM operators in Primus-Turbo to running the `nccl_pad` and `odc_nopad` arms and computing the speedup, plus a pitfalls checklist — lives in the Primus repository:

**Full reproduction guide:** [`examples/odc/README.md` in AMD-AGI/Primus](https://github.com/AMD-AGI/Primus/blob/feat/odc-consume-turbo/examples/odc/README.md)

## Additional Resources

- Paper: [*On-Demand Communication for FSDP*](https://openreview.net/pdf?id=iIEEgI6WsF) (ICLR 2026) — the motivation and method figures, the Collective LB-Micro baseline, and the scaling study showing speedup growing with device count
- Official repository: [sail-sg/odc](https://github.com/sail-sg/odc) — the FSDP patch, the gather and scatter-accumulate primitives, and the original CUDA IPC and NVSHMEM substrate
- Communication substrate: [ROCm/mori](https://github.com/ROCm/mori) — MORI-SHMEM and MORI-IR, the replacement for NVSHMEM
- Training framework: [Primus](https://github.com/AMD-AGI/Primus), with high-performance operators in [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo)
- Ported source, in-tree under the [Primus repository](https://github.com/AMD-AGI/Primus):
  - Algorithm layer (Python): `primus/core/odc/primitives/{gather,scatter_accumulate,_rocshmem_backend,shmem_triton,utils}.py`, `primus/core/odc/fsdp/{fsdp1,fsdp2}.py`, `primus/core/odc/odc_early/sitecustomize.py`, `primus/backends/megatron/patches/odc_{lb_mini,torch_fsdp2}_patches.py`
  - rocSHMEM runtime and launch scripts: `primus/core/odc/rocshmem_runtime/`, including `scripts/run_odc.sh` and its `README.md`
  - Comm operators, in the [Primus-Turbo repository](https://github.com/AMD-AGI/Primus-Turbo): `csrc/kernels/odc_rocshmem/{odc_rocshmem_host,odc_rocshmem_gda}.cu` plus the thin pybind wrapper in `csrc/pytorch/dist/` (`primus_turbo.pytorch._C.odc_rocshmem_host` and `primus_turbo.pytorch._C.odc_rocshmem_gda`)

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
