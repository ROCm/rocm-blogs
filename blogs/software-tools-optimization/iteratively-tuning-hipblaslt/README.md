---
blogpost: true
blog_title: "Iteratively Tuning hipBLASLt TensileLite Kernels: A Smaller Search, a Faster Kernel"
date: "09 Sep 2026"
author: "Yuchen Lin, Clement Lin, Chunhung Wang"
thumbnail: 'iteratively-tuning-hipblaslt-thumbnail.png'
tags: "AI/ML"
category: "Software tools & optimizations"
target_audience: "AI developers and enthusiasts"
key_value_propositions: "Tune hipBLASLt TensileLite kernels with an iterative, warm-start coordinate descent that compiles about 156 kernels per shape, beats a one-shot search over roughly ten thousand candidates in 116 of 144 GEMMs, takes a quarter of the tuning time, and cannot regress below the kernel it started from."
language: English
myst:
    html_meta:
        "author": "Yuchen Lin, Clement Lin, Chunhung Wang"
        "description lang=en": "Learn why a small iterative search beats a big one-shot sweep for hipBLASLt TensileLite kernels, in less time and with no risk of regression."
        "keywords": "LLM, Kernels, Inference, hipBLASLt, GEMM tuning, TensileLite, iterative tuning, warm-start, coordinate descent, tile expansion, AMD Instinct MI325X"
        "vertical": "AI, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Yuchen Lin, Clement Lin, Chunhung Wang"
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

# Iteratively Tuning hipBLASLt TensileLite Kernels: A Smaller Search, a Faster Kernel

A TensileLite run can only return a kernel its configuration's search space can express. That sounds like a technicality, but it is the one fact that decides whether a tuning campaign helps or hurts. The configuration is not a hint handed to an optimizer that is free to look elsewhere; it is the complete enumeration of what the run is allowed to consider, and anything outside it stays unreachable no matter how long the run is given. In a previous blog, [Reverse-Engineering hipBLASLt TensileLite Kernels](https://rocm.blogs.amd.com/artificial-intelligence/reverse-hipblaslt-tensilelite/README.html), we used that constraint defensively: decode a kernel you trust from its solution name, pin its roughly one hundred parameters as single-value `ForkParameters`, and the run cannot return anything slower, because the only kernel it can express is the one you started from. That gives tuning a floor.

A floor is not the goal, so the search space has to grow outward from it, and that is where a search goes wrong. The tempting move is to open every promising parameter at once into one grid, on the reasoning that a wider net cannot hurt. It can, in two separate ways. The first is cost: the axes multiply, so a handful of plausible values on each becomes tens of thousands of candidates to generate, compile, and time. The second is worse, because it is silent. A grid assembled from a generic template is built without reference to the kernel you already have, so nothing guarantees that kernel is inside it, and a run that cannot express your starting point has nothing holding it above that point. We measured the consequence: such a grid came back **slower than the untuned library default in 109 of 144 GEMMs**.

This post is about the other way to expand. Open one small group of parameters at a time, keep the running best inside every candidate list, and adopt a round's winner only when it is strictly faster. The search then stays cheap, because it never materializes the full product of the axes, and stays safe, because the kernel it is currently holding is always one of the candidates it is measuring against.

## The Iteration: Principle and Design

The method is a coordinate descent over groups of parameters: each round opens one group, searches it while everything else stays pinned at the running best, adopts the winner, and moves on. Coordinate descent here is a deliberate simplification rather than an attempt at optimality. The parameters do interact, so walking one group at a time is not guaranteed to land on the best point of the full product. What it buys in exchange is a cost that grows with the number of groups instead of with their product, and a trajectory in which every step is a measured comparison against the kernel already in hand. Three design choices make it work.

**Warm start.** Round 0 is the decoded kernel run as-is, so the descent begins with a long tail of roughly seventy parameters already tuned. That tail is exactly what nobody puts in a hand-built grid: load widths, scheduling and prefetch details, assertion and store options, all settled when the pool kernel was originally tuned and all invisible in the headline knobs. A cold start over a generic grid leaves them at defaults and spends its budget climbing back to where a warm start began. The climb is not reliably successful either, because those parameters interact with the tile and the split, so ground given up in the tail is not necessarily won back by opening the big axes wider.

**Group by mechanism, order by impact.** Each group collects the parameters controlling one hardware mechanism, so a round explores one coherent trade-off and stays buildable. The grouping cuts both ways. Parameters that trade directly against each other belong in the same round, because searching them separately lets the first settle on a value the second would have overturned. Parameters governing unrelated mechanisms belong in different rounds, because combining them multiplies the candidate count without adding information. The rounds then run in order of impact and dependency: tiling onto the matrix engine, splitting the contraction, placing workgroups on the chiplets, the read-side and write-side vector widths, and finally Stream-K, which must come last because it is mutually exclusive with Global-Split-K. Ordering by impact has a second benefit: the decisions that move latency most are made while the budget is intact, and the cheap tail rounds are left to refine a kernel that is already close.

**Union the incumbent, adopt only on a strict win.** Every candidate list includes the running best's own value, so the current best is always inside the search space and the round's winner is at least as fast. A round with no headroom holds instead of regressing. That makes monotonicity structural rather than lucky: it does not depend on the ranges being well chosen, on the measurement being quiet, or on the round having anything to find, because the worst available outcome is re-selecting the kernel you brought in. It is the first property to check in any implementation, and it is easy to lose by accident. Dropping the incumbent value while generating a list, or adopting a winner on a tie, is enough to turn a guaranteed floor back into a coin flip.

The cost is small and worth quantifying precisely. A full trajectory **enumerates a median of 1,126 parameter combinations** (610 to 1,814) and, after TensileLite rejects the illegal ones, **compiles a median of 156 kernels** (124 to 419). The two numbers are often conflated, and the distinction matters as soon as you have to budget a campaign: enumeration is what the configuration asks for, compilation is what actually costs build time, and the gap between them is the legality filter discarding combinations that cannot be generated for that shape and datatype. Against the one-shot grid's 10,000 enumerated candidates, this enumerates one-ninth as many and compiles fewer than one-sixtieth as many, which is why the wall-clock gap reported later is wider than the candidate counts alone would suggest.

## The Six Rounds

**1. Tile.** Open a generated `MatrixInstruction` grid, a power-of-two spread of wave-group by wave-tile decompositions, together with `DepthU`, then reseed from the winner. The tile decides how the output is partitioned across workgroups and waves and how much of the contraction is staged per iteration, so it fixes the arithmetic intensity that every later round has to work within. Generating the grid instead of drawing candidates from the pool is the whole point of the round: it reaches decompositions the pool never held. Its value depends on your pool rather than your shape, which is why it gets a section of its own later.

**2. Split-K.** Open `GlobalSplitU` *and* `GlobalSplitUAlgorithm`. Splitting the contraction across workgroups trades one pass for several partial results that then have to be reduced, which pays exactly when a shape cannot otherwise fill the device: low-tile-count and short-contraction shapes leave compute units idle, and splitting K is how you give them work. The algorithm choice between `MultipleBuffer` and `MultipleBufferSingleKernel` decides how those partials are staged and reduced, and it is a separate lever that a single pinned value gets badly wrong. Opening the count while leaving the algorithm pinned misses most of this round.

**3. Mapping.** Open `WorkGroupMapping`, `WorkGroupMappingXCC`, and `StaggerU`. These decide how workgroups land on compute units and spread across the chiplets, changing L2 locality and cross-chiplet traffic without touching the math at all. That is what makes the round cheap: the kernel body is identical, only the order in which tiles are handed out changes, and on a multi-chiplet part that order decides how much of what a workgroup needs is already resident nearby. `WorkGroupMappingXCC` must be `-1` or a power of two dividing the CU count; the 304-CU MI325X admits `{1, 2, 4, 8, 16}`, yet many hand-built grids stop at 8. Not harmless: **16 won in 31 of 144 cases**.

**4. Global-read vector widths.** Open `GlobalReadVectorWidthA`, `GlobalReadVectorWidthB`, and `LocalReadVectorWidth`, the widths of the loads feeding the matrix engine. Wider loads move more bytes per instruction and cut issue pressure, but they also constrain the addressing and alignment the kernel can use, so the width that wins is a property of the shape's leading dimensions rather than a setting that is globally better.

**5. Store and LDS vector widths.** Open `VectorWidthA`, `VectorWidthB`, `StoreVectorWidth`, and the LDS pads `LdsPadA` and `LdsPadB`, trading LDS padding against bank conflicts on the write side. Padding spends shared memory, which can cost occupancy; refusing to pad can cost a bank conflict on every access. Which way that trade falls is decided by the tile chosen four rounds earlier, which is why this round runs after the tile is settled rather than beside it.

**6. Stream-K.** A terminal branch: set `GlobalSplitU` to zero and open `StreamK` in `{1, 2, 3}`. Stream-K parallelizes the contraction across a fixed grid instead of replicating partial sums, so work is spread evenly by construction rather than by however the tiles happen to divide, which suits skinny and low-tile-count shapes where Global-Split-K leaves compute units idle. Because it replaces the data-parallel decomposition rather than refining it, it is mutually exclusive with Global-Split-K and can only be evaluated once that decomposition has been settled. Its codegen fails for some heavy-epilogue fp8 kernels, so the branch must be try-and-fallback: keep the Global-Split-K best when Stream-K yields no winner, which is what keeps it monotone-safe.

Some parameters are never opened because they are fixed by good practice or are derived rather than tunable: `1LDSBuffer`, `ClusterLocalRead`, `ScheduleIterAlg`, `SourceSwap`, and the kernel language stay constant, and quantities such as `LoopUnroll` are not tuned directly. Leaving them out is part of keeping each round small rather than an oversight. Every axis added multiplies the round it joins, so an axis earns a place only if it is both independently meaningful and genuinely free to vary.

## Experiment Setup

**Hardware and toolchain.** AMD Instinct MI325X, `gfx942`, 304 CUs, SPX / NPS1, ROCm 7.2.0. Every microsecond comes from the same TensileLite client with a frozen `GlobalParameters` block, because the client's own settings change what a given configuration measures, and a trajectory is only monotone if every stage was timed the same way. **Each case runs its whole trajectory on one GPU**, since latencies from different cards are not comparable and a card change mid-trajectory would show up as a round that appeared to win or lose for reasons the configuration cannot explain. The hipBLASLt tree is pinned to one commit throughout: TensileLite's legality rules change between revisions, enough that a config decoded under one can be rejected outright by a later one, which would silently shrink a round's candidate list rather than fail loudly.

**Problem matrix: 144 GEMMs.** 24 shapes x 3 precisions x 2 layouts (NN, TN), batch = 1. The shapes are six probes (small and large square, tall-M, wide-N, fat-K, thin-K) plus eighteen LLM-serving shapes: dense-4k and dense-8k prefill and decode projections, MoE expert projections, and a small-N family including GEMV-like cases. The split is deliberate: the probes isolate one geometric extreme each, so a result can be attributed to a shape property rather than to a model, while the LLM shapes are the ones that actually have to run fast in production and cover both the compute-bound prefill regime and the latency-bound decode regime. Precisions are `bfloat16`, `float16`, and `fp8` in and out (the `gfx942` FNUZ type). `Base` is round 0, the decoded pool kernel run as-is; `Tuned` is the kernel after all six rounds.

**Measurement integrity.** Timing the same configuration by two independent paths is worth doing, because ours disagreed. Under a parallel campaign one path intermittently returned a 10x to 100x inflated latency, and in the worst case 117x. An inflated baseline shows up as a spectacular round-1 "gain", which is exactly the kind of number a reader should not have to trust. Every latency reported here comes from a re-measured campaign in which the two paths agree to within 3.56x at worst, and the residual is the seed difference described below rather than measurement error, so **no case is excluded on measurement grounds**. If a tuning harness reports a number that looks too good, measure the same configuration by a second path before believing it.

**One stated limitation.** Each trajectory seeds from the pool's *top-ranked* candidate, which keeps the tile comparison clean but is not the strongest available start: selecting the fastest of several buildable candidates instead produces a starting kernel that is a median 1.18x faster (p90 1.66x). That choice is why the speedups below are quoted against the baseline the descent actually starts from.

## Experiment Results

Table 1 shows eight cases spanning the range of outcomes, from the largest win to one where every round held; the complete matrix is in Table 2, in the collapsible section below it. Every round column is the latency in microseconds *after* that round, so a row read left to right traces the descent and a round that found nothing repeats the previous number. Reading the rows this way is more informative than reading the final column alone, because it shows *where* a shape's gain came from: a row that drops once and then flattens was decided by a single mechanism, while a row that steps down repeatedly was improved by several independent ones. Latency is non-increasing through **all six rounds in all 144 cases**, the structural guarantee behaving as designed rather than a property we had to check for afterwards.

| Shape | M x N x K | Prec | Layout | Base | Tile | Split-K | Mapping | ReadVW | MemVW | StreamK | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `d4k_down_pf` | 4096x4096x16384 | fp16 | NN | 2669.6 | 948.0 | 946.6 | 946.6 | 863.9 | 849.7 | 849.7 | 3.14x |
| `sn_64_pf` | 4096x64x4096 | fp16 | NN | 35.2 | 17.3 | 17.2 | 16.5 | 16.5 | 16.4 | 16.4 | 2.14x |
| `sqS` | 512x512x512 | fp16 | TN | 8.2 | 6.3 | 5.5 | 4.6 | 4.6 | 4.4 | 4.4 | 1.86x |
| `moe_up_pf` | 4096x2048x8192 | fp8 | NN | 210.2 | 123.8 | 120.9 | 116.1 | 116.1 | 116.1 | 116.1 | 1.81x |
| `d4k_qkv_pf` | 4096x8192x4096 | bf16 | TN | 548.2 | 548.2 | 485.3 | 473.4 | 473.4 | 473.4 | 473.4 | 1.16x |
| `d4k_ffn_dec` | 16x16384x4096 | fp16 | NN | 45.7 | 44.2 | 41.5 | 39.4 | 39.4 | 39.4 | 39.4 | 1.16x |
| `thinK` | 4096x4096x512 | bf16 | NN | 51.8 | 50.9 | 49.8 | 47.1 | 47.1 | 47.1 | 47.1 | 1.10x |
| `fatK` | 1024x1024x8192 | fp16 | TN | 43.5 | 43.5 | 43.5 | 43.5 | 43.5 | 43.5 | 43.5 | 1.00x |

Table 1. Selected cases from the 144-GEMM matrix, latency in microseconds after each round, chosen to span the range of outcomes.

```{dropdown} Table 2. Every case in the 144-GEMM matrix, latency in microseconds after each round *(click to show)*
<!-- BEGIN per-case table -->
| Shape | M x N x K | Prec | Layout | Base | Tile | Split-K | Mapping | ReadVW | MemVW | StreamK | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sn_16_dec` | 16x16x4096 | bf16 | NN | 6.0 | 5.9 | 5.5 | 5.4 | 5.3 | 5.3 | 5.3 | 1.13x |
| `sn_16_dec` | 16x16x4096 | bf16 | TN | 5.6 | 5.6 | 5.0 | 4.8 | 4.8 | 4.8 | 4.8 | 1.18x |
| `sn_16_dec` | 16x16x4096 | fp16 | NN | 5.5 | 5.3 | 5.2 | 5.1 | 5.1 | 5.1 | 5.1 | 1.08x |
| `sn_16_dec` | 16x16x4096 | fp16 | TN | 5.5 | 5.2 | 5.0 | 4.8 | 4.8 | 4.8 | 4.8 | 1.14x |
| `sn_16_dec` | 16x16x4096 | fp8 | NN | 7.1 | 6.6 | 5.7 | 5.6 | 5.5 | 5.4 | 5.4 | 1.31x |
| `sn_16_dec` | 16x16x4096 | fp8 | TN | 4.4 | 4.4 | 4.4 | 4.3 | 4.3 | 4.3 | 4.3 | 1.02x |
| `sn_16_pf` | 8192x16x2048 | bf16 | NN | 14.3 | 13.8 | 13.8 | 13.1 | 13.1 | 13.1 | 13.1 | 1.09x |
| `sn_16_pf` | 8192x16x2048 | bf16 | TN | 13.0 | 13.0 | 13.0 | 12.8 | 12.8 | 12.7 | 12.7 | 1.02x |
| `sn_16_pf` | 8192x16x2048 | fp16 | NN | 13.2 | 13.2 | 13.2 | 12.8 | 12.8 | 12.8 | 12.8 | 1.03x |
| `sn_16_pf` | 8192x16x2048 | fp16 | TN | 14.0 | 13.4 | 13.1 | 12.8 | 12.7 | 12.7 | 12.7 | 1.10x |
| `sn_16_pf` | 8192x16x2048 | fp8 | NN | 12.6 | 12.6 | 11.9 | 11.3 | 11.2 | 11.1 | 11.1 | 1.14x |
| `sn_16_pf` | 8192x16x2048 | fp8 | TN | 8.7 | 7.6 | 7.6 | 7.6 | 7.6 | 7.6 | 7.6 | 1.15x |
| `sn_32_dec` | 16x32x8192 | bf16 | NN | 6.5 | 6.0 | 6.0 | 5.9 | 5.9 | 5.9 | 5.9 | 1.11x |
| `sn_32_dec` | 16x32x8192 | bf16 | TN | 6.2 | 5.6 | 5.6 | 5.6 | 5.6 | 5.6 | 5.6 | 1.11x |
| `sn_32_dec` | 16x32x8192 | fp16 | NN | 6.7 | 6.7 | 6.1 | 6.0 | 6.0 | 6.0 | 6.0 | 1.12x |
| `sn_32_dec` | 16x32x8192 | fp16 | TN | 5.6 | 5.6 | 5.6 | 5.6 | 5.6 | 5.6 | 5.6 | 1.00x |
| `sn_32_dec` | 16x32x8192 | fp8 | NN | 7.0 | 7.0 | 6.1 | 5.9 | 5.9 | 5.9 | 5.9 | 1.20x |
| `sn_32_dec` | 16x32x8192 | fp8 | TN | 6.5 | 6.5 | 5.6 | 5.0 | 5.0 | 5.0 | 5.0 | 1.29x |
| `sn_64_pf` | 4096x64x4096 | bf16 | NN | 16.6 | 16.6 | 16.6 | 16.4 | 16.4 | 16.1 | 16.1 | 1.03x |
| `sn_64_pf` | 4096x64x4096 | bf16 | TN | 15.0 | 14.7 | 14.5 | 14.4 | 14.4 | 14.4 | 14.4 | 1.04x |
| `sn_64_pf` | 4096x64x4096 | fp16 | NN | 35.2 | 17.3 | 17.2 | 16.5 | 16.5 | 16.4 | 16.4 | 2.14x |
| `sn_64_pf` | 4096x64x4096 | fp16 | TN | 15.3 | 15.0 | 14.6 | 14.4 | 14.4 | 14.4 | 14.4 | 1.06x |
| `sn_64_pf` | 4096x64x4096 | fp8 | NN | 16.7 | 15.7 | 14.0 | 13.9 | 13.6 | 13.5 | 13.5 | 1.24x |
| `sn_64_pf` | 4096x64x4096 | fp8 | TN | 11.4 | 11.4 | 10.4 | 10.1 | 10.1 | 10.1 | 10.1 | 1.13x |
| `sn_128_dec` | 16x128x4096 | bf16 | NN | 6.4 | 6.4 | 6.4 | 6.3 | 6.3 | 6.3 | 6.3 | 1.02x |
| `sn_128_dec` | 16x128x4096 | bf16 | TN | 6.1 | 6.1 | 6.1 | 6.1 | 6.1 | 6.1 | 6.1 | 1.01x |
| `sn_128_dec` | 16x128x4096 | fp16 | NN | 7.6 | 6.4 | 6.3 | 6.2 | 6.2 | 6.2 | 6.2 | 1.22x |
| `sn_128_dec` | 16x128x4096 | fp16 | TN | 6.1 | 6.1 | 6.1 | 6.0 | 6.0 | 6.0 | 6.0 | 1.02x |
| `sn_128_dec` | 16x128x4096 | fp8 | NN | 6.9 | 6.9 | 6.5 | 6.3 | 6.3 | 6.3 | 6.3 | 1.10x |
| `sn_128_dec` | 16x128x4096 | fp8 | TN | 4.6 | 4.6 | 4.6 | 4.4 | 4.4 | 4.4 | 4.4 | 1.04x |
| `sn_256_pf` | 4096x256x8192 | bf16 | NN | 53.6 | 53.6 | 47.9 | 45.3 | 45.3 | 45.3 | 45.3 | 1.18x |
| `sn_256_pf` | 4096x256x8192 | bf16 | TN | 54.9 | 53.6 | 52.1 | 47.5 | 47.5 | 47.5 | 47.5 | 1.15x |
| `sn_256_pf` | 4096x256x8192 | fp16 | NN | 54.4 | 54.4 | 49.5 | 46.2 | 46.2 | 46.2 | 46.2 | 1.18x |
| `sn_256_pf` | 4096x256x8192 | fp16 | TN | 55.2 | 54.9 | 51.0 | 48.2 | 48.2 | 48.2 | 48.2 | 1.14x |
| `sn_256_pf` | 4096x256x8192 | fp8 | NN | 40.0 | 38.0 | 35.4 | 34.3 | 34.3 | 34.3 | 34.3 | 1.17x |
| `sn_256_pf` | 4096x256x8192 | fp8 | TN | 31.9 | 31.9 | 30.1 | 28.7 | 28.7 | 28.7 | 28.7 | 1.11x |
| `sqS` | 512x512x512 | bf16 | NN | 6.8 | 6.8 | 6.7 | 6.6 | 6.6 | 6.4 | 6.4 | 1.07x |
| `sqS` | 512x512x512 | bf16 | TN | 5.9 | 4.9 | 4.4 | 4.3 | 4.3 | 4.3 | 4.3 | 1.37x |
| `sqS` | 512x512x512 | fp16 | NN | 8.2 | 5.8 | 5.8 | 5.8 | 5.8 | 5.7 | 5.7 | 1.43x |
| `sqS` | 512x512x512 | fp16 | TN | 8.2 | 6.3 | 5.5 | 4.6 | 4.6 | 4.4 | 4.4 | 1.86x |
| `sqS` | 512x512x512 | fp8 | NN | 8.6 | 6.1 | 6.1 | 6.1 | 6.1 | 6.1 | 6.1 | 1.42x |
| `sqS` | 512x512x512 | fp8 | TN | 7.0 | 5.5 | 4.9 | 4.9 | 4.9 | 4.9 | 4.9 | 1.44x |
| `fatK` | 1024x1024x8192 | bf16 | NN | 54.4 | 46.5 | 46.5 | 44.2 | 44.2 | 44.2 | 44.2 | 1.23x |
| `fatK` | 1024x1024x8192 | bf16 | TN | 49.1 | 49.1 | 44.8 | 41.2 | 41.2 | 41.2 | 41.2 | 1.19x |
| `fatK` | 1024x1024x8192 | fp16 | NN | 53.6 | 53.6 | 48.1 | 44.1 | 44.1 | 44.0 | 44.0 | 1.22x |
| `fatK` | 1024x1024x8192 | fp16 | TN | 43.5 | 43.5 | 43.5 | 43.5 | 43.5 | 43.5 | 43.5 | 1.00x |
| `fatK` | 1024x1024x8192 | fp8 | NN | 43.5 | 35.4 | 34.3 | 30.5 | 30.5 | 30.5 | 30.5 | 1.42x |
| `fatK` | 1024x1024x8192 | fp8 | TN | 28.2 | 28.2 | 26.7 | 25.0 | 25.0 | 25.0 | 25.0 | 1.13x |
| `tallM` | 8192x1024x2048 | bf16 | NN | 59.4 | 55.2 | 55.2 | 53.4 | 53.4 | 53.4 | 53.4 | 1.11x |
| `tallM` | 8192x1024x2048 | bf16 | TN | 106.5 | 72.7 | 71.2 | 65.8 | 65.8 | 65.8 | 65.8 | 1.62x |
| `tallM` | 8192x1024x2048 | fp16 | NN | 58.1 | 58.1 | 57.2 | 56.0 | 56.0 | 56.0 | 56.0 | 1.04x |
| `tallM` | 8192x1024x2048 | fp16 | TN | 104.7 | 104.7 | 103.6 | 95.9 | 95.9 | 95.9 | 95.5 | 1.10x |
| `tallM` | 8192x1024x2048 | fp8 | NN | 42.1 | 41.8 | 38.1 | 33.3 | 33.3 | 33.3 | 33.3 | 1.26x |
| `tallM` | 8192x1024x2048 | fp8 | TN | 39.6 | 35.2 | 35.2 | 33.9 | 33.9 | 33.9 | 33.9 | 1.17x |
| `sqL` | 2048x2048x2048 | bf16 | NN | 39.8 | 39.8 | 37.1 | 34.6 | 34.6 | 34.6 | 34.6 | 1.15x |
| `sqL` | 2048x2048x2048 | bf16 | TN | 42.6 | 42.5 | 40.7 | 36.8 | 36.8 | 36.1 | 36.1 | 1.18x |
| `sqL` | 2048x2048x2048 | fp16 | NN | 42.5 | 41.4 | 37.8 | 33.2 | 33.2 | 33.2 | 33.2 | 1.28x |
| `sqL` | 2048x2048x2048 | fp16 | TN | 45.6 | 45.0 | 42.4 | 38.1 | 38.1 | 38.1 | 38.1 | 1.20x |
| `sqL` | 2048x2048x2048 | fp8 | NN | 33.7 | 32.8 | 24.7 | 24.4 | 24.4 | 24.4 | 24.4 | 1.38x |
| `sqL` | 2048x2048x2048 | fp8 | TN | 31.5 | 21.9 | 20.8 | 19.3 | 19.3 | 19.3 | 19.3 | 1.63x |
| `moe_up_pf` | 4096x2048x8192 | bf16 | NN | 205.3 | 205.3 | 202.7 | 195.4 | 195.4 | 195.4 | 195.4 | 1.05x |
| `moe_up_pf` | 4096x2048x8192 | bf16 | TN | 336.0 | 336.0 | 335.8 | 329.2 | 329.2 | 329.2 | 329.2 | 1.02x |
| `moe_up_pf` | 4096x2048x8192 | fp16 | NN | 254.4 | 229.9 | 228.3 | 217.7 | 217.7 | 217.7 | 217.7 | 1.17x |
| `moe_up_pf` | 4096x2048x8192 | fp16 | TN | 387.4 | 379.4 | 322.4 | 307.7 | 307.7 | 307.7 | 307.7 | 1.26x |
| `moe_up_pf` | 4096x2048x8192 | fp8 | NN | 210.2 | 123.8 | 120.9 | 116.1 | 116.1 | 116.1 | 116.1 | 1.81x |
| `moe_up_pf` | 4096x2048x8192 | fp8 | TN | 197.4 | 197.4 | 179.3 | 175.3 | 175.3 | 175.3 | 175.3 | 1.13x |
| `d4k_down_dec` | 16x4096x16384 | bf16 | NN | 39.2 | 38.5 | 37.5 | 36.1 | 36.1 | 36.1 | 36.1 | 1.08x |
| `d4k_down_dec` | 16x4096x16384 | bf16 | TN | 46.4 | 38.3 | 36.0 | 35.5 | 35.5 | 35.5 | 35.5 | 1.31x |
| `d4k_down_dec` | 16x4096x16384 | fp16 | NN | 39.1 | 38.4 | 36.0 | 34.5 | 34.5 | 34.5 | 34.5 | 1.13x |
| `d4k_down_dec` | 16x4096x16384 | fp16 | TN | 45.9 | 37.7 | 36.3 | 35.5 | 35.5 | 35.5 | 35.5 | 1.29x |
| `d4k_down_dec` | 16x4096x16384 | fp8 | NN | 20.9 | 20.9 | 20.4 | 20.0 | 19.9 | 19.9 | 19.9 | 1.05x |
| `d4k_down_dec` | 16x4096x16384 | fp8 | TN | 20.4 | 20.4 | 19.5 | 19.4 | 19.4 | 19.4 | 19.4 | 1.05x |
| `thinK` | 4096x4096x512 | bf16 | NN | 51.8 | 50.9 | 49.8 | 47.1 | 47.1 | 47.1 | 47.1 | 1.10x |
| `thinK` | 4096x4096x512 | bf16 | TN | 37.5 | 32.1 | 31.9 | 31.4 | 31.4 | 31.4 | 31.4 | 1.20x |
| `thinK` | 4096x4096x512 | fp16 | NN | 38.2 | 38.2 | 37.8 | 35.2 | 35.0 | 35.0 | 35.0 | 1.09x |
| `thinK` | 4096x4096x512 | fp16 | TN | 32.4 | 32.4 | 32.3 | 31.6 | 31.6 | 31.6 | 31.6 | 1.03x |
| `thinK` | 4096x4096x512 | fp8 | NN | 29.3 | 29.3 | 28.2 | 26.1 | 26.1 | 26.1 | 25.9 | 1.13x |
| `thinK` | 4096x4096x512 | fp8 | TN | 25.3 | 25.3 | 24.4 | 21.6 | 21.6 | 21.6 | 21.6 | 1.17x |
| `d4k_down_pf` | 4096x4096x16384 | bf16 | NN | 1137.3 | 1137.3 | 1123.8 | 1089.8 | 1085.0 | 1083.4 | 1071.4 | 1.06x |
| `d4k_down_pf` | 4096x4096x16384 | bf16 | TN | 1324.9 | 1319.9 | 1317.2 | 1252.3 | 1252.3 | 1252.3 | 1247.3 | 1.06x |
| `d4k_down_pf` | 4096x4096x16384 | fp16 | NN | 2669.6 | 948.0 | 946.6 | 946.6 | 863.9 | 849.7 | 849.7 | 3.14x |
| `d4k_down_pf` | 4096x4096x16384 | fp16 | TN | 1340.5 | 1340.5 | 1340.5 | 1280.0 | 1280.0 | 1280.0 | 1252.7 | 1.07x |
| `d4k_down_pf` | 4096x4096x16384 | fp8 | NN | 519.4 | 519.4 | 519.4 | 472.9 | 472.9 | 472.9 | 472.9 | 1.10x |
| `d4k_down_pf` | 4096x4096x16384 | fp8 | TN | 696.6 | 696.6 | 696.4 | 659.9 | 651.4 | 651.4 | 651.4 | 1.07x |
| `d4k_qkv_dec` | 16x8192x4096 | bf16 | NN | 24.1 | 21.2 | 21.2 | 20.4 | 20.4 | 20.4 | 20.4 | 1.18x |
| `d4k_qkv_dec` | 16x8192x4096 | bf16 | TN | 25.2 | 20.7 | 20.5 | 20.2 | 20.2 | 20.2 | 20.2 | 1.25x |
| `d4k_qkv_dec` | 16x8192x4096 | fp16 | NN | 29.3 | 20.9 | 20.8 | 20.2 | 20.2 | 20.2 | 20.2 | 1.45x |
| `d4k_qkv_dec` | 16x8192x4096 | fp16 | TN | 25.2 | 20.8 | 20.7 | 20.3 | 20.3 | 20.2 | 20.2 | 1.25x |
| `d4k_qkv_dec` | 16x8192x4096 | fp8 | NN | 11.5 | 11.5 | 11.3 | 10.9 | 10.8 | 10.8 | 10.8 | 1.06x |
| `d4k_qkv_dec` | 16x8192x4096 | fp8 | TN | 13.0 | 11.6 | 11.3 | 11.3 | 11.3 | 11.3 | 11.3 | 1.15x |
| `wideN` | 1024x8192x2048 | bf16 | NN | 92.8 | 92.8 | 90.6 | 86.9 | 86.9 | 86.9 | 86.9 | 1.07x |
| `wideN` | 1024x8192x2048 | bf16 | TN | 80.0 | 80.0 | 80.0 | 61.1 | 61.0 | 60.7 | 60.7 | 1.32x |
| `wideN` | 1024x8192x2048 | fp16 | NN | 96.3 | 92.0 | 89.4 | 60.5 | 60.5 | 60.3 | 60.3 | 1.60x |
| `wideN` | 1024x8192x2048 | fp16 | TN | 83.1 | 82.7 | 81.9 | 61.0 | 61.0 | 59.8 | 59.8 | 1.39x |
| `wideN` | 1024x8192x2048 | fp8 | NN | 52.3 | 36.9 | 33.9 | 33.0 | 33.0 | 33.0 | 33.0 | 1.58x |
| `wideN` | 1024x8192x2048 | fp8 | TN | 38.0 | 38.0 | 33.1 | 33.1 | 33.1 | 33.1 | 33.1 | 1.15x |
| `d8k_down_pf` | 2048x8192x32768 | bf16 | NN | 2382.7 | 1725.3 | 1721.5 | 1650.6 | 1650.6 | 1650.6 | 1650.6 | 1.44x |
| `d8k_down_pf` | 2048x8192x32768 | bf16 | TN | 3168.7 | 3163.5 | 3162.6 | 2733.3 | 2733.3 | 2726.5 | 2726.5 | 1.16x |
| `d8k_down_pf` | 2048x8192x32768 | fp16 | NN | 2737.9 | 2737.9 | 2706.3 | 2603.2 | 2603.2 | 2603.2 | 2603.2 | 1.05x |
| `d8k_down_pf` | 2048x8192x32768 | fp16 | TN | 3262.1 | 3221.6 | 3197.3 | 2796.2 | 2796.2 | 2796.2 | 2796.2 | 1.17x |
| `d8k_down_pf` | 2048x8192x32768 | fp8 | NN | 1434.7 | 1011.0 | 1011.0 | 903.0 | 903.0 | 874.2 | 870.9 | 1.65x |
| `d8k_down_pf` | 2048x8192x32768 | fp8 | TN | 1442.9 | 1442.9 | 1441.5 | 1305.1 | 1305.1 | 1305.1 | 1305.1 | 1.11x |
| `moe_down_pf` | 4096x8192x2048 | bf16 | NN | 237.5 | 237.5 | 237.5 | 231.9 | 231.9 | 231.9 | 231.9 | 1.02x |
| `moe_down_pf` | 4096x8192x2048 | bf16 | TN | 245.5 | 245.5 | 245.5 | 242.0 | 242.0 | 242.0 | 242.0 | 1.01x |
| `moe_down_pf` | 4096x8192x2048 | fp16 | NN | 289.0 | 289.0 | 242.2 | 241.3 | 239.8 | 239.8 | 239.8 | 1.21x |
| `moe_down_pf` | 4096x8192x2048 | fp16 | TN | 304.4 | 299.3 | 259.1 | 247.0 | 247.0 | 247.0 | 247.0 | 1.23x |
| `moe_down_pf` | 4096x8192x2048 | fp8 | NN | 156.1 | 125.4 | 124.3 | 114.3 | 114.3 | 114.3 | 114.3 | 1.37x |
| `moe_down_pf` | 4096x8192x2048 | fp8 | TN | 152.3 | 152.3 | 145.0 | 138.1 | 138.1 | 138.1 | 138.1 | 1.10x |
| `d4k_qkv_pf` | 4096x8192x4096 | bf16 | NN | 536.1 | 531.5 | 531.5 | 517.4 | 517.4 | 517.4 | 517.4 | 1.04x |
| `d4k_qkv_pf` | 4096x8192x4096 | bf16 | TN | 548.2 | 548.2 | 485.3 | 473.4 | 473.4 | 473.4 | 473.4 | 1.16x |
| `d4k_qkv_pf` | 4096x8192x4096 | fp16 | NN | 552.4 | 545.6 | 544.7 | 475.4 | 475.4 | 475.4 | 475.4 | 1.16x |
| `d4k_qkv_pf` | 4096x8192x4096 | fp16 | TN | 580.9 | 580.3 | 571.9 | 549.8 | 549.8 | 549.8 | 549.8 | 1.06x |
| `d4k_qkv_pf` | 4096x8192x4096 | fp8 | NN | 286.5 | 286.5 | 286.5 | 278.7 | 278.7 | 278.7 | 278.7 | 1.03x |
| `d4k_qkv_pf` | 4096x8192x4096 | fp8 | TN | 273.0 | 273.0 | 273.0 | 260.4 | 260.4 | 255.5 | 255.5 | 1.07x |
| `d4k_ffn_dec` | 16x16384x4096 | bf16 | NN | 46.8 | 44.9 | 41.7 | 39.5 | 39.5 | 39.5 | 39.5 | 1.19x |
| `d4k_ffn_dec` | 16x16384x4096 | bf16 | TN | 46.2 | 43.9 | 40.7 | 38.8 | 38.8 | 38.8 | 38.8 | 1.19x |
| `d4k_ffn_dec` | 16x16384x4096 | fp16 | NN | 45.7 | 44.2 | 41.5 | 39.4 | 39.4 | 39.4 | 39.4 | 1.16x |
| `d4k_ffn_dec` | 16x16384x4096 | fp16 | TN | 46.2 | 44.0 | 41.0 | 39.2 | 39.2 | 39.2 | 39.2 | 1.18x |
| `d4k_ffn_dec` | 16x16384x4096 | fp8 | NN | 18.4 | 18.4 | 18.2 | 18.1 | 18.0 | 17.9 | 17.9 | 1.03x |
| `d4k_ffn_dec` | 16x16384x4096 | fp8 | TN | 17.1 | 16.9 | 16.9 | 16.8 | 16.8 | 16.8 | 16.8 | 1.01x |
| `d8k_qkv_dec` | 16x16384x8192 | bf16 | NN | 77.8 | 76.1 | 74.8 | 70.6 | 70.1 | 70.1 | 70.1 | 1.11x |
| `d8k_qkv_dec` | 16x16384x8192 | bf16 | TN | 72.5 | 72.4 | 70.8 | 67.0 | 67.0 | 67.0 | 67.0 | 1.08x |
| `d8k_qkv_dec` | 16x16384x8192 | fp16 | NN | 114.8 | 76.7 | 72.1 | 68.9 | 68.9 | 68.9 | 68.9 | 1.67x |
| `d8k_qkv_dec` | 16x16384x8192 | fp16 | TN | 72.7 | 72.7 | 69.4 | 67.0 | 67.0 | 67.0 | 67.0 | 1.08x |
| `d8k_qkv_dec` | 16x16384x8192 | fp8 | NN | 38.4 | 35.2 | 34.6 | 31.7 | 31.7 | 31.7 | 31.7 | 1.21x |
| `d8k_qkv_dec` | 16x16384x8192 | fp8 | TN | 39.2 | 34.1 | 30.9 | 30.9 | 30.9 | 30.8 | 30.8 | 1.27x |
| `d8k_qkv_pf` | 2048x16384x8192 | bf16 | NN | 890.6 | 890.6 | 890.0 | 878.3 | 878.3 | 878.3 | 878.3 | 1.01x |
| `d8k_qkv_pf` | 2048x16384x8192 | bf16 | TN | 994.0 | 994.0 | 990.7 | 904.3 | 900.3 | 900.3 | 900.3 | 1.10x |
| `d8k_qkv_pf` | 2048x16384x8192 | fp16 | NN | 849.0 | 849.0 | 848.7 | 848.7 | 848.7 | 848.7 | 848.7 | 1.00x |
| `d8k_qkv_pf` | 2048x16384x8192 | fp16 | TN | 995.0 | 995.0 | 995.0 | 938.8 | 938.8 | 938.8 | 938.8 | 1.06x |
| `d8k_qkv_pf` | 2048x16384x8192 | fp8 | NN | 563.3 | 561.5 | 556.7 | 518.6 | 518.6 | 518.6 | 518.6 | 1.09x |
| `d8k_qkv_pf` | 2048x16384x8192 | fp8 | TN | 609.2 | 585.7 | 585.7 | 533.4 | 533.4 | 533.4 | 533.4 | 1.14x |
| `d4k_ffn_pf` | 4096x16384x4096 | bf16 | NN | 834.0 | 824.3 | 824.3 | 770.2 | 770.2 | 770.2 | 770.2 | 1.08x |
| `d4k_ffn_pf` | 4096x16384x4096 | bf16 | TN | 955.0 | 856.6 | 856.6 | 823.5 | 823.5 | 823.5 | 823.5 | 1.16x |
| `d4k_ffn_pf` | 4096x16384x4096 | fp16 | NN | 858.9 | 857.2 | 826.0 | 806.3 | 806.3 | 806.3 | 806.3 | 1.07x |
| `d4k_ffn_pf` | 4096x16384x4096 | fp16 | TN | 912.5 | 901.4 | 901.4 | 806.0 | 806.0 | 806.0 | 806.0 | 1.13x |
| `d4k_ffn_pf` | 4096x16384x4096 | fp8 | NN | 572.7 | 570.9 | 569.2 | 545.0 | 544.9 | 475.8 | 475.8 | 1.20x |
| `d4k_ffn_pf` | 4096x16384x4096 | fp8 | TN | 541.8 | 540.5 | 540.5 | 519.5 | 519.5 | 519.5 | 450.8 | 1.20x |
| `d8k_ffn_pf` | 2048x32768x8192 | bf16 | NN | 1814.5 | 1796.0 | 1676.6 | 1549.8 | 1549.8 | 1549.8 | 1549.8 | 1.17x |
| `d8k_ffn_pf` | 2048x32768x8192 | bf16 | TN | 1747.4 | 1747.4 | 1747.4 | 1626.1 | 1626.1 | 1626.1 | 1626.1 | 1.07x |
| `d8k_ffn_pf` | 2048x32768x8192 | fp16 | NN | 1774.8 | 1774.8 | 1751.1 | 1592.3 | 1592.3 | 1592.3 | 1592.3 | 1.11x |
| `d8k_ffn_pf` | 2048x32768x8192 | fp16 | TN | 1926.7 | 1852.8 | 1852.8 | 1678.5 | 1663.6 | 1663.6 | 1663.6 | 1.16x |
| `d8k_ffn_pf` | 2048x32768x8192 | fp8 | NN | 985.0 | 985.0 | 972.0 | 926.2 | 925.9 | 925.9 | 925.9 | 1.06x |
| `d8k_ffn_pf` | 2048x32768x8192 | fp8 | TN | 926.2 | 926.2 | 926.2 | 888.3 | 885.0 | 885.0 | 885.0 | 1.05x |
<!-- END per-case table -->
```

### Per-Round Contributions

Table 3 breaks the descent down by round.

<!-- BEGIN per-round table -->
| Round | Mean speedup | Median speedup | Best case | Cases improved |
| --- | --- | --- | --- | --- |
| Tile | 1.086x | 1.004x | 2.82x | 85 / 144 |
| Split-K | 1.039x | 1.015x | 1.33x | 112 / 144 |
| Mapping | 1.054x | 1.040x | 1.48x | 136 / 144 |
| ReadVW | 1.002x | 1.000x | 1.10x | 26 / 144 |
| MemVW | 1.003x | 1.000x | 1.15x | 26 / 144 |
| StreamK | 1.001x | 1.000x | 1.15x | 7 / 144 |
<!-- END per-round table -->

Table 3. Per-round speedup over the kernel entering that round, across all 144 GEMMs.

Overall the tuned kernel is a median **1.139x** faster than its baseline (mean 1.196x, best 3.14x). Measured instead against the library default it is a median 1.067x, and 44 of 144 cases end below it: with a top-1 seed the descent cannot always close a gap it did not open, which is the limitation noted above and the reason the headline is quoted against the baseline. The distinction is worth keeping straight when reading any tuning result, including this one. A monotone method guarantees you will not lose ground relative to the kernel it was handed; it says nothing about whether that kernel was the best one available to hand it, and choosing a weak seed is a mistake the descent cannot repair.

Three rounds carry the descent and they are near-equals in the mean, Tile at 1.086x, Mapping at 1.054x and Split-K at 1.039x, so none can be dropped while keeping most of the benefit. They differ in shape, though, which is why the median column matters as much as the mean. Mapping is the steady one, firing in 136 of 144 cases for a median 1.040x: a small, almost universal gain, which is what you expect from a round that only reorders work instead of changing the kernel. Tile is lumpy, with the highest mean of any round but a median of only 1.004x, because it fires in 85 of 144 and pays up to 2.82x when it does. Averaging those two profiles into a single headline number would hide the difference that matters operationally: Mapping is worth running because it nearly always helps a little, Tile is worth running because it occasionally helps enormously.

The three lower rounds have a median of exactly 1.000x, so their value is the tail: read-side widths and store-and-LDS fire in 26 cases each, Stream-K in only 7 — but Stream-K is the only round serving those 7 at all, since no amount of Global-Split-K tuning reaches the decomposition it provides. A zero-median round costs few candidates and cannot regress, so it is insurance rather than optimization, and it should be budgeted that way: you are not paying for an expected gain, you are paying a small fixed premium so that the minority of shapes which need that mechanism are covered.

### The Tile Round: Why It Fires, and When It Matters

The tile round looks like it should be redundant here: sweeping the full generated grid shows **the best MacroTile was already in the library pool in all 144 cases**. The resolution is that a MacroTile is not a kernel. Of the 85 cases where the round improved on its baseline, **41 kept the same MacroTile** and won on a better wave decomposition at that tile, which only a generated grid reaches. It fires in 65% of fp16 cases, 58% of bf16 and 54% of fp8, and in 64% of NN against 54% of TN. The largest single win is `d4k_down_pf` in fp16 NN, where a move from `MT256x224x64` to `MT256x256x32` cuts latency from 2669.6 us to 948.0 us.

Because this pool is mature, the matrix understates the round for a bring-up, so we simulated pools of varying maturity: group the landscape by distinct MacroTile, sample N of them, let the baseline be the best tile *inside* that sparse pool, and compare against opening the full grid. The construction matters for the conclusion to mean anything. No arm is handicapped, since the baseline at every maturity is the best kernel that pool could offer, so the gap being measured is what tile expansion adds over the best already-available option rather than over an arbitrary one. Landscapes span 3 to 176 tiles (median 23), so maturity is expressed as the share of a shape's own landscape the pool holds instead of an absolute kernel count, which lets shapes with very different landscape sizes be compared on the same axis. Figure 1, below, shows how the gain from opening the tile decays as that share rises.

```{figure} ./images/tile-gain-vs-pool-maturity.png
:align: center
:alt: Tile-expansion gain against pool maturity on MI325X, falling from about 22 percent when the pool covers under 5 percent of the tile landscape to about 6 percent when it covers most of it.

Figure 1. Tile expansion pays off in proportion to how young the kernel pool is. Median gain with the interquartile range across 144 GEMMs.
```

Leaving the tile pinned costs a median **22.29%** when the pool has tuned under 5% of a shape's landscape, falling by roughly a factor of four to **5.83%** once it covers most of it. The decay is the finding, not the endpoints. The absolute level is an upper bound, because the comparison arm is the fastest of several single-shot candidates and so inherits a winner's-curse bias; since the same comparison is used at every maturity, that bias shifts the whole curve by roughly a constant and leaves the slope intact. The practical reading is a scheduling rule rather than a number to quote: **gate the tile round on pool maturity**, running it first during a bring-up on a new architecture or datatype, and dropping it first once the pool has been swept and the budget is better spent on the rounds below it.

### Against a One-Shot Sweep

The iterative search dominates the one-shot sweep in the strict sense: the sweep is beaten on the kernel it finds *and* on the time it takes, so there is no budget at which it becomes the better choice. Figure 2 is the first half. Against the tuned kernel from Table 2 the one-shot grid is slower in **116 of 144 cases**, by a geomean of 1.18x overall, and it loses at every precision: 1.21x in fp8, 1.19x in fp16, 1.13x in bf16. Losing consistently across all three precisions is the part worth noting, because it rules out the explanation that one datatype happened to suit the grid's template.

One point sits far off to the right, and it is worth reading correctly. For `d4k_ffn_dec` in fp16 NN the one-shot grid returned a **429 us** kernel where ours runs at 39.4 us. That 10.9x is not the method excelling; our kernel is an unremarkable 1.26x over the library default there. It is the sweep collapsing: **429 us is 8.6x slower than not tuning at all**. A grid that cannot see the kernel you already have has no floor to stop it, and this is what that looks like when it goes wrong.

```{figure} ./images/oneshot-quality.png
:align: center
:alt: Per-case comparison showing the one-shot sweep slower than the iterative kernel at every precision, geomean 1.18x.

Figure 2. A one-shot sweep of roughly 10,000 candidates loses to a search that compiles about 156 kernels. One point per GEMM, split by precision; the bar is the geomean.
```

Figure 3 is the second half, and it is the cleaner result of the two: **every one of the 144 shapes cost more to tune with the one-shot grid**, most of them by more than 4x, with a worst case of 21.9x. Cumulatively the grid spent **1,691 minutes** against **431 minutes**.

```{figure} ./images/oneshot-cost.png
:align: center
:alt: Log-log scatter of per-shape tuning time; all 144 shapes lie above the equal line, most above the 4x line.

Figure 3. Per-shape tuning wall time, paired on the same GPU. Every shape sits above the equal line, so the larger budget buys a worse kernel and pays more for it.
```

More candidates do not help when they are in the wrong place. The one-shot grid does not lack search budget, it lacks an anchor: it spends an order of magnitude more compilations exploring a region chosen without reference to the kernel already in hand, and it has no mechanism that prevents it from finishing below where it started. The iterative descent searches a far smaller region, but every point in that region is reached from a kernel that has already been measured, and no round can hand back something worse than what it was given.

## Summary

Growing a search outward from a kernel you already trust, one small group of parameters at a time, dominates opening everything at once on both axes that matter: it compiles about 156 kernels per shape instead of enumerating 10,000, and it wins in 116 of 144 GEMMs at a quarter of the wall time.

Four things are worth carrying away. Monotonicity is structural, not lucky: union the incumbent into every candidate list, adopt only on a strict win, and latency is non-increasing through all six rounds in all 144 cases. The procedure enforces that by construction, so it does not depend on the ranges being well chosen or on the measurements being quiet. The top three rounds are near-equals in the mean but differ in profile, so ordering by impact matters more than trusting any one of them, and dropping the round with the smallest median would remove the one that pays the largest single wins. The tile round's worth is a property of your pool rather than of the method, from a median 5.83% on a mature pool to 22.29% on a young one, and should be gated accordingly instead of run by default. And a campaign needs a second independent measurement of its own baseline: the most eye-catching number this study first produced was a timing artifact, and an inflated baseline is indistinguishable from a spectacular first round until something else measures it.

Together with the [previous blog](https://rocm.blogs.amd.com/artificial-intelligence/reverse-hipblaslt-tensilelite/README.html), this completes the method: decoding a kernel establishes a floor, and iterative warm-start expansion grows the search space outward from it while keeping that floor intact. Follow the AMD ROCm Blogs for more hipBLASLt and TensileLite deep dives from our team.

## Acknowledgement

We would like to express our thanks to our colleagues [Brian Chang](../../authors/brian-chang.md), [Eveline Chen](../../authors/eveline-chen.md), [Bobo Fang](../../authors/bobo-fang.md), [Bill Ku](../../authors/bill-ku.md), [Kaiping Lu](../../authors/kaiping-lu.md), and [Menghsuan Yang](../../authors/menghsuan-yang.md) for their insightful feedback and technical assistance.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED "AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
