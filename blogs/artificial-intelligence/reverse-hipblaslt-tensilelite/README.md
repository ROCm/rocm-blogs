---
blogpost: true
blog_title: "Reverse-Engineering hipBLASLt TensileLite Kernels: From Solution Name to a Tuning Config"
date: "04 Aug 2026"
author: 'Yuchen Lin, Clement Lin, Chunhung Wang'
thumbnail: 'reverse-hipblaslt-tensilelite-thumbnail.png'
tags: AI/ML, Performance, LLM
category: Applications & models
target_audience: AI developers and enthusiasts
key_value_propositions: Combine reverse-engineering and iterative expansion tuning so that a TensileLite run is guaranteed to produce a kernel at least as fast as the pool's current best, and usually faster.
language: English
myst:
    html_meta:
        "author": "Yuchen Lin, Clement Lin, Chunhung Wang"
        "description lang=en": "Pin the pool's best kernel into a TensileLite tuning config by decoding its solution name, so an expanded re-tune can only match or beat it."
        "keywords": "LLM, Kernels, Inference, hipBLASLt, GEMM tuning, TensileLite, AMD Instinct MI300X"
        "vertical": "Developers, AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
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

# Reverse-Engineering hipBLASLt TensileLite Kernels: From Solution Name to a Tuning Config

In a previous blog, [Customizing Kernels with hipBLASLt TensileLite GEMM Tuning](https://rocm.blogs.amd.com/artificial-intelligence/hipblaslt-tensilelite-tuning/README.html), we showed how TensileLite Tuning generates brand-new GEMM kernels by searching a parameter space and selecting the fastest valid candidate for a given problem size. The search space is defined entirely by the tuning configuration YAML you provide. Its `ForkParameters` lists are expanded into a Cartesian product, and the tuning run can only ever return a kernel that is expressible as one combination within that product.

That property cuts both ways. TensileLite can generate genuinely new kernels, but it is only as good as the search space you hand it. If the parameter combinations in your configuration cannot reproduce the best kernel you already have, that kernel is not even a candidate, and the run can return something slower than where you started. The strongest kernel you currently hold is usually the one **Offline Tuning** selected from the existing solution pool for that shape, so the real question is sharper than "can tuning find a fast kernel?" It is "can tuning **guarantee** a kernel at least as fast as the pool's current best?"

This blog answers that question with a method that turns TensileLite Tuning into a process with a provable floor, built on two ideas:

* **Reverse-engineer the current best kernel into a pinned configuration.** Decode the pool-best kernel's solution name back into its exact parameters and write them as single-value `ForkParameters`. The resulting search space is exactly one point: that kernel. This is the anchor the run can never fall below.
* **Expand the search space outward from that anchor.** Once the pinned configuration is in hand, you can widen the parameters to look for something faster, while keeping the anchor kernel inside the search space so the floor never drops.

Together they guarantee the result never regresses: the search space always contains the pool's best kernel, so the kernel TensileLite generates is at least as fast as the pool's current best, and usually faster. This blog focuses on the first idea, the reverse-engineering that establishes the anchor; the expansion strategy is a topic for a follow-up.

## Why the Search Space Must Contain Your Current Best Kernel

Treat the tuning configuration as the definition of a set: its `ForkParameters`, with the problem type, enumerate a finite set of candidate kernels, and a run keeps the fastest valid member. One hard constraint governs everything:

> A TensileLite run can only ever return a kernel that is a member of the search space its configuration defines.

Offline Tuning, by contrast, generates nothing; it selects the best existing kernel from the pool for your shape, which is by definition your current floor. Combine the two and the requirement is unavoidable:

* If your current best kernel is **not** a member of the configuration's search space, the TensileLite run cannot produce it, and there is no guarantee the winner will be at least as fast. The outcome could easily be a regression.
* If your current best kernel **is** a member of the search space, then in the worst case the run reproduces it and ties, and in the better case it finds a neighbor that is faster. Either way you cannot lose.

Figure 1 makes this contrast concrete and previews the contribution of the method. A generic tuning configuration is built from a fixed template, blind to your golden kernel, so the golden can fall outside its search space (left); the run then returns the fastest kernel it happens to contain, which may be slower than the golden. Reverse-engineering instead builds the search space around the golden and keeps it inside (right), so the run can only match or beat it.

```{figure} ./images/fig1.png
:align: center
:alt: A generic search space that misses the golden kernel versus a reverse-anchored space that contains it

Figure 1. A TensileLite run can only return a kernel inside the search space its configuration defines. Left: a generic configuration is built from a template without reference to the golden, so the pool's best kernel can fall outside the search space and be unreachable, and the best kernel found may be slower than the golden. Right: reverse-engineering builds the search space around the golden and keeps it inside, so the tuned result can only match or beat it.
```

So the first goal is precise: make the configuration's search space a **superset** that includes your current best kernel. The catch is that you usually only have that kernel's solution name from the hipBLASLt logs, not the configuration that produced it, and user-tuned kernels are frequently absent from the committed library logic. The only general way to recover its parameters is to decode the name itself, and because a TensileLite solution name is a deterministic encoding of the kernel's solution parameters rather than an opaque hash, the decode is exact. That decode is the subject of the rest of this blog.

Figure 2 summarizes the two-stage idea: reverse-engineering pins the pool's best kernel inside the search space, and any later expansion keeps it there, so the result can only match or beat it.

```{figure} ./images/fig2.png
:align: center
:alt: Keep the pool's best kernel inside the tuning search space

Figure 2. The tuning configuration defines a search space (the outer region): the set of all kernels expressible as a combination of its `ForkParameters`. A TensileLite run can only return a kernel inside this region. Stage 1 places the pool's best kernel inside the region by reverse-engineering it; Stage 2 grows the region outward while keeping that kernel inside, so the run can only match or beat it.
```

## Reverse-Engineering the Current Best Kernel

The goal is to turn the pool's best kernel, which you only know as a solution name, into a pinned tuning configuration whose search space is exactly that one kernel. This is the heart of the method and the part you must get exactly right, because pinning the current best inside the search space is precisely what makes the guarantee hold. The following sections work through the decode end to end: the structure of a solution name, the rules that map each token back to a parameter, the reconstruction of the matrix instruction, and the recovery of the problem type. The output is a configuration you can hand to TensileLite as-is, or use as the starting point for the expansion step, which is a topic for a follow-up.

## Anatomy of a TensileLite Solution Name

A TensileLite solution name (the `SolutionNameMin` string) is produced by the naming logic in `tensilelite/Tensile/SolutionStructs/Naming.py`. Consider this example:

```text
Cijk_Alik_Bljk_BBS_STA_BH_UserArgs_MT32x16x256_MI16x16x1_SN_LDSB0_AFC0_..._PGR2_PLR1_..._SIA3_SS1_..._VWA1_VWB1_WS64_WG32_4_1_WGM1_WGMXCC4_WGMXCCGn1
```

It has three logical regions.

### 1. The Problem-Type Prefix

The leading tokens describe the operation and its features, not the tunable kernel knobs. They come from `ProblemType.__str__` in `tensilelite/Tensile/SolutionStructs/Problem.py`:

* `Cijk_Alik_Bljk` encodes the index assignments (the transpose layout).
* `BBS` is the datatype signature. Each letter is a datatype character: `B` for `bfloat16`, `H` for `float16`, `S` for `float32`, and so on. The trailing letters capture destination and compute types.
* `STA` marks `SwizzleTensorA`, `BH` marks `UseBeta` plus `HighPrecisionAccumulate`, `Bias` marks bias support, `SAV` marks `UseScaleAlphaVec`, `SABV` marks `UseScaleAB` in `Vector` mode, and `HA` marks the `hipblaslt_all` activation set.

These prefix tokens map back to `ProblemType` fields, which belong in the `BenchmarkProblems` problem type, not in `ForkParameters`.

### 2. The Tile and Matrix-Instruction Anchors

Two composite tokens anchor the tile shape:

* `MT{MacroTile0}x{MacroTile1}x{DepthU}`, for example `MT32x16x256`.
* `MI{M}x{N}x{B}`, for example `MI16x16x1`.

Note that the `MI` token encodes only `M`, `N`, and `B`. It does **not** carry the matrix-instruction K dimension. We reconstruct K from the datatype and the MI dimensions, as described below.

### 3. The Solution-Parameter Tokens

Everything after the anchors is a sequence of underscore-separated tokens, one per non-default solution parameter. Each token is a parameter-name abbreviation immediately followed by a value abbreviation, so decoding means reversing two rules in turn.

#### The name-abbreviation rule

The prefix keeps only the uppercase letters of the parameter name, a deterministic transformation you can reproduce for any parameter:

* `PrefetchGlobalRead` keeps `P`, `G`, `R`, giving `PGR`.
* `ScheduleIterAlg` keeps `S`, `I`, `A`, giving `SIA`.
* `WorkGroupMappingXCC` keeps `W`, `G`, `M`, `X`, `C`, `C`, giving `WGMXCC`.
* `LocalWritePerMfma` keeps `L`, `W`, `P`, `M`, giving `LWPM`.

The rule also explains cryptic tokens. `UseSgprForGRO` keeps the uppercase letters of every word, including the `F` of `For`, giving `USFGRO`. `Use64bShadowLimit` gives `USL` (digits and the lowercase `b` drop). A leading digit counts as non-uppercase, so `1LDSBuffer` reduces to `LDSB`.

Because the rule is lossy, different names can collapse to the same prefix; see disambiguation below.

#### The value-abbreviation rule

The suffix of a token encodes the value. The encoding depends on the value's type, summarized in Table 1:

| Value type | Encoding | Example token | Decodes to |
| --- | --- | --- | --- |
| Boolean | `1` for true, `0` for false | `SS1` | `SourceSwap = true` |
| Non-negative integer | the decimal digits | `PGR2` | `PrefetchGlobalRead = 2` |
| Negative integer | `n` followed by the magnitude | `LWPMn1` | `LocalWritePerMfma = -1` |
| Float | integer part, then `p`, then two hundredths digits; a whole number drops the `p` part | `GRPM1` or `0p50` | `1.0` or `0.50` |
| String | the uppercase letters of the string value | `GSUAMB` | `GlobalSplitUAlgorithm = MultipleBuffer` |
| ISA tuple | the three components concatenated, last in hex | `ISA942` | `ISA = [9, 4, 2]` |

Table 1. Value-abbreviation encodings by value type.

The string rule stacks two abbreviations: `GlobalSplitUAlgorithm` gives `GSUA` and its value `MultipleBuffer` gives `MB`, so the token is `GSUAMB` (likewise `GSUAMBSK` for `MultipleBufferSingleKernel` and `GSUASB` for `SingleBuffer`). Read the value by matching the trailing letters against that parameter's known values, not a generic dictionary.

#### Tokenizing with longest-prefix matching

Because prefix and value are concatenated with no separator, split a token by longest-prefix matching against known parameter abbreviations. In `GSUAMBSK`, a naive match stops at `GSU` (`GlobalSplitU`) and leaves the meaningless `AMBSK`; the correct split takes the longer `GSUA` (`GlobalSplitUAlgorithm`) and reads `MBSK` as `MultipleBufferSingleKernel`. Always prefer the longest valid abbreviation, then read the remainder as the value.

#### Disambiguating shared abbreviations

Some parameters share a prefix because the rule is lossy. `AssertFree0ElementMultiple` and `AssertFree1ElementMultiple` both reduce to `AFEM`, so a name carries two, for example `AFEM1_AFEM1`. They are emitted in sorted key order, so the first is `AssertFree0ElementMultiple` and the second `AssertFree1ElementMultiple`; preserve that order when decoding.

#### Only non-default parameters appear

The name lists only parameters whose value differs from the default in the reduced ("min") set used for naming; everything absent takes its default. That is exactly the shape a tuning configuration wants, so a faithful decode lists only the non-default knobs and does not pad with defaults, which would add noise without changing the kernel.

### A reference map of common tokens

Table 2 groups the abbreviations you will most often encounter on `gfx942` GEMM kernels. It is not exhaustive, but combined with the name-abbreviation rule it lets you decode essentially any token by inspection. Each row lists the token prefix and the parameter it denotes.

| Category | Token prefix | Parameter |
| --- | --- | --- |
| Tile and MFMA | `MT...` | `MacroTile0` x `MacroTile1` x `DepthU` (anchor) |
| Tile and MFMA | `MI...` | `MatrixInstruction` M x N x B (anchor) |
| Tile and MFMA | `MIWT` | `MIWaveTile` |
| Tile and MFMA | `WG` | `WorkGroup` (third value is `LocalSplitU`) |
| Tile and MFMA | `WS` | `WavefrontSize` |
| Tile and MFMA | `MIAV` | `MIArchVgpr` |
| Tile and MFMA | `IU` | `InnerUnroll` |
| Global read | `PGR` | `PrefetchGlobalRead` |
| Global read | `GRVWA` / `GRVWB` | `GlobalReadVectorWidthA` / `...B` |
| Global read | `DTVA` / `DTVB` | `DirectToVgprA` / `...B` |
| Global read | `USFGRO` | `UseSgprForGRO` |
| Global read | `UIOFGRO` | `UseInstOffsetForGRO` |
| Global read | `NLCA` / `NLCB` | `NumLoadsCoalescedA` / `...B` |
| Global read | `GLS` | `GroupLoadStore` |
| Global read | `ULSGRO` | `UnrollLoopSwapGlobalReadOrder` |
| Local read and LDS | `PLR` | `PrefetchLocalRead` |
| Local read and LDS | `LRVW` | `LocalReadVectorWidth` |
| Local read and LDS | `CLR` | `ClusterLocalRead` |
| Local read and LDS | `LDSB` | `1LDSBuffer` |
| Local read and LDS | `LPA` / `LPB` / `LPM` | `LdsPadA` / `...B` / `...Metadata` |
| Local read and LDS | `LBSPPA` / `LBSPPB` / `LBSPPM` | `LdsBlockSizePerPadA` / `...B` / `...Metadata` |
| Local read and LDS | `TLDS` | `TransposeLDS` |
| Scheduling | `SIA` | `ScheduleIterAlg` |
| Scheduling | `GRPM` | `GlobalReadPerMfma` |
| Scheduling | `LWPM` | `LocalWritePerMfma` |
| Scheduling | `EPS` | `ExpandPointerSwap` |
| Scheduling | `ONLL` | `OptNoLoadLoop` |
| Scheduling | `SS` | `SourceSwap` |
| GSU and StreamK | `GSU` | `GlobalSplitU` |
| GSU and StreamK | `GSUA` | `GlobalSplitUAlgorithm` |
| GSU and StreamK | `GSUC` | `GlobalSplitUCoalesced` |
| GSU and StreamK | `GSUWGMRR` | `GlobalSplitUWorkGroupMappingRoundRobin` |
| GSU and StreamK | `SK` | `StreamK` |
| GSU and StreamK | `SKXCCM` | `StreamKXCCMapping` |
| GSU and StreamK | `SKFTR` | `StreamKFixupTreeReduction` |
| Workgroup mapping | `WGM` | `WorkGroupMapping` |
| Workgroup mapping | `WGMXCC` | `WorkGroupMappingXCC` |
| Workgroup mapping | `WGMXCCG` | `WorkGroupMappingXCCGroup` |
| Workgroup mapping | `SU` / `SUS` / `SUM` | `StaggerU` / `StaggerUStride` / `StaggerUMapping` |
| Store and epilogue | `VWA` / `VWB` | `VectorWidthA` / `...B` |
| Store and epilogue | `VS` | `VectorStore` |
| Store and epilogue | `SVW` | `StoreVectorWidth` |
| Store and epilogue | `SRVW` | `StoreRemapVectorWidth` |
| Store and epilogue | `SPO` | `StorePriorityOpt` |
| Store and epilogue | `SSO` | `StoreSyncOpt` |
| Store and epilogue | `NEPBS` | `NumElementsPerBatchStore` |
| Store and epilogue | `AFC` | `ActivationFuncCall` |
| Cache hint | `NT` | `NonTemporal` |
| Cache hint | `NTA` / `NTB` / `NTC` / `NTD` / `NTM` | `NonTemporalA` / `...B` / `...C` / `...D` / `...Metadata` |
| Assertion and meta | `ASEM` | `AssertSummationElementMultiple` |
| Assertion and meta | `AFEM` | `AssertFree0ElementMultiple` then `AssertFree1ElementMultiple` |
| Assertion and meta | `MO` | `MaxOccupancy` |
| Assertion and meta | `USL` | `Use64bShadowLimit` |
| Assertion and meta | `PKA` | `PreloadKernArgs` |
| Assertion and meta | `FDSI` | `ForceDisableShadowInit` |
| Assertion and meta | `CADS` | `ConvertAfterDS` |

Table 2. Common token prefixes on `gfx942` GEMM kernels and the parameters they denote.

When you meet a token that is not in this table, apply the name-abbreviation rule in reverse: list the candidate parameters whose uppercase letters match the prefix, pick the longest valid match, and read the remainder as the value with the type-based rules above.

## Reconstructing the Matrix Instruction

The single most important field to recover correctly is `MatrixInstruction`, because it pins the tile shape and the wave tiling. TensileLite accepts the canonical nine-element form:

```text
[M, N, K, B, MIBlockM, WaveTileM, WaveTileN, WaveM, WaveN]
```

We rebuild it from the name and one hardware-determined lookup:

* `M`, `N`, and `B` come from the `MI` token.
* `WaveTileM` and `WaveTileN` come from the `MIWT` token.
* `MIBlockM` defaults to 1.
* `WaveM` and `WaveN` are derived from the macro tile: `WaveM = MacroTile0 / (M * WaveTileM)` and `WaveN = MacroTile1 / (N * WaveTileN)`.
* `K` is the one value not in the name; it is fixed by the hardware MFMA instruction and must be looked up, as the next subsection covers.

We then validate the reconstruction by checking that `M * WaveTileM * WaveM` equals `MacroTile0` and `N * WaveTileN * WaveN` equals `MacroTile1`.

One detail is easy to miss: the nine-element form does not carry `LocalSplitU`. TensileLite reads it from the third element of `WorkGroup` (see `matrixInstructionToMIParameters` in `tensilelite/Tensile/SolutionStructs/Validators/MatrixInstruction.py`), so the recovered configuration must keep `WorkGroup`; dropping it silently loses `LocalSplitU` for any kernel that splits the summation locally.

### Looking Up the Matrix-Instruction K

K is fixed by the MFMA (Matrix Fused Multiply-Add) instruction the kernel issues, which is fully determined by the architecture, the tile dimensions `M` and `N`, and the input datatype. The authoritative source is the AMD CDNA ISA documentation: each MFMA is named `V_MFMA_<accumulator>_<M>X<N>X<K>_<input-type>`, spelling out M, N, K, and the datatype. A lookup keyed by `(architecture, M, N, datatype)`, built from that instruction set, yields an exact K for every supported shape.

The critical subtlety is that K depends on `M` and `N`, not the datatype alone: the 16x16 and 32x32 instructions of one datatype use different K. Table 3 lists the dense MFMA K values for `gfx942` (CDNA3):

| Instruction shape | bf16 / f16 | fp8 / bf8 | int8 |
| --- | --- | --- | --- |
| 16x16 | 16 | 32 | 32 |
| 32x32 | 8 | 16 | 16 |

Table 3. Matrix-instruction K by instruction shape and datatype on `gfx942`.

Reusing the 16x16 row for 32x32 is an easy mistake; the values really do differ. Newer architectures make this harder: on `gfx950` (CDNA4), double-rate variants let one `(M, N, datatype)` map to two valid K values (a 16x16 `bfloat16` instruction can use K=16 or K=32), and microscaling formats add more shapes, so there you need an extra signal such as `DepthU` (a multiple of K) to disambiguate. For unique-K architectures like `gfx942`, the lookup is unambiguous.

## Recovering the Problem Type

The `ProblemType` fields come from two sources that complement each other:

* The `hipblaslt-bench` command provides the datatypes, transpose flags, and problem size. For example, `--a_type bf16_r` maps to `B`, `--transA T --transB N` maps to a TN layout, and `-m 96 -n 12 -k 5120 --batch_count 1` becomes the exact size `[96, 12, 1, 5120]`.
* The solution-name prefix provides the features that the bench command alone cannot always convey, such as `SwizzleTensorA`, `UseScaleAB`, `UseScaleAlphaVec`, bias support, and the activation set.

On `gfx942`, the 8-bit float types are the FNUZ variants, so `f8_r` maps to the `F8N` character. When the input datatype differs from the compute datatype, `HighPrecisionAccumulate` is enabled.

One nuance is worth calling out, because it changes which kernel you target. The epilogue features encoded in the name, such as the activation set or the scale modes, often describe a general-purpose `UserArgs` kernel that can switch those features off at runtime. In many solution pools there is no narrower kernel for a specific epilogue. For instance, a bias kernel may always be emitted with the activation and scale-alpha-vector features attached, so a pure bias-only name simply does not exist. This gives you two legitimate targets:

* **Reproduce the exact golden kernel.** Enrich the problem type with every epilogue feature encoded in the name. The configuration then pins the precise kernel you observed, including features the benchmark may not actually exercise.
* **Target the leanest kernel for the real workload.** Build the problem type from the benchmark command alone and ignore the name's epilogue tokens. Tuning then generates the kernel whose prefix matches the benchmark's true needs, which can be leaner than the fat `UserArgs` kernel.

Either choice keeps the core solution parameters (tile, matrix instruction, wave tile, and the rest) from the name; they differ only in which epilogue features the problem type declares.

## Worked Example: Decoding a Name by Hand

The rules above are enough to decode a real kernel end to end with nothing but the name and the benchmark command. Consider this benchmark command and the solution name it selected on `gfx942`:

```bash
hipblaslt-bench --api_method c \
    -m 96 -n 12 -k 5120 \
    --transA T --swizzleA --transB N \
    --batch_count 1 \
    --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r \
    --scale_type f32_r --bias_type f32_r --compute_type f32_r
```

```text
Cijk_Alik_Bljk_BBS_STA_BH_UserArgs_MT32x16x256_MI16x16x1_SN_..._DU256_PGR2_PLR1_..._SIA3_SS1_..._VWA1_VWB1_WS64_WG32_4_1_WGM1_WGMXCC4_WGMXCCGn1
```

**Step 1, read the prefix into a problem type.** `Cijk_Alik_Bljk` is the TN layout (`--transA T --transB N`); `BBS` is `bfloat16` input with `bfloat16` output and `float32` compute (`--a_type bf16_r ... --compute_type f32_r`); `STA` sets `SwizzleTensorA` (`--swizzleA`); `BH` sets `UseBeta` and `HighPrecisionAccumulate`. With no `Bias`, `SAV`, or `SABV` token, no bias or scaling is enabled. The size comes from the command as `Exact: [96, 12, 1, 5120]` in `[M, N, batch, K]` order.

**Step 2, read the anchors.** `MT32x16x256` gives `MacroTile0 = 32`, `MacroTile1 = 16`, and `DepthU = 256`. `MI16x16x1` gives matrix-instruction `M = 16`, `N = 16`, `B = 1`.

**Step 3, rebuild the matrix instruction.** The wave tile token is absent, so `MIWaveTile` is `[1, 1]` and `MIBlockM` is 1. K is not in the name; a 16x16 `bfloat16` instruction on `gfx942` gives `K = 16` (a 32x32 one would use `K = 8`, which is why the lookup keys on `M` and `N`, not the datatype alone). From the macro tile, `WaveM = 32 / (16 * 1) = 2` and `WaveN = 16 / (16 * 1) = 1`, giving `MatrixInstruction = [16, 16, 16, 1, 1, 1, 1, 2, 1]`; the checks `16 * 1 * 2 = 32` and `16 * 1 * 1 = 16` match the macro tile.

**Step 4, decode the remaining tokens with the rules.** Walking left to right and applying longest-prefix matching plus the value rules:

* `PGR2` is `PrefetchGlobalRead = 2`.
* `PLR1` is `PrefetchLocalRead = 1`.
* `SIA3` is `ScheduleIterAlg = 3`.
* `SS1` is `SourceSwap = true`.
* `VWA1` and `VWB1` are `VectorWidthA = 1` and `VectorWidthB = 1`.
* `WS64` is `WavefrontSize = 64`.
* `WG32_4_1` is `WorkGroup = [32, 4, 1]`, so `LocalSplitU = 1`.
* `WGM1` is `WorkGroupMapping = 1`.
* `WGMXCC4` is `WorkGroupMappingXCC = 4`.
* `WGMXCCGn1` is `WorkGroupMappingXCCGroup = -1`.

**Step 5, assemble the configuration.** Place the problem-type fields under the `BenchmarkProblems` problem type, the exact size under `BenchmarkFinalParameters`, and every decoded solution parameter as a single-value list under `ForkParameters`. Crucially, keep `WorkGroup` alongside the nine-element `MatrixInstruction`, because its third element carries `LocalSplitU`. The result is a configuration whose entire search space is exactly one kernel: the one you started from.

Because the `ForkParameters` are single-valued, feeding this configuration into the TensileLite workflow reproduces the exact kernel. To explore its neighborhood while preserving the guarantee, widen a few axes into multi-value lists, for example `DepthU: [128, 256]` or `GlobalSplitU: [1, 2, 4]`, and leave the rest fixed. The original kernel remains a member of the enlarged search space, so the run can only match or beat it.

## Caveats and Limitations

This technique is deterministic, but a few practical limits are worth stating clearly.

* **K is recovered from the ISA, not stored in the name.** The matrix-instruction K dimension is fixed by the hardware MFMA instruction, so it is recovered by looking up `(architecture, M, N, datatype)` against the architecture's matrix-instruction set. On unique-K architectures such as `gfx942` this is exact. On `gfx950`, double-rate and microscaling instructions make K ambiguous from the name alone, so an extra signal such as `DepthU` is required to disambiguate. When the kernel exists in the library logic, the logic file also gives the authoritative value.
* **Version drift in tokens.** Names produced by a different TensileLite version can contain tokens that the current parameter set does not recognize, for example a stray marker that no longer maps to any parameter. Treat unrecognized tokens as a signal to check the version, skip the ones with no mapping, and fold derived load tokens back into their source parameter where possible.
* **Defaults are intentional.** A decoded configuration lists only the non-default knobs. Do not pad it with default values. That subset is the faithful tuning input.
* **Architecture-specific datatype mapping.** The 8-bit float mapping is architecture dependent. On `gfx942` the FNUZ variants apply; other architectures use different encodings.
* **`gfx942` is not a single device.** The `gfx942` ISA string is shared by several SKUs (for example MI300A, MI300X, MI308X, and MI325X) and partition modes, and Tensile dispatches by CU count and `DeviceNames`, not by the ISA string. So the pinned configuration must set its `LibraryLogic` `CUCount` and `DeviceNames` to the SKU and partition you actually deploy on, and you should re-check `WorkGroupMappingXCC` (whose valid values are powers of two that divide the CU count, which changes with the partition). Otherwise you recover a valid but poorly tuned kernel.

## Results on MI325X

To check the guarantee in practice, we applied the method to a matrix of 72 GEMMs (eight shapes times three layouts times three precisions) on an AMD Instinct MI325X (`gfx942`, 304 CU, SPX partition). Every case was measured in the same tuning client on the same device, median of three runs. The sweep is now complete, so all 72 cases are reported.

Each GEMM is compared at three points:

* **Library default**: the kernel the library heuristic picks out of the box.
* **Anchor (pool best)**: the fastest existing kernel for that shape, recovered by reverse-engineering and pinned as described above. Because the library default is always one of the candidates, the anchor is by construction no slower than the default.
* **Tuned (final)**: the kernel produced by expanding the search space outward from the anchor.

The central prediction of Figure 2 is that latency is monotone: tuned is at least as fast as the anchor, which is at least as fast as the default. This held in **all 72 cases**, with zero regressions.

### Aggregate by precision

Table 4 aggregates the tuned-versus-default latency reduction by precision.

| Precision | Cases | Median | Mean | Max |
| --- | --- | --- | --- | --- |
| `bfloat16` | 24 | 15.3% | 15.9% | 32.6% |
| `float16` | 24 | 26.0% | 26.5% | 66.2% |
| `fp8` | 24 | 19.0% | 18.2% | 32.0% |
| All | 72 | 18.0% | 20.2% | 66.2% |

Table 4. Latency reduction of the tuned kernel versus the library default, by precision.

### Aggregate by shape

Table 5 breaks the same reductions down by problem shape.

| Shape | M x N x K | Cases | Median | Mean | Max |
| --- | --- | --- | --- | --- | --- |
| `sqS` | 512 x 512 x 512 | 9 | 23.4% | 25.2% | 66.2% |
| `sqL` | 2048 x 2048 x 2048 | 9 | 20.6% | 22.3% | 32.0% |
| `tallM` | 8192 x 1024 x 2048 | 9 | 16.1% | 18.2% | 39.0% |
| `wideN` | 1024 x 8192 x 2048 | 9 | 16.7% | 17.5% | 32.0% |
| `fatK` | 1024 x 1024 x 8192 | 9 | 19.4% | 25.7% | 56.1% |
| `thinK` | 4096 x 4096 x 512 | 9 | 18.2% | 19.0% | 26.1% |
| `decode` | 64 x 8192 x 4096 | 9 | 21.2% | 22.1% | 32.6% |
| `llm` | 4096 x 4096 x 4096 | 9 | 8.5% | 11.6% | 32.0% |

Table 5. Latency reduction of the tuned kernel versus the library default, by problem shape.

Across all 72 cases, the tuned kernel is **18.0%** faster than the library default at the median (20.2% faster on average and up to **66.2%** faster). Reverse-engineering the anchor already recovers a mean 10.6% over the default, and expanding from that anchor adds a further mean 10.6% on top, improving on the anchor in 70 of the 72 cases while never falling below it. The largest gains appear for `float16` and for transpose-N GEMMs with A swizzling; the biggest single case, a small square `float16` GEMM with swizzling, drops from 12.20 us to 4.12 us (66.2%).

### Full per-case results

The complete matrix follows in Table 6, and it doubles as the data for the next section. Latencies are microseconds. `Reverse vs default` is the tuned kernel's reduction over the library default (higher is better); `Generic vs reverse` is how much slower the generic one-shot config, analyzed in the next section, is than the reverse-anchored kernel (positive means slower). The five tiny `sqS` generic winners were re-measured median-of-five.

| Precision | Shape | M x N x K | Layout | Default (us) | Golden (us) | Reverse (us) | Generic (us) | Reverse vs default | Generic vs reverse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bfloat16` | `sqS` | 512 x 512 x 512 | NN | 6.46 | 5.61 | 5.48 | 5.49 | 15.2% | +0.2% |
| `bfloat16` | `sqS` | 512 x 512 x 512 | TN | 4.27 | 4.27 | 4.25 | 4.35 | 0.5% | +2.4% |
| `bfloat16` | `sqS` | 512 x 512 x 512 | TN+swizzle | 5.52 | 4.46 | 3.85 | 3.98 | 30.3% | +3.4% |
| `bfloat16` | `sqL` | 2048 x 2048 x 2048 | NN | 42.04 | 39.56 | 33.15 | 50.53 | 21.1% | +52.4% |
| `bfloat16` | `sqL` | 2048 x 2048 x 2048 | TN | 41.23 | 39.69 | 34.73 | 45.97 | 15.8% | +32.4% |
| `bfloat16` | `sqL` | 2048 x 2048 x 2048 | TN+swizzle | 37.79 | 35.26 | 29.23 | 52.70 | 22.7% | +80.3% |
| `bfloat16` | `tallM` | 8192 x 1024 x 2048 | NN | 64.30 | 62.61 | 53.92 | 71.59 | 16.1% | +32.8% |
| `bfloat16` | `tallM` | 8192 x 1024 x 2048 | TN | 66.19 | 64.33 | 59.73 | 83.37 | 9.8% | +39.6% |
| `bfloat16` | `tallM` | 8192 x 1024 x 2048 | TN+swizzle | 61.98 | 59.27 | 52.86 | 95.67 | 14.7% | +81.0% |
| `bfloat16` | `wideN` | 1024 x 8192 x 2048 | NN | 61.64 | 61.24 | 53.14 | 72.28 | 13.8% | +36.0% |
| `bfloat16` | `wideN` | 1024 x 8192 x 2048 | TN | 68.97 | 68.97 | 61.20 | 81.27 | 11.3% | +32.8% |
| `bfloat16` | `wideN` | 1024 x 8192 x 2048 | TN+swizzle | 56.98 | 56.98 | 47.45 | 90.26 | 16.7% | +90.2% |
| `bfloat16` | `fatK` | 1024 x 1024 x 8192 | NN | 51.68 | 43.66 | 43.66 | 53.17 | 15.5% | +21.8% |
| `bfloat16` | `fatK` | 1024 x 1024 x 8192 | TN | 48.62 | 48.62 | 41.89 | 55.94 | 13.8% | +33.5% |
| `bfloat16` | `fatK` | 1024 x 1024 x 8192 | TN+swizzle | 50.38 | 43.76 | 38.68 | 59.58 | 23.2% | +54.0% |
| `bfloat16` | `thinK` | 4096 x 4096 x 512 | NN | 37.11 | 37.11 | 30.67 | 43.72 | 17.4% | +42.5% |
| `bfloat16` | `thinK` | 4096 x 4096 x 512 | TN | 39.80 | 37.83 | 30.41 | 45.08 | 23.6% | +48.2% |
| `bfloat16` | `thinK` | 4096 x 4096 x 512 | TN+swizzle | 36.74 | 36.74 | 31.02 | 50.42 | 15.6% | +62.5% |
| `bfloat16` | `decode` | 64 x 8192 x 4096 | NN | 29.32 | 27.34 | 25.05 | 29.26 | 14.6% | +16.8% |
| `bfloat16` | `decode` | 64 x 8192 x 4096 | TN | 28.87 | 21.40 | 19.45 | 23.92 | 32.6% | +23.0% |
| `bfloat16` | `decode` | 64 x 8192 x 4096 | TN+swizzle | 26.60 | 26.15 | 22.94 | 25.39 | 13.8% | +10.7% |
| `bfloat16` | `llm` | 4096 x 4096 x 4096 | NN | 194.48 | 185.63 | 180.79 | 258.29 | 7.0% | +42.9% |
| `bfloat16` | `llm` | 4096 x 4096 x 4096 | TN | 232.09 | 208.55 | 204.47 | 288.77 | 11.9% | +41.2% |
| `bfloat16` | `llm` | 4096 x 4096 x 4096 | TN+swizzle | 179.38 | 179.38 | 170.85 | 335.59 | 4.8% | +96.4% |
| `float16` | `sqS` | 512 x 512 x 512 | NN | 7.41 | 5.54 | 5.49 | 5.44 | 25.9% | -0.9% |
| `float16` | `sqS` | 512 x 512 x 512 | TN | 7.97 | 4.79 | 4.20 | 4.32 | 47.3% | +2.9% |
| `float16` | `sqS` | 512 x 512 x 512 | TN+swizzle | 12.20 | 4.48 | 4.12 | 4.37 | 66.2% | +6.1% |
| `float16` | `sqL` | 2048 x 2048 x 2048 | NN | 42.81 | 40.80 | 33.97 | 45.87 | 20.6% | +35.0% |
| `float16` | `sqL` | 2048 x 2048 x 2048 | TN | 50.21 | 38.42 | 35.58 | 47.32 | 29.1% | +33.0% |
| `float16` | `sqL` | 2048 x 2048 x 2048 | TN+swizzle | 48.38 | 38.42 | 32.90 | 52.53 | 32.0% | +59.7% |
| `float16` | `tallM` | 8192 x 1024 x 2048 | NN | 63.66 | 61.02 | 56.05 | 75.59 | 12.0% | +34.9% |
| `float16` | `tallM` | 8192 x 1024 x 2048 | TN | 73.69 | 72.48 | 66.39 | 85.58 | 9.9% | +28.9% |
| `float16` | `tallM` | 8192 x 1024 x 2048 | TN+swizzle | 88.25 | 64.47 | 53.81 | 94.35 | 39.0% | +75.3% |
| `float16` | `wideN` | 1024 x 8192 x 2048 | NN | 63.05 | 62.33 | 58.03 | 75.91 | 8.0% | +30.8% |
| `float16` | `wideN` | 1024 x 8192 x 2048 | TN | 75.89 | 71.10 | 63.09 | 82.89 | 16.9% | +31.4% |
| `float16` | `wideN` | 1024 x 8192 x 2048 | TN+swizzle | 75.46 | 62.05 | 51.28 | 90.57 | 32.0% | +76.6% |
| `float16` | `fatK` | 1024 x 1024 x 8192 | NN | 52.96 | 52.79 | 44.29 | 52.94 | 16.4% | +19.5% |
| `float16` | `fatK` | 1024 x 1024 x 8192 | TN | 87.29 | 43.61 | 43.61 | 55.90 | 50.0% | +28.2% |
| `float16` | `fatK` | 1024 x 1024 x 8192 | TN+swizzle | 101.97 | 54.85 | 44.78 | 61.39 | 56.1% | +37.1% |
| `float16` | `thinK` | 4096 x 4096 x 512 | NN | 42.59 | 37.68 | 31.47 | 44.49 | 26.1% | +41.4% |
| `float16` | `thinK` | 4096 x 4096 x 512 | TN | 40.22 | 38.63 | 32.88 | 46.55 | 18.2% | +41.6% |
| `float16` | `thinK` | 4096 x 4096 x 512 | TN+swizzle | 40.95 | 37.63 | 35.60 | 50.20 | 13.1% | +41.0% |
| `float16` | `decode` | 64 x 8192 x 4096 | NN | 34.19 | 26.21 | 24.18 | 30.11 | 29.3% | +24.5% |
| `float16` | `decode` | 64 x 8192 x 4096 | TN | 28.92 | 24.49 | 21.20 | 23.88 | 26.7% | +12.6% |
| `float16` | `decode` | 64 x 8192 x 4096 | TN+swizzle | 31.75 | 25.12 | 22.25 | 26.11 | 29.9% | +17.3% |
| `float16` | `llm` | 4096 x 4096 x 4096 | NN | 206.33 | 191.19 | 188.82 | 264.18 | 8.5% | +39.9% |
| `float16` | `llm` | 4096 x 4096 x 4096 | TN | 250.13 | 215.09 | 205.62 | 295.48 | 17.8% | +43.7% |
| `float16` | `llm` | 4096 x 4096 x 4096 | TN+swizzle | 198.53 | 198.51 | 190.23 | 335.82 | 4.2% | +76.5% |
| `fp8` | `sqS` | 512 x 512 x 512 | NN | 5.57 | 5.31 | 5.09 | 5.09 | 8.6% | 0.0% |
| `fp8` | `sqS` | 512 x 512 x 512 | TN | 3.99 | 3.99 | 3.63 | 3.89 | 9.0% | +7.2% |
| `fp8` | `sqS` | 512 x 512 x 512 | TN+swizzle | 4.48 | 3.65 | 3.43 | 3.44 | 23.4% | +0.3% |
| `fp8` | `sqL` | 2048 x 2048 x 2048 | NN | 28.98 | 26.59 | 23.06 | 32.77 | 20.4% | +42.1% |
| `fp8` | `sqL` | 2048 x 2048 x 2048 | TN | 26.68 | 26.36 | 21.71 | 26.84 | 18.6% | +23.6% |
| `fp8` | `sqL` | 2048 x 2048 x 2048 | TN+swizzle | 20.25 | 18.13 | 16.14 | 31.16 | 20.3% | +93.1% |
| `fp8` | `tallM` | 8192 x 1024 x 2048 | NN | 44.66 | 44.66 | 37.13 | 49.41 | 16.9% | +33.1% |
| `fp8` | `tallM` | 8192 x 1024 x 2048 | TN | 45.25 | 40.04 | 35.18 | 44.63 | 22.3% | +26.9% |
| `fp8` | `tallM` | 8192 x 1024 x 2048 | TN+swizzle | 39.30 | 38.57 | 30.15 | 49.93 | 23.3% | +65.6% |
| `fp8` | `wideN` | 1024 x 8192 x 2048 | NN | 44.90 | 35.78 | 35.75 | 49.68 | 20.4% | +39.0% |
| `fp8` | `wideN` | 1024 x 8192 x 2048 | TN | 44.52 | 39.35 | 33.46 | 45.99 | 24.8% | +37.4% |
| `fp8` | `wideN` | 1024 x 8192 x 2048 | TN+swizzle | 38.52 | 37.37 | 33.35 | 49.70 | 13.4% | +49.0% |
| `fp8` | `fatK` | 1024 x 1024 x 8192 | NN | 35.33 | 35.33 | 30.31 | 37.36 | 14.2% | +23.3% |
| `fp8` | `fatK` | 1024 x 1024 x 8192 | TN | 30.36 | 25.56 | 24.47 | 32.58 | 19.4% | +33.1% |
| `fp8` | `fatK` | 1024 x 1024 x 8192 | TN+swizzle | 28.93 | 27.83 | 22.35 | 35.08 | 22.7% | +57.0% |
| `fp8` | `thinK` | 4096 x 4096 x 512 | NN | 28.04 | 27.82 | 22.84 | 29.43 | 18.5% | +28.9% |
| `fp8` | `thinK` | 4096 x 4096 x 512 | TN | 30.30 | 27.65 | 23.36 | 29.37 | 22.9% | +25.7% |
| `fp8` | `thinK` | 4096 x 4096 x 512 | TN+swizzle | 25.74 | 25.74 | 21.78 | 32.86 | 15.4% | +50.9% |
| `fp8` | `decode` | 64 x 8192 x 4096 | NN | 17.29 | 15.31 | 14.15 | 20.59 | 18.2% | +45.5% |
| `fp8` | `decode` | 64 x 8192 x 4096 | TN | 15.78 | 12.90 | 12.44 | 14.45 | 21.2% | +16.2% |
| `fp8` | `decode` | 64 x 8192 x 4096 | TN+swizzle | 15.28 | 14.89 | 13.37 | 13.95 | 12.5% | +4.3% |
| `fp8` | `llm` | 4096 x 4096 x 4096 | NN | 123.73 | 116.37 | 105.58 | 154.03 | 14.7% | +45.9% |
| `fp8` | `llm` | 4096 x 4096 x 4096 | TN | 150.82 | 113.51 | 102.50 | 147.68 | 32.0% | +44.1% |
| `fp8` | `llm` | 4096 x 4096 x 4096 | TN+swizzle | 103.15 | 103.15 | 99.97 | 176.79 | 3.1% | +76.8% |

Table 6. Full 72-GEMM results on MI325X. Default, golden (reverse-engineered anchor), reverse-anchored tuned, and generic one-shot winner, with the reverse win and the generic gap in the last two columns.

## A Generic Config Is Not Enough

The guarantee above hinges on one thing: the pool's best kernel being inside the search space. To show what happens without it, we compared reverse-anchored tuning against a generic one-shot config on the same 72-GEMM matrix on MI325X. The generic config is built straight from the official hipBLASLt template: one config per precision and layout, covering all eight shapes with a single shared grid (25 representative power-of-two MacroTiles times `DepthU`, `GlobalSplitU`, `StaggerU`, and `WorkGroupMapping`), about 10,000 candidate solutions per shape, which is roughly ten times the budget of the reverse-anchored run. The point is that this grid is generic: it is not guaranteed to contain each shape's golden kernel. (Winner latencies on the tiny 512-cubed shape were re-measured median-of-five to remove a min-over-many-runs bias.)

As Figure 1 anticipated, the search space is a hard ceiling: a generic config that does not contain the golden cannot return it, no matter how large its budget. The results are decisive:

* Even with about ten times the budget, the generic config is slower than reverse-anchored tuning in **70 of 72 cases** (median +35%, mean +37%, up to **+96%**). The single exception is a 0.9% noise-level tie.
* It is slower than the golden kernel it was supposed to beat in **61 of 72 cases**. This is direct evidence that a generic search space often does not contain the golden.
* It is not even monotone against the library default: faster on 22 cases but slower on 49 (up to +87%). A generic config provides no floor at all.
* The gap is largest for swizzled transpose-N GEMMs, the most tile-shape-sensitive case.

Table 7 counts how often the generic config is slower than each reference point.

| Generic config is slower than | Cases (of 72) |
| --- | --- |
| Reverse-anchored tuned | 70 |
| Golden (pool best) | 61 |
| Library default | 49 |

Table 7. Number of cases (of 72) where the generic config is slower than each reference point.

Table 8 breaks the generic-versus-reverse gap down by layout.

| Layout | Generic vs reverse-anchored (mean) |
| --- | --- |
| NN | +30.3% |
| TN | +28.7% |
| TN + swizzle | +52.7% |

Table 8. Mean gap of the generic config versus reverse-anchored tuning, by layout.

The per-case numbers are already in the full results table above: its `Generic` column against the `Default`, `Golden`, and `Reverse` columns. Every positive `Generic vs reverse` entry is a case where the generic config, despite far more search budget, loses to the reverse-anchored kernel.

Reverse-anchored tuning avoids all of this by construction. Because the golden kernel is pinned into the search space by reverse-engineering, the tuned result satisfies tuned <= golden <= default in every case: strict, monotone, and never worse than where you started. The generic config, lacking that anchor, routinely performs worse than the golden and even worse than the library default. More search budget does not fix this; only putting the golden inside the search space does.

## Summary

In this blog, you learned why a TensileLite run can only return a kernel its configuration can express, which makes the search space the most important design choice in tuning. If your current best kernel, typically the one Offline Tuning selected from the pool, is not inside that space, a re-tune has no guaranteed floor and may regress.

You then learned a deterministic recipe that removes this risk: decode a kernel's solution name back into its exact parameters, rebuild the nine-element `MatrixInstruction` while keeping `WorkGroup` so that `LocalSplitU` survives, recover the `ProblemType` from the name prefix and the `hipblaslt-bench` command, and pin the result as single-value `ForkParameters`. Pinning the pool's best kernel as an anchor and expanding outward without ever dropping it guarantees a tuned kernel at least as fast as the pool's best, and usually faster. On the 72-GEMM matrix on MI325X, this held with zero regressions and a median latency reduction of 18.0% over the library default.

The decode rests on three ideas worth remembering. First, the name's prefix maps back to `ProblemType` fields, while the remaining tokens map back to solution parameters. Second, `MatrixInstruction` must be rebuilt as the nine-element form, with `WorkGroup` retained so that `LocalSplitU` survives. Third, because user-tuned kernels are often absent from the committed library logic, decoding the name directly is the most general recovery path.

This blog focused on establishing the anchor. In a follow-up, we will cover the second half of the method: how to expand the search space outward from the anchor efficiently, deciding which parameters to widen and by how much to reach faster kernels without ever losing the guarantee. Follow the AMD ROCm Blogs for that next installment and for more hipBLASLt and TensileLite tuning deep dives from our team.

## Acknowledgement

We would like to express our thanks to our colleagues [Brian Chang](../../authors/brian-chang.md), [Eveline Chen](../../authors/eveline-chen.md), [Bobo Fang](../../authors/bobo-fang.md), [Bill Ku](../../authors/bill-ku.md), [Kaiping Lu](../../authors/kaiping-lu.md), and [Menghsuan Yang](../../authors/menghsuan-yang.md) for their insightful feedback and technical assistance.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
