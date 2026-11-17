---
blogpost: true
blog_title: "An AITER Pin Keyed by Kernel Text: Faster Glue Than the Inductor Default"
date: "17 Nov 2026"
author: "Yuchen Lin, Huan Li, Chunhung Wang, Chao Chen"
thumbnail: 'inductor-aiter-conversion-thumbnail.png'
tags: "AI/ML, Performance"
category: "Software tools & optimizations"
target_audience: "AI developers and enthusiasts"
key_value_propositions: "An AITER pin keyed by the kernel text carries across compiles and beats Inductor's single default glue config. Against open-list autotune it is faster on one model and inside the compile-to-compile spread on the other."
language: English
myst:
    html_meta:
        "author": "Yuchen Lin, Huan Li, Chunhung Wang, Chao Chen"
        "description lang=en": "Learn how an AITER pin keyed by kernel text beats Inductor's default glue config, and how it compares with open-list autotune."
        "keywords": "Triton, TorchInductor, AOTInductor, AITER, rocprof, kernel tuning, AMD Instinct, ROCm"
        "vertical": "AI, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Yuchen Lin, Huan Li, Chunhung Wang, Chao Chen"
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

# An AITER Pin Keyed by Kernel Text: Faster Glue Than the Inductor Default

GEMM and attention kernels already have offline tuning paths. The kernels around them often do not. A compiled inference graph still launches a long tail of small Triton kernels for casts, concatenations, normalizations, activations, gathers, and histogram-style updates. TorchInductor generates those kernels. With autotune on, it times a candidate list and writes the winner into a cache keyed by a fusion number. Figure 1, below, is why that cache is not what ships.

```{figure} ./images/why-pin.png
:align: center
:alt: Autotune cache keyed by a fusion number, beside a pinned config keyed by the kernel text.

Figure 1. The autotune cache is one compile's winner. A pin is the kernel text plus one checked config, written into the package.
```

Two limits of that flag are why the search is pulled out of the compile. It is a closed choice: it times the list the compiler emits, and the winner it stores is the config you get. A faster tile that the list does not contain is not a choice it can make. Another process on the GPU can make the fastest config in that list look slow. Model A's three open-list compiles were not screened for a second process on the GPU. What they measured is an unstable winner: 1618.9, 1769.8, and 1863.5 us, a spread of 245 us. The offline search runs apart from that compile, keeps a config only when the check passes and the trace is faster, and writes it into the package. Startup does not time candidates. On glue the kernel speedup against Inductor's single default config is 1.388x on Model A and 2.846x on Model B. Against the open-list autotune, Model B is 1.15x. Model A's edge over its fastest autotune compile is smaller than the 245 us spread, so it is not a listed speedup. Table 1 and Table 9 hold the times.

That tail is the glue in this post: pointwise, reduction, and persistent-reduction kernels. The tables compare the three stages.

* **Inductor** keeps the TorchInductor body and turns glue autotune off, so every glue kernel launches the compiler's single default config.
* **AITER** replaces that body with another Triton function that has the same arguments, and leaves the config at the Inductor default. AITER here is a library of Triton kernels for AMD Instinct GPUs. A body that fails the numerical check stays on the compiler source.
* **Tuned** is the package that ships. On Model A, 12 kept configs are in the compiler list and 55 are not. Forty-six of those 55 change a block, warp, or stage, counted in the stage definition below.

In the tables, the AITER row is that body swap at the default config. The AITER pin in the title is the Tuned config file. A fourth build is a control, not a stage. It puts that file back on the Inductor body, which is the last row of Figure 2.

Two unnamed production models are the measurements. Model A has 143 glue kernels and Model B has 25. GEMM and attention template timings are copied across builds. The glue autotune cache is not, so a glue change is not a GEMM being selected again. The GEMM binary stays the same, though its duration can still move. Request latency, profiler off, is the end-to-end number.

In this blog, you will learn why that search leaves the Inductor compile, how a pin is keyed by the kernel text, and what the traces say about a body swap and a config change.

## Where This Is Worth Doing

The shape is fixed, so a measured config is the one production launches. A unit in the last place (ULP) is one step of a bf16 or fp16 value. A kernel that fails the check stays on the compiler body. This does not help when the hot path is still an untuned GEMM, the shape changes every step, or the time is a launch the body cannot fuse away.

## What the Glue Kernels Are

TorchInductor fuses elementwise work, small reductions, and indexing into Triton so the graph does not bounce through global memory between ops. A pointwise kernel owns `XBLOCK` elements. A reduction loops over `RBLOCK` and then collapses it. A persistent reduction keeps that axis in one tile. Warp count, stages, and `waves_per_eu` sit beside the tile. A wave on this GPU is 64 threads. `waves_per_eu` is how many of those waves the launch asks to keep resident on one execution unit.

## From a Compiler Kernel to a Pinned Triton Kernel

The conversion does not rewrite the graph. The compiled graph still calls one function per fusion, with the same pointers and the same grid. Two things about that function are allowed to change: the launch config, and the statements inside the `@triton.jit` function.

### Capture

Compile the graph once the way production compiles it, and keep the generated Python for every glue kernel. Each file is a heuristic decorator plus one `@triton.jit` function. The decorator records the signature, the size hints, and the configs autotune is allowed to try. The function is the body.

Kernels are matched by a fingerprint: the SHA-256 of the `@triton.jit` function with the numeric suffix removed, truncated to 20 hex characters. Inductor renumbers fusions when an upstream template changes, so the suffix is not an identity. When two pins share a fingerprint and disagree, a pin whose kernel name still matches is applied by name.

The artifact that ships is an AOTInductor package plus the runtime graph. The build uses one compile thread, so the patch is visible to the compiler, and the log records how many pins and bodies were applied.

### The Three Stages

Figure 2, below, changes the body and the config one at a time. Doing both in one step would credit a tile to the body, or a body to the tile. The control then puts the pins back on the compiler body.

```{figure} ./images/why-two-changes.png
:align: center
:alt: Four stages in a column. The body changes only at the swap. The config changes only at the pin. A control puts the pins back on the compiler body.

Figure 2. Each step changes one thing. The AITER row names both models: three SiLU kernels on Model A, and on Model B one file that differs only in literals. AITER keeps the default config, so a difference there would belong to the body. Tuned then changes the config. The control puts those pins on the compiler body.
```

The Inductor stage turns glue autotune off and keeps the compiler body. On Model A, max-autotune still offers several candidates, so the stage slices that list to the first config. On Model B the flag alone yields one config, and that build is the Inductor package below. Every later ratio divides by this stage.

The AITER stage swaps the `@triton.jit` source and leaves the signature, the decorator, and the arguments alone. On Model A the swap is three SiLU kernels, `a * sigmoid(a) * b`, out of 143. On Model B, 24 of 25 files match the compiler text and the 25th differs only in constants the build reinitializes. The one-config slice stays. An earlier Model B install left the list open and moved the tile, so the AITER row is a rebuild that cuts the list to one config first. The open-list build is reported beside it.

Each kernel is screened for three rounds of five repetitions. The three fastest that pass are confirmed for seven rounds of ten, interleaved with the baseline. A candidate is kept when its median is at least 0.5 percent faster and its maximum over those seven rounds sits below the baseline's minimum. That interval is the min and the max, not a confidence interval. Inside that noise, the grid that can fill the 80 compute units wins. A kernel the sweep cannot improve keeps the config it arrived with.

On Model A, pointwise candidates use `XBLOCK` from 64 up to the power of two covering the launch, warps in {1, 2, 4, 8}, `waves_per_eu` in {0, 2}, and one stage. Reduction candidates use `R0_BLOCK` from 64 to 4096, the same warps, and one or two stages. A tile that would put more than 64 elements on a thread is dropped. Of the 143, 69 are pointwise, 64 are persistent reductions, and 10 are reductions. A pointwise kernel that also has a `YBLOCK` axis is searched on both tiles together, with the product of the two blocks kept between 256 and 16384 elements. Across the 143 that is 5220 candidates. 1329 failed the bitwise check: 119 pointwise, 789 persistent reduction, and 421 reduction. None failed to compile. 67 kernels took a new config, and the other 76 kept the baseline. Of the 67, 12 match a config the compiler heuristic emits for that kernel's size hints on the open-list compile. The other 55 do not. Nine of the 55 keep the block size, warp count, and stage count and set `waves_per_eu` to 2. The other 46 change a block size, the warp count, or the stage count. Those 46 are the wider part of the search. Model B refuses a tile larger than the config in hand, so the `XBLOCK` 128 already chosen by the compiler list stayed the baseline. The 119 pointwise failures are five kernels, none of them atomic and none a reduction, split 45, 45, 17, 8, and 4. The first three are a masked split across inputs and a divide. The other two are masked pointwise kernels as well: 38 of 46 candidates passed on one, and 41 of 45 on the other. The remaining 64 pointwise kernels had no gate failures.

On Model B, `XBLOCK` is 1 or a power of two from 8 to 4096, warps are {1, 2, 4, 8, 16}, stages are 1 to 4, and `waves_per_eu` is {0, 2, 4, 8}. The tile cannot exceed the config in hand, and the grid stays under about a thousand waves per compute unit. Of the 25 kernels, 17 took a new config, 6 kept the previous one because nothing cleared 0.5 percent, and 2 had no passing candidate.

A fast sigmoid, allowed one ULP of bf16 error, made glue slower on both models and was not kept. A reciprocal hoist that leaves the stored bits unchanged is in the ceiling section. It is not in the Tuned package.

The rule used to accept or reject a body is the same one used for the config sweep.

* Integer outputs stay equal.
* bf16 and fp16 may move by at most one ULP when the change is an explicit fast-math form such as the sigmoid. Everything else stays bitwise.
* fp32 stays bitwise. Atomics are compared after the outputs are zeroed, because addition order changes the low bits. Buffers that are also inputs are left intact. Clearing those inputs caused false rejects.

A body that fails is dropped. The compiler's version of that kernel stays.

## How the Traces Were Read

Kernel time comes from `rocprofv3` kernel traces. The benchmark's duration field ends the run, because the tool writes the file only when the process exits on its own.

```bash
rocprofv3 --kernel-trace --output-format csv -d prof -o trace -- <benchmark>
```

Each row is one dispatch. Duration is end minus start. The forward count is the most common glue-kernel call count, and glue time is the sum of per-forward durations over the glue kernels only.

Three clocks stay separate. Issued time sums every dispatch, so overlap is counted twice. Glue share is issued glue divided by all issued time, a fraction of issued work. Device busy time is the union of the intervals, so one microsecond with two kernels in flight counts once. Request latency is a different harness, profiler off, the median of the benchmark histogram. A completion rate at a fixed stream count is not used.

Byte counters are a second pair of passes. `FETCH_SIZE` and `WRITE_SIZE` cannot be collected together. The ceiling used later is a lower bound, not a peak a math-heavy kernel is supposed to hit.

```text
ceiling time = max(1.7 us, bytes / 3.4 TB/s)
percent      = ceiling time / measured time
```

The 1.7 us term is a launch floor. Seven kernels sit between 1.26 and 2.09 us, so percent can exceed 100. The 3.4 TB/s term is a large bf16 copy on this card whose dimensions were not kept, so it cannot be recomputed here. "At the ceiling" means 80 percent or above. Atomics, a cache miss, and the arithmetic sit outside the bound.

## Test Environment

| Component | Specification |
| --- | --- |
| **CPU** | 2x Intel Xeon Platinum 8480C, 56 cores per socket with hyperthreading, 224 logical CPUs, 2 NUMA nodes |
| **Memory** | 2.0 TiB |
| **OS** | Linux, kernel 5.10.134 |
| **GPU** | GFX942 (80 CUs) |
| **GPU driver** | amdgpu 6.16.13 |
| **ROCm** | 7.2.2 |
| **PyTorch** | 2.9.1+rocm7.2.2 |
| **Triton** | 3.5.1+rocm7.2.2 |

Model A traces use one stream on four GPUs, five rounds, unless a row says otherwise. Its latency uses ten streams, one GPU, six rounds of 120 seconds, profiler off. Model B uses six host threads and one stream. Inductor and Tuned traces there are five rounds of 45 seconds. The AITER rebuild and the config control are three rounds of 45 seconds. Latency is up to six rounds of 120 seconds, and rounds with another process on the GPU were dropped. The tables compare medians. Percentiles are not the comparison.

## Model A: 143 Glue Kernels

Table 1 is issued glue per forward, one stream: the mean of four per-GPU medians of five rounds. Every speedup is the mean of the four per-GPU ratios against that GPU's own Inductor median. The ratio of the averaged times is separate. For Tuned it is 2134.3 / 1540.1 = 1.386x, the mean of the ratios is 1.388x, and their median is 1.372x. Table 2 is that column. For the three open-list compiles the mean of the ratios is 1.319x, 1.206x, and 1.146x. Dividing the averaged times gives 1.318x, 1.206x, and 1.145x.

| Stage | What changed | Glue us/forward | vs default config | Glue share |
| --- | --- | --- | --- | --- |
| Default config | Compiler body, list sliced to one | 2134.3 | 1.000x | 60.0% |
| Open list | Compiler body, list left open, three compiles | 1618.9, 1769.8, 1863.5 | 1.319x, 1.206x, 1.146x | 53.6%, 55.6%, 56.6% |
| AITER | Three SiLU bodies, that same one config | 2116.0 | 1.009x | 59.6% |
| Tuned | Pinned config on that body | 1540.1 | 1.388x | 52.3% |

Table 1. Model A glue, one stream. Default, AITER, and Tuned are means of four GPUs. The open-list cells are the three compiles in the same order, fastest glue first. Every ratio is the mean of the four per-GPU ratios, and the 1.388x is against the default config only. The open-list shares are issued glue divided by all issued time, in that same order. As you can see in Figure 3, below, the open-list compiles already sit next to Tuned, and the three of them are not the same compile.

```{figure} ./images/glue-by-stage.png
:align: center
:alt: Glue time per forward for the default config, three open-list compiles, the body swap, and the pinned package on two models.

Figure 3. Kernel speedup versus Inductor's single default config. On Model A the open-list compiles are 1.319x, 1.206x, and 1.146x, and the pin is 1.388x. On Model B the open-list compile is 2.470x and the pin is 2.846x. Bar labels are rounded to the nearest microsecond, so a ratio read off the bars will not match these speedups. The Model A speedups are the mean of the per-GPU ratios in Table 1.
```

| GPU | Inductor us | Tuned us | Ratio |
| --- | --- | --- | --- |
| 4 | 2145.1 | 1453.0 | 1.476x |
| 5 | 2109.7 | 1584.9 | 1.331x |
| 6 | 2138.9 | 1598.7 | 1.338x |
| 7 | 2143.7 | 1523.8 | 1.407x |

Table 2. Model A per-GPU glue medians, one stream, five rounds. The 1.388x in Table 1 is the mean of this ratio column. The 1.372x is its median.

Table 1 is a later replay of the pinned package. It is not the timing log from the search.

The three open-list compiles use max-autotune with no pins and retain the same GEMM and attention kernel names as the default build. The second and third were seeded from the first compile's template cache. Glue and latency do not rank together: 1769.8 us has the slowest latency in Table 4. The 245 us spread among the three compiles is larger than Tuned's edge over the fastest. The search was sharded across GPUs 0, 1, 4, and 5. Table 2 replays the package on GPUs 4 to 7.

Table 3 splits issued time by category. Attention is flat. hipBLASLt GEMM moves by 6 us, the Triton GEMM template by 25 us, and glue by 594 us.

| Stage | Glue | hipBLASLt GEMM | Triton GEMM template | Attention | Other |
| --- | --- | --- | --- | --- | --- |
| Inductor | 2134.3 | 672.7 | 349.8 | 122.2 | 279.0 |
| Tuned | 1540.1 | 678.9 | 325.1 | 122.3 | 277.4 |

Table 3. Model A issued microseconds per forward by category, one stream, mean of four GPU medians. GEMM template choices were copied across stages. The template column still moved by 25 us, which is small next to the glue column and is not zero.

### Request Latency, Profiler Off

Table 4 is unprofiled request time, ten streams, one GPU. Each 120-second round contributes its latency-histogram median, and the table is the median of six rounds. The 99th percentile is omitted. Times are milliseconds.

| Stage | Median latency (ms) | Round range (ms) |
| --- | --- | --- |
| Default config | 20.57 | 20.50 to 20.63 |
| Open list, three compiles | 16.14, 16.46, 16.29 | 15.96 to 16.21, 16.38 to 16.57, 16.20 to 16.34 |
| AITER | 20.60 | 20.46 to 20.71 |
| Tuned | 15.25 | 15.16 to 15.32 |

Table 4. Model A unprofiled request latency, ten streams, one GPU, six rounds. The open-list order matches Table 1. Tuned at 15.25 ms is 0.89 ms under the fastest open-list compile, and the other two, in that same order, are 1.21 ms and 1.04 ms slower than Tuned.

The body swap leaves the median where Inductor put it. The config retune is the step that moves it. Table 5 is one 30-second profiled window of the same ten-stream harness, one window per stage, for device busy time.

| Stage | Issued glue us/forward | Device busy us/forward | All issued / busy |
| --- | --- | --- | --- |
| Inductor | 3809 | 2277 | 2.83 |
| AITER | 3680 | 2219 | 2.81 |
| Tuned | 2388 | 1775 | 2.54 |

Table 5. Model A, ten streams, one profiled window of 30 seconds. Device busy time is the union of dispatch intervals. The last column is all issued time divided by busy time. Issued glue divided by busy time is smaller (3809 / 2277 = 1.67 at Inductor).

Table 6 is one forward on one stream. The gap before the next forward, near 0.11 ms, is excluded. Figure 4, below, draws the mean of each pair.

| Window | Stage | Device us | Wait us | Span us |
| --- | --- | --- | --- | --- |
| 1 | Default | 5998 | 15301 | 21299 |
| 2 | Default | 6004 | 15655 | 21659 |
| 1 | Open list | 4670 | 13854 | 18524 |
| 2 | Open list | 4705 | 13618 | 18323 |
| 1 | AITER | 6008 | 15263 | 21271 |
| 2 | AITER | 5897 | 16545 | 22442 |
| 1 | Tuned | 4442 | 13579 | 18021 |
| 2 | Tuned | 4524 | 12904 | 17428 |

Table 6. Model A, one forward on one stream. Device and span are medians across forwards in that window. Wait is span minus device, so each row adds. The median of the positive gaps inside the forward differs from this wait by at most 148 us, because kernels on the stream overlap. That 148 us is the gap between two definitions of one row. Open list is the fastest of the three compiles.

```{figure} ./images/forward-split.png
:align: center
:alt: Stacked bars of device time and in-forward wait. Model A includes the fastest open-list compile.

Figure 4. Time inside one forward, mean of two windows. The gray band is span minus device. On Model A the open-list compile is close to the pinned package, and the body swap stays with the default.
```

Against Tuned, those windows average 699 us more span: 204.5 us device and 494.5 us wait. The 494.5 us exceeds the 148 us residual, so the wait gap is not the overlap term. Issued glue is 2602.5 and 2581.2 us against Tuned 2367.7 and 2396.1 us. The other categories are a net 21 us higher, not a sum of absolute moves. Glue is a window mean, 210 us, and device is a median, 204.5 us. Adding 210 and 21 leaves 26 us against that device figure, inside the 148 us residual. The 79 us is the one-stream mean in Table 1. The 0.89 ms latency gap is the unprofiled harness, and most of this profiled gap is wait.

### The Same Configs on the Inductor Body for Model A

The Tuned configs were searched while the AITER bodies were installed. Model A replaces three SiLU bodies. The other 140 glue kernels keep the compiler source. Table 7 puts the Tuned config file on the Inductor body, with autotune still off, and measures it on GPU 6, the same GPU as that row of Table 2. Of 143 pins, 135 were applied. Two fingerprints were ambiguous, and eight pins were not applied. On one Tuned round on GPU 6 those eight kernels are 63 us of 1593 us of glue, about 4 percent. The five rounds on that GPU already span 1468 to 1583 us.

| Build on GPU 6 | Glue us/forward | Ratio vs this GPU's Inductor |
| --- | --- | --- |
| Inductor | 2138.9 | 1.000x |
| AITER, same config | 2107.8 | 1.015x |
| Tuned | 1598.7 | 1.338x |
| Inductor body, Tuned configs | 1567.2 | 1.365x |

Table 7. Model A control, one stream, GPU 6. Inductor, AITER, and Tuned are the medians of five rounds from the four-GPU set. The last row is the median of five new rounds: 1468, 1543, 1567, 1581, and 1583 us.

On this GPU the kernel speedup versus Inductor is 1.365x for the compiler body with the Tuned configs, and 1.338x for Tuned. The two fast rows match on glue. The three SiLU replacements are the only bodies swapped, so they are not required for that speedup. Request latency matches across six unprofiled rounds.

### What the Ceiling Says About the 143

At Tuned, **7 of 143** glue kernels are at or above 80 percent of the ceiling. All seven are short dispatches, 1.26 to 2.09 us, already about one launch.

On the byte-counter trace the 143 glue kernels sum to about 1573 us per forward, the same order as the 1540.1 us mean in Table 1 and not that average. Table 8 is that heavy end. It is not a launch:

| Role | Launches per forward | us per launch | us per forward | Percent of ceiling | What the time is |
| --- | --- | --- | --- | --- | --- |
| Histogram reduction | 1 | 192.3 | 192.3 | 1% | Little traffic (about 12 GB/s). The body is a short reduction, then atomics into a handful of bins. |
| Zero fill | 19 | 4.65 | 88.3 | 36% | The body is a store of zero. The launches are the cost. |
| Wide concatenation | 6 | 10.67 | 64.0 | 64% | About 2.2 TB/s. Bytes dominate. |
| Second histogram reduction | 1 | 53.8 | 53.8 | 5% | Many atomics, modest traffic. |
| Second zero fill | 10 | 5.12 | 51.2 | 33% | Same pattern as the first fill: repeated dispatch. |
| Concatenation plus SiLU | 2 | 23.31 | 46.6 | 44% | About 1.5 TB/s. |
| Third histogram reduction | 1 | 37.0 | 37.0 | 5% | Atomics, low traffic. |
| Fourth histogram reduction | 1 | 36.6 | 36.6 | 5% | Atomics, low traffic. |

Table 8. The slowest glue roles on Model A at Tuned, from one kernel trace plus byte counters. Percent is ceiling time divided by measured time. The 1.26 us dispatches elsewhere in the same model are shorter than the 1.7 us floor, so percent there is above 100. Roles are grouped by what the kernel does, not by compiler name.

Across five rounds on GPU 4, hoisting the divisor out of the largest histogram kernel reduces the number of divides from 512 to 32 while keeping the bin counts bitwise identical. The kernel moves from 192.3 us (192.2 to 192.6) to 89.9 us (89.7 to 90.0). Glue does not: the medians are 1453 us against 1466 us. On one paired round the glue sum is 1453.0 us against 1466.3 us, up 13.3 us. The histogram kernel falls 102.62 us. Thirty-eight kernels slower by more than 1 us sum to +113.29 us, four faster by more than 1 us sum to -5.43 us, and 100 kernels that move by at most 1 us sum to +8.06 us. Those pieces add to 13.3 us. The edit stays out. A block-histogram rewrite measured 193.7 us on one trace and was not kept.

## Model B: 25 Glue Kernels

Model B is the smaller model. Table 9 has the three stages and the open-list compile. Inductor and Tuned are medians of five 45-second traces. AITER is the rebuild that installs the body files and cuts the candidate list to one config, median of three 45-second traces on one GPU. The speedup column is that stage's glue time divided into Inductor's, so it matches the time column.

| Stage | Glue us/forward | vs default config | Request latency (ms) |
| --- | --- | --- | --- |
| Default config | 998.56 | 1.000x | 4.866 |
| Open list | 404.3 | 2.470x | 2.998 |
| AITER | 1009.9 | 0.989x | 4.837 |
| Tuned | 350.88 | 2.846x | 2.935 |

Table 9. Model B. Glue for the default and Tuned rows is the five-round median. Open list is the compile that left the candidate list on. AITER is the one-config body rebuild. The 2.846x is against the default config. Against the open-list row, Tuned is 1.15x on glue, and 2.935 ms sits inside the open-list latency range 2.916 to 3.076 ms.

Of the 25 body files, 24 match the Inductor `@triton.jit` text. The 25th differs in literal constants the build reinitializes. The slow Inductor tile launches one program per row, grid `xnumel`, over a reduction of length 16. Tuned uses `XBLOCK` 128, grid `ceil(xnumel / 128)`, so one program owns 128 rows and the launch is amortized. The offline sweep did not find that 128. Its baseline was already 128, two warps, four stages, from an earlier compile that left the candidate list open. The rule that refuses a larger tile never had to admit 128. The compiler's own list selected it. The single-default grid, `xnumel`, is the Inductor stage.

A repeat of the open-list package measured 399.8, 416.4, and 419.1 us. That row is a tile change, which is why it is not the AITER row.

Table 10 splits the same traces by category. Each cell is a median of five rounds, so a row is not one round's sum. hipBLASLt falls by 165 us, attention by 28 us, other by 56 us, and glue by 648 us. Across the five rounds hipBLASLt is 703 to 855 us at Inductor and 575 to 635 us at Tuned. Those ranges do not overlap, so the drop is not only a shift of the median.

| Stage | Glue | hipBLASLt GEMM | Attention | Other |
| --- | --- | --- | --- | --- |
| Inductor | 998.6 | 758.1 | 146.4 | 138.0 |
| Tuned | 350.9 | 593.3 | 118.0 | 81.7 |

Table 10. Model B issued microseconds per forward by category, median of five rounds. There is no separate Triton GEMM template column on this model. The hipBLASLt column moves, so a claim that only glue changed would be wrong here.

Table 11 is unprofiled request latency. Rounds with another process on the GPU were dropped.

| Stage | Rounds kept | Median latency (ms) | 99th percentile (ms) |
| --- | --- | --- | --- |
| Inductor | 4 | 4.866 | 5.098 |
| AITER | 6 | 4.837 | omitted |
| Tuned | 5 | 2.935 | 3.155 |

Table 11. Model B unprofiled request latency, six host threads, one stream, 120-second rounds. Rounds with another process on the GPU were dropped. The median is the comparison. AITER's 99th percentile is omitted: two of six rounds had a dirty tail. Its median is 4.837 ms, range 4.765 to 4.903. Open-list latency is in Table 9.

Table 12 is two profiled windows of one forward on the stream:

| Window | Stage | Device us | Wait us | Span us |
| --- | --- | --- | --- | --- |
| 1 | Inductor | 1902 | 2635 | 4537 |
| 2 | Inductor | 1905 | 2708 | 4613 |
| 1 | AITER | 1942 | 2856 | 4798 |
| 2 | AITER | 1932 | 2693 | 4625 |
| 1 | Tuned | 1048 | 1951 | 2999 |
| 2 | Tuned | 1042 | 2033 | 3075 |

Table 12. Model B, one forward on the stream. Device and span are medians. Wait is span minus device, so each row adds. The gap before the next forward stays near 0.03 ms and is not in the wait column. The median of the positive gaps is within 16 us of this wait.

Mean span falls from 4575 us to 3037 us, by 1538 us: 858.5 us of device time and 679.5 us of wait. AITER stays with Inductor, which matches its 4.837 ms latency.

### The Same Configs on the Inductor Body for Model B

Table 13 is the control. The Tuned pin file is applied to the Inductor body, with no body replacement. All 25 pins were applied. The grid of the largest kernel is `ceil(xnumel / 128)`, the same tile Tuned ships, and the second kernel is `ceil(xnumel / 32)`, also Tuned's tile.

| Build | Glue us/forward | Median request latency (ms) |
| --- | --- | --- |
| Inductor | 998.56 | 4.866 |
| AITER, one config | 1009.9 | 4.837 |
| Inductor body, Tuned configs | 391.2 | 2.953 |
| Tuned | 350.88 | 2.935 |

Table 13. Model B control. The config row is the median of three profiled rounds (381.7, 391.2, 403.1 us) and of four unprofiled latency rounds (2.943 to 2.964 ms). AITER's latency is the six-round median from Table 11.

On glue the compiler body with the same pins is 2.55x versus Inductor, and Tuned is 2.846x. The two fast rows match. Request latency matches.

None of the 25 kernels is at 80 percent of the ceiling. The closest is a residual normalization at 64 percent, 14.5 us against a ceiling near 9.2 us.

## Summary

The pin stays in the package. The kernel speedup against Inductor's single default config is 1.388x on Model A (with a median per-GPU ratio of 1.372x) and 2.846x on Model B. Against the open-list autotune, Model B glue is 1.15x. Model A's edge over its fastest autotune compile is smaller than the 245 us spread. Model B request latency stays inside the open-list range. The one-config body swap does not move the kernel ratio. The same pins on the compiler body reproduce the package. A body edit that moves one kernel and not the forward stays out.

In this blog, you explored why an autotune cache keyed by a fusion number does not survive the next compile, and how a pin keyed by the kernel text keeps one checked config in the package. The traces separate a body swap from a config change, and they keep the open-list comparison inside what the repeats can support. Follow the AMD ROCm Blogs for more kernel tuning from our team.

## Acknowledgement

We would like to express our thanks to our colleagues [Brian Chang](../../authors/brian-chang.md), [Eveline Chen](../../authors/eveline-chen.md), [Bobo Fang](../../authors/bobo-fang.md), [Bill Ku](../../authors/bill-ku.md), [Clement Lin](../../authors/clement-lin.md), [Kaiping Lu](../../authors/kaiping-lu.md), and [Menghsuan Yang](../../authors/menghsuan-yang.md) for their insightful feedback and technical assistance.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED "AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
