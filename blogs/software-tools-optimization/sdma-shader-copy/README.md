---
blogpost: true
blog_title: "Inside hipMemcpy: SDMA vs shader copy"
date: 29 Oct 2026
author: "Clement Lin, Jeremy Liao"
thumbnail: 'sdma-shader-copy.png'
tags: Hardware, Memory, Performance, Optimization, Profiling
category: Software tools & optimizations
target_audience: "HIP and ROCm developers who need to choose or debug hipMemcpy paths on Instinct GPUs. Intermediate-to-advanced; familiar with HIP streams and kernels."
key_value_propositions: "Shows which hardware path hipMemcpy actually takes, why HIP defaults to shader copy for D2D and small H2D, and when SDMA is the right host-copy path because CUs are busy."
language: English
myst:
  html_meta:
    "author": "Clement Lin, Jeremy Liao"
    "description lang=en": "HIP hipMemcpy is two hardware paths. MI308X measurements show when SDMA vs shader copy wins, including the 16 KiB knee."
    "keywords": "HIP, SDMA, shader copy, hipMemcpy, Instinct GPUs, ROCm, memcpy, performance"
    "property=og:locale": "en_US"
    "vertical": "Systems, Developers"
    "amd_category": "Developer Resources"
    "amd_asset_type": "Blog"
    "amd_technical_blog_type": "Tools, Features, and Optimizations"
    "amd_blog_hardware_platforms": "Instinct GPUs"
    "amd_blog_development_tools": "ROCm Software"
    "amd_blog_applications": "AI Inference, AI Training, Design, Simulation & Modeling"
    "amd_blog_topic_categories": "Software & Ecosystem"
    "amd_blog_authors": "Clement Lin, Jeremy Liao"
---

# Inside hipMemcpy: SDMA vs shader copy

`hipMemcpy` looks like one API. On ROCm™ the same bytes can take two hardware paths.
**Shader copy** dispatches a blit kernel on the HIP stream’s AQL compute queue and
moves data with CU waves. **SDMA** writes packets onto a separate SDMA ring and lets
a DMA engine copy without using CUs.

Measurements in this post used one AMD Instinct™ MI308X GPU (gfx942, SPX, 80 CUs,
192 GiB) with TheRock nightly 10.2.0a20260925 (HIP 7.17 / ROCr 1.21), matching
`ROCm/rocm-systems` @ 820ea79c. Timing is `hipMemcpyAsync` plus `hipEvent`, with
the path forced except where the HIP default is plotted. Results may vary.

```{note}
HIP’s default is not “SDMA is the fast path.” Same-GPU D2D always uses a CLR blit
kernel, at every size. H2D / D2H switch at 16 KiB. The chart that matters is not
the 26× D2D peak. It is what happens to H2D / D2H when the CUs are already busy.
```

## The two paths differ in who does the work

| | Shader copy (CLR blit kernel) | SDMA (DMA engine) |
| --- | --- | --- |
| Who | Runs on CU waves; contends with user kernels | Independent DMA engine (two per GPU on this SPX node) |
| Queue | HIP stream’s own AQL compute queue | ROCr `BlitSdma` RLC queue — not the stream queue |
| Packet | 64 B AQL dispatch + kernarg | `COPY_LINEAR` and related SDMA packets |
| Ordering | Ordered with later kernels on the same queue via the barrier bit | Cross-queue ordering via HSA signals (`POLL_REGMEM` / barrier-AND) |
| CLR picks it for | D2D and small copies: same-queue ordering, no cross-engine signal | Large H2D / D2H / P2P: does not consume compute |

There are two shader-copy implementations. Do not mix them up.

HIP’s default is the **CLR blit kernel** `__amd_rocclr_copyBuffer`, compiled at
runtime and dispatched on the HIP stream queue. **ROCr BlitKernel**
(`CopyAligned` / `CopyMisaligned`) is hand-written ISA on ROCr’s internal blit
queue. `HSA_ENABLE_SDMA=0`, a failed SDMA queue, or the A8 redirect below all
land on ROCr BlitKernel — not the CLR kernel HIP uses every day.

## How CLR chooses a path

The fork is `useShaderCopyBufferPath()` in `rocblit.cpp`. HIP defaults:

| Copy | When | Path |
| --- | --- | --- |
| Same-GPU D2D | any size | Shader |
| D2D + `hipMemcpyDeviceToDeviceNoCU` | any size | SDMA (`force_copy_on_sdma`, no blit fallback) |
| Pinned H2D / D2H | ≤ 16 KiB | Shader |
| Pinned H2D / D2H | > 16 KiB | SDMA |
| Pageable H2D / D2H | ≤ 16 KiB: staging + shader; 16–64 KiB: staging + SDMA; > 64 KiB: pin + SDMA | Shader, then SDMA |
| P2P / IPC / VMM import | ≤ 1 MiB shader (WG cap 16); > 1 MiB SDMA | Shader / SDMA |

16 KiB is `GPU_FORCE_BLIT_COPY_SIZE` (OpenCL defaults it to 0). 1 MiB is
`ROC_P2P_SDMA_SIZE`. If an SDMA submit fails, CLR falls back to the blit kernel
unless `DEBUG_CLR_DISABLE_FALLBACK` or `requireSDMA` is set. The source comment
for default D2D is blunt: resolving an SDMA dependent signal is slow; a
same-queue blit is cheaper. The numbers below agree.

## A8: a same-GPU SDMA API call is not always SDMA

Even after the caller reaches ROCr’s `hsa_amd_memory_async_copy_on_engine`,
same-GPU copies hit another gate in `GpuAgent::DmaCopyOnEngine`:

```cpp
engine_offset = is_same_gpu && !force_copy_on_sdma ? BlitDevToDev : engine_offset;
```

Same GPU and no `force_copy_on_sdma` rewrites the engine to `BlitDevToDev` —
ROCr BlitKernel at `blits_[0]`. Logs still look like a DMA copy. The hardware is
not SDMA.

Default HIP D2D never reaches this API; CLR calls `shaderCopyBuffer` directly.
You hit A8 if you:

- call the HSA SDMA API for a same-GPU copy without `force_copy_on_sdma`
- copy IPC / VMM-imported pointers that CLR treats as P2P-like (> 1 MiB intended
  SDMA, then redirected if ROCr sees one GPU)
- set `HSA_ENABLE_SDMA=0`, which forces every blit to `BlitDevToDev`

The HIP switch that actually forces same-GPU D2D onto SDMA is
`hipMemcpyDeviceToDeviceNoCU`.

## D2D: the default is shader because it is much faster

The default heuristic mixes path changes into a size sweep. These runs pin the
path: D2D shader via `hipMemcpyDeviceToDevice`, D2D SDMA via
`hipMemcpyDeviceToDeviceNoCU`.

| D2D shader peak | D2D SDMA saturate | Peak ratio | Small-copy launch |
| --- | --- | --- | --- |
| 1514 GB/s @ 64 MiB | ~58 GB/s | 26× | 3.2 vs 8.5 µs |

```{figure} ./images/d2d-bandwidth.png
:align: center
:alt: On-device D2D bandwidth for shader copy versus SDMA across transfer sizes
:width: 80%
Figure 1: On-device D2D bandwidth. Shader uses 80 CUs into HBM. SDMA is a narrow DMA engine, not the HBM peak. `hipMemcpyDeviceToDeviceNoCU` means “do not use CUs”, not “go faster”.
```

```{figure} ./images/d2d-latency.png
:align: center
:alt: On-device D2D latency for shader copy versus SDMA in the launch-bound range
:width: 80%
Figure 2: D2D latency for the launch-bound range. Shader sits at ~3.2 µs; SDMA at ~8.5–9.7 µs until the copy itself grows. That gap is the cross-engine signal, not the few kilobytes moved. Past 1 MiB, time follows bandwidth (shader 44 µs at 64 MiB, SDMA 1.2 ms).
```

## H2D: the 16 KiB policy is visible on the curve

Pinned H2D with HIP’s default next to both paths forced. The default line follows
shader through 16 KiB, then snaps onto SDMA at 32 KiB.

```{figure} ./images/h2d-bandwidth.png
:align: center
:alt: Pinned H2D bandwidth for HIP default, forced shader copy, and forced SDMA
:width: 80%
Figure 3: Pinned H2D bandwidth. The dashed marker is `GPU_FORCE_BLIT_COPY_SIZE` = 16 KiB. At 32 KiB the default is slower than the 16 KiB shader copy: the threshold spends bandwidth to stop using CUs. By 16 MiB every path is ~53–55 GB/s — the host link, not the engine.
```

```{figure} ./images/h2d-latency.png
:align: center
:alt: Pinned H2D latency through 1 MiB showing the 16 KiB policy knee
:width: 80%
Figure 4: Pinned H2D latency through 1 MiB, where the policy knee is visible. Default tracks shader (~3 µs) through 16 KiB, then jumps to ~10 µs at 32 KiB with SDMA. D2H is the same shape: 16 KiB default ~2.9 µs / 5.7 GB/s, 32 KiB ~9.0 µs / 3.6 GB/s.
```

## When CUs are full, that is why SDMA exists

The D2D curves invite the wrong question. Idle H2D / D2H already tie on PCIe.
The useful score is wall-clock while compute is running. A second HIP stream
filled all 80 CUs with an ALU spin (grid 80×8, block 256, almost no HBM
traffic), then copied 16 MiB.

| Path | Idle GB/s | CUs occupied GB/s |
| --- | --- | --- |
| D2D shader | 1316 | 0.36 |
| H2D shader | 54.0 | 0.36 |
| H2D SDMA | 53.3 | 53.2 |
| D2H SDMA | 54.8 | 54.9 |

```{figure} ./images/cu-occupied.png
:align: center
:alt: 16 MiB H2D and D2H bandwidth with CUs idle versus fully occupied
:width: 80%
Figure 5: 16 MiB, forced paths. Shader bars collapse; SDMA bars do not move. D2D shader is omitted from the plot (1316 vs 55 does not share a y-axis). D2D + `hipMemcpyDeviceToDeviceNoCU` under this infinite occupy stuck in `hipEventSynchronize` — the engine should not wait on CUs; the HIP event completion path did. H2D / D2H SDMA did not stick, so that comparison holds.
```

```{important}
Runtime is not picking the faster engine. Large host copies use SDMA because
only SDMA still delivers PCIe peak while compute is running. A finite ~10 ms
kernel, instead of a spin that never yields, slows shader H2D (~55 → 33 GB/s)
rather than freezing it. Same 64 MiB × 8 work: SDMA makespan 10.2 ms, shader
16.5 ms.
```

## What to actually use

1. **Leave same-GPU D2D on the default.** It is already the CLR blit kernel.
   `DeviceToDeviceNoCU` drops large copies from TB/s to ~58 GB/s on this MI308X.
2. **`hipMemcpyDeviceToDeviceNoCU` means “the copy must not eat CUs.”** It is
   not a speed switch.
3. **Keep small H2D / D2H on the 16 KiB threshold.** Setting
   `GPU_FORCE_BLIT_COPY_SIZE=0` makes tiny copies pay the SDMA launch tax.
4. **Large H2D / D2H should stay on SDMA.** Idle bandwidth matches shader; under
   a full CU grid shader falls to 0.36 GB/s and SDMA stays ~53 GB/s.
5. **Do not use `HSA_ENABLE_SDMA=0` to “compare against CLR blit.”** That is
   ROCr BlitKernel.
6. **If you call the HSA SDMA API for a same-GPU copy, set
   `force_copy_on_sdma`.** Otherwise `DmaCopyOnEngine` silently sends you to
   `blits_[0]`.
7. **SDMA queue slots are a per-node bitmap shared by processes.** Exhaustion
   logs `No more SDMA queue to allocate`; that process’s H2D / D2H falls back
   to ROCr BlitKernel and starts using CUs.

## Summary

`hipMemcpy` is two engines. Same-GPU D2D belongs on the CLR blit kernel; large
H2D / D2H belong on SDMA because only SDMA still delivers host-link bandwidth
when the CUs are busy. The 16 KiB H2D / D2H switch is a runtime policy, not a
hardware crossover, and `hipMemcpyDeviceToDeviceNoCU` means “do not use CUs,”
not “go faster.”

## Disclaimers

The information presented in this document is for informational purposes only
and may contain technical inaccuracies, omissions, and typographical errors.
The information contained herein is subject to change and may be rendered
inaccurate for many reasons, including but not limited to product and roadmap
changes, component and motherboard version changes, new model and/or product
releases, product differences between differing manufacturers, software
changes, BIOS flashes, firmware upgrades, or the like. Any computer system has
risks of security vulnerabilities that cannot be completely prevented or
mitigated. AMD assumes no obligation to update or otherwise correct or revise
this information. However, AMD reserves the right to revise this information
and to make changes from time to time to the content hereof without obligation
of AMD to notify any person of such revisions or changes. THIS INFORMATION IS
PROVIDED “AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO
THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS,
OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS
ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY
PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY
RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING
FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY
ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

AMD, the AMD Arrow logo, AMD Instinct, ROCm, HIP, and combinations thereof are
trademarks of Advanced Micro Devices, Inc. Other product names used in this
publication are for identification purposes only and may be trademarks of their
respective companies.

© 2026 Advanced Micro Devices, Inc. All rights reserved.
