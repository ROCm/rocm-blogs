---
blogpost: true
blog_title: "From Wave64 to Wave32: Rebuilding gsplat Kernels for AMD MI45x"
date: "17 November 2026"
author: "Deeksha Goplani, Karthik Kashyap Thatipamula, Grant Pinkert, Vikas C Sajjan"
thumbnail: "mi45x-gsplat-wave32-thumbnail.png"
tags: "Gaussian Splatting, gsplat, HIP, PyTorch, AMD Instinct, ROCm, Performance"
category: "Software tools & optimizations"
target_audience: "GPU kernel engineers and researchers porting CUDA graphics and AI code to AMD Instinct"
key_value_propositions: "What changes when the gsplat 3DGS kernels move from wave64 on MI300 to wave32 on MI45x, and which of those changes are worth making"
language: English
myst:
    html_meta:
        "author": "Deeksha Goplani, Karthik Kashyap Thatipamula, Grant Pinkert, Vikas C Sajjan"
        "description lang=en": "How the gsplat backward rasterizer was rebuilt for wave32 on AMD Instinct MI45x, from 6.244 ms to 1.541 ms per launch."
        "keywords": "Gaussian Splatting, gsplat, HIP, PyTorch, MI45x, ROCm, wave32, 3DGS"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "Software & Tools"
        "amd_blog_authors": "Deeksha Goplani, Karthik Kashyap Thatipamula, Grant Pinkert, Vikas C Sajjan"
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

# From Wave64 to Wave32: Rebuilding gsplat Kernels for AMD MI45x

[gsplat](https://github.com/AMD-Ecosystem/gsplat) renders a scene as millions of 3D Gaussians. Its hot kernels were written for a 64-lane wave. AMD Instinct MI45x (gfx1250) runs a 32-lane wave, and a recompile is not enough: the same 256-thread backward rasterizer issues twice as many atomics and its warp reduction drops onto the LDS pipe. This post rebuilds that kernel, then the four kernels that show up once it is no longer the bottleneck. On the Mip-NeRF 360 stump scene the backward rasterizer goes from 6.244 ms to 1.541 ms per launch, and a 30,000-step training run averages 88.7 steps/s.

## Environment and setup

| | |
| --- | --- |
| GPU | AMD Instinct MI45x, gfx1250, A0, wave32 |
| Baseline GPU | AMD Instinct MI325X, gfx942, wave64 |
| ROCm | 10.3.0a20261105, nightly `whl-next` |
| PyTorch | 2.14.0+rocm10.3.0a20261105 |
| Python | 3.12 |
| Source | [AMD-Ecosystem/gsplat](https://github.com/AMD-Ecosystem/gsplat), branch `sync/mi45x_support` |

On this nightly the A0 device enumerates as `gfx1250-strict` unless `HSA_DISABLE_GFX12_STRICT=1` is set, and kernel loads then fail. Create the virtual environment outside the repository. If the ROCm SDK sits inside the tree, hipify rewrites its headers and the build dies with redefinition errors.

```bash
git clone --branch sync/mi45x_support https://github.com/AMD-Ecosystem/gsplat.git
cd gsplat
git submodule update --init --recursive

python3.12 -m venv ~/venvs/gsplat-eco
source ~/venvs/gsplat-eco/bin/activate

python -m pip install -i https://nightly.repo.amd.com/rocm/whl-next/ --pre \
  "rocm[libraries,devel]==10.3.0a20261105" \
  "rocm-sdk-device-gfx1250==10.3.0a20261105"

python -m pip install -i https://nightly.repo.amd.com/rocm/whl-next/ --pre \
  "torch==2.14.0+rocm10.3.0a20261105" \
  "amd-torch-device-gfx1250==2.14.0+rocm10.3.0a20261105" \
  "torchvision==0.29.0+rocm10.3.0a20261105" \
  "amd-torchvision-device-gfx1250==0.29.0+rocm10.3.0a20261105"

source ./path.sh
export PYTORCH_ROCM_ARCH=gfx1250
rm -rf build dist gsplat/hip
python setup.py bdist_wheel
pip install dist/gsplat-*.whl
pip install -r examples/requirements.txt

cd examples
mkdir -p data/360_v2
curl -L -o /tmp/360_v2.zip http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip /tmp/360_v2.zip "stump/*" -d data/360_v2

export HSA_DISABLE_GFX12_STRICT=1
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --disable_viewer \
  --data_dir data/360_v2/stump/ --data_factor 4 \
  --result_dir ./results/stump
```

`torchvision` has to be the ROCm wheel. `torchmetrics` imports it, and the PyPI build fails with `operator torchvision::nms does not exist`.

## The 3DGS pipeline: project, sort, composite

```{figure} ./images/3dgs-pipeline.jpg
:align: center
:alt: Three stages of 3D Gaussian Splatting. Project maps Gaussians to 2D ellipses, sort assigns them to tiles and orders them by depth, and composite blends them front to back.
Project, sort, and composite.
```

Rendering runs in three stages. **Project** maps each Gaussian to a 2D ellipse on the image plane and evaluates its color for the current view. **Sort** assigns each ellipse to the tiles it overlaps and orders those entries by depth. **Composite** walks each pixel's list front to back, blending by opacity and remaining transmittance, and stops when transmittance is near zero.

```{figure} ./images/rasterizer-forward-backward.png
:align: center
:alt: Forward rasterizer walks pixels front to back with one thread per pixel. Backward walks Gaussians back to front, reduces partial gradients across the wave, and writes one atomic per Gaussian per tile.
Forward and backward rasterizers.
```

The backward kernel reverses the walk. It goes back to front, and instead of recomputing transmittance from 1 it reconstructs it from the saved final value. The mapping flips too: each lane takes a turn owning one Gaussian and broadcasts it to the other lanes, because many pixels now contribute to the same Gaussian. Those partial gradients have to be summed across the wave before a single atomic write per Gaussian per tile. That reduction is the piece of machinery the forward pass never needed.

## Porting for MI45x

MI45x is wave32, so the hardcoded warp width of 64 had to become a build-time choice before the kernels would run correctly.

* **Warp width is a build flag.** `setup.py` reads the target architecture and passes `-DGSPLAT_WARP_SIZE=32` for gfx1250 and `64` for CDNA.
* **A missing flag fails the build.** If `GSPLAT_WARP_SIZE` is not defined, compilation stops with `#error`. A wrong width would otherwise pass and corrupt results.
* **The substitution is mechanical.** Across `Utils.cuh`, both `Projection*` kernels, and both `RasterizeToPixels*Bwd` kernels, every `rocprim::warp_reduce<float,64>`, `cg::tiled_partition<64>`, `threadIdx.x % 64`, and `for (i < 64)` lane loop now uses `GSPLAT_WARP_SIZE`.
* **The wave64 fast path stays gated.** It is wrapped in `#if GSPLAT_WARP_SIZE == 64`. On MI45x the build uses the generic kernel.

## Profiling on MI45x

Right after the port, end-to-end 3DGS training of the Mip-NeRF 360 stump scene through `simple_trainer.py` ran at about 16 steps/s. The same benchmark on MI325X, which shares the gfx942 wave64 architecture with MI300X, was 49 steps/s.

The profile below is `torch.profiler` around step 21100. On ROCm its "CUDA" columns are HIP device time. `TILE_SIZE=16` selects the generic backward kernel. The default tile of 8 dispatches the multi-tile kernel from the next section, so a profile without this variable will not reproduce the table. Compare runs on `Time avg`, not `Self CUDA`: the total scales with the number of profiled calls.

```bash
cd /path/to/gsplat/examples
PROFILE=1 PROFILE_STEPS=3 PROFILE_START=21100 TILE_SIZE=16 \
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
  --disable_viewer --data_factor 4 --render_traj_path ellipse \
  --data_dir data/360_v2/stump/ --result_dir results/benchmark/stump/
```

| Kernel | Self CUDA | Self CUDA % | Time avg | Calls |
| --- | --- | --- | --- | --- |
| `rasterize_to_pixels_3dgs_bwd_kernel` | 18.731 ms | 10.42% | 6.244 ms | 3 |
| `rasterize_to_pixels_3dgs_fwd_kernel` | 11.308 ms | 6.29% | 3.769 ms | 3 |
| `projection_ewa_3dgs_fused_bwd_kernel` | 8.919 ms | 4.96% | 2.973 ms | 3 |

The backward rasterizer is the largest kernel in the profile, 10.4% of device time. Per launch it averages 6.244 ms. The 18.731 ms figure is that average times the three profiled calls.

### Why it isn't a recompile

The gfx942 reduction does not carry over. On wave64 the 64-lane sum is a DPP `row_shr` / `row_bcast` ladder and stays in VGPRs. gfx1250 has no `row_bcast:15` or `row_bcast:31`, so a pure DPP 32-lane reduction is not expressible. `rocprim::warp_reduce<float,32>` lowers the cross-16 step to `ds_swizzle_b32` / `ds_bpermute_b32`, and those LDS-unit ops serialize on `SQ_WAIT_INST_LDS` in this kernel.

Atomic traffic doubles. The 16×16 tile is still a 256-thread block, which is 4 waves on wave64 and 8 waves on wave32. Each wave reduces its partial gradients and one lane writes them, so a Gaussian gets 4 atomic writes per tile on wave64 and 8 on wave32, to the same addresses.

## The new kernel: 4-wave multi-tile

Three different kernels show up in the rest of the post, and the measurements do not transfer between them.

* The generic kernel, `rasterize_to_pixels_3dgs_bwd_kernel` at tile 16, is what the profile above measured.
* The multi-tile kernel, `rasterize_bs32_multitile_to_pixels_3dgs_bwd_kernel` at tile 8, replaces it.
* The optimizations in the next section sit on top of the multi-tile kernel.

A 128-thread workgroup is 4 wave32 waves. Each wave rasterizes a different 8×8 tile: 64 pixels across 32 lanes, so each lane owns 2 pixels. The waves share nothing. Because the tiles are disjoint there is no `block.sync()`, no shared Gaussian batch, divergent per-wave trip counts are safe, and a whole-wave early return cannot deadlock the block. The workgroup is a packing container. It asks the hardware for 4 waves of resources at once. It is not a cooperation unit.

```text
tile_linear = blockIdx.y * MT_WARPS + warp_id     // MT_WARPS = 4
grid        = { I, ceil(tile_height * tile_width / MT_WARPS), 1 }
threads     = { 128, 1, 1 }
```

Tile size is baked into the intersection and binning structures upstream. Shrinking it from the generic kernel's 16 grows the Gaussian-to-tile intersection list. Tile 8 is the size this kernel is built around: 64 pixels fill two pixels per lane on a 32-lane wave.

### Why 4 waves and not 8

The kernel was originally 8 waves (256 threads). Cutting the group to 4 waves (128 threads) measured:

| CDIM | 8-wave to 4-wave |
| --- | --- |
| 3 | neutral |
| 8 | 4.8% faster |
| 16 | 2.6% faster |

Register allocation is identical either way. The win is packing. At CDIM=8 the register budget allows 36 resident waves per CU (9 waves per SIMD32, times 4 SIMD32s per CU). Waves only arrive in whole workgroups:

| Group size | Groups that fit | Waves resident | Wasted |
| --- | --- | --- | --- |
| 8 waves | floor(36/8) = 4 | 32 | 4 waves of registers |
| 4 waves | floor(36/4) = 9 | 36 | 0 |

The kernel did not change. Only how its waves pack onto a CU did. Going smaller than 4 strands capacity the other way: a CU has a fixed number of workgroup slots as well as a register budget, and below 4 waves per group the slots run out first. Four waves also map one-to-one onto the CU's four SIMD32s, so each group puts one wave on each SIMD.

## Optimizations on top of the new kernel

### Getting LDS out of the inner loop

The multi-tile kernel staged four arrays in shared memory: `s_vrc`, `s_tf`, `s_va`, and `s_buf`. Each was indexed only by the lane's own pixel (`p = lane + 32*s`) and never read across lanes, so the shared memory added latency and no reuse. All four were offsets into one `extern __shared__` block. The compiler could not prove the `s_buf` stores did not alias the others, and it reissued the loop-invariant reads on every inner iteration.

Three changes, shipped together, remove that dynamic shared memory from the inner loop.

* Gaussian colors are broadcast with `warp.shfl`, as the conic and `xy`/opacity already were.
* Per-pixel state (`v_render_colors`, `T_final`, `v_render_alpha`) moves into registers as `r_vrc`, `r_tf`, and `r_va`. `r_vrc` is still `2·CDIM` registers, one vector for each of the lane's two pixels.
* The CDIM-length color accumulator collapses to one scalar per pixel, `r_sdot`.

`buf_k` is only ever consumed dotted against `vrc_k`, which is constant per pixel:

```text
v_alpha += Σ_k (rgb_k · T − buf_k · ra) · vrc_k
         = T · Σ_k rgb_k·vrc_k  −  ra · Σ_k buf_k·vrc_k
         = T · R                −  ra · S

where R := Σ_k rgb_k·vrc_k  and  S := Σ_k buf_k·vrc_k
      buf_k += rgb_k · fac   ⟹   S += fac · R
```

The kernel uses the current `S` in `v_alpha`, then updates it. `R` is recomputed for each Gaussian and then discarded. The reassociation replaces `buf[CDIM]` with one running scalar per pixel: 2 VGPRs instead of `2·CDIM`. At CDIM=3 the accumulator update drops from about 12 VALU operations to about 6. The dot that builds `R` is still CDIM long, and the exponential, conic, and means work is unchanged. The sum is reassociated, so this result is checked against tolerance rather than bit-exact.

What remains in shared memory is the static rocprim scratch for the bin reduction, outside this loop. Holding `r_vrc` in registers is also the change that drops occupancy from 12 to 10 waves per SIMD. `-ffast-math` below gets those waves back.

### Rebuilding the warp reduction

`rocprim::warp_reduce<float,32>` lowers to LDS-unit operations on gfx1250, and the CDNA cross-16 DPP modes that would avoid them are not on this part. The replacement keeps the intra-16 reduction on the DPP `row_shr` ladder and does the single cross-16 combine with `v_readlane_b32`, which is VALU and does not touch the LDS unit.

```cpp
__device__ __forceinline__ float dpp_permlane_warpSum32(float v) {
    v = dpp_row_shr_add<0x118>(v); // row_shr:8
    v = dpp_row_shr_add<0x114>(v); // row_shr:4
    v = dpp_row_shr_add<0x112>(v); // row_shr:2
    v = dpp_row_shr_add<0x111>(v); // row_shr:1
    int vi = __float_as_int(v);
    float lo = __int_as_float(__builtin_amdgcn_readlane(vi, 15)); // sum(0..15)
    float hi = __int_as_float(__builtin_amdgcn_readlane(vi, 31)); // sum(16..31)
    return lo + hi;                                               // all lanes
}
```

After the ladder, lane 15 holds the sum of lanes 0 through 15 and lane 31 holds the sum of lanes 16 through 31. Two readlanes give every lane the full sum. This is an all-reduce, which is what the lane-scatter epilogue needs: any lane can write any component.

### Lane-scatter atomic epilogue

The conventional epilogue leaves every reduced component in lane 0, and that lane issues `NGRAD` atomic adds one after another. Lane-scatter gives each of lanes `[0, NGRAD)` one component and one address, so the wave issues one `global_atomic_add_f32` with `NGRAD` lanes active.

```text
NGRAD = CDIM + 6 + (ABSGRAD ? 2 : 0)

Lane map (CDIM=3, ABSGRAD=0, so NGRAD=9):
  [0, CDIM)        -> v_colors[CDIM*g + k]
  CDIM+0 .. CDIM+2 -> v_conics[3*g + 0..2]
  CDIM+3 .. CDIM+4 -> v_means2d[2*g + 0..1]
  CDIM+5           -> v_opacities[g]
  CDIM+6 .. CDIM+7 -> v_means2d_abs[2*g + 0..1]   (ABSGRAD only)
```

The atomic count and the values stay the same. Each component is still one `f32` atomic add to the same address, so the result is bitwise identical. What changes is the issue pattern: the writes go out together. `NGRAD` is 9 at CDIM=3 and 22 at CDIM=16 with `ABSGRAD` off. `static_assert(NGRAD <= 32)` caps the kernel. With `ABSGRAD` on, `NGRAD = CDIM + 8`, so CDIM=24 fills the wave and is the hard ceiling.

### Phase split

Each lane owns 2 pixels, so each Gaussian requires 2 `__expf` evaluations, and the two chains share no data. Guarding each with its own `if (valid)` writes `EXEC` between them, which splits the basic block. The scheduler cannot overlap the two transcendentals, and the second waits behind the first.

The fix is a two-phase inner loop.

Phase 1 evaluates the Gaussian test for both pixels in one straight-line block, so the two `__expf` calls issue back to back and their latency overlaps. `px` and `py` are initialized for outside pixels too, and the `__expf` result is consumed only in phase 2 when `valid_s` is set, so an invalid pixel costs the extra VALU and nothing else.

Phase 2 runs the gradient bodies, which read and write per-pixel state (`T[s]`, `r_sdot[s]`), so those keep one `EXEC` region each.

The validity test uses bitwise `&` for the same reason. `&&` short-circuits, and each short circuit is another `EXEC` write in the middle of phase 1.

```cpp
// Bitwise & on purpose: && short-circuits, and each short
// circuit is another EXEC write in the middle of phase 1.
valid_s[s] = (batch_end - t <= bin_final_arr[s]) &
             !(sigma < 0.f) & !(alpha_s[s] < ALPHA_THRESHOLD);
```

Outside pixels get `bin_final = -1` rather than a separate `inside[]` array, so the existing range test excludes them.

### `-ffast-math`

The CUDA path has always built with `nvcc --use_fast_math`. The HIP port never carried that across, so `1.0f / (1.0f - alpha)` compiled to the IEEE division sequence (`v_div_scale`, `v_div_fmas`, `v_div_fixup`) instead of one `v_rcp_f32`. That divide sits on the critical path of every Gaussian, twice per lane.

| Metric | Before | After |
| --- | --- | --- |
| VGPR | 83 | 74 |
| Occupancy | 10 waves/SIMD | 12 |
| VALU instructions | — | −13.7% |

The two changes fit together. Moving per-pixel state into registers spent occupancy (12 to 10 waves per SIMD) to buy latency, and `-ffast-math` returns it (10 to 12) while cutting VALU. The flag also turns on finite-math-only, so the compiler may drop NaN and Inf handling around tests such as `!(sigma < 0.f)`. Rebuild with `FAST_MATH=0` to compare the IEEE divide.

### `-munsafe-fp-atomics`

This flag emits a hardware `global_atomic_add_f32` instead of a compare-and-swap loop. It matters here because wave32 already doubled the atomic count per Gaussian, and the lane-scatter epilogue issues those atomics across lanes. The hardware instruction does not match the CAS loop on denormals. These atomics target device-allocated gradient buffers, which is the case this flag is meant for.

### `ABSGRAD` as a template parameter

`v_means2d_abs` is optional and usually absent. Making it a compile-time template parameter rather than a runtime `nullptr` check means that when the caller does not request absolute gradients, the accumulator, its warp reduction, and its atomics are compiled out. `NGRAD` drops by 2.

## The remaining kernels

The backward rasterizer went from 6.244 ms per launch to 1.541 ms. That moves the bottleneck. Five kernels are left, none of them that rasterizer, and each one fails differently.

The times below are the minimum of 15 dispatches from `rocprofv3`, on the same stump benchmark. `projection_ewa_3dgs_packed_bwd` is not exercised by stump. Its numbers come from a separate large-scale multi-camera workload in packed mode. Every delta in this section is a reduction in kernel time.

| Kernel | Before | After | Delta |
| --- | --- | --- | --- |
| `projection_ewa_3dgs_fused_bwd` | 1.141 ms | 0.287 ms | −74.8% |
| `intersect_tile` (pass 2) | 1.924 ms | 0.231 ms | −88.0% |
| `spherical_harmonics_bwd` | 0.918 ms | 0.747 ms | −18.6% |
| `rasterize_to_pixels_3dgs_fwd` | 1.618 ms | 1.395 ms | −13.8% (CDIM ≤ 8) |
| `projection_ewa_3dgs_packed_bwd` | 2.452 ms | 0.911 ms | −62.9% |

### A reduction that could never find anything

`projection_ewa_3dgs_fused_bwd` wrote its gradients back by labelling lanes with `gid`. For each gradient it scanned the whole wave with `__shfl_sync` to build a label mask, reduced only the lanes that matched, then scanned again to elect a leader. ROCm has no `cg::labeled_partition`, so this was hand-written. Means plus covariance is two of those scans. Means plus quaternion and scale is three.

The labels make the scan useless. `gid = idx % N` over consecutive `idx`, so when `N` is at least the wave size every active lane carries a distinct `gid`. Early exits remove lanes, but they cannot create a duplicate. Every launch, each scan searched for a second member that the index arithmetic made impossible.

For `N >= GSPLAT_WARP_SIZE` that path is gone. The original scan stays as the fallback below it. Means, covariance, and the quaternion/scale pair all take the same branch. The `v_viewmats` reduction stays too: it is labelled by `cid`, and lanes of one camera do share that label.

```cpp
// gid = idx % N over consecutive idx, so for N >= the wave size every
// lane's gid differs. The scan can never find a second member.
if (N >= GSPLAT_WARP_SIZE) {
    const bool excl = (C == 1);   // this thread owns the address grid-wide
    scalar_t *p = v_means + bid * N * 3 + gid * 3;
    if (excl) {
        p[0] = v_mean.x; p[1] = v_mean.y; p[2] = v_mean.z;  // buffer is zeroed
    } else {
        unsafeAtomicAdd(p,     v_mean.x);                   // no in-wave reduction
        unsafeAtomicAdd(p + 1, v_mean.y);
        unsafeAtomicAdd(p + 2, v_mean.z);
    }
}
```

With `C == 1` each address is written by exactly one thread in the grid, so the atomic add onto a zeroed buffer becomes a plain store. With more cameras the same Gaussian is written from several warps, and the atomic add is what combines them. Kernel time dropped 74.8%. The scan was correct for any labels, which is why the impossible case survived. Check what a reduction's labels can actually be before assuming it reduces anything.

### The same machinery, where it is load-bearing

`projection_ewa_3dgs_packed_bwd` runs the same emulation, and the argument above does not transfer. Here `gid = gaussian_ids[idx]` is data dependent. A store would drop gradients: the same Gaussian is written once per camera, usually from different warps, and those stores would overwrite each other. Inside one wave a shared `gid` is uncommon, because a camera row lays Gaussians out in order, but the atomics are what make the write correct either way. Dropping the in-wave reduction needs no `N` guard. Every lane `atomicAdd`s its own contribution.

The `v_viewmats` reduction is kept. It is labelled by `cid`. Lanes from the same camera share that label, so the partition really does have more than one member. Kernel time dropped 62.9%.

### `intersect_tile`: one rectangle at a time, with the whole wave

Pass 2 writes one entry per (Gaussian, tile) intersection. Each thread walked its own Gaussian's tile bounding box serially, so a wave retired only when its slowest lane did. The longest rectangle in a wave was 6.0× the mean, and 83% of lane-slots were idle. About 3 in 4 lanes hold no rectangle at all. That is a separate count from the 6.0× imbalance.

The fix inverts the mapping. The wave takes its rectangles one at a time and drains each with all 32 lanes. A `__ballot_sync` over `my_cnt != 0` builds the source list, so empty lanes are never visited. A fixed 32-source loop would pay a shuffle for `cnt`, `w`, `ty`, `tx`, `base`, `hdr`, and `sidx` on each of them. Row and column advance with adds. The only divisions are `lane / w` and `GSPLAT_WARP_SIZE / w`, once per rectangle.

```cpp
unsigned long long src_mask = __ballot_sync(m, my_cnt != 0);
while (src_mask != 0) {
    const int src = __ffsll(src_mask) - 1;
    src_mask &= ~(1ULL << src);
    // wave-uniform
    const uint32_t cnt = __shfl_sync(m, my_cnt, src);
    const uint32_t w   = __shfl_sync(m, my_w,   src);
    /* ... ty, tx, base, hdr, sidx likewise ... */

    uint32_t r = lane / w, c = lane - r * w;
    const uint32_t dr = GSPLAT_WARP_SIZE / w;
    const uint32_t dc = GSPLAT_WARP_SIZE - dr * w;   // == warp size % w
    for (uint32_t k = lane; k < cnt; k += GSPLAT_WARP_SIZE) {
        const int64_t tile_id = (int64_t)(ty + r) * tile_width + (tx + c);
        isect_ids[base + k]   = hdr | (tile_id << 32);
        flatten_ids[base + k] = sidx;
        r += dr; c += dc;
        if (c >= w) { c -= w; ++r; }  // dc < w and c < w: one correction suffices
    }
}
```

Output is bit-identical. Kernel time dropped 88.0%, from 1.924 ms to 0.231 ms.

No lane may return before the cooperative loop. Every lane writes a strided share of each rectangle (`k = lane, lane+32, …`), so a lane that exits early drops its entries. Lanes with no rectangle stay resident with `my_cnt = 0` and fall through with everyone else.

### `spherical_harmonics_bwd`: one thread per Gaussian

The kernel parallelised over `N × 3`, one thread per (Gaussian, channel). Three threads loaded the same direction, evaluated the same spherical-harmonic basis, and then contended on the same `v_dirs` address with atomics.

One thread per Gaussian covering all three channels gives three independent FMA chains in one lane, the basis evaluated once, and `v_dir` accumulated in the thread so three same-address atomics become three stores. Kernel time dropped 18.6%, and the grid shrinks by 3×. This kernel was never short of parallelism.

One trap: `sh_coeffs_to_color_fast_vjp` assigns to `*v_dir` rather than accumulating into it, so the caller has to sum the per-channel results.

### The forward rasterizer, and where the win stops

Two changes to the LDS batch. Pack each Gaussian into two `float4` values, so an entry is one `ds_load_b128` plus one `ds_load_b96` off a single base, replacing four loads off three addresses. Then prefetch entry `t+1` before consuming entry `t`, so its LDS latency overlaps this iteration's arithmetic.

Both only pay while the loop body is short relative to LDS latency, and the body grows with CDIM:

| CDIM | Delta |
| --- | --- |
| 3 | −14.5% |
| 16 | +2.8% |

So the path is gated. `GSPLAT_FWD_F4_MAXCDIM` is 8 on wave32. Larger CDIM keeps the stock three-array layout. On wave64 the batch loses at every CDIM, so it is off there.

This is the third time on this page the answer is a range rather than a verdict. The workgroup-size choice was made against the register budget, the register move spent occupancy to buy latency, and here a load pattern wins at small CDIM and loses at large. None of the three has a setting that is right everywhere.

### Build knobs in one place

The tuning constants now sit together at the top of `Utils.cuh`. Override any of them at build time with `GSPLAT_EXTRA_DEFINES="GSPLAT_FWD_F4_MAXCDIM=0"`.

| Knob | Default | Bounded by |
| --- | --- | --- |
| `GSPLAT_WARP_SIZE` | injected by `setup.py` (32 or 64) | the part |
| `GSPLAT_FWD_F4_MAXCDIM` | 8 on wave32, 0 otherwise | LDS latency versus loop body |
| `GSPLAT_BS32_MULTITILE_MAXCDIM` | 16 | VGPRs ≈ 69 + 5·CDIM; hard ceiling 24 |
| `GSPLAT_BS32_MULTITILE_THREADS` | 128 | wave packing per CU |

`GSPLAT_WARP_SIZE` no longer has a fallback. It used to guess from `__AMDGCN_WAVEFRONT_SIZE__` and default to 64. It is now an `#error` if `setup.py` does not inject it. A wrong wave width is silent numerical corruption on a wave32 part.

These paths are ROCm-only. `setup.py` injects `GSPLAT_WARP_SIZE` on the HIP build, and the header stops compilation if it is missing. The CUDA path does not define the flag. It still uses `cg::labeled_partition`, and it keeps the original `intersect_tile` loop.

### Validation

199 tests pass, plus a production-scale check of twelve output tensors against a known-good reference. `intersect_tile` is bit-identical by construction. The projection and spherical-harmonic changes alter the order of floating-point accumulation, fewer atomics and a different grouping, so they are checked against tolerance rather than exactly. The color-accumulator reassociation is in that same category.

## Results

```{figure} ./images/bwd-kernel-time.png
:align: center
:alt: Bar chart of backward rasterizer time on MI45x. Baseline 6.244 ms, new multi-tile kernel 2.287 ms, after the optimizations 1.541 ms.
Backward rasterizer time per launch on MI45x. 6.244 ms to 1.541 ms is 4.05×.
```

The optimized kernel was profiled with `TILE_SIZE=8` at the same stump step window:

```bash
PROFILE=1 PROFILE_STEPS=3 PROFILE_START=21100 TILE_SIZE=8 \
CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
  --disable_viewer --data_factor 4 --render_traj_path ellipse \
  --data_dir data/360_v2/stump/ --result_dir results/benchmark/stump/
```

| Metric | New multi-tile kernel | After the optimizations |
| --- | --- | --- |
| Duration, mean | 2.287 ms | 1.541 ms |
| `SQ_INSTS_LDS` | 70.0M | 29.2M |
| LDS / VALU ratio | 14.0% | 5.7% |
| `SQ_WAIT_ANY` | 83.2% | 61.8% |
| `GL2C_ATOMIC` | 19.1M | 8.8M |
| Atomic latency | 592.5 cyc | 390 cyc |
| L2 hit rate | 82.4% | 90.8% |
| `SQ_INSTS_VALU` | 499.7M | 516M |
| Dynamic LDS / block | 3,072 B | 0 B |
| Resident waves / CU | 28.72 | 33.46 |
| Effective HBM read bandwidth | 152.3 GB/s | 207.7 GB/s |

The first column is the multi-tile kernel before the inner-loop shared-memory removal, which is why it still reports 3,072 bytes of dynamic LDS. Three counters are worth pulling out.

* Atomics halved (19.1M to 8.8M) and their average latency fell from 592.5 to 390 cycles. That is the lane-scatter epilogue.
* L2 misses halved (20.4M to 10.1M), lifting the hit rate from 82.4% to 90.8%.
* Effective HBM bandwidth stays around 150 to 210 GB/s, a small fraction of what the part can do. This kernel was stall-bound.

### End to end

On the Mip-NeRF 360 stump scene at `data_factor` 4, the 30,000-step run averages 88.7 steps/s, computed as 30,000 divided by `ellipse_time` from `train_step29999_rank0.json`. At step 29999 the held-out result is PSNR 26.824, SSIM 0.7791, LPIPS 0.145, with 5,676,550 Gaussians. The ~16 steps/s figure from the start of the post is the rate right after the port, not this same full-run average.

### MI300 kernel versus the MI45x kernel

The comparison is the specialized 8×8 backward kernels, not the generic tile-16 kernel from the first profile.

| | MI300X / MI325X (gfx942) | MI45x (gfx1250) |
| --- | --- | --- |
| Wave width | 64 lanes | 32 lanes |
| Kernel | `rasterize_bs64_to_pixels_3dgs_bwd_kernel` | `rasterize_bs32_multitile_to_pixels_3dgs_bwd_kernel` |
| Block shape | 64 threads = 1 wave = 1 tile | 128 threads = 4 waves = 4 tiles |
| Pixels per lane | 1 | 2 |
| Cross-wave sync | `block.sync()` | none (`warp.sync()` only) |
| Gaussian attributes | LDS-staged batch | registers + `warp.shfl` |
| Dynamic LDS | `max_batch_size × CDIM × 4` bytes | 0 bytes |
| Warp reduction | `rocprim::warp_reduce<T,64>` (DPP) | DPP ladder + `v_readlane` |
| Atomic epilogue | serialized from lane 0 | lane-scatter, one instruction |
| Absolute-gradient path | runtime `nullptr` check | compile-time `ABSGRAD` template |

## Lessons

1. **Occupancy is a means.** A drop of 2 waves per SIMD still bought 13.5% on kernel time. The win was in stall cycles. Chase the stall counters.
2. **Workgroup size is a packing decision.** The same kernel and the same registers, in smaller workgroups (8 waves down to 4), recovered 4 stranded waves per CU and 12.5% more resident waves.
3. **Shared-memory aliasing has a silent cost.** Four unrelated arrays in one `extern __shared__` block defeated alias analysis, and the compiler reloaded loop-invariant data every iteration.
4. **Check build flags before touching code.** CUDA had `--use_fast_math` for years. The HIP port did not carry it over. One flag: VGPR 83 to 74, occupancy 10 to 12, VALU −13.7%.
5. **A correct reduction can still be pure overhead.** Index math made every `gid` partition a singleton. It was correct, so nobody caught it.
6. **Lane imbalance hides in plain sight.** `intersect_tile` looked cheap, but 83% of lane-slots sat idle. Redistributing the rectangles across the wave cut kernel time 88%, bit-identical.
7. **Know why an optimization failed.** The rejected ideas spent registers to hide latency on a kernel that was already register-limited. That only showed up when the two versions were measured side by side.
