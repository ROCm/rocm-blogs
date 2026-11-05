/***************************************************************************
 Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

// Host launchers, one per stage. Each performs a single kernel launch and does not
// synchronize; main.hip handles warm-up, timing, and verification.
//
// Stages 2 to 4 hold the tile at 256x256 with a 4x4 wave grid. Stage 5 shrinks it to
// 128x96, which a deeper K slice needs to fit in LDS.
#pragma once

#include "gemm_common.hpp"

using LaunchFunction = void (*)(Arch, const _Float16*, const _Float16*, _Float16*, int, int, int);

/// Stage 1: one wave, one WMMA tile at a time.
void launch_stage01(Arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K);
/// Stage 2: tiling, wide copies, double buffering, staged commit.
void launch_stage02(Arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K);
/// Stage 3: LDS bank conflicts, by padding and by permutation.
void launch_stage03(Arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K);
void launch_stage03_swizzled(
    Arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K);
/// Stage 4: buffer loads, clamping out-of-bounds reads to zero. Verified but not timed;
/// it keeps stage 3's tile.
void launch_stage04(Arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K);
/// Stage 5: four WMMA steps of K per tile, on a 128x96 tile.
void launch_stage05(Arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K);
/// Stage 6: tiles assigned to blocks in a cache-reusing order.
void launch_stage06(Arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K);
/// Stage 7: the next tile's reads issued from inside the fold.
void launch_stage07(Arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K);
/// Stage 8: one LDS buffer instead of two.
void launch_stage08(Arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K);
/// Stage 8 with a larger 192x224 4x2 tile, tuned for gfx1201.
void launch_stage08_tuned(
    Arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K);
