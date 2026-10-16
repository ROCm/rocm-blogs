/***************************************************************************
 Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

// Stage 4: buffer loads.
//
// Global reads go through a buffer resource descriptor rather than a plain pointer, so the
// hardware returns zero past the end of the allocation instead of faulting. This allows
// tiles that do not divide the matrix: a 96-row tile against M = 8192 would otherwise read
// past the end of A.
//
// The 256x256 tile of stage 3 is unchanged and neither benchmark size is out of bounds at
// that tile, so no clamping is exercised here; stage 5 uses it for a 128x96 tile. The
// driver verifies this stage but does not time it.
//
// The address arithmetic is identical to stage 3 — address.global(...) once per chunk —
// and only the consuming instruction differs. A flat read carries a 64-bit address per
// lane; a buffer read carries a 32-bit offset against a descriptor shared by the wave.
//
// The tile shape is a template parameter here. The copy still gives every thread the same
// whole number of 128-bit chunks, so the tile's element count must divide evenly among
// them.
#include "gemm_common.hpp"
#include "stages.hpp"

namespace
{
/// A GEMM kernel whose block tile is a template parameter.
///
/// \tparam BlockM, BlockN  output tile computed by one block
/// \tparam BlockK          K values held in LDS at a time
/// \tparam WavesM, WavesN  wave grid within the block
template<typename Policy, int BlockM, int BlockN, int BlockK, int WavesM, int WavesN>
__global__ __launch_bounds__(wave_grid_threads(WavesM, WavesN)) void gemm_srd(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    _Float16* __restrict__ C,
    int M,
    int N,
    int K)
{
    constexpr int threads  = wave_grid_threads(WavesM, WavesN);
    constexpr int tiles_m  = BlockM / WavesM / wmma_tile;
    constexpr int tiles_n  = BlockN / WavesN / wmma_tile;
    constexpr int stride   = BlockK + lds_pad;
    constexpr int chunks_a = (BlockM * BlockK) / (threads * elems_per_b128);
    constexpr int chunks_b = (BlockN * BlockK) / (threads * elems_per_b128);

    __shared__ _Float16 tile_a[2][BlockM * stride];
    __shared__ _Float16 tile_b[2][BlockN * stride];

    const int thread = threadIdx.x;
    const int lane   = thread % wave_size;
    const int wave   = thread / wave_size;

    const int block_row = blockIdx.y * BlockM;
    const int block_col = blockIdx.x * BlockN;

    const int wave_row = (wave / WavesN) * (BlockM / WavesM);
    const int wave_col = (wave % WavesN) * (BlockN / WavesN);

    const __amdgpu_buffer_rsrc_t buffer_a
        = make_buffer(A, static_cast<unsigned>(M) * K * sizeof(_Float16));
    const __amdgpu_buffer_rsrc_t buffer_b
        = make_buffer(B, static_cast<unsigned>(K) * N * sizeof(_Float16));

    const CopyAddress<BlockK, threads, stride> address(thread);

    /// Reads one chunk through the descriptor. Addresses beyond the allocation return
    /// zero, so BlockM and BlockN need not divide the matrix.
    auto load_chunk = [&](__amdgpu_buffer_rsrc_t resource, int origin, int k_tile, int chunk)
    {
        const int      offset = address.global(chunk, origin, k_tile * BlockK, K);
        const b128_raw raw
            = __builtin_amdgcn_raw_buffer_load_b128(resource,
                                                    offset * static_cast<int>(sizeof(_Float16)),
                                                    0,
                                                    0);
        half8 value;
        __builtin_memcpy(&value, &raw, sizeof(value));
        return value;
    };

    auto store_chunk = [&](_Float16* tile, const half8& value, int chunk)
    { *reinterpret_cast<half8*>(&tile[address.lds(chunk)]) = value; };

    // Declared outside the loop: inside, an unrolled loop can force these to scratch.
    typename Policy::Accumulator acc[tiles_m][tiles_n] = {};
    typename Policy::FragmentA   a[tiles_m];
    typename Policy::FragmentB   b[tiles_n];

    // The chunks of the next K tile, held between the load and the commit.
    half8 staged_a[chunks_a];
    half8 staged_b[chunks_b];

    const int num_k_tiles = (K + BlockK - 1) / BlockK;
    int       buffer      = 0;

    auto multiply_current = [&]()
    {
        const _Float16* pa = &tile_a[buffer][0];
        const _Float16* pb = &tile_b[buffer][0];

        for(int m = 0; m < tiles_m; ++m)
        {
            Policy::load_a(a[m], pa, wave_row + m * wmma_tile, 0, lane, stride);
        }
        for(int n = 0; n < tiles_n; ++n)
        {
            Policy::load_b(b[n], pb, wave_col + n * wmma_tile, 0, lane, stride);
        }
        for(int m = 0; m < tiles_m; ++m)
        {
            for(int n = 0; n < tiles_n; ++n)
            {
                Policy::mma(acc[m][n], a[m], b[n]);
            }
        }
    };

    for(int chunk = 0; chunk < chunks_a; ++chunk)
    {
        store_chunk(&tile_a[buffer][0], load_chunk(buffer_a, block_row, 0, chunk), chunk);
    }
    for(int chunk = 0; chunk < chunks_b; ++chunk)
    {
        store_chunk(&tile_b[buffer][0], load_chunk(buffer_b, block_col, 0, chunk), chunk);
    }
    __syncthreads();

    for(int k_tile = 0; k_tile < num_k_tiles - 1; ++k_tile)
    {
        for(int chunk = 0; chunk < chunks_a; ++chunk)
        {
            staged_a[chunk] = load_chunk(buffer_a, block_row, k_tile + 1, chunk);
        }
        for(int chunk = 0; chunk < chunks_b; ++chunk)
        {
            staged_b[chunk] = load_chunk(buffer_b, block_col, k_tile + 1, chunk);
        }

        multiply_current();

        for(int chunk = 0; chunk < chunks_a; ++chunk)
        {
            store_chunk(&tile_a[buffer ^ 1][0], staged_a[chunk], chunk);
        }
        for(int chunk = 0; chunk < chunks_b; ++chunk)
        {
            store_chunk(&tile_b[buffer ^ 1][0], staged_b[chunk], chunk);
        }
        __syncthreads();
        buffer ^= 1;
    }
    multiply_current();

    for(int m = 0; m < tiles_m; ++m)
    {
        for(int n = 0; n < tiles_n; ++n)
        {
            Policy::store_c(acc[m][n],
                            C,
                            block_row + wave_row + m * wmma_tile,
                            block_col + wave_col + n * wmma_tile,
                            M,
                            N,
                            lane);
        }
    }
}

template<int BlockM, int BlockN, int BlockK, int WavesM, int WavesN>
void launch_config(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    const dim3 grid((N + BlockN - 1) / BlockN, (M + BlockM - 1) / BlockM);
    const dim3 block(wave_grid_threads(WavesM, WavesN));
    if(arch == Arch::gfx11)
    {
        gemm_srd<WmmaRdna3Policy, BlockM, BlockN, BlockK, WavesM, WavesN>
            <<<grid, block>>>(A, B, C, M, N, K);
    }
    else
    {
        gemm_srd<WmmaRdna4Policy, BlockM, BlockN, BlockK, WavesM, WavesN>
            <<<grid, block>>>(A, B, C, M, N, K);
    }
}
} // namespace

/// The stage 3 tile, now reading through a descriptor.
void launch_stage04(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    launch_config<256, 256, wmma_tile, 4, 4>(arch, A, B, C, M, N, K);
}
