/***************************************************************************
 Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

// Stage 5: four WMMA steps of K per LDS tile rather than one.
//
// This cuts the barriers to a quarter and gives each wave four rounds of matrix
// instructions per copy. The fold gains an outer loop over K slices; only the current
// slice's fragments are live, so the depth costs no registers.
//
// LDS grows in proportion to the K depth and decides the tile shape. A 256x256 tile needs
// 80 KB at two K steps; 128x128 at four K steps needs 72 KB, past the 64 KB a workgroup
// may use. 128x96 fits in 63 KB, and is only expressible because stage 4 clamps
// out-of-bounds reads: 96 divides neither 4096 nor 8192.
//
// Stage 6 is in this file too: it hands tiles to blocks in a cache-reusing order.
#include "gemm_common.hpp"
#include "stages.hpp"

namespace
{
/// Four WMMA steps of K per tile, rather than one.
constexpr int block_k = 4 * wmma_tile;
constexpr int block_m = 128;
constexpr int block_n = 96;
constexpr int waves_m = 2;
constexpr int waves_n = 2;
constexpr int threads = wave_grid_threads(waves_m, waves_n);
constexpr int tiles_m = block_m / waves_m / wmma_tile;
constexpr int tiles_n = block_n / waves_n / wmma_tile;
/// WMMA steps held in one LDS tile.
constexpr int k_slices = block_k / wmma_tile;
constexpr int stride   = block_k + lds_pad;

constexpr int chunks_a = (block_m * block_k) / (threads * elems_per_b128);
constexpr int chunks_b = (block_n * block_k) / (threads * elems_per_b128);

/// \tparam SingleBuffer  use one LDS buffer instead of two.
template<typename Policy, bool Swizzle, bool SingleBuffer>
__global__ __launch_bounds__(threads) void gemm_deep_k(const _Float16* __restrict__ A,
                                                       const _Float16* __restrict__ B,
                                                       _Float16* __restrict__ C,
                                                       int M,
                                                       int N,
                                                       int K)
{
    constexpr int buffers = SingleBuffer ? 1 : 2;
    constexpr int other   = SingleBuffer ? 0 : 1;

    __shared__ _Float16 tile_a[buffers][block_m * stride];
    __shared__ _Float16 tile_b[buffers][block_n * stride];

    const int thread = threadIdx.x;
    const int lane   = thread % wave_size;
    const int wave   = thread / wave_size;
    // Tiles are assigned in groups of rows rather than row by row, so concurrent blocks
    // sit in a tall region of the output and share the rows of A and columns of B.
    int block_row;
    int block_col;
    if constexpr(Swizzle)
    {
        map_block_swizzled<block_m, block_n>(blockIdx.x,
                                             (M + block_m - 1) / block_m,
                                             (N + block_n - 1) / block_n,
                                             &block_row,
                                             &block_col);
    }
    else
    {
        block_row = blockIdx.y * block_m;
        block_col = blockIdx.x * block_n;
    }
    const int wave_row = (wave / waves_n) * (block_m / waves_m);
    const int wave_col = (wave % waves_n) * (block_n / waves_n);

    const __amdgpu_buffer_rsrc_t buffer_a
        = make_buffer(A, static_cast<unsigned>(M) * K * sizeof(_Float16));
    const __amdgpu_buffer_rsrc_t buffer_b
        = make_buffer(B, static_cast<unsigned>(K) * N * sizeof(_Float16));

    const CopyAddress<block_k, threads, stride> address(thread);

    auto load_chunk = [&](__amdgpu_buffer_rsrc_t resource, int origin, int k_tile, int chunk)
    {
        const int      offset = address.global(chunk, origin, k_tile * block_k, K);
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
    typename Policy::FragmentA   a[k_slices * tiles_m];
    typename Policy::FragmentB   b[k_slices * tiles_n];

    // The chunks of the next K tile, held between the load and the commit.
    half8 staged_a[chunks_a];
    half8 staged_b[chunks_b];

    const int num_k_tiles = (K + block_k - 1) / block_k;
    int       buffer      = 0;

    /// Multiplies the tile one K step at a time. Only the current step's fragments are
    /// live, so extra K depth costs no registers.
    auto multiply_current = [&]()
    {
        const _Float16* pa = &tile_a[buffer][0];
        const _Float16* pb = &tile_b[buffer][0];

        // Fragment addresses are rebuilt from a row and a K position. The reads are
        // issued as a burst and stall on LDS, so the arithmetic overlaps the stall.
        // Stage 7 spreads these reads out, where it does not.
        for(int slice = 0; slice < k_slices; ++slice)
        {
            for(int m = 0; m < tiles_m; ++m)
            {
                Policy::load_a(a[slice * tiles_m + m],
                               pa,
                               wave_row + m * wmma_tile,
                               slice * wmma_tile,
                               lane,
                               stride);
            }
            for(int n = 0; n < tiles_n; ++n)
            {
                Policy::load_b(b[slice * tiles_n + n],
                               pb,
                               wave_col + n * wmma_tile,
                               slice * wmma_tile,
                               lane,
                               stride);
            }
            for(int m = 0; m < tiles_m; ++m)
            {
                for(int n = 0; n < tiles_n; ++n)
                {
                    Policy::mma(acc[m][n], a[slice * tiles_m + m], b[slice * tiles_n + n]);
                }
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

        // With one buffer the commit below overwrites the tile just read.
        if constexpr(SingleBuffer)
        {
            __syncthreads();
        }

        for(int chunk = 0; chunk < chunks_a; ++chunk)
        {
            store_chunk(&tile_a[buffer ^ other][0], staged_a[chunk], chunk);
        }
        for(int chunk = 0; chunk < chunks_b; ++chunk)
        {
            store_chunk(&tile_b[buffer ^ other][0], staged_b[chunk], chunk);
        }
        __syncthreads();
        buffer ^= other;
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

template<bool Swizzle, bool SingleBuffer>
void launch_buffering(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    const int  tiles_row = (M + block_m - 1) / block_m;
    const int  tiles_col = (N + block_n - 1) / block_n;
    const dim3 grid(Swizzle ? dim3(tiles_row * tiles_col) : dim3(tiles_col, tiles_row));
    const dim3 block(threads);
    if(arch == Arch::gfx11)
    {
        gemm_deep_k<WmmaRdna3Policy, Swizzle, SingleBuffer><<<grid, block>>>(A, B, C, M, N, K);
    }
    else
    {
        gemm_deep_k<WmmaRdna4Policy, Swizzle, SingleBuffer><<<grid, block>>>(A, B, C, M, N, K);
    }
}
} // namespace

/// Stage 5: the deeper K slice with the earlier row-by-row block order.
void launch_stage05(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    launch_buffering<false, false>(arch, A, B, C, M, N, K);
}

/// Stage 6: the same kernel with the cache-reusing block order.
void launch_stage06(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    launch_buffering<true, false>(arch, A, B, C, M, N, K);
}
