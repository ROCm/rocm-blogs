/***************************************************************************
 Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

// Stage 2: 256x256 block tile over a 4x4 wave grid, 128-bit copies, double-buffered LDS,
// and the copy staged through registers.
//
// The loop is prefetch, multiply, commit, one barrier. Later stages keep this structure
// and change how memory is addressed and how the work is ordered.
#include "gemm_common.hpp"
#include "stages.hpp"

namespace
{
constexpr int block_k = wmma_tile;
constexpr int block_m = 256;
constexpr int block_n = 256;
constexpr int waves_m = 4;
constexpr int waves_n = 4;
constexpr int threads = wave_grid_threads(waves_m, waves_n);
/// 16x16 WMMA tiles each wave accumulates.
constexpr int tiles_m = block_m / waves_m / wmma_tile;
constexpr int tiles_n = block_n / waves_n / wmma_tile;

/// 128-bit chunks each thread copies per operand.
constexpr int chunks_a = (block_m * block_k) / (threads * elems_per_b128);
constexpr int chunks_b = (block_n * block_k) / (threads * elems_per_b128);

/// Reads one 128-bit chunk of a K slice into registers.
///
/// \param base    A, or B; both are contiguous along K with leading dimension K.
/// \param origin  the block's first row (for A) or column (for B).
using Address = CopyAddress<block_k, threads, block_k>;

__device__ __forceinline__ half8 load_chunk(
    const _Float16* base, int origin, int k_tile, int K, const Address& address, int chunk)
{
    half8 value;
    __builtin_memcpy(&value,
                     &base[address.global(chunk, origin, k_tile * block_k, K)],
                     sizeof(value));
    return value;
}

/// Writes one 128-bit chunk from registers into an LDS tile.
__device__ __forceinline__ void
    store_chunk(_Float16* tile, const half8& value, const Address& address, int chunk)
{
    *reinterpret_cast<half8*>(&tile[address.lds(chunk)]) = value;
}

template<typename Policy>
__global__ __launch_bounds__(threads) void gemm_pipeline(const _Float16* __restrict__ A,
                                                         const _Float16* __restrict__ B,
                                                         _Float16* __restrict__ C,
                                                         int M,
                                                         int N,
                                                         int K)
{
    __shared__ _Float16 tile_a[2][block_m * block_k];
    __shared__ _Float16 tile_b[2][block_n * block_k];

    const int thread    = threadIdx.x;
    const int lane      = thread % wave_size;
    const int wave      = thread / wave_size;
    const int block_row = blockIdx.y * block_m;
    const int block_col = blockIdx.x * block_n;
    // Origin of this wave's quadrant within the block tile.
    const int wave_row = (wave / waves_n) * (block_m / waves_m);
    const int wave_col = (wave % waves_n) * (block_n / waves_n);

    const Address address(thread);

    // Declared outside the loop: inside, an unrolled loop can force these to scratch.
    typename Policy::Accumulator acc[tiles_m][tiles_n] = {};
    typename Policy::FragmentA   a[tiles_m];
    typename Policy::FragmentB   b[tiles_n];

    // The chunks of the next K tile, held between the load and the commit.
    half8 staged_a[chunks_a];
    half8 staged_b[chunks_b];

    const int num_k_tiles = K / block_k;
    // Which of the two tiles is being read. Flipping a counter rather than swapping
    // pointers keeps the tile base a compile-time constant.
    int buffer = 0;

    auto multiply_current = [&]()
    {
        const _Float16* pa = &tile_a[buffer][0];
        const _Float16* pb = &tile_b[buffer][0];

        // Each fragment is read once and reused across a row or column of the grid.
        for(int m = 0; m < tiles_m; ++m)
        {
            Policy::load_a(a[m], pa, wave_row + m * wmma_tile, 0, lane, block_k);
        }
        for(int n = 0; n < tiles_n; ++n)
        {
            Policy::load_b(b[n], pb, wave_col + n * wmma_tile, 0, lane, block_k);
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
        store_chunk(&tile_a[buffer][0],
                    load_chunk(A, block_row, 0, K, address, chunk),
                    address,
                    chunk);
    }
    for(int chunk = 0; chunk < chunks_b; ++chunk)
    {
        store_chunk(&tile_b[buffer][0],
                    load_chunk(B, block_col, 0, K, address, chunk),
                    address,
                    chunk);
    }
    __syncthreads();

    for(int k_tile = 0; k_tile < num_k_tiles - 1; ++k_tile)
    {
        // Issue the next slice's reads, then compute while they are still outstanding.
        for(int chunk = 0; chunk < chunks_a; ++chunk)
        {
            staged_a[chunk] = load_chunk(A, block_row, k_tile + 1, K, address, chunk);
        }
        for(int chunk = 0; chunk < chunks_b; ++chunk)
        {
            staged_b[chunk] = load_chunk(B, block_col, k_tile + 1, K, address, chunk);
        }

        multiply_current();

        // Only here do the values have to have arrived.
        for(int chunk = 0; chunk < chunks_a; ++chunk)
        {
            store_chunk(&tile_a[buffer ^ 1][0], staged_a[chunk], address, chunk);
        }
        for(int chunk = 0; chunk < chunks_b; ++chunk)
        {
            store_chunk(&tile_b[buffer ^ 1][0], staged_b[chunk], address, chunk);
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
} // namespace

void launch_stage02(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    const dim3 grid((N + block_n - 1) / block_n, (M + block_m - 1) / block_m);
    const dim3 block(threads);
    if(arch == Arch::gfx11)
    {
        gemm_pipeline<WmmaRdna3Policy><<<grid, block>>>(A, B, C, M, N, K);
    }
    else
    {
        gemm_pipeline<WmmaRdna4Policy><<<grid, block>>>(A, B, C, M, N, K);
    }
}
