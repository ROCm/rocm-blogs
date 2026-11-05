/***************************************************************************
 Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

// Stage 1: one wave per block, 32x32 output tile, one WMMA step of K per iteration.
// Two barriers per K step, no overlap between copying and computing.
#include "gemm_common.hpp"
#include "stages.hpp"

namespace
{
/// K values processed per iteration: one WMMA step.
constexpr int block_k = wmma_tile;
/// Output tile computed by one block.
constexpr int block_m = 32;
constexpr int block_n = 32;
/// 16x16 WMMA tiles per wave in each dimension.
constexpr int tiles_m = block_m / wmma_tile;
constexpr int tiles_n = block_n / wmma_tile;

template<typename Policy>
__global__ __launch_bounds__(wave_size) void gemm_naive(const _Float16* __restrict__ A,
                                                        const _Float16* __restrict__ B,
                                                        _Float16* __restrict__ C,
                                                        int M,
                                                        int N,
                                                        int K)
{
    // K-contiguous: tile_a[m][k] and tile_b[n][k]. B is column-major, so neither operand
    // is transposed on the way in.
    __shared__ _Float16 tile_a[block_m][block_k];
    __shared__ _Float16 tile_b[block_n][block_k];

    const int lane      = threadIdx.x;
    const int block_row = blockIdx.y * block_m;
    const int block_col = blockIdx.x * block_n;

    // Declared outside the loop: inside, an unrolled loop can force these to scratch.
    typename Policy::Accumulator acc[tiles_m][tiles_n] = {};
    typename Policy::FragmentA   a[tiles_m];
    typename Policy::FragmentB   b[tiles_n];

    const int num_k_tiles = K / block_k;
    for(int k_tile = 0; k_tile < num_k_tiles; ++k_tile)
    {
        // One element per lane per step.
        for(int i = lane; i < block_m * block_k; i += wave_size)
        {
            const int row  = i / block_k;
            const int k    = i % block_k;
            tile_a[row][k] = A[(block_row + row) * K + k_tile * block_k + k];
        }
        for(int i = lane; i < block_n * block_k; i += wave_size)
        {
            const int col  = i / block_k;
            const int k    = i % block_k;
            tile_b[col][k] = B[(block_col + col) * K + k_tile * block_k + k];
        }
        __syncthreads();

        for(int m = 0; m < tiles_m; ++m)
        {
            Policy::load_a(a[m], &tile_a[0][0], m * wmma_tile, 0, lane, block_k);
        }
        for(int n = 0; n < tiles_n; ++n)
        {
            Policy::load_b(b[n], &tile_b[0][0], n * wmma_tile, 0, lane, block_k);
        }
        for(int m = 0; m < tiles_m; ++m)
        {
            for(int n = 0; n < tiles_n; ++n)
            {
                Policy::mma(acc[m][n], a[m], b[n]);
            }
        }

        // The next iteration overwrites the tiles.
        __syncthreads();
    }

    for(int m = 0; m < tiles_m; ++m)
    {
        for(int n = 0; n < tiles_n; ++n)
        {
            Policy::store_c(acc[m][n],
                            C,
                            block_row + m * wmma_tile,
                            block_col + n * wmma_tile,
                            M,
                            N,
                            lane);
        }
    }
}
} // namespace

void launch_stage01(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    const dim3 grid((N + block_n - 1) / block_n, (M + block_m - 1) / block_m);
    const dim3 block(wave_size);
    if(arch == Arch::gfx11)
    {
        gemm_naive<WmmaRdna3Policy><<<grid, block>>>(A, B, C, M, N, K);
    }
    else
    {
        gemm_naive<WmmaRdna4Policy><<<grid, block>>>(A, B, C, M, N, K);
    }
}
