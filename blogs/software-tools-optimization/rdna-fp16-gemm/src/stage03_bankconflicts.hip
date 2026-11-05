/***************************************************************************
 Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

// Stage 3: LDS bank conflicts.
//
// Every tile row so far has been block_k elements long, so all rows start in the same
// bank. The sixteen lanes reading a fragment each take a different row, hit the same bank,
// and serialise.
//
// Two kernels are built here. The first appends lds_pad elements to each row, shifting
// consecutive rows into different banks and growing the tile by lds_pad / block_k. The
// second keeps the rows compact and XOR-permutes addresses within them instead. The
// permutation unit is one fragment, so a wider fragment leaves fewer positions to permute
// between; see the table in the blog post for the counts per architecture.
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
constexpr int tiles_m = block_m / waves_m / wmma_tile;
constexpr int tiles_n = block_n / waves_n / wmma_tile;

constexpr int chunks_a = (block_m * block_k) / (threads * elems_per_b128);
constexpr int chunks_b = (block_n * block_k) / (threads * elems_per_b128);

/// Padded row stride.
constexpr int padded_stride = block_k + lds_pad;

/// Both layouts need the (row, k) pair rather than a single offset, so the address is
/// kept unpadded here and each layout applies its own mapping.
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

// ===========================================================================
// Padding: give each row a longer stride than it needs.
// ===========================================================================
__device__ __forceinline__ void
    store_chunk_padded(_Float16* tile, const half8& value, const Address& address, int chunk)
{
    const int row = address.row + chunk * Address::rows_per_chunk;
    *reinterpret_cast<half8*>(&tile[row * padded_stride + address.k]) = value;
}

template<typename Policy>
__global__ __launch_bounds__(threads) void gemm_padded(const _Float16* __restrict__ A,
                                                       const _Float16* __restrict__ B,
                                                       _Float16* __restrict__ C,
                                                       int M,
                                                       int N,
                                                       int K)
{
    __shared__ _Float16 tile_a[2][block_m * padded_stride];
    __shared__ _Float16 tile_b[2][block_n * padded_stride];

    const int thread    = threadIdx.x;
    const int lane      = thread % wave_size;
    const int wave      = thread / wave_size;
    const int block_row = blockIdx.y * block_m;
    const int block_col = blockIdx.x * block_n;
    const int wave_row  = (wave / waves_n) * (block_m / waves_m);
    const int wave_col  = (wave % waves_n) * (block_n / waves_n);

    const Address address(thread);

    // Declared outside the loop: inside, an unrolled loop can force these to scratch.
    typename Policy::Accumulator acc[tiles_m][tiles_n] = {};
    typename Policy::FragmentA   a[tiles_m];
    typename Policy::FragmentB   b[tiles_n];

    // The chunks of the next K tile, held between the load and the commit.
    half8 staged_a[chunks_a];
    half8 staged_b[chunks_b];

    const int num_k_tiles = K / block_k;
    int       buffer      = 0;

    auto multiply_current = [&]()
    {
        const _Float16* pa = &tile_a[buffer][0];
        const _Float16* pb = &tile_b[buffer][0];

        for(int m = 0; m < tiles_m; ++m)
        {
            Policy::load_a(a[m], pa, wave_row + m * wmma_tile, 0, lane, padded_stride);
        }
        for(int n = 0; n < tiles_n; ++n)
        {
            Policy::load_b(b[n], pb, wave_col + n * wmma_tile, 0, lane, padded_stride);
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
        store_chunk_padded(&tile_a[buffer][0],
                           load_chunk(A, block_row, 0, K, address, chunk),
                           address,
                           chunk);
    }
    for(int chunk = 0; chunk < chunks_b; ++chunk)
    {
        store_chunk_padded(&tile_b[buffer][0],
                           load_chunk(B, block_col, 0, K, address, chunk),
                           address,
                           chunk);
    }
    __syncthreads();

    for(int k_tile = 0; k_tile < num_k_tiles - 1; ++k_tile)
    {
        for(int chunk = 0; chunk < chunks_a; ++chunk)
        {
            staged_a[chunk] = load_chunk(A, block_row, k_tile + 1, K, address, chunk);
        }
        for(int chunk = 0; chunk < chunks_b; ++chunk)
        {
            staged_b[chunk] = load_chunk(B, block_col, k_tile + 1, K, address, chunk);
        }

        multiply_current();

        for(int chunk = 0; chunk < chunks_a; ++chunk)
        {
            store_chunk_padded(&tile_a[buffer ^ 1][0], staged_a[chunk], address, chunk);
        }
        for(int chunk = 0; chunk < chunks_b; ++chunk)
        {
            store_chunk_padded(&tile_b[buffer ^ 1][0], staged_b[chunk], address, chunk);
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

// ===========================================================================
// XOR permutation: store block j of row r at block j XOR (r % blocks), so consecutive
// rows begin in different banks without the tile growing.
//
// A permutation may not split the run of elements one lane reads for a fragment, so its
// unit is Policy::fragment_elems: 16 FP16 on RDNA3, 8 on RDNA4. Permutable blocks per row
// is block_k / fragment_elems:
//
//     block_k   RDNA3 blocks   RDNA4 blocks
//        16          1              2
//        32          2              4
//        64          4              8
//
// At block_k = 16 the RDNA3 permutation is the identity. The rest of the progression uses
// padding, which staggers rows at any fragment width.
// ===========================================================================
template<typename Policy>
__global__ __launch_bounds__(threads) void gemm_swizzled(const _Float16* __restrict__ A,
                                                         const _Float16* __restrict__ B,
                                                         _Float16* __restrict__ C,
                                                         int M,
                                                         int N,
                                                         int K)
{
    constexpr int unit   = Policy::fragment_elems;
    constexpr int blocks = block_k / unit;

    __shared__ _Float16 tile_a[2][block_m * block_k];
    __shared__ _Float16 tile_b[2][block_n * block_k];

    const int thread    = threadIdx.x;
    const int lane      = thread % wave_size;
    const int wave      = thread / wave_size;
    const int block_row = blockIdx.y * block_m;
    const int block_col = blockIdx.x * block_n;
    const int wave_row  = (wave / waves_n) * (block_m / waves_m);
    const int wave_col  = (wave % waves_n) * (block_n / waves_n);

    /// Offset of element (row, k) once the row's blocks have been permuted.
    auto swizzle = [](int row, int k)
    { return row * block_k + ((k / unit) ^ (row % blocks)) * unit + (k % unit); };

    // Declared outside the loop: inside, an unrolled loop can force these to scratch.
    typename Policy::Accumulator acc[tiles_m][tiles_n] = {};
    typename Policy::FragmentA   a[tiles_m];
    typename Policy::FragmentB   b[tiles_n];

    // The chunks of the next K tile, held between the load and the commit.
    half8 staged_a[chunks_a];
    half8 staged_b[chunks_b];

    const int num_k_tiles = K / block_k;
    int       buffer      = 0;

    const Address address(thread);

    auto store_chunk = [&](_Float16* tile, const half8& value, int chunk)
    {
        *reinterpret_cast<half8*>(
            &tile[swizzle(address.row + chunk * Address::rows_per_chunk, address.k)])
            = value;
    };

    auto multiply_current = [&]()
    {
        const _Float16* pa = &tile_a[buffer][0];
        const _Float16* pb = &tile_b[buffer][0];

        // The fragment read undoes the same permutation, so it forms the address itself
        // rather than going through Policy::load_a.
        for(int m = 0; m < tiles_m; ++m)
        {
            const int row = wave_row + m * wmma_tile + lane % wmma_tile;
            __builtin_memcpy(&a[m], &pa[swizzle(row, Policy::k_offset(lane))], sizeof(a[m]));
        }
        for(int n = 0; n < tiles_n; ++n)
        {
            const int col = wave_col + n * wmma_tile + lane % wmma_tile;
            __builtin_memcpy(&b[n], &pb[swizzle(col, Policy::k_offset(lane))], sizeof(b[n]));
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
        store_chunk(&tile_a[buffer][0], load_chunk(A, block_row, 0, K, address, chunk), chunk);
    }
    for(int chunk = 0; chunk < chunks_b; ++chunk)
    {
        store_chunk(&tile_b[buffer][0], load_chunk(B, block_col, 0, K, address, chunk), chunk);
    }
    __syncthreads();

    for(int k_tile = 0; k_tile < num_k_tiles - 1; ++k_tile)
    {
        for(int chunk = 0; chunk < chunks_a; ++chunk)
        {
            staged_a[chunk] = load_chunk(A, block_row, k_tile + 1, K, address, chunk);
        }
        for(int chunk = 0; chunk < chunks_b; ++chunk)
        {
            staged_b[chunk] = load_chunk(B, block_col, k_tile + 1, K, address, chunk);
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
} // namespace

void launch_stage03(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    const dim3 grid((N + block_n - 1) / block_n, (M + block_m - 1) / block_m);
    const dim3 block(threads);
    if(arch == Arch::gfx11)
    {
        gemm_padded<WmmaRdna3Policy><<<grid, block>>>(A, B, C, M, N, K);
    }
    else
    {
        gemm_padded<WmmaRdna4Policy><<<grid, block>>>(A, B, C, M, N, K);
    }
}

void launch_stage03_swizzled(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    const dim3 grid((N + block_n - 1) / block_n, (M + block_m - 1) / block_m);
    const dim3 block(threads);
    if(arch == Arch::gfx11)
    {
        gemm_swizzled<WmmaRdna3Policy><<<grid, block>>>(A, B, C, M, N, K);
    }
    else
    {
        gemm_swizzled<WmmaRdna4Policy><<<grid, block>>>(A, B, C, M, N, K);
    }
}
