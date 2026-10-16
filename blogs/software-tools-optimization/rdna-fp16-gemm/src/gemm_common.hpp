/***************************************************************************
 Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

// Shared definitions for the FP16 GEMM stages.
//
// A is row-major [M][K], B is column-major [N][K]. Both are contiguous along K, so LDS
// tiles are K-contiguous and a WMMA fragment is a contiguous run of K values per lane.
#pragma once

#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iostream>

constexpr int error_exit_code = -1;

#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = condition;                                                 \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cerr << "An error encountered: \"" << hipGetErrorString(error) << "\" at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                          \
            std::exit(error_exit_code);                                                     \
        }                                                                                   \
    }

#define HIPBLASLT_CHECK(condition)                                                             \
    {                                                                                          \
        const hipblasStatus_t status = condition;                                              \
        if(status != HIPBLAS_STATUS_SUCCESS)                                                   \
        {                                                                                      \
            std::cerr << "hipBLASLt error encountered: " << static_cast<int>(status) << " at " \
                      << __FILE__ << ':' << __LINE__ << std::endl;                             \
            std::exit(error_exit_code);                                                        \
        }                                                                                      \
    }

/// Eight FP16 values: what a single 128-bit access moves.
using half8 = _Float16 [[clang::ext_vector_type(8)]];
/// The same 128 bits as returned by the buffer load intrinsics, before reinterpreting.
using b128_raw = unsigned int [[clang::ext_vector_type(4)]];

/// Size of the WMMA instruction's output tile, and of its K step.
constexpr int wmma_tile = 16;
/// Elements moved by one 128-bit access.
constexpr int elems_per_b128 = 8;
/// Lanes in a wave32 wavefront.
constexpr int wave_size = 32;
/// Rows of a 16x16 output tile one lane holds. Two groups of 16 lanes cover the tile on
/// both architectures; they differ in which rows each group holds.
constexpr int acc_rows_per_lane = wmma_tile / 2;
/// Elements appended to each LDS tile row so consecutive rows start in different bank
/// groups (stage 3).
constexpr int lds_pad = 8;

/// Threads in a block holding a WavesM x WavesN grid of waves.
constexpr int wave_grid_threads(int waves_m, int waves_n)
{
    return waves_m * waves_n * wave_size;
}

enum class Arch
{
    gfx11,
    gfx12,
    unknown
};

// Compute policies. The fragment types, the LDS-to-fragment read, the matrix multiply and
// the C write-out differ between architectures and are selected by template argument:
//
//   RDNA3 (gfx11)  Every lane holds all 16 K values, so a fragment is 16 FP16 wide. With
//                  opsel = false the results land in acc[2 * i].
//   RDNA4 (gfx12)  Lanes 0-15 hold K values 0-7 and lanes 16-31 hold 8-15, so a fragment
//                  is 8 FP16 wide and the results are acc[i].
//
// Fragment reads use __builtin_memcpy: a vector register cannot be indexed dynamically,
// so an indexed loop over the elements forces the fragment to scratch.

/// RDNA3: 16-wide fragments, results in the even accumulator halves.
struct WmmaRdna3Policy
{
    using FragmentA   = _Float16 [[clang::ext_vector_type(16)]];
    using FragmentB   = _Float16 [[clang::ext_vector_type(16)]];
    using Accumulator = _Float16 [[clang::ext_vector_type(16)]];

    /// Elements of K one lane reads for a fragment. Also the swizzle granularity in
    /// stage 3: a permutation may not split this run.
    static constexpr int fragment_elems = 16;

    /// Every lane reads the same K values, so there is no per-lane offset.
    __device__ static int k_offset(int /*lane*/)
    {
        return 0;
    }

    /// The lane-dependent part of a fragment's LDS address. Added to a tile pointer once,
    /// it leaves every fragment in that tile at a fixed offset. See load_at.
    __device__ static int lane_base(int lane, int stride)
    {
        return (lane % wmma_tile) * stride;
    }

    __device__ static void
        load_a(FragmentA& fragment, const _Float16* tile, int row, int k, int lane, int stride)
    {
        __builtin_memcpy(&fragment, &tile[(row + lane % wmma_tile) * stride + k], sizeof(fragment));
    }

    __device__ static void
        load_b(FragmentB& fragment, const _Float16* tile, int col, int k, int lane, int stride)
    {
        __builtin_memcpy(&fragment, &tile[(col + lane % wmma_tile) * stride + k], sizeof(fragment));
    }

    /// Reads a fragment from an address the caller has already formed. Both operands read
    /// the same way, so one function serves either. Used by the interleaved fold.
    template<typename Fragment>
    __device__ static void load_at(Fragment& fragment, const _Float16* at)
    {
        __builtin_memcpy(&fragment, at, sizeof(fragment));
    }

    __device__ static void mma(Accumulator& acc, const FragmentA& a, const FragmentB& b)
    {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) \
    || defined(__gfx1151__) || defined(__gfx1152__)
        acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, acc, /*opsel=*/false);
#endif
    }

    /// Lane L holds column L % 16 and rows (2 * i) + L / 16 of the 16x16 output tile. C is
    /// column-major with leading dimension M, matching hipBLASLt's output layout.
    __device__ static void store_c(
        const Accumulator& acc, _Float16* C, int tile_row, int tile_col, int M, int N, int lane)
    {
        const int col = tile_col + lane % wmma_tile;
        if(col >= N)
        {
            return;
        }
        const int first_row = tile_row + lane / wmma_tile;
        for(int i = 0; i < acc_rows_per_lane; ++i)
        {
            // Rows alternate between the lane groups and results sit in even elements.
            const int row = first_row + 2 * i;
            if(row < M)
            {
                C[static_cast<size_t>(col) * M + row] = acc[2 * i];
            }
        }
    }
};

/// RDNA4: 8-wide fragments split across two lane groups.
struct WmmaRdna4Policy
{
    using FragmentA   = _Float16 [[clang::ext_vector_type(8)]];
    using FragmentB   = _Float16 [[clang::ext_vector_type(8)]];
    using Accumulator = _Float16 [[clang::ext_vector_type(8)]];

    static constexpr int fragment_elems = 8;

    /// Lanes 0-15 read K values 0-7; lanes 16-31 read 8-15.
    __device__ static int k_offset(int lane)
    {
        return elems_per_b128 * (lane / wmma_tile);
    }

    /// The lane-dependent part of a fragment's LDS address. Added to a tile pointer once,
    /// it leaves every fragment in that tile at a fixed offset. See load_at.
    __device__ static int lane_base(int lane, int stride)
    {
        return (lane % wmma_tile) * stride + k_offset(lane);
    }

    __device__ static void
        load_a(FragmentA& fragment, const _Float16* tile, int row, int k, int lane, int stride)
    {
        const int offset = (row + lane % wmma_tile) * stride + k + k_offset(lane);
        __builtin_memcpy(&fragment, &tile[offset], sizeof(fragment));
    }

    __device__ static void
        load_b(FragmentB& fragment, const _Float16* tile, int col, int k, int lane, int stride)
    {
        const int offset = (col + lane % wmma_tile) * stride + k + k_offset(lane);
        __builtin_memcpy(&fragment, &tile[offset], sizeof(fragment));
    }

    /// Reads a fragment from an address the caller has already formed. Both operands read
    /// the same way, so one function serves either. Used by the interleaved fold.
    template<typename Fragment>
    __device__ static void load_at(Fragment& fragment, const _Float16* at)
    {
        __builtin_memcpy(&fragment, at, sizeof(fragment));
    }

    __device__ static void mma(Accumulator& acc, const FragmentA& a, const FragmentB& b)
    {
#if defined(__gfx1200__) || defined(__gfx1201__)
        acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a, b, acc);
#endif
    }

    /// Lane L holds column L % 16 and rows 8 * (L / 16) + i of the 16x16 output tile. C is
    /// column-major with leading dimension M, matching hipBLASLt's output layout.
    __device__ static void store_c(
        const Accumulator& acc, _Float16* C, int tile_row, int tile_col, int M, int N, int lane)
    {
        const int col = tile_col + lane % wmma_tile;
        if(col >= N)
        {
            return;
        }
        // Each lane group owns a contiguous half of the rows, and the results are packed,
        // so the rows and the accumulator indices both run consecutively.
        const int first_row = tile_row + acc_rows_per_lane * (lane / wmma_tile);
        for(int i = 0; i < acc_rows_per_lane; ++i)
        {
            const int row = first_row + i;
            if(row < M)
            {
                C[static_cast<size_t>(col) * M + row] = acc[i];
            }
        }
    }
};

/// Where one thread's chunks of a tile live, in LDS and in global memory.
///
/// Chunks are `Threads * elems_per_b128` elements apart, which is a whole number of tile
/// rows, so chunk `c` sits `c * rows_per_chunk` rows below chunk 0 at the same K offset.
/// The divide and remainder converting a flat index to (row, k) therefore depend only on
/// the thread and are evaluated once here, leaving each chunk a fixed stride away.
template<int BlockK, int Threads, int Stride>
struct CopyAddress
{
    /// Rows of the tile between one chunk and the next.
    static constexpr int rows_per_chunk = (Threads * elems_per_b128) / BlockK;

    static_assert((Threads * elems_per_b128) % BlockK == 0,
                  "chunk spacing must be a whole number of tile rows");

    /// This thread's position in the tile, shared by all of its chunks.
    int row;
    int k;

    __device__ __forceinline__ explicit CopyAddress(int thread)
        : row(thread * elems_per_b128 / BlockK), k(thread * elems_per_b128 % BlockK)
    {}

    /// Element offset of chunk `chunk` within an LDS tile.
    __device__ __forceinline__ int lds(int chunk) const
    {
        return (row + chunk * rows_per_chunk) * Stride + k;
    }

    /// Element offset of chunk `chunk` in a K-contiguous matrix with leading dimension
    /// `K`, for a tile starting at row `origin` and K position `k_origin`.
    __device__ __forceinline__ int global(int chunk, int origin, int k_origin, int K) const
    {
        return (origin + row + chunk * rows_per_chunk) * K + k_origin + k;
    }
};

/// The reads one operand needs during an interleaved fold, as base-plus-constant.
///
/// Each read has a thread-dependent part and a fragment- or chunk-dependent part. This
/// holds the first and takes the second as a template argument, leaving nothing to
/// compute at the point of use.
template<typename Policy, int BlockK, int Threads, int Stride>
struct OperandReader
{
    using Address = CopyAddress<BlockK, Threads, Stride>;

    static constexpr int element_bytes = static_cast<int>(sizeof(_Float16));

    /// This wave's first element within an LDS tile, and the LDS tile it currently
    /// reads from. Every fragment sits a compile-time distance from `tile`.
    int             lane_offset;
    const _Float16* tile;

    /// The matrix in global memory, this thread's byte offset into the tile being
    /// prefetched, and the distance from one chunk of it to the next.
    __amdgpu_buffer_rsrc_t resource;
    int                    offset;
    int                    chunk_bytes;

    __device__ __forceinline__ OperandReader(int                    wave_offset,
                                             int                    lane,
                                             __amdgpu_buffer_rsrc_t resource,
                                             const Address&         address,
                                             int                    origin,
                                             int                    K)
        : lane_offset(wave_offset * Stride + Policy::lane_base(lane, Stride))
        , tile(nullptr)
        , resource(resource)
        , offset(address.global(0, origin, BlockK, K) * element_bytes)
        , chunk_bytes(Address::rows_per_chunk * K * element_bytes)
    {}

    /// Points the reader at one of the LDS buffers.
    __device__ __forceinline__ void read_from(const _Float16* lds)
    {
        tile = lds + lane_offset;
    }

    /// Moves the prefetch on to the next K tile.
    __device__ __forceinline__ void advance()
    {
        offset += BlockK * element_bytes;
    }

    /// Element offset of the fragment at tile row `Row` of K slice `Slice`.
    template<int Row, int Slice>
    static constexpr int fragment_offset()
    {
        return Row * wmma_tile * Stride + Slice * wmma_tile;
    }

    template<int Row, int Slice, typename Fragment>
    __device__ __forceinline__ void read_fragment(Fragment& fragment) const
    {
        Policy::load_at(fragment, tile + fragment_offset<Row, Slice>());
    }

    template<int Chunk>
    __device__ __forceinline__ void prefetch(half8* into) const
    {
        const b128_raw raw
            = __builtin_amdgcn_raw_buffer_load_b128(resource, offset + Chunk * chunk_bytes, 0, 0);
        __builtin_memcpy(&into[Chunk], &raw, sizeof(half8));
    }
};

/// Config word for a raw buffer on gfx11 and gfx12: an untyped buffer, bounds-checked
/// against the size given below. The field layout is in the ISA guides, under the buffer
/// resource descriptor.
constexpr unsigned buffer_config_word = 0x31004000;

/// Wraps a pointer in a buffer resource descriptor. Loads against it return zero for
/// out-of-bounds addresses instead of faulting, which allows block tiles that do not
/// divide the matrix evenly (stage 5).
///
/// \param bytes  the size the hardware clamps against, so it must be the whole allocation.
///               The size field is 32 bits, bounding one operand at 4 GB.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer(const void* ptr, unsigned bytes)
{
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(ptr),
                                             /*stride=*/0,
                                             bytes,
                                             buffer_config_word);
}

/// Maps a 1D block index onto an output tile, grouping concurrent blocks so they share
/// rows of A and columns of B in L2 (stage 6).
///
/// Blocks are handed out in groups of `GroupSize` tile rows, the index running down the
/// rows before moving across the columns, so a group covers a tall region of C rather than
/// a wide one. A tall region reads `GroupSize` rows of A and few columns of B.
///
/// This is composable_kernel's BlockToCTileMap_M00_N0_M01Adapt, in
/// include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp, reduced to the one case
/// needed here. CK names the group height M01 and defaults it to 8, which is GroupSize
/// below; `group_rows` is its `M01_adapt`, handling a tile count that is not a multiple of
/// the group height by making the final short group divide by its own height.
///
/// `tiles_m` and `tiles_n` are the tile counts in each dimension.
template<int BlockM, int BlockN, int GroupSize = 8>
__device__ __forceinline__ void
    map_block_swizzled(int block, int tiles_m, int tiles_n, int* tile_row, int* tile_col)
{
    if(tiles_m == 1 || tiles_n == 1)
    {
        *tile_row = (block / tiles_n) * BlockM;
        *tile_col = (block % tiles_n) * BlockN;
        return;
    }

    const int index_n = block % tiles_n;
    const int index_m = block / tiles_n;

    // The last group is short when the rows do not divide evenly, and then covers only
    // the rows that remain.
    const int group_rows
        = (index_m < tiles_m - tiles_m % GroupSize) ? GroupSize : tiles_m % GroupSize;

    const int group = index_m / GroupSize;
    const int local = index_n + (index_m % GroupSize) * tiles_n;

    *tile_row = (local % group_rows + group * GroupSize) * BlockM;
    *tile_col = (local / group_rows) * BlockN;
}
