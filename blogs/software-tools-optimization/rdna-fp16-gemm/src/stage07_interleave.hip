/***************************************************************************
 Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

// Stage 7: interleaving the copy with the multiply. Stage 8: one LDS buffer.
//
// Stage 2 issues all of the next tile's global reads before the fold, which puts every
// read in one burst and leaves the rest of the fold with no memory traffic behind it. With
// four K slices per tile there are 48 matrix instructions to distribute 14 reads between,
// so each read is issued one instruction ahead of its consumer, alternating operands.
// Fragments are read the same way, one step before the instruction that needs them.
//
// The step index selects which fragment is read and which accumulator is updated, so it
// must be a compile-time constant: a fragment lives in vector registers, and an index from
// a runtime counter forces the array to scratch. A recursive template supplies it, each
// instantiation performing one step and calling the next.
//
// Addresses must also be free here. Both reads — a fragment from LDS, a chunk from global
// — have a thread-dependent part and a fragment- or chunk-dependent part. Stages 2 to 6
// rebuild both per read, which their burst of loads hides behind its own memory stall.
// Here a read sits between two matrix instructions, so the thread-dependent part is
// computed once before the loop and folded into a pointer, and the rest is a template
// argument. See OperandReader in gemm_common.hpp.
#include "gemm_common.hpp"
#include "stages.hpp"

namespace
{
/// What one step of the fold needs, so the recursion carries a single reference.
template<typename Policy, int BlockM, int BlockN, int BlockK, int WavesM, int WavesN>
struct FoldContext
{
    static constexpr int threads = wave_grid_threads(WavesM, WavesN);

    static constexpr int tiles_m  = BlockM / WavesM / wmma_tile;
    static constexpr int tiles_n  = BlockN / WavesN / wmma_tile;
    static constexpr int k_slices = BlockK / wmma_tile;
    static constexpr int stride   = BlockK + lds_pad;
    static constexpr int chunks_a = (BlockM * BlockK) / (threads * elems_per_b128);
    static constexpr int chunks_b = (BlockN * BlockK) / (threads * elems_per_b128);
    /// Matrix instructions per K tile.
    static constexpr int steps = k_slices * tiles_m * tiles_n;

    using Reader = OperandReader<Policy, BlockK, threads, stride>;

    Reader reader_a;
    Reader reader_b;

    typename Policy::FragmentA* a;
    typename Policy::FragmentB* b;
    typename Policy::Accumulator (*acc)[tiles_n];

    half8* staged_a;
    half8* staged_b;
};

/// One matrix instruction, together with the reads that belong with it.
/// Where step `Step` of the traversal lands: row by row through the wave's tiles, one K
/// slice after another.
template<int Step, int TilesM, int TilesN>
struct FoldPosition
{
    static constexpr int outer = Step / TilesN;

    static constexpr int m = outer % TilesM;
    static constexpr int n = Step % TilesN;

    /// Which K slice this step multiplies.
    static constexpr int slice = outer / TilesM;

    static constexpr int a_index = slice * TilesM + m;
    static constexpr int b_index = slice * TilesN + n;
};

template<typename Policy,
         int  Step,
         int  BlockM,
         int  BlockN,
         int  BlockK,
         int  WavesM,
         int  WavesN,
         bool Prefetch>
struct FoldStep
{
    using Context = FoldContext<Policy, BlockM, BlockN, BlockK, WavesM, WavesN>;

    static constexpr int tiles_m = Context::tiles_m;
    static constexpr int tiles_n = Context::tiles_n;

    using here = FoldPosition<Step, tiles_m, tiles_n>;
    using next = FoldPosition<Step + 1 < Context::steps ? Step + 1 : Step, tiles_m, tiles_n>;

    __device__ static void run(const Context& ctx)
    {
        // One chunk of the next tile per instruction, alternating operands. Whether to
        // prefetch is a template parameter rather than a runtime flag: a test here would
        // put a branch in front of all 48 instructions.
        if constexpr(Prefetch)
        {
            constexpr int piece = Step / 2;
            if constexpr(Step % 2 == 0)
            {
                if constexpr(piece < Context::chunks_a)
                {
                    ctx.reader_a.template prefetch<piece>(ctx.staged_a);
                }
            }
            else if constexpr(piece < Context::chunks_b)
            {
                ctx.reader_b.template prefetch<piece>(ctx.staged_b);
            }
        }

        // Read what the next instruction needs, not this one: the fragment this step
        // multiplies was read one step earlier, so its LDS latency overlaps a matrix
        // instruction.
        //
        // The traversal runs n fastest, so B's index cycles 0,1,2,0,1,2 while A advances
        // once per row. Reading B on every index change would re-read the same fragments.
        // Instead the whole K slice of B is read when the traversal enters it.
        if constexpr(Step + 1 < Context::steps)
        {
            if constexpr(next::a_index != here::a_index)
            {
                ctx.reader_a.template read_fragment<next::m, next::slice>(ctx.a[next::a_index]);
            }
            // B advances only in the first row of a slice; the fragments are then reused
            // down the remaining rows.
            if constexpr(next::m == 0)
            {
                ctx.reader_b.template read_fragment<next::n, next::slice>(ctx.b[next::b_index]);
            }
        }

        // Raise this wave's issue priority around the matrix instruction so it wins
        // arbitration for the SIMD over the interleaved reads competing with it. Explicit
        // schedule constraints on the compiler were not needed to hold the interleaved order.
        __builtin_amdgcn_s_setprio(1);
        Policy::mma(ctx.acc[here::m][here::n], ctx.a[here::a_index], ctx.b[here::b_index]);
        __builtin_amdgcn_s_setprio(0);

        if constexpr(Step + 1 < Context::steps)
        {
            FoldStep<Policy, Step + 1, BlockM, BlockN, BlockK, WavesM, WavesN, Prefetch>::run(ctx);
        }
    }

    /// Reads the operands the first step multiplies. Every later fragment is read one
    /// step ahead of its use.
    __device__ static void prologue(const Context& ctx)
    {
        ctx.reader_a.template read_fragment<here::m, here::slice>(ctx.a[here::a_index]);
        ctx.reader_b.template read_fragment<here::n, here::slice>(ctx.b[here::b_index]);
    }
};

/// \tparam BlockM, BlockN  output tile computed by one block
/// \tparam BlockK          K values held in LDS at a time
/// \tparam SingleBuffer    one LDS buffer instead of two
template<typename Policy,
         int  BlockM,
         int  BlockN,
         int  BlockK,
         int  WavesM,
         int  WavesN,
         bool SingleBuffer>
__global__ __launch_bounds__(wave_grid_threads(WavesM, WavesN)) void gemm_interleaved(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    _Float16* __restrict__ C,
    int M,
    int N,
    int K)
{
    using Context = FoldContext<Policy, BlockM, BlockN, BlockK, WavesM, WavesN>;

    constexpr int threads = Context::threads;

    constexpr int tiles_m  = Context::tiles_m;
    constexpr int tiles_n  = Context::tiles_n;
    constexpr int k_slices = Context::k_slices;
    constexpr int stride   = Context::stride;
    constexpr int chunks_a = Context::chunks_a;
    constexpr int chunks_b = Context::chunks_b;
    constexpr int buffers  = SingleBuffer ? 1 : 2;
    constexpr int other    = SingleBuffer ? 0 : 1;

    __shared__ _Float16 tile_a[buffers][BlockM * stride];
    __shared__ _Float16 tile_b[buffers][BlockN * stride];

    const int thread = threadIdx.x;
    const int lane   = thread % wave_size;
    const int wave   = thread / wave_size;

    int block_row;
    int block_col;
    map_block_swizzled<BlockM, BlockN>(blockIdx.x,
                                       (M + BlockM - 1) / BlockM,
                                       (N + BlockN - 1) / BlockN,
                                       &block_row,
                                       &block_col);

    const int wave_row = (wave / WavesN) * (BlockM / WavesM);
    const int wave_col = (wave % WavesN) * (BlockN / WavesN);

    const __amdgpu_buffer_rsrc_t buffer_a
        = make_buffer(A, static_cast<unsigned>(M) * K * sizeof(_Float16));
    const __amdgpu_buffer_rsrc_t buffer_b
        = make_buffer(B, static_cast<unsigned>(K) * N * sizeof(_Float16));

    const CopyAddress<BlockK, threads, stride> address(thread);

    auto store_chunk = [&](_Float16* tile, const half8& value, int chunk)
    { *reinterpret_cast<half8*>(&tile[address.lds(chunk)]) = value; };

    /// Reads one chunk of K tile `k_tile`, for the prologue. Later tiles are staged
    /// through registers from inside the fold.
    auto load_direct = [&](__amdgpu_buffer_rsrc_t resource, int origin, int k_tile, int chunk)
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

    // Declared outside the loop: inside, an unrolled loop can force these to scratch.
    typename Policy::Accumulator acc[tiles_m][tiles_n] = {};
    typename Policy::FragmentA   a[k_slices * tiles_m];
    typename Policy::FragmentB   b[k_slices * tiles_n];

    // The chunks of the next K tile, held between the load and the commit.
    half8 staged_a[chunks_a];
    half8 staged_b[chunks_b];

    const int num_k_tiles = (K + BlockK - 1) / BlockK;
    int       buffer      = 0;

    for(int chunk = 0; chunk < chunks_a; ++chunk)
    {
        store_chunk(&tile_a[buffer][0], load_direct(buffer_a, block_row, 0, chunk), chunk);
    }
    for(int chunk = 0; chunk < chunks_b; ++chunk)
    {
        store_chunk(&tile_b[buffer][0], load_direct(buffer_b, block_col, 0, chunk), chunk);
    }
    __syncthreads();

    typename Context::Reader reader_a(wave_row, lane, buffer_a, address, block_row, K);
    typename Context::Reader reader_b(wave_col, lane, buffer_b, address, block_col, K);

    auto context_for = [&]()
    {
        reader_a.read_from(&tile_a[buffer][0]);
        reader_b.read_from(&tile_b[buffer][0]);
        return Context{reader_a, reader_b, a, b, acc, staged_a, staged_b};
    };

    for(int k_tile = 0; k_tile < num_k_tiles - 1; ++k_tile)
    {
        FoldStep<Policy, 0, BlockM, BlockN, BlockK, WavesM, WavesN, true>::prologue(context_for());
        FoldStep<Policy, 0, BlockM, BlockN, BlockK, WavesM, WavesN, true>::run(context_for());

        // With one buffer the commit overwrites the tile just read.
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
        reader_a.advance();
        reader_b.advance();
    }

    // The last tile has nothing to prefetch, so its fold is instantiated with the loads
    // compiled out.
    FoldStep<Policy, 0, BlockM, BlockN, BlockK, WavesM, WavesN, false>::prologue(context_for());
    FoldStep<Policy, 0, BlockM, BlockN, BlockK, WavesM, WavesN, false>::run(context_for());

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

template<int BlockM, int BlockN, int BlockK, int WavesM, int WavesN, bool SingleBuffer = false>
void launch_config(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    const int  blocks = ((M + BlockM - 1) / BlockM) * ((N + BlockN - 1) / BlockN);
    const dim3 grid(blocks);
    const dim3 block(wave_grid_threads(WavesM, WavesN));
    if(arch == Arch::gfx11)
    {
        gemm_interleaved<WmmaRdna3Policy, BlockM, BlockN, BlockK, WavesM, WavesN, SingleBuffer>
            <<<grid, block>>>(A, B, C, M, N, K);
    }
    else
    {
        gemm_interleaved<WmmaRdna4Policy, BlockM, BlockN, BlockK, WavesM, WavesN, SingleBuffer>
            <<<grid, block>>>(A, B, C, M, N, K);
    }
}
} // namespace

/// Stage 7: the interleaved fold with two LDS buffers.
void launch_stage07(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    launch_config<128, 96, 4 * wmma_tile, 2, 2>(arch, A, B, C, M, N, K);
}

/// Stage 8: one LDS buffer instead of two. Half the LDS and twice the resident blocks,
/// without the copy/compute overlap the second buffer provided.
void launch_stage08(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    launch_config<128, 96, 4 * wmma_tile, 2, 2, true>(arch, A, B, C, M, N, K);
}

/// Step 8 with a larger tile tuned for gfx1201: a 192x224 output tile on a 4x2 wave grid, same
/// K depth and single buffering. Fits the 64 KB budget at 58.5 KB.
void launch_stage08_tuned(
    Arch arch, const _Float16* A, const _Float16* B, _Float16* C, int M, int N, int K)
{
    launch_config<192, 224, 4 * wmma_tile, 4, 2, true>(arch, A, B, C, M, N, K);
}
