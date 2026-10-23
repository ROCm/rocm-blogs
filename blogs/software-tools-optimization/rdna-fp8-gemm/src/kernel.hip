// FP8 GEMM kernel for RDNA 4, with the copy interleaved into the fold.
//
// The kernel carries over the final shape from the FP16 progression:
//
//   - buffer descriptors for out-of-bounds-safe global loads
//   - a deep K slice in LDS, amortizing barriers over more arithmetic
//   - blocks scheduled in a cache-reusing order
//   - global reads issued from inside the fold, one per second matrix instruction
//   - one LDS buffer or two, as a template parameter
//
// Each step index in the fold must be a compile-time constant so that the fragment and
// accumulator it selects — which live in vector registers — are not forced to scratch.
// A recursive template supplies the constant: each instantiation handles one step and
// calls the next, so the compiler unrolls the chain and the indices fold away.
//
// Addresses are hoisted before the loop. Both a fragment read and a prefetch chunk have
// a thread-dependent part and a position-dependent part. The first is folded into a
// pointer; the second is a template argument and becomes an immediate.
#include "configs.hpp"
#include "gemm_common.hpp"

namespace
{
/// Everything one step of the fold needs, gathered so the recursion carries a single
/// reference rather than a long parameter list.
template<typename Policy, int BlockM, int BlockN, int BlockK, int WavesM, int WavesN>
struct FoldContext
{
    static constexpr int threads = wave_grid_threads(WavesM, WavesN);

    static constexpr int tiles_m  = BlockM / WavesM / wmma_tile;
    static constexpr int tiles_n  = BlockN / WavesN / wmma_tile;
    static constexpr int k_slices = BlockK / wmma_tile;
    static constexpr int stride   = BlockK + lds_pad;
    /// Elements one pass of the whole thread block moves.
    static constexpr int pass = threads * elems_per_b128;
    /// Passes that cover the tile completely, and the total including a partial last one.
    static constexpr int whole_a  = (BlockM * BlockK) / pass;
    static constexpr int whole_b  = (BlockN * BlockK) / pass;
    static constexpr int chunks_a = (BlockM * BlockK + pass - 1) / pass;
    static constexpr int chunks_b = (BlockN * BlockK + pass - 1) / pass;
    /// Matrix instructions per K tile.
    static constexpr int steps = k_slices * tiles_m * tiles_n;

    using Reader = OperandReader<Policy, BlockK, threads, stride>;

    Reader reader_a;
    Reader reader_b;

    typename Policy::FragmentA* a;
    typename Policy::FragmentB* b;
    typename Policy::Accumulator (*acc)[tiles_n];

    fp8x16* staged_a;
    fp8x16* staged_b;
};

/// Writes staged chunks of one operand into its LDS tile.
///
/// `Whole` is the number of passes fully covered by the tile. Those passes carry no
/// guard; only the trailing partial pass is predicated.
template<typename Addr, int Rows, int Chunks, int Whole>
__device__ __forceinline__ void
    commit_chunks(fp8* tile, const Addr& address, const fp8x16* staged)
{
    for(int chunk = 0; chunk < Whole; ++chunk)
    {
        *reinterpret_cast<fp8x16*>(&tile[address.lds(chunk)]) = staged[chunk];
    }
    if constexpr(Chunks > Whole)
    {
        if(address.row + Whole * Addr::rows_per_chunk < Rows)
        {
            *reinterpret_cast<fp8x16*>(&tile[address.lds(Whole)]) = staged[Whole];
        }
    }
}

/// The (m, n, slice) position step `Step` occupies in the traversal.
/// n runs fastest within each (slice, m) pair.
template<int Step, int TilesM, int TilesN>
struct FoldPosition
{
    static constexpr int outer = Step / TilesN;

    static constexpr int m = outer % TilesM;
    static constexpr int n = Step % TilesN;

    /// K slice this step multiplies.
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
        // One chunk of the next tile per instruction, alternating operands. `Prefetch` is
        // a template parameter so the loads compile out entirely on the last tile.
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

        // Each fragment is read one step ahead of the instruction that consumes it.
        // B is read only in the first row of each K slice and reused down the remaining rows.
        if constexpr(Step + 1 < Context::steps)
        {
            if constexpr(next::a_index != here::a_index)
            {
                ctx.reader_a.template read_fragment<next::m, next::slice>(ctx.a[next::a_index]);
            }
            if constexpr(next::m == 0)
            {
                ctx.reader_b.template read_fragment<next::n, next::slice>(ctx.b[next::b_index]);
            }
        }

        __builtin_amdgcn_s_setprio(1);
        Policy::mma(ctx.acc[here::m][here::n], ctx.a[here::a_index], ctx.b[here::b_index]);
        __builtin_amdgcn_s_setprio(0);

        if constexpr(Step + 1 < Context::steps)
        {
            FoldStep<Policy, Step + 1, BlockM, BlockN, BlockK, WavesM, WavesN, Prefetch>::run(ctx);
        }
    }

    /// Reads the operands the first step multiplies. Every later fragment is read one step
    /// ahead of its use; only this pair has nowhere earlier to come from.
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
__global__ __launch_bounds__(wave_grid_threads(WavesM,
                                               WavesN)) void gemm_fp8(const fp8* __restrict__ A,
                                                                      const fp8* __restrict__ B,
                                                                      float* __restrict__ C,
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
    constexpr int whole_a  = Context::whole_a;
    constexpr int whole_b  = Context::whole_b;
    constexpr int buffers  = SingleBuffer ? 1 : 2;

    constexpr int other = SingleBuffer ? 0 : 1;

    __shared__ fp8 tile_a[buffers][BlockM * stride];
    __shared__ fp8 tile_b[buffers][BlockN * stride];

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
        = make_buffer(A, static_cast<unsigned>(M) * K * sizeof(fp8));
    const __amdgpu_buffer_rsrc_t buffer_b
        = make_buffer(B, static_cast<unsigned>(K) * N * sizeof(fp8));

    using CopyAddr = CopyAddress<BlockK, threads, stride>;
    const CopyAddr address(thread);

    // Registers declared once, outside the loop, so the compiler keeps them live rather
    // than re-establishing the same values on every iteration.
    typename Policy::Accumulator acc[tiles_m][tiles_n] = {};
    typename Policy::FragmentA   a[k_slices * tiles_m];
    typename Policy::FragmentB   b[k_slices * tiles_n];

    // Staging registers hold the next K tile between the global load and the LDS commit.
    fp8x16 staged_a[chunks_a];
    fp8x16 staged_b[chunks_b];

    const int num_k_tiles = (K + BlockK - 1) / BlockK;
    int       buffer      = 0;

    // Load the first K tile into staging and commit it to LDS.
    constexpr int element_bytes = static_cast<int>(sizeof(fp8));
    for(int chunk = 0; chunk < chunks_a; ++chunk)
    {
        const int offset = address.global(chunk, block_row, 0, K);
        load_chunk(staged_a[chunk], buffer_a, offset * element_bytes);
    }
    for(int chunk = 0; chunk < chunks_b; ++chunk)
    {
        const int offset = address.global(chunk, block_col, 0, K);
        load_chunk(staged_b[chunk], buffer_b, offset * element_bytes);
    }
    commit_chunks<CopyAddr, BlockM, chunks_a, whole_a>(&tile_a[buffer][0], address, staged_a);
    commit_chunks<CopyAddr, BlockN, chunks_b, whole_b>(&tile_b[buffer][0], address, staged_b);
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

        // With one buffer the commit overwrites the tile just read, so every wave must
        // finish reading before the write begins.
        if constexpr(SingleBuffer)
        {
            __syncthreads();
        }
        commit_chunks<CopyAddr, BlockM, chunks_a, whole_a>(&tile_a[buffer ^ other][0],
                                                           address,
                                                           staged_a);
        commit_chunks<CopyAddr, BlockN, chunks_b, whole_b>(&tile_b[buffer ^ other][0],
                                                           address,
                                                           staged_b);
        __syncthreads();
        buffer ^= other;
        reader_a.advance();
        reader_b.advance();
    }

    // The last tile has nothing to prefetch; its fold is instantiated with the loads
    // compiled out and the staging registers go unread.
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

} // namespace

/// One instantiation per configuration in the sweep.
template<typename Cfg>
void launch(const fp8* A, const fp8* B, float* C, int M, int N, int K)
{
    const int blocks
        = ((M + Cfg::block_m - 1) / Cfg::block_m) * ((N + Cfg::block_n - 1) / Cfg::block_n);
    const dim3 grid(blocks);
    const dim3 block(Cfg::threads);
    gemm_fp8<WmmaFp8Policy,
             Cfg::block_m,
             Cfg::block_n,
             Cfg::block_k,
             Cfg::waves_m,
             Cfg::waves_n,
             Cfg::single_buffer><<<grid, block>>>(A, B, C, M, N, K);
}

#define INSTANTIATE(M, N, K, WM, WN, SINGLE)                                \
    template void launch<Config<M, N, K, WM, WN, SINGLE>>(const fp8*,       \
                                                          const fp8*,       \
                                                          float*,           \
                                                          int,              \
                                                          int,              \
                                                          int);
FP8_GEMM_CONFIGS(INSTANTIATE)
#undef INSTANTIATE
