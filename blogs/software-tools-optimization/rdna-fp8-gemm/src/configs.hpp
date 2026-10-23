// Tile shape configurations for the FP8 GEMM sweep.
//
// The kernel takes its shape as template parameters, so a configuration is a type rather
// than a runtime argument: every one is compiled separately and dispatched from a table.
//
// A configuration must satisfy four constraints, all checked by static_assert in Config:
//
//   1. The wave grid divides the tile into whole WMMA tiles.
//   2. A chunk fits inside one row of the K tile.
//   3. The chunk spacing is a whole number of tile rows.
//   4. There are enough matrix instructions in the fold to carry every prefetch chunk.
//
// One constraint is easy to trip: block_k must divide K. The buffer descriptor bounds
// reads against the whole allocation rather than one row, so a K tile that runs past the
// end of a row reads the next row rather than returning zeros.
#pragma once

#include "gemm_common.hpp"

/// One tile shape, as a type. `SingleBuffer` selects one LDS buffer or two.
template<int BlockM, int BlockN, int BlockK, int WavesM, int WavesN, bool SingleBuffer>
struct Config
{
    static constexpr int  block_m       = BlockM;
    static constexpr int  block_n       = BlockN;
    static constexpr int  block_k       = BlockK;
    static constexpr int  waves_m       = WavesM;
    static constexpr int  waves_n       = WavesN;
    static constexpr bool single_buffer = SingleBuffer;

    static constexpr int threads = wave_grid_threads(WavesM, WavesN);
    static constexpr int tiles_m = BlockM / WavesM / wmma_tile;
    static constexpr int tiles_n = BlockN / WavesN / wmma_tile;

    /// Elements one pass of the whole thread block moves, and the passes each tile needs.
    /// The count rounds up: a tile that does not divide evenly ends on a partial pass.
    static constexpr int pass     = threads * elems_per_b128;
    static constexpr int chunks_a = (BlockM * BlockK + pass - 1) / pass;
    static constexpr int chunks_b = (BlockN * BlockK + pass - 1) / pass;
    static constexpr int steps    = (BlockK / wmma_tile) * tiles_m * tiles_n;

    /// LDS bytes one block occupies, with the row padding included.
    static constexpr int lds_bytes
        = (BlockM + BlockN) * (BlockK + lds_pad) * (SingleBuffer ? 1 : 2);

    static_assert(BlockM % (WavesM * wmma_tile) == 0 && BlockN % (WavesN * wmma_tile) == 0,
                  "the wave grid must divide the tile into whole WMMA tiles");
    static_assert(BlockK % elems_per_b128 == 0, "a chunk must fit inside one row of the K tile");
    static_assert(pass % BlockK == 0, "chunk spacing must be a whole number of tile rows");
    static_assert(2 * chunks_a <= steps && 2 * chunks_b <= steps,
                  "the fold needs at least two steps per chunk to interleave the prefetch");
};

/// Runs one configuration. Defined in kernel.hip, instantiated once per Config below.
template<typename Cfg>
void launch(const fp8* A, const fp8* B, float* C, int M, int N, int K);

// ---------------------------------------------------------------------------
// The three configurations compared in the post.
//
// Config 1 (128x96, K=64, 2x2): the FP16 final kernel with element type changed.
// Config 2 (256x256, K=64, 4x4): larger tile, fits at FP8 (40 KB), not at FP16 (80 KB).
// Config 3 (256x160, K=128, 4x2): deeper K, fills more of the LDS budget.
// Config 4 (192x192, K=128, 2x4): the tuned tile, deeper K on a 2x4 wave grid.
// ---------------------------------------------------------------------------
#define FP8_GEMM_CONFIGS(X)              \
    /*  M    N    K   WM  WN   single */ \
    X(128, 96, 64, 2, 2, true)          \
    X(256, 256, 64, 4, 4, true)         \
    X(256, 160, 128, 4, 2, true)        \
    X(192, 192, 128, 2, 4, true)
