#mma_b16 = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @blog_example(%arg1: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %2 = ttg.local_load %arg1 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_b16, kWidth = 8}>>
    %ptr2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma_b16, kWidth = 8}>>
    tt.store %ptr2, %2 : tensor<64x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mma_b16, kWidth = 8}>>
    tt.return
  }
}
