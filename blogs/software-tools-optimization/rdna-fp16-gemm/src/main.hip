/***************************************************************************
 Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

// Benchmark driver for the stage progression.
//
// A is row-major [M][K] and B is column-major [N][K], each element drawn as -1 or +1. hipBLASLt
// runs the same problem, called directly rather than through rocBLAS. Each stage is timed and
// checked against its output; with +/-1 inputs the two match exactly, so a mismatch is a real bug.
//
// The full progression runs at M = N = K = 8192; the final kernel is repeated at a
// non-square shape. Build with build.sh and run ./gemm_stages.
#include "gemm_common.hpp"
#include "stages.hpp"

#include <hipblaslt/hipblaslt.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace
{
/// With +/-1 inputs every true output is an integer, so it is either zero or a normal FP16 value.
/// A subnormal can only be the rounding residue of a cancelled-to-zero output -- on RDNA 3 the
/// reduction leaves ~1e-7 behind instead of a clean zero -- so flushing subnormals to zero
/// collapses those residues while leaving every real value untouched.
float flush_subnormal(_Float16 value)
{
    constexpr float smallest_normal = 6.103515625e-5f; // 2^-14
    const float     wide            = static_cast<float>(value);
    return std::fabs(wide) < smallest_normal ? 0.0f : wide;
}

Arch detect_arch()
{
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    const std::string name(properties.gcnArchName);
    if(name.rfind("gfx12", 0) == 0)
    {
        return Arch::gfx12;
    }
    if(name.rfind("gfx11", 0) == 0)
    {
        return Arch::gfx11;
    }
    return Arch::unknown;
}

/// Runs the progression at one problem shape.
///
/// \param full_progression  when false, only the final kernel is measured.
void run(int M, int N, int K, Arch arch, hipblasLtHandle_t handle, bool full_progression = true)
{
    const size_t elements_a = static_cast<size_t>(M) * K;
    const size_t elements_b = static_cast<size_t>(N) * K;
    const size_t elements_c = static_cast<size_t>(M) * N;

    // A and B are dense, with each element drawn as -1 or +1. This lets every stage be checked
    // against hipBLASLt exactly, with no tolerance: every product is then +/-1 and every partial
    // sum an integer. With random inputs those sums stay near +/-sqrt(K), well inside the range
    // FP16 represents exactly, so no rounding occurs whatever order each side reduces in, and a
    // correct kernel must match hipBLASLt bit for bit.
    std::vector<_Float16>       host_a(elements_a);
    std::vector<_Float16>       host_b(elements_b);
    std::mt19937                generator(42);
    std::bernoulli_distribution coin(0.5);
    for(auto& value : host_a)
    {
        value = static_cast<_Float16>(coin(generator) ? 1.0f : -1.0f);
    }
    for(auto& value : host_b)
    {
        value = static_cast<_Float16>(coin(generator) ? 1.0f : -1.0f);
    }

    _Float16* device_a = nullptr;
    _Float16* device_b = nullptr;
    _Float16* device_c = nullptr;
    HIP_CHECK(hipMalloc(&device_a, elements_a * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&device_b, elements_b * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&device_c, elements_c * sizeof(_Float16)));
    HIP_CHECK(
        hipMemcpy(device_a, host_a.data(), elements_a * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(device_b, host_b.data(), elements_b * sizeof(_Float16), hipMemcpyHostToDevice));

    // A row-major with B column-major is a TN GEMM: op(A) transposed with lda = K,
    // op(B) untransposed with ldb = K, and a column-major C with ldc = M.
    //
    // HIPBLAS_COMPUTE_16F accumulates in FP16, as these kernels do. HIPBLAS_COMPUTE_32F
    // would accumulate in FP32: more accurate, and not what is measured here.
    const _Float16 alpha = static_cast<_Float16>(1.0f);
    const _Float16 beta  = static_cast<_Float16>(0.0f);

    hipblasLtMatmulDesc_t   operation = nullptr;
    hipblasLtMatrixLayout_t layout_a  = nullptr;
    hipblasLtMatrixLayout_t layout_b  = nullptr;
    hipblasLtMatrixLayout_t layout_c  = nullptr;
    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_16F, HIP_R_16F));

    const hipblasOperation_t transpose_a = HIPBLAS_OP_T;
    const hipblasOperation_t transpose_b = HIPBLAS_OP_N;
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(operation,
                                                    HIPBLASLT_MATMUL_DESC_TRANSA,
                                                    &transpose_a,
                                                    sizeof(transpose_a)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(operation,
                                                    HIPBLASLT_MATMUL_DESC_TRANSB,
                                                    &transpose_b,
                                                    sizeof(transpose_b)));

    // Layouts describe the operands before op() is applied, so A is K rows by M columns.
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layout_a, HIP_R_16F, K, M, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layout_b, HIP_R_16F, K, N, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layout_c, HIP_R_16F, M, N, M));

    // Stop if the heuristic returns no solution, rather than falling back silently.
    constexpr int                    requested_solutions = 1;
    hipblasLtMatmulPreference_t      preference          = nullptr;
    hipblasLtMatmulHeuristicResult_t heuristic[requested_solutions];
    int                              returned_solutions = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&preference));

    constexpr uint64_t workspace_bytes = 32 * 1024 * 1024;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(preference,
                                                          HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                          &workspace_bytes,
                                                          sizeof(workspace_bytes)));

    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle,
                                                    operation,
                                                    layout_a,
                                                    layout_b,
                                                    layout_c,
                                                    layout_c,
                                                    preference,
                                                    requested_solutions,
                                                    heuristic,
                                                    &returned_solutions));
    if(returned_solutions == 0)
    {
        std::cerr << "hipBLASLt returned no solution for M=" << M << " N=" << N << " K=" << K
                  << std::endl;
        std::exit(error_exit_code);
    }

    void* workspace = nullptr;
    HIP_CHECK(hipMalloc(&workspace, workspace_bytes));

    auto call_reference = [&]()
    {
        HIPBLASLT_CHECK(hipblasLtMatmul(handle,
                                        operation,
                                        &alpha,
                                        device_a,
                                        layout_a,
                                        device_b,
                                        layout_b,
                                        &beta,
                                        device_c,
                                        layout_c,
                                        device_c,
                                        layout_c,
                                        &heuristic[0].algo,
                                        workspace,
                                        workspace_bytes,
                                        nullptr));
    };

    /// Mean milliseconds per launch over 30 iterations, after 10 warm-up launches.
    //
    // GEMM_PROFILE reduces each stage to a single dispatch and skips timing, so a kernel
    // trace holds one record per stage rather than 40.
    const bool profiling = std::getenv("GEMM_PROFILE") != nullptr;

    auto time_launch = [&](auto launch) -> float
    {
        if(profiling)
        {
            launch();
            HIP_CHECK(hipDeviceSynchronize());
            return 0.0f;
        }

        for(int i = 0; i < 10; ++i)
        {
            launch();
        }
        HIP_CHECK(hipDeviceSynchronize());

        hipEvent_t start;
        hipEvent_t stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        float total = 0.0f;
        for(int i = 0; i < 30; ++i)
        {
            float elapsed = 0.0f;
            HIP_CHECK(hipEventRecord(start));
            launch();
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));
            HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));
            total += elapsed;
        }

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        return total / 30.0f;
    };

    auto tflops
        = [&](float milliseconds) { return 2.0 * M * N * K / (milliseconds * 1e-3) / 1e12; };

    // hipBLASLt and the kernels both write C column-major with ld = M, so checking a stage is a
    // straight element-wise match against this reference, after flushing subnormals.
    std::vector<float> reference(elements_c);

    std::printf("=== M=%d N=%d K=%d ===\n", M, N, K);
    if(!profiling)
    {
        const float reference_ms = time_launch(call_reference);
        std::printf("   %-30s %9.4f ms  %8.2f TFLOPS  (reference)\n",
                    "hipBLASLt",
                    reference_ms,
                    tflops(reference_ms));
    }
    else
    {
        call_reference();
        HIP_CHECK(hipDeviceSynchronize());
    }
    std::vector<_Float16> reference_fp16(elements_c);
    HIP_CHECK(hipMemcpy(reference_fp16.data(),
                        device_c,
                        elements_c * sizeof(_Float16),
                        hipMemcpyDeviceToHost));
    for(size_t i = 0; i < elements_c; ++i)
    {
        reference[i] = flush_subnormal(reference_fp16[i]);
    }

    std::vector<_Float16> host_c(elements_c);

    /// Times a stage, checks it against hipBLASLt, and reports it.
    ///
    /// \param timed  false for stage 4, which swaps a flat read for a buffer read at the
    ///               same tile. It is launched and checked but has no time worth reporting.
    auto benchmark = [&](const char* name, LaunchFunction function, bool timed = true)
    {
        auto launch = [&]() { function(arch, device_a, device_b, device_c, M, N, K); };

        // In profile mode this launch is the only dispatch, so a kernel trace holds one row
        // per stage. Stages that share a kernel differ in their template arguments, which the
        // mangled symbol distinguishes.
        HIP_CHECK(hipMemset(device_c, 0, elements_c * sizeof(_Float16)));
        launch();
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(
            host_c.data(), device_c, elements_c * sizeof(_Float16), hipMemcpyDeviceToHost));

        size_t mismatches = 0;
        for(size_t i = 0; i < elements_c; ++i)
        {
            if(flush_subnormal(host_c[i]) != reference[i])
            {
                ++mismatches;
            }
        }
        const char* verdict = mismatches == 0 ? "PASS" : "FAIL";

        const float milliseconds = (profiling || !timed) ? 0.0f : time_launch(launch);
        if(profiling || !timed)
        {
            std::printf("   %-30s %s\n", name, verdict);
        }
        else
        {
            std::printf("   %-30s %9.4f ms  %8.2f TFLOPS  %s\n",
                        name,
                        milliseconds,
                        tflops(milliseconds),
                        verdict);
        }
        if(mismatches != 0)
        {
            std::printf("   %-30s   %zu of %zu entries differ from hipBLASLt\n",
                        "",
                        mismatches,
                        elements_c);
        }
    };

    if(full_progression)
    {
        benchmark("1 one wave, 32x32", launch_stage01);
        benchmark("2 tiled pipeline 256x256", launch_stage02);
        benchmark("3 padded LDS rows", launch_stage03);
        benchmark("3 XOR-permuted LDS", launch_stage03_swizzled);
        benchmark("4 buffer loads", launch_stage04, /*timed=*/false);
        benchmark("5 four K steps, 128x96", launch_stage05);
        benchmark("6 swizzled block order", launch_stage06);
        benchmark("7 interleaved", launch_stage07);
    }
    benchmark("8 single LDS buffer", launch_stage08);
    // The larger tile is tuned for gfx1201, so only measure it on RDNA 4.
    if(arch == Arch::gfx12)
    {
        benchmark("8 tuned 192x224 4x2 (gfx1201)", launch_stage08_tuned);
    }

    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceDestroy(preference));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(layout_c));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(layout_b));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutDestroy(layout_a));
    HIPBLASLT_CHECK(hipblasLtMatmulDescDestroy(operation));

    HIP_CHECK(hipFree(workspace));
    HIP_CHECK(hipFree(device_a));
    HIP_CHECK(hipFree(device_b));
    HIP_CHECK(hipFree(device_c));
}
} // namespace

int main()
{
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    std::printf("Device: %s (%s)\n", properties.name, properties.gcnArchName);

    // These kernels are written against the gfx11 and gfx12 WMMA instructions, and the
    // policy that supplies them is chosen from this. Say so here rather than launching a
    // kernel whose matrix instruction the target does not have, which would return zeros
    // and look like a correctness bug.
    const Arch arch = detect_arch();
    if(arch == Arch::unknown)
    {
        std::cerr << "This benchmark targets RDNA 3 (gfx11) and RDNA 4 (gfx12). Device "
                  << properties.gcnArchName << " is neither." << std::endl;
        return error_exit_code;
    }

    // Library setup — loading hipBLASLt's solution data and picking an algorithm — happens
    // on the first call and is not timed: every measurement is preceded by ten warm-up
    // launches that time_launch discards.
    hipblasLtHandle_t handle = nullptr;
    HIPBLASLT_CHECK(hipblasLtCreate(&handle));

    run(8192, 8192, 8192, arch, handle);
    run(8192, 28672, 8192, arch, handle, /*full_progression=*/false);

    HIPBLASLT_CHECK(hipblasLtDestroy(handle));
    return 0;
}
