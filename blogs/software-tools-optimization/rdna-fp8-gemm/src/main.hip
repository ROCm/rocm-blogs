// Benchmark driver for the FP8 GEMM kernel.
//
// A is row-major [M][K] and B is column-major [N][K], each element drawn as -1 or +1. hipBLASLt
// runs the same problem with FP8 inputs and FP32 output, matching the WMMA instruction used here.
// Each config is timed and checked against hipBLASLt's output: with +/-1 inputs the two match
// exactly, so a mismatch is a real bug (see the correctness note in run()).
//
// Build with `bash build.sh` and run ./gemm_fp8.
#include "configs.hpp"
#include "gemm_common.hpp"

#include <hipblaslt/hipblaslt.h>

#include <random>
#include <string>
#include <vector>

/// String label for the single-buffer flag, selected by token-paste at macro expansion.
#define true_LABEL "1buf"
#define false_LABEL "2buf"

namespace
{
bool is_gfx12()
{
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    return std::string(properties.gcnArchName).rfind("gfx12", 0) == 0;
}

/// Runs the kernel and the reference at one problem shape, timing and checking each config.
void run(int M, int N, int K, hipblasLtHandle_t handle)
{
    const size_t elements_a = static_cast<size_t>(M) * K;
    const size_t elements_b = static_cast<size_t>(N) * K;
    const size_t elements_c = static_cast<size_t>(M) * N;

    // A and B are dense, with each element drawn as -1 or +1. This lets every config be checked
    // against hipBLASLt exactly, with no tolerance: E4M3 represents +/-1 exactly, so every product
    // is +/-1 and every partial sum an integer. With random inputs those sums stay near
    // +/-sqrt(K), well inside the range FP32 accumulation represents exactly, so a correct kernel
    // must match hipBLASLt bit for bit.
    std::vector<fp8>            host_a(elements_a);
    std::vector<fp8>            host_b(elements_b);
    std::mt19937                generator(42);
    std::bernoulli_distribution coin(0.5);
    const fp8 plus_one  = __hip_cvt_float_to_fp8(1.0f, __HIP_SATFINITE, __HIP_E4M3);
    const fp8 minus_one = __hip_cvt_float_to_fp8(-1.0f, __HIP_SATFINITE, __HIP_E4M3);
    for(auto& value : host_a)
    {
        value = coin(generator) ? plus_one : minus_one;
    }
    for(auto& value : host_b)
    {
        value = coin(generator) ? plus_one : minus_one;
    }

    fp8*   device_a = nullptr;
    fp8*   device_b = nullptr;
    float* device_c = nullptr;
    HIP_CHECK(hipMalloc(&device_a, elements_a * sizeof(fp8)));
    HIP_CHECK(hipMalloc(&device_b, elements_b * sizeof(fp8)));
    HIP_CHECK(hipMalloc(&device_c, elements_c * sizeof(float)));
    HIP_CHECK(hipMemcpy(device_a, host_a.data(), elements_a * sizeof(fp8), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_b, host_b.data(), elements_b * sizeof(fp8), hipMemcpyHostToDevice));

    // A row-major with B column-major: op(A) transposed with lda = K, op(B) untransposed
    // with ldb = K, and a column-major C with ldc = M.
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    hipblasLtMatmulDesc_t   operation = nullptr;
    hipblasLtMatrixLayout_t layout_a  = nullptr;
    hipblasLtMatrixLayout_t layout_b  = nullptr;
    hipblasLtMatrixLayout_t layout_c  = nullptr;
    HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&operation, HIPBLAS_COMPUTE_32F, HIP_R_32F));

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
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layout_a, HIP_R_8F_E4M3, K, M, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layout_b, HIP_R_8F_E4M3, K, N, K));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layout_c, HIP_R_32F, M, N, M));

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
        std::cerr << "hipBLASLt returned no FP8 solution for M=" << M << " N=" << N << " K=" << K
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

    std::printf("=== M=%d N=%d K=%d ===\n", M, N, K);
    if(!profiling)
    {
        const float reference_ms = time_launch(call_reference);
        std::printf("   %-34s %9.4f ms  %8.2f TFLOPS  (reference)\n",
                    "hipBLASLt",
                    reference_ms,
                    tflops(reference_ms));
    }
    else
    {
        call_reference();
    }
    HIP_CHECK(hipDeviceSynchronize());

    // hipBLASLt's output is the correctness reference every config is checked against. Both it
    // and the kernels write C column-major with ld = M, so checking a config is a straight
    // element-wise match against this reference.
    std::vector<float> reference(elements_c);
    HIP_CHECK(
        hipMemcpy(reference.data(), device_c, elements_c * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<float> host_c(elements_c);
    auto               benchmark = [&](const char* name,
                         void (*launch_fn)(const fp8*, const fp8*, float*, int, int, int),
                         int lds_bytes)
    {
        auto launch = [&]() { launch_fn(device_a, device_b, device_c, M, N, K); };

        HIP_CHECK(hipMemset(device_c, 0, elements_c * sizeof(float)));
        launch();
        const hipError_t launch_status = hipGetLastError();
        if(launch_status != hipSuccess)
        {
            std::printf("   %-28s %26s  (%s)\n",
                        name,
                        "did not launch",
                        hipGetErrorString(launch_status));
            return;
        }
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(
            hipMemcpy(host_c.data(), device_c, elements_c * sizeof(float), hipMemcpyDeviceToHost));

        size_t mismatches = 0;
        for(size_t i = 0; i < elements_c; ++i)
        {
            if(host_c[i] != reference[i])
            {
                ++mismatches;
            }
        }
        const bool passed = mismatches == 0;

        const float milliseconds = profiling ? 0.0f : time_launch(launch);

        if(profiling)
        {
            std::printf("   %-34s %s\n", name, passed ? "PASS" : "FAIL");
        }
        else
        {
            std::printf("   %-34s %9.4f ms  %8.2f TFLOPS  %5.1f KB  %s\n",
                        name,
                        milliseconds,
                        tflops(milliseconds),
                        lds_bytes / 1024.0,
                        passed ? "PASS" : "FAIL");
        }
        if(!passed)
        {
            std::printf("   %-34s   %zu of %zu entries differ from hipBLASLt\n",
                        "",
                        mismatches,
                        elements_c);
        }

        HIP_CHECK(hipDeviceSynchronize());
    };

#define RUN_CONFIG(BM, BN, BK, WM, WN, SINGLE)                              \
    benchmark(#BM "x" #BN " k" #BK " " #WM "x" #WN " " SINGLE##_LABEL,      \
              launch<Config<BM, BN, BK, WM, WN, SINGLE>>,                   \
              Config<BM, BN, BK, WM, WN, SINGLE>::lds_bytes);
    FP8_GEMM_CONFIGS(RUN_CONFIG)
#undef RUN_CONFIG
#undef true_LABEL
#undef false_LABEL

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

    if(!is_gfx12())
    {
        std::cerr << "This benchmark targets RDNA 4 (gfx12). "
                  << "Device " << properties.gcnArchName << " is not gfx12." << std::endl;
        return error_exit_code;
    }

    hipblasLtHandle_t handle = nullptr;
    HIPBLASLT_CHECK(hipblasLtCreate(&handle));

    // K = 8192 is the hidden size of 70B-class models. N varies by layer type.
    run(8192, 8192, 8192, handle);   // attention output projection
    run(8192, 28672, 8192, handle);  // MLP up-projection
    run(8192, 10240, 8192, handle);  // GQA QKV projection
    run(3072, 4096, 8192, handle);

    HIPBLASLT_CHECK(hipblasLtDestroy(handle));
    return 0;
}
