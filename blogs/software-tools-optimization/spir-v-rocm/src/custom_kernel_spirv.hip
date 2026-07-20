/**
 * custom_kernel_spirv.hip
 *
 * A vector-add HIP program compiled with the SPIR-V abstract target
 * (`amdgcnspirv`). Apples-to-apples counterpart of
 * custom_kernel_standard.hip: the two files do identical work — same
 * kernel, same launches, same host logic — and differ only in how
 * per-arch dispatch is expressed.
 *
 * Compile:
 *   clang++ -x hip --offload-arch=amdgcnspirv \
 *       custom_kernel_spirv.hip -o vecadd_spirv
 *
 * Recommended stack: ROCm 7.2 or later.
 *
 * The resulting binary runs on any supported AMD GPU; the SPIR-V
 * intermediate representation is lowered to native ISA at runtime,
 * on the first kernel launch.
 *
 * How per-arch dispatch is expressed here (vs. the standard version):
 *   - Compile-time arch macros (__gfx942__, ...) are NOT used — they
 *     do not exist for the abstract amdgcnspirv target.
 *   - In-kernel arch dispatch uses __builtin_amdgcn_processor_is(),
 *     resolved during SPIR-V -> native lowering (one branch survives
 *     per arch).
 *   - Host code reads hipDeviceProp_t::gcnArchName at runtime and
 *     picks the launch config from there.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Intra-kernel arch dispatch: items processed per thread.
// __builtin_amdgcn_processor_is() resolves during SPIR-V->native JIT
// lowering, so exactly one branch survives per arch and the loop in
// vector_add unrolls to a per-arch constant. Same SPIR-V binary,
// different ISA per GPU. (The standard build expresses the same
// dispatch with compile-time #if __gfx__ macros instead.)
__device__ __forceinline__ int items_per_thread_for_arch() {
    if (__builtin_amdgcn_processor_is("gfx942"))  return 4;  // CDNA3 (MI300)
    if (__builtin_amdgcn_processor_is("gfx950"))  return 4;  // CDNA4 (MI350)
    if (__builtin_amdgcn_processor_is("gfx90a"))  return 4;  // CDNA2 (MI200)
    if (__builtin_amdgcn_processor_is("gfx1201")) return 2;  // RDNA4
    if (__builtin_amdgcn_processor_is("gfx1100")) return 2;  // RDNA3
    return 1;
}

// Vector add with per-arch work-per-thread tuning chosen at JIT time.
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    const int items = items_per_thread_for_arch();
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * items;
    #pragma unroll
    for (int k = 0; k < items; ++k) {
        int i = base + k;
        if (i < N) C[i] = A[i] + B[i];
    }
}

// Host-side block-size selection. With SPIR-V, compile-time arch macros
// are unavailable, so the host reads gcnArchName at runtime and picks
// the launch config before the kernel runs.
static int pick_block_size_host(int deviceId) {
    hipDeviceProp_t props{};
    hipGetDeviceProperties(&props, deviceId);
    const char* arch = props.gcnArchName;  // e.g. "gfx950:sramecc+:xnack-"
    if (strncmp(arch, "gfx942",  6) == 0) return 256;  // CDNA3
    if (strncmp(arch, "gfx950",  6) == 0) return 256;  // CDNA4
    if (strncmp(arch, "gfx90a",  6) == 0) return 256;  // CDNA2
    if (strncmp(arch, "gfx1201", 7) == 0) return 128;  // RDNA4
    if (strncmp(arch, "gfx1100", 7) == 0) return 128;  // RDNA3
    return 256;
}

// Mirror of items_per_thread_for_arch() so the host can compute the
// matching grid for the kernel. Keep the two tables in sync.
static int items_per_thread_for_arch_host(int deviceId) {
    hipDeviceProp_t props{};
    hipGetDeviceProperties(&props, deviceId);
    const char* arch = props.gcnArchName;
    if (strncmp(arch, "gfx942",  6) == 0) return 4;
    if (strncmp(arch, "gfx950",  6) == 0) return 4;
    if (strncmp(arch, "gfx90a",  6) == 0) return 4;
    if (strncmp(arch, "gfx1201", 7) == 0) return 2;
    if (strncmp(arch, "gfx1100", 7) == 0) return 2;
    return 1;
}

int main() {
    const int N = 1 << 20;  // 1M elements
    size_t bytes = N * sizeof(float);

    printf("=== SPIR-V HIP Vector Add (portable amdgcnspirv) ===\n");
    printf("Binary is portable -- native code generated at runtime\n\n");

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, bytes);
    hipMalloc(&d_B, bytes);
    hipMalloc(&d_C, bytes);

    hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice);

    // Host-side: pick the launch config from gcnArchName at runtime.
    int deviceId = 0;
    hipGetDevice(&deviceId);
    hipDeviceProp_t props{};
    hipGetDeviceProperties(&props, deviceId);
    int blockSize = pick_block_size_host(deviceId);
    int itemsPerThread = items_per_thread_for_arch_host(deviceId);
    int gridSize = (N + blockSize * itemsPerThread - 1) / (blockSize * itemsPerThread);
    printf("Device: %s (%s)\n", props.name, props.gcnArchName);
    printf("Block size: %d, %d items/thread, grid: %d (chosen on host "
           "from gcnArchName)\n", blockSize, itemsPerThread, gridSize);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // First launch -- includes SPIR-V -> native JIT lowering.
    printf("\nFirst launch (includes SPIR-V JIT compilation):\n");
    hipEventRecord(start);
    vector_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms_first = 0;
    hipEventElapsedTime(&ms_first, start, stop);
    printf("  Kernel time: %.3f ms  (includes JIT overhead)\n", ms_first);

    // Second launch -- native code is cached.
    printf("Second launch (native code cached):\n");
    hipEventRecord(start);
    vector_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms_cached = 0;
    hipEventElapsedTime(&ms_cached, start, stop);
    printf("  Kernel time: %.3f ms  (native speed)\n", ms_cached);

    // Verify
    hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) { correct = false; break; }
    }
    printf("\nResult: %s\n", correct ? "PASS" : "FAIL");

    // Cleanup
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    free(h_A); free(h_B); free(h_C);
    hipEventDestroy(start); hipEventDestroy(stop);

    return 0;
}
