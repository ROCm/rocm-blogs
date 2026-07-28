/**
 * custom_kernel_standard.hip
 *
 * A vector-add HIP program compiled for SPECIFIC GPU architectures
 * (a per-`gfx` "fat binary"). Apples-to-apples counterpart of
 * custom_kernel_spirv.hip: the two files do identical work — same
 * kernel, same launches, same host logic — and differ only in how
 * per-arch dispatch is expressed.
 *
 * Compile:
 *   clang++ -x hip --offload-arch=gfx942 --offload-arch=gfx950 \
 *       custom_kernel_standard.hip -o vecadd_fat
 *
 * The resulting binary embeds one native code object per target and
 * only runs on those GPUs.
 *
 * How per-arch dispatch is expressed here (vs. the SPIR-V version):
 *   - DEVICE code uses compile-time macros (__gfx942__, __gfx1201__,
 *     ...). Each --offload-arch device-codegen pass defines its own,
 *     so the #if chain is resolved once per arch, at build time.
 *   - HOST code cannot: those macros are defined only in the device
 *     passes, never the host pass. So host-side launch decisions read
 *     hipDeviceProp_t::gcnArchName at runtime — exactly as the SPIR-V
 *     version does. Host-side arch dispatch is always a runtime query.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Intra-kernel arch dispatch: items processed per thread.
// Compile-time macros are defined per device-codegen pass, so this #if
// chain is resolved once per --offload-arch at build time and the loop
// in vector_add unrolls to a per-arch constant. (The SPIR-V build
// expresses the same dispatch with __builtin_amdgcn_processor_is(),
// resolved at JIT instead — same source shape, different bind time.)
__device__ __forceinline__ int items_per_thread_for_arch() {
#if defined(__gfx942__)
    return 4;  // CDNA3 (MI300)
#elif defined(__gfx950__)
    return 4;  // CDNA4 (MI350)
#elif defined(__gfx90a__)
    return 4;  // CDNA2 (MI200)
#elif defined(__gfx1201__)
    return 2;  // RDNA4
#elif defined(__gfx1100__)
    return 2;  // RDNA3
#else
    return 1;
#endif
}

// Vector add with per-arch work-per-thread tuning baked in at build time.
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    const int items = items_per_thread_for_arch();
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * items;
    #pragma unroll
    for (int k = 0; k < items; ++k) {
        int i = base + k;
        if (i < N) C[i] = A[i] + B[i];
    }
}

// Host-side block-size selection. The compile-time arch macros above
// are unavailable in the host pass, so the host reads gcnArchName at
// runtime and picks the launch config before the kernel runs.
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

    printf("=== Standard HIP Vector Add (per-gfx fat binary) ===\n");
    printf("Binary embeds native code for its --offload-arch targets\n\n");

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

    // First launch -- native code object selected from the fat binary.
    printf("\nFirst launch (native code object from fat binary):\n");
    hipEventRecord(start);
    vector_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms_first = 0;
    hipEventElapsedTime(&ms_first, start, stop);
    printf("  Kernel time: %.3f ms\n", ms_first);

    // Second launch.
    printf("Second launch:\n");
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
