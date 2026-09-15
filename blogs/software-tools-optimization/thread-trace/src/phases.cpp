// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace
{
constexpr uint32_t kBlockSize = 512;
constexpr uint32_t kLdsElements = 8192; // 32 KiB
constexpr uint32_t kLdsValuesPerThread = kLdsElements / kBlockSize;
constexpr uint32_t kComputeIterations = 12288;
constexpr uint32_t kLdsIterations = 16384;
constexpr uint32_t kMixIterations = 12288;
constexpr uint32_t kMemoryBatch = 8;
constexpr uint32_t kMixBatch = 4;
constexpr uint32_t kHotRows = 8;
constexpr uint32_t kCacheRepeats = 512;
constexpr size_t kWorkingSetBytes = 256ULL << 20;
constexpr size_t kElementCount = kWorkingSetBytes / sizeof(float);

void check(hipError_t status, const char* call)
{
    if (status == hipSuccess) return;
    std::cerr << call << ": " << hipGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
}

#define HIP_CHECK(call) check((call), #call)

using index_t = uint64_t;
using lds_float = float __attribute__((address_space(3)));

__device__ __forceinline__ float phase_1_bad_coalescing(
    const float* data,
    index_t element_mask,
    index_t thread_count,
    index_t stream_iterations,
    index_t offset,
    float result
)
{
    const index_t tid = static_cast<index_t>(blockIdx.x) * kBlockSize + threadIdx.x;
    index_t index = ((offset + tid) * 33) & element_mask;
    const index_t step = (thread_count * 33) & element_mask;
    float sums[kMemoryBatch] = {result};

    // The odd 33-float stride is a full permutation of the power-of-two buffer,
    // but adjacent lanes land on separate cache lines.
    for (index_t i = 0; i < stream_iterations; i += kMemoryBatch)
    {
        float values[kMemoryBatch];
#pragma unroll
        for (uint32_t j = 0; j < kMemoryBatch; ++j)
        {
            values[j] = data[index];
            index = (index + step) & element_mask;
        }
#pragma unroll
        for (uint32_t j = 0; j < kMemoryBatch; ++j) sums[j] += values[j];
    }
#pragma unroll
    for (uint32_t j = 1; j < kMemoryBatch; ++j) sums[0] += sums[j];
    return sums[0];
}

__device__ __forceinline__ float phase_2_valu(float seed)
{
    float x0 = seed + 0.01f;
    float x1 = seed + 0.02f;
    float x2 = seed + 0.03f;
    float x3 = seed + 0.04f;
    float x4 = seed + 0.05f;
    float x5 = seed + 0.06f;
    float x6 = seed + 0.07f;
    float x7 = seed + 0.08f;

    for (uint32_t i = 0; i < kComputeIterations; ++i)
    {
        x0 = fmaf(x0, 1.000001f, 0.000001f);
        x1 = fmaf(x1, 0.999999f, 0.000002f);
        x2 = fmaf(x2, 1.000002f, 0.000003f);
        x3 = fmaf(x3, 0.999998f, 0.000004f);
        x4 = fmaf(x4, 1.000003f, 0.000005f);
        x5 = fmaf(x5, 0.999997f, 0.000006f);
        x6 = fmaf(x6, 1.000004f, 0.000007f);
        x7 = fmaf(x7, 0.999996f, 0.000008f);
    }
    return x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}

__device__ __forceinline__ float phase_3_good_coalescing(
    const float* data,
    index_t element_mask,
    index_t thread_count,
    index_t stream_iterations,
    index_t offset,
    float result
)
{
    const index_t tid = static_cast<index_t>(blockIdx.x) * kBlockSize + threadIdx.x;
    index_t index = (offset + tid) & element_mask;
    float sums[kMemoryBatch] = {result};

    // Every lane reads an adjacent element and the complete working set is
    // streamed once, so reuse within L1/L2 is negligible.
    for (index_t i = 0; i < stream_iterations; i += kMemoryBatch)
    {
        float values[kMemoryBatch];
#pragma unroll
        for (uint32_t j = 0; j < kMemoryBatch; ++j)
        {
            values[j] = data[index];
            index = (index + thread_count) & element_mask;
        }
#pragma unroll
        for (uint32_t j = 0; j < kMemoryBatch; ++j) sums[j] += values[j];
    }
#pragma unroll
    for (uint32_t j = 1; j < kMemoryBatch; ++j) sums[0] += sums[j];
    return sums[0];
}

__device__ __forceinline__ float phase_4_lds_bound(lds_float* lds, float result)
{
    // Each lane owns its values, while the lane-major layout intentionally
    // creates LDS bank conflicts.
    const uint32_t base = threadIdx.x * kLdsValuesPerThread;
    float last = 0.0f;
    for (uint32_t i = 0; i < kLdsIterations; i += kMemoryBatch)
    {
        const uint32_t slot = i & (kLdsValuesPerThread - 1);
        float values[kMemoryBatch];
#pragma unroll
        for (uint32_t j = 0; j < kMemoryBatch; ++j) values[j] = lds[base + slot + j];
#pragma unroll
        for (uint32_t j = 0; j < kMemoryBatch; ++j)
        {
            values[j] += 0.00001f;
            lds[base + slot + j] = values[j];
        }
        asm volatile("" ::: "memory");
        last = values[0];
    }
    return result + last;
}

__device__ __forceinline__ float phase_5_cache_hits(
    const float* data,
    index_t element_mask,
    index_t thread_count,
    index_t stream_iterations,
    index_t offset,
    float result
)
{
    const index_t tid = static_cast<index_t>(blockIdx.x) * kBlockSize + threadIdx.x;
    const index_t first_row = stream_iterations - kHotRows;
    float sums[kHotRows] = {result};

    // Re-read the tail of phase 3. It is 16 KiB per workgroup and remains hot
    // in L1/L2; repeated passes dominate the one initial refill if evicted.
    for (uint32_t repeat = 0; repeat < kCacheRepeats; ++repeat)
    {
        index_t index = (offset + first_row * thread_count + tid) & element_mask;
        float values[kHotRows];
#pragma unroll
        for (uint32_t row = 0; row < kHotRows; ++row)
        {
            values[row] = data[index];
            index = (index + thread_count) & element_mask;
        }
#pragma unroll
        for (uint32_t row = 0; row < kHotRows; ++row) sums[row] += values[row];
    }
#pragma unroll
    for (uint32_t row = 1; row < kHotRows; ++row) sums[0] += sums[row];
    return sums[0];
}

__device__ __forceinline__ float phase_6_valu_lds_mix(lds_float* lds, float seed)
{
    float x0 = seed * 0.0001f;
    float x1 = seed * 0.0002f;

    // The row-major layout is conflict-free across lanes. Each batch mixes
    // four LDS load/stores with dependent VALU operations.
    for (uint32_t i = 0; i < kMixIterations; i += kMixBatch)
    {
        uint32_t indices[kMixBatch];
        float values[kMixBatch];
#pragma unroll
        for (uint32_t j = 0; j < kMixBatch; ++j)
        {
            indices[j] = ((i + j) & (kLdsValuesPerThread - 1)) * kBlockSize + threadIdx.x;
            values[j] = lds[indices[j]];
        }
#pragma unroll
        for (uint32_t j = 0; j < kMixBatch; ++j)
        {
            x0 = fmaf(x0, 0.99991f, values[j] * 0.00009f);
            x1 = fmaf(x1, 0.99989f, x0 * 0.00011f);
            values[j] = fmaf(values[j], 0.99987f, x1 * 0.00013f);
            lds[indices[j]] = values[j];
        }
        asm volatile("" ::: "memory");
    }
    return x0 + x1;
}

__global__ void phases(
    const float* bad_data,
    const float* good_data,
    float* output,
    index_t element_mask,
    index_t thread_count,
    index_t stream_iterations,
    uint32_t phase_repeats
)
{
    __shared__ float lds[kLdsElements];
    auto* lds_ptr = (lds_float*) lds;

    for (uint32_t i = threadIdx.x; i < kLdsElements; i += kBlockSize) lds[i] = static_cast<float>(i & 31) * 0.001f;
    __syncthreads();

    float result = static_cast<float>(threadIdx.x + 1) * 0.0001f;
    for (uint32_t repeat = 0; repeat < phase_repeats; ++repeat)
    {
        const index_t bad_offset = static_cast<index_t>(repeat) * 130363;
        const index_t good_offset = static_cast<index_t>(repeat) * 104729;

        result = phase_1_bad_coalescing(bad_data, element_mask, thread_count, stream_iterations, bad_offset, result);
        __syncthreads();

        result = phase_2_valu(result);
        __syncthreads();

        result = phase_3_good_coalescing(good_data, element_mask, thread_count, stream_iterations, good_offset, result);
        __syncthreads();

        result = phase_4_lds_bound(lds_ptr, result);
        __syncthreads();

        result = phase_5_cache_hits(good_data, element_mask, thread_count, stream_iterations, good_offset, result);
        __syncthreads();

        result = phase_6_valu_lds_mix(lds_ptr, result);
        __syncthreads();
    }

    const index_t tid = static_cast<index_t>(blockIdx.x) * kBlockSize + threadIdx.x;
    output[tid] = result;
}
} // namespace

int main(int argc, char** argv)
{
    const uint32_t phase_repeats = argc > 1 ? std::max(1, std::atoi(argv[1])) : 3;

    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));

    const uint32_t block_count = properties.multiProcessorCount;
    const size_t thread_count = static_cast<size_t>(block_count) * kBlockSize;
    const index_t stream_iterations = (kElementCount / thread_count) & ~(static_cast<index_t>(kMemoryBatch) - 1);
    if (stream_iterations < kHotRows)
    {
        std::cerr << "Working set is too small for this GPU\n";
        return EXIT_FAILURE;
    }

    float* bad_data = nullptr;
    float* good_data = nullptr;
    float* output = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&bad_data), kWorkingSetBytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&good_data), kWorkingSetBytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&output), thread_count * sizeof(float)));
    HIP_CHECK(hipMemset(bad_data, 0x3f, kWorkingSetBytes));
    HIP_CHECK(hipMemset(good_data, 0x3e, kWorkingSetBytes));

    // One workgroup per CU keeps the phases approximately aligned without
    // requiring cooperative-launch support.
    phases<<<block_count, kBlockSize>>>(
        bad_data, good_data, output, kElementCount - 1, thread_count, stream_iterations, phase_repeats
    );
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    float result = 0.0f;
    HIP_CHECK(hipMemcpy(&result, output, sizeof(result), hipMemcpyDeviceToHost));
    std::cout << "Completed " << phase_repeats << " phase cycles on " << properties.name << "; result = " << result
              << '\n';

    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(good_data));
    HIP_CHECK(hipFree(bad_data));
    return EXIT_SUCCESS;
}
