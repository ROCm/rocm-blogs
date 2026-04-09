---
blogpost: true
blog_title: "Introducing hipThreads: A C++ - Style Concurrency Library for AMD GPUs"
date: 19 Feb 2026
author: 'Charles Boyd, Kelvin Lui, Daniel Mcintosh, Marko Savic'
thumbnail: 'introducing-hipthreads-cpp-gpu-concurrency.png'
tags: C++, HPC, Installation, Performance
category: Software tools & optimizations
target_audience: Developers new to GPU programming, and existing GPU developers.
key_value_propositions: This blog introduces a new simpler way to program AMD GPUs by providing C++ concurrency primitives like threads, mutexes, and condition variables.
language: English
myst:
    html_meta:
        "author": "Charles Boyd, Kelvin Lui, Daniel Mcintosh, Marko Savic"
        "description lang=en": "Discover how hipThreads lets you write hip::thread just like std::thread and unlock GPU acceleration with minimal code changes."
        "keywords": "hipThreads, ROCm, HIP, GPU, AMD, HPC, parallel programming, C++, thread, mutex, memory"
        "vertical": "Developers, HPC"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Radeon Graphics, Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Generative AI, Computer Vision"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Charles Boyd, Kelvin Lui, Daniel Mcintosh, Marko Savic"
---

<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->

# Introducing hipThreads: A C++ - Style Concurrency Library for AMD GPUs

In this blog, you will learn how to accelerate C++ code developed for the CPU to run on AMD GPUs using **hipThreads** by incrementally porting familiar
`std::thread` patterns to GPU-resident `hip::thread` code.
We walk through a step-by-step SAXPY example, explain key concepts like persistent threads and fibers, and share real performance
results to help you evaluate when this model fits your workload.

hipThreads introduces a GPU execution model that lets you launch and coordinate work using idioms you already know from the C++
Concurrency Support Library.
Instead of beginning your GPU journey by learning kernel configuration, grid/block semantics, and ad-hoc synchronization, you can write
`hip::thread`, `hip::mutex`, `hip::lock_guard`, and `hip::condition_variable` code that feels structurally similar to your existing
`std::thread`-driven CPU programs—making first contact with GPU compute feel like an incremental extension of existing C++ expertise,
not a wholesale shift in mental model.

The design of hipThreads lets teams port CPU concurrency regions incrementally:
replace `std::thread` with `hip::thread`, adapt synchronization where needed,
and move logic onto the GPU without immediately restructuring everything into bulk kernels.
For newcomers to GPU programming, it reduces cognitive load -
developers can experiment with parallelism using concepts they already trust,
then dive deeper into HIP specifics only as optimization demands.

In short, hipThreads aims to make AMD GPU compute more accessible, more maintainable,
and more aligned with modern C++ concurrency practices,
accelerating both learning curves and codebase evolution.

## Blog Roadmap

This post assumes you're comfortable with modern C++ concepts, especially
multi-threading with `std::thread`. If you can write CPU parallel code with
`std::thread`, you already have most of what you need for hipThreads.

You'll encounter a few GPU-specific concepts as you work through the examples:

- **Device vs. host memory**: GPUs have separate memory from CPUs
- **Device functions**: Mark code to run on the GPU with `__device__`

Don't worry - we introduce each concept exactly when you need it in the step-by-step
SAXPY tutorial below, with clear explanations and working code. You'll learn these
GPU fundamentals naturally as you port your first parallel algorithm. For deeper
background, see the [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html).

We've structured this introduction around the following core sections to provide a clear starting point for exploring hipThreads:

- [Understanding hipThreads](#understanding-hipthreads)
- [How hipThreads work](#how-hipthreads-work)
- [Prerequisites](#hipthreads-prerequisites)
- [Build and Installation](#build-and-installation)
- [Use hipThreads](#how-to-use-hipthreads-in-a-cmake-project)
- [Sample code](#saxpy-sample-code)
- [Performance benefits](#performance-benefits)

> **Note**:
> Our discussion here emphasizes the overall hipThreads approach rather than detailed API usage.
The synchronization primitives mentioned (`hip::mutex`, `hip::lock_guard`, `hip::condition_variable`, etc.)
work similarly to their `std::` counterparts but are designed for GPU execution.
For complete API documentation and detailed explanations of how these primitives work,
please refer to our [hipThreads API Reference Documentation](https://github.com/ROCm/hipThreads/blob/release/0.1.0/README.md#Documentation).

## Understanding hipThreads

hipThreads fills a gap between two existing ecosystems.
rocThrust offers powerful host-side facilities (algorithms, memory management, STL-like utilities)
but does not provide a `std::thread`-style mechanism to start GPU-resident threaded control flows.
libhipcxx delivers device-side building blocks (e.g. `hip::atomic` mirroring `std::atomic`)
once code is already running on the GPU,
but it does not define how you initiate structured, thread-like execution there.
hipThreads bridges these worlds by introducing `hip::thread`, a `std::thread`-inspired abstraction,
plus supporting synchronization primitives,
giving developers a familiar on-ramp from CPU concurrency to persistent, cooperative GPU execution.

The introduction of hipThreads delivers a suite of familiar C++‑style GPU concurrency abstractions
enabling dynamic parallelism without excessive kernel launches:

- **Familiar API**: Launch GPU work with `hip::thread` much like `std::thread`
- **Less overhead**: Persistent scheduler eliminates constant kernel launching
- **Cooperative**: Threads can yield to another thread
- **Standard primitives**: Use familiar synchronization primitives that mirror their CPU counterparts

## How hipThreads work

Instead of launching many separate GPU kernels,
hipThreads launches one long-running "manager" that stays on your GPU.
When you create a `hip::thread`, it gets added to a work queue, and
the manager picks it up and runs it alongside other threads.

Think of it as having a persistent worker pool on your GPU
rather than hiring and firing workers for every small task.
This approach gives you a familiar API where you can write `hip::thread` just like `std::thread`,
while eliminating the overhead of constant kernel launching.

Additionally, each `hip::thread` can run with multiple **"fibers"** - parallel execution lanes that work together within a single virtual thread, taking advantage of the GPU's SIMD (Single Instruction, Multiple Data) architecture.
This allows you to distribute work efficiently across the GPU's vector units
while maintaining the familiar threading abstraction.
For example, when processing an image,
different fibers within the same thread might handle different pixels (0, 4, 8... vs 1, 5, 9...),
automatically spreading the workload across the GPU's parallel processing capabilities
without requiring you to manually manage the low-level details.

This fiber-based execution model is conceptually similar to CPU vectorization technologies like AVX-512.
While an AVX-512 instruction processes 16 single-precision data elements in parallel,
a `hip::thread` running on an AMD GPU typically uses a width of 32 fibers,
offering significantly wider data parallelism within the same programming abstraction.
This means code that already benefits from CPU SIMD can often achieve even greater speedups
when ported to `hip::thread` with minimal structural changes.

## hipThreads Prerequisites

hipThreads has the following prerequisites:

- Linux OS (Ubuntu 22.04+ is recommended)
- CMake 3.21+
- Build tools (e.g., `make` or `ninja`)
- **ROCm 7.12+** (recommended) or **ROCm 7.0.2** — hipThreads depends on HIP and libhipcxx. The code samples also use rocThrust. All dependencies are included in [TheRock](https://rocm.docs.amd.com/en/7.12.0-preview/) builds (ROCm 7.12+).

  For **ROCm 7.12+**: See the [hipThreads 0.1.1 prerequisites](https://github.com/ROCm/hipThreads/blob/release/0.1.1/README.md#prerequisites) for installation and environment setup instructions.

  For **ROCm 7.0.2**: See the [hipThreads 0.1.0 prerequisites](https://github.com/ROCm/hipThreads/blob/release/0.1.0/README.md#prerequisites) for manual installation of ROCm 7.0.2, libhipcxx, and rocThrust.

> **Note:** ROCm 7.12 is part of a technology preview release stream (starting from 7.9.0) and is separate from the 7.0–7.2 production releases. The last supported ROCm 7 production release is 7.0.2.

## Build and Installation

Before building, make sure your ROCm environment is configured properly (e.g., `ROCM_PATH`, `PATH`, `LD_LIBRARY_PATH`). See the [hipThreads Prerequisites](https://github.com/ROCm/hipThreads/blob/release/0.1.1/README.md#Prerequisites) for details.

By default, hipThreads installs under `$ROCM_PATH` (matching other ROCm components). You can override this by adding `-DCMAKE_INSTALL_PREFIX=<path>` to the CMake configure command.

``` bash
git clone https://github.com/ROCm/hipThreads.git
cd hipThreads
cmake -B build
cmake --build ./build
sudo cmake --install ./build
```

> **Note**: Installing to `$ROCM_PATH` usually requires `sudo`.

## How to use hipThreads in a CMake project

To use hipThreads in your own project, add the following lines to your `CMakeLists.txt` file:

``` cpp
  find_package(hipthreads REQUIRED)
  
  [...]

  target_link_libraries(<your_target> hipthreads::hipthreads)
```

## SAXPY Sample Code

We use the well-known SAXPY (Single-precision Alpha X Plus Y) operation
to demonstrate how to incrementally port a CPU-parallel algorithm to the GPU using `hip::thread`.
SAXPY computes `y[i] = alpha * x[i] + y[i]` for each element in arrays `x` and `y`.
This example shows the natural progression from CPU threading to GPU execution,
illustrating how `hip::thread` provides a familiar on-ramp without requiring immediate mastery of GPU-specific concepts.

> **hipThreads SAXPY repository**: <https://github.com/ROCm/hipThreads/tree/release/0.1.0/examples/saxpy>

Build and run (from project root):

```bash
cd examples/saxpy/step3-simdize
mkdir build && cd build
[CXX=hipcc] cmake ..
make -j
./bin/saxpy
```

> **Note**: If hipThreads is not installed in the default `/opt/rocm`,
> add `-DCMAKE_PREFIX_PATH=/path/to/hipThreads` to your CMake configure command.

The sample processes 268,435,456 elements (0x10000000),
with each element undergoing 512 iterations of the computation.
You can adjust these by modifying the `N` and `NUM_ITERATIONS` constants in `main.cxx`.

### Step 1: CPU Baseline with `std::thread`

We start with a standard CPU implementation using `std::thread`.
This establishes our baseline and shows the familiar threading pattern we'll be porting from:

```cpp
#include <thread>
#include <vector>

#define N 0x10000000U
#define NUM_ITERATIONS 512

int main() {
    std::vector<float> x(N, 1.0F);
    std::vector<float> y(N, 2.0F);
    const float alpha = 2.0F;

    std::vector<std::thread> threads(std::thread::hardware_concurrency());
    
    for (unsigned int i = 0; i < threads.size(); ++i) {
        size_t chunk_size = (i < N % threads.size()) ? (N / threads.size() + 1) : (N / threads.size());
        size_t offset =     (i < N % threads.size()) ? (i * chunk_size)         : (i * chunk_size + N % threads.size());
        
        threads[i] = std::thread(
            [] (uint32_t n, float a, const float *x, float *y) {
                for (uint32_t i = 0; i < n; ++i) {
                    float t = x[i];
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < NUM_ITERATIONS; ++j) {
                        t = a * t + y[i];
                    }
                    y[i] = t;
                }
            },
            chunk_size, alpha, x.data() + offset, y.data() + offset);
    }
    
    for (auto &t : threads) {
        t.join();
    }
    
    return 0;
}
```

This CPU version divides the work across available CPU cores,
with each thread processing its assigned chunk of the arrays sequentially.

### Step 2: First GPU Port - Drop-in Replacement with 1-Wide `hip::thread`

The first step in porting to GPU is a minimal change: replace `std::thread` with `hip::thread`.
At this stage, we use the default thread width (1 fiber per thread),
making `hip::thread` a near drop-in replacement for `std::thread`.
This lets you verify GPU execution works before optimizing for GPU-specific features.

**Key changes from Step 1**:

- Line 1: Change `#include <thread>` to `#include <hip/thread>`
- Lines 5-11: Handle GPU memory allocation (explained below)
- Line 13: Change `std::thread` to `hip::thread`
- Line 18: Add `__device__` annotation to the lambda

```cpp
#include <hip/thread>
#include <thrust/unique_ptr.h>
#include <thrust/copy.h>
#include <vector>

#define N 0x10000000U
#define NUM_ITERATIONS 512

int main() {
    // Create and initialize vectors on the host
    std::vector<float> x_host(N, 1.0F);
    std::vector<float> y_host(N, 2.0F);
    const float alpha = 2.0F;

    // Allocate device memory and copy data to GPU
    thrust::unique_ptr<float[]> x = thrust::make_unique<float[]>(N);
    thrust::unique_ptr<float[]> y = thrust::make_unique<float[]>(N);
    thrust::copy(x_host.begin(), x_host.end(), x.get());
    thrust::copy(y_host.begin(), y_host.end(), y.get());

    { // hip::thread scope starts here
        std::vector<hip::thread> threads(hip::thread::hardware_concurrency());
        
        for (unsigned int i = 0; i < threads.size(); ++i) {
            size_t chunk_size = (i < N % threads.size()) ? (N / threads.size() + 1) : (N / threads.size());
            size_t offset =     (i < N % threads.size()) ? (i * chunk_size)         : (i * chunk_size + N % threads.size());
            
            threads[i] = hip::thread(
                [] __device__(uint32_t n, float a, const float *x, float *y) {
                    for (uint32_t i = 0; i < n; ++i) {
                        float t = x[i];
                        for (int j = 0; j < NUM_ITERATIONS; ++j) {
                            t = a * t + y[i];
                        }
                        y[i] = t;
                    }
                },
                chunk_size, alpha, x.get_raw() + offset, y.get_raw() + offset);
        }
        
        for (auto &t : threads) {
            t.join();
        }
    } // hip::thread scope ends here
    
    // Copy results back from GPU to host if necessary
    thrust::copy(y.get(), y.get() + N, y_host.begin());
    
    return 0;
}
```

**Understanding GPU Memory Management**:
The CPU and GPU have separate memory spaces.
We use `thrust::unique_ptr` to allocate GPU-accessible memory and
`thrust::copy` to transfer data between host and device.
The scoped block (`{ ... }`) ensures all `hip::thread` objects are destroyed
before we copy results back, preventing deadlocks with synchronous HIP operations (explained below).

**Important**: Notice the loop structure inside the lambda remains identical to the CPU version.
Each thread still processes elements sequentially (`for (uint32_t i = 0; i < n; ++i)`).
We haven't yet leveraged the GPU's SIMD capabilities - this is purely a threading model change.

### Step 3: Optimizing with Multi-Fiber Execution

Now we unlock the GPU's true potential by using **wide threads** with multiple fibers.
Instead of each thread processing elements sequentially,
we create threads with 32 fibers that process 32 elements in parallel using the GPU's SIMD architecture.

**Key changes from Step 2**:

- Line 28: Add `hip::thread::max_width()` parameter to create a 32-fiber thread
- Line 30: Change loop to distribute work across fibers using `hip::this_thread::get_fiber_id()` and `hip::this_thread::get_width()`

```cpp
#include <hip/thread>
#include <thrust/unique_ptr.h>
#include <thrust/copy.h>
#include <vector>

#define N 0x10000000U
#define NUM_ITERATIONS 512

int main() {
    // Create and initialize vectors on the host
    std::vector<float> x_host(N, 1.0F);
    std::vector<float> y_host(N, 2.0F);
    const float alpha = 2.0F;

    // Allocate device memory and copy data to GPU
    thrust::unique_ptr<float[]> x = thrust::make_unique<float[]>(N);
    thrust::unique_ptr<float[]> y = thrust::make_unique<float[]>(N);
    thrust::copy(x_host.begin(), x_host.end(), x.get());
    thrust::copy(y_host.begin(), y_host.end(), y.get());

    { // hip::thread scope starts here
        std::vector<hip::thread> threads(hip::thread::hardware_concurrency());
        
        for (unsigned int i = 0; i < threads.size(); ++i) {
            size_t chunk_size = (i < N % threads.size()) ? (N / threads.size() + 1) : (N / threads.size());
            size_t offset =     (i < N % threads.size()) ? (i * chunk_size)         : (i * chunk_size + N % threads.size());
            
            threads[i] = hip::thread(hip::thread::max_width(),
                [] __device__(uint32_t n, float a, const float *x, float *y) {
                    for (uint32_t i = hip::this_thread::get_fiber_id(); i < n; i += hip::this_thread::get_width()) {
                        float t = x[i];
                        for (int j = 0; j < NUM_ITERATIONS; ++j) {
                            t = a * t + y[i];
                        }
                        y[i] = t;
                    }
                },
                chunk_size, alpha, x.get_raw() + offset, y.get_raw() + offset);
        }
        
        for (auto &t : threads) {
            t.join();
        }
    } // hip::thread scope ends here
    
    // Copy results back from GPU to host if necessary
    thrust::copy(y.get(), y.get() + N, y_host.begin());
    
    return 0;
}
```

**Understanding Fibers**:
Each fiber in the thread is numbered 0 to 31 (for a width of 32).
By starting the loop at `hip::this_thread::get_fiber_id()` and incrementing by `hip::this_thread::get_width()`,
we distribute the work: fiber 0 processes elements 0, 32, 64..., fiber 1 processes 1, 33, 65..., and so on.
The GPU executes all 32 fibers in lockstep, processing 32 elements simultaneously.

**Scoping and Synchronization**:
The scoped block (`{ ... }`) around `hip::thread` objects is required in this example because we're using rocThrust,
which makes synchronous HIP API calls (like `thrust::copy`).
The persistent kernel scheduler holds the GPU context until all `hip::thread` objects are destroyed.
Without the scope, synchronous HIP calls would wait for the GPU context that the scheduler is holding, causing a deadlock.

If your application uses only asynchronous HIP APIs (e.g., `hipMemcpyAsync`, `hipMallocAsync`) with streams,
you don't need this scoping pattern - async calls don't block waiting for the GPU context.
However, many libraries (like rocThrust) use synchronous HIP functions internally,
so the scoping pattern is a safe practice when mixing `hip::thread` with external libraries.

## Performance Benefits

A small baseline port already cuts frame time from
**271.88ms** on a *Ryzen™ 9 9900X* (`std::thread` + AVX‑512, 16 SIMD lanes) to
**42.60ms** on an *AMD Radeon™ AI PRO R9700* (`hip::thread`, 32 lanes) — about a 6.4× speedup.<sup>[1](#footnote-1)</sup>
This gain comes largely from wider parallelism and the persistent multi‑fiber execution model,
without deep algorithmic changes.
Single run, untuned numbers; they illustrate approach's potential, not peak performance.

| Platform | Implementation | Time (ms) | Lanes | Relative |
|----------|----------------|---------:|-------|----------|
| AMD Ryzen™ 9 9900X 12-Core | `std::thread` + AVX-512 | 271.88 | 16 | 1.00× (baseline) |
| AMD Radeon™ AI PRO R9700 | `hip::thread` | 42.60 | 32 | 6.4× faster<sup>[1](#footnote-1)</sup> |

**Validated Across Diverse Workloads**: Beyond SAXPY, we've tested hipThreads on additional real-world applications, consistently achieving 2.9-6.4× speedups:

| Example | Workload Type | CPU Performance | GPU Performance | Speedup |
|---------|---------------|-----------------|-----------------|---------|
| **Sparse Matrix Multiply** | Complex data structures | 188.57s | 52.30s | **3.6×**<sup>[3](#footnote-3)</sup> |
| **InOneWeekend Ray Tracer** | Graphics rendering | 1.610s | 0.559s | **2.9×**<sup>[2](#footnote-2)</sup> |

These results demonstrate that hipThreads delivers meaningful performance gains across different algorithm types—from complex data structures to graphics and AI workloads—all with minimal code changes and familiar C++ patterns.

> **Note**: All benchmarks were conducted on the same test system: AMD Ryzen™ 9 9900X, AMD Radeon™ AI PRO R9700, 64GB DDR5-4800 RAM, ASUS TUF GAMING B850-PLUS WIFI motherboard, Ubuntu 24.04.2 LTS, using ROCm 7.0.2 and AMDGPU driver 6.16.6.

## Incremental Porting Summary

The three-step progression demonstrates hipThreads' design philosophy:

1. **Step 1 (CPU Baseline)**: Standard `std::thread` implementation
2. **Step 2 (Drop-in Replacement)**: Minimal changes to move to GPU with 1-wide threads
3. **Step 3 (Optimized)**: Leverage GPU SIMD with multi-fiber execution

**Total code changes from Step 1 to Step 3**: 16 lines

- **Added**: 11 lines (4 includes, 4 memory allocation/transfer, 2 scoping, 1 result copy)
- **Modified**: 5 lines (namespace, device annotation, fiber width, fiber-aware loop, pointer access)

This incremental approach lets developers:

- Verify GPU execution works before optimizing
- Understand each concept (memory management, then fibers) separately
- Maintain familiar threading semantics throughout
- Achieve significant speedups with minimal code changes

The 96% code similarity between CPU and optimized GPU implementations shows that
`hip::thread` successfully abstracts GPU complexity,
making high-performance computing accessible with a minimal learning curve.

## Summary

In this blog, you learned how to accelerate C++ code on AMD GPUs using **hipThreads** by incrementally porting familiar `std::thread`
patterns to GPU-resident `hip::thread` code. Here's what we covered:

- **Core concepts**: How hipThreads bridges CPU and GPU programming through persistent threads and multi-fiber execution
- **Step-by-step porting**: A three-stage progression from CPU baseline to optimized GPU code with only 16 lines changed
- **Performance gains**: Real benchmarks showing 2.9-6.4× speedups across diverse workloads (SAXPY, ray tracing, sparse matrix multiply)
- **Practical patterns**: Memory management with `thrust::unique_ptr`, and fiber-aware loop structures

hipThreads lowers the barrier to GPU programming by letting you write `hip::thread` just like `std::thread`, while a persistent
scheduler and multi-fiber execution model handle the GPU-specific complexity behind the scenes. Whether you're new to GPU programming
or looking for a more maintainable way to structure GPU code, hipThreads offers an incremental path from CPU concurrency to
high-performance GPU execution.

Get the most out of hipThreads by exploring the **[hipThreads API Reference Documentation](https://github.com/ROCm/hipThreads/blob/release/0.1.0/README.md#Documentation)**, which covers the complete API,
real-world usage examples, and detailed explanations of every synchronization primitive.

As hipThreads evolve,
planned enhancements include tighter async API integration,
expanded synchronization constructs, and
utilities to simplify common porting patterns.
Community feedback is encouraged:
share use cases, feature requests, pain points, or integration concerns
to help refine priorities and guide upcoming releases.

Stay tuned for future updates,
deeper performance tuning guides, and
additional samples.

## Footnotes

<a id="footnote-1"></a>*[1] Testing by AMD as of February 2026 on the AMD Radeon™ AI PRO R9700 using ROCm 7.0.2 and AMDGPU driver 6.16.6, with HIP Threads running on the GPU versus standard threads on the CPU, on a test system configured with an AMD Ryzen™ 9 9900X, AMD Radeon™ AI PRO R9700, 64GB DDR5-4800 RAM, ASUS TUF GAMING B850-PLUS WIFI motherboard, and Ubuntu 24.04.2 LTS, using the SAXPY (Single-precision A times X plus Y) computation function test. System manufacturers may vary configurations, yielding different results. RPS-167.*

<a id="footnote-2"></a>*[2] Testing by AMD as of February 2026 on the AMD Radeon™ AI PRO R9700 using ROCm 7.0.2 and AMDGPU driver 6.16.6, with HIP Threads running on the GPU versus standard threads on the CPU, on a test system configured with an AMD Ryzen™ 9 9900X, AMD Radeon™ AI PRO R9700, 64GB DDR5-4800 RAM, ASUS TUF GAMING B850-PLUS WIFI motherboard, and Ubuntu 24.04.2 LTS, using the "Ray Tracing in One Weekend" ray traced rendering test. System manufacturers may vary configurations, yielding different results. RPS-168.*

<a id="footnote-3"></a>*[3] Testing by AMD as of February 2026 on the AMD Radeon™ AI PRO R9700 using ROCm 7.0.2 and AMDGPU driver 6.16.6, with HIP Threads running on the GPU versus standard threads on the CPU, on a test system configured with an AMD Ryzen™ 9 9900X, AMD Radeon™ AI PRO R9700, 64GB DDR5-4800 RAM, ASUS TUF GAMING B850-PLUS WIFI motherboard, and Ubuntu 24.04.2 LTS, using the Sparse Matrix Multiply (pwtk.mtx) test. System manufacturers may vary configurations, yielding different results. RPS-169.*

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is
not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS”
WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT
YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

```{update} Apr 9, 2026
Updated hipThreads library to support the newest ROCm 7.12+ in addition to ROCm 7.0.2.
```
