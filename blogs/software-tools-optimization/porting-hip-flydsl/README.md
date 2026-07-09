---
blogpost: true
blog_title: "Porting High-Performance HIP Kernels to FlyDSL"
date: "09 Jul 2026"
author: "Christian Gilli, Amanzhol Salykov, Andy Luo"
thumbnail: 'porting-hip-flydsl-thumbnail.png'
tags: "AI/ML, Linear Algebra, C++, HPC, Performance"
category: "Software tools & optimizations"
target_audience: "AI/ML/HPC engineers, researchers, developers"
key_value_propositions: "This blog shows how to port an high-performance GEMM kernel from HIP C++ to FlyDSL, AMD new purpose-built DSL for authoring GPU kernels, showcasing its strengths over C++."
language: English
myst:
    html_meta:
        "author": "Christian Gilli, Amanzhol Salykov, Andy Luo"
        "description lang=en": "This blog post shows how to port HIP C++ GPU kernels to FlyDSL, AMD's new Python DSL, matching hand-tuned C++ performance with less code."
        "keywords": "FlyDSL, HIP, CDNA™4, GEMM, ROCm"
        "vertical": "HPC"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Christian Gilli, Amanzhol Salykov, Andy Luo"
---

<!---
Copyright (c) 2026 Advanced Micro Devices, Inc. (AMD)

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

# Porting High-Performance HIP Kernels to FlyDSL

In our series of posts we have explored how to utilize [Matrix Core instructions](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html) and how to design high-performance [GEMM](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html) [kernels](https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html) using HIP C++. In this post we'll show how to port those kernels to [FlyDSL](https://github.com/ROCm/FlyDSL), a new Python DSL developed at AMD to simplify kernel development and testing. By working through real-world examples, we'll explore the key concepts of FlyDSL and how they map to low-level HIP C++ code. Finally, we'll show how — despite being higher-level — FlyDSL kernels can match or even exceed the performance of hand-tuned C++ kernels with a fraction of the complexity.

By the end of this post you'll know how to:

- Structure a FlyDSL kernel
- Access global memory using buffer resource descriptors
- Manage LDS in kernel code
- Work with MFMA instructions
- Pre-compile kernels for minimal launch overhead

## What is FlyDSL?

FlyDSL stands for "Flexible Layout Python DSL". It's an MLIR-based Python DSL for authoring high-performance GPU kernels. Unlike [Triton](https://openai.com/index/triton/), which abstracts away a lot of details, FlyDSL gives expert users fine-grained low-level control over the kernel code, enabling speed-of-light performance.

"Flexible Layout" because FlyDSL has first-class support for [CuTe layout algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html). This provides a formally verified mathematical framework for expressing tensor computations that is consistent and portable. Users already familiar with CUTLASS/CuTe DSL should feel immediately at home.

Layouts are not the only way of using FlyDSL though, regular indexing is well-supported and it's easier to understand for users not already familiar with layout concepts.

## Getting FlyDSL

```{note}
All the codes for this article were tested on an AMD Instinct™ MI355X GPU using ROCm 7.2.2 and FlyDSL 0.2.0. FlyDSL is still under very active development, so some of the code examples may not work as intended in future versions.
```

The easiest way to install FlyDSL is via `pip`:

```bash
pip install flydsl
```

If you want to install the latest nightly version run:

```bash
pip install --extra-index-url https://rocm.frameworks-nightlies.amd.com/whl/gfx942-gfx950/ flydsl
```

Or check out the [ROCm nightlies](https://rocm.frameworks-nightlies.amd.com/whl/gfx942-gfx950/flydsl/) and add a specific tag to the `pip install` command above.

You can also build from source, for this, follow the instructions in the [FlyDSL GitHub repository](https://github.com/ROCm/FlyDSL#build-from-source).
Note that building from source will take quite some time and is required only if you're developing FlyDSL itself. If you're not developing the language, nightlies are more than enough for you to stay at the bleeding edge.

## Our First FlyDSL Kernel

To get a feel for what FlyDSL code looks like let's start with the simplest useful kernel: vector addition.

For the sake of simplicity, let's assume both A and B are 1D vectors with the same number of FP32 elements and that we can launch enough blocks to handle all of them.

The HIP C++ code will look something like this:

```c++
#include <hip/hip_runtime.h>

__global__ void add_kernel(float *A, float *B, float *C, int ne) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= ne) return;

    C[tid] = A[tid] + B[tid];
}

// Launch function
void add(float *A, float *B, float *C, int ne) {
    int block_dim = 256;
    int grid_dim = (ne + block_dim - 1) / block_dim;
    add_kernel<<<grid_dim, block_dim>>>(A, B, C, ne);
}

```

### Structure of a FlyDSL Kernel

Each FlyDSL kernel has two components:

1. **Kernel code** — defined as a Python function marked with `@flyc.kernel` where we'll actually write our kernel logic using the DSL. Can only be called inside a `@flyc.jit` function.
2. **Launch code** — marked with `@flyc.jit`. Host-side function that launches the kernel, gets JIT-compiled on first call.

For our vector addition we'll have something like this:

```python
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx

@flyc.kernel
def add_kernel(A: fx.Pointer, B: fx.Pointer, C: fx.Pointer, ne: fx.Int32):
    ...

@flyc.jit
def add(A: fx.Pointer, B: fx.Pointer, C: fx.Pointer, ne: fx.Int32, stream: fx.Stream):
    block_dim = 256
    grid_x = (ne + block_dim - 1) // block_dim
    add_kernel(A, B, C, ne).launch(grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)
```

The launch code is the same as in the HIP C++ version. The only differences are that we have to spell out that `grid` and `block` are tuples, and we must specify a stream. We could also use `fx.Stream(None)` instead of passing a stream parameter, but this will create a new HIP stream for each call, needlessly adding latency.

Finally, we can write a small harness to exercise our kernel:

```python
n = 10
A = torch.rand(n, device="cuda")
B = torch.rand(n, device="cuda")
C = torch.empty(n, dtype=torch.float32, device="cuda")

add(A, B, C, n, torch.cuda.default_stream())
torch.cuda.synchronize()

print(f'Add correct: {torch.allclose(C, A + B)}')
```

PyTorch tensors are automatically converted to FlyDSL `Pointer` (or `Tensor`) objects via DLPack, same goes for HIP streams.

The kernel body is empty so it won't produce correct results, but it will compile and run.
To verify that FlyDSL is actually compiling something we can dump the generated IR and inspect it,
to do so specify the environment variables `FLYDSL_DUMP_IR=1` and `FLYDSL_DUMP_DIR=./dumps` and run the test harness.
If everything is configured correctly you'll find a bunch of MLIR files and a `final_isa.s` file that contains the actual
assembly that gets executed by the GPU, inside that you'll find (amongst many other things):

```asm
add_kernel_0:
    s_endpgm
```

A kernel that simply exits, exactly what we expect.

### Vector Addition in FlyDSL

For the addition kernel we need to understand how to express these concepts in FlyDSL:

1. Access built-in constant (i.e. `threadIdx`)
2. **Load** from global memory
3. **Store** to global memory

Built-in constants are part of the `expr` package:

| HIP C++ | FlyDSL |
| ----------- | --------------- |
| `threadIdx` | `fx.thread_idx` |
| `blockIdx` | `fx.block_idx` |
| `blockDim` | `fx.block_dim` |
| `gridDim` | `fx.grid_dim` |

For loading and storing data, FlyDSL gives you 3 options:

1. Raw pointer indexing
2. Buffer resource descriptors
3. Copy atoms using layout algebra

We'll show only methods 1 and 2 in this article and leave layouts for future ones.

#### Raw Pointers in FlyDSL

Using raw pointers is the most straightforward way of porting a HIP C++ kernel to FlyDSL:

```python
@flyc.kernel
def add_kernel(A: fx.Pointer, B: fx.Pointer, C: fx.Pointer, ne: fx.Int32):
    idx = fx.thread_idx.x + fx.block_idx.x * fx.block_dim.x
    
    if idx >= ne:
        return
    
    C[idx] = A[idx] + B[idx]
```

One nice quality-of-life improvement over regular C++ pointers is that `Pointer` objects know their shape, stride and dtype.
This means that if `X` is a 2D row-major matrix, we can access the j-th element of the i-th row as `el = X[i, j]`.

#### Buffer Resource Descriptors

Buffer resource descriptors are objects that wrap a raw pointer with some metadata and are used by the hardware to do out-of-band checking[^1].
Utilities to work with buffer resources are also defined in the `expr` package.

[^1]: To learn more, check out the [CDNA™4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf), there's a section dedicated to buffer resources.

With these, we can write our add kernel in FlyDSL as follows:

```python
from flydsl.expr import buffer_ops

@flyc.kernel
def add_kernel(A: fx.Pointer, B: fx.Pointer, C: fx.Pointer, ne: fx.Int32):
    tid = fx.thread_idx.x + fx.block_idx.x * fx.block_dim.x
    
    # Not needed, buffer resources guard us against OOB access. 
    # if tid >= ne:
    #     return
    
    # Step 1: create a buffer resource for every tensor we have
    A_rsrc = buffer_ops.create_buffer_resource(A)
    B_rsrc = buffer_ops.create_buffer_resource(B)
    C_rsrc = buffer_ops.create_buffer_resource(C)

    # Step 2: load data from global memory to registers.
    #         By specifying `vec_width` > 1 we can easily issue a vectorized load.
    a_v = buffer_ops.buffer_load(A_rsrc, fx.Int32(tid), vec_width=1, dtype=fx.Float32)
    b_v = buffer_ops.buffer_load(B_rsrc, fx.Int32(tid), vec_width=1, dtype=fx.Float32)

    # Step 3: add
    c_v = a_v + b_v

    # Step 4: store the result back in global memory
    buffer_ops.buffer_store(c_v, C_rsrc, fx.Int32(tid))
```

Since we're using buffer resources to access our data, we don't even need to guard against out-of-band read/writes, the hardware will take care
of everything for us.

If you now run this kernel you'll see the test passes.

## A More Complex Kernel

Now that we laid down the basics of FlyDSL, let's tackle a more complex kernel: matrix multiplication.
Specifically, we'll take the 4-Wave interleave kernel that we developed in [our previous post](https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html) and rewrite it in FlyDSL.
The full HIP C++ code is available on [GitHub](https://github.com/nirw4nna/hipgemm). We won't go into the details of the implementation but rather discuss how we can express low-level patterns in FlyDSL.

Figure 1 shows a high-level overview of the algorithm we'll be implementing:

```{figure} ./images/gemm_struct.png
:align: center
:alt: Figure 1: Algorithm Overview
Figure 1: Algorithm Overview
```

For a more detailed breakdown, check out our previous post.

We already introduced buffer resources, but we still need to cover some core concepts:

1. Allocate LDS
2. Load from global memory to LDS without going through registers
3. Load from LDS to registers in a vectorized way
4. Issue MFMA instructions

Let's start by writing down the kernel stub, we'll introduce a new pattern that you'll find in most of the production FlyDSL kernels: wrapping the `@flyc.kernel` and `@flyc.jit` function in a Python-level factory that returns the launcher.
This is done so we can do **metaprogramming** at the Python-level, sort of like what you'd do with C++ templates but in a much more convenient way.

```python
# These are all the imports we need for the 4-wave interleave kernel.
# To make the code less verbose, all the subsequent snippets will assume these are present.
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

def compile_fp8_gemm_4wave_256x256x128(
        *,
        M: int,
        N: int,
        K: int,
):
    # Tile sizes are constants, for now even though we have OOB protection
    # we will work only with matrices that are multiple of the tile size.
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 128

    @flyc.kernel
    def kernel_gemm(
        A: fx.Pointer,
        B_T: fx.Pointer,
        C: fx.Pointer,
    ):
        ...

    @flyc.jit
    def launch_gemm(
        A: fx.Pointer,
        B_T: fx.Pointer,
        C: fx.Pointer,
        stream: fx.Stream
    ):
        grid_x = (M * N) // (256 * 256)
        kernel_gemm(A, B_T, C).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
```

In the add example, we directly called the launch function. This is fine for infrequent calls or when we don't really care about latency. Often, what we want is to pre-compile the launcher once and then be able to call it with minimal overhead over and over again. This can be done easily in FlyDSL by explicitly invoking the compiler on the launcher function:

```python
# Prepare the kernel by invoking the factory function once.
launch_gemm_fn = compile_fp8_gemm_4wave_256x256x128(M=M, N=N, K=K)
stream = torch.cuda.current_stream()

# Compile the kernel once. Only positional arguments are allowed.
# `Constexpr` values are baked in at compile time and ignored on subsequent calls.
compiled_gemm = flyc.compile(launch_gemm_fn, A, B_T, C, stream)

# Call the compiled kernel with runtime arguments however many times you need.
compiled_gemm(A, B_T, C, stream)
```

### LDS in FlyDSL

In HIP C++, we have `__shared__` arrays:

```c++
__shared__ fp8 A_lds[2][2][128 * 128];
__shared__ fp8 B_lds[2][2][128 * 128];
```

In FlyDSL, we can use simple Python lists:

```python
# (inside a `@flyc.kernel` function)
# ...
lds_allocator = fx.SharedAllocator()

A_lds = [
    # Each element is an array of 128x128 FP8 with 16 bytes alignment. These are FlyDSL `Pointer` objects.
    [lds_allocator.allocate(fx.Array[fx.Float8E4M3FN, 128*128, 16]).peek().ptr for _ in range_constexpr(2)]
    for _ in range_constexpr(2) # Number of stages in the pipeline (2=double-buffering)
]

B_lds = [
    [lds_allocator.allocate(fx.Array[fx.Float8E4M3FN, 128*128, 16]).peek().ptr for _ in range_constexpr(2)]
    for _ in range_constexpr(2)
]
# ...
```

`range_constexpr` is a FlyDSL construct that maps to a fully unrolled loop at compile-time. This can be used together with
Python list comprehension to provide a powerful mechanism for compile-time unrolling to build static data structures.
Right now we hard-coded 2 because we're doing double buffering but we can actually make this parametric. For example, the user could specify a `n_lds_stages` parameter when calling `compile_fp8_gemm_4wave_256x256x128` or, we could come up with a strategy for using more LDS stages based on the actual LDS footprint of our kernel.

### Loading LDS

For loading data from global memory to LDS without touching VGPRs, we need to issue the `llvm.amdgcn.raw.buffer.load.lds` LLVM instruction. The signature for that can be found in the [LLVM documentation](https://clang.llvm.org/docs/AMDGPUBuiltinReference.html#builtin-amdgcn-raw-ptr-buffer-load-lds).

```c++
using i32x4 = int __attribute__((__vector_size__(4 * sizeof(int))));
using as3_uint32_ptr = uint32_t __attribute__((address_space(3))) *;

extern "C" __device__ void
llvm_amdgcn_raw_buffer_load_lds(i32x4 rsrc,
                                as3_uint32_ptr lds_ptr,
                                int size,
                                int voffset,
                                int soffset,
                                int offset,
                                int aux) __asm("llvm.amdgcn.raw.buffer.load."
                                               "lds");

static __device__ i32x4 make_buffer_resource(const void *ptr,
                                             const int32_t range) {
    struct BufferResource {
        const void *ptr;
        int32_t range;
        int32_t config;
    };

    BufferResource res{ ptr, range, 0x110000 };
    return std::bit_cast<i32x4>(res);
}

template <int K>
static __device__ void
load_lds(fp8 *shmem, const fp8 *gmem_base, const int k_offset, const int4 &offsets) {
    const int wave_id = threadIdx.x / 64;
    auto lds_ptr = shmem + wave_id * 1024;

    auto buffer_dsc = make_buffer_resource(gmem_base + k_offset, K * 128);
    for (int step = 0; step < 4; ++step) {
        auto lds_ptr_raw = (as3_uint32_ptr)((uintptr_t)lds_ptr + step * 4096);
        llvm_amdgcn_raw_buffer_load_lds(buffer_dsc, lds_ptr_raw, 16,
                                        offsets[step], 0, 0, 0);
    }
}
```

```python
def load_lds(gl_src, lds_dst, k_offset, offsets):
    wave_id = fx.thread_idx.x // 64
    # The LDS allocator gives back a `Pointer` object which is a FlyDSL primitive,
    # we need to cast that to an LLVM pointer.
    lds_base = fx.Int32(fx.ptrtoint(lds_dst))
    for step in range_constexpr(4):
        # Create the LLVM pointer as `raw_ptr_buffer_load_lds` expects.
        lds_ptr = buffer_ops.create_llvm_ptr(
            lds_base + fx.Int32(wave_id * 1024 + step * 4096),
            address_space=3 # LDS address space
        )
        rocdl.raw_ptr_buffer_load_lds(
            gl_src,                     # SRC buffer resource descriptor
            lds_ptr,                    # DST pointer (`void address_space<3> *` in LLVM jargon)
            fx.Int32(16),               # 16 bytes per thread --> buffer_load_dwordx4
            fx.Int32(offsets[step]),    # voffset (each thread can have a different value)
            fx.Int32(k_offset),         # soffset (all threads within a block share the same value)
            fx.Int32(0),
            fx.Int32(0)
        )
```

In C++, creating an LLVM pointer was a simple cast, here we have to be explicit but the result is exactly the same.

While the result is exactly the same, the developer experience matters too: the HIP C++ version feels a bit off due to all the gymnastics
required to interact with LLVM and have it spit out the instruction we want, in FlyDSL everything feels more "first-class".

### Loading to Registers

In HIP C++, we had to rely on inline assembly to load data from LDS to registers and manually insert waits before using those registers:

```c++
static __device__ void load_rt(const fp8 *lds_src, RT_ABt &dst, const int wave_idx) {
    const int lane_id = threadIdx.x % 64;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = wave_idx * 64 + i * 16 + lane_id % 16;
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const int col = (lane_id / 16) * 16 + step * 64;
            const auto swizzle = swizzle_128(row, col);
            const auto lds_addr =
            (uint32_t)(((uintptr_t)lds_src) + (swizzle.row * 128 + swizzle.col));
            asm volatile("ds_read_b128 %0, %1\n"
                         : "=v"(dst.tiles[i].half[step])
                         : "v"(lds_addr)
                         : "memory");
        }
    }
}
```

In FlyDSL this is no longer necessary, we can simply issue loads from an LLVM pointer in LDS to register and have the compiler automatically insert appropriate waits based on the usage of those registers.

```python
# As in C++, we need to declare a type to store the results of our load.
# This is a 128 bit type that will hold 16 FP8 values, same as `i32x4` in C++.
I32x4_t = Vec.make_type(4, fx.Int32)

def pack_i32x4_i32x8(lo, hi):
    # Pack two i32x4 as one i32x8
    return lo.shuffle(hi, list(range(8)))

def load_i32x4(lds_ptr, offset):
    # Issue an llvm.load on an !llvm.ptr<3>, the AMDGPU backend
    # lowers a 16B vector load from addrspace 3 to ds_read_b128.
    base = fx.Int32(fx.ptrtoint(lds_ptr))
    addr = buffer_ops.create_llvm_ptr(base + fx.Int32(offset), address_space=3)
    return Vec(_llvm.load(I32x4_t, addr))

def load_rt(lds_src, wave_idx):
    # Load a 64x128 fragment of A/B from LDS to registers
    # Each 16x128 fragment requires 2 i32x4 (2 ds_read_b128)
    lane_id = fx.thread_idx.x % 64
    frag = []
    for i in range_constexpr(4):
        row = wave_idx * 64 + i * 16 + lane_id % 16
        halves = []
        for step in range_constexpr(2):
            col = (lane_id // 16) * 16 + step * 64
            row_swz, col_swz = swizzle_128(row, col)
            halves.append(load_i32x4(lds_src, row_swz * 128 + col_swz))
        frag.append(pack_i32x4_i32x8(halves[0], halves[1])) # i32x8
    return frag
```

### MFMA in FlyDSL

In the 4-wave interleave kernel, each wave handles a 128x128 tile of the output in a 2x2 configuration. We need 4 zero-initialized FP32 fragments, each one big enough
for a 64x64 sub-tile.

```python
# Each MFMA requires a thread to hold 4 FP32 values
CFrag_init = Vec.filled(4, 0.0, fx.Float32)
# Each value is a vector of 4 floats that holds a 16x16 fragment of the output, we need 16 of those in a 4x4 configuration for a 64x64.
c00_frag = [
    [CFrag_init for _ in range_constexpr(4)]
    for _ in range_constexpr(4)   
]
# Same for c01, c10 and c11
```

Then, we can use this fragment as part of our MFMA routine:

```python
# First we need to declare the data type of the accumulator
AccumDtype_t = Vec.make_type(4, fx.Float32)

def mfma_ABt_all(a, b, c):
    for i in range_constexpr(4):
        for j in range_constexpr(4):
            # 0x7F7F7F7F is the identity scale: 4 E8M0 values that translate to 1.0 (no scale)
            c[i][j] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(AccumDtype_t, [a[i], b[j], c[i][j], 0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F])
    return c
```

### Putting it all Together

Finally, we can put everything together and validate our kernel performance against HIP C++.

```python
# ...
# (inside a @flyc.kernel function)
#
# Setup: 
# - compute runtime values used throughout the kernel by all the routines (i.e. lane_id, wave_id...)
# - create buffer descriptors
# - initialize LDS, accumulators...

def wait_barrier(count):
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({count})\n\ts_barrier",
        constraints="",
        has_side_effects=True
    )

# Prologue: pre-load A/B cur
load_lds(A, A_lds[0][0]) # Each one of these issues 4 vmem ops
load_lds(B, B_lds[0][0])
load_lds(B, B_lds[0][1])
load_lds(A, A_lds[0][1])

# Issue load for next tile
load_lds(A, A_lds[1][0])
load_lds(B, B_lds[1][0])
load_lds(B, B_lds[1][1])
load_lds(A, A_lds[1][1])

# Wait just for the first tile of A to be ready
wait_barrier(28)

a0_frag = load_rt(A_lds[0][0])

# Wait for the first tile of B
wait_barrier(24)

b0_frag = load_rt(B_lds[0][0])

# Compute on cur, load on next. Swap after each iteration.
cur, next = 0, 1

# Main loop fully unrolled (equivalent to C++ #pragma unroll)
for k in range_constexpr(K_ITERS - 2):
    wait_barrier(16)

    c00_frag, b1_frag = interleaved_cluster(
        A_lds[cur][0],      # The LDS pointer where to store the next tile
        A,                  # Where to load the next LDS tile from
        B_lds[cur][1],      # The LDS pointer to load from for the next register fragment
        a0_frag, b0_frag,   # Current MFMA operands (registers)
        c00_frag            # Accumulator
    )

    c01_frag, a1_frag = interleaved_cluster(B_lds[cur][0], B, A_lds[cur][1], a0_frag, b1_frag, c01_frag) # c[0][1] += a[0] * b[1]

    wait_barrier(16)

    c10_frag, a0_frag = interleaved_cluster(B_lds[cur][1], B, A_lds[next][0], a1_frag, b0_frag, c10_frag) # c[1][0] += a[1] * b[0]

    c11_frag, b0_frag = interleaved_cluster(A_lds[cur][1], A, B_lds[next][0], a1_frag, b1_frag, c11_frag) # c[1][1] += a[1] * b[1]

    # Swap cur and next
    cur ^= 1
    next ^= 1

# EPILOGUE
# Last two k steps + store.
```

The full listing with the test and performance harness is available on [GitHub](https://github.com/nirw4nna/flydsl-gemm).

Note that `rocdl.s_waitcnt()` expects the raw [bitfield encoding](https://github.com/llvm/llvm-project/blob/90f71456ec5d0a1716716c8477eed0ab720c037b/mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp#L514), not the shorthand `vmcnt(N)` syntax used in assembly. If you prefer the familiar syntax, inline assembly is simpler.

## Results

Finally, we can benchmark our FlyDSL kernel against the hand-tuned HIP C++ reference from our previous blogpost. We used 1000 warm-up iterations followed by 1000 benchmark iterations with rotating buffers and normally distributed data, and report the average TFLOPS.

```{figure} ./images/tflops.png
:align: center
:alt: Figure 2: Performance Comparison
Figure 2: Performance Comparison
```

As you can see from figure 2, across all four matrix sizes the two implementations are within 1% of each other. At 8192 and 12288 FlyDSL even edges ahead slightly (3312 vs 3292 and 3444 vs 3362 TFLOPS respectively), while at 4096 and 16384 HIP C++ leads by a similarly narrow margin. These differences are well within run-to-run variance and confirm that moving to a higher-level language does not mean leaving performance on the table: FlyDSL generates the same quality of machine code while being significantly more concise and maintainable.

## Summary

In this post we introduced FlyDSL by porting two kernels of increasing complexity.

Starting with vector addition we covered the core building blocks: the
`@flyc.kernel` / `@flyc.jit` pair, buffer resource descriptors for global
memory access, and the built-in constant mapping (`fx.thread_idx`, etc.).

For the FP8 GEMM kernel we went deeper, covering:

- **LDS allocation** via `fx.SharedAllocator` and `range_constexpr` list comprehensions
- **Direct-to-LDS loads** via `rocdl.raw_ptr_buffer_load_lds`, a direct replacement for the HIP C++ intrinsic
- **Vectorized register loads** with the `Vector` class
- **MFMA** via `rocdl.mfma_scale_f32_16x16x128_f8f6f4`

Along the way we saw several ergonomic advantages over HIP C++: first-class support for
buffer resource descriptors to eliminate out-of-band guards,
`range_constexpr` replaces template unrolling, Python-level factory functions replace C++ templates
and the compiler inserts `s_waitcnt` instructions automatically rather than requiring explicit inline assembly for register loading.

Most importantly, FlyDSL matched or exceeded the hand-tuned HIP C++ kernel at every tested matrix size — confirming that higher-level doesn't have to mean slower.

FlyDSL has much more to offer — in particular, its CuTe-compatible layout algebra
can replace manual index arithmetic with formally verified tensor expressions,
making kernels more composable and portable. That will be the subject of a future post.

## Further Reading

For additional background and related work, see:

1. [FlyDSL Introduction](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html)
2. [Working with FlyDSL Nightlies](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-nightly-wheel/README.html)
3. [Deep Dive Into 4-Wave Interleave FP8 GEMM](https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html)

---

## Configuration Details

Data measured by AMD by the authors on June 23rd, 2026. All the experiments were conducted on the AMD Instinct™ MI355X platform. System configuration:

- GPU: MI355X-DLC 1.4KW
- CPU: AMD EPYC 9575F 64-Core Processor x 2
- ROCm Version: 7.2.2
- OS: Linux (Ubuntu 22.04)

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
