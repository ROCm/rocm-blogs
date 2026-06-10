---
blogpost: true
blog_title: "Quickly Developing Powerful Flash Attention Using TileLang on AMD Instinct MI300X GPU"
date: 20 Jan 2026
author: 'Daniel Huang, George Wang'
thumbnail: 'rocm-tilelang-kernel.png'
tags: AI/ML, Hardware
category: Ecosystems and Partners
target_audience: The GPU kernel developers want to author high-peformance LLM kernels for AMD GPUs
key_value_propositions: This article introduces the new and efficient kernel DSL on AMD Instinct GPUs
language: English
myst:
    html_meta:
        "author": "Daniel Huang, George Wang"
        "description lang=en": "Learn how to leverage TileLang to develop your own kernel. Explore the power to fully utilize AMD GPUs"
        "keywords": "ROCm, kernel DSL, TileLang"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Ecosystem and Partners"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Daniel Huang, George Wang"
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

# Quickly Developing Powerful Flash Attention Using TileLang on AMD Instinct MI300X GPU

Against the backdrop of the rapid development of the AMD ROCm™ software ecosystem, the high barrier to operator development has long been a bottleneck. The emergence of TileLang provides developers with an efficient solution. As an emerging AI operator development framework, tilelang encapsulates low-level GPU details with concise syntax, enabling developers to fully tap into the computing potential of AMD GPUs without requiring in-depth knowledge of low-level languages such as HIP. The AMD Instinct™ MI300X GPU, as a flagship GPU for AI workloads, boasts ultra-high bandwidth memory and powerful compute units, but it requires adaptive high-performance operators to unleash its capabilities. In this blog, we will take Flash Attention, a key kernel in both LLM training and inference, as an example to fully demonstrate the development process based on TileLang on the MI300X, highlighting the dual benefits of efficiency and performance that TileLang brings to AMD operator development.

## Understanding TileLang: A GPU-Friendly Kernel Development Framework

### Core Positioning of TileLang

TileLang is an open-source AI operator programming domain-specific language. The core goal is to simplify the development process of complex GPU operators while achieving performance comparable to handwritten low-level code. It takes "Tile" as the core programming unit and encapsulates GPU low-level details through high-level APIs. This allows developers to efficiently utilize hardware resources such as GPU shared memory and registers without mastering low-level languages like HIP/CUDA. The DeepSeek team open-sourced TileLang-based operators in its V3.2-Exp model and recommends them for rapid iteration and debugging in research experiments, which fully confirms its industrial applicability.

### Core Advantages Over Triton

Although OpenAI's Triton pioneered high-level kernel programming, it still has limitations in AMD ecosystem adaptation and performance tuning. TileLang offers significant advantages in addressing these pain points:

- **Higher Development Efficiency**: Through tile-level abstraction and built-in optimization primitives, TileLang significantly reduces code volume. According to data from the TileLang team, the code length of its Flash Attention kernel implementation is reduced from over 500 lines in CUDA to fewer than 80 lines while maintaining equivalent performance. Compared with Triton, TileLang's APIs are more aligned with kernel optimization logic, reducing redundant syntax overhead.

- **Smarter Autotuning**: TileLang includes a flexible autotuning framework that supports multi-dimensional parameter combination search. This enables rapid matching of optimal configurations for different hardware and workload scenarios, eliminating the tedious manual tuning process for developers.

- **Excellent Ecosystem Compatibility**: It has completed adaptation with domestic and mainstream GPUs. Meanwhile, it has been adopted by mainstream large model projects like DeepSeek, showing rapid growth in ecosystem maturity.

- **Strong User-Friendliness and Scalability**: It supports kernel development for developers at all proficiency levels, whether they are beginners, experienced developers, or domain experts. Each group can find a suitable development approach. In contrast, Triton has a relatively high threshold for beginners. TileLang, through its layered API design, allows beginners to quickly get started with built-in primitives, while experts can dive into low-level customization and optimization, balancing usability and scalability.

## Flash Attention Introduction

### Core Pain Points Addressed

The computational complexity of traditional attention mechanisms grows quadratically with sequence length. Additionally, frequent reading and writing of large-scale intermediate results (such as attention score matrices) make GPU memory volume and bandwidth a bottleneck. Flash Attention employs tiled computation and recomputation techniques to shift memory access from HBM (High Bandwidth Memory) to GPU SRAM. This greatly reduces memory read/write volume and improves the utilization of computing resources through optimized parallel strategies, achieving a 2-4x performance improvement.

### Core Formulas and Computational Process

The core computational process of Flash Attention is consistent with traditional attention, but its computation order is restructured to adapt to the tiling mechanism. The core formulas are as follows:

1. **Attention Score Calculation**: Perform matrix multiplication between queries (Q) and the transpose of keys (K), then normalize using the square root of the dimension to avoid softmax saturation caused by excessively large values.

$$S = \frac{QK^T}{\sqrt{d}}$$

Here, Q ∈ ℝ<sup>B×L×H×D</sup> (where B = batch size, L = sequence length, H = number of attention heads, D = feature dimension), K ∈ R, and d = feature dimension D.

2. **Causal Masking (Optional)**: In generation tasks, mask future position information using a lower triangular mask to ensure causal consistency:

$$S_{i,j} = \begin{cases} S_{i,j} & i \geq j \\ -\infty & i < j \end{cases}$$

3. **Softmax Normalization**: Normalize the attention scores to obtain attention weights:

$$A = \text{softmax}(S)$$

4. **Output Calculation**: Perform matrix multiplication between attention weights and values (V) to get the final output:

$$O = AV$$

Here, V ∈ R and O ∈ R.

The core innovation of Flash Attention lies in decomposing the above computations into multiple small tiles. Through a "load-compute-update" pipeline operation, most computations are completed in SRAM, and only the final results are written to HBM, thereby breaking through the memory bandwidth bottleneck.

## Flash Attention Implementation with TileLang: In-Depth Code Analysis

Based on the provided source code, the following section analyzes the core logic of Flash Attention implemented with TileLang, organized by functional module. The overall code is divided into two parts: core operator functions and auxiliary functions. The core operator realizes tiled computation and hardware optimization through TileLang's high-level APIs.

### Core Operator: Analysis of the Main Flash Attention Function

As the core of the Flash Attention implementation with TileLang, this function realizes autotuning and just-in-time compilation through the `@tilelang.autotune` and `@tilelang.jit` decorators. It is mainly divided into two parts: parameter initialization and kernel definition.

#### Decorators and Parameter Definition

```Python
@tilelang.autotune(configs=get_configs(), cache_input_tensors=True, supply_prog=supply_tensors_gpu)
@tilelang.jit(out_idx=[3])
def fast_flashattn(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    groups,
    block_M: int,
    block_N: int,
    num_split_q: int,
    threads: int,
    num_stages: int,
    enable_rasterization: bool,
    k_pack: int,
    panel_size: int,
    qk_coalesced_width: int,
    v_coalesced_width: int,
):
```

- `@tilelang.autotune`: Enables autotuning, specifying the candidate configuration set, tensor caching strategy, and GPU tensor supply function.

- `@tilelang.jit`: Enables just-in-time compilation to convert TileLang code into GPU-executable HIP kernels. out_idx=[3] specifies that the 4th parameter (Output) is the output tensor.

#### Initialization and Kernel Entry

```Python
scale = (1.0 / dim)**0.5
head_kv = heads // groups
q_shape = [batch, seq_len, heads, dim]
kv_shape = [batch, seq_len, head_kv, dim]
dtype = "float16"
accum_dtype = "float"

@T.prim_func
def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
):
    with T.Kernel(num_split_q, batch * heads, threads=threads) as (b_split, byz_combined):
        T.use_swizzle(panel_size, enable=enable_rasterization)

        bz = byz_combined // heads
        by = byz_combined % heads
```

Core Logic:

1. Initialization: Calculate the normalization factor (scale) and the number of KV heads (head_kv), and define tensor shapes and data types (float16 for computation to improve performance, float for accumulation to ensure precision).

2. Kernel Definition: Define the core computation function with `@T.prim_func`.`T.Kernel` specifies parallel dimensions—num_split_q (parallel splitting of Q) and batch*heads (combined parallelism of batch and heads), while threads specify the number of threads per thread block.

3. Hardware Optimization: `T.use_swizzle` enables memory reordering optimization to improve the memory access efficiency of the MI300X. Decompose byz_combined into batch index (bz) and attention head index (by) to realize parallel processing of batches and heads.

#### Q Tile Processing and Cache Initialization

```Python
num_q_blocks = T.ceildiv(seq_len, block_M)
bx = T.alloc_var("int32")
bx = b_split
with T.While(bx < num_q_blocks):
    # Initialize accumulators and numerical stability variables
    acc_o = T.alloc_fragment([block_M, dim], accum_dtype)  # Output accumulator
    m_i = T.alloc_fragment([block_M], accum_dtype)        # Row-wise maximum (for numerical stability)
    l_i = T.alloc_fragment([block_M], accum_dtype)        # Row-wise sum (for numerical stability)
    T.fill(acc_o, 0)
    T.fill(m_i, -T.infinity(accum_dtype))
    T.fill(l_i, 0)

    q_block_offset = bx * block_M
    # Allocate shared memory and registers
    Q_shared = T.alloc_shared([block_M, dim], dtype)      # Q shared memory cache
    K_shared = T.alloc_shared([block_N, dim], dtype)      # K shared memory cache
    V_shared = T.alloc_shared([block_N, dim], dtype)      # V shared memory cache
    acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)  # Register cache (reduces LDS usage)

    # Load Q tile to shared memory
    T.copy(
        Q[bz, q_block_offset:q_block_offset + block_M, by, :],
        Q_shared,
        coalesced_width=qk_coalesced_width)
```

Core Logic:

1. Q Tile Traversal: Calculate the number of Q tiles (num_q_blocks) and traverse each Q tile via a While loop.

2. Numerical Stability Initialization: Create acc_o (output accumulator), m_i (row-wise maximum), and l_i (row-wise sum) to solve the numerical overflow problem in softmax computation.

3. Memory Allocation: Allocate shared memory (Q_shared/K_shared/V_shared) to cache tile data and allocate registers (acc_s_cast) to cache intermediate results, reducing shared memory (LDS) usage.

4. Q Tile Loading: Load the current Q tile from HBM to shared memory. coalesced_width specifies the memory coalescing width to improve the memory bandwidth utilization of the MI300X.

#### K/V Tile Traversal and Core Computation

```Python
loop_end_k = T.ceildiv(q_block_offset + block_M,
                       block_N) if is_causal else T.ceildiv(seq_len, block_N)

for k in T.Pipelined(loop_end_k, num_stages=num_stages):
    kv_idx = k * block_N

    # Load K/V tiles to shared memory
    T.copy(
        K[bz, kv_idx:kv_idx + block_N, by // groups, :],
        K_shared,
        coalesced_width=qk_coalesced_width)
    T.copy(
        V[bz, kv_idx:kv_idx + block_N, by // groups, :],
        V_shared,
        coalesced_width=v_coalesced_width)

    # Causal masking initialization
    if is_causal:
        for i, j in T.Parallel(block_M, block_N):
            acc_s[i, j] = T.if_then_else(q_block_offset + i >= kv_idx + j, 0,
                                         -T.infinity(acc_s.dtype))
    else:
        T.clear(acc_s)

    # QK^T matrix multiplication (attention score calculation)
    T.gemm(
        Q_shared,
        K_shared,
        acc_s,
        transpose_B=True,
        k_pack=k_pack,
        policy=GemmWarpPolicy.FullRow,
    )

    # Numerical stability processing (stepwise softmax computation)
    T.copy(m_i, m_prev)
    T.reduce_max(acc_s, m_i, dim=1, clear=False)  # Update row-wise maximum
    for i in T.Parallel(block_M):
        m_i[i] = T.max(m_i[i], m_prev[i])
        sf = T.exp(m_prev[i] * scale - m_i[i] * scale)
        l_i[i] *= sf  # Update row sum (scaling)
        scale_factor[i] = sf

    # Output accumulator scaling
    for i, j in T.Parallel(block_M, dim):
        acc_o[i, j] *= scale_factor[i]

    # Softmax normalization
    for i, j in T.Parallel(block_M, block_N):
        acc_s[i, j] = T.exp(acc_s[i, j] * scale - m_i[i] * scale)
    T.reduce_sum(acc_s, row_sum, dim=1)  # Calculate row-wise sum
    for i in T.Parallel(block_M):
        l_i[i] += row_sum[i]

    # Attention weight and V multiply-accumulate (output calculation)
    T.copy(acc_s, acc_s_cast)
    T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.FullRow)
```

This is the core logic of Flash Attention tiled computation, which realizes efficient "load-compute-update" pipeline by traversing K/V tiles:

1. K/V Tile Range: Determine the traversal end of K/V tiles based on whether causality is enabled. In causal mode, only K/V tiles before the current Q position are traversed to avoid future information leakage.

2. K/V Loading: Load K/V tiles to shared memory, adopting the same memory coalescing strategy as Q.

3. Causal Masking: Assign values to acc_s (score cache) via parallel loops, setting non-compliant positions to -∞ to implement causal constraints.

4. GEMM Computation: Call TileLang's built-in GEMM primitive to compute QK. transpose_B=True specifies transposing K, k_pack optimizes data packing, and GemmWarpPolicy.FullRow adapts to the MI300X's warp scheduling strategy to improve parallel efficiency.

5. Numerical Stability Processing: Compute softmax in steps (first find maximum value, then scale, finally compute exp) to avoid numerical overflow caused by direct exp computation.

6. OV Multiply-Accumulate: Perform GEMM computation between normalized attention weights (acc_s) and V and accumulate the results into acc_o to realize tile result fusion.

#### Final Output and Q Tile Iteration

```Python
# Final normalization (reciprocal of row sum)
l_inv = T.alloc_fragment([block_M], accum_dtype)
for i in T.Parallel(block_M):
    safe_l = T.if_then_else(l_i[i] > 1e-6, l_i[i], 1.0)  # Avoid division by zero
    l_inv[i] = 1.0 / safe_l
# Write output results to HBM
for i, j in T.Parallel(block_M, dim):
    Output[bz, q_block_offset + i, by, j] = acc_o[i, j] * l_inv[i]
# Iterate to next Q tile
bx = current_bx + num_split_q
```

Core Logic: Calculate the reciprocal of the row sum (l_inv), perform final normalization on the accumulator acc_o, and write the results to the output tensor (Output). After processing the current Q tile, update bx to enter the loop for the next Q tile.

### Auxiliary Functions: Supporting Modules

#### Allocate Tensor on GPU

This function is an auxiliary interface for the TileLang autotuning framework. It ensures that input tensors are created on AMD GPUs (ROCm/HIP environment) to avoid computation errors caused by device inconsistency. Its core logic is to iterate over input parameters: for tensors with defined shape and dtype, it forces the generation of random tensors on the "cuda" device (compatible in ROCm environment); non-tensor parameters are returned directly.

#### Reference Implementation

This is a standard attention computation implementation based on PyTorch, used for correctness verification and performance comparison with the TileLang-implemented operator. The core steps include:

1. Parameter Validation: Ensure the number of heads in Q matches that in K/V and the number of groups.
2. KV Head Expansion: Repeat the head dimension of K/V according to the number of groups to align with Q.
3. Score Calculation and Normalization: Implement QK computation via einsum and normalize using the square root of the dimension.
4. Causal Masking: If causality is enabled (is_causal), generate a lower triangular mask to block future positions.
5. Softmax and Output Calculation: Normalize the scores and perform multiply-accumulate with V to get the output.

#### Autotuning Configuration Generation

This function generates a candidate configuration set for the TileLang autotuning tool, with parameter dimensions designed for the parallel characteristics of Flash Attention V2, including:

- Tile Size (block_M/block_N): Tiling dimensions for Q and K/V, which determine SRAM utilization efficiency.
- Number of Threads (threads): Number of threads per GPU thread block, matching the MI300X's computing core architecture.
- Parallel Splitting (num_split_q): Number of parallel splits for the Q sequence, improving multi-unit GPU utilization.
- Pipeline Stages (num_stages): Number of stages in the computation pipeline, optimizing data prefetching and computation overlap.
- Memory Coalescing Width (qk_coalesced_width/v_coalesced_width): Memory access coalescing width for QK and V, improving memory bandwidth utilization.

It generates 108 candidate configurations by traversing all parameter combinations via itertools.product, which are used by the autotuning tool to select the optimal solution.

### Main Function: Performance Testing and Verification

The main function implements operator autotuning triggering, correctness verification, and performance testing:

1. Computation Volume Statistics: Calculate the total number of floating-point operations (FLOPs) based on parameters such as batch size and sequence length.

2. Autotuning: Call fast_flashattn to trigger autotuning and search for the optimal configuration.

3. Correctness Verification: Compare the results of the TileLang implementation with the PyTorch reference implementation via profiler.assert_allclose to ensure the precision error is within an acceptable range (rtol=0.01, atol=0.01).

4. Performance Testing: Test the latency and computing power of both implementations via the do_bench function, with warmup=100 to ensure the GPU enters a stable state.

## Performance Comparison: Highlighting the Advantages of TileLang Implementation

On the AMD MI300X GPU, performance test results for the typical scenario (batch=1, heads=8, seq_len=4096, dim=128) are shown in table 1:

Table 1: Performance results for batch=1, heads=8, seq_len=4096, dim=128

|Implementation|Latency (ms)|Performance Improvement|Core Parameters of Optimal Configuration|
|---|---|---|---|
|PyTorch Reference (ref_program)|0.97|——|None (Native Implementation)|
|TileLang Implementation (fast_flashattn)|0.36|2.69x|block_M=128, block_N=32, threads=512, enable_rasterization=True|
|Triton Implementation|0.55|1.53x|block_M=128, block_N=32, threads=512|

Key Conclusions:

- **Significant Latency Reduction**: The latency of the TileLang implementation is only 37.1% of that of the native PyTorch implementation and 65.5% of that of the Triton implementation, achieving a performance improvement of nearly 2.7x over PyTorch and 1.53x over Triton [1]. This fully demonstrates the effectiveness of tiled computation and hardware optimization tailored for AMD GPUs.
- **Efficient Autotuning**: The search for 108 configurations takes only about 1 second. The optimal configuration achieves a balance between SRAM utilization and parallel efficiency through a tile size of 128×32 and 512 threads.
- **Reliable Precision**: The results are fully aligned with the PyTorch reference implementation, verifying the correctness of the TileLang implementation.

## Summary

Through the complete case of implementing Flash Attention on the AMD Instinct MI300X GPU with TileLang, this blog verifies the core value of TileLang as a high-level operator development framework—it achieves performance comparable to handwritten low-level code with concise code while significantly lowering the development threshold for AMD GPU kernels. Compared with the native PyTorch implementation, the TileLang version achieves a 2.7x performance improvement through tiled computation, memory optimization, and autotuning, with more concise code and stronger maintainability.

In the future, with in-depth adaptation of TileLang in the AMD ecosystem (such as supporting Tensor Core acceleration of the AMD Instinct MI300X GPUs) and optimization of autotuning algorithms, it will have broader application prospects in the development of core operators for large models. For developers, TileLang provides a new operator development paradigm that "enables efficient utilization of hardware performance without in-depth knowledge of hardware details", facilitating the rapid deployment and performance release of large models on AMD GPUs.

## Endnotes

[1] Test Environment

Hardware:

- AMD Instinct MI300X GPU
- Intel(R) Xeon(R) Platinum 8568Y+.

Software:

- ROCm v7.0.1
- Pytorch v2.9.0
- Triton v3.0.0
- TileLang v0.1.7.

Input configuration:

- batch_size = 1
- head_nums = 8
- seq_len = 4096
- dim = 128

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
