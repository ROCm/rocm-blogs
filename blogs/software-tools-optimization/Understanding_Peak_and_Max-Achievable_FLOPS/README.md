---
blogpost: true
blog_title: "Understanding Peak, Max-Achievable & Delivered FLOPs"
date: 14 Feb, 2025
author: Ben Sander
thumbnail: 'maf_bg.JPG'
tags: AI/ML, Hardware
target_audience: "AI Core Developer, HPC Core Developer"
key_value_propositions: ""
category: Software tools & optimizations
language: English
myst:
    html_meta:
            "author": "Ben Sander"
            "description lang=en": "Understanding Peak, Max-Achievable & Delivered FLOPs"
            "keywords": "AI, HPC, Optimization, MI300X"
            "amd_category": "Developer Resources"
            "amd_asset_type": "Blog"
            "amd_technical_blog_type": "Benchmarks and Testing"
            "amd_blog_hardware_platforms": "Instinct GPUs"
            "amd_blog_development_tools": "ROCm Software"
            "amd_blog_applications": "Design, Simulation & Modeling"
            "amd_blog_topic_categories": "Software & Ecosystem, AI & Intelligent Systems, HPC & Scientific Computing"
            "property=og:locale": "en_US"
---

# Understanding Peak, Max-Achievable & Delivered FLOPs, Part 1

The purpose of this blog post is to provide information on the
differences between Peak FLOPs and Max-achievable FLOPs. After reading,
users will know how AMD measures maximum delivered performance, and how
AMD recommends measured device performance is used.

Historically, terms such as peak FLOPs, max achievable FLOPs, and
delivered FLOPs have been used interchangeably, creating confusion and
incorrect comparisons. While all three concepts are useful, it is
important that they be used in a consistent and well-defined manner.

## Peak FLOPs

Peak FLOPs (or Peak Theoretical FLOPs) are a well-established metric
used in specification sheets to describe the maximum theoretical
performance capabilities of CPU and GPU devices. FLOPs are the
floating-point operations per second. The formula is simple and easy to
understand:

```{math}
Peak_{FLOPs} = Max\_Boost\_Clock\_Frequency \times Number\_Cores \times \frac{OPS}{Core \cdot Cycle}
```

Boost_Clock_Frequency<sup>[^1]</sup>: The maximum clock rate achievable
by the device. The Boost clock (or Max boost) is not a theoretical
metric - it can be achieved by the device (typically in a lightly loaded
scenario such as a single core or a low-power workload phase). Boost
clock is determined by hardware design characteristics such as the
transistor gates/clock, the power consumption, and the process
technology.

Number_Cores : For CPUs this is the number of processing cores, and for
GPUs the number of Compute Units (aka Streaming Multiprocessors).

OPS/Core/Cycle : The number of floating point operations per core that
can be processed in a single cycle. Peak FLOPs is traditionally focused
on matrix multiplication (aka \"GEMM\") and devices will have different
rates depending on the data type (with faster rates for ML-focused data
types such as F16 or FP8). For processors with a fused
multiply-accumulate (FMA) instruction, both the multiply and the
accumulate count as an \"op\" for the purposes of this metric.

Table 1 compares the Peak half precision (BF16) FLOPs of a few
current-generation CPU and GPU products:

| Product | Max Boost Clock Freq. | Number of Cores | OPS/Core/Cycle | BF16 Peak TFLOPs (without sparsity) |
|:---:|:---:|:---:|:---:|:---:|
| AMD EPYC 9965 | 3700 Mhz | 192 (CPU Cores) | 128 | 91 |
| AMD Instinct MI325X<sup>[MI325-008](https://www.amd.com/en/legal/claims/instinct.html#q=MI325-008&sortCriteria=%40title%20ascending)</sup> | 2100 Mhz | 304 (Compute Units) | 2048 | 1307 |
| NVIDIA H200<sup>[MI325-008](https://www.amd.com/en/legal/claims/instinct.html#q=MI325-008&sortCriteria=%40title%20ascending)</sup> | 1803 Mhz | 134 (Compute Units) | 4096 | 989  |

Table 1: Example Peak BF16 FLOPs (without sparsity) calculations for
modern CPUs and GPUs.
  
The AMD Instinct MI325X has a higher boost clock and more compute units
than the Nvidia H200, which enables the MI325X to drive high Peak FLOPs.
The AMD CPU has a high Boost clock (useful for single-thread workloads)
and a high number of cores, with a modest vectorized floating point
investment via the AVX512 ISA.

Both GPUs in the table have dedicated tensor core arrays which deliver
very high floating-point rates for ML data types -- more than 10X higher
"Ops_Per_Core" as compared to the EPYC 9965 CPU.

The Peak FLOPs provide a consistent cross-vendor comparison point based
on pure hardware metrics and can be computed without needing to run a
benchmark on the target device.

Additionally, the Boost clock is an important component of application
performance - the processor can boost to the highest clock during
lightly loaded situations and improve latency through critical-path
execution regions. In machine learning applications, this can improve
latency of small kernels, element-wise operations, and kernel sections
like address computations and setup code.

## Max-Achievable FLOPs

The \"Max-Achievable FLOPs\" (MAF) are the maximum achievable FLOPs
under realistic workload situations (e.g. normalized data
initialization, stable thermal conditions, cold caches) using a
device-specific optimal matrix size chosen to maximize architectural
efficiency. MAF has a specific definition and with care can be measured
on any target device -- we'll share our measurement methodology later in
part two of this discussion.

Understanding MAF is crucial since it removes the opaque hardware
components (e.g., the actual processing frequency) from the target
metric and allows developers to focus on factors that they can control.
For example, developers optimizing compute kernels (e,g. flash
attention), writing library routines (e.g. AMD hipBLASlt), or developing
compilers (e.g. Triton or MLIR) can use MAF to understand if the
performance they measure is close to the achievable hardware
capabilities. Essentially, MAF helps optimizers to know when they are
done optimizing.

One specific use case for MAF is in refactoring the classic "MFU"
metric:

```{math}
MFU = \frac{FLOPS used by the model}{Total Available HW FLOPs}
```

MFU is common metric used in model optimization to determine the current
software optimization level. Model optimizers often pursue model-level
optimizations (fusing kernels, overlapping communication, tuning
hyper-parameters, adjusting batch, etc.) until the model performance
approaches the MFU. Traditionally MFU has been computed using Peak FLOPs
in the denominator, but AMD recommends replacing this with MAF as this
is a more achievable target.

## History of Peak and Max-Achievable FLOPs

Peak FLOPs is a long-standing industry standard that has been used for
decades to compare CPU, GPU, and HPC products. Historically, the MAF
tracked the peak FLOPs relatively closely - often within a few percent -
and the two terms could be used nearly interchangeably.

![Peak and estimated max achievable FLOPS diverging over time](./images/image1.png)

Figure 1: Peak and MAF FLOPs Diverging over Time[^2]

However, with the rise of ML workloads, the silicon area and power
invested in matrix floating point multiplication has increased
significantly - evolving from packed vector FMAs to powerful tensorcore
arrays. For example, a modern GPU core now has more than 10X the
floating-point compared to earlier-generation GPUs.

| Product | OPS/Core/Cycle | Ratio | Architecture |
|:---:|:---:|:---:|:---:|
| P100 (Pascal) | 256 | 1.0 X | Vector FMA |
| V100 (Volta) | 1024 | 4.0 X | Tensor Core |
| A100 (Ampere) | 2048 | 8.0 X | Tensor Core |
| H100 (Hopper) | 4096 | 16.0 X | Tensor Core |

Table 2: Increasing silicon investment in matrix-multiplication over
product generations.

This dedicated silicon consumes a significant amount of power, and the
optimal operating point runs a wider set of logic at a clock frequency
which is lower than the Boost clock frequency. Modern GPUs have
sophisticated power management algorithms which detect the power
consumption of the device - raising the clock rate to the Boost clock
when possible and lowering it in the dense compute phases. These
transitions are handled automatically by the hardware and can occur
frequently (i.e. in micro-seconds) - even during the execution of a GPU
kernel.

In simpler times, the ratio of MAF:Peak could be used as a measure of
software optimization efficiency. However, on modern CPUs and GPUs, with
their specialized matrix hardware and diverse workload requirements
requiring sophisticated power and frequency management, this is no
longer true -- we estimate based on preliminary testing that the gap is
at 44-70% as shown on the right side of Figure 1. Both high Peak FLOPs
(with associated high Boost frequency) and high Max-achievable FLOPs are
desirable features for a modern processor.

## Delivered FLOPs

You may see the term "Delivered FLOPs" -- sometimes this can refer to
the MAF, or sometimes it refers to the GEMM performance inside the
application, etc. Recall MAF is a well-defined and measurable number: it
is the *maximum* achievable FLOPs when using an optimally chosen matrix
size under a specific environment. In an actual application, with less
optimal sizes and less controlled conditions, performance results may
differ from MAF. Some reasons the FLOPs seen during a GEMM inside an
application may differ from MAF include:

- Memory-bound kernels -- kernels which are skinny in at least one
    dimension will be bounded by the memory bandwidth and not the
    compute FLOPs.

- Small K dimension -- the K value determines the amount of time spent
    in the inner loop of the GEMM. If K is large, it amortizes the
    startup effects, and the performance approaches the MAF. If K is
    small -- these startup effects become significant. Each GPU kernel
    incurs a startup delay of a few micro-seconds -- the hardware has to
    set up the state for the new kernel, fetch the kernel arguments,
    compute addresses, and load the first pieces of data from memory.
    Additionally, some or all of the final epilogue and store to memory
    is exposed and cannot be overlapped.

- Unused Compute Units -- classic GPU matrix multiplication algorithms
    divide the output space into a number of tiles, and each tile is
    assigned to a compute unit. If the number of tiles does not divide
    evenly among the CUs, the GEMM will experience a tail phase in which
    only a subset of the machine is used. In the near future, AMD
    libraries plan to include a "STREAM-K" algorithm that provides
    better load-balancing for these GEMM shapes.

- Environmental effects -- ambient temperatures, part-to-part
    variability, quality of the cooling solution.

- Data initialization -- zeros or constant values consume less power
    than numbers with variation in exponents and toggle rates.

- Power management -- the power management may not run the GEMM at its
    optimal frequency.

- Software optimization -- there are a wide variety of different GEMM
    optimizations techniques. The MAF focuses on core compute kernel
    performance -- real sizes may require additional optimization
    attention in other parts of the kernel, or different algorithmic
    techniques.

- Selection heuristics - GEMMs have a number of different possible
    algorithms -- different tile sizes, unroll factors, load strategies,
    split-k, stream-k, etc. The hipBLASLt library is responsible for
    mapping each new set of M, N, K and other parameters to the best
    available solution in the library. Choosing the wrong solution may
    provide lower performance. In some cases, this performance can be
    recovered by applying a user-driven auto-tuning flow that searches
    and find the optimal algorithm from the "menu" provided by the
    library.

## Summary and Next Steps

Hopefully this article helps to understand the history of FLOPs
measurement, and why both peak and MAF are important metrics as we move forward
into the age of ML-optimized silicon. For most cases, AMD recommends
using MAF as a target for kernel and model optimization - as this metric
gives a more realistic target for the optimization knobs that are under
the developers control.

Max-Achievable FLOPs are important but just one component of the overall
workload performance. Other key factors that influence performance
include memory bandwidth, communication latency and bandwidth, and
memory capacity. Machine learning workloads exhibit different
sensitivities to these factors:

- The \"prefill\" or \"prompt\" phase of generative AI inference tends
to have large matrices and be sensitive to MAF performance. The
summarization use case (e.g. summarize a long document or email) is
dominated by the prefill phase - by definition the summarization
process boils down a large number of input tokens to a small number
of output tokens.

- On the other hand \"decode\" or token-generation phase is dominated
by an auto-regressive algorithm which is sensitive to memory
bandwidth. The chat use case - where a user provides a relatively
short prompt and expects a large number of output tokens - would
have a short \"prefill\" phase followed by a relatively large
\"token\" phase. Some recent reasoning models generate an even
larger number of "thinking" tokens which further emphasize the
importance of this phase.

- Parallelizing a workload across multiple GPUs will sensitize the
communication performance, but in some cases a GPU with high memory
capacity can fit the model and KV cache states in a single GPU or
single node and substantially reduce or even eliminate the
communication. Capacity can also provide spectacular throughput
gains by enabling higher batch sizes.

For more information on workload performance, see the recent blogs on
[Best practices for competitive inference optimization on AMD Instinct™ MI300X](https://rocm.blogs.amd.com/artificial-intelligence/LLM_Inference/README.html)
and [Enhancing AI Training with AMD ROCm Software](https://rocm.blogs.amd.com/artificial-intelligence/training_rocm_pt/README.html).

[^1]: Max boost for AMD EPYC processors - EPYC-018 ; Max boost clock frequency for AMD Instinct™ accelerators represent
    the maximum frequency on the GPU under bursty workloads. Actual
    boost clock speeds may vary based on multiple factors, including but
    not limited to: thermal conditions, system cooling efficiency, power
    limits, workload characteristics, board design, firmware settings,
    the latest AMD ROCm™ driver, and system software updates. GD-249

[^2]: Peak Theoretical FLOPs

    Published specifications on the AMD Instinct™ MI250 Accelerator:
    632.1 TFLOPs peak theoretical half precision (FP16, non-sparse)
    floating-point performance.[Spec sheet](https://www.amd.com/en/products/accelerators/instinct/mi200/mi250.html)

    Published specifications on the AMD Instinct™ MI300X Accelerator:
    1307.4 TFLOPs peak theoretical half peak theoretical half precision
    floating-point performance.[Spec sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)

    Published specifications on the AMD Instinct™ MI325X Accelerator:
    1307.4 TFLOPs peak theoretical half precision (FP16, non-sparse)
    floating-point performance.[Spec sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf)

    Published specifications on the NVIDIA Tesla P100 PCIe accelerator:
    18.7 TFLOPs peak theoretical half precision tensor (FP16, non-sparse
    Tensor) floating point performance.[Spec sheet](https://images.nvidia.com/content/tesla/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf)

    Published specifications on the NVIDIA Tesla V100 PCIe accelerator:
    112 TFLOPs peak theoretical half precision tensor (FP16, non-sparse
    Tensor) floating point performance.[Spec sheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf)

    Published specifications on the NVIDIA Ampere A100 (80GB) GPU
    accelerator: 312 TFLOPs peak theoretical half precision tensor
    (FP16, non-sparse Tensor) floating point performance.[Spec sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf)

    Published specifications on Nvidia H100 SXM (80GB) GPU:   989.5
    TFLOPs peak theoretical half precision tensor (FP16, non-sparse
    Tensor) floating point performance. AMD converted these numbers to
    non-sparsity/dense by dividing by 2, and this number appears above.[Spec sheet](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet)

    Published specifications on Nvidia H200 SXM (141GB) GPU:   989.5
    TFLOPs peak theoretical half precision tensor (FP16, non-sparse
    Tensor) floating point performance. AMD converted these numbers to
    non-sparsity/dense by dividing by 2, and this number appears above.[Spec sheet](https://resources.nvidia.com/en-us-data-center-overview-mc/en-us-data-center-overview/hpc-datasheet-sc23-h200)
