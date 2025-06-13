---
blogpost: true
blog_title: "Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X"
date: 21 Mar 2025
author: 'Peng Sun, Andy Luo, Seungrok Jung, Liz Li, Hai Xiao'
thumbnail: 'deep-seek-part2_thumbnail.png'
tags: AI/ML
category: Software tools & optimizations
target_audience: AI Developers and Enthusiast
key_value_propositions: deepseek r2 performance gain vs nvidia h200
language: English
myst:
    html_meta:
        "author": "Peng Sun, Andy Luo, Seungrok Jung, Liz Li"
        "description lang=en": "Learn how to optimize DeepSeek-R1 on AMD MI300X with SGLang, AITER kernels and hyperparameter tuning for up to 5× throughput and 60% lower latency over Nvidia H200"
        "keywords": "Deepseek, AMD Instinct MI 300X, Inference, SGLang, Optimization, AITER, Chunked Prefill size"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_developer_type": "ML/AI Developer"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_hardware_platforms": 'Instinct GPUs'
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "Generative AI, AI Inference"
        "amd_blog_topic_categories": 'Software & Ecosystem'
        "amd_blog_releasedate": Fri Mar 21, 12:00:00 PST 2025
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

# Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X

Our previous [blog post on this topic](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html)
discussed how DeepSeek-R1 achieves competitive performance on AMD
Instinct™ MI300X GPUs. We also included performance comparisons against
Nvidia H200 GPUs and a short demo application illustrating real-world
usage. In this blog we will delve into how using the SGLang framework,
critical kernel optimizations like AI Tensor Engine for ROCm™, and
hyperparameter tuning helps to achieve performance boosts.

## At a Glance

- **By using the latest SGLang framework, compared to Nvidia H200,MI300X achieves**
      - 2X--5X higher throughput at the same latency[^1]
      - Up to 75% higher throughput and 60% lower latency for same concurrency [^1].
- **AI Tensor Engine for ROCm** **software (AITER)** kernels are optimized providing +2X **GEMM** [^2], +3X **MoE** [^3], +17x **MLA decode**[^4], +14X **MHA prefill**[^5]
- **SGLang** serving **hyperparameter tuning**, thanks to the larger memory capacity of MI300X, boosted throughput at large concurrency

## Key Takeaways from DeepSeek-R1 Serving Benchmark

MI300X GPU demonstrates significantly better performance with SGLang
across the board regarding total throughput vs. end-to-end latency with
various optimization techniques. Figure 1 below shows that using SGLang
framework and key optimization techniques, MI300X achieved up to 5X
higher throughput at similar latencies vs NVIDIA H200.

```{figure} ./images/FIGURE1.png
:align: center
:alt: Scaling performance
Figure 1. DeepSeek R1 Total Throughput (tks) vs. Latency (ms)[^1]
```

As shown in Figure 2, H200 GPUs can serve up to 16 concurrent requests
with inter-token latency (ITL) below 50ms in a single node. We
benchmarked the performance of SGLang on the Nvidia H200 GPUs, using
SGLang 0.4.4.post1 and flashinfer MLA library. A single MI300X node,
which consists of 8 GPUs, can serve up to 128 concurrent requests
while maintaining inter-token latency (ITL) below 50ms, which shows
MI300X has higher user capacity to complete response without violating
user experience.

Note: Setting the chunked prefill size parameter to 131,072 enables
single batching of input sequences, but this may lead to out-of-memory
(OOM) errors for very large input sequences. Reducing the chunked
prefill size allows for batched prefill cache computation, making better
use of the model's total context length budget. However, this comes at
the cost of increased decode latency, as the input must be processed in
smaller batches and retrieved during decoding. Readers are encouraged to
optimize this parameter for their specific use-case.

```{figure} ./images/FIGURE2.png
:align: center
:alt: Scaling performance
Figure 2. DeepSeek R1 Higher concurrency under 50 ms Inter Token Latency limit [^1]
```

## Key Optimizations

Optimization techniques enable developers to significantly improve the
performance of applications running on GPUs, leveraging the full
potential of parallel processing and removing memory bottlenecks. The
AMD AI Tensor Engine for ROCm (AITER) is a centralized repository filled
with high-performance AI operators designed to accelerate various AI
workloads.

- **AI Tensor Engine for ROCm (AITER)**

    AITER is a brand-new high-performance open-source AI operator library.
    It provides Python and C++ APIs that can be easily integrated into
    SGlang, vLLM, and other custom frameworks. The following kernels for
    DeepSeek V3/R1 have been optimized in AITER to achieve significant
    uplift on the MI300X GPUs so that users can experienced significant
    performance boost using this library.
        - AITER block-scale GEMM (up to 2X boost)
        - AITER block-scale fused MoE (up to 3X boost)
        - AITER MLA for decode (up to 17X boost)
        - AITER MHA for prefill (up to 14X boost)

- **Hyperparameter Tuning**

    When running programs with a high number of threads (e.g., 128 or more),
    the system faces a bottleneck due to slow prefill throughput. We found
    that using a higher value of
    [chunked_prefill_size](https://docs.SGLang.ai/backend/server_arguments.html)
    can accelerate the prefill phase with the cost of more VRAM consumption
    as shown in Figure 3.

```{figure} ./images/FIGURE3.png
:align: center
:alt: Scaling performance
Figure 3. DeepSeek R1 Higher Total Throughput (tks) with Hyperparameter tuning on SGLang [^1]
```

## How to Reproduce the Benchmark

Now let's reproduce the same performance boost on your system and apply
the same techniques to your application for optimal performance on the
MI300X GPUs.

The following instructions assume that the user already downloaded a
model.

Note: The image provided for replicating the MI300X benchmark is a
pre-upstream staging version. The optimizations and performance
enhancements in this release are expected to be included in the upcoming
[lmsysorg](https://hub.docker.com/r/lmsysorg/sglang/tags) upstream
production release.

### AMD Instinct MI300X GPU with SGLang

1. Set relevant environment variables and launch the AMD SGLang container.

```shell
docker pull rocm/sgl-dev:upstream_20250312_v1
export MODEL_DIR=<DeepSeek-R1 saved_path>
docker run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --shm-size 32G \
    --cap-add=CAP_SYS_ADMIN \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --security-opt apparmor=unconfined \
    -v $MODEL_DIR:/model \
    rocm/sgl-dev:upstream_20250312_v1
```

2. Start the SGLang server.

```shell
python3 -m sglang.launch_server \
    --model /model \
    --tp 8 \
    --trust-remote-code \
    --chunked-prefill-size 131072 \
    --enable-torch-compile \
    --torch-compile-max-bs 256 &
```

3. Run the SGLang benchmark serving script for the user defined concurrency values and desired parameters.

```shell
# Run after “The server is fired up and ready to roll!”
concurrency_values=(128 64 32 16 8 4 2 1)
for concurrency in "${concurrency_values[@]}"; do
python3 -m sglang.bench_serving \
    --dataset-name random \
    --random-range-ratio 1 \
    --num-prompt 500 \
    --random-input 3200 \
    --random-output 800 \
    --max-concurrency "${concurrency}"
done
```

Note: Using the torch compile flags will result in longer server launch time

### NVIDIA H200 GPU with SGLang

1. Set relevant environment variables and launch the NVIDIA SGLang container.

```shell
docker pull lmsysorg/sglang:v0.4.4.post1-cu125
export MODEL_DIR=<DeepSeek-R1 saved_path>
docker run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --shm-size 32G \
    --gpus all \
    -v $MODEL_DIR:/model \
    lmsysorg/sglang:v0.4.4.post1-cu125
```

2. Start the SGLang server.

```shell
export SGL_ENABLE_JIT_DEEPGEMM=1
python3 -m sglang.launch_server \
    --model /model \
    --trust-remote-code \
    --tp 8 \
    --mem-fraction-static 0.9 \
    --enable-torch-compile \
    --torch-compile-max-bs 256 \
    --chunked-prefill-size 131072 \
    --enable-flashinfer-mla &
```

3. Run the SGLang benchmark serving script for the user defined concurrency values and desired parameters.

```shell
# Run after “The server is fired up and ready to roll!”
concurrency_values=(128 64 32 16 8 4 2 1)
for concurrency in "${concurrency_values[@]}"; do
python3 -m sglang.bench_serving \
    --dataset-name random \
    --random-range-ratio 1 \
    --num-prompt 500 \
    --random-input 3200 \
    --random-output 800 \
    --max-concurrency "${concurrency}"
done
```

## Summary

This blog showed you how to achieves breakthrough inference performance using DeepSeek-R1 on AMD Instinct™ MI300X GPUs by leveraging the SGLang framework, AMD’s AI Tensor Engine for ROCm (AITER), and targeted hyperparameter tuning. We demonstrated how MI300X outperforms Nvidia’s H200 with up to 5× higher throughput and significantly lower latency, especially under high concurrency workloads. We showed how core kernel optimizations—including GEMM, MoE, MLA decode, and MHA prefill—delivered major performance boosts, while tuning parameters like chunked_prefill_size enhanced throughput scaling. We also provided step-by-step benchmarking instructions to help you replicate our findings, showcasing MI300X’s strength for serving large-scale machine learning models efficiently

More optimizations are coming soon in future AMD ROCm software releases.
We expect further performance boost, including but not limited to:

- Expert Parallelism (EP)

- Prefill and decode disaggregation

- Speculative decoding

## Try Today

Download the [prebuilt SGLang docker
container](https://hub.docker.com/layers/lmsysorg/sglang/v0.4.2.post3-rocm630/images/sha256-f81946dac80889123dd91586832cb0e5d6d3530780e6577cb353a8c3e8b5c288)
from Docker hub and follow the instructions here to get started on the
MI300X GPUs:
<https://github.com/sgl-project/SGLang/blob/main/docs/references/amd.md>.

## Additional Resources

- [ROCm AI Developer Hub](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai.html)
- [SGLang Docker Images](https://hub.docker.com/r/lmsysorg/sglang/tags)
- SGLang Docker files for production deployment: [Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile.rocm)
- GitHub For AITER: <https://github.com/ROCm/aiter>.
- [AITER Blog](https://rocm.blogs.amd.com/software-tools-optimization/aiter:-ai-tensor-engine-for-rocm™/README.html)

[^1]: On average, a system configured with an AMD Instinct™ MI300X GPU running  DeepSeek R1 with SGLang offers 2X–5X higher throughput at the same latency, 75% better throughput and 60% lower latency for same batch size than Nvidia HGX H200. Testing done by AMD on 03/13/2025, results may vary based on configuration, usage, software version, and optimizations.

    **SYSTEM CONFIGURATION:**  \
    AMD Instinct ™ MI300X platform \
    System Model: Supermicro AS-8125GS-TNMR2 \
    CPU: 2x AMD EPYC 9654 96-Core Processor \
    NUMA: 2 NUMA node per socket. NUMA auto-balancing disabled/
    Memory: 2304 GiB (24 DIMMs x 96 GiB Micron Technology MTC40F204WS1RC48BB1 DDR5 4800 MT/s) \
    Disk: 16,092 GiB (4x SAMSUNG MZQL23T8HCLS-00A07 3576 GiB, 2x SAMSUNG MZ1L2960HCJR-00A07 894 GiB) \
    GPU: 8x AMD Instinct MI300X 192GB HBM3 750W \
    Host OS: Ubuntu 22.04.4 \
    System BIOS: 3.2 \
    System Bios Vendor: American Megatrends International, LLC. \
    Host GPU Driver: (amdgpu version): ROCm 6.3.1 \
    \
    **SYSTEM CONFIGURATION:** \
    NVIDIA HGX H200 Platform \
    System Model: Supermicro SYS-821GE-TNHR \
    CPU: 2x Intel Xeon Platinum 8592V 64-Core Processor \
    NUMA: 2 NUMA node per socket. NUMA auto-balancing enabled \
    Memory: 3072 GiB (32 DIMMs x 96 GiB Micron Technology MTC40F204WS1RC56BB1 DDR5 5600 MT/s) \
    Disk: 432TiB (16 x 27TiB SOLIDIGM SBFPF2BU307T) \
    GPU: 8x NVIDIA Hopper H200 141GB HBM 700W \
    Host OS: Ubuntu 22.04.5 \
    System BIOS: 2.1 \
    System Bios Vendor: American Megatrends International, LLC. \
    Host GPU Driver: Cuda 12.5

[^2]: On average, a system configured with an AMD Instinct™ MI300X GPU shows that AITER block-scale GEMM offers a 2x performance boost substantially accelerating general matrix multiplication tasks. Testing done by AMD on 03/011/2025, results may vary based on configuration, usage, software version, and optimizations.
    **SYSTEM CONFIGURATION:** \
    AMD Instinct™ MI300X platform: \
    System Model: Supermicro GPU A+ Server AS - 8125GS-TNMR2 \
    CPU: 2x AMD EPYC 9654 96-core Processor (2 sockets, 96 cores per socket, 2 threads per core) \
    NUMA Config: 2 NUMA node per socket \
    Memory: 2.3 TiB (24 DIMMs, 4800 mts, 96 GiB/DIMM) \
    Disk: Root drive + Data drive combined: \
    2x 960GB Samsung MZ1L2960HCJR-00A07 \
    4x 3.84TB Samsung MZQL23T8HCLS-00A07 \
    GPU: 8x AMD MI300X 192GB HBM3 750W \
    Host OS: Ubuntu 22.04.4 LTS with Linux kernel 5.15.0-116-generic. \
    System BIOS: 3.2 \
    System Bios Vendor:American Megatrends International, LLC. \
    Host GPU Driver (amdgpu version): 6.10.5

[^3]: On average, a system configured with an AMD Instinct™ MI300X GPU shows AITER block-scale fused MoE offers a 3x performance boost, optimizing the efficiency of Mixture of Experts (MoE) operations. Testing done by AMD on 03/011/2025, results may vary based on configuration, usage, software version, and optimizations.
    **SYSTEM CONFIGURATION:**\
    AMD Instinct ™ MI300X platform \
    System Model: Supermicro AS-8125GS-TNMR2 \
    CPU: 2x AMD EPYC 9654 96-Core Processor \
    NUMA: 2 NUMA node per socket. NUMA auto-balancing disabled \
    Memory: 2304 GiB (24 DIMMs x 96 GiB Micron Technology MTC40F204WS1RC48BB1 DDR5 4800 MT/s) \
    Disk: 16,092 GiB (4x SAMSUNG MZQL23T8HCLS-00A07 3576 GiB, 2x SAMSUNG MZ1L2960HCJR-00A07 894 GiB) \
    GPU: 8x AMD Instinct MI300X 192GB HBM3 750W \
    Host OS: Ubuntu 22.04.4 \
    System BIOS: 3.2 \
    System Bios Vendor: American Megatrends International, LLC \
    Host GPU Driver: (amdgpu version): ROCm 6.3.1

[^4]: On average, a system configured with an AMD Instinct™ MI300X GPU hows that AITER MLA for decode offers a 17x performance boost enhancing decoding efficiency.Testing done by AMD on 03/011/2025, results may vary based on configuration, usage, software version and optimizations.
    **SYSTEM CONFIGURATION:** \
    AMD Instinct™ MI300X platform \
    System Model: Supermicro GPU A+ Server AS - 8125GS-TNMR2 \
    CPU: 2x AMD EPYC 9654 96-core Processor (2 sockets, 96 cores per socket, 2 threads per core \
    NUMA Config: 2 NUMA node per socket \
    Memory: 2.3 TiB (24 DIMMs, 4800 mts, 96 GiB/DIMM) \
    Disk: Root drive + Data drive combined: \
    2x 960GB Samsung MZ1L2960HCJR-00A07 \
    4x 3.84TB Samsung MZQL23T8HCLS-00A07 \
    GPU: 8x AMD MI300X 192GB HBM3 750W \
    Host OS: Ubuntu 22.04.4 LTS with Linux kernel 5.15.0-116-generic \
    System BIOS: 3.2 \  
    System Bios Vendor:   American Megatrends International, LLC. \
    Host GPU Driver (amdgpu version): 6.10.5  

[^5]: On average, a system configured with an AMD Instinct™ MI300X GPU with AITER MHA for prefill shows a14x performance boost, improving Multi-Head Attention (MHA) performance during prefill stages. Testing done by AMD on 03/011/2025, results may vary based on configuration, usage, software version, and optimizations.
    **SYSTEM CONFIGURATION:** \
    AMD Instinct™ MI300X platform \
    System Model: Supermicro GPU A+ Server AS - 8125GS-TNMR2 \
    CPU: 2x AMD EPYC 9654 96-core Processor (2 sockets, 96 cores per socket, 2 threads per core \
    NUMA Config: 2 NUMA node per socket \
    Memory: 2.3 TiB (24 DIMMs, 4800 mts, 96 GiB/DIMM) \
    Disk: Root drive + Data drive combined \
    2x 960GB Samsung MZ1L2960HCJR-00A07 \  
    4x 3.84TB Samsung MZQL23T8HCLS-00A07 \  
    GPU: 8x AMD MI300X 192GB HBM3 750W \  
    Host OS: Ubuntu 22.04.4 LTS with Linux kernel 5.15.0-116-generic. \
    System BIOS: 3.2 \
    System Bios Vendor:   American Megatrends International, LLC. \
    Host GPU Driver (amdgpu version): 6.10.5

### Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
