---
blogpost: true
blog_title: "Understanding RCCL Bandwidth and xGMI Performance on AMD Instinct MI300X"
date: 2 March 2025
author: Jayacharan Kolla, Pedram Alizadeh, Gilbert Lee
thumbnail: 'mi300x-rccl-xgmi.png'
tags: System-Tuning, Performance, Optimization
category: Software tools & optimizations
target_audience: AI / HPC Developers
key_value_propositions: Understand the performance differences and how the scale-up links work within MI300X.
language: English
myst:
    html_meta:
        "author": "Jayacharan Kolla, Pedram Alizadeh, Gilbert Lee"
        "description lang=en": "The blog explains the reasons behind RCCL bandwidth limitations and xGMI performance constraints, and provides actionable steps to maximize link efficiency on AMD MI300X"
        "keywords": "RCCL, xGMI, Collectives, AllReduce, MI300X"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_developer_type": "ML/AI Developer"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_hardware_platforms": 'Instinct GPUs'
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_releasedate": "Mon Feb 24 07:00:00 PST 2025"
---

# Understanding RCCL Bandwidth and xGMI Performance on AMD Instinct™ MI300X

Efficient inter-GPU communication is the backbone of high-performance AI
and HPC workloads, where technologies like RCCL and xGMI play pivotal
roles. However, some limitations in achieving theoretical peak bandwidth
have raised questions about performance bottlenecks. In this blog we explain
the limitations to achieve the theoretical maximum bandwidth in multi-GPU clusters,
and teach you how to perform a set of diagnostics and performance-tuning strategies
that will help you optimize RCCL and xGMI bandwidth on AMD MI300X systems. We will
first introduce you to xGMI and its performance constraints, to RCCL and its bandwidth
limitations, and then cover several practical benchmarks and best practices for maximizing RCCL efficiency.

## xGMI Overview and Performance Expectations

### What is xGMI?

xGMI (External Global Memory Interconnect) is AMD's high-speed
GPU-to-GPU interconnect based on Infinity Fabric™ technology. It
is designed to enable efficient peer-to-peer GPU communication for
multi-GPU AI and HPC workloads. This interconnect is crucial for
ensuring that large-scale AI models and distributed HPC simulations
run efficiently across multiple GPUs. Each AMD Instinct MI300X system
GPU is connected to its seven peer GPUs via xGMI links, forming a
fully connected mesh with high-bandwidth, low-latency
communication. For more information related to the MI300X Platform, refer to this
[datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf).

### Theoretical Performance Claims vs. Achievable

xGMI's raw bandwidth is rated up to 64 GB/s for each
point-to-point link between GPUs. In an ideal situation, this means that
each GPU in a system can communicate with every other GPU at this peak
rate. However, in reality, this bandwidth is constrained by factors like
CRC bit correction and protocol overhead, which reduce the usable
bandwidth to approximately 48 GB/s per link. This represents about 75%
of the theoretical peak bandwidth, which is a critical factor when
evaluating performance in practical workloads.

| Specification                                      | Theoretical Max           | Realized Performance      |
|----------------------------------------------------|---------------------------|---------------------------|
| Aggregated unidirectional bandwidth per GPU       | 448 GB/s (7×64 GB/s)      | 315 - 336 GB/s (7× 45-48 GB/s) |
| GPU-GPU Unidirectional BW via xGMI Link           | 64 GB/s                   | 45 - 48 GB/s                   |

## Why Does RCCL Bandwidth shuffle between 310-330 GB/s for the best-case scenario?

While in reality, the aggregated unidirectional bandwidth per GPU is 336
GB/s, in practical AI workloads, the slowest link limits the total
available bandwidth.

Example: Let\'s practically test out the xGMI Link performances between GPUs in a
single-node MI300X machine. For this experimentation, we will be using
the Transferbench tool. *TransferBench.* is a benchmarking utility designed to measure the
performance of simultaneous data transfers between user-specified
devices, such as CPUs and GPUs. For this example, we will use
Transferbench to identify Inter-GPU xGMI scale-up link bandwidth. Refer to these steps to [install and build Transferbench](https://rocm.docs.amd.com/projects/TransferBench/en/latest/install/install.html#install-transferbench).

The following TransferBench command can be used on MI300X systems to
measure the achievable bandwidth between any pair of directly connected
GPUs, as well as the total All-to-All aggregate bandwidth. This example
below evaluates AllToAll bandwidth using fine-grained memory, an unroll
factor = 2, and 8 Compute Units (CUs) per GPU. The results provide an
excellent upper-bound estimate of the bandwidth that RCCL can achieve.

<div style="max-height: 700px; overflow-y: auto; overflow-x: auto;">

```bash
#Transferbench All to All Run
./TransferBench a2a 64M 8
TransferBench v1.60.00 (with NIC support)
===============================================================
[Common]                              (Suppress by setting HIDE_ENV=1)
ALWAYS_VALIDATE      =            0 : Validating after all iterations
BLOCK_BYTES          =          256 : Each CU gets a mulitple of 256 bytes to copy
BYTE_OFFSET          =            0 : Using byte offset of 0
CLOSEST_NIC          =         auto : Per-GPU closest NIC is set as auto
CU_MASK              =            0 : All
FILL_PATTERN         =            0 : Element i = ((i * 517) modulo 383 + 31) * (srcBufferIdx + 1)
GFX_BLOCK_SIZE       =          256 : Threadblock size of 256
GFX_SINGLE_TEAM      =            1 : Combining CUs to work across entire data array
GFX_UNROLL           =            2 : Using GFX unroll factor of 2
GFX_WAVE_ORDER       =            0 : Using GFX wave ordering of Unroll,Wavefront,CU
IP_ADDRESS_FAMILY    =            4 : IP address family is set to IPv4
IB_GID_INDEX         =           -1 : RoCE GID index is set to auto
IB_PORT_NUMBER       =            1 : IB port number is set to 1
MIN_VAR_SUBEXEC      =            1 : Using at least 1 subexecutor(s) for variable subExec tranfers
MAX_VAR_SUBEXEC      =            0 : Using up to all available subexecutors for variable subExec transfers
NIC_RELAX_ORDER      =            1 : Using relaxed ordering for NIC RDMA
NUM_ITERATIONS       =           10 : Running 10  timed iteration(s)
NUM_SUBITERATIONS    =            1 : Running 1 subiterations
NUM_WARMUPS          =            3 : Running 3 warmup iteration(s) per Test
ROCE_VERSION         =            2 : RoCE version is set to 2
SHOW_ITERATIONS      =            0 : Hiding per-iteration timing
USE_HIP_EVENTS       =            1 : Using HIP events for GFX/DMA Executor timing
USE_HSA_DMA          =            0 : Using hipMemcpyAsync for DMA execution
USE_INTERACTIVE      =            0 : Running in non-interactive mode
USE_SINGLE_STREAM    =            1 : Using single stream per GFX device
VALIDATE_DIRECT      =            0 : Validate GPU destination memory via CPU staging buffer
VALIDATE_SOURCE      =            0 : Do not perform source validation after prep

[AllToAll Related]
A2A_DIRECT           =            1 : Only using direct links
A2A_LOCAL            =            0 : Exclude local transfers
A2A_MODE             =            0 : Copy
NUM_GPU_DEVICES      =            8 : Using 8 GPUs
NUM_QUEUE_PAIRS      =            0 : Using 0 queue pairs for NIC transfers
NUM_SUB_EXEC         =            8 : Using 8 subexecutors/CUs per Transfer
USE_DMA_EXEC         =            0 : Using GFX executor
USE_FINE_GRAIN       =            1 : Using fine-grained memory
USE_REMOTE_READ      =            0 : Using SRC as executor

GPU-GFX All-To-All benchmark:
==========================
- Copying 67108864 bytes between directly connected pairs of GPUs using 8 CUs (56 Transfers)
Test 1:
 Executor: GPU 00 |  315.552 GB/s |    1.489 ms |    469762048 bytes | 326.317 GB/s (sum)
     Transfer 00  |   47.056 GB/s |    1.426 ms |     67108864 bytes | F0 -> G000:008 -> F1
     Transfer 01  |   46.826 GB/s |    1.433 ms |     67108864 bytes | F0 -> G000:008 -> F2
     Transfer 02  |   47.126 GB/s |    1.424 ms |     67108864 bytes | F0 -> G000:008 -> F3
     Transfer 03  |   46.432 GB/s |    1.445 ms |     67108864 bytes | F0 -> G000:008 -> F4
     Transfer 04  |   46.883 GB/s |    1.431 ms |     67108864 bytes | F0 -> G000:008 -> F5
     Transfer 05  |   46.784 GB/s |    1.434 ms |     67108864 bytes | F0 -> G000:008 -> F6
     Transfer 06  |   45.210 GB/s |    1.484 ms |     67108864 bytes | F0 -> G000:008 -> F7
 Executor: GPU 01 |  317.663 GB/s |    1.479 ms |    469762048 bytes | 325.040 GB/s (sum)
     Transfer 07  |   47.000 GB/s |    1.428 ms |     67108864 bytes | F1 -> G001:008 -> F0
     Transfer 08  |   46.751 GB/s |    1.435 ms |     67108864 bytes | F1 -> G001:008 -> F2
     Transfer 09  |   46.935 GB/s |    1.430 ms |     67108864 bytes | F1 -> G001:008 -> F3
     Transfer 10  |   46.436 GB/s |    1.445 ms |     67108864 bytes | F1 -> G001:008 -> F4
     Transfer 11  |   45.543 GB/s |    1.474 ms |     67108864 bytes | F1 -> G001:008 -> F5
     Transfer 12  |   46.260 GB/s |    1.451 ms |     67108864 bytes | F1 -> G001:008 -> F6
     Transfer 13  |   46.116 GB/s |    1.455 ms |     67108864 bytes | F1 -> G001:008 -> F7
 Executor: GPU 02 |  317.041 GB/s |    1.482 ms |    469762048 bytes | 325.612 GB/s (sum)
     Transfer 14  |   47.098 GB/s |    1.425 ms |     67108864 bytes | F2 -> G002:008 -> F0
     Transfer 15  |   46.785 GB/s |    1.434 ms |     67108864 bytes | F2 -> G002:008 -> F1
     Transfer 16  |   47.032 GB/s |    1.427 ms |     67108864 bytes | F2 -> G002:008 -> F3
     Transfer 17  |   46.048 GB/s |    1.457 ms |     67108864 bytes | F2 -> G002:008 -> F4
     Transfer 18  |   46.829 GB/s |    1.433 ms |     67108864 bytes | F2 -> G002:008 -> F5
     Transfer 19  |   45.451 GB/s |    1.476 ms |     67108864 bytes | F2 -> G002:008 -> F6
     Transfer 20  |   46.370 GB/s |    1.447 ms |     67108864 bytes | F2 -> G002:008 -> F7
 Executor: GPU 03 |  315.094 GB/s |    1.491 ms |    469762048 bytes | 326.649 GB/s (sum)
     Transfer 21  |   46.931 GB/s |    1.430 ms |     67108864 bytes | F3 -> G003:008 -> F0
     Transfer 22  |   47.009 GB/s |    1.428 ms |     67108864 bytes | F3 -> G003:008 -> F1
     Transfer 23  |   46.941 GB/s |    1.430 ms |     67108864 bytes | F3 -> G003:008 -> F2
     Transfer 24  |   45.142 GB/s |    1.487 ms |     67108864 bytes | F3 -> G003:008 -> F4
     Transfer 25  |   46.718 GB/s |    1.436 ms |     67108864 bytes | F3 -> G003:008 -> F5
     Transfer 26  |   46.982 GB/s |    1.428 ms |     67108864 bytes | F3 -> G003:008 -> F6
     Transfer 27  |   46.925 GB/s |    1.430 ms |     67108864 bytes | F3 -> G003:008 -> F7
 Executor: GPU 04 |  318.348 GB/s |    1.476 ms |    469762048 bytes | 325.427 GB/s (sum)
     Transfer 28  |   46.551 GB/s |    1.442 ms |     67108864 bytes | F4 -> G004:008 -> F0
     Transfer 29  |   46.333 GB/s |    1.448 ms |     67108864 bytes | F4 -> G004:008 -> F1
     Transfer 30  |   46.084 GB/s |    1.456 ms |     67108864 bytes | F4 -> G004:008 -> F2
     Transfer 31  |   45.614 GB/s |    1.471 ms |     67108864 bytes | F4 -> G004:008 -> F3
     Transfer 32  |   46.670 GB/s |    1.438 ms |     67108864 bytes | F4 -> G004:008 -> F5
     Transfer 33  |   47.033 GB/s |    1.427 ms |     67108864 bytes | F4 -> G004:008 -> F6
     Transfer 34  |   47.140 GB/s |    1.424 ms |     67108864 bytes | F4 -> G004:008 -> F7
 Executor: GPU 05 |  317.032 GB/s |    1.482 ms |    469762048 bytes | 326.659 GB/s (sum)
     Transfer 35  |   46.831 GB/s |    1.433 ms |     67108864 bytes | F5 -> G005:008 -> F0
     Transfer 36  |   45.424 GB/s |    1.477 ms |     67108864 bytes | F5 -> G005:008 -> F1
     Transfer 37  |   46.821 GB/s |    1.433 ms |     67108864 bytes | F5 -> G005:008 -> F2
     Transfer 38  |   46.768 GB/s |    1.435 ms |     67108864 bytes | F5 -> G005:008 -> F3
     Transfer 39  |   46.734 GB/s |    1.436 ms |     67108864 bytes | F5 -> G005:008 -> F4
     Transfer 40  |   47.073 GB/s |    1.426 ms |     67108864 bytes | F5 -> G005:008 -> F6
     Transfer 41  |   47.008 GB/s |    1.428 ms |     67108864 bytes | F5 -> G005:008 -> F7
 Executor: GPU 06 |  317.454 GB/s |    1.480 ms |    469762048 bytes | 325.979 GB/s (sum)
     Transfer 42  |   46.844 GB/s |    1.433 ms |     67108864 bytes | F6 -> G006:008 -> F0
     Transfer 43  |   46.278 GB/s |    1.450 ms |     67108864 bytes | F6 -> G006:008 -> F1
     Transfer 44  |   45.483 GB/s |    1.475 ms |     67108864 bytes | F6 -> G006:008 -> F2
     Transfer 45  |   46.713 GB/s |    1.437 ms |     67108864 bytes | F6 -> G006:008 -> F3
     Transfer 46  |   46.936 GB/s |    1.430 ms |     67108864 bytes | F6 -> G006:008 -> F4
     Transfer 47  |   46.862 GB/s |    1.432 ms |     67108864 bytes | F6 -> G006:008 -> F5
     Transfer 48  |   46.864 GB/s |    1.432 ms |     67108864 bytes | F6 -> G006:008 -> F7
 Executor: GPU 07 |  314.759 GB/s |    1.492 ms |    469762048 bytes | 325.207 GB/s (sum)
     Transfer 49  |   45.095 GB/s |    1.488 ms |     67108864 bytes | F7 -> G007:008 -> F0
     Transfer 50  |   46.475 GB/s |    1.444 ms |     67108864 bytes | F7 -> G007:008 -> F1
     Transfer 51  |   46.049 GB/s |    1.457 ms |     67108864 bytes | F7 -> G007:008 -> F2
     Transfer 52  |   46.713 GB/s |    1.437 ms |     67108864 bytes | F7 -> G007:008 -> F3
     Transfer 53  |   46.907 GB/s |    1.431 ms |     67108864 bytes | F7 -> G007:008 -> F4
     Transfer 54  |   47.182 GB/s |    1.422 ms |     67108864 bytes | F7 -> G007:008 -> F5
     Transfer 55  |   46.786 GB/s |    1.434 ms |     67108864 bytes | F7 -> G007:008 -> F6
 Aggregate (CPU)  | 1937.924 GB/s |    1.939 ms |   3758096384 bytes | Overhead: 0.447 ms

Summary: [67108864 bytes per Transfer]
==========================================================
SRC\DST  GPU 00     GPU 01     GPU 02     GPU 03     GPU 04     GPU 05     GPU 06     GPU 07        STotal      Actual
GPU 00      N/A     47.056     46.826     47.126     46.432     46.883     46.784     45.210       326.317     316.469
GPU 01   47.000        N/A     46.751     46.935     46.436     45.543     46.260     46.116       325.040     318.799
GPU 02   47.098     46.785        N/A     47.032     46.048     46.829     45.451     46.370       325.612     318.160
GPU 03   46.931     47.009     46.941        N/A     45.142     46.718     46.982     46.925       326.649     315.994
GPU 04   46.551     46.333     46.084     45.614        N/A     46.670     47.033     47.140       325.427     319.297
GPU 05   46.831     45.424     46.821     46.768     46.734        N/A     47.073     47.008       326.659     317.968
GPU 06   46.844     46.278     45.483     46.713     46.936     46.862        N/A     46.864       325.979     318.378
GPU 07   45.095     46.475     46.049     46.713     46.907     47.182     46.786        N/A       325.207     315.664

RTotal  326.350    325.359    324.955    326.901    324.636    326.686    326.370    325.634      2606.891     315.664     319.297

Average   bandwidth (GPU Timed):   46.552 GB/s
Aggregate bandwidth (GPU Timed): 2606.891 GB/s
Aggregate bandwidth (CPU Timed): 1937.924 GB/s
```

</div>

Based on the above results we obtain the Inter-GPU xGMI Link Bandwidth as shown below,

| GPU  |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    |  Total  | Actual  |
|------|--------|--------|--------|--------|--------|--------|--------|--------|---------|---------|
|  0   |  N/A   | 47.056 | 46.826 | 47.126 | 46.432 | 46.883 | 46.784 | 45.210 | 326.317 | 316.469 |
|  1   | 47.000 |  N/A   | 46.751 | 46.935 | 46.436 | 45.543 | 46.260 | 46.116 | 325.040 | 318.799 |
|  2   | 47.098 | 46.785 |  N/A   | 47.032 | 46.048 | 46.829 | 45.451 | 46.370 | 325.612 | 318.160 |
|  3   | 46.931 | 47.009 | 46.941 |  N/A   | 45.142 | 46.718 | 46.982 | 46.925 | 326.649 | 315.994 |
|  4   | 46.551 | 46.333 | 46.084 | 45.614 |  N/A   | 46.670 | 47.033 | 47.140 | 325.427 | 319.297 |
|  5   | 46.831 | 45.424 | 46.821 | 46.768 | 46.734 |  N/A   | 47.073 | 47.008 | 326.659 | 317.968 |
|  6   | 46.844 | 46.278 | 45.483 | 46.713 | 46.936 | 46.862 |  N/A   | 46.864 | 325.979 | 318.378 |
|  7   | 45.095 | 46.475 | 46.049 | 46.713 | 46.907 | 47.182 | 46.786 |  N/A   | 325.207 | 315.664 |

Let\'s consider the interaction between GPU 0 to all other GPUs as represented in the below diagram, The
slowest link from **GPU 0** is between **GPU 0 → GPU 7** = **45.21 GB /
s** which translates to an Actual Aggregated BW of **\~ 316.47 GB / s (
7 \* 45.21** **GB/s** **)** for your collective operation in an AI
Training Workload although your total is \~326.317 GB/s. Since the
slowest link is what defines the total bandwidth in this case, that
causes the RCCL Bandwidth to be effectively capped at 310-330 GB/s.

```{figure} ./images/image1.png
:align: center
:alt: GPU Bandwidth Analysis
:width: 50%
Figure 1: xGMI Bandwidth between GPU 0 to other GPUs
```

Furthermore, you can also use RCCL-Test to benchmark bandwidth and
latency for various collective operations across different message
sizes. The following snippets demonstrates running an AllReduce benchmark for
message sizes ranging from 8 bytes to 16 GB, using one process per GPU,
with a multiplicative factor of 2 between message sizes.

```bash
#AllReduce RCCL Test
mpirun -np 8 --bind-to numa -env NCCL_DEBUG=VERSION rccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
rccl-tests: Version develop:448c4c7
# Using devices
#   Rank  0 Pid 2509534 on host device  0 [0000:0c:00.0] AMD Instinct MI300X
#   Rank  1 Pid 2509535 on host device  1 [0000:22:00.0] AMD Instinct MI300X
#   Rank  2 Pid 2509536 on host device  2 [0000:38:00.0] AMD Instinct MI300X
#   Rank  3 Pid 2509537 on host device  3 [0000:5c:00.0] AMD Instinct MI300X
#   Rank  4 Pid 2509538 on host device  4 [0000:9f:00.0] AMD Instinct MI300X
#   Rank  5 Pid 2509539 on host device  5 [0000:af:00.0] AMD Instinct MI300X
#   Rank  6 Pid 2509540 on host device  6 [0000:bf:00.0] AMD Instinct MI300X
#   Rank  7 Pid 2509541 on host device  7 [0000:df:00.0] AMD Instinct MI300X
RCCL version : 2.22.3-develop:f5b15f2
HIP version  : 6.3.42131-fa1d09cbd
ROCm version : 6.3.0.0-39-ce2be3b
Hostname     : host
Librccl path : /home/User/all2all/doc/rccl/build/release/librccl.so.1
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    16.61    0.00    0.00      0    16.53    0.00    0.00      0
          16             4     float     sum      -1    16.12    0.00    0.00      0    15.94    0.00    0.00      0
          32             8     float     sum      -1    16.56    0.00    0.00      0    16.68    0.00    0.00      0
          64            16     float     sum      -1    17.94    0.00    0.01      0    18.36    0.00    0.01      0
         128            32     float     sum      -1    22.94    0.01    0.01      0    22.78    0.01    0.01      0
         256            64     float     sum      -1    22.93    0.01    0.02      0    22.85    0.01    0.02      0
         512           128     float     sum      -1    22.91    0.02    0.04      0    23.20    0.02    0.04      0
        1024           256     float     sum      -1    13.11    0.08    0.14      0    12.79    0.08    0.14      0
        2048           512     float     sum      -1    13.09    0.16    0.27      0    12.96    0.16    0.28      0
        4096          1024     float     sum      -1    13.27    0.31    0.54      0    12.71    0.32    0.56      0
        8192          2048     float     sum      -1    13.30    0.62    1.08      0    12.89    0.64    1.11      0
       16384          4096     float     sum      -1    13.54    1.21    2.12      0    13.01    1.26    2.20      0
       32768          8192     float     sum      -1    13.31    2.46    4.31      0    13.10    2.50    4.38      0
       65536         16384     float     sum      -1    13.77    4.76    8.33      0    13.38    4.90    8.57      0
      131072         32768     float     sum      -1    14.14    9.27   16.22      0    13.77    9.52   16.66      0
      262144         65536     float     sum      -1    16.87   15.54   27.20      0    16.55   15.84   27.72      0
      524288        131072     float     sum      -1    24.41   21.48   37.59      0    23.88   21.95   38.42      0
     1048576        262144     float     sum      -1    27.42   38.25   66.93      0    27.13   38.65   67.64      0
     2097152        524288     float     sum      -1    34.81   60.25  105.43      0    34.84   60.20  105.35      0
     4194304       1048576     float     sum      -1    51.72   81.09  141.91      0    129.0   32.52   56.92      0
     8388608       2097152     float     sum      -1    83.68  100.24  175.43      0    86.67   96.78  169.37      0
    16777216       4194304     float     sum      -1    148.8  112.72  197.26      0    155.6  107.81  188.66      0
    33554432       8388608     float     sum      -1    244.1  137.47  240.57      0    256.0  131.07  229.38      0
    67108864      16777216     float     sum      -1    429.1  156.38  273.66      0    440.6  152.30  266.52      0
   134217728      33554432     float     sum      -1    797.4  168.32  294.55      0    805.8  166.56  291.48      0
   268435456      67108864     float     sum      -1   1537.1  174.64  305.62      0   1546.0  173.63  303.85      0
   536870912     134217728     float     sum      -1   3026.2  177.41  310.46      0   3034.2  176.94  309.64      0
  1073741824     268435456     float     sum      -1   5970.6  179.84  314.72      0   5974.5  179.72  314.51      0
  2147483648     536870912     float     sum      -1    11891  180.59  316.04      0    11895  180.54  315.94      0
  4294967296    1073741824     float     sum      -1    23680  181.37  317.40      0    23769  180.69  316.21      0
  8589934592    2147483648     float     sum      -1    47278  181.69  317.96      0    47323  181.52  317.66      0
17179869184    4294967296     float     sum      -1    94254  182.27  318.98      0    94328  182.13  318.73      0
# Errors with asterisks indicate errors that have exceeded the maximum threshold.
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 116.669
```

You can further see that running with Graph mode enabled `-G 1`
flag will improve the latency in small message sizes where the
collective operations are latency-bound.

```bash
mpirun -np 8 --bind-to numa -env NCCL_DEBUG=VERSION rccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 1 -G 1
# nThread 1 nGpus 1 minBytes 8 maxBytes 17179869184 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
rccl-tests: Version develop:448c4c7
# Using devices
#   Rank  0 Pid 2510387 on host device  0 [0000:0c:00.0] AMD Instinct MI300X
#   Rank  1 Pid 2510388 on host device  1 [0000:22:00.0] AMD Instinct MI300X
#   Rank  2 Pid 2510389 on host device  2 [0000:38:00.0] AMD Instinct MI300X
#   Rank  3 Pid 2510390 on host device  3 [0000:5c:00.0] AMD Instinct MI300X
#   Rank  4 Pid 2510391 on host device  4 [0000:9f:00.0] AMD Instinct MI300X
#   Rank  5 Pid 2510392 on host device  5 [0000:af:00.0] AMD Instinct MI300X
#   Rank  6 Pid 2510393 on host device  6 [0000:bf:00.0] AMD Instinct MI300X
#   Rank  7 Pid 2510394 on host device  7 [0000:df:00.0] AMD Instinct MI300X
RCCL version : 2.22.3-develop:f5b15f2
HIP version  : 6.3.42131-fa1d09cbd
ROCm version : 6.3.0.0-39-ce2be3b
Hostname     : host
Librccl path : /home/User/all2all/doc/rccl/build/release/librccl.so.1
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    16.38    0.00    0.00      0    16.29    0.00    0.00      0
          16             4     float     sum      -1    15.75    0.00    0.00      0    15.79    0.00    0.00      0
          32             8     float     sum      -1     6.32    0.01    0.01      0     6.17    0.01    0.01      0
          64            16     float     sum      -1     6.00    0.01    0.02      0     6.51    0.01    0.02      0
         128            32     float     sum      -1     6.53    0.02    0.03      0     6.20    0.02    0.04      0
         256            64     float     sum      -1     6.77    0.04    0.07      0     7.18    0.04    0.06      0
         512           128     float     sum      -1     7.44    0.07    0.12      0     7.84    0.07    0.11      0
        1024           256     float     sum      -1     7.47    0.14    0.24      0     7.76    0.13    0.23      0
        2048           512     float     sum      -1     7.46    0.27    0.48      0     7.54    0.27    0.48      0
        4096          1024     float     sum      -1     7.61    0.54    0.94      0     7.61    0.54    0.94      0
        8192          2048     float     sum      -1     7.64    1.07    1.88      0     8.10    1.01    1.77      0
       16384          4096     float     sum      -1     8.71    1.88    3.29      0     8.10    2.02    3.54      0
       32768          8192     float     sum      -1     9.05    3.62    6.33      0     8.79    3.73    6.52      0
       65536         16384     float     sum      -1     9.14    7.17   12.54      0     9.14    7.17   12.54      0
      131072         32768     float     sum      -1     9.51   13.78   24.12      0     9.53   13.75   24.06      0
      262144         65536     float     sum      -1    10.26   25.54   44.69      0    10.32   25.39   44.43      0
      524288        131072     float     sum      -1    14.26   36.75   64.32      0    14.64   35.81   62.66      0
     1048576        262144     float     sum      -1    19.40   54.06   94.61      0    19.72   53.18   93.06      0
     2097152        524288     float     sum      -1    35.03   59.86  104.76      0    34.83   60.20  105.35      0
     4194304       1048576     float     sum      -1    51.83   80.93  141.62      0    52.99   79.15  138.52      0
     8388608       2097152     float     sum      -1    83.74  100.17  175.30      0    86.66   96.80  169.41      0
    16777216       4194304     float     sum      -1    149.1  112.55  196.96      0    156.3  107.32  187.81      0
    33554432       8388608     float     sum      -1    243.4  137.85  241.23      0    255.5  131.33  229.82      0
    67108864      16777216     float     sum      -1    427.9  156.83  274.45      0    439.1  152.83  267.46      0
   134217728      33554432     float     sum      -1    795.3  168.77  295.35      0    805.9  166.54  291.44      0
   268435456      67108864     float     sum      -1   1534.4  174.94  306.15      0   1546.7  173.55  303.72      0
   536870912     134217728     float     sum      -1   3024.0  177.54  310.69      0   3034.5  176.92  309.62      0
  1073741824     268435456     float     sum      -1   5961.9  180.10  315.18      0   5974.7  179.71  314.50      0
  2147483648     536870912     float     sum      -1    11880  180.77  316.34      0    11897  180.50  315.88      0
  4294967296    1073741824     float     sum      -1    23658  181.55  317.71      0    23681  181.37  317.40      0
  8589934592    2147483648     float     sum      -1    47244  181.82  318.18      0    47287  181.65  317.89      0
17179869184    4294967296     float     sum      -1    94182  182.41  319.22      0    94540  181.72  318.01      0
# Errors with asterisks indicate errors that have exceeded the maximum threshold.
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 120.69
```

## Best Practices to Maximize RCCL Efficiency

- Utilization of all GPUs within a node

  - The MI300X architecture features dedicated links between GPUs, forming a *fully connected topology*.
  - For collective operations, the highest performance is achieved when *all 8 GPUs are used*, ensuring all inter GPU links are active.
  - When performing collective operations with only 2 or 4 GPUs, only a *fraction of the available bandwidth* on the node is utilized. But, if your workload limits you to use less than 8 MI300X GPUs on a system, you can use the run-time variable `NCCL_MIN_NCHANNELS` to increase the number of channels or CU\'s used, thereby improving RCCL performance.
    - Use the following environment variable to increase the number of channels used by RCCL when using RCCL in single-node E2E workloads to potentially improve the performance:
            `export NCCL_MIN_NCHANNELS=112`

- Disable NUMA Auto balancing

  - NUMA (Non-Uniform Memory Access) auto-balancing is designed to improve memory access efficiency on multi-socket systems. It dynamically moves memory pages between NUMA nodes to balance memory load across available processors. While this is beneficial for some workloads, it can introduce performance overheads in GPU-specific collective operations. Disabling NUMA Auto balancing is required for reducing performance variability and to achieve better results.
  - How to check if NUMA Auto balancing is disabled?
    - Run the following command to check its status

      ```bash
      cat /proc/sys/kernel/numa_balancing
      ```

      - If the output is **0**, NUMA auto-balancing is already disabled.
      - If the output is **1**, NUMA auto-balancing is enabled and should be turned off.

    - To disable it run the following command.

      ```bash
      sudo sysctl kernel.numa_balancing=0
      ```

    - Note: Please make sure daemons like `numad` are also disabled.

- Use a Single Process per GPU

  - RCCL delivers the best performance for collectives when it is configured in a one-process-per-GPU mode. This is because, with a one-process-per-multiple-GPUs configuration, you can run into kernel launch latency issues, this is because ROCm serializes kernel launches on multiple GPUs from one process which hurts performance. The examples are specific to RCCL-Tests only.

    - How to enable Single Process per GPU mode:

    ```bash
    mpirun -np #number_of_GPUs ./build/all_reduce_perf -b minbytes -e maxbytes -f multiplier -g 1
    ```

    - If one-process-per-GPU mode is not feasible, a single-process, multi-threaded approach provides better performance than a single-threaded single-process

    ```bash
    mpirun -np #number_of_GPUs ./build/all_reduce_perf -b minbytes -e maxbytes -f multiplier -g 1 -t threads_per_process
    ```

    - Enabling Graph could further improve performance by saving on kernel launch overhead:

    ```bash
    mpirun -np #number_of_GPUs ./build/all_reduce_perf -b minbytes -e maxbytes -f multiplier -g gpus_per_thread -t threads_per_process -G number_of_kernel_graph_launch
    ```

## Resources

- [RCCL GitHub Repository](https://github.com/ROCmSoftwarePlatform/rccl)

- [TransferBench Github Repository](https://github.com/ROCm/TransferBench)

- [RCCL-Test Github Repository](https://github.com/ROCm/rccl-tests)

## Summary

In this blog, we talked about xGMI bandwidth and RCCL performance on the MI300X
giving key insights into inter-GPU communication efficiency,
highlighting both theoretical expectations and real-world performance
considerations. While xGMI provides a high-bandwidth, low-latency
interconnect, practical limitations such as protocol overhead and bit
correction and the slowest link constraints impact overall bandwidth
utilization in AI and HPC workloads.

We also walked through the benchmarking with *TransferBench* and *RCCL-Test* to understand xGMI bandwidth and RCCL performance, we
observed that the aggregated theoretical bandwidth per GPU reaches *336 GB/s*, and practical execution yields values between *310-330 GB/s*,
largely dictated by the slow link in the communication topology. By
leveraging efficient configurations as mentioned in the best practices
above users can further enhance bandwidth utilization and maximize
efficiency. These optimizations empower developers to fully harness the
MI300X platform, driving faster AI training and high-performance
computing workloads with greater scalability and efficiency.
