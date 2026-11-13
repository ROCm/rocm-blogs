---
blogpost: true
blog_title: "GPU-Initiated DMA Copy Engines in RCCL and rocSHMEM"
date: "13 Nov 2026"
author: "Aurelien Bouteiller, Nusrat Islam, Edgar Gabriel, Maria Garzaran"
thumbnail: 'rccl_thumbnail_image_blog.png'
tags: "Performance, Optimization, HPC"
category: "Software tools & optimizations"
target_audience: "HPC/GPU software engineers and performance engineers"
key_value_propositions: "Offload GPU communication to SDMA engines, freeing compute units for computation while achieving full scale-up bandwidth with one Compute Unit."
language: English
myst:
    html_meta:
        "author": "Aurelien Bouteiller, Nusrat Islam, Edgar Gabriel, Maria Garzaran"
        "description lang=en": "Learn how GPU-initiated SDMA engines in RCCL and rocSHMEM offload data transfers to free compute units and maximize scale-up bandwidth. "
        "keywords": "GPU-initiated SDMA, RCCL, rocSHMEM, multi-GPU communication"
        "vertical": "HPC"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Deploying AI at Scale"
        "amd_blog_topic_categories": "HPC & Scientific Computing"
        "amd_blog_authors": "Aurelien Bouteiller, Nusrat Islam, Edgar Gabriel, Maria Garzaran"
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

# GPU-Initiated DMA Copy Engines in RCCL and rocSHMEM

As GPU-accelerated workloads scale across multiple devices and nodes, the efficiency of inter-GPU communication becomes a decisive factor in overall application performance. This is especially true for inference workloads, where communication sits on the critical path and traditional approaches to overlapping compute and communication fall short without fine-grained, kernel-level orchestration.

In this blog, you will explore a powerful new capability available in both RCCL and rocSHMEM: GPU-initiated SDMA (System DMA) copy engines for intra-node data movement. You will learn how this approach decouples data transfer from compute unit execution, freeing up GPU resources for computation while still saturating available bandwidth. Through performance results collected on AMD Instinct™ MI350X hardware, you will see how GPU-initiated SDMA achieves bandwidth comparable to or exceeding traditional implementations, using as few as a single compute unit instead of 64.

Device-side APIs are critical because they enable the fusion of computation and communication. This is a capability that is particularly essential for optimizing inference workloads, where communication lies on the critical path and where overlapping compute and communication is impractical without fine-grained orchestration between the two.

Recently, RCCL has introduced support for GPU-initiated communication, enabling data transfers within the scale-up domain through regular load/store operations as well as across nodes through GPU-initiated networking (GIN). rocSHMEM was architected with this capability from the outset. Both GPU-initiated communication APIs provide a clean abstraction to access hardware features that enable high efficiency scale-up data transfers, maximizing bandwidth while using fewer resources.

## Highlights

- High bandwidth with fewer compute units (CUs): in eight-GPU MI350X tests, GPU-initiated SDMA achieves comparable or higher large-message bandwidth using one versus 64 CUs for default RCCL `AlltoAll` and `AllGather`. Similarly, in rocSHMEM `putnbi` tests, SDMA delivers the highest bandwidth among tested configurations for messages 512 KiB and larger, using one CU and one wavefront.
- More resources for computation: GPU kernels initiate transfers while dedicated SDMA engines move the data, reducing CU usage in fused compute-and-communication kernels.
- Demonstrated across libraries: generic solution that works across libraries.

## A New Approach: GPU-Initiated DMA Copy Engines

GPUs can move data using either CUs, which execute load/store instructions, or dedicated DMA copy engines called SDMA on AMD hardware. A separate distinction is who initiates the transfer: the CPU or a running GPU kernel.

| Transfer approach | Who initiates the transfer? | Who performs the data movement? | CU involvement |
| --- | --- | --- | --- |
| GPU load/store communication | GPU kernel | CUs executing load/store instructions | CUs execute the copying work |
| Host-initiated SDMA | CPU | SDMA engines | CUs do not execute the copy |
| GPU-initiated SDMA | GPU kernel | SDMA engines | CUs issue commands and coordinate completion |

While GPU-initiated communication itself is not novel, what distinguishes this release is the addition of support for GPU-initiated communication within the scale-up domain using SDMA copy engines. This is a significant advancement: this release enables GPU-initiated communication that leverages SDMA engines rather than compute units for data movement. AMD GPUs are equipped with enough SDMA engines to fully saturate the available bandwidth within the scale-up domain.

The key advantage of this approach lies in resource utilization. In conventional fused compute-and-communication kernels, compute units must service both the computation and the data transfer, creating contention for shared resources. By offloading the communication to SDMA engines through GPU-initiated commands, compute resources are fully available for the computational portion of the kernel. Notably, this approach can saturate the available bandwidth using only a single CU, the one required to issue the SDMA commands.

## Performance Evaluation

### RCCL Results

To demonstrate the benefits of this approach, Figure 1 and Figure 2 compare the performance of three different implementations for `AlltoAll` and `Allgather`, respectively:

- The default RCCL implementation based on load/store instructions
- The host-initiated SDMA (Host-SDMA)
- The GPU-initiated SDMA (GIN-SDMA)

Host-SDMA and GIN-SDMA rely on symmetric memory registration, whereas the default implementation does not. Experimental results were collected on eight GPUs of an AMD Instinct™ MI350X node, with bus bandwidth reported as a function of message size. As shown in the figures, the SDMA implementations achieve bandwidth comparable to—and slightly higher than—the default implementation for the large message sizes evaluated. For alltoall, GIN-SDMA backend is used for message sizes greater than 32MiB; smaller sizes use vector load-store operations (red line in Figure-1). For allgather, GIN-SDMA is used for messages 512KiB or larger. Notably, GIN-SDMA delivers higher bandwidth than the default while using only a single CU, compared to the 64 CUs required by the default RCCL algorithm.

```{figure} images/rccl-alltoall-busbw-8gpu-mi350x.svg
:alt: Figure 1: AlltoAll bus bandwidth comparison across three implementations on 8-GPU MI350X
:name: fig-alltoall-busbw
Figure 1: `AlltoAll` bus bandwidth across eight AMD Instinct MI350X GPUs in a single node
```

```{figure} images/rccl-allgather-busbw-8gpu-mi350x.svg
:alt: Figure 2: Allgather bus bandwidth comparison across three implementations on 8-GPU MI350X
:name: fig-allgather-busbw
Figure 2: `Allgather` bus bandwidth across eight AMD Instinct MI350X GPUs in a single node
```

### rocSHMEM Results

GPU-initiated SDMA is also supported in rocSHMEM, as illustrated in Figure 3 and Figure 4.

Figure 3 shows the bus bandwidth of `rocshmem_alltoall_wg` running on eight GPUs of an AMD Instinct™ MI350X GPU node with a single CU. Performance is compared across two configurations: (a) SDMA enabled (SDMA ON) with one CU and one wavefront, and (b) SDMA disabled (SDMA OFF) with one CU and 16 wavefronts. As the figure shows, SDMA ON delivers significantly higher performance because a single CU cannot saturate the available bandwidth using regular loads and stores.

| Configuration | Details | Results |
| --- | --- | --- |
| SDMA OFF: one CU, 16 wavefronts | Threads on the CU execute the copying instructions | Lower bandwidth |
| SDMA ON: one CU, one wavefront | Threads issue commands; SDMA engines copy the data | Significantly higher bandwidth |

Figure 4 shows the bandwidth of a `putnbi` benchmark between two GPUs on an AMD Instinct™ MI350X GPU node. Results are shown for SDMA ON with one CU using one and four wavefronts, and for SDMA OFF with various CU and wavefront configurations. At 128 KiB, the bandwidth achieved by a single CU is similar whether SDMA is enabled or not, although using a larger number of CUs with SDMA OFF can drive higher bandwidth at this message size. For messages of 512 KiB or larger, SDMA ON delivers the highest bandwidth regardless of configuration. The current rocSHMEM implementation does not use SDMA for messages smaller than 128 KiB; consequently, the configuration with four wavefronts outperforms the single-wavefront configuration at 64 KiB, as the load/store path benefits from additional wavefronts. However, once SDMA is used, a single CU with a single wavefront is sufficient to achieve peak bandwidth.

```{figure} images/rocshmem-alltoall-busbw-8gpu-mi350x.svg
:alt: Figure 3: rocshmem_alltoall_wg bus bandwidth with SDMA ON vs OFF on 8-GPU MI350X
:name: fig-rocshmem-alltoall-busbw
Figure 3: `rocshmem_alltoall_wg` bus bandwidth across eight AMD Instinct MI350X GPUs in a single node
```

```{figure} images/putnbi-bandwidth-two-gpus-mi350x.svg
:alt: Figure 4: putnbi bandwidth between two GPUs on MI350X with SDMA ON vs OFF configurations
:name: fig-putnbi-bw
Figure 4: `putnbi` bandwidth between two AMD Instinct MI350X GPUs
```

## Architectural Perspective

Each graphics compute die (GCD) has multiple SDMA engines available to perform copy operations. At a low level, an SDMA engine is a programmable DMA engine (similar in spirit to an RDMA NIC) but designed for inter-GPU communication within a node or pod. Figure 5 illustrates how SDMA engines are controlled at the low level. A ring buffer is populated with DMA commands (e.g., linear copies, copies with fused signal, or atomic operations), and a doorbell mechanism is used to trigger consumption of the pending entries in the ring buffer. For host-initiated operations, the `hipMemcpyAsync` API from the standard ROCm runtime abstracts this complexity, making SDMA engines readily accessible to the programmer. For device-initiated operations, RCCL-GIN and rocSHMEM prepare the SDMA packets internally, providing a similarly transparent interface to device kernels.

```{figure} images/sdma-engine-low-level-details.png
:alt: Figure 5: SDMA engine ring buffer and doorbell mechanism diagram
:name: fig-sdma-engine
Figure 5: SDMA engine low-level details, abstracted from end-users due to `hipMemcpyAsync` (host initiated) and RCCL-GIN or rocSHMEM (device initiated)
```

## Availability and Reproducibility

This feature is available in RCCL and rocSHMEM from [ROCm 10.1](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html). To use the GPU-initiated SDMA feature, download [RCCL](https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl) or [rocSHMEM](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocshmem) from GitHub and build from source.

Measurements were done by AMD engineers from the collectives team. Experiments were repeated between 5 and 7 times. Since the results were stable across the executions, one set of the results is reported.

### RCCL Configuration

Symmetric memory registration and cuMem (`NCCL_CUMEM_ENABLE=1`) are prerequisites for GIN enablement in RCCL. RCCL currently supports multiple GIN backends, such as proxy, SDMA, and GDA. The SDMA backend can be enabled by using the environment variable `NCCL_GIN_TYPE=7`.

Build command:

```console
install.sh --rocshmem-gin
```

Versions used for results:

```console
RCCL version  : 2.32.3-develop:d20dae4
HIP version   : 7.0.51831-b20d3d7ac
ROCm version  : 7.0.2.2.0-16-9428210
MPI version   : open MPI 4.1.8
```

#### RCCL Hardware Configuration

Single server with eight AMD Instinct MI350X GPUs. Configuration: Quanta server with two AMD EPYC 9655 96-Core processors per server, 3 TB of system memory, BIOS F0TC3A07, CentOS Stream 9.

`rccl-tests` `AlltoAll` command (uses RCCL internal `AlltoAll` kernel using GIN APIs):

```console
mpirun -np 8 --map-by ppr:8:node -x NCCL_IGNORE_CPU_AFFINITY=1 \
  -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_ENV_PLUGIN=none -x NCCL_CUMEM_ENABLE=1 -x NCCL_GIN_TYPE=7 \
  rccl-tests/build/alltoall_perf -b 1 -e 4G -f 2 -g 1 -R 2 -w 100 -n 20
```

`rccl-tests` `Allgather` command (uses `rccl-tests` `Allgather` kernel using GIN APIs):

```console
mpirun -np 8 --map-by ppr:8:node -x NCCL_IGNORE_CPU_AFFINITY=1 \
  -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_ENV_PLUGIN=none -x NCCL_CUMEM_ENABLE=1 -x NCCL_GIN_TYPE=7 \
  rccl-tests/build/all_gather_perf -b 1 -e 4G -f 2 -g 1 -R 2 -D 3 -w 100 -n 20
```

### rocSHMEM configuration

Build command:

```console
./scripts/build_configs/all_backends -DUSE_SDMA=ON
```

Versions used for results:

```console
version      : rocSHMEM_v3.8.0_GDA(MUX:IONIC/BNXT/MLX5)+RO+IPC+SDMA
git Hash     : 1b28700e0c9d6ce2e7e5917bb555181859e5f9da
ROCm version : 10.1.0
```

#### rocSHMEM Hardware Configuration

Single server with eight AMD Instinct MI350X GPUs. Configuration: single Supermicro AS-8126GS-TNMR server, 2x AMD EPYC 9575F 64-Core Processor, 3TB memory (1.5TB per NUMA node), BIOS version 1.8. Ubuntu® 24.04 LTS, ROCm™ 10.1.0.

`AlltoAll` rocSHMEM functional test:

```console
mpiexec -n 8 --map-by numa --bind-to none \
  -x ROCSHMEM_TEST_UUID=1 \
  -x ROCSHMEM_HEAP_SIZE=$((64 * 1024 * 1024 * 1024)) \
  -x ROCSHMEM_SDMA_ENABLED=1 -x ROCSHMEM_SDMA_THRESHOLD=2048 \
  tests/functional_tests/rocshmem_functional_tests -a teamalltoall -w 1 -z 64 -v $((1 * 1024 * 1024 * 1024)) -n 20
```

`Putnbi` rocSHMEM functional test:

```console
mpiexec -n 2 --map-by numa --bind-to none \
  -x ROCSHMEM_TEST_UUID=1 \
  -x ROCSHMEM_HEAP_SIZE=$((64 * 1024 * 1024 * 1024)) \
  -x ROCSHMEM_SDMA_ENABLED=1 -x ROCSHMEM_SDMA_THRESHOLD=1024 \
  tests/functional_tests/rocshmem_functional_tests -a wgputnbi -w 1 -z 64 -v $((1 * 1024 * 1024 * 1024)) -n 20
```

Important parameters for rocSHMEM functional tests:

| Parameter | Description |
| --------- | ----------- |
| `ROCSHMEM_SDMA_ENABLED` | Turn SDMA ON/OFF |
| `ROCSHMEM_SDMA_THRESHOLD` | Controls switch point between IPC/LSA and SDMA |
| `-w` | Number of workgroups (workgroups normally map to CUs) |
| `-z` | Number of threads/workgroup (64 is one wavefront on MI3XX) |

## Summary

This blog explored how GPU-initiated SDMA in RCCL and rocSHMEM enables a new class of fused compute-and-communication kernels where data movement is offloaded entirely to dedicated DMA engines. Performance benchmarks on AMD Instinct™ MI350X hardware demonstrated how this approach frees compute units to focus on computation, eliminates resource contention, and achieves full scale-up bandwidth with as little as a single Compute Unit. Available from ROCm 10.1, this capability is especially valuable for inference workloads where communication falls on the critical path and fine-grained overlap between compute and communication is essential.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED 'AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2026 Advanced Micro Devices, Inc. All rights reserved
