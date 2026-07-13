---
blogpost: true
blog_title: "QuickReduce INT3 Quantization and Benchmarking on MI355"
date: 13 Jul 2026
author: 'Haoyang Li, Wei Luo, Xinjun Niu, Spandan Tiwari, Ke Wang, Ashish Sirasao'
thumbnail: 'quickreduce-int3-mi355-thumbnail.jpg'
tags: Systems, Performance, AI/ML, LLM
category: Applications & models
target_audience: AI DEVELOPERS AND ENTHUSIAST
key_value_propositions: An Efficient AllReduce Communication Method for AMD GPUs
language: English
myst:
    html_meta:
        "author": "Haoyang Li, Wei Luo, Xinjun Niu, Spandan Tiwari, Ke Wang, Ashish Sirasao"
        "description lang=en": "Learn how QuickReduce uses INT3 quantization to accelerate all-reduce communication and evaluate its performance and accuracy on AMD Instinct MI355 GPUs."
        "keywords": "MI355, QuickReduce, INT3, AllReduce"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
        "amd_blog_authors": "Haoyang Li, Wei Luo, Xinjun Niu, Spandan Tiwari, Ke Wang, Ashish Sirasao"
---

# QuickReduce INT3 Quantization and Benchmarking on MI355

Large Language Models (LLMs) typically contain billions — or even tens of billions — of parameters. During inference, tensor parallelism (TP) is a widely used technique that distributes the compute across multiple GPUs. This approach, however, requires frequent, large-scale data synchronization between layers, introducing significant communication latency and placing enormous pressure on interconnect bandwidth.

Among the various communication patterns, **all-reduce** stands out as one of the most critical. It aggregates data from all participating devices (for example, via summation) and broadcasts the result back to each device, enabling synchronized multi-GPU computation.

[QuickReduce](https://github.com/mk1-project/quickreduce/) is a high-performance all-reduce library designed for AMD ROCm that supports **inline compression**: the send path quantizes the data, the reduction is performed in the compressed domain, and the receive path dequantizes the result. Compared to RCCL (AMD's collective communication primitives for multi-GPU and multi-node communication), QuickReduce achieves up to **2.25x speedup** on 2×MI300X and 4×MI300X configurations, and outperforms RCCL across all multi-GPU (single-node) configurations when optimized, as reported in [our previous work](https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html).

In [our previous work](https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce/README.html), we discussed the design principles and performance characteristics of QuickReduce in detail, integrated it into the popular inference frameworks [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang), and provided comprehensive performance and accuracy benchmarks for integer quantization (INT8/INT6/INT4) on MI300.

In a [subsequent post](https://rocm.blogs.amd.com/artificial-intelligence/quick-reduce-2/README.html), we further introduced a hardware-accelerated FP4 quantization path on MI355 and presented updated performance and accuracy evaluations, showing that on MI355 FP4 and INT4 are essentially interchangeable.

In this blog post, we go one step further and implement the **INT3** quantization mode of QuickReduce. INT3 trades a higher compression ratio for a smaller on-wire transfer volume. We describe its design and vLLM integration, and demonstrate its latency advantage over INT4 and its model-dependent accuracy on AMD Instinct MI355.

## QuickReduce with INT3: Design and Implementation

### Design Highlights

INT3 reuses the same "block-level FP16 scale + symmetric quantization" framework as INT4, with the key difference that each value is represented in just **3 bits**:

- Each quantization block covers 32 FP16/BF16 values (8 threads × 4 `fp16x2` lanes) that share one FP16 scale.
- Values are scaled, clipped to the symmetric range [-4, +3], rounded to integers, and biased to the unsigned domain [0, 7] for packing.
- Unlike INT4 (eight 4-bit values packed into one `int32`), INT3 splits each 3-bit value into a low and a high part:
  - low 2 bits → one `uint16` per thread (denoted `q2w`)
  - high 1 bit → one `uint8` per thread (denoted `q1w`)
- Each group of 32 values shares one decoding scale, written by the thread-group leader.

This packing shrinks the per-rank transmitted tile from 1152 bytes (INT4) to **896 bytes**, a reduction of about **22%**.
The smaller on-wire data directly reduces the communication volume of the two-shot QuickReduce kernel in Phase-1/Phase-2;
this benefit only takes effect once the message length exceeds the vLLM activation threshold.

### Quantization and Dequantization Paths

The send path is implemented in `CodecQ3::send` (`csrc/quickreduce/quick_reduce_impl.cuh`): it first computes the absolute maximum within the thread group and derives the FP16 encode/decode scale; it then scales and clips the values to [-4, +3], rounds them, and biases them to [0, 7]; finally it splits each 3-bit value into `q2w` and `q1w`, writes them into the communication buffer, and the group leader writes the block scale.

The receive path is implemented in `CodecQ3::recv`: it reads `q2w`, `q1w`, and the scale from the buffer using non-temporal loads, recombines the low 2 bits and high 1 bit into an unsigned quantized value, maps it back to the symmetric integer domain, and multiplies by the decoding scale to reconstruct FP16/BF16. The two-shot kernel calls `recv` during Phase-1B (in-segment reduction) and Phase-2 (gather), consistent with the other QuickReduce quantization modes.

## QuickReduce INT3 Benchmarking on MI355

### Kernel Performance

All experiments in this post were run on a **single-node** MI355 server (8× AMD Instinct MI355X GPUs in one node), so every TP configuration below uses GPUs within that single node (for the full software and system configuration, see the [Appendix](#appendix-configuration-details)). On this server we benchmark three all-reduce implementations:

- **RCCL**
- **Custom AllReduce (CR)** — A widely adopted all-reduce acceleration kernel used in vLLM and SGLang, which significantly outperforms RCCL at low data volumes.
- **QuickReduce (QR)** — This post focuses on comparing the **INT3** and **INT4** quantization configurations.

The tests cover message sizes ranging from **4 KB to 1 GB**, and all latency values are reported in **μs (microseconds)**.

To better visualize relative performance across the full message-size range, we plot **speedup over RCCL** rather than raw latency. The y-axis is defined as **RCCL latency ÷ each method's latency**, so a value greater than 1 indicates better performance than RCCL.

#### TP = 2

The figure below shows how the TP=2 speedup over RCCL varies with message size, highlighting where QuickReduce INT3 and INT4 overtake CR.

```{figure} images/int3-tp2.jpg
:alt: Speedup over RCCL on MI355 at TP=2
:align: center

Speedup over RCCL on MI355 at TP=2. The y-axis is computed as RCCL latency divided by the measured latency of each method.
```

#### TP > 2

Based on our kernel experiments, INT3 and INT4 perform very similarly under TP=4 and TP=8.
We believe that as TP grows, the number of GPUs that must communicate with one another increases multiplicatively, requiring more synchronization and scheduling time;
in this regime, synchronization and latency overheads — rather than the transferred data volume — increasingly become the bottleneck, so the transfer time saved by further bit-width compression (INT3) becomes a smaller fraction of the total and the benefit is diluted.

Because INT3 compresses the bit-width more aggressively and is more sensitive to accuracy, taking both performance and accuracy into account, we do not recommend using INT3 outside of TP=2.

### Key Observations

From the results, we draw the following conclusions:

1. **For message sizes beyond the crossover point, QuickReduce delivers the highest large-message speedup over RCCL**: at a 1 GB message size, QR INT3 reaches roughly **5x speedup** at TP=2.
2. **Compared to INT4, INT3 only shows a clear additional speedup at large communication volumes** (above ~4 MB); for smaller messages the two are essentially on par.

### End-to-End Performance and Accuracy

We selected five representative models spanning both Dense (Qwen3-32B, Qwen2.5-72B-Instruct) and MoE (DeepSeek-V4-Flash, MiniMax-M2.5, Qwen3-235B-A22B) architectures, and used vLLM for both performance and accuracy evaluation.
As with INT4, we set a **1 MB activation threshold** for INT3: even with QR INT3 enabled, the QR INT3 communication path is only used when a single communication exceeds 1 MB;
below 1 MB, the framework falls back to RCCL or CR.

**Server launch command:**

```bash
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT3 vllm serve <model_path> \
    --no-enable-prefix-caching \
    --trust-remote-code \
    -tp 2 \
    --dtype auto \
    --port 12340
```

**Client commands:**

```bash
# Accuracy evaluation
python3 vllm/tests/evals/gsm8k/gsm8k_eval.py --port 12340

# Performance benchmark
vllm bench serve \
    --model <model_path> \
    --backend vllm \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 10 \
    --num-prompts 500 \
    --ignore-eos \
    --request-rate inf \
    --max-concurrency 128 \
    --port 12340
```

All end-to-end results below were collected at **TP=2** under a throughput-oriented load (4096-token random inputs, `--request-rate inf`, `--max-concurrency 128`). For each model we compare the no-quantization baseline (NONE), INT4, and INT3. Speedup and throughput gain are normalized to the NONE baseline, and **Accuracy Recovery** is GSM8K accuracy relative to the NONE baseline.

#### DeepSeek-V4-Flash (MoE)

| Quantization | TTFT (ms) | TPOT (ms) | Output Throughput (tok/s) | TTFT Speedup | TPOT Speedup | Throughput Gain | GSM8K  | Accuracy Recovery |
|:------------:|----------:|----------:|--------------------------:|-------------:|-------------:|----------------:|-------:|------------------:|
| NONE         |  26226.98 |    519.49 |                     37.50 |        1.000 |        1.000 |           1.000 | 0.9447 |            1.0000 |
| INT4         |  24329.79 |    479.63 |                     40.55 |        1.078 |        1.083 |           1.081 | 0.9443 |            0.9996 |
| **INT3**     |**23794.17**|**465.70**|                **41.48**  |    **1.102** |    **1.116** |       **1.106** | 0.9373 |            0.9922 |

#### Qwen3-32B (Dense)

| Quantization | TTFT (ms) | TPOT (ms) | Output Throughput (tok/s) | TTFT Speedup | TPOT Speedup | Throughput Gain | GSM8K  | Accuracy Recovery |
|:------------:|----------:|----------:|--------------------------:|-------------:|-------------:|----------------:|-------:|------------------:|
| NONE         |  25671.79 |    513.31 |                     37.94 |        1.000 |        1.000 |           1.000 | 0.6200 |            1.0000 |
| INT4         |  19903.11 |    395.44 |                     48.83 |        1.290 |        1.298 |           1.287 | 0.6670 |            1.0758 |
| **INT3**     |**18962.92**|**377.71**|                **51.20**  |    **1.354** |    **1.359** |       **1.349** | 0.5867 |            0.9462 |

#### Qwen2.5-72B-Instruct (Dense)

| Quantization | TTFT (ms)  | TPOT (ms) | Output Throughput (tok/s) | TTFT Speedup | TPOT Speedup | Throughput Gain | GSM8K  | Accuracy Recovery |
|:------------:|-----------:|----------:|--------------------------:|-------------:|-------------:|----------------:|-------:|------------------:|
| NONE         |   48670.30 |    971.36 |                     20.06 |        1.000 |        1.000 |           1.000 | 0.7120 |            1.0000 |
| INT4         |   35127.15 |    698.54 |                     27.69 |        1.386 |        1.391 |           1.380 | 0.7290 |            1.0239 |
| **INT3**     |**33363.36**|**664.74** |                **29.11**  |    **1.459** |    **1.461** |       **1.451** | 0.7207 |            1.0122 |

#### MiniMax-M2.5 (MoE)

| Quantization | TTFT (ms) | TPOT (ms) | Output Throughput (tok/s) | TTFT Speedup | TPOT Speedup | Throughput Gain | GSM8K  | Accuracy Recovery |
|:------------:|----------:|----------:|--------------------------:|-------------:|-------------:|----------------:|-------:|------------------:|
| NONE         |  18668.83 |    372.06 |                     52.06 |        1.000 |        1.000 |           1.000 | 0.9317 |            1.0000 |
| INT4         |  15575.94 |    304.56 |                     62.67 |        1.199 |        1.222 |           1.204 | 0.9307 |            0.9989 |
| **INT3**     |**15121.72**|**295.86**|                **64.54**  |    **1.235** |    **1.258** |       **1.240** | 0.9273 |            0.9953 |

#### Qwen3-235B-A22B (MoE)

| Quantization | TTFT (ms) | TPOT (ms) | Output Throughput (tok/s) | TTFT Speedup | TPOT Speedup | Throughput Gain | GSM8K  | Accuracy Recovery |
|:------------:|----------:|----------:|--------------------------:|-------------:|-------------:|----------------:|-------:|------------------:|
| NONE         |  33496.04 |    673.56 |                     28.88 |        1.000 |        1.000 |           1.000 | 0.9067 |            1.0000 |
| INT4         |  26412.86 |    531.00 |                     36.50 |        1.268 |        1.268 |           1.264 | 0.9057 |            0.9989 |
| **INT3**     |**25501.79**|**512.93**|                **37.75**  |    **1.313** |    **1.313** |       **1.307** | 0.9117 |            1.0055 |

### Analysis

On performance, QR INT3 provides a further speedup over QR INT4 because it puts less data on the wire. Across the five models, relative to INT4, INT3 reduces TTFT by about **3.6%** on average (up to ~5% on Qwen2.5-72B-Instruct), reduces TPOT by about **3.7%**, and improves output throughput by about **3.7%**. The TTFT and TPOT gains are comparable here: under this throughput-oriented load (many concurrent requests), the decode phase also accumulates substantial all-reduce traffic above the activation threshold, so it benefits from the smaller INT3 payload as well — not just the prefill phase. These gains come on top of the speedup INT4 already delivers over the no-quantization baseline.

On accuracy, INT3 preserves accuracy well across the MoE models and the stronger dense model; the only model with a clearly visible drop is Qwen3-32B. Our analysis is as follows:

On one hand, compared to INT4, INT3 has fewer quantization levels (8 vs 16) and therefore injects more quantization noise. The three MoE models (DeepSeek-V4-Flash, MiniMax-M2.5, and Qwen3-235B-A22B) are the most robust — each retains at least 99% of its baseline accuracy (Accuracy Recovery 0.9922, 0.9953, and 1.0055, respectively). One likely explanation is that, for MoE models, the output is a weighted sum of multiple experts; combined with their larger parameter redundancy and hidden dimensions, this may further dilute and average out the quantization noise during computation, making them less sensitive.

On the other hand, the determining factor is the strength of the baseline rather than the Dense-vs-MoE distinction. Qwen3-32B is the only model with a notable drop (Accuracy Recovery 0.9462), and it already has a low GSM8K baseline (0.62), indicating limited inherent capability: many problems sit near the decision boundary, so the extra noise introduced by INT3 can easily flip the model's answers. By contrast, the other dense model, Qwen2.5-72B-Instruct, has a stronger baseline (0.712) and is essentially unaffected (Accuracy Recovery 1.0122). For models that are inherently strong and have high baseline scores, most problems are answered with high confidence, and 3-bit noise is not enough to change the correct outcome.

## Summary

In this blog post, we benchmarked the INT3 quantization configuration of QuickReduce on MI355 for both performance and accuracy.

- From the kernel tests: under TP=2, INT3 is clearly faster than INT4 starting from about 4 MB, and reaches roughly 5x speedup over RCCL on 1 GB large messages; however, under TP=4 / TP=8, INT3 and INT4 converge.
- From the end-to-end tests (TP=2, throughput-oriented load): compared to INT4, INT3 reduces TTFT by about 3.6% and TPOT by about 3.7% on average, while improving output throughput by about 3.7%.
- On accuracy, based on the models evaluated in this post, INT3 performed best on the tested MoE models and on models with stronger baseline accuracy; on models with weaker baselines (such as Qwen3-32B in our tests), the added 3-bit noise can flip near-boundary answers.

## Appendix - Configuration Details

For full reproducibility, the exact software stack and system configuration used for all benchmarks in this post are listed below.

Results were obtained using a system configured with AMD Instinct™ MI355X GPUs. Tests for vLLM were conducted by AMD on June 10, 2026. Actual results may vary depending on system configuration, usage, software version, and applied optimizations.

### Software Setup for vLLM

```text
HIPBLASLT_BRANCH: 1.2.2.70203-90~22.04 (libhipblaslt.so.1.2.70203)
TRITON_BRANCH: 3.6.0
TRITON_REPO: https://github.com/triton-lang/triton.git
PYTORCH_BRANCH: 2.10.0+git8514f05
PYTORCH_REPO: https://github.com/ROCm/pytorch.git (commit 8514f05131610dab50233027b2fab9c01235081b)
PYTORCH_VISION_BRANCH: 0.24.1+d801a34
PYTORCH_VISION_REPO: https://github.com/pytorch/vision.git
AITER_BRANCH: 0.1.13 (amd_aiter)
AITER_REPO: https://github.com/ROCm/aiter.git
VLLM_BRANCH: 0.20.2rc1.dev713+ge380238bd.d20260609
VLLM_REPO: https://github.com/vllm-project/vllm
ROCm: 7.2.3 (HIP runtime 7.2.53211-c2d9476115)

```

### System Configuration

```text
AMD Instinct MI355X platform (gfx950)
CPU: 2x AMD EPYC 9575F 64-Core Processor (256 logical CPUs)
NUMA: 2 NUMA node(s), 1 NUMA node per socket
    node0 CPUs: 0-63,128-191
    node1 CPUs: 64-127,192-255
Memory: ~3.0 TiB total
Disk: 1x Micron 7450 14 TB NVMe + 2x Micron 7450 3.5 TB NVMe
GPU: 8x AMD Instinct MI355X (gfx950, 256 CUs each, ~288 GiB HBM each — VRAM total 309 GB/GPU reported)
Host OS: Ubuntu 22.04.5 LTS (Jammy)
Host Kernel: 5.15.0-144-generic
Host GPU Driver (amdgpu / KFD): 6.14.14
```

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED 'AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, ROCm, Instinct, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
