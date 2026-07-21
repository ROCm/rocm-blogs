---
blogpost: true
blog_title: "Efficient MiniMax-M3 Inference on AMD Instinct GPUs with ATOM and ATOMesh"
date: 21 Jul 2026
author: 'Chang Liu, Seungrok Jung, Hattie Wu, Lingpeng Jin, Hongxia Yang'
thumbnail: 'minimax-m3-atomesh-thumbnail.png'
tags: AI/ML, LLM, Serving, Optimization, Performance
category: Applications & models
target_audience: AI infrastructure engineers, LLM serving engineers, and performance optimization practitioners deploying large MoE models on AMD GPUs.
key_value_propositions: Demonstrate efficient single-node and multi-node serving of MiniMax-M3 on AMD Instinct MI355X GPUs using ATOM and ATOMesh, with EAGLE3 speculative decoding and disaggregated inference.
language: English
myst:
    html_meta:
        "author": "Chang Liu, Seungrok Jung, Hattie Wu, Lingpeng Jin, Hongxia Yang"
        "description lang=en": "Serve and benchmark MiniMax-M3 on AMD Instinct MI355X GPUs using ATOM and ATOMesh with EAGLE3 speculative decoding."
        "keywords": "MiniMax-M3, ATOMesh, ATOM, ROCm, AITER, MORI, MoE, LLM inference, multi-node serving, AMD Instinct, MI355X"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Chang Liu, Seungrok Jung, Hattie Wu, Lingpeng Jin, Hongxia Yang"

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

# Efficient MiniMax-M3 Inference on AMD Instinct GPUs with ATOM and ATOMesh

[MiniMax-M3](https://www.minimax.io/models/text/m3) is a natively multimodal Mixture-of-Experts (MoE) foundation model with 428 billion total parameters and 22 billion activated per token. Released in June 2026, M3 is the first open-weight model to combine frontier coding and agentic capabilities, a 1-million-token context window, and native text, image, and video understanding within a single architecture. Its key innovation, MiniMax Sparse Attention (MSA), replaces standard quadratic attention with a block-level KV cache selection mechanism, reducing per-token compute to 1/20th of its predecessor at 1M context.

Serving a model of this scale efficiently requires both optimized single-node execution and scalable multi-node orchestration. In this post, we demonstrate how to serve MiniMax-M3 on AMD Instinct MI355X GPUs using **ATOM** — AMD's ROCm-native inference engine — for single-node serving with EAGLE3 speculative decoding, and **ATOMesh** for multi-node disaggregated prefill/decode inference. We present performance benchmarks on both configurations, provide step-by-step reproduction instructions, and walk through the AMD inference stack.

---

## Performance Results

We evaluated MiniMax-M3 serving performance on AMD Instinct MI355X GPUs using ATOM in a single-node configuration and ATOMesh + ATOM in a multi-node configuration. All benchmarks use ISL/OSL = 8K/1K with concurrency sweeps. Results are from the [SemiAnalysis InferenceX](https://inferencex.semianalysis.com/) open-source benchmark platform, as of June 26, 2026. Performance may vary based on model version, precision, topology, software stack, benchmark settings, workload, and system configuration.

### Single-Node Performance

Figure 1 shows the single-node token throughput per GPU vs. interactivity curve for MiniMax-M3 on AMD Instinct MI355X with ATOM (pink curves) and ATOM with EAGLE3 speculative decoding (red curves), in both FP4 and FP8, compared against NVIDIA B200 with vLLM (light green curves) and vLLM with EAGLE3 speculative decoding in FP8 (dark green curve).

At the high-interactivity end of the curve, the MI355X with ATOM EAGLE3 FP4 and the B200 vLLM EAGLE3 FP8 both reach 340–370 tok/s/user, while the MI355X delivers ~0.6k tok/s/GPU throughput compared to ~0.2k tok/s/GPU for the B200. At the high-throughput end, across the 17–135 tok/s/user interactivity range, the MI355X with ATOM EAGLE3 FP4 achieves 3.6k–8.5k tok/s/GPU, surpassing the B200 at 2.5k–7.7k tok/s/GPU. These results demonstrate strong serving efficiency for MiniMax-M3 on AMD hardware with AITER-optimized kernels.

```{figure} images/minimax-m3-atomesh-singlenode.png
:width: 80%
:align: center
:alt: MiniMax-M3 single-node token throughput per GPU vs. interactivity on AMD Instinct MI355X with ATOM

Figure 1: MiniMax-M3 single-node token throughput per GPU vs. interactivity — AMD MI355X with ATOM (FP4, FP8) and EAGLE3 speculative decoding vs. NVIDIA B200 with vLLM (FP8) and EAGLE3. ISL/OSL = 8K/1K. Source: SemiAnalysis InferenceX™, Jun 26, 2026.
```

### Multi-Node Performance

Figure 2 shows the multi-node token throughput per GPU vs. interactivity curve for MiniMax-M3 on AMD Instinct MI355X with ATOMesh + ATOM in FP4 (red curve, right) and FP8 (red curve, left) using Mooncake for KV cache transfer, compared against NVIDIA B300 with Dynamo vLLM FP8 (green curve).

In the 60–150 tok/s/user interactivity range, the MI355X with ATOMesh FP4 achieves 0.3k–5k tok/s/GPU throughput, exceeding the B300 Dynamo vLLM FP8 at 0.2k–3.9k tok/s/GPU. These multi-node results demonstrate that ATOMesh with Mooncake disaggregated inference on MI355X delivers competitive per-GPU efficiency against NVIDIA's Dynamo vLLM stack.

```{figure} images/minimax-m3-atomesh-multinode.png
:width: 80%
:align: center
:alt: MiniMax-M3 multi-node token throughput per GPU vs. interactivity on AMD Instinct MI355X with ATOMesh

Figure 2: MiniMax-M3 multi-node token throughput per GPU vs. interactivity — AMD MI355X with ATOMesh + ATOM (FP4, FP8) using Mooncake disaggregated inference vs. NVIDIA GB300 NVL72 and B300 with Dynamo vLLM (FP8). ISL/OSL = 8K/1K. Source: SemiAnalysis InferenceX™, Jun 26, 2026.
```

---

## The Optimization Race

One of the most valuable aspects of an open, continuously updated benchmark like [SemiAnalysis InferenceX](https://inferencex.semianalysis.com/) is that it captures not just a single snapshot, but the *evolution* of performance over time. The story of MiniMax-M3 serving over the past month is a great example of how quickly the open-source inference stack and its optimization now move — and this time, AMD has been setting the pace.

**AMD took the lead early, and NVIDIA caught up fast.** Within days of MiniMax-M3's launch, AMD took the lead on Instinct MI355X GPUs with FP4 support in the ATOM inference engine, combining ROCm-native execution, AITER-optimized attention and MoE kernels, and tensor-parallel single-node serving ([InferenceX #1813](https://github.com/SemiAnalysisAI/InferenceX/pull/1813)). As Figure 3 shows (InferenceX snapshot from June 18, 2026), the MI355X with ATOM FP4 sat well above both NVIDIA B300 and B200 with vLLM FP8 across the entire curve.

```{figure} images/minimax-m3-atomesh-story-20260618.png
:width: 80%
:align: center
:alt: MiniMax-M3 performance snapshot June 18, 2026 — AMD MI355X ATOM FP4 leading NVIDIA B300 and B200 vLLM FP8

Figure 3: MiniMax-M3 token throughput per GPU vs. interactivity — AMD MI355X with ATOM (FP4) vs. NVIDIA B300 and B200 with vLLM (FP8). ISL/OSL = 8K/1K. Source: SemiAnalysis InferenceX™, Jun 18, 2026.
```

**AMD kept pushing the frontier forward.** Building on the day-zero baseline, the AMD team layered on several serving optimizations to widen the lead by June 26 (Figure 4, [InferenceX #1917](https://github.com/SemiAnalysisAI/InferenceX/pull/1917)). The biggest lever was **EAGLE3 speculative decoding on ATOM** — pairing the MiniMax-M3 target with the lightweight MiniMax-M3-EAGLE3 draft model lets the engine propose several tokens per step and verify them in a single forward pass, cutting the number of memory-bound decode steps, while chat-template-aligned drafting keeps the acceptance-length distribution close to real-world traffic. This was combined with an **ASM-optimized paged-attention kernel** that accelerates MiniMax Sparse Attention's block-level KV selection on the MI355X, and **AITER INT4 quick-reduce**, which routes the tensor-parallel all-reduce path through a low-precision reduction to trim cross-GPU communication cost during decode. Together with the MXFP4 weight path, these optimizations lifted MI355X throughput across the interactivity curve and kept the ATOM FP4 configuration ahead of the NVIDIA B200 across the plotted InferenceX interactivity range.

```{figure} images/minimax-m3-atomesh-story-20260626.png
:width: 80%
:align: center
:alt: MiniMax-M3 snapshot June 26, 2026 — AMD MI355X with ATOM and EAGLE3 FP4 leading

Figure 4: MiniMax-M3 token throughput per GPU vs. interactivity — AMD MI355X with ATOM and EAGLE3 speculative decoding (FP4) vs. NVIDIA B200 with vLLM (FP8) and EAGLE3. ISL/OSL = 8K/1K. Source: SemiAnalysis InferenceX™, Jun 26, 2026.
```

**AMD keeps optimizing, and the game is still on.** The AMD team has continued to iterate. By July 8 (Figure 5, [InferenceX #2001](https://github.com/SemiAnalysisAI/InferenceX/pull/2001)), a newer ATOM nightly build added **online PTPC-FP8 quantization** — converting the attention and dense MLP layers to per-tensor-per-channel FP8 at load time while leaving the MoE weights untouched — and switched the attention path to the **Triton backend**, extending the MI355X coverage to higher throughput per GPU and higher interactivity across the curve. As of today, the contest remains genuinely close, and it is often hard to declare a definitive winner: the answer depends on precision, topology, interactivity level, and workload.

The notable shift this round is that AMD has been in the driver's seat from the start, leading rather than following on a brand-new frontier model. This kind of competition can help accelerate inference optimization and expand the range of serving options available to developers. That is precisely the outcome the industry has been hoping for.

```{figure} images/minimax-m3-atomesh-story-20260708.png
:width: 80%
:align: center
:alt: MiniMax-M3 performance snapshot July 8, 2026 — AMD MI355X with ATOM extends coverage with PTPC-FP8 and Triton attention

Figure 5: MiniMax-M3 token throughput per GPU vs. interactivity — AMD MI355X with ATOM, EAGLE3, and online PTPC-FP8 quantization with Triton attention backend vs. NVIDIA B200 and B300 with vLLM. ISL/OSL = 8K/1K. Source: SemiAnalysis InferenceX™, Jul 8, 2026.
```

---

## Step-by-Step Reproduction

All benchmarks in this post were produced on AMD Instinct MI355X GPUs using the ATOM inference engine. This section provides the commands for reproducing the single-node and multi-node results.

### Environment Setup

Pull the ATOM Docker image with MiniMax-M3 support:

```bash
docker pull rocm/atom-dev:MiniMax-M3-20260623
```

Launch the container:

```bash
docker run -it --rm --network=host --privileged \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size 16G --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /path/to/models:/models \
  rocm/atom-dev:MiniMax-M3-20260623 /bin/bash
```

### Single-Node: Launching the ATOM Server

The single-node setup runs the ATOM server on 4 MI355X GPUs (TP4). The commands below show the MXFP4 configuration; for MXFP8, replace the model path with `MiniMaxAI/MiniMax-M3-MXFP8`.

**Standard serving:**

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export ATOM_M3_SPARSE_USE_ASM_PA=1

python3 -m atom.entrypoints.openai_server \
  --model amd/MiniMax-M3-MXFP4 \
  --server-port 8000 \
  --trust-remote-code \
  -tp 4 \
  --block-size 128 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 256 \
  --kv_cache_dtype fp8 \
  --no-enable_prefix_caching
```

**With EAGLE3 speculative decoding:** add three flags to enable the [MiniMax-M3-EAGLE3](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3) draft model, which proposes multiple tokens per step for lossless acceleration:

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export ATOM_M3_SPARSE_USE_ASM_PA=1

python3 -m atom.entrypoints.openai_server \
  --model amd/MiniMax-M3-MXFP4 \
  --server-port 8000 \
  --trust-remote-code \
  -tp 4 \
  --block-size 128 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 256 \
  --kv_cache_dtype fp8 \
  --no-enable_prefix_caching \
  --method eagle3 \
  --draft-model Inferact/MiniMax-M3-EAGLE3 \
  --num-speculative-tokens 3
```

### Multi-Node: Launching ATOM with ATOMesh Disaggregation

The multi-node setup uses a **1P1D disaggregated topology**: one prefill worker and one decode worker, each with TP4. The prefill node also hosts the ATOMesh router that coordinates requests between the two workers via Mooncake RDMA KV transfer.

**Prefill server (kv_producer)** — run on the prefill node:

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

python3 -m atom.entrypoints.openai_server \
  --model amd/MiniMax-M3-MXFP4 \
  --host 0.0.0.0 --server-port 8010 \
  --trust-remote-code \
  -tp 4 \
  --block-size 128 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 128 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --no-enable_prefix_caching \
  --kv-transfer-config \
    '{"kv_role":"kv_producer","kv_connector":"mooncake","proxy_ip":"<PREFILL_NODE_IP>","handshake_port":6301}'
```

**Decode server (kv_consumer)** — run on the decode node:

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

python3 -m atom.entrypoints.openai_server \
  --model amd/MiniMax-M3-MXFP4 \
  --host 0.0.0.0 --server-port 8020 \
  --trust-remote-code \
  -tp 4 \
  --block-size 128 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 128 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --no-enable_prefix_caching \
  --kv-transfer-config \
    '{"kv_role":"kv_consumer","kv_connector":"mooncake","proxy_ip":"<DECODE_NODE_IP>","handshake_port":6301}'
```

**ATOMesh Router** — once both servers are healthy, start the router on the prefill node:

```bash
atomesh launch \
  --host 0.0.0.0 --port 8000 \
  --pd-disaggregation \
  --prefill http://<PREFILL_NODE_IP>:8010 \
  --decode http://<DECODE_NODE_IP>:8020 \
  --policy random \
  --backend atom \
  --log-level info \
  --disable-health-check \
  --disable-circuit-breaker \
  --prometheus-port 29100
```

The router exposes OpenAI-compatible APIs at port 8000 and routes requests to the prefill and decode workers using prefill/decode disaggregation. For MXFP8, replace the model path with `MiniMaxAI/MiniMax-M3-MXFP8`.

### Running Benchmarks

The InferenceX benchmarks sweep concurrency levels against the ATOM server (single-node) or ATOMesh router (multi-node). In our runs, ISL and OSL are set to 8K and 1K respectively:

```bash
python3 -m atom.benchmarks.benchmark_serving \
  --model amd/MiniMax-M3-MXFP4 \
  --backend vllm \
  --base-url http://localhost:8000 \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 1024 \
  --random-range-ratio 1.0 \
  --num-prompts 160 \
  --max-concurrency 16 \
  --request-rate inf \
  --ignore-eos \
  --save-result \
  --trust-remote-code \
  --percentile-metrics "ttft,tpot,itl,e2el"
```

For the full command reference and model-specific recipes, see the [ATOM MiniMax-M3 recipe](https://github.com/ROCm/ATOM/blob/main/recipes/MiniMax-M3.md).

---

## MiniMax-M3 and the AMD Serving Stack

### MiniMax-M3 Model Architecture

[MiniMax-M3](https://github.com/MiniMax-AI/MiniMax-M3) is built on a transformer-based Mixture-of-Experts architecture with 428B total parameters and approximately 22B activated per token. Three architectural features define how M3 interacts with the serving stack:

**MiniMax Sparse Attention (MSA).** MSA replaces standard quadratic attention with a KV-block selection mechanism that attends only to the most relevant cache blocks per query. According to the [MiniMax Sparse Attention technical report](https://arxiv.org/abs/2606.13392), compared to the prior generation (M2) at 1M tokens, MSA delivers **9x faster prefill**, **15x faster decoding**, and reduces per-token compute to **1/20th** — without compressing key-values or sacrificing precision.

**Large-scale MoE.** M3 uses **128 total experts** with **4 activated per token**, keeping per-token compute at ~22B parameters while retaining 428B-parameter capacity. At scale, this architecture benefits from expert parallelism (EP) and RDMA-optimized all-to-all communication.

**Native multimodality.** M3 undergoes mixed-modality training from step zero using a ViT-based vision encoder (~600M parameters) for image and video input, achieving deep semantic fusion across text, image, and video.

### The AMD Inference Stack

Serving a 428B-parameter MoE model with million-token context at production quality requires tight integration across every layer of the inference stack. As shown in Figure 6, the AMD inference stack is built from the hardware up through five layers:

```{figure} images/minimax-m3-atomesh-stack.png
:width: 80%
:align: center
:alt: The AMD LLM inference stack, from hardware through ATOMesh distributed orchestration

Figure 6: The AMD LLM inference stack, from hardware through ATOMesh distributed orchestration.
```

**[ATOMesh](https://github.com/ROCm/ATOM)** is the distributed inference gateway that sits above the engine layer, turning a collection of ATOM worker instances into a coordinated, scalable serving system. For large MoE models like MiniMax-M3 that benefit from disaggregated prefill/decode across nodes, ATOMesh provides the cluster-level orchestration. Its key responsibilities include:

- **Prefill/decode disaggregation** — routing prefill and decode phases to separate worker pools, critical for long-context MoE workloads where prefill and decode have vastly different resource profiles.
- **KV-aware scheduling** — selecting workers based on KV cache locality to reduce redundant prefill work and improve time-to-first-token.
- **Multi-engine backend management** — routing to ATOM, vLLM, or SGLang backends through a unified placement core, with ATOM as the primary ROCm-optimized execution path.
- **Worker lifecycle management** — handling registration, warmup, health checks, draining, reconnects, and graceful shutdown without disrupting the scheduling hot path.
- **Observability** — built-in metrics, structured logging, and centralized metric naming for production monitoring.

ATOMesh centralizes all routing decisions in a single placement engine that can assign requests to either a single worker or a prefill/decode worker pair for disaggregated serving. This keeps scheduling consistent regardless of how requests arrive or which backend engine handles them. For the full design details, see the [ATOMesh blog post](https://rocm.blogs.amd.com/software-tools-optimization/atomesh-inference/README.html).

**[ATOM](https://github.com/ROCm/ATOM)** (AiTer Optimized Model) is the ROCm-native inference engine that orchestrates model execution end-to-end. Designed with ROCm-first priorities and AITER-native operators on the inference-critical path, ATOM provides OpenAI-compatible serving APIs, prefill-first continuous batching with KV cache block lifecycle management, composable TP/DP/EP parallelism, graph replay and piecewise compilation for low decode-stage overhead, and quantization-aware execution with FP8 and MXFP4 formats auto-detected from model config. Under the engine layer, **[MORI](https://github.com/ROCm/mori)** provides the modular RDMA and traffic-control stack for EP dispatch/combine and KV transfer, while **[RCCL](https://github.com/ROCm/rccl)** provides collective communication primitives. For a deeper walkthrough, see the [ATOM blog post](https://rocm.blogs.amd.com/software-tools-optimization/atom-inference-engine/README.html).

**[AITER](https://github.com/ROCm/aiter)** (AI Tensor Engine for ROCm) is the high-performance kernel library for inference-critical operators: Flash/Paged Attention for efficient KV cache access, GEMM across FP8/MXFP4/INT8/INT4 formats, fused MoE kernels for expert dispatch and combine, and norm/activation/position-encoding fusions. AITER sources these from Composable Kernel (CK), Triton, and HIP/C++ backends with per-architecture dispatch, ensuring each AMD GPU generation runs the fastest available code path.

**[ROCm](https://rocm.docs.amd.com/)** is AMD's open-source accelerator software platform, providing the runtime, compiler, and core libraries (HIP, RCCL, MIOpen, rocBLAS) that all higher layers build on.

---

## Summary

In this blog, we explored how to serve MiniMax-M3 — a 428B-parameter, 128-expert MoE model with million-token context — on AMD Instinct MI355X GPUs using ATOM and ATOMesh. We saw single-node performance with ATOM FP4 and EAGLE3 speculative decoding outperforming NVIDIA B200 with vLLM, and multi-node performance with ATOMesh disaggregated inference delivering competitive per-GPU efficiency against NVIDIA GB300 NVL72 and B300. We also walked through the full reproduction steps and the AMD inference stack architecture.

Looking ahead, AMD continues to optimize multi-node scaling, longer-context workloads leveraging MSA, and compute-communication overlap. We will publish follow-up results as ATOM and ATOMesh mature. Stay tuned for future ROCm blog posts covering these developments.

---

## Additional Resources

- [MiniMax-M3 Official Page](https://www.minimax.io/models/text/m3)
- [MiniMax-M3 GitHub Repository](https://github.com/MiniMax-AI/MiniMax-M3)
- [MiniMax Sparse Attention (MSA) Technical Report — arXiv:2606.13392](https://arxiv.org/abs/2606.13392)
- [ATOM Repository](https://github.com/ROCm/ATOM)
- [ATOM Recipes](https://github.com/ROCm/ATOM/tree/main/recipes)
- [ATOM Benchmark Dashboard](https://rocm.github.io/ATOM/benchmark-dashboard/#)
- [ATOM Blog: Unlocking Extreme AMD Instinct Inference](https://rocm.blogs.amd.com/software-tools-optimization/atom-inference-engine/README.html)
- [ATOMesh Blog: Unlocking AMD Hardware for Scalable LLM Serving](https://rocm.blogs.amd.com/software-tools-optimization/atomesh-inference/README.html)
- [ROCm AITER](https://github.com/ROCm/aiter)
- [ROCm MORI](https://github.com/ROCm/mori)
- [InferenceX Dashboard](https://inferencex.semianalysis.com/inference)

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED 'AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
© 2026 Advanced Micro Devices, Inc. All rights reserved.
