---
blogpost: true
blog_title: "Sparsity Is Not Speedup: Porting BLASST Block-Sparse Attention to AMD Instinct MI350X and MI355X"
date: TODO
author: Sonali Singh, TODO
thumbnail: 'TODO.png'
tags: AI/ML, GenAI, LLM, Performance, Optimization
category: Software tools & optimizations
target_audience: GPU kernel developers, ML systems developers, and performance engineers
key_value_propositions: Learn how BLASST prefill was ported to the unified attention kernels on AMD Instinct MI350X and MI355X, why data-dependent block skipping exposed chiplet-level load imbalance, and how the fixes turned skipped attention blocks into measured speedups.
language: English
myst:
    html_meta:
        "author": "Sonali Singh, TODO"
        "description lang=en": "Porting BLASST prefill to the unified attention kernels on AMD Instinct MI350X and MI355X: diagnosis of chiplet load imbalance, optimizations, and accuracy/sparsity tradeoffs."
        "keywords": "BLASST, sparse attention, block sparsity, softmax thresholding, FlashAttention, long context, prefill, unified attention, Triton, AITER, ATOM, ROCm, MI350X, MI355X, AMD Instinct"
        "property=og:locale": "en_US"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Software Tools and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Sonali Singh, TODO"
---

<!--
AUTHORING NOTES (remove before publication)
- Section plan follows working/blog_matrix.xlsx.
- Status: Introduction, Motivation, and Background are drafted. All later
  sections are skeletons with OWNER / TODO markers and pointers to the
  evidence that should back them (data/, PRs, profiler output).
- Rules for contributors are in AGENT.md at the repo root. In short: every
  number must trace back to data/, profiler output, a PR, or a cited source;
  keep measurement and interpretation separate; leave a TODO where evidence
  is missing.
- Keep these terms distinct: target sparsity, observed sparsity / tile
  elision, threshold (lambda) and calibration, skipped blocks, accuracy
  impact, kernel speedup, end-to-end speedup.
- Scope of this post: attention prefill only. On the ATOM call path the
  kernels are unified_*. Decode is out of scope.
- Do not add MI350X or MI355X sections. Organize by kernel path. The Gluon
  path comes before the Triton path. The MI350X experiments belong to
  another contributor and have equal weight: they live in the Gluon
  sections, which are marked OWNER. Name the GPU on every number. Do not
  pool the two GPUs' results.
-->

# Sparsity Is Not Speedup: Porting BLASST Block-Sparse Attention to AMD Instinct MI350X and MI355X

Long-context inference is increasingly dominated by attention. For a sequence of length $n$, attention performs $O(n^2)$ work, and even with FlashAttention-style tiling the kernel still computes every query-key interaction. **BLASST** (BLocked Attention Sparsity via Softmax Thresholding) [[1](#references)] is a training-free, drop-in technique that reuses statistics already computed by FlashAttention's online softmax to detect attention blocks whose contribution to the output is negligible, and skips the work for those blocks at run time. The original BLASST work provides optimized CUDA kernels for NVIDIA Hopper and Blackwell GPUs, integrated into TensorRT-LLM and FlashInfer [[1](#references), [2](#references)].

In this blog, we describe our experience bringing BLASST **prefill** to AMD Instinct **MI350X** and **MI355X** on ROCm. The two sets of experiments are reported side by side: the MI350X results are from a colleague, and they carry the same weight as the MI355X results. The kernel work is limited to the attention kernels on the **ATOM** [[4](#references)] call path, the `unified_*` Triton kernels in **AITER** [[3](#references)]. Decode, and attention kernels that ATOM does not call, are outside the scope of this post.

The algorithm itself ports with few changes. Turning skipped blocks into wall-clock speedup did not. A straightforward port of the skip rule left a large part of the expected speedup unrealized, and in some configurations was slower than the dense kernel. Diagnosing why exposed dependencies between the sparse-attention algorithm, the compiler, the kernel's data layout, and the way work is scheduled across the chiplets of the GPU. Most of this post is about those dependencies: how we observed them, how we diagnosed them, and what we changed to fix them.

> **TODO:** Add a one-paragraph summary of the headline results once the numbers in `data/optimized/` and `data/atom/` are finalized: kernel speedup, end-to-end speedup, and accuracy impact, each with its GPU, model, sequence length, observed sparsity, and baseline.

By the end of this post, you will:

- Understand how BLASST decides which attention blocks to skip, and which work a skipped block saves.
- See how the BLASST skip rule is added to ATOM's `unified_*` prefill kernels, with results on MI350X and MI355X.
- Learn why data-dependent block skipping can create chiplet-level load imbalance on a multi-chiplet GPU.
- See the scheduling and compiler changes that recovered the expected speedup.
- See the measured accuracy and sparsity tradeoffs, and how to reproduce them.

> **TODO:** Confirm the list above matches the final section contents.

## Motivation

### Long-context attention is expensive

Recent models support context windows of 128K tokens or more, and workloads such as repository-level code understanding, long-document analysis, and long multi-turn agent sessions use them [[1](#references)]. At these lengths, attention cost grows quadratically with sequence length while the cost of the linear layers grows linearly, so attention becomes an increasingly large fraction of prefill time.

FlashAttention [[5](#references)] and its successors reduce the memory traffic of attention through tiling and kernel fusion, but they still compute the full attention matrix. The quadratic term remains.

> **TODO:** Add a short measurement showing the share of prefill time spent in attention as a function of sequence length for one of our evaluation models (for example, Qwen3-8B). Source: `data/baseline/` or a profiler trace. Record the GPU (MI350X or MI355X), model, sequence lengths, batch size, and software versions.

### Sparse attention, and why adoption is hard

In practice, most of the post-softmax attention mass for a given query is concentrated on a small subset of keys. Sparse-attention methods exploit this by computing only part of the attention matrix. The BLASST authors identify five recurring obstacles to deploying such methods [[1](#references)]:

1. Some methods need an expensive pre-computation pass to choose the sparsity pattern, which can cancel the speedup.
2. Some methods add layers that require fine-tuning, or require training a new architecture.
3. Most methods accelerate only prefill or only decode.
4. Many lack kernels for current GPUs, so it is unclear whether their theoretical savings become real speedups.
5. Many require intrusive changes to model code or attention APIs.

### Why BLASST, and why port it to AMD GPUs

BLASST addresses these obstacles directly. It needs no training and no pre-computation. Its skip decision uses only values the FlashAttention kernel already computes. The paper applies it to both prefill and decode and to the common attention variants (MHA, GQA, MQA, MLA), and its only new input is a single scalar threshold [[1](#references)]. On NVIDIA B200, the authors report a 1.52x prefill speedup over a FlashAttention-3 baseline at 71.9% observed sparsity, with small accuracy differences on RULER and LongBench [[1](#references)]. This post ports the prefill half of that idea. Decode is out of scope.

**TODO:** should we quote NVIDIA numbers?

While porting BLASST to AMD Instinct GPUs, we realized that its speedup is not a property of the algorithm alone. It depends on whether the kernel actually reclaims the time of the skipped work on the target hardware. Questions such as how evenly skipped work is spread across compute units, whether the compiler still generates efficient code when the inner loop becomes conditional, and how the kernel's tile shape affects the number of blocks that can be skipped all have hardware- and implementation-specific answers. The published results were measured on NVIDIA Hopper and Blackwell, and we do not assume they transfer to MI350X or MI355X. This post reports what we measured on those two GPUs, with the GPU named on each result.

## Background

### FlashAttention and online softmax

For a query block {math}`Q_i`, attention computes

```{math}
O_i = \mathrm{softmax}\left(\frac{Q_i K^\top}{\sqrt{d_k}}\right) V .
```

FlashAttention [[5](#references)] avoids materializing the full score matrix by splitting the keys and values into blocks {math}`K_j, V_j` of {math}`B_c` tokens and processing them one at a time. For each query row it keeps a running maximum {math}`m_i^{(j)}`, a running softmax denominator {math}`l_i^{(j)}`, and an unnormalized output accumulator {math}`O_i^{(j)}`. For each key/value block {math}`j` the kernel:

1. Computes the scores {math}`S_{ij} = Q_i K_j^\top` (the first matrix multiply, often called BMM1 or QK).
2. Computes the block-local row maximum {math}`\tilde{m}_i^{(j)} = \mathrm{rowmax}(S_{ij})` and updates the running maximum {math}`m_i^{(j)} = \max(m_i^{(j-1)}, \tilde{m}_i^{(j)})`.
3. Exponentiates, {math}`\tilde{P}_{ij} = \exp(S_{ij} - m_i^{(j)})`, and rescales and updates {math}`l_i` and {math}`O_i`, including the second matrix multiply {math}`\tilde{P}_{ij} V_j` (BMM2 or PV).

After the last block, the output is normalized, {math}`O_i = O_i^{(T_c)} / l_i^{(T_c)}`.

### The BLASST skip rule

BLASST adds one test between steps 2 and 3. If a block's local maximum is far below the running maximum,

```{math}
\tilde{m}_i^{(j)} - m_i^{(j)} < \ln \lambda ,
```

then every score in that block satisfies {math}`\exp(S_{ij} - m_i^{(j)}) < \lambda`, so the block's post-softmax weights are negligible and the kernel skips the rest of the iteration [[1](#references)]. A skipped prefill block avoids:

- the exponentials and row-sum reduction for {math}`\tilde{P}_{ij}`, and
- the {math}`\tilde{P}_{ij} V_j` matrix multiply.

The scores {math}`S_{ij}` for every block are still computed, because the skip decision needs them. Only work after the decision can be saved, so the maximum possible kernel speedup depends on how much of each iteration's cost comes after the test. In the paper's prefill kernels, which are compute-bound, the value block is still loaded, and the savings come from skipping the softmax and the second matrix multiply [[1](#references)]. The paper also describes a decode kernel. This post does not: everything that follows is about prefill.

The skip decision is made per tile, not per element. In a GPU kernel, a tile covers several query rows, and the block is skipped only if all rows in the tile agree. This means the tile shape chosen by a kernel directly affects how many blocks can be skipped at a given threshold.

> **TODO:** Insert the BLASST diagram (planned in `blog_matrix.xlsx`). Ideally one figure from a real model and dataset showing the attention tile grid with skipped and computed tiles. Save the figure under `images/` and the script under `scripts/`.

> **TODO:** Confirm and state how our `unified_*` kernels evaluate the test. The working notes describe the comparison in base-2 (`log2(lambda)`), consistent with kernels that compute softmax with `exp2`. State this precisely, with the scaling of the scores, so readers can map our {math}`\lambda` values to the paper's.

### Threshold calibration

A fixed {math}`\lambda` does not give a fixed sparsity. For the same threshold, longer sequences produce much higher sparsity, because softmax weights are spread over more keys and each one is smaller on average. The BLASST authors find that accuracy degradation is mainly a function of observed sparsity, and that the threshold for a given target sparsity scales inversely with context length {math}`L` [[1](#references)]:

```{math}
\lambda = \frac{a}{L} .
```

Their calibration procedure sweeps candidate thresholds over a calibration set, records the observed sparsity {math}`s` for each, and fits

```{math}
\lambda \cdot L = \alpha \, e^{\beta s} ,
```

so that at inference time the threshold for a target sparsity {math}`S` is {math}`\lambda = \alpha e^{\beta S} / L` [[1](#references)]. The fitted parameters depend on the model and, in our experience, also on the kernel's tile geometry and the GPU.

> **TODO:** Back the last sentence with our data (for example, calibrated parameters for two tile shapes or two GPUs) or soften it. Source: `data/atom/`, `data/quark/`.

### Terminology used in this post

Several quantities related to sparsity are easy to conflate. We use the following terms consistently:

| Term | Meaning in this post |
| --- | --- |
| Threshold ({math}`\lambda`) | The scalar passed to the kernel that sets the skip test. |
| Target sparsity | The sparsity that calibration aims for when choosing {math}`\lambda`. |
| Observed sparsity (tile elision) | The fraction of attention tiles the kernel actually skipped on a given input. |
| Kernel speedup | Dense attention-kernel time divided by BLASST attention-kernel time, for the same inputs. |


> **TODO:** Confirm the exact denominator used for "observed sparsity / tile elision" in our tooling (for example, all tiles versus only tiles not removed by the causal mask), and state it here. The paper's definition should be checked as well before comparing against it.

### AMD Instinct MI350X, MI355X, and the ROCm software stack

The measurements in this post come from two GPUs, **MI350X** and **MI355X**, and the two sets of results are presented with equal weight. The MI350X experiments are a colleague's. Each figure and table names the GPU it was measured on. A result on one GPU is not reported as a result on the other.

> **TODO:** Short description of the MI350X and MI355X features relevant to this work: accelerator complex dies (XCDs, or chiplets), compute units per XCD, the per-XCD L2 cache, and how the hardware distributes workgroups across XCDs. Cite AMD architecture documentation for each fact, and state what the two GPUs share and where they differ. The profiling section relies on the workgroup-to-XCD mapping, so state it here with a citation. Do not describe a behavior measured on one GPU as a property of the other.

The software components involved in this work are:

- **ROCm** [[6](#references)]: the AMD GPU software stack.
- **Triton** [[7](#references)]: the Python-embedded kernel language and compiler. One of our fixes is a change in the Triton compiler.
- **AITER** [[3](#references)]: AMD's library of optimized AI operators. On the ATOM prefill path we use its `unified_*` Triton attention kernels.
- **Gluon** [[8](#references)]: a lower-level, layout-explicit kernel language built on the Triton compiler. The Gluon prefill work is a separate path.
- **ATOM** [[4](#references)]: the inference engine. Its prefill call path selects the `unified_*` kernels. Accuracy and prefill performance measurements go through ATOM.

> **TODO:** Pin software versions or commit hashes for every result in this post (ROCm, Triton, AITER, ATOM, Gluon kernel source), ideally in a single table in the reproduction section. Name the GPU for each.

## Implementing BLASST on AMD GPUs

> **Evidence:** `data/baseline/gluon/`, `data/baseline/triton/aiter/`, Gluon PR, AITER PR, ATOM PR.

Prefill only. The Triton kernels in scope are the `unified_*` kernels on the ATOM call path.

### Gluon attention kernel

> **OWNER:** TODO, colleague who ran the MI350X experiments. Do not rewrite this subsection.
>
> **TODO:** Describe the Gluon prefill implementation, what Gluon's explicit layout control allowed, and how the skip test is coordinated across the workgroup. Name the GPU.

### AITER Triton attention kernels

> **TODO:** Describe where the skip test is inserted in the `unified_*` prefill kernels (the 2D and 3D `unified_attention` variants), and the flag or condition that puts them on the ATOM call path. Explain what one skip decision covers. Working notes say the 2D tile packs several query heads of one KV group into one 128-row tile, so all of those heads must agree before the tile is skipped. Verify against the source before stating it.

### Baseline performance after porting

> **TODO:** Report prefill kernel speedup versus observed sparsity for the Gluon path and the Triton path before any optimization, on random and real inputs. Each plot needs: metric, dense baseline kernel, GPU, model or shape (query heads, KV heads, head dimension), sequence length, threshold, and observed sparsity. Note where speedup falls short of the observed sparsity, and any configuration where BLASST is slower than dense.

## Profiling and diagnosis: why skipping did not become speedup

> **OWNER:** TBD
>
> **Evidence:** `data/profiling/`.

> **TODO:** Present the investigation as observation, then hypothesis, then the measurement that confirms or rejects it. Prefill only. Name the GPU on every trace. Points from the working notes to verify and write up:
>
> - How the launch grid of the `unified_attention` 2D kernel maps workgroups to XCDs, and the consequence that all workgroups of one KV head land on the same XCD when there are 8 KV heads. Include the profiler or instrumentation evidence for this mapping.
> - Why this is harmless for dense attention (every head costs about the same) but creates imbalance under BLASST (observed sparsity varies by head, so the XCD holding the least-sparse head finishes last).
> - The effect of causal masking: later query blocks do more work, and the default launch order issues them last.
> - Use a diagram based on a real model and dataset (planned in `blog_matrix.xlsx`).

## Optimizations: Gluon path

> **OWNER:** TODO, colleague who ran the MI350X experiments. Do not rewrite this section.
>
> **Evidence:** `data/optimized/gluon/`, Gluon PR.

Prefill only. Name the GPU on every measurement.

### Fixing chiplet imbalance with atomic tickets and a persistent kernel

> **TODO:** Describe the Gluon scheduling fix. If this path keeps a persistent kernel while the Triton path does not, explain what differs between the two kernels that leads to different choices.

### Other optimizations

> **TODO:** List and describe any other Gluon-specific optimizations, or remove this subsection.

## Optimizations: Triton path

> **OWNER:** Sonali Singh
>
> **Evidence:** `data/optimized/triton/`, AITER PR, Triton PR.

Prefill only, in the `unified_*` kernels. Name the GPU on every measurement.

### Fixing chiplet imbalance with atomic-ticket scheduling and descending query-block order

> **TODO:** Describe the atomic-ticket work assignment, the descending query-block (longest-job-first) ordering, and why the two are needed together. Report the dense-path cost (working notes say these are compile-time options enabled only with block skipping, so the dense path is unchanged; verify). Mention the persistent-kernel alternative that was evaluated and not adopted, with the measurement that decided it.

### Triton compiler optimization

> **TODO:** Describe the code-generation problem caused by placing the PV matrix multiply inside a conditional, the compiler change that fixed it, how it is enabled, and its measured effect. Include the cases where it had no measurable effect, with configurations.

### Kernel-driven anomalies

> **TODO:** Describe any remaining anomalies observed in the `unified_*` kernels and how they were resolved or explained.

## Performance results

Prefill only. Report the Gluon path first, then the Triton path. Every number names its GPU, model, sequence length, threshold, observed sparsity, and dense baseline.

### Gluon path

> **OWNER:** TODO, colleague who ran the MI350X experiments. Do not rewrite this subsection.
>
> **TODO:** Results and reproduction steps. Evidence: `data/optimized/gluon/`.

### Triton path

> **TODO:** Results and reproduction steps. Evidence: `data/optimized/triton/`. Report dense, BLASST, and BLASST with the scheduling fixes against the same dense baseline. Also report the dense kernel with the scheduling fixes alone, so the scheduling gain is visible without BLASST.

## Accuracy and calibration results

> **OWNER:** TBD

### ATOM path

> **TODO:** Prefill accuracy (for example, RULER at 32K) with dense and BLASST at calibrated target sparsities, observed sparsity, and prefill speedup. Report MI350X and MI355X separately, with equal weight. Evidence: `data/atom/`, ATOM PR. State that BLASST is enabled only on the `unified_*` prefill path.

### Quark path

> **TODO:** Include this subsection only if the Quark runs call the same `unified_*` prefill kernels. Otherwise delete it. Evidence: `data/quark/`. Name the GPU on every number.

## Additional insights and future directions

> **OWNER:** TBD

> **TODO:** Layer-dependent switching of the BLASST mechanism, and grid search for the best thresholds for each kernel variant. Other practical lessons and open questions.

## Summary

> **TODO:** Write after the technical sections are complete.

## References

1. [J. Yuan, C. Shinn, et al. *BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding.* MLSys 2026. arXiv:2512.12087.](https://arxiv.org/abs/2512.12087)
2. [NVIDIA Technical Blog. *Accelerating Long-Context Inference with Skip Softmax in NVIDIA TensorRT-LLM.*](https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/)
3. [AITER: AI Tensor Engine for ROCm.](https://github.com/ROCm/aiter)
4. [ATOM.](https://github.com/ROCm/ATOM) TODO: verify link.
5. [T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.](https://arxiv.org/abs/2205.14135)
6. [ROCm documentation.](https://rocm.docs.amd.com)
7. [Triton.](https://github.com/triton-lang/triton)
8. TODO: Gluon reference (Triton repository / documentation link).

> **TODO:** Add links to the Gluon, AITER, ATOM, and Triton pull requests once their final numbers and status are confirmed.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED "AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

AMD, the AMD Arrow logo, AMD Instinct, ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
