---
blogpost: true
blog_title: "Implementing a High-Performance Custom Diffusion Attention Kernel with FlyDSL"
date: "17 Sep 2026"
author: "Lorri Rao, Kyle Zhao, Yuankai Chen, Kailash Gogineni, Gopal Muthukrishnan, Yao Fu, Zhenyu Gu, Jinzhao Wang, Rishi Iyer"
thumbnail: 'flydsl-customized-attention-thumbnail.png'
tags: "Reinforcement Learning, Partner Applications, AI/ML, LLM, Optimization, Performance, PyTorch, Serving, Diffusion Model"
category: "Software tools & optimizations"
target_audience: "AI Customers"
key_value_propositions: "This post demonstrates FlyDSL's flexible support for custom, high-performance GPU diffusion attention kernels"
language: English
myst:
    html_meta:
        "author": "Lorri Rao, Kyle Zhao, Yuankai Chen, Kailash Gogineni, Gopal Muthukrishnan, Yao Fu, Zhenyu Gu, Jinzhao Wang, Rishi Iyer"
        "description lang=en": "Learn how to implement and optimize flexible, high-performance diffusion attention kernels with FlyDSL."
        "keywords": "FlyDSL, diffusion attention, GPU kernels, AMD Instinct GPUs, ROCm, performance optimization"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Lorri Rao, Kyle Zhao, Yuankai Chen, Kailash Gogineni, Gopal Muthukrishnan, Yao Fu, Zhenyu Gu, Jinzhao Wang, Rishi Iyer"
---

# Implementing a High-Performance Custom Diffusion Attention Kernel with FlyDSL

Readers may be familiar with traditional Transformer models and their attention mechanisms. The traditional autoregressive transformers generate tokens iteratively. Since this feature significantly limits the inference throughput, researchers have begun exploring approaches such as diffusion models that can generate multiple tokens in each iteration.

Diffusion models present unique challenges for kernel implementation, for the following reasons:

1. **Flexible KV-cache layouts.** Our customers want kernels that work with vLLM in TiDAR mode (Think in Diffusion, Talk in Autoregression). This requires support for paged KV caches and scratch storage for speculatively generated tokens.
2. **Configurability.** Customers are exploring different settings to optimize real-world workload performance, so the kernel must be highly configurable.

3. **High performance yet flexible.** Although FlexAttention offers considerable flexibility, it can have performance limitations. We aim to deliver optimizations across the stack, including FlashAttention and split-K at the algorithm and dataflow levels, as well as register-usage and data-movement optimizations at the low-level hardware layer.

This is where FlyDSL can help. FlyDSL addresses these needs by exposing low-level hardware details that kernel developers can optimize while retaining the flexibility required for customization.

As AI coding agents are increasingly used to implement GPU kernels, this post presents an step-by-step workflow with rich references. Developers can use this post to guide an agent for implementing or optimizing kernels. We also tested this workflow by giving an LLM agent our customer requirements and guide it as listed below. The resulting implementation performed well.

## Optimizing Attention With FlyDSL, Step by Step

### Learn the FlyDSL Basics

The FlyDSL repository provides a comprehensive starting guide:

* [FlyDSL documentation](https://github.com/ROCm/FlyDSL/tree/main/docs)
* [Webpage version](https://rocm.github.io/FlyDSL/)

### Choose a Starting Point

AMD's public repositories provide several attention kernel implementations. Review these examples and reuse relevant code when possible. In addition to the examples in the [FlyDSL repository](https://github.com/ROCm/FlyDSL/tree/main/examples), FlyDSL kernels are available in:

* [aiter](https://github.com/ROCm/aiter/tree/main/aiter/ops/flydsl)
* [Primus Turbo](https://github.com/AMD-AGI/Primus-Turbo/tree/main/primus_turbo/flydsl)

### Understand Architectural Differences Across AMD GPU Generations

The examples target different GPU architectures. It is important to understand how hardware differences affect kernel implementations. Key considerations include:

* **LDS buffer size.** MI350 and MI355 GPUs (gfx950/CDNA 4) provide more LDS capacity per compute unit (CU) than MI300 and MI325 GPUs (gfx942/CDNA 3), enabling better data prefetching and pipelining.
* **Transpose-load instructions.** MI350 and MI355 GPUs provide specialized instructions that transpose data while loading it from LDS into vector general-purpose registers (VGPRs). MI300 and MI325 GPUs require explicit transposition.
* **MFMA instructions.** MI350 and MI355 GPUs introduce 16x16x32 and 32x32x16 MFMA instructions (`mfma_f32_16x16x32_f16` and `mfma_f32_32x32x16_f16`). These provide higher throughput than the previous 16x16x16 and 32x32x8 variants.

### Debug Systematically

* **Use a Triton baseline.** End users often prototype in Triton before moving to FlyDSL for better performance. A Triton implementation therefore provides a useful correctness baseline. Compare intermediate results at steps such as the log-sum-exp (LSE) calculation and before and after register operations such as permutations and XOR reductions.
* **Start simple and add one feature at a time.** Begin with a straightforward implementation, then add features such as paged-attention support, the split-K algorithm, and data pipelining.

### Optimize Performance

#### Match MFMA Operands to the Desired Fragment Layout

For fused attention, MFMA operand order determines how score and probability fragments are distributed across lanes. The mathematical QK GEMM is `A = Q`, `B = K^T`, and `D = QK^T`, where M is the query dimension and N is the token dimension. However, logical row-major contiguity is not the same as per-lane register contiguity.

For gfx942's `V_MFMA_F32_16X16X16_BF16`, refer to the general output layout in section 7.1.4 of the [AMD Instinct MI300 CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf). Use AMD's [Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator/tree/2ef91896bcdc4d26624f952e5c905c787cd9bc9e) to inspect the mapping:

```bash
./matrix_calculator.py \
  -a gfx942 \
  -i v_mfma_f32_16x16x16_bf16 \
  --matrix-layout --D-matrix
```

Consequently, one lane fixes the N coordinate `j` and stores four consecutive M rows. As shown in the output:

```text
lane 0:  v0=D[0][0]  v1=D[1][0]  v2=D[2][0]  v3=D[3][0]
lane 16: v0=D[4][0]  v1=D[5][0]  v2=D[6][0]  v3=D[7][0]
```

On gfx942, using K as operand A and Q as operand B produces score fragments in the layout required by the subsequent P×V MFMA. This operand-swapped QK formulation allows the probability fragment to feed P×V directly, eliminating the probability LDS transpose, its synchronization barrier, and the associated LDS traffic. This optimization is illustrated in the figure below.

```{figure} ./mfma_operand_layout.svg
:align: center
:width: 100%
:alt: Conventional and operand-swapped MFMA workflows
```

The AMD FlyDSL FlashAttention kernel demonstrates the same layout. The QK loop invokes [`mfma_acc(k_pack, q_pack, accumulator)`](https://github.com/ROCm/FlyDSL/blob/b8ed73fe6d9e17e101093324b1a7af518b1a0f29/kernels/attention/flash_attn_generic.py#L415-L416), placing K before Q. Its P×V invokes [`mfma_acc(v_transposed_pack, p_pack, accumulator)`](https://github.com/ROCm/FlyDSL/blob/b8ed73fe6d9e17e101093324b1a7af518b1a0f29/kernels/attention/flash_attn_utils.py#L2809-L2810).

#### Analyze Register Usage

To dump the generated assembly, set `FLYDSL_RUNTIME_ENABLE_CACHE=0` to avoid reusing stale cache entries and set `FLYDSL_DUMP_IR=1`. You can also set `FLYDSL_DUMP_DIR=/tmp/xx` to select the output directory; the default is `/root/.flydsl/debug/`. Because FlyDSL uses just-in-time (JIT) compilation, run the kernel at least once to generate the output.

Relevant fields in `<DUMP_DIR>/kernel_<name>_0/21_final_isa.s` include:

* `.set kernel.num_vgpr` / `.vgpr_count` and `num_agpr` / `.agpr_count`
* `accum_offset` and `next_free_vgpr`
* `group_segment_fixed_size`: LDS size
* `.vgpr_spill_count`: memory spills

#### Analyze Performance

Use `rocprofv3` to collect runtime statistics. The following hardware counters can help identify bottlenecks:

| Purpose | Counters |
| --- | --- |
| L2 cache hits and coalescing | `TCC_HIT_sum`, `TCC_MISS_sum`, `TCC_REQ_sum` |
| HBM efficiency and traffic distribution | `TCC_EA0_RDREQ_sum`, `TCC_EA0_RDREQ_32B_sum`, `TCC_EA0_RDREQ_DRAM_sum`, `TCP_TCC_READ_REQ_sum` |
| MFMA utilization | `SQ_INSTS_MFMA`, `SQ_INSTS_VALU_MFMA_MOPS_*`, `MfmaUtil` (MFMA busy percentage) |
| VMEM and LDS activity | `SQ_INSTS_VMEM_*`, `SQ_INSTS_LDS`, `SQ_LDS_BANK_CONFLICT`, `LDSBankConflict` |

See these guides for detailed collection and analysis instructions:

* [Capture cache and HBM counters in PMC mode](https://github.com/ROCm/FlyDSL/blob/421935cc6f09fd9b27d5d5ae52e0960e18834bd5/.claude/skills/capture-kernel-trace/SKILL.md#pmc-mode-cache--hbm-counter-capture-separate-from-att)
* [Perform line-level stall analysis with ATT](https://github.com/ROCm/FlyDSL/blob/421935cc6f09fd9b27d5d5ae52e0960e18834bd5/.claude/skills/capture-kernel-trace/SKILL.md#step-4-run-rocprofv3-with-att)
* [Analyze line-specific stalls related to LDS optimization](https://github.com/ROCm/FlyDSL/blob/421935cc6f09fd9b27d5d5ae52e0960e18834bd5/.claude/skills/lds-optimization/SKILL.md?plain=1#L118)

#### Address LDS Bank Conflicts

See the [GEMM optimization guide's section on LDS bank conflicts](https://github.com/ROCm/FlyDSL/blob/421935cc6f09fd9b27d5d5ae52e0960e18834bd5/.claude/skills/gemm-optimization/SKILL.md?plain=1#L181).

#### Prefetch Data

See the [prefetch data-load guide](https://github.com/ROCm/FlyDSL/blob/421935cc6f09fd9b27d5d5ae52e0960e18834bd5/.claude/skills/prefetch-data-load/SKILL.md).

For more data-movement optimizations, see the [LDS optimization guide](https://github.com/ROCm/FlyDSL/blob/421935cc6f09fd9b27d5d5ae52e0960e18834bd5/.claude/skills/lds-optimization/SKILL.md).

## Integration Notes

### Graph Capture

Because FlyDSL uses JIT compilation, run the kernel once before starting graph capture. When designing the kernel interface, expose configuration values that can change between invocations as runtime parameters. Configuration values that remain fixed, such as the KV-cache page size and split-K factor, can be kernel template parameters.

### Additional Reference

See the [FlyDSL kernel tuning guide](https://github.com/ROCm/FlyDSL/blob/421935cc6f09fd9b27d5d5ae52e0960e18834bd5/docs/kernel_tuning_guide.md).

## Summary

As AI models and attention algorithms evolve rapidly, traditional kernel libraries face challenges in both performance and flexibility. More and more customers require customized kernels for their algorithms.

In this blog, you gained a high-level yet actionable understanding of how we used FlyDSL to develop a customized diffusion attention kernel. We recommend that readers use this guide as a reference when implementing their own customized attention kernels. We tested this approach by providing an LLM agent with our customer kernel specification and carefully guide the agent step-by-step, and the resulting implementation worked well.

For other FlyDSL guide, please refer to other posts on our [ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html).

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, AMD Instinct, AMD ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
