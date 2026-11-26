---
blogpost: true
blog_title: "Bringing MXFP4 MoE to AMD Instinct MI300 Series GPUs with FlyDSL"
date: "26 Nov 2026"
author: "Menghsuan Yang, Felix Li, Clement Lin, Chunhung Wang"
thumbnail: 'gfx942-moe-mxfp4-thumbnail.png'
tags: "Optimization, Performance, Profiling, AI/ML"
category: "Applications & models"
target_audience: "AI kernel developers"
key_value_propositions: "Bit-exact MXFP4 byte-LUT dequant for gfx942 MoE GEMMs in AITER / FlyDSL"
language: English
myst:
    html_meta:
        "author": "Menghsuan Yang, Felix Li, Clement Lin, Chunhung Wang"
        "description lang=en": "Profile a VALU-bound MXFP4 MoE GEMM on AMD Instinct MI300X GPUs and swap software FP4 decode for a bit-exact v_perm_b32 LUT in AITER and FlyDSL."
        "keywords": "MXFP4, MoE, gfx942, FlyDSL, AITER, MI300X, dequant, LUT, profiling, VALU"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Menghsuan Yang, Felix Li, Clement Lin, Chunhung Wang"
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

# Bringing MXFP4 MoE to AMD Instinct MI300 Series GPUs with FlyDSL

The AMD CDNA™ 3 architecture of the AMD Instinct™ MI300 Series reached the market in December 2023, two months after the Open Compute Project (OCP) released the [Microscaling (MX) Formats specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) [[1]](#gfx942-moe-mxfp4-ref-1), which defines MXFP4. Hardware support for MX formats came with the next generation: AMD CDNA™ 4 in the AMD Instinct™ MI350 Series (gfx950, 2025) converts MXFP4 weights to bf16 with a single instruction, `v_cvt_scalef32_pk_bf16_fp4`, and CDNA 3 GPUs such as the MI300X (gfx942) have no such instruction. Serving a mixture-of-experts (MoE) model with MXFP4 (E2M1) weights, such as gpt-oss, on gfx942 with bf16 activations therefore needs every 4-bit weight decoded in software before it can feed a matrix core.

This post measures what that decode costs in `moe_2stage_a16wmix`, the MoE GEMM kernels that [AITER](https://github.com/ROCm/aiter) (the AMD AI Tensor Engine for ROCm™) [[2]](#gfx942-moe-mxfp4-ref-2) implements in [FlyDSL](https://github.com/ROCm/FlyDSL) [[3]](#gfx942-moe-mxfp4-ref-3), a Python domain-specific language for GPU kernels on ROCm. SQ counters read with [rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) [[4]](#gfx942-moe-mxfp4-ref-4) show that the waves are limited by VALU issue rather than by waiting on memory. [Advanced thread trace (ATT)](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html) [[5]](#gfx942-moe-mxfp4-ref-5) shows which instructions take those cycles. FlyDSL source-line tables tie those instructions to the Python lines that decode the weights, allowing us to follow the decoding code from FlyDSL to the compiled ISA. Replacing that decode with a bit-exact `v_perm_b32` byte lookup table (LUT) from [ROCm/aiter#5583](https://github.com/ROCm/aiter/pull/5583) [[6]](#gfx942-moe-mxfp4-ref-6) makes GEMM1 **2.2×–2.4×** and GEMM2 **1.7×–2.5×** faster on MI300X than a reference software decode with the same tiles, from 1 to 8192 tokens [[a]](#gfx942-moe-mxfp4-endnote-a), and the nonzero decoded bf16 weights are bit-identical for every e8m0 scale up to `0xFC`.

## The `a16wmix` Kernels

`moe_2stage_a16wmix` is a two-stage MoE GEMM: gate/up (GEMM1) then down (GEMM2), bf16 activations × MXFP4 weights, with the gated activation fused into the stage-1 epilogue. The measurements here use SwiGLU in the form gpt-oss uses: clamp(gate, max = 7) · σ(1.702 · gate) · (clamp(up, −7, 7) + 1). MXFP4 stores each weight as a 4-bit E2M1 code and shares one e8m0 scale per group of 32 weights. An e8m0 byte `e` encodes 2^(e − 127), and `0xFF` is NaN ([OCP MX specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)). Both stages load packed nibbles through `upconvert_b`, a FlyDSL helper function in AITER (`moe_2stage_a16wmix/utils.py`) that GEMM1 and GEMM2 share. On gfx950 that helper uses the hardware convert; on gfx942 it must synthesize the bf16 values itself.

Two terms recur below. `BM` is the number of sorted token rows that one workgroup tile covers (16 or 32 here). A K32 step is one 32-element slice along the reduction dimension; gfx942 has no K=32 bf16 MFMA, so each K32 step of a 16-row block takes two `16×16×16` bf16 MFMAs [[7]](#gfx942-moe-mxfp4-ref-7).

Shapes below use `E=896, model_dim=3584, inter_dim=384, topk=16` unless noted — a per-GPU shard of a wide MoE at TP=8.

(gfx942-moe-mxfp4-s1)=

## Profiling the Reference Software Decode

We profiled GEMM1 and GEMM2 on one MI300X with rocprofv3 in three layers, each narrower than the last:

1. **Kernel time** (`--kernel-trace`), from GPU timestamps rather than host wall clock, to see which stage dominates and how it scales with tokens. At small token counts, host overhead and run-to-run variance can swamp a host-side timer.
2. **SQ hardware counters** (`--pmc`), to see what the waves spend their cycles on.
3. **Advanced thread trace** (ATT, `--att`) on one CU, to see which instructions those cycles belong to.

A fourth step leaves the profiler: FlyDSL source-line tables tie each of those instructions to the Python line that produced it ([Mapping the Instructions Back to FlyDSL Lines](#gfx942-moe-mxfp4-s1-4)).

Kernel time covers seven token counts from 1 to 8192 ([Kernel Time on MI300X](#gfx942-moe-mxfp4-s3-4)). Counters and ATT use two of them: a decode batch of 8 tokens (`BM=16`) and a prefill batch of 2048 tokens (`BM=32`).

All measurements use upstream AITER. Before `#5583`, the FlyDSL MoE kernels in AITER did not support MXFP4 weights on gfx942. The first version we wrote while adding that support decoded the weights in software ([Root Cause: The Software E2M1→bf16 Decode](#gfx942-moe-mxfp4-s2)); it was never merged, and `#5583` shipped the LUT instead. To measure what the LUT removes, we put that software decode back at the same `upconvert_b` call site in the `#5583` merge commit `baef8156a` [[a]](#gfx942-moe-mxfp4-endnote-a).

(gfx942-moe-mxfp4-s1-1)=

### Counters That Locate the Limiter

All of these are SQ counters that rocprofv3 exposes on gfx942, and each metric is a ratio between two counters of the same kernel. Counters that cannot be collected together go into separate passes, so each ratio uses the median of each counter over its own pass.

| Metric | Counters | What it answers |
| --- | --- | --- |
| VALU per MFMA | (`SQ_INSTS_VALU` − `SQ_INSTS_MFMA`) / `SQ_INSTS_MFMA` | How much non-matrix vector work is issued for every matrix instruction |
| VALU share of issue | `SQ_ACTIVE_INST_VALU` / `SQ_ACTIVE_INST_ANY` | Of the cycles waves spend on instructions, how many are VALU |
| VMEM / LDS share of issue | `SQ_ACTIVE_INST_VMEM`, `SQ_ACTIVE_INST_LDS`, each over `SQ_ACTIVE_INST_ANY` | Whether memory instructions, rather than ALU work, take the issue cycles |
| Wave time ratios | `SQ_ACTIVE_INST_ANY`, `SQ_WAIT_INST_ANY`, `SQ_WAIT_ANY`, each over `SQ_WAVE_CYCLES` | How much of its lifetime a wave spends issuing, ready but waiting to issue, or waiting on a dependency such as an outstanding load or a barrier |

*Table 1. The SQ counter ratios used in this post.*

`SQ_INSTS_VALU` includes MFMA instructions. Its ratio to `SQ_INSTS_MFMA`, 73.0 for software GEMM1 at 8 tokens, matches the ratio of all VALU instructions (MFMA included) to MFMA that ATT counts on the traced CU, so the first row subtracts `SQ_INSTS_MFMA`. The two ratios agree within 0.1% for every dispatch we traced, which also shows that the single CU ATT traces runs the same instruction mix as the whole GPU. rocprofv3 ships the three wave time ratios as derived metrics: `WaveExec`, `WaveIssueWait`, and `WaveDepWait`.

(gfx942-moe-mxfp4-s1-2)=

### What the Counters Showed

Table 2 lists the counters for the software-decode path. Each value is the median over five dispatches, collected in four counter passes.

| Software decode | GEMM1, 8 tokens | GEMM2, 8 tokens | GEMM1, 2048 tokens | GEMM2, 2048 tokens |
| --- | ---: | ---: | ---: | ---: |
| VALU per MFMA | 72.0 | 79.5 | 35.8 | 40.9 |
| VALU share of issue | 97.4% | 95.7% | 98.4% | 93.4% |
| VMEM / LDS share of issue | 0.3% / 1.5% | 0.4% / 0.7% | 0.4% / 0.8% | 0.6% / 2.1% |
| Wave time issuing (`WaveExec`) | 45% | 38% | 30% | 50% |
| Wave time ready but waiting to issue (`WaveIssueWait`) | 32% | 49% | 60% | 28% |
| Wave time waiting on a dependency (`WaveDepWait`) | 22% | 13% | 10% | 21% |

*Table 2. SQ counters for the software-decode GEMMs on MI300X [[a]](#gfx942-moe-mxfp4-endnote-a). The three wave time ratios come from separate counter passes, so they need not sum to exactly 100%.*

Three things stand out:

- For every MFMA, the software path issues 36 to 80 other vector instructions. Decode work is per weight while MFMA work grows with the rows in a block, so `BM=32` at 2048 tokens halves the ratio, but it stays high.
- VALU takes 93–98% of the cycles in which waves issue anything. Memory and LDS instructions together take at most 2.7%.
- Waves spend 77–90% of their time either issuing or ready to issue, and only 10–23% waiting on a dependency such as an outstanding load or a barrier. With almost every issue cycle going to VALU, the slot a ready wave waits for is almost always a VALU slot.

That points at VALU issue, not at waiting on memory. [After the Fix: Counters and Thread Trace](#gfx942-moe-mxfp4-s3-3) adds a direct check: the LUT leaves the memory work and the time spent waiting on it unchanged, yet kernel time is reduced by a factor of 2.2. The counters cannot say which VALU instructions are the problem; ATT can.

(gfx942-moe-mxfp4-s1-3)=

### What the Thread Trace Showed

ATT records every instruction that the waves on one CU (CU 1 of shader engine 0) execute in one dispatch, with how often each instruction ran and how many cycles it accounted for. Table 3 groups the software-decode GEMMs at 2048 tokens by instruction class.

| Instruction class | GEMM1 instructions | GEMM1 cycles | GEMM2 instructions | GEMM2 cycles |
| --- | ---: | ---: | ---: | ---: |
| MFMA (`v_mfma_f32_16x16x16_bf16`) | 2.5% | 1.1% | 2.2% | 1.3% |
| Compare / select (`v_cmp_*`, `v_cndmask_*`) | 22.3% | 21.2% | 27.0% | 23.0% |
| Other VALU (shifts, bit-field extracts, masks, integer adds, f32 multiplies) | 66.0% | 62.7% | 60.3% | 50.8% |
| `v_perm_b32` | 2.5% | 2.4% | 2.3% | 2.1% |
| `s_waitcnt` | 0.5% | 5.5% | 1.3% | 14.4% |
| `s_nop` | 4.9% | 2.3% | 1.5% | 1.0% |
| VMEM and LDS | 0.8% | 0.7% | 1.7% | 2.3% |
| Scalar, branch, and other (including `s_barrier`) | 0.4% | 4.2% | 3.8% | 5.0% |

*Table 3. ATT instruction mix of the software-decode GEMMs at 2048 tokens on one MI300X CU [[a]](#gfx942-moe-mxfp4-endnote-a). "Cycles" is the per-instruction latency that ATT reports, summed over all executions and waves. Waves overlap in time, so these are shares of wave-cycles on that CU, not of kernel time. `s_barrier` alone is 4.0% of GEMM1 cycles and 3.1% of GEMM2 cycles.*

Table 3 shows that the cycles go to VALU work: VALU instructions other than MFMA (compare/select, other VALU, and `v_perm_b32`) are 91% of the instructions in software GEMM1 at 2048 tokens and 86% of its cycles, and 90% and 76% in GEMM2, while MFMA is 2–3% of the instructions and about 1% of the cycles. Within those VALU cycles, seven instructions take 60% of all GEMM1 cycles: `v_lshlrev_b32` (17%), `v_cmp_eq_u32` (10%), `v_pk_mul_f32` (9%), `v_bfe_u32` (7%), `v_and_or_b32` (6%), `v_cmp_u_f32` (5.5%), and `v_add3_u32` (5%). `v_cmp_u_f32` alone executes 216,000 times on the traced CU in one GEMM1 dispatch, counting both of its encodings. The 8-token dispatches show the same picture: MFMA is 1.3% of software GEMM1 instructions and 0.8% of its cycles, while compare/select is 23% and 18%.

The counters and ATT say what the waves issue, but not yet which code it belongs to. Shifts, masks, compares, and f32 multiplies could just as well come from address arithmetic, the scale loads, or the SwiGLU epilogue.

(gfx942-moe-mxfp4-s1-4)=

### Mapping the Instructions Back to FlyDSL Lines

ATT reports each instruction by its address in the kernel. To tie those addresses to FlyDSL source, we rebuilt the same kernels with FlyDSL source-line tables. With `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`, FlyDSL records the Python file and line of every operation it emits and passes them to the compiler as a DWARF line table. `FLYDSL_DUMP_IR=1` dumps the compilation stages, and the ISA built from them puts a `.loc` directive with that file and line before each instruction.

The rebuilt GEMM1 at 2048 tokens has the same 70,642 instructions, in the same order, as the kernel ATT traced for Table 3, and the same holds for every kernel in Table 4 [[a]](#gfx942-moe-mxfp4-endnote-a). Each row of the ATT listing can therefore be labeled with the FlyDSL line that produced it. Table 4 groups the rows by the FlyDSL function those lines belong to.

| FlyDSL source of the instruction | GEMM1, 8 tokens | GEMM2, 8 tokens | GEMM1, 2048 tokens | GEMM2, 2048 tokens |
| --- | ---: | ---: | ---: | ---: |
| Weight decode (`_fp4_nibble_to_bf16x8_sw`) | 93.1% / 73.4% | 90.1% / 66.8% | 92.3% / 80.5% | 84.5% / 69.3% |
| MFMA and operand packing | 3.5% / 2.8% | 3.8% / 3.1% | 4.8% / 3.3% | 5.0% / 5.4% |
| Weight and scale loads | 1.0% / 11.2% | 1.2% / 7.9% | 0.8% / 3.3% | 1.3% / 4.1% |
| Activation tile: global memory → LDS → registers | 0.6% / 8.0% | 0.4% / 3.1% | 0.6% / 4.5% | 1.3% / 6.4% |
| Epilogue: SwiGLU (GEMM1), weighted atomic store (GEMM2) | 0.2% / 0.1% | 0.6% / 0.8% | 0.4% / 0.1% | 3.1% / 5.0% |
| Indexing, barriers, and other | 1.6% / 4.5% | 3.8% / 18.3% | 1.1% / 8.2% | 4.8% / 9.8% |

*Table 4. Software decode: share of executed instructions / share of ATT cycles by FlyDSL function, on one MI300X CU [[a]](#gfx942-moe-mxfp4-endnote-a). Executed instructions are ATT hit counts; cycles are as in Table 3. "Weight and scale loads" includes the e8m0→f32 scale conversion.*

The weight decode helper, `_fp4_nibble_to_bf16x8_sw`, issues 85–93% of the executed instructions and takes 67–81% of the cycles, in both GEMMs and at both token counts. Of the 72.0 non-MFMA VALU instructions per MFMA in GEMM1 at 8 tokens (Table 2), 69.3 come from it, and across the four kernels it supplies 93–96% of that ratio. The matrix work itself, MFMA plus packing its operands, is 3.5–5.0% of the instructions, and no other group issues more than 5%. That ties the VALU-bound profile to the weight decode, and it is why we optimized the dequant first, rather than memory access or tiling.

(gfx942-moe-mxfp4-s2)=

## Root Cause: The Software E2M1→bf16 Decode

AMD CDNA 4 (gfx950) and later architectures have `v_cvt_scalef32_pk_bf16_fp4`, which converts the two FP4 weights in one byte into two bf16 values and multiplies them by an f32 scale in a single instruction; four of them decode the eight weights of a K32 step. The CDNA 3 GPU we ran on (gfx942) has no such instruction, so the software decode rebuilds each nibble (one FP4 weight) as an f32 bit pattern, multiplies it by the group scale, and converts the result to bf16. Figure 1 follows one nibble through those steps.

```{figure} ./images/sw_decode_bits.png
:align: center
:width: 100%
:alt: The software decode of one FP4 weight in four steps, drawn as bit cells colored by sign, exponent, and mantissa. Nibble 1011 (sign 1, exponent 01, mantissa 1, value −1.5) becomes the f32 bit pattern 1 01111111 1000... (0xBFC00000); multiplying by the scale 2.0 changes the exponent to 10000000 (0xC0400000, −3.0); casting to bf16 keeps the high 16 bits, 1100 0000 0100 0000 (0xC040).

Figure 1: The software decode of one FP4 weight, bit by bit. Nibble `1011` (−1.5) becomes the f32 bit pattern `0xBFC00000`, the scale 2.0 changes only its exponent, and rounding to bf16 keeps the high 16 bits, `0xC040` (−3.0).
```

In FlyDSL, abridged, for the eight nibbles `j` of one packed 32-bit word:

```python
for j in range_constexpr(8):
    n = raw.shrui(fx.Int32(4 * j)) & fx.Int32(0xF)
    e = n.shrui(fx.Int32(1)) & fx.Int32(0x3)
    mant = n & fx.Int32(0x1)
    sign = n.shrui(fx.Int32(3)) & fx.Int32(0x1)
    normal_bits = ((e + fx.Int32(126)) << fx.Int32(23)) | (mant << fx.Int32(22))
    sub_bits = (mant == fx.Int32(1)).select(fx.Int32(0x3F000000), fx.Int32(0))
    mag_bits = (e == fx.Int32(0)).select(sub_bits, normal_bits)
    bits = mag_bits | (sign << fx.Int32(31))
    v = fx.Float32(_raw(bits).bitcast(T.f32)) * fx.Float32(scale_f32)
    bf16s.append(v.to(fx.BFloat16))
```

Figure 2 maps each line of that loop to the gfx942 instructions it compiles to.

```{figure} ./images/sw_decode_flow.png
:align: center
:width: 100%
:alt: Flow of the software E2M1 decode for one nibble, in five steps with the FlyDSL code and the gfx942 instructions of each step: extract the fields, build the f32 bit pattern with selects for subnormals, multiply by the scale, convert to bf16 with round-to-nearest-even and a NaN check, and repeat for eight nibbles. A worked example decodes nibble 0x5 with scale 2.0 to bf16 0x40C0 (6.0). About 17 VALU instructions per weight, 139 per K32 step.

Figure 2: The FlyDSL code of the software decode, step by step, with the gfx942 instructions each step compiles to and a worked example.
```

(gfx942-moe-mxfp4-s2-1)=

### What the Software Decode Compiles To

Below is the ISA for the first two weights of software GEMM1 at 8 tokens, from the debug-info build described in [Mapping the Instructions Back to FlyDSL Lines](#gfx942-moe-mxfp4-s1-4). Each instruction sits under the FlyDSL line that its `.loc` names. `v28` holds eight packed nibbles from the weight load, and `v54` holds the f32 scale of the group. The compiler decodes the two nibbles side by side, so most lines produce one instruction per weight. The `sign` line has no instructions of its own; its shift and mask are folded into the `bits = mag_bits | (sign << 31)` line.

```text
# n = raw.shrui(fx.Int32(4 * j)) & fx.Int32(0xF)
  v_lshrrev_b32_e32 v32, 4, v28                 ; j = 1 (j = 0 needs no shift)
# e = n.shrui(fx.Int32(1)) & fx.Int32(0x3)
  v_lshrrev_b32_e32 v33, 5, v28
  v_lshrrev_b32_e32 v34, 1, v28
  v_and_b32_e32     v33, 3, v33
  v_and_b32_e32     v34, 3, v34
# mant = n & fx.Int32(0x1)
  v_and_b32_e32     v32, 1, v32
  v_and_b32_e32     v35, 1, v28
# normal_bits = ((e + fx.Int32(126)) << fx.Int32(23)) | (mant << fx.Int32(22))
  v_lshlrev_b32_e32 v36, 23, v33
  v_lshlrev_b32_e32 v37, 23, v34
  v_lshlrev_b32_e32 v38, 22, v32
  v_lshlrev_b32_e32 v39, 22, v35
  v_or_b32_e32      v36, v36, v38
  v_or_b32_e32      v37, v37, v39
  v_add_u32_e32     v37, 0.5, v37               ; + (126 << 23) = 0x3F000000, printed as 0.5
  v_add_u32_e32     v36, 0.5, v36
# sub_bits = (mant == fx.Int32(1)).select(fx.Int32(0x3F000000), fx.Int32(0))
  v_cmp_eq_u32_e64  s[0:1], 0, v32
  v_cmp_eq_u32_e64  s[2:3], 0, v35
  v_cndmask_b32_e64 v32, 0.5, 0, s[2:3]         ; 0.5 is 0x3F000000 again
  v_cndmask_b32_e64 v35, 0.5, 0, s[0:1]
# mag_bits = (e == fx.Int32(0)).select(sub_bits, normal_bits)
  v_cmp_eq_u32_e64  s[0:1], 0, v34
  v_cmp_eq_u32_e64  s[2:3], 0, v33
  v_cndmask_b32_e64 v32, v37, v32, s[0:1]
  v_cndmask_b32_e64 v33, v36, v35, s[2:3]
# bits = mag_bits | (sign << fx.Int32(31))
  v_lshlrev_b32_e32 v34, 24, v28                ; nibble bit 3 -> bit 31
  v_lshlrev_b32_e32 v35, 28, v28
  v_and_or_b32      v33, v34, s27, v33          ; s27 = 0x80000000
  v_and_or_b32      v32, v35, s27, v32
# v = fx.Float32(_raw(bits).bitcast(T.f32)) * fx.Float32(scale_f32)
  v_pk_mul_f32      v[32:33], v[54:55], v[32:33] op_sel_hi:[0,1]   ; both weights x v54
# bf16s.append(v.to(fx.BFloat16))
  v_bfe_u32         v34, v32, 16, 1             ; lowest bit that bf16 keeps
  v_add3_u32        v34, v34, v32, s26          ; + 0x7FFF + that bit: round to nearest even
  v_or_b32_e32      v35, 0x400000, v32          ; quiet-NaN pattern
  v_cmp_u_f32_e64   s[0:1], v32, v32            ; is v NaN?
  v_cndmask_b32_e64 v32, v34, v35, s[0:1]
  ...                                           ; the same five for the second weight
# fx.gemm(mma_atom, acc, _bf16_frag4(a8, h), _bf16_frag4(b8, h), acc)
  v_perm_b32        v34, v33, v32, s28          ; s28 = 0x07060302: high halves of v32, v33 -> two bf16
```

That is 38 VALU instructions for two weights, plus the last `v_perm_b32`, which packs the two bf16 results into one register. The line table charges it to the MFMA call (`_mma_bf16`) rather than to the decode helper, because the packing happens where the MFMA operand is built. Over the whole kernel, the line table puts 69.3 of the 72.0 non-MFMA VALU instructions per MFMA (Table 2) in this helper: about 17 per weight, or 139 per K32 step.

These are the instructions at the top of Table 3. Shifts (`v_lshlrev_b32`) build the exponent and sign fields, each `select` becomes a `v_cmp_eq_u32` and `v_cndmask_b32` pair, and the scale multiply becomes `v_pk_mul_f32`. The last line hides the most work. gfx942 has no `v_cvt_pk_bf16_f32`, so `v.to(fx.BFloat16)` becomes a five-instruction round-to-nearest-even (RNE) with a NaN check (`v_cmp_u_f32`). E2M1 has no NaN encoding, so for every e8m0 scale below `0xFF` that compare can never fire. By the line table, that one line accounts for 21% of software GEMM1 cycles at 2048 tokens.

(gfx942-moe-mxfp4-s2-2)=

### The Decode Is a Bit Pattern

None of that arithmetic is needed. Each weight is a 4-bit FP4 (E2M1) code, and the MFMA takes 16-bit bf16 operands, so every weight has to grow from 4 bits to 16. But a 4-bit code has only 16 values, and E2M1, with a 2-bit exponent and a 1-bit mantissa, fits inside bf16, so every one of them is **exactly** representable. The 16 output bits of every code are therefore known in advance: turning FP4 into bf16 is a table lookup, not a computation. The sign is a single bit, so the table needs only the eight magnitudes in Table 5. The e8m0 scale does not change this: it is a power of two, so for every scale up to `0xFC` the scaled value is still exact in bf16 (above that, the largest magnitudes overflow, and `0xFF` is NaN).

| nibble `& 7` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| value | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 6.0 |
| bf16 bits | `0x0000` | `0x3F00` | `0x3F80` | `0x3FC0` | `0x4000` | `0x4040` | `0x4080` | `0x40C0` |

*Table 5. The eight E2M1 magnitudes and their bf16 bit patterns.*

Across the eight magnitudes, the high bytes take only three distinct values `{0x00, 0x3F, 0x40}` and the low bytes four `{0x00, 0x40, 0x80, 0xC0}`, so each half fits an 8-entry byte table indexed by the magnitude. `v_perm_b32` treats its two 32-bit sources as an 8-byte table, and each selector byte from 0 to 7 picks one table byte; selector values 8 and above have special meanings (sign replication, `0x00`, `0xFF`) [[8]](#gfx942-moe-mxfp4-ref-8). The 3-bit magnitude `nibble & 7` is therefore a valid selector, so one `v_perm_b32` returns the four high bytes, and one more the four low bytes, for four nibbles with no compare chain. The sign is bit 3 of each nibble. For the high nibble of a byte it already sits at bit 7, where the bf16 high byte needs it; for the low nibble, a 4-bit shift moves it there.

(gfx942-moe-mxfp4-s3)=

## Fix: Bit-Exact `v_perm_b32` LUT ([aiter#5583](https://github.com/ROCm/aiter/pull/5583))

`#5583` (merged 2026-09-22 as `baef8156a`) adds the gfx942 branch of `upconvert_b`. It decodes with:

1. Two 8-entry byte tables, holding the high and low bytes of the eight unsigned bf16 magnitudes, stored as four 32-bit constants.
2. Four `v_perm_b32` lookups that decode the high and low bytes of eight nibbles, and eight more that interleave those bytes into four 32-bit words of bf16 pairs. Masks build the selectors, and one `v_and_or_b32` per byte group folds the sign back in.
3. The e8m0 scale, applied by shifting each bf16 into an f32 (low halves with a 16-bit shift, high halves with a mask), multiplying, and truncating back to bf16. The scale is a power of two, so the product is exact in bf16 and truncation gives the same bits as round-to-nearest.

Figure 3 follows eight weights through these steps. The letters `a` to `h` stand for the eight FP4 weights in one 32-bit word, written from bit 31 down to bit 0, so `h` is element 0 of the K32 step.

```{figure} ./images/lut_decode_bits.png
:align: center
:width: 100%
:alt: The LUT decode of eight FP4 weights a to h in one 32-bit word, in seven steps. The word [a b c d e f g h] splits into sel_even [0 b 0 d 0 f 0 h] and sel_odd [0 a 0 c 0 e 0 g], where each byte [0 w] is the table index of one weight (index_h = 0000 0101 = 5, index_g = 0000 0011 = 3); two 8-entry byte tables (high bytes 00 3F 3F 3F 40 40 40 40, low bytes 00 00 80 C0 00 40 80 C0) are stored in the four _FP4_MAG constants; four v_perm_b32 look up HI[x] and LO[x] for each weight; the sign bits go back into the high bytes; eight more v_perm_b32 build four dwords of bf16 pairs [g h] [e f] [c d] [a b]; the scale is applied last. Example byte g h = 1011 0101 gives (−1.5, +3.0) and, with scale 2.0, (−3.0, +6.0).

Figure 3: The LUT decode of eight FP4 weights, step by step. Each weight, widened to one byte `[0 w]` with its sign bit cleared, is an index into two tables stored in the four `_FP4_MAG` constants, and each `_fp4_perm` call is one `v_perm_b32` that looks up four of those indices at once. The example byte `g h = 1011 0101` decodes to (−1.5, +3.0) and, with scale 2.0, to (−3.0, +6.0).
```

Each byte of `sel_even` and `sel_odd` is one weight widened to a byte, `[0 w]`, with its sign bit cleared, so it is a table index from 0 to 7. The four `_FP4_MAG` constants in the code below are the two tables. A 32-bit constant holds four table entries with its lowest byte first: `_FP4_MAG_HI_LO = 0x3F3F3F00` holds the high bytes 00, 3F, 3F, 3F of magnitudes 0–3, and `_FP4_MAG_HI_HI = 0x40404040` those of magnitudes 4–7, while `_FP4_MAG_LO_LO` and `_FP4_MAG_LO_HI` hold the low bytes. `_fp4_perm(src_hi, src_lo, sel)` is one `v_perm_b32`: `src_lo` supplies table entries 0–3, `src_hi` supplies entries 4–7, and each byte of `sel` picks one entry for the same byte of the result. With `sel_even` as the selector, one call returns the bf16 high bytes of the four even weights, and a second call their low bytes. The interleave uses the same instruction with constant selectors: `0x05010400` takes byte 0 of `src_lo`, byte 0 of `src_hi`, byte 1 of `src_lo`, and byte 1 of `src_hi`, which puts each low byte next to its high byte.

The helper, abridged from [`aiter/ops/flydsl/kernels/moe_2stage_a16wmix/utils.py`](https://github.com/ROCm/aiter/blob/baef8156aa222809e7aa77801f4b7d064ca94149/aiter/ops/flydsl/kernels/moe_2stage_a16wmix/utils.py) at `baef8156a` (AITER, MIT License):

```python
_FP4_MAG_HI_LO = 0x3F3F3F00  # bf16 high bytes for nibble&7 = 0,1,2,3
_FP4_MAG_HI_HI = 0x40404040  # bf16 high bytes for nibble&7 = 4,5,6,7
_FP4_MAG_LO_LO = 0xC0800000  # bf16 low bytes  for nibble&7 = 0,1,2,3
_FP4_MAG_LO_HI = 0xC0804000  # bf16 low bytes  for nibble&7 = 4,5,6,7


def _fp4_mag_dwords(raw_i32):
    """Decode 8 FP4 nibbles to 4 dwords, each holding 2 bf16, scale not applied."""
    raw = fx.Int32(raw_i32)
    sel_even = raw & fx.Int32(0x07070707)  # low nibble of each byte -> elements 0,2,4,6
    sel_odd = raw.shrui(fx.Int32(4)) & fx.Int32(0x07070707)  # -> elements 1,3,5,7

    hb_even = _fp4_perm(_FP4_MAG_HI_HI, _FP4_MAG_HI_LO, sel_even)
    lb_even = _fp4_perm(_FP4_MAG_LO_HI, _FP4_MAG_LO_LO, sel_even)
    hb_odd = _fp4_perm(_FP4_MAG_HI_HI, _FP4_MAG_HI_LO, sel_odd)
    lb_odd = _fp4_perm(_FP4_MAG_LO_HI, _FP4_MAG_LO_LO, sel_odd)

    # Sign: nibble bit 3 -> bf16 bit 15, i.e. bit 7 of the high byte.
    hb_even = hb_even | ((raw & fx.Int32(0x08080808)) << fx.Int32(4))
    hb_odd = hb_odd | (raw & fx.Int32(0x80808080))

    # dword d = [lb_even[d], hb_even[d], lb_odd[d], hb_odd[d]].
    ev01 = _fp4_perm(hb_even, lb_even, fx.Int32(0x05010400))
    od01 = _fp4_perm(hb_odd, lb_odd, fx.Int32(0x05010400))
    ev23 = _fp4_perm(hb_even, lb_even, fx.Int32(0x07030602))
    od23 = _fp4_perm(hb_odd, lb_odd, fx.Int32(0x07030602))
    return [
        _fp4_perm(od01, ev01, fx.Int32(0x05040100)),
        _fp4_perm(od01, ev01, fx.Int32(0x07060302)),
        _fp4_perm(od23, ev23, fx.Int32(0x05040100)),
        _fp4_perm(od23, ev23, fx.Int32(0x07060302)),
    ]


def _fp4_nibble_to_bf16x8(raw_i32, scale_f32):
    """FP4 (E2M1) -> v8bf16 for one MFMA K32 step, with the groupwise scale applied."""
    scale = fx.Float32(scale_f32)
    out = []
    for d in _fp4_mag_dwords(raw_i32):
        lo_f = fx.Float32(_raw(d << fx.Int32(16)).bitcast(T.f32))
        hi_f = fx.Float32(_raw(d & fx.Int32(0xFFFF0000)).bitcast(T.f32))
        lo_b = fx.Int32(_raw(lo_f * scale).bitcast(T.i32)).shrui(fx.Int32(16))
        hi_b = fx.Int32(_raw(hi_f * scale).bitcast(T.i32)) & fx.Int32(0xFFFF0000)
        out.append(lo_b | hi_b)
    return fx.Vector.from_elements([_raw(x) for x in out], fx.Int32).bitcast(fx.BFloat16)
```

`_fp4_perm(src_hi, src_lo, sel)` is a thin wrapper around `rocdl.perm_b32`. The same helper is the gfx942 branch of `upconvert_b` for **both** GEMM1 and GEMM2.

Figure 4 maps each step of the helper to the gfx942 instructions it compiles to.

```{figure} ./images/lut_decode_flow.png
:align: center
:width: 100%
:alt: Flow of the LUT E2M1 decode for one packed dword of eight nibbles, in five steps with the FlyDSL code and the gfx942 instructions of each step: build 3-bit selectors, look up the bf16 high and low bytes with four v_perm_b32, fold in the sign bits, interleave the bytes into four dwords of bf16 pairs with eight v_perm_b32, and apply the power-of-two scale. A worked example decodes byte 0xB5 with scale 2.0 to the bf16 pair 6.0 and −3.0. About 5 VALU instructions per weight, 39 per K32 step.

Figure 4: The FlyDSL code of the `#5583` LUT decode for one K32 step, with the gfx942 instructions each step compiles to and a worked example on byte `0xB5`.
```

(gfx942-moe-mxfp4-s3-1)=

### What the LUT Decode Compiles To

Below are the instructions that produce the weight operand of the first MFMA, four weights, in LUT GEMM1 at 8 tokens, from the same debug-info build as the listing in [What the Software Decode Compiles To](#gfx942-moe-mxfp4-s2-1). They are grouped under the FlyDSL lines of `_fp4_mag_dwords` and `_fp4_nibble_to_bf16x8` above. The line table charges every `v_perm_b32` to the one-line `_fp4_perm` wrapper, so the table and selector operands (`s1`, `v45`, `s2`, `s25`, `s28`, `s29`) show which call each one is. `v24` first holds eight packed nibbles from the weight load; once the sign step has read it, the compiler reuses it for the f32 scale of the group.

```text
# sel_even = raw & fx.Int32(0x07070707)
  v_and_b32_e32     v30, 0x7070707, v24
# sel_odd = raw.shrui(fx.Int32(4)) & fx.Int32(0x07070707)
  v_lshrrev_b32_e32 v31, 4, v24
  v_and_b32_e32     v31, 0x7070707, v31
# hb_even = _fp4_perm(_FP4_MAG_HI_HI, _FP4_MAG_HI_LO, sel_even)
  v_perm_b32        v34, s1, v45, v30           ; s1 = 0x40404040, v45 = 0x3F3F3F00
# lb_even = _fp4_perm(_FP4_MAG_LO_HI, _FP4_MAG_LO_LO, sel_even)
  v_perm_b32        v30, s2, -4.0, v30          ; s2 = 0xC0804000, -4.0 = 0xC0800000
# hb_odd = _fp4_perm(_FP4_MAG_HI_HI, _FP4_MAG_HI_LO, sel_odd)
  v_perm_b32        v5, s1, v45, v31
# lb_odd = _fp4_perm(_FP4_MAG_LO_HI, _FP4_MAG_LO_LO, sel_odd)
  v_perm_b32        v6, s2, -4.0, v31
# hb_even = hb_even | ((raw & fx.Int32(0x08080808)) << fx.Int32(4))
  v_lshlrev_b32_e32 v32, 4, v24
  v_and_or_b32      v7, v32, s3, v34            ; s3 = 0x80808080
# hb_odd = hb_odd | (raw & fx.Int32(0x80808080))
  v_and_or_b32      v5, v24, s3, v5
# ev01 = _fp4_perm(hb_even, lb_even, fx.Int32(0x05010400))
  v_perm_b32        v15, v7, v30, s25           ; s25 = 0x05010400
# od01 = _fp4_perm(hb_odd, lb_odd, fx.Int32(0x05010400))
  v_perm_b32        v13, v5, v6, s25
# _fp4_perm(od01, ev01, fx.Int32(0x05040100)), _fp4_perm(od01, ev01, fx.Int32(0x07060302))
  v_perm_b32        v14, v13, v15, s28          ; dword 0
  v_perm_b32        v15, v13, v15, s29          ; dword 1
# scale loader, shared with the software decode: e8m0 byte -> f32
  v_lshlrev_b32_e32 v16, s33, v21               ; byte into the exponent field
  v_and_b32_e32     v24, 0x7f800000, v16        ; v24 = 2^(e8m0 - 127)
# lo_f = fx.Float32(_raw(d << fx.Int32(16)).bitcast(T.f32))
  v_lshlrev_b32_e32 v4, 16, v14
  v_lshlrev_b32_e32 v5, 16, v15
# hi_f = fx.Float32(_raw(d & fx.Int32(0xFFFF0000)).bitcast(T.f32))
  v_and_b32_e32     v14, 0xffff0000, v14
  v_and_b32_e32     v15, 0xffff0000, v15
# lo_f * scale, hi_f * scale
  v_pk_mul_f32      v[4:5], v[24:25], v[4:5] op_sel_hi:[0,1]       ; both lanes x v24
  v_pk_mul_f32      v[14:15], v[24:25], v[14:15] op_sel_hi:[0,1]
# lo_b = fx.Int32(_raw(lo_f * scale).bitcast(T.i32)).shrui(fx.Int32(16))
  v_lshrrev_b32_e32 v4, 16, v4
  v_lshrrev_b32_e32 v5, 16, v5
# hi_b = ... & fx.Int32(0xFFFF0000);  out.append(lo_b | hi_b)
  v_and_or_b32      v96, v14, s27, v4           ; s27 = 0xFFFF0000
  v_and_or_b32      v97, v15, s27, v5           ; v[96:97]: weight operand of the MFMA
```

Apart from the scale loader, that is 24 VALU instructions for four weights. The other four weights of the K32 step reuse the selectors, lookups, and sign, and need only the remaining four `v_perm_b32` and their scale steps: 14 more. Over the whole kernel, the line table puts 19.4 of the 20.5 non-MFMA VALU instructions per MFMA (Table 6) in the decode helpers: about 5 per weight, against about 17 for the software decode.

The scale loader builds the f32 scale by shifting the e8m0 byte into the exponent field and masking it. That maps e8m0 `0x00` to 0.0 rather than 2^-127, and `0xFF` to `+inf` rather than NaN. Both decodes use the same loader, so this does not differ between them.

(gfx942-moe-mxfp4-s3-2)=

### Checking Bit-Exactness on MI300X

To check that the two decodes produce the same bits on hardware, we read decoded weights back through GEMM2. With `topk=1`, a routing weight of 1.0, and a one-hot intermediate row, each output element is exactly one decoded weight: one nonzero product per dot product, exact in fp32, written onto a zeroed bf16 output. GEMM1 calls the same helper; the test reads back through GEMM2 because GEMM2 has no activation epilogue.

- The test reads back 22,020,096 weights (16 experts × 3584 × 384) with random packed bytes and e8m0 scales from `0x00` to `0xFC`. That covers all 64,768 pairs of packed byte value and scale.
- Every nonzero weight is bit-identical between the software and LUT outputs. Zeros match by value; the test cannot see their sign, because adding −0 to a +0 accumulator gives +0.
- Against the exact E2M1 × 2^(e8m0 − 127) values, the only differences are the 76,208 nonzero weights with scale `0x00`. Both decodes return 0 for them, because of the scale loader described in [What the LUT Decode Compiles To](#gfx942-moe-mxfp4-s3-1).
- Scales `0xFD`–`0xFF` are left out: they overflow bf16 for the largest magnitudes or are NaN, and a single non-finite weight would turn its whole dot product into NaN.

(gfx942-moe-mxfp4-s3-3)=

### After the Fix: Counters and Thread Trace

Table 6 repeats the counters from [What the Counters Showed](#gfx942-moe-mxfp4-s1-2) with the LUT in place.

| Software → LUT | GEMM1, 8 tokens | GEMM2, 8 tokens | GEMM1, 2048 tokens | GEMM2, 2048 tokens |
| --- | ---: | ---: | ---: | ---: |
| VALU per MFMA | 72.0 → 20.5 | 79.5 → 20.9 | 35.8 → 10.1 | 40.9 → 11.6 |
| Non-MFMA VALU instructions, LUT / software | 0.28 | 0.26 | 0.28 | 0.28 |
| MFMA and LDS instructions, waves | unchanged | unchanged | unchanged | unchanged |
| VMEM instructions | unchanged | unchanged | −3% | unchanged |
| VALU share of issue | 97.4% → 92.9% | 95.7% → 85.7% | 98.4% → 94.4% | 93.4% → 80.7% |
| Wave time waiting on a dependency | 22% → 24% | 13% → 29% | 10% → 20% | 21% → 49% |

*Table 6. SQ counters before and after the LUT on MI300X [[a]](#gfx942-moe-mxfp4-endnote-a).*

The LUT issues 3.5×–3.8× fewer non-MFMA VALU instructions for exactly the same MFMA instructions, in both GEMMs and at both token counts. VALU is still most of what the waves issue, but there is far less of it between consecutive MFMAs.

This is also the direct check promised in [What the Counters Showed](#gfx942-moe-mxfp4-s1-2). The LUT leaves the LDS instruction count and the number of waves unchanged, and the VMEM instruction count unchanged or 3% lower. On the traced CU at 2048 tokens, the cycles spent in `s_waitcnt` do not grow in GEMM1 (2.93M → 2.11M) and grow 10% in GEMM2 (2.76M → 3.04M) [[a]](#gfx942-moe-mxfp4-endnote-a). Yet GEMM1 and GEMM2 kernel times drop 2.2× and 1.8× ([Kernel Time on MI300X](#gfx942-moe-mxfp4-s3-4)). If waiting on memory had been the limit, removing VALU work could not have moved the kernel time this much.

ATT shows where the remaining cycles go. In LUT GEMM1 at 2048 tokens, compare/select drops from 21% to 0.6% of the cycles, and `v_cmp_u_f32` falls from 216,000 executions on the traced CU to 1,024, none of them in the decode. `v_perm_b32` becomes 25% of the instructions but only 16% of the cycles. The MFMA share of the cycles rises from 1.1% to 4.9% in GEMM1 and from 1.3% to 5.3% in GEMM2.

The line table shows that the decode, although cheaper, is still most of GEMM1. Its three helpers issue 85% of the instructions in LUT GEMM1 at 8 tokens and 81% at 2048 tokens, and take 62% and 75% of the cycles, against 93% and 92% of the instructions and 73% and 81% of the cycles for the software decode (Table 4) [[a]](#gfx942-moe-mxfp4-endnote-a). In LUT GEMM2 the decode share falls further, to 75% and 62% of the instructions and 53% and 43% of the cycles. That is why VALU still dominates issue after the fix.

The next limits are visible in the same data:

- **Waits are now exposed.** With the decode cheaper, `s_waitcnt` is 11% of GEMM1 cycles and 36% of GEMM2 cycles (up from 5.5% and 14%). `s_barrier` grows from 4.0% to 4.8% of GEMM1 cycles and from 3.1% to 5.5% of GEMM2 cycles, and GEMM2 waves wait on a dependency 49% of the time.
- **Stalls at the scale multiply.** ATT charges 33% of LUT GEMM1 cycles at 2048 tokens to `v_pk_mul_f32`, the `lo_f * scale` and `hi_f * scale` multiplies of `_fp4_nibble_to_bf16x8`, and 94% of those cycles are waiting before issue. The software decode shows the same pattern at that instruction (97%), so the stall belongs to this instruction rather than to the lookup.

Neither is the bit-pattern lookup itself, so they are separate optimizations.

(gfx942-moe-mxfp4-s3-4)=

### Kernel Time on MI300X

Table 7 compares kernel time on one AMD Instinct MI300X GPU, taken from the GEMM1 and GEMM2 kernels of full `fused_moe` calls. The only source difference between the two arms is the decode helper called at that one site in `upconvert_b`. Each arm has its own FlyDSL kernel cache, so neither reuses a kernel compiled for the other decode. Tiles come from the tuned table that `#5583` ships, and both arms pick the same kernels at every token count: `BM=16` up to 512 tokens, and `BM=32` at 2048 and 8192. Routing follows the op test in that commit: the top 16 of random scores per token.

| Tokens | Distinct experts | Software GEMM1 (µs) | LUT GEMM1 (µs) | GEMM1 × | Software GEMM2 (µs) | LUT GEMM2 (µs) | GEMM2 × | Total × |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 49.0 | 20.8 | 2.36 | 20.0 | 8.6 | 2.32 | 2.35 |
| 2 | 32 | 75.8 | 31.4 | 2.41 | 34.3 | 14.8 | 2.31 | 2.38 |
| 4 | 64 | 109.7 | 49.5 | 2.22 | 51.1 | 23.5 | 2.17 | 2.20 |
| 8 | 128 | 231.7 | 95.5 | 2.43 | 107.8 | 49.1 | 2.19 | 2.35 |
| 512 | 896 | 1306.1 | 591.9 | 2.21 | 774.3 | 313.2 | 2.47 | 2.30 |
| 2048 | 896 | 2336.1 | 1040.7 | 2.24 | 1587.3 | 884.7 | 1.79 | 2.04 |
| 8192 | 896 | 6661.1 | 2846.0 | 2.34 | 4217.1 | 2534.0 | 1.66 | 2.02 |

*Table 7. Software vs LUT decode, rocprofv3 kernel time on MI300X [[a]](#gfx942-moe-mxfp4-endnote-a). Mean of three repetitions of 100 `fused_moe` calls after 10 warm-up calls. Total × is (software GEMM1 + GEMM2) / (LUT GEMM1 + GEMM2); MoE sorting is not included. Per repetition, the whole MoE call agrees within 2.2%, except the LUT arm at 2048 tokens (6.5%).*

Both GEMMs speed up, because both were spending their issue cycles on the decode. Kernel time drops less than the VALU instruction count does (for GEMM1 at 2048 tokens, 2.2× against 3.5×): the LUT kernel still spends most of its issue cycles on VALU, and the waits described in [After the Fix: Counters and Thread Trace](#gfx942-moe-mxfp4-s3-3) are now exposed. Prefill GEMM2 gains the least (1.8× at 2048 tokens and 1.7× at 8192), which matches Table 6: after the fix, its waves spend 49% of their time waiting on a dependency rather than issuing.

(gfx942-moe-mxfp4-try-it)=

## Try It

`#5583` made the LUT the gfx942 decode for MXFP4 weights with bf16 activations, and AITER used it without a flag from `baef8156a` until `#6144` replaced it at `f2cb249` ([Follow-Up Decode Optimizations in AITER](#gfx942-moe-mxfp4-followup)). Check out `baef8156a` to reproduce the numbers of this post. We ran the steps below with FlyDSL 0.3.4.1, ROCm 10.0.0, and PyTorch 2.12.0 [[a]](#gfx942-moe-mxfp4-endnote-a). Save this script as `moe.py`. It sends the shape of this post through `fused_moe` with SwiGLU, with inputs and weights set up as in the AITER op test for these kernels:

```python
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

E, topk, H, I, tokens = 896, 16, 3584, 384, 2048
x = torch.randn((tokens, H), dtype=dtypes.bf16, device="cuda")
w1 = torch.randn((E, 2 * I, H), dtype=dtypes.bf16, device="cuda")
w2 = torch.randn((E, H, I), dtype=dtypes.bf16, device="cuda")
scores = torch.randn((tokens, E), dtype=dtypes.bf16, device="cuda")
topk_weights, topk_ids = fused_topk(x, scores, topk, True)

quant = aiter.get_torch_quant(QuantType.per_1x32)
w1_q, w1_s = quant(w1, quant_dtype=dtypes.fp4x2)
w2_q, w2_s = quant(w2, quant_dtype=dtypes.fp4x2)
out = fused_moe(
    x,
    shuffle_weight_a16w4(w1_q.view(E, 2 * I, H // 2), 16, False),
    shuffle_weight_a16w4(w2_q.view(E, H, I // 2), 16, False),
    topk_weights,
    topk_ids,
    w1_scale=shuffle_scale_a16w4(w1_s, E, False),
    w2_scale=shuffle_scale_a16w4(w2_s, E, False),
    quant_type=QuantType.per_1x32,
    activation=ActivationType.Swiglu,
    quant_dtype_a=dtypes.bf16,
    doweight_stage1=False,
    gate_mode=GateMode.SEPARATED.value,
)
torch.cuda.synchronize()
print("out", tuple(out.shape), out.dtype, "nan" if out.isnan().any().item() else "finite")
```

To profile the two GEMMs the way [Profiling the Reference Software Decode](#gfx942-moe-mxfp4-s1) does, run the script under rocprofv3: kernel time, the four counter passes behind Table 2, and ATT on one CU.

```bash
rocprofv3 --kernel-trace --output-format csv -d trace -o trace -- python3 moe.py
rocprofv3 --kernel-include-regex gemm --output-format csv -d pmc -o pmc \
  --pmc 'GRBM_GUI_ACTIVE SQ_WAVES SQ_WAVE_CYCLES SQ_BUSY_CYCLES' \
  --pmc 'GRBM_GUI_ACTIVE SQ_INSTS_VALU SQ_INSTS_MFMA SQ_INSTS_SALU SQ_INSTS_VMEM' \
  --pmc 'GRBM_GUI_ACTIVE SQ_ACTIVE_INST_VALU SQ_VALU_MFMA_BUSY_CYCLES SQ_ACTIVE_INST_ANY SQ_INSTS_LDS' \
  --pmc 'GRBM_GUI_ACTIVE SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_ACTIVE_INST_VMEM SQ_ACTIVE_INST_LDS' \
  -- python3 moe.py
rocprofv3 --att --kernel-include-regex gemm --att-target-cu 1 -d att -o att -- python3 moe.py
```

For the source lines of [Mapping the Instructions Back to FlyDSL Lines](#gfx942-moe-mxfp4-s1-4), build the kernels again with line tables, starting from an empty FlyDSL cache (`~/.flydsl/cache` by default):

```bash
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=dump python3 moe.py
```

Each kernel gets a directory under `dump/`, named after the kernel. Its `*_final_isa.s` file puts a `.loc` directive (file number and line) before each instruction, and the `.file` lines at the top of the file map those numbers to Python files. Before joining that file with the instruction list in an ATT `stats_*.csv`, check that both list the same instructions in the same order. If a FlyDSL compile fails with `lld invocation failed`, set `ROCM_PATH` to the ROCm install, for example `/opt/rocm`; in our container, the FlyDSL link step found `ld.lld` only with it set.

## Limitations

These bound what the measurements above support:

- **One GPU and one shape.** Every number comes from one MI300X running GEMM1 and GEMM2 at `E=896, model_dim=3584, inter_dim=384, topk=16` with SwiGLU and bf16 activations. Other shapes, activations, and gfx942 products are not measured.
- **Synthetic weights and routing.** Inputs and weights are N(0,1) before MXFP4 quantization, and routing takes the top 16 of random scores, as in the AITER op test for these kernels. Weights and routing from a real checkpoint are not measured.
- **A reference baseline.** The speedups compare the LUT with the first, straightforward software decode written during development, placed at the same call site; it was never merged into AITER.
- **GEMM time only.** Table 7 covers the two GEMM kernels. MoE sorting and the rest of the model are not included, so it is not an end-to-end speedup.
- **One process per point.** Each kernel time comes from one process. Across its three repetitions the whole MoE call agrees within 2.2% (6.5% for the LUT at 2048 tokens), but variation between processes was not measured.
- **One CU in the thread trace.** ATT covers CU 1 of shader engine 0 for one dispatch per GEMM, and its cycles are wave-cycles on that CU, not kernel time. Its VALU-per-MFMA ratio matches the whole-GPU counters within 0.1% ([Counters That Locate the Limiter](#gfx942-moe-mxfp4-s1-1)).
- **Bit-exactness range.** The check covers e8m0 scales `0x00`–`0xFC`. Scale `0x00` decodes to 0 rather than 2^-127 in both decodes, and `0xFD`–`0xFF` are left out ([Checking Bit-Exactness on MI300X](#gfx942-moe-mxfp4-s3-2)).
- **Innermost source lines.** The line table names the innermost FlyDSL line of each instruction. The `v_perm_b32` lookups land on the one-line `_fp4_perm` wrapper and the LUT scale arithmetic on the final `return` of `_fp4_nibble_to_bf16x8`, so the listing in [What the LUT Decode Compiles To](#gfx942-moe-mxfp4-s3-1) places them under their calling lines by their operands. The categories in Table 4 are our grouping of FlyDSL functions.

(gfx942-moe-mxfp4-followup)=

## Follow-Up Decode Optimizations in AITER

Two later AITER pull requests cut the VALU instructions of the gfx942 decode further and keep every nonzero decoded weight bit-identical to the `#5583` LUT. [ROCm/aiter#6144](https://github.com/ROCm/aiter/pull/6144) [[9]](#gfx942-moe-mxfp4-ref-9), merged as `f2cb249`, looks up E4M3FNUZ fp8 codes instead of bf16 bytes and converts them with the hardware `v_cvt_pk_f32_fp8`, which takes the lookups from 12 `v_perm_b32` to 2 for every 8 weights. [ROCm/aiter#8184](https://github.com/ROCm/aiter/pull/8184) [[10]](#gfx942-moe-mxfp4-ref-10), open as of November 19, 2026, takes that decode from 28 to 21 VALU instructions for every 8 weights.

## Summary

In this blog you explored why MXFP4 MoE weights need a software decode on AMD Instinct MI300X GPUs, and what that decode costs. You profiled the `moe_2stage_a16wmix` GEMMs in AITER with GPU kernel time, SQ counter ratios, ATT, and FlyDSL source-line tables, and traced their VALU-bound profile to a single helper: the E2M1→bf16 weight decode. You then saw that this decode is a fixed bit pattern rather than arithmetic, and how the `v_perm_b32` byte LUT from `#5583` turns it into byte lookups. On MI300X, the LUT makes GEMM1 2.2×–2.4× and GEMM2 1.7×–2.5× faster than the reference software decode, and every nonzero decoded weight stays bit-identical to the software decode for every e8m0 scale up to `0xFC` [[a]](#gfx942-moe-mxfp4-endnote-a).

| Step | What you learn / change | Accuracy | Status |
| --- | --- | --- | --- |
| Kernel time + SQ counters + ATT + FlyDSL line tables | VALU takes 93–98% of issue cycles, and the weight-decode helper issues 92–93% of the GEMM1 instructions on the software path | — | Method |
| Software E2M1→bf16 | f32 rebuild + RNE (including NaN checks that cannot fire) | Reference without the hardware convert | Reference baseline (not shipped) |
| `v_perm_b32` bf16 LUT | Bit-pattern lookup in both GEMMs; VALU per MFMA 72→20 (GEMM1, 8 tokens) | Nonzero decoded weights bit-identical for e8m0 ≤ `0xFC` | [aiter#5583](https://github.com/ROCm/aiter/pull/5583) merged |

If you run MXFP4 MoE models with bf16 activations on MI300X through AITER, you already use the result: since `#5583`, AITER decodes these weights on gfx942 with a table lookup, and the follow-up pull requests keep every nonzero decoded weight bit-identical to it. Beyond this one kernel, you can take away a profiling loop: time kernels on the GPU, use counter ratios such as VALU per MFMA and the wave time ratios to see what the waves are busy with, use ATT to tie those cycles to specific instructions, use the line tables from the compiler to tie the instructions to source lines, and fix the code those lines point at. The commands in [Try It](#gfx942-moe-mxfp4-try-it) are a starting point for running the same loop on your own FlyDSL kernels.

There is more to come. The counters and thread trace in this post already point at the next limits: the exposed waits and the stalls at the scale multiply. Meanwhile, the follow-up pull requests keep cutting the decode itself ([Follow-Up Decode Optimizations in AITER](#gfx942-moe-mxfp4-followup)). Follow [AITER](https://github.com/ROCm/aiter) and [FlyDSL](https://github.com/ROCm/FlyDSL) on GitHub, and check back on ROCm Blogs for more kernel work on AMD Instinct GPUs from our team.

## Endnotes

(gfx942-moe-mxfp4-endnote-a)=

[a] Test Environment

Tables 2–4, 6, and 7, [Checking Bit-Exactness on MI300X](#gfx942-moe-mxfp4-s3-2), and [After the Fix: Counters and Thread Trace](#gfx942-moe-mxfp4-s3-3): testing conducted by AMD on October 30, 2026. GEMM1 and GEMM2 of `moe_2stage_a16wmix` at `E=896, model_dim=3584, inter_dim=384, topk=16`, with bf16 activations, MXFP4 weights, and SwiGLU (gpt-oss form). Every `fused_moe` call passes `quant_dtype_a=dtypes.bf16`: for SwiGLU, `fused_moe` otherwise quantizes the activations to fp4 from about 512 tokens up, which leaves the bf16-activation kernels this post measures. LUT arm: AITER `baef8156a` (the `#5583` merge commit) with FlyDSL 0.3.4.1. Software arm: the same commit with the reference software decode from [Root Cause: The Software E2M1→bf16 Decode](#gfx942-moe-mxfp4-s2) called from `upconvert_b` instead of `_fp4_nibble_to_bf16x8`. Inputs and weights follow the `op_tests/test_flydsl_moe_a16wfp4.py` setup of that commit, and tiles come from its tuned table. The bit-exactness check in [Checking Bit-Exactness on MI300X](#gfx942-moe-mxfp4-s3-2) calls GEMM2 directly and does not involve the activation. Kernel time from rocprofv3 `--kernel-trace`: mean of three repetitions of 100 `fused_moe` calls after 10 warm-up calls. Counters from rocprofv3 `--pmc` in four passes: median of five dispatches after 2 warm-up dispatches. ATT from rocprofv3 `--att` on CU 1 of shader engine 0, one dispatch per GEMM. Source lines (Table 4, [What the Software Decode Compiles To](#gfx942-moe-mxfp4-s2-1), [What the LUT Decode Compiles To](#gfx942-moe-mxfp4-s3-1), and [After the Fix: Counters and Thread Trace](#gfx942-moe-mxfp4-s3-3)): the same kernels rebuilt on November 4, 2026 from the same commits with `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`; the ISA of each rebuilt kernel has the same instructions, in the same order, as the kernel ATT traced, and the line of each instruction is joined to the ATT row of that instruction. Server manufacturers may vary configurations, which can yield different results. Performance may also vary based on the use of the latest drivers and optimizations.

AMD system configuration:

- 1x AMD Instinct™ MI300X, 192 GB HBM3 per GPU
- Host OS: Ubuntu 24.04.4 LTS
- Host GPU driver: amdgpu 7.1.3
- Container: ROCm™ 10.0.0, PyTorch 2.12.0, AITER `baef8156a`, FlyDSL 0.3.4.1

## References

(gfx942-moe-mxfp4-ref-1)=

[1] [OCP Microscaling Formats (MX) v1.0 Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf): defines MXFP4 and the e8m0 scale.

(gfx942-moe-mxfp4-ref-2)=

[2] [AITER](https://github.com/ROCm/aiter): AMD AI Tensor Engine for ROCm, with the `moe_2stage_a16wmix` kernels.

(gfx942-moe-mxfp4-ref-3)=

[3] [FlyDSL](https://github.com/ROCm/FlyDSL): Python domain-specific language for GPU kernels on ROCm.

(gfx942-moe-mxfp4-ref-4)=

[4] [Using rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html): kernel tracing and counter collection.

(gfx942-moe-mxfp4-ref-5)=

[5] [Using thread trace](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html): advanced thread trace (ATT) with rocprofv3.

(gfx942-moe-mxfp4-ref-6)=

[6] [ROCm/aiter#5583: gfx942 `v_perm_b32` E2M1 decode](https://github.com/ROCm/aiter/pull/5583): the LUT decode measured in this post.

(gfx942-moe-mxfp4-ref-7)=

[7] [Matrix Core Programming on AMD CDNA™3 and CDNA™4 architecture](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html): MFMA instructions and their operands.

(gfx942-moe-mxfp4-ref-8)=

[8] [AMD Instinct MI300 CDNA 3 Instruction Set Architecture](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf): including `v_perm_b32`.

(gfx942-moe-mxfp4-ref-9)=

[9] [ROCm/aiter#6144: gfx942 a16w4 MXFP4 decode through an fp8 LUT and the hardware fp8 convert](https://github.com/ROCm/aiter/pull/6144): the first follow-up decode.

(gfx942-moe-mxfp4-ref-10)=

[10] [ROCm/aiter#8184: gfx942 a16w4 MXFP4 decode with a negative LUT, `v_pk_mul_f32` scale, and `v_perm_b32` pack](https://github.com/ROCm/aiter/pull/8184): the second follow-up decode.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, AMD Instinct, AMD CDNA, AMD ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. PyTorch, the PyTorch logo and any related marks are trademarks of The Linux Foundation.
© 2026 Advanced Micro Devices, Inc. All rights reserved
