---
blogpost: true
blog_title: "Serve Kimi-K2.5-MXFP4 on MI355X with ATOM"
date: "16 Jun 2026"
author: "Menghsuan Yang, Clement Lin, Bobo Fang, Eveline Chen, Chunhung Wang"
thumbnail: ''
tags: "AI/ML, Optimization, LLM"
category: "Software tools & optimizations"
target_audience: "AI developers, LLM serving engineers, ROCm users, AMD Instinct customers, performance engineers working on large-scale inference"
key_value_propositions: "Shows how ATOM enables out-of-the-box optimized Kimi-K2.5-MXFP4 serving on AMD Instinct MI355X, including standalone serving, vLLM plugin integration, MXFP4 metadata handling, gfx950 scaled-MFMA dispatch, end-to-end benchmark results, and GSM8K accuracy validation."
language: English
myst:
    html_meta:
        "author": "Menghsuan Yang, Clement Lin, Bobo Fang, Eveline Chen, Chunhung Wang"
        "description lang=en": "Serve Kimi-K2.5-MXFP4 on MI355X with ATOM and gfx950 block-scaled FP4 kernels for optimized LLM inference."
        "keywords": "ROCm, ATOM, AITER, vLLM, Kimi-K2.5, MXFP4, MI355X, gfx950, FP4, block-scaled MFMA, LLM inference, quantization"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Menghsuan Yang, Clement Lin, Bobo Fang, Eveline Chen, Chunhung Wang"
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

# Serve Kimi-K2.5-MXFP4 on MI355X with ATOM

In our previous Kimi-K2.5 posts, we moved from [fused MoE optimization with FlyDSL](../../artificial-intelligence/kimi-k2.5-optimize/README.md) [[8]](#references) to [W4A8 and W8A8 quantization with AMD Quark](../../artificial-intelligence/kimi-k2.5-w4a8/README.md) [[9]](#references). Those posts focused on kernel and quantization work: how to make the dominant MoE path faster, and how to trade off INT4, INT8, and FP8 formats on AMD Instinct(TM) MI300X and MI325X GPUs.

This post shifts the focus to the serving stack. We show how to serve the pre-quantized [`amd/Kimi-K2.5-MXFP4`](https://huggingface.co/amd/Kimi-K2.5-MXFP4) checkpoint [[2]](#references) on AMD Instinct MI355X with [ATOM](https://github.com/ROCm/ATOM) [[3]](#references), so the runtime can recognize MXFP4 metadata, dispatch optimized gfx950 FP4 kernels, and expose an OpenAI-compatible endpoint without manually wiring every model and kernel component.

ATOM connects model-specific execution, [AITER](https://github.com/ROCm/aiter) [[4]](#references) kernels, `torch.compile`, CUDA graph capture, distributed execution, and serving interfaces into a deployable stack. In this post, we use ATOM's standalone OpenAI-compatible server for measurement and describe the vLLM out-of-tree plugin path for users who want to keep [vLLM](https://github.com/vllm-project/vllm) [[5]](#references) as their serving framework.

## ATOM: optimized serving for AMD GPUs

ATOM, short for AiTer Optimized Model, is a lightweight vLLM-like implementation focused on integrating and optimizing AMD GPU model execution with AITER kernels. In practice, ATOM provides the framework layer around the optimized kernels:

- **ROCm and AITER kernel integration**: ATOM routes model operations to AITER kernels, including ASM, CK, Triton, and FlyDSL-backed paths.
- **OpenAI-compatible serving**: ATOM provides `/v1/completions` and `/v1/chat/completions` endpoints for standalone serving.
- **Compilation and graph capture**: ATOM supports piecewise `torch.compile` and CUDA graph capture for low-latency decode.
- **Multi-GPU execution**: ATOM supports tensor parallelism, data parallelism, and expert-parallel execution paths for large MoE models.
- **Quantization-aware dispatch**: ATOM reads HuggingFace quantization metadata and maps supported formats, including MXFP4, to runtime quantization and kernel choices.

For Kimi-K2.5-MXFP4, that last point is key. The checkpoint already contains packed FP4 weights and E8M0 scale tensors. ATOM maps those metadata fields to the `per_1x32` FP4 runtime path and dispatches AITER gfx950 kernels that execute scaled MFMA instructions.

### Standalone ATOM server

ATOM can run as its own OpenAI-compatible server. This mode is useful for benchmarking and for validating new model support because the optimized ATOM model implementation and AITER kernels are directly visible in the serving stack.

```bash
python3 -m atom.entrypoints.openai_server \
    --model /models/Kimi-K2.5-MXFP4 \
    --trust-remote-code \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --host 0.0.0.0 \
    --server-port 18002
```

### ATOM as a vLLM out-of-tree plugin backend

For many deployments, vLLM is the serving framework users already operate. ATOM can also work as a vLLM out-of-tree (OOT) plugin backend: ATOM is installed as a separate Python package and plugs into vLLM through vLLM's official plugin interfaces.

In that mode, vLLM remains the framework-level serving runtime, while ATOM contributes the AMD-optimized model execution path:

- ATOM registers a vLLM platform plugin so vLLM can resolve the ATOM platform.
- ATOM registers selected model wrappers such as `ATOMForCausalLM` and `ATOMMoEForCausalLM`.
- ATOM supplies AITER-backed attention backends for supported models.
- vLLM continues to drive request scheduling and serving, while the hot model execution path runs through ATOM model code and AITER kernels.

This split matters for adoption. The standalone server is the shortest path to validate a new optimized model stack; the vLLM plugin path lets users keep the vLLM interface and operational model while benefiting from ATOM's AMD-specific model and kernel optimizations.

## From W4A8 Quark configs to MXFP4

The W4A8 post used [AMD Quark](https://github.com/amd/quark) [[6]](#references) to produce a checkpoint compatible with the INT8 MFMA runtime path on gfx942. That required a different weight layout from W4A16: INT4 weights with per-channel scales, plus dynamic activation quantization at runtime. The reason was structural: the INT8 MFMA inner loop could not apply a different floating-point weight scale every 32 K values, so W4A8 moved weight scales to a per-channel layout that could be applied in the epilogue.

For MXFP4 on gfx950, the config changes in a different direction. Instead of coarsening the weight scale to make an INT8 epilogue work, MXFP4 keeps a fine-grained 32-value block scale and uses a hardware path that can consume that block scale in the matrix instruction.

At a high level, the Quark-style configuration changes look like this:

| Config choice | W4A8 direction | MXFP4 direction |
| --- | --- | --- |
| Weight dtype | INT4 | FP4 |
| Activation dtype | Dynamic 8-bit activation path | Dynamic FP4 activation path |
| Weight scale granularity | Per-channel | Per-group, `group_size=32` |
| Scale format | Floating-point scale tensors for INT4/activation dequantization | E8M0 block scales |
| Observer style | INT4/FP8 quantization specs | `PerBlockMXObserver` |
| Runtime target | INT8 MFMA with scale application in the epilogue | gfx950 scaled FP4 MFMA with payload and scale operands |

For MXFP4, both weights and input activations use `dtype=fp4`, `qscheme=per_group`, `group_size=32`, and `scale_format=e8m0`. The weights are stored statically in the checkpoint as packed FP4 payloads plus E8M0 `weight_scale` tensors; activations are quantized dynamically at runtime into the same 1x32 block-scaled representation.

That makes the checkpoint and runtime contract different from W4A8. W4A8 changes scale granularity so the INT8 path can stay efficient on gfx942. MXFP4 keeps per-32 block scaling because gfx950 adds the scaled MFMA family needed to consume those scales directly.

## The Kimi-K2.5-MXFP4 model on gfx950

MXFP4 combines two ideas:

- **4-bit element storage**: the value payload uses FP4 E2M1, packed two elements per byte.
- **Microscaling**: each block of 32 K-dimension values carries an E8M0 scale, giving the block a shared exponent.

The Kimi-K2.5-MXFP4 model configuration uses this scheme for both activations and weights:

| Field | Value observed in `config.json` |
| --- | --- |
| `dtype` | `fp4` |
| `qscheme` | `per_group` |
| `group_size` | `32` |
| `scale_format` | `e8m0` |
| `observer_cls` | `PerBlockMXObserver` |

### The opportunity: block-scaled FP4 MFMA

The W4A8 post focused on a gfx942 question: how do we move the MoE path from BF16 MFMA to an 8-bit MFMA path without losing too much accuracy? MXFP4 asks the next-generation gfx950 version of that question: if the hardware can consume block-scaled 4-bit floating-point operands directly, can we keep both the compact payload and a fine-grained scale layout in the matrix core?

That is the important architectural change. On CDNA3/gfx942, the common choices were BF16, INT8, FP8, and BF8 MFMA forms. On CDNA4/gfx950, the MFMA family adds block-scaled low-precision formats, including FP4, FP6, and BF6, with an E8M0 block exponent shared by 32 K-dimension elements [[7]](#references).

| Path | Value payload | Scale placement | Matrix instruction implication |
| --- | ---: | --- | --- |
| BF16 | 16 bits/value | No external block scale | Stable, simple, but higher data movement |
| FP8 / W8A8 | 8 bits/value | Encoded in FP8 values and runtime scaling path | Mature 8-bit floating-point path |
| W4A8 | 4-bit weights, 8-bit activations | Weight and activation scales applied after INT8 accumulation | Efficient on gfx942, but per-group weight scales do not fit the INT8 inner loop |
| MXFP4 | 4-bit FP4 payload, packed two values per byte | E8M0 scale per 32 K values | gfx950 scaled MFMA can consume payloads and block scales together |

Compared with FP8, the main MXFP4 advantage is not just "a smaller number." It is a different memory/computation trade-off:

1. **Half the element bytes of FP8**: FP4 payloads are 4 bits instead of 8 bits, reducing weight and activation traffic for quantized matrix paths.
2. **Hardware-visible block scales**: gfx950 can consume the FP4 payload and the E8M0 scale operands in scaled MFMA instructions, rather than requiring a full software dequantization back to BF16 first.
3. **Better range than raw FP4**: the shared E8M0 exponent lets a 32-element block shift its dynamic range, which is essential for using FP4 in large MoE models.

There is still a trade-off. FP8 keeps more mantissa/exponent information per element and is often simpler for accuracy-sensitive paths. MXFP4 wins when the model can tolerate the lower per-element precision and when memory bandwidth, model footprint, or MoE weight traffic dominate.

### MXFP4 tensor anatomy

MXFP4 is easiest to understand as a two-tensor representation for each quantized matrix:

```text
FP4 payload:  [two E2M1 values packed into each uint8 byte]
E8M0 scales: [one uint8 scale for every 32 original K-dimension values]
```

For a representative Kimi-K2.5 expert `gate_proj` or `up_proj` shard, the logical weight matrix has 2,048 output channels and 7,168 K-dimension values per row. In the MXFP4 checkpoint, this appears as:

| Tensor | Stored dtype | Stored shape | What it represents |
| --- | --- | ---: | --- |
| `weight` | `uint8` | `[2048, 3584]` | Two FP4 E2M1 values packed per byte, so the K dimension is halved in storage |
| `weight_scale` | `uint8` | `[2048, 224]` | One E8M0 scale for each block of 32 original K values (`7168 / 32 = 224`) |

For a representative expert `down_proj` shard, the same rule applies in the other projection shape:

| Tensor | Stored dtype | Stored shape | What it represents |
| --- | --- | ---: | --- |
| `weight` | `uint8` | `[7168, 1024]` | Packed FP4 payload for 2,048 logical K values |
| `weight_scale` | `uint8` | `[7168, 64]` | One E8M0 scale for each 32-value block (`2048 / 32 = 64`) |

The scale overhead is small compared with the payload. A 32-value block needs 16 bytes of FP4 payload plus 1 byte of E8M0 scale, or about 0.53125 bytes per original value before additional tensor metadata and alignment effects. The key point is that this scale is not a dequantization side table that must disappear before compute. On gfx950, it is part of the block-scaled operand stream consumed by the FP4 matrix instruction.

```text
BF16 activations
      |
      v
dynamic 1x32 MXFP4 quantization
      |
      +--> activation FP4 payload --+
      +--> activation E8M0 scales --+
                                    +--> gfx950 V_MFMA_SCALE_F32_* --> accumulator/output
      +--> weight FP4 payload ------+
      +--> weight E8M0 scales ------+
```

**Figure 1.** MXFP4 dataflow on gfx950. Both activations and weights are represented as packed FP4 payloads plus E8M0 block scales. The important difference from a storage-only FP4 path is that the block scales stay visible to the matrix instruction through the gfx950 scaled MFMA family.

### Choosing the right weight layout

The previous W4A8 post framed quantization as a layout problem as much as a datatype problem: the target MFMA instruction determines where scales can be applied, and that determines which checkpoint layout is efficient at runtime. MXFP4 follows the same logic, but with a different answer on gfx950.

| Format | Weight payload | Activation path | Scale layout | Runtime compute path | Best fit |
| --- | --- | --- | --- | --- | --- |
| **W4A16** | INT4 | BF16 | Per-group, `group_size=32` | Dequantize to BF16, then BF16 MFMA | Accuracy-oriented INT4 serving when BF16 compute is acceptable |
| **W4A8** | INT4 | BF16 to INT8 dynamically | Per-channel weight scales plus per-token activation scales | INT8 MFMA with scale application in the epilogue | Smaller weight footprint with 8-bit compute on gfx942 |
| **W8A8** | FP8 | FP8 | FP8 weight and activation path | FP8 MFMA through AITER CK/ASM kernels | Simpler low-precision pipeline when HBM capacity allows |
| **MXFP4** | FP4 E2M1 packed two values per byte | FP4 dynamic quantization for quantized matrix paths | E8M0 scale per 32 K-dimension values | gfx950 scaled MFMA with FP4 payloads and block scales | Lowest payload width with hardware-visible block scaling |

The key MXFP4 difference is that the scale is not merely checkpoint metadata. The E8M0 scale is part of the block-scaled compute contract: ATOM maps the model metadata to 1x32 FP4 quantization, and AITER can dispatch gfx950 kernels that keep the FP4 payload and scale operands close to the matrix instruction.

### When is the scale applied? The gfx950 scaled-MFMA difference

The most important question is not only "how many bits does each value use?" It is "where does the scale enter the GEMM?"

That placement determines whether a scale layout is cheap, expensive, or structurally incompatible with the target MFMA instruction.

**W4A16: scale is absorbed before MFMA.**

In the W4A16 path, INT4 weights are dequantized into BF16 before the matrix instruction:

```text
w_bf16 = int4_to_bf16(w_int4) * group_scale[n, k // 32]
acc += w_bf16 * act_bf16
```

After dequantization, the scale has been baked into the BF16 value. The BF16 MFMA instruction sees only BF16 operands; it does not know whether the original checkpoint had one scale per row or one scale per 32 values. This makes fine-grained per-group scaling accuracy-friendly, but it also keeps the compute path in BF16.

**W4A8 on gfx942: no block-scaled MFMA, so scale is deferred until the epilogue.**

In the W4A8 INT8 path, the kernel wants to keep the inner loop entirely in the integer domain:

```text
w_int8 = sign_extend(w_int4)          # no floating-point scale here
int32_sum += w_int8 * act_int8        # INT8 MFMA, accumulates to INT32
output = weight_scale[n] * act_scale[token] * int32_sum
```

The weight scale cannot be applied before INT8 MFMA, because multiplying by a floating-point scale would turn the operand into a floating-point value. The gfx942 W4A8 path also does not have a block-scaled MFMA instruction that can consume INT4 or INT8 payloads together with per-32 scale operands. The scale therefore moves to the epilogue, after the dot product is complete. That is why W4A8 prefers per-channel weight scales: one row scale can be applied once at the end. A per-group scale every 32 K values would force the dot product into many separately scaled partial sums, which defeats the clean INT8 inner loop.

**MXFP4 on gfx950: scaled MFMA makes per-32 scaling a hardware compute path.**

MXFP4 takes a different route. Its E8M0 scale is neither fully baked into a BF16 value before MFMA nor delayed as a single correction after accumulation. The scale is part of the block-scaled FP4 operand representation:

```text
w_fp4x2, w_scale_e8m0 = load_mxfp4_weight_block(...)
a_fp4x2, a_scale_e8m0 = dynamic_quantize_activation_block(...)
acc = scaled_mfma(w_fp4x2, a_fp4x2, w_scale_e8m0, a_scale_e8m0)
```

This is the key gfx950 distinction. CDNA4 adds the `V_MFMA_SCALE_F32_*` family for block-scaled low-precision inputs, including FP4 [[7]](#references). The scaled MFMA path can consume low-precision FP4 payloads together with their block scales, so a 1x32 scale layout remains compatible with the high-throughput matrix instruction. In other words, MXFP4 keeps the local dynamic-range benefit of per-32 scaling without reverting the main compute path to BF16 dequantization or forcing all scale application into a late epilogue.

| Format | Scale applied | Consequence |
| --- | --- | --- |
| **W4A16** | Before MFMA, during dequantization to BF16 | Fine-grained per-group scales are natural, but compute uses BF16 MFMA |
| **W4A8** | After INT8 MFMA, in the epilogue | Efficient INT8 inner loop, but weight scales must be coarse enough to apply after accumulation |
| **W8A8** | Encoded in FP8 values and handled by FP8 compute path | Simple 8-bit floating-point pipeline, with a wider payload than FP4 |
| **MXFP4** | Alongside FP4 payloads in gfx950 `V_MFMA_SCALE_F32_*` | Per-32 block scaling stays in the matrix path with 4-bit payloads |

This is why MXFP4 is more than "FP8 but smaller." The format changes where scale arithmetic lives, and gfx950 is the hardware generation that turns that scale placement into a first-class MFMA path. On gfx950, the serving stack can keep FP4 values compact while still giving each 32-value block its own exponent-like scale at compute time.

### Checkpoint layout and quantization metadata

The MXFP4 checkpoint carries the runtime contract in `config.json`, not only in tensor names. Both weights and activations use the same high-level quantization scheme:

| Config field | Weight | Input activation |
| --- | --- | --- |
| `dtype` | `fp4` | `fp4` |
| `is_dynamic` | `false` | `true` |
| `qscheme` | `per_group` | `per_group` |
| `group_size` | `32` | `32` |
| `scale_format` | `e8m0` | `e8m0` |
| `observer_cls` | `PerBlockMXObserver` | `PerBlockMXObserver` |

This split is important:

- **Weights are static**: the packed FP4 payload and E8M0 `weight_scale` tensors are stored in the checkpoint.
- **Activations are dynamic**: ATOM quantizes live BF16 activation blocks into FP4 payloads and E8M0 scales at runtime.
- **Some layers remain outside MXFP4**: the checkpoint excludes attention projections, MoE router gates, the language model head, and vision/projector components from this global FP4 quantization path. That keeps precision-sensitive or non-MoE components on their own kernels while concentrating MXFP4 on the matrix-heavy MLP and expert projections.

In the checkpoint index, quantized modules appear as paired `weight` and `weight_scale` tensors. For Kimi-K2.5-MXFP4, the index contains 69,303 `weight_scale` tensors, matching 69,303 modules with stored block-scale metadata. That is the checkpoint-side reason ATOM can map the model to `QuantType.per_1x32` rather than treating the FP4 payload as a generic packed byte tensor.

## Runtime pipeline: from BF16 activations to scaled FP4 MFMA

The MXFP4 serving path matters because it keeps FP4 payloads close to the matrix core instead of expanding the model back to a wider format before the main compute. At runtime, the path has three connected stages: parse the checkpoint metadata, quantize live activations into the same 1x32 block-scaled format, then dispatch GEMM and MoE kernels that use gfx950 scaled MFMA.

### Stage A: Quantization config parsing

ATOM maps `per_group` quantization to `QuantType.per_1x32`. The same parser recognizes `fp4`, `float4`, and `mxfp4` as `fp4x2`, then forces MXFP4 into `per_1x32` block scaling when needed.

This matches the model config: FP4 values with group size 32 and E8M0 scales.

### Stage B: Dynamic activation quantization

For weight tensors, the FP4 payload and E8M0 scales are already stored in the checkpoint. Activations are different: they are produced live in BF16 by the surrounding transformer layers, so ATOM dynamically quantizes activation blocks at inference time.

For the MXFP4 linear path, ATOM calls `get_hip_quant(QuantType.per_1x32)`, views activations as `torch.float4_e2m1fn_x2`, views scales as `torch.float8_e8m0fnu`, and passes both into the low-precision GEMM path. Conceptually:

```text
hidden_bf16
  -> dynamic 1x32 FP4 quantization
  -> activation_fp4x2 + activation_e8m0_scale
  -> scaled FP4 GEMM
```

This is the activation-side counterpart to the checkpoint's packed `weight` and `weight_scale` tensors.

### Stage C: MXFP4 linear GEMM

During warmup, AITER loaded multiple gfx950 `f4gemm` kernels, including:

```text
f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co
f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128.co
f4gemm_bf16_per1x32Fp4_BpreShuffle_128x128.co
```

The loaded FP4 GEMM path uses the block-scaled MFMA instruction family:

```text
v_mfma_scale_f32_16x16x128_f8f6f4 ...
```

This is the key gfx950 instruction family for the FP4 path: the kernel is not simply converting FP4 weights to BF16 and using BF16 MFMA; it is using the scaled MFMA form that accepts low-precision block-scaled operands.

### Stage D: MXFP4 fused MoE

ATOM's `Mxfp4MoEMethod` creates FP4 expert weights and uint8 scale tensors with a hard-coded MXFP4 block size of 32. For MoE execution, it passes:

- `quant_type=self.quant_type`
- `w1_scale=layer.w13_weight_scale`
- `w2_scale=layer.w2_weight_scale`

into AITER `fused_moe`.

AITER selected FlyDSL 2-stage fused MoE kernels with FP4 dtypes:

```text
flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_bnt0_xcd4
flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce_xcd4
```

For smaller token regimes, the run also used:

```text
flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w4
flydsl_moe2_afp4_wfp4_bf16_t32x256x256_atomic_bnt2
```

The dispatch line included:

```text
torch.float4_e2m1fn_x2, torch.float4_e2m1fn_x2, QuantType.per_1x32
```

Together, these linear and MoE paths show the core benefit of MXFP4 on gfx950: FP4 payloads and 1x32 scaling can be carried into the compute kernels instead of serving only as a compressed checkpoint format.

### Verifying scaled MFMA in the generated kernels

The scaled-MFMA claim was verified by disassembling representative gfx950 FP4 code objects used by the ATOM/AITER stack. The FP4 GEMM kernels contain `v_mfma_scale_f32_16x16x128_f8f6f4` instructions, for example:

```text
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v208, v200 ...
```

Representative `f4gemm_bf16_per1x32Fp4_*` code objects contained hundreds of `v_mfma_scale_f32_16x16x128_f8f6f4` instructions. The MXFP4 MoE code objects also contained the same scaled MFMA family, with typical counts in the hundreds per code object. This confirms that the serving path is not merely loading a compressed FP4 checkpoint and expanding it to BF16 before the matrix core; the gfx950 kernels really do execute the block-scaled MFMA path.

### Non-MXFP4 kernels still appear

The serving stack also includes supporting kernels that are not MXFP4:

- `mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co` for MLA decode
- BF16 GEMM fallback kernels such as `bf16gemm_fp32bf16_tn_80x64_splitk_clean.co`
- normalization, quantization, cache, and sampling modules

This is expected. The model is not "FP4 everywhere." The MXFP4 path is concentrated in quantized linear/MoE compute, while attention, routing, normalization, logits, and other framework components still use their own precision choices.

## End-to-end performance on 8x MI355X

We benchmarked the ATOM OpenAI-compatible server using the same input/output shape used in the previous Kimi-K2.5 ROCm blog posts:

- TP = 8
- GPU = 8x MI355X / gfx950
- ISL/OSL = 10,240 / 512
- low concurrency = 2, `num_prompts = 10`
- high concurrency = 40, `num_prompts = 160`
- `request_rate = inf`
- `random_range_ratio = 1.0`

This keeps the benchmark shape aligned with the earlier MI300X and MI325X posts. The hardware and serving stack are different, so the numbers below should be read as the MI355X/gfx950 MXFP4 result under the same workload shape, not as a same-system speedup over W4A16, W4A8, or W8A8.

### Test environment

| Component | Configuration |
| --- | --- |
| Hardware | 8x AMD Instinct MI355X, gfx950 |
| Serving stack | ATOM OpenAI-compatible server with AITER kernels |
| Model | `amd/Kimi-K2.5-MXFP4` |
| Tensor parallelism | TP = 8 |
| KV cache dtype | `fp8` |
| Docker image | `rocm/atom-dev:latest@sha256:eba8c90854e5ce6a6f5b5ac4d35d41f7df76981229e5d91f32a9d2176c52a2e8` |

### Launching the server

The following command shows the serving setup used for the 8-GPU run. Replace `/models/Kimi-K2.5-MXFP4` with the location where the MXFP4 checkpoint is mounted inside the container.

```bash
export IMAGE="rocm/atom-dev:latest@sha256:eba8c90854e5ce6a6f5b5ac4d35d41f7df76981229e5d91f32a9d2176c52a2e8"
export MODEL_DIR="/models/Kimi-K2.5-MXFP4"
export PORT=18002

docker run --rm --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host --shm-size=256G \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -v /path/to/Kimi-K2.5-MXFP4:${MODEL_DIR}:ro \
    ${IMAGE} \
    bash -lc "cd /app/ATOM && \
      python3 -m atom.entrypoints.openai_server \
        --model ${MODEL_DIR} \
        --trust-remote-code \
        -tp 8 \
        --kv_cache_dtype fp8 \
        --host 0.0.0.0 \
        --server-port ${PORT}"
```

### Benchmark commands

We use the same random input/output lengths as the earlier Kimi-K2.5 posts: 10,240 input tokens and 512 output tokens.

**Prefill-dominated (max concurrency = 2):**

```bash
python3 -m atom.benchmarks.benchmark_serving \
    --model=/models/Kimi-K2.5-MXFP4 \
    --backend=vllm \
    --base-url=http://127.0.0.1:18002 \
    --trust-remote-code \
    --dataset-name=random \
    --random-input-len=10240 \
    --random-output-len=512 \
    --random-range-ratio=1.0 \
    --num-prompts=10 \
    --max-concurrency=2 \
    --request-rate=inf \
    --percentile-metrics=ttft,tpot,itl,e2el
```

**Decode-dominated (max concurrency = 40):**

```bash
python3 -m atom.benchmarks.benchmark_serving \
    --model=/models/Kimi-K2.5-MXFP4 \
    --backend=vllm \
    --base-url=http://127.0.0.1:18002 \
    --trust-remote-code \
    --dataset-name=random \
    --random-input-len=10240 \
    --random-output-len=512 \
    --random-range-ratio=1.0 \
    --num-prompts=160 \
    --max-concurrency=40 \
    --request-rate=inf \
    --percentile-metrics=ttft,tpot,itl,e2el
```

**Concurrency = 2 (prefill-dominated):**

| Metric | Statistic | MXFP4 |
| --- | --- | ---: |
| **TTFT (ms)** ↓ | Mean | 421.24 |
| | Median | 380.50 |
| | P99 | 607.20 |
| **TPOT (ms/token)** ↓ | Mean | 11.30 |
| | Median | 11.15 |
| | P99 | 12.49 |
| **Output throughput (tok/s)** ↑ | - | 165.30 |

**Concurrency = 40 (decode-dominated):**

| Metric | Statistic | MXFP4 |
| --- | --- | ---: |
| **TTFT (ms)** ↓ | Mean | 4,167.25 |
| | Median | 3,992.10 |
| | P99 | 8,897.46 |
| **TPOT (ms/token)** ↓ | Mean | 29.34 |
| | Median | 29.57 |
| | P99 | 39.36 |
| **Output throughput (tok/s)** ↑ | - | 1,055.95 |

The low-concurrency case highlights first-token latency and per-token decode cost when only two requests are active. The high-concurrency case stresses parallel decoding, where aggregate output throughput rises to 1,055.95 tok/s while mean TPOT increases to 29.34 ms/token.

These are end-to-end serving measurements, not isolated GEMM or MoE microbenchmarks. They include request scheduling, attention, MoE, sampling, KV cache management, and framework overhead around the MXFP4 kernels.

### How to interpret these results

The strongest claim supported by this experiment is that Kimi-K2.5-MXFP4 can be served end-to-end on 8x MI355X with ATOM while exercising gfx950 scaled-MFMA FP4 kernels in the matrix-heavy paths. The benchmark includes the full serving stack, so it is more relevant to deployment than an isolated GEMM microbenchmark.

The comparison below is useful for context, but it is not a controlled same-system speedup experiment. The previous W4A16, W4A8, and W8A8 rows were measured on MI325X with SGLang/AITER, while the MXFP4 row is measured on MI355X with ATOM/AITER. That means several variables change at once: hardware generation, quantization format, serving framework, kernel dispatch path, and runtime scheduler behavior.

For this reason, the MXFP4 numbers should be read as a gfx950 enablement result and a method-aligned workload comparison. A strict speedup claim would require a same-system baseline, such as another Kimi-K2.5 format on the same 8x MI355X ATOM stack, or an MXFP4 fallback path on the same hardware.

### Method-aligned comparison with the previous post

The table below mirrors the benchmark comparison style from the W4A8/W8A8 post, but the interpretation is different. The W4A16, W4A8, and W8A8 rows were reported on an 8x MI325X SGLang/AITER setup; the MXFP4 row here is measured on an 8x MI355X ATOM/AITER setup. The workload shape and GPU count are aligned, but the hardware generation, quantization format, and serving stack are not identical, so this is context rather than a same-system speedup claim.

**Concurrency = 2 (prefill-dominated):**

| Format | System / stack | Weight layout | Mean TTFT (ms) ↓ | Mean TPOT (ms/token) ↓ | Output throughput (tok/s) ↑ |
| --- | --- | --- | ---: | ---: | ---: |
| W4A16 | 8x MI325X / SGLang | INT4 per-group, BF16 compute | 723.96 | 20.17 | 92.78 |
| W4A8 | 8x MI325X / SGLang | INT4 per-channel, INT8 compute | 695.30 | 16.95 | 109.35 |
| W8A8 | 8x MI325X / SGLang | FP8 weights and activations | 747.52 | 14.25 | 127.47 |
| MXFP4 | 8x MI355X / ATOM | FP4 E2M1 with E8M0 1x32 scales | 421.24 | 11.30 | 165.30 |

**Concurrency = 40 (decode-dominated):**

| Format | System / stack | Weight layout | Mean TTFT (ms) ↓ | Mean TPOT (ms/token) ↓ | Output throughput (tok/s) ↑ |
| --- | --- | --- | ---: | ---: | ---: |
| W4A16 | 8x MI325X / SGLang | INT4 per-group, BF16 compute | 10,183.74 | 51.99 | 556.94 |
| W4A8 | 8x MI325X / SGLang | INT4 per-channel, INT8 compute | 9,801.37 | 46.78 | 607.24 |
| W8A8 | 8x MI325X / SGLang | FP8 weights and activations | 9,764.04 | 53.57 | 551.12 |
| MXFP4 | 8x MI355X / ATOM | FP4 E2M1 with E8M0 1x32 scales | 4,167.25 | 29.34 | 1,055.95 |

The comparison highlights why MXFP4 is interesting on gfx950: it combines the small payload width of 4-bit weights with a scale layout that the hardware can consume directly in the FP4 matrix path. That is the key distinction from both W4A8, which changes scale granularity to make INT8 MFMA efficient, and W8A8, which uses a wider FP8 payload to keep the compute pipeline simple.

### When should you choose MXFP4?

MXFP4 is attractive when the deployment target has gfx950 block-scaled MFMA support and the model is available in a checkpoint layout that preserves FP4 payloads plus E8M0 scales. In that setting, MXFP4 gives the serving stack a compact 4-bit payload without forcing per-32 scales to be baked into BF16 before compute or delayed into a single epilogue scale.

| Format | Best fit | Main trade-off |
| --- | --- | --- |
| **W4A8** | gfx942-era deployments where INT8 MFMA is the optimized path and INT4 weight footprint matters | Per-channel weight scales are needed for the INT8 inner loop, which is coarser than per-group scaling |
| **W8A8** | Deployments that prefer a simpler FP8 pipeline and can afford the larger weight footprint | FP8 payloads are wider than FP4, so memory traffic and model footprint are higher |
| **MXFP4** | gfx950 deployments that can use block-scaled FP4 MFMA and want the smallest payload width with per-32 block scaling | Requires gfx950 scaled-MFMA support and an MXFP4-aware serving stack |

In practice, MXFP4 is a good candidate when weight bandwidth, MoE expert traffic, or model footprint are important constraints, and when the workload can tolerate FP4 element precision with E8M0 block scaling. FP8 remains a simpler choice when maximum precision margin or broader kernel availability is more important than the smallest payload. W4A8 remains useful for gfx942 paths where INT8 MFMA is the available high-throughput option.

## Accuracy validation: GSM8K

To validate that the ATOM MXFP4 path preserves model quality on a downstream reasoning benchmark, we ran GSM8K 10-shot evaluation against the same ATOM server. This matches the GSM8K protocol used in the previous W4A8/W8A8 post. The run used greedy generation (`do_sample=False`, `temperature=0.0`) and evaluated the full GSM8K test set of 1,319 samples.

The evaluation used `lm-eval` through the server's OpenAI-compatible completions endpoint:

```bash
lm_eval \
    --model local-completions \
    --model_args model=/models/Kimi-K2.5-MXFP4,base_url=http://127.0.0.1:18002/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=32 \
    --tasks gsm8k \
    --num_fewshot 10 \
    --batch_size 1
```

The `strict-match` filter follows the canonical GSM8K answer marker, while `flexible-extract` extracts the final numeric answer more permissively. Reporting both filters makes the result easier to compare with the earlier Kimi-K2.5 posts.

| Benchmark | Filter | n-shot | Metric | Value | Stderr |
| --- | --- | ---: | --- | ---: | ---: |
| GSM8K | flexible-extract | 10 | exact_match | 0.9401 | 0.0065 |
| GSM8K | strict-match | 10 | exact_match | 0.9401 | 0.0065 |

The result shows that the MXFP4 serving path maintains strong downstream reasoning quality while using the lower-precision FP4 payload and gfx950 block-scaled compute path.

## Summary

This third Kimi-K2.5 optimization note extends the earlier MI300X/MI325X story to MI355X and gfx950 by moving from kernel and quantization work to optimized serving:

1. **ATOM makes the optimized path deployable**: users can run a standalone OpenAI-compatible server, while the vLLM OOT plugin path lets vLLM remain the serving framework and ATOM provide AMD-optimized model execution.
2. **The checkpoint is truly MXFP4**: the model configuration records FP4 tensors with `group_size=32` and E8M0 scales, and stores packed FP4 payloads plus `weight_scale` tensors for quantized modules.
3. **ATOM parses MXFP4 into a 1x32 FP4 execution path**: `per_group` maps to `QuantType.per_1x32`, and MXFP4 is represented as `torch.float4_e2m1fn_x2` plus `torch.float8_e8m0fnu` scales at runtime.
4. **AITER dispatches gfx950 FP4 kernels**: the serving path uses `f4gemm` kernels and FlyDSL FP4 fused MoE kernels built around gfx950 scaled MFMA support.
5. **End-to-end serving is strong on 8x MI355X**: under the aligned 10,240/512 workload, MXFP4 reached 165.30 output tok/s at concurrency 2 and 1,055.95 output tok/s at concurrency 40.
6. **Accuracy is maintained on GSM8K**: the same ATOM server scored 94.01% flexible-extract exact match on GSM8K 10-shot (`0.9401 +/- 0.0065`) on the full 1,319-sample test set.

The broader takeaway is that gfx950 turns MXFP4 into a first-class compute format, and ATOM turns that hardware capability into a serving path. Compared with FP8, MXFP4 reduces data movement through 4-bit payloads while preserving dynamic range through block scales; compared with older FP4-as-storage approaches, gfx950 scaled MFMA lets the kernel keep the low-precision representation much closer to the matrix core. ATOM is the framework layer that makes this usable: it loads the HuggingFace checkpoint, maps quantization metadata to runtime operators, chooses the AITER/FlyDSL kernels, exposes a serving API, and provides benchmark tooling for the full stack rather than isolated kernels.

## References

[1] [Kimi-K2.5 on HuggingFace](https://huggingface.co/moonshotai/Kimi-K2.5)

[2] [Kimi-K2.5-MXFP4 on HuggingFace](https://huggingface.co/amd/Kimi-K2.5-MXFP4)

[3] [ATOM](https://github.com/ROCm/ATOM)

[4] [AITER](https://github.com/ROCm/aiter)

[5] [vLLM](https://github.com/vllm-project/vllm)

[6] [AMD Quark](https://github.com/amd/quark)

[7] [AMD Instinct CDNA4 Instruction Set Architecture Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)

[8] [Previous ROCm blog: Accelerating Kimi-K2.5 on AMD Instinct MI300X: Optimizing Fused MoE with FlyDSL](../../artificial-intelligence/kimi-k2.5-optimize/README.md)

[9] [Previous ROCm blog: Further Accelerating Kimi-K2.5 on AMD Instinct MI325X: W4A8 and W8A8 Quantization with AMD Quark](../../artificial-intelligence/kimi-k2.5-w4a8/README.md)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
