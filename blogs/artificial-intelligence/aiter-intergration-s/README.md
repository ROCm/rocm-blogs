---
blogpost: true
blog_title: "Accelerate DeepSeek-R1 Inference: Integrate AITER into SGLang"
date: 16 May 2025
author: 'Bruce Xue, George Wang'
thumbnail: 'Thumbn.jpg'
tags: Optimization
category: Applications & models
target_audience: AI enthusiast and developers
key_value_propositions: AITER-based optimizations in SGLang for Deepseek architecture
language: English
myst:
    html_meta:
        "author": "Bruce Xue, George Wang"
        "description lang=en": "Boost DeepSeek-R1 with AITER: Step-by-step SGLang integration for high-performance MoE, GEMM, and attention ops on AMD GPUs"
        "keywords": "AITER, SGLang, Deepseek"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends, AI & Intelligent Systems"
        "amd_blog_authors": "Bruce Xue, George Wang"
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

# Accelerate DeepSeek-R1 Inference: Integrate AITER into SGLang

To achieve optimized LLM performance on GPUs, high-performance AI
operators/kernels are very critical. AMD recently announced
[AITER](https://github.com/ROCm/aiter), a centralized repository
designed to accelerate AI workloads by providing a unified collection of
high-performance AI operators. It serves as a comprehensive hub for
customer-level operator requests, supporting diverse needs across
private, public, or custom frameworks. With both C++ and Python APIs,
AITER enables developers to focus on operator development while offering
flexible backend kernel implementations using Triton, CK, or assembly.
AITER supports inference, training kernels, GEMM, and communication
kernels, allowing flexibility across different kernel-framework pairings
and architectural limitations. In this blog we will provide a
comprehensive, step-by-step hands-on guide on integrating AITER
operators into SGLang for DeepSeek-R1.
[SGLang](https://github.com/sgl-project/sglang) is a fast serving
framework for large language and vision language models. For
DeepSeek-R1, SGLang incorporates MLA (Multi-Head Latent Attention)
optimizations and supports FP8 precision (specifically W8A8 format).
These enhancements enable the identification of target modules that can
be replaced with AITER-optimized solutions, improving overall efficiency
and performance. AITER integration delivers significant performance
improvements across the entire inference pipeline while maintaining full
functional equivalence with the original architecture.

## AITER Optimization in SGLang

<!-- markdownlint-disable -->

For Deepseek R1\'s architecture, we have implemented AITER-based
optimizations in SGLang by replacing the following key components:

1.  **Complete MoE** **implementation**

    - Top-K routing

    - MoE sorting

    - BlockScale FP8 MoE computation

2.  **General linear layers**

    - FP8 Blockscale GEMM

3.  **Attention mechanisms**

    - Decode phase: Latent Attention (MLA)

    - Prefill phase: Multi-Head Attention (MHA)

4.  **Other operators**

    - Customer All Reduce
      
<!-- markdownlint-restore -->

## Operator Integration for AITER Library

The AITER library provides complete operator implementation with
well-documented interfaces under
[op_tests](https://github.com/BruceXcluding/aiter/blob/aiter_customer/op_tests)
for each operator as shown in Figure 1. For integration, users can
locate corresponding modules containing:

1. Operator APIs,

2. Tensor layout specifications,

3. Framework integration entry point

```{figure} ./images/image1.png
:align: center
:alt: Scaling performance
Figure 1: Hierarchy to navigate op_tests
```

Note : *This blog is based on a private repository and presents a fully
integrated AITER solution for SGLang version v0.4.4. Please refer to
private repository for details on [Customer
SGLang](https://github.com/BruceXcluding/sglang/tree/aiter_customer_v0.4.4)
and [Customer
AITER](https://github.com/BruceXcluding/aiter/tree/aiter_customer)*

## Complete MoE Implementation

<!-- markdownlint-disable -->
**Top-K** **routing**
<!-- markdownlint-restore -->

AITER provides a fused biased grouped topk implementation with a HIP
kernel. This function computes the expert selection probability for each token in the MoE layer. The grouped Top-K mechanism
used in DeepSeek-R1/V3, which allows each token to select only a fixed
number of expert groups and then selects Top-K experts from each expert.

Please refer to
[op_tests/test_moeTopkSoftmax.py](https://github.com/BruceXcluding/aiter/blob/master/op_tests/test_moeTopkSoftmax.py) for implementation.

```python

...
def test_biased_grouped_topk(
    token, expert, group, topk, topk_group, need_renorm, dtype, scale_factor=1.0
):
...
    w_ref = w_ref * scale_factor
    w_aiter = torch.empty_strided((token, topk), (topk + 10, 1), dtype=torch.float32)
    id_aiter = torch.empty_strided((token, topk), (topk + 10, 1), dtype=torch.int32)
    _, us_aiter = run_perftest(
        aiter.biased_grouped_topk,
        gating_output,
        correction_bias,
        w_aiter,
        id_aiter,
        group,
        topk_group,
        need_renorm,
        scale_factor,
    )
...
```

APIs entry point aiter/ops/topk.py:

```python
...
def biased_grouped_topk(
    gating_output: Tensor,
    correction_bias: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    routed_scaling_factor: float=1.0  # mul to topk_weights
): ...

```
<!-- markdownlint-disable -->
**MoE Sorting**
<!-- markdownlint-restore -->

AITER includes a Mixture-of-Experts (MoE) sorting implementation using a composable kernel (ck) for MoE alignment. Gating functions are used to route activations based on top-k gating logits. This functionality depends on the sorting logic provided by the MoE align and sort mechanism.

Please refer to
[op_tests/test_moe_blockscale.py](https://github.com/BruceXcluding/aiter/blob/aiter_customer/op_tests/test_moe_blockscale.py) for implementation

```python

def asm_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,
    fc2_scale=None,
    a1_scale=None,
    scale_blk=(128, 128),
):

    model_dim = hidden_states.shape[-1]
    topk = topk_ids.shape[-1]
    E = w1.shape[0]
    sorted_token_ids, sorted_weight_buf, sorted_expert_ids, num_valid_ids, out_asm = (
        moe_sorting(topk_ids, topk_weights, E, model_dim, dtype)
    )
...
```

<!-- markdownlint-disable -->
**Blockscale FP8 MoE** **Computation**
<!-- markdownlint-restore -->

AITER provides a fused MoE fp8 blockscale implementation with an
assembly kernel. The MoE layer is essentially a layer of multiple expert
feed-forward networks (FFN) and computes logits by group gemm on
selected FFN layer after gating scores. AITER asm MoE is currently the
best performance on the AMD platform. The acceleration comes from the
operations in group gemm of different shapes, where the weights of each
expert are multiplied by hidden state tokens.

Please refer to
[op_tests/test_moe_blockscale.py](https://github.com/BruceXcluding/aiter/blob/aiter_customer/op_tests/test_moe_blockscale.py) for implementation.

```python
...
def asm_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,
    fc2_scale=None,
    a1_scale=None,
    scale_blk=(128, 128),
):

...
    scale_blk_n, scale_blk_k = scale_blk
    aiter.fmoe_fp8_blockscale_g1u1(
        out_asm,
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_weight_buf,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        a1_scale,
        fc1_scale,
        fc2_scale,
        scale_blk_n,
        scale_blk_k,
        None,
    )
    return out_asm
```

We unified the interface of the MoE Sorting & MoE Computation APIs entry point at aiter/fused_moe_bf16_asm.py:

```python

def asm_moe():

    # MoE sorting
    sorted_token_ids, sorted_weight_buf, sorted_expert_ids, num_valid_ids, out_asm = moe_sorting_ck(topk_ids, topk_weights, E,
                                                                                                    model_dim, dtype)
    ...                                                                                               
    elif block_shape is not None:
        assert dtype == torch.bfloat16, "asm_moe for block_scale only support bfloat16 hidden_states"
        assert block_shape == (
            128, 128), "asm_moe for block_scale only support (128, 128)"
        assert w1.dtype == torch.float8_e4m3fnuz, "asm_moe for block_scale only support float8_e4m3fnuz weight"
        assert w2.shape[2] * 2 == w1.shape[1], "aiter moe for block_scale only support g1u1"
        scale_blk_n, scale_blk_k = block_shape
        hidden_states = hidden_states.view(M *
                                        model_dim//scale_blk_k, scale_blk_k)
        a8 = torch.empty(
            (M, model_dim), dtype=w1.dtype, device=device)
        a8_scale = torch.empty((M, model_dim//scale_blk_k), dtype=torch.float, device=device)
        aiter.dynamic_per_token_scaled_fp8_quant(a8, hidden_states, a8_scale)
        
        # MoE Blockscale FP8
        aiter.fmoe_fp8_blockscale_g1u1(moe_buf, a8, w1, w2, sorted_ids,
                                    sorted_weights, sorted_expert_ids, num_valid_ids,
                                    topk,
                                    fc1_scale.view(E, -1),
                                    fc2_scale.view(E, -1),
                                    a8_scale.t().contiguous(),
                                    scale_blk_n,
                                    scale_blk_k,
                                    fc2_smooth_scale)
        return moe_buf
    ...

```

Plug into Framework at sglang/python/sglang/srt/layers/quantization/fp8.py

```python
from aiter import biased_grouped_topk
from aiter.fused_moe_bf16_asm import asm_moe
...

class FP8MoEMethod:
    ...
    def apply():
        ...
        # Top-K routing
        if _is_hip and get_bool_env_var("AITER_MOE") and correction_bias is not None:
            token = x.shape[0]
            biased_grouped_topk(
                router_logits,
                correction_bias,
                layer.ns_topk_weights[:token],
                layer.ns_topk_ids[:token],
                num_expert_group,
                topk_group,
                renormalize,
                layer.routed_scaling_factor,
            )
            topk_ids = layer.total_topk_ids[:token]
            topk_weights = layer.total_topk_weights[:token]
        ...
        # MoE (Moe Sorting & MoE blockscale operators)
        if _is_hip and get_bool_env_var("AITER_MOE"):
            assert (
                activation == "silu"
            )
            ...
            if self.block_quant:
                return asm_moe(
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    topk_weights,
                    topk_ids,
                    layer.w13_weight_scale_inv,
                    layer.w2_weight_scale_inv,
                    block_shape=tuple(self.quant_config.weight_block_size),
                    expert_mask=None,
                )
        ...
```

<!-- markdownlint-disable -->
## General Linear Layer
<!-- markdownlint-restore -->

AITER provides two types of FP8 blockwise GEMM implementation with a ck
fp8 blockscale GEMM and a ck pre-shuffle fp8 blockscale GEMM. We use
shuffle gemm here. For activation, every 1x128 pixel needs one scale,
and for weights, every 128x128 pixel needs one scale. The pre-shuffle is
a permutation for optimized layout.

Please refer to
[op_tests/test_gemm_a8w8_blockscale.py](https://github.com/BruceXcluding/aiter/blob/aiter_customer/op_tests/test_gemm_a8w8_blockscale.py) for implementation.

```python
...
def run_gemm_ck_wpreshuffle(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    return aiter.gemm_a8w8_blockscale_wpreshuffle_CK(x, weight, x_scale, w_scale, dtype)
...
def test_gemm(dtype, m, n, k):
    dim = (m, n, k)
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=torch.float16, device="cuda") / 10).to(
        torch.float8_e4m3fnuz
    )
    ...
    weight_shulle = shuffle_weight(weight, layout=(16, 16))
    ...
    c, avg_c = run_gemm_ck_wpreshuffle(x, weight_shulle, x_scale, w_scale, dtype)
...
APIs entry point aiter/ops/gemm_op_a8w8.py:
...
def gemm_a8w8_blockscale_wpreshuffle_CK(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype=torch.bfloat16
):
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in gemm_a8w8"
    m = XQ.shape[0]
    n = WQ.shape[0]
    k = XQ.shape[-1]
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    return gemm_a8w8_blockscale_wpreshuffle(XQ, WQ, x_scale, w_scale, Y)
```

To plug into Framework:Shuffle the weights at sglang/python/sglang/srt/layers/quantization/fp8.py

```python
from aiter.ops.shuffle import shuffle_weight
...
class FP8LinearMethod(LinearMethodBase):
...
def process_weights_after_loading(self, layer: Module) -> None:
    ...
                if get_bool_env_var("AITER_MOE"):
                    # Pre-shuffle weights
                    layer.weight.data = shuffle_weight(
                        layer.weight.contiguous(), (16, 16)
                    )
    ...
```

GEMM at sglang/python/sglang/srt/layers/quantization/fp8_utils.py

```python
from aiter import gemm_a8w8_blockscale_wpreshuffle_CK
...
def apply_w8a8_block_fp8_linear():
...
elif _is_hip and get_bool_env_var("AITER_MOE"):
        q_input, x_scale = per_token_group_quant_fp8(
            input_2d, block_size[1], column_major_scales=False
        )
        output = gemm_a8w8_blockscale_wpreshuffle_CK(q_input, weight, x_scale, weight_scale, dtype=input.dtype)
...
```

Note: This GEMM has been tuned for DeepSeek-V3/R1's shapes under aiter/configs/a8w8_blockscale_tuned_gemm.csv. For understanding method for tuning different shapes, refer to csrc/ck_gemm_a8w8_blockscale/README.md

## Attention Mechanisms

AITER provides a latent attention implementation with an assembly kernel
for decode (head dim 576/512) weights absorption. AITER provides a multi
head attention implementation with a ck kernel (head dim 192/128) and a
latent attention implementation with an assembly kernel (limited to q
extend \< 160) for deepseek prefill phase.
<!-- markdownlint-disable -->
**MLA For Decode**
<!-- markdownlint-restore -->
Please refer to
[op_tests/test_mla.py](https://github.com/BruceXcluding/aiter/blob/aiter_customer/op_tests/test_mla.py) for implementation.

Note: A new parameter, kv_last_page_lens, has been added to the function. Additionally, kv_buffer is now required to be viewed as a 4-dimension.

```python

# AITER Implementation

kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
out_asm = torch.empty((batch_size, nhead, v_head_dim), dtype=dtype).fill_(-1)
(attn_logits, attn_lse), us_asm = run_perftest(
    aiter.mla.mla_decode_fwd,
    q,
    kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
    out_asm,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    sm_scale,
)
...
```

APIs entry point aiter/aiter/mla.py:

```python

def mla_decode_fwd(
    q,
    kv_buffer,
    o,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    sm_scale=None,  # 1.0 / (qk_head_dim**0.5)
    logit_cap=0.0,
    num_kv_splits=None,  # for experts only!!!
):
```

To plug into Framework: Prepare data for the new parameter kv_last_page_lens at sglang/python/sglang/srt/layer/attention/triton_backend.py:

```python

def init_forward_metadata():
...
if forward_batch.forward_mode.is_decode_or_idle():
    ...
    kv_last_page_len = torch.ones(bs, dtype=torch.int)
    ...
```

Decode MLA at sglang/python/sglang/srt/layer/attention/triton_ops/decode_attention.py, adding implementation into decode_attention.py:

```python

from aiter.mla import mla_decode_fwd
...
def decode_attention_fwd():
...
elif _is_hip and get_bool_env_var("AITER_MOE"):
    # ROCM MLA
    mla_decode_fwd(
        q,
        k_buffer.view(-1, 1, 1, q.shape[-1]),
        o,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        sm_scale,
        logit_cap,
    )
    k_buffer = k_buffer.reshape(-1, 1, q.shape[-1])
...
```

<!-- markdownlint-disable -->


**MHA/MLA For Prefill**

**MHA:**
<!-- markdownlint-restore -->

Please refer to [op_tests/test_mla.py](https://github.com/BruceXcluding/aiter/blob/aiter_customer/op_tests/test_mla.py) for implementation.

```python

...
out_aiter, us_aiter = run_perftest(
    aiter.flash_attn_varlen_func,
    q,
    k,
    v,
    qo_indptr,
    kv_indptr,
    max_seqlen_qo,
    max_seqlen_kv,
    softmax_scale=sm_scale,
    causal=True,
)
...
```

APIs entry point aiter/aiter/ops/mha.py:

```python

def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    block_table=None,
):
```

**MLA:**

Please refer to [op_tests/test_mla.py](https://github.com/BruceXcluding/aiter/blob/aiter_customer/op_tests/test_mla.py) for implementation.

```python

kv_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_kv, dim=0)
kv_indices = torch.randint(
    0, num_page, (kv_indptr[-1].item() + 1,), dtype=torch.int
)
...
# aiter implementation
out_asm = torch.empty((total_qo, nhead, v_head_dim), dtype=dtype).fill_(-1)
(attn_logits, attn_lse), us_asm = run_perftest(
    aiter.mla.mla_prefill_fwd,
    q,
    kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
    out_asm,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_qo,
    sm_scale,
)
...

```

APIs entry point aiter/aiter/mla.py:

```python

def mla_prefill_fwd(
    q,  # [num_seqs, num_heads, head_size]
    kv_buffer,  # [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    o,  # [num_seqs, num_heads, v_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    sm_scale=None,  # 1.0 / (qk_head_dim**0.5)
    logit_cap=0.0,
    num_kv_splits=None,  # for experts only!!!
):
```

Integrate with the framework at sglang/python/sglang/srt/layer/attention/triton_backend.py. In sglang, the kv_indices are initially allocated as a prefix; however, when performing absorption, we need to convert them to an extend + prefix format.

```python

def init_forward_metadata():
...
else:
    if _is_hip and get_bool_env_var("AITER_MOE"):
        max_prefix_extend_len = torch.max(
            forward_batch.extend_seq_lens + forward_batch.extend_prefix_lens
        ).item()
        kv_indptr += qo_indptr
        if sum(forward_batch.extend_seq_lens_cpu) - sum(forward_batch.extend_prefix_lens_cpu) <= 160:
            prefix_kv_indices = kv_indices
            extend_kv_indices = forward_batch.out_cache_loc
            prefix = torch.split(
                prefix_kv_indices, forward_batch.extend_prefix_lens_cpu
            )
            extend = torch.split(
                extend_kv_indices, forward_batch.extend_seq_lens_cpu
            )
            kv_indices = torch.cat(
                [x for el in zip(prefix, extend) for x in el]
            ).to(torch.int)
...

```

Add implementation into forward extend, forward_normarl goes to mha and forward_extend goes to mla:

```python

from aiter import flash_attn_varlen_func
from aiter.mla import mla_prefill_fwd
...
def forward_extend():
...
if _is_hip and get_bool_env_var("AITER_MOE"):
    max_extend_len = self.forward_metadata.max_extend_len
    max_prefix_extend_len = self.forward_metadata.max_prefix_extend_len
    kv_indptr = self.forward_metadata.kv_indptr
    kv_indices = self.forward_metadata.kv_indices
    kv_last_page_lens = self.forward_metadata.kv_last_page_len
    qo_indptr = self.forward_metadata.qo_indptr
    K_Buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
    V_Buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
    kv_lora_rank = V_Buffer.shape[-1]
    qk_rope_head_dim = K_Buffer.shape[-1] - kv_lora_rank
    qk_nope_head_dim = k.shape[-1] - qk_rope_head_dim
    assert len(q.shape) == 3
    assert len(k.shape) == 3
    assert len(v.shape) == 3
    if layer.tp_k_head_num != 1:
        if kv_indices.shape[0] == 0:
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                qo_indptr,
                qo_indptr,
                max_extend_len,
                max_extend_len,
                softmax_scale=layer.scaling,
                causal=True,
            )
            return o
        elif layer.qk_head_dim != (kv_lora_rank + qk_rope_head_dim):
            K_Buffer = torch.index_select(K_Buffer, 0, kv_indices)
            kvc, k_pe = torch.split(
                K_Buffer, [kv_lora_rank, qk_rope_head_dim], dim=-1
            )
            kvprefix = layer.kv_b_proj(kvc.contiguous())[0]

            kvprefix = kvprefix.view(
                -1, layer.tp_k_head_num, qk_nope_head_dim + layer.v_head_dim
            )
            k_prefix, v_prefix = torch.split(
                kvprefix, [qk_nope_head_dim, layer.v_head_dim], dim=-1
            )
            k_prefix = torch.cat(
                [k_prefix, torch.broadcast_to(k_pe, (k_pe.shape[0], layer.tp_k_head_num, k_pe.shape[2]),),], dim=-1,
            )
            assert (
                forward_batch.extend_prefix_lens.shape
                == forward_batch.extend_seq_lens.shape
            )
            k_prefix = torch.split(
                k_prefix, forward_batch.extend_prefix_lens_cpu
            )
            k_extend = torch.split(k, forward_batch.extend_seq_lens_cpu)
            assert len(k_prefix) == len(forward_batch.extend_prefix_lens_cpu)
            k = torch.cat([x for el in zip(k_prefix, k_extend) for x in el])
            v_prefix = torch.split(
                v_prefix, forward_batch.extend_prefix_lens_cpu
            )
            v_extend = torch.split(v, forward_batch.extend_seq_lens_cpu)
            v = torch.cat([x for el in zip(v_prefix, v_extend) for x in el])

            o = flash_attn_varlen_func(
                q,
                k,
                v,
                qo_indptr,
                kv_indptr,
                max_extend_len,
                max_prefix_extend_len,
                softmax_scale=layer.scaling,
                causal=True,
            )
            return o
        else:
            token_num = forward_batch.extend_num_tokens

            mla_prefill_fwd(
                q.view(token_num, layer.tp_q_head_num, layer.qk_head_dim),
                K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                o.view(token_num, layer.tp_q_head_num, layer.v_head_dim),
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_lens,
                max_extend_len,
                layer.scaling,
                layer.logit_cap,
            )
            K_Buffer = K_Buffer.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
            return o
    ...
```

## Other Operators
<!-- markdownlint-disable -->
**Customer All Reduce**
<!-- markdownlint-restore -->
The Customer All-Reduce implementation in AITER is functionally
equivalent to the one used in vLLM, supporting out-of-place operations.
However, the AITER version is specifically optimized for AMD MI300X
architecture, enabling enhanced performance on that platform.

APIs entry point: aiter/ops/custom_all_reduce.py

```python
def init_custom_ar(
    out: Tensor, exp_sums: Tensor, handles, offsets, rank: int, full_nvlink: bool
) -> int: ...
...

```

Plug into Framework at sglang/python/sglang/srt/\_custome_ops.py

```python

...
else:
    ...
    if get_bool_env_var("AITER_MOE"):

        import aiter.ops.custom_all_reduce as aiter_custom_ar

        def init_custom_ar() -> int:
            ...
            return aiter_custom_ar.init_custom_ar(meta, rank_data, handles, offsets, rank, full_nvlink)

        def all_reduce_reg(fa: int, inp: torch.Tensor, out: torch.Tensor) -> None:
            ...
            aiter_custom_ar.all_reduce_reg(fa, inp, out)

        def all_reduce_unreg() -> None:
            ...
            aiter_custom_ar.all_reduce_unreg(fa, inp, reg_buffer, out)

        def dispose(fa: int) -> None:
            (aiter_custom_ar.dispose(fa))

        def meta_size() -> int:
            return aiter_custom_ar.meta_size

        def register_buffer(
            fa: int, t: torch.Tensor, handles: List[str], offsets: List[int]
        ) -> None:
            return aiter_custom_ar.register_buffer(fa, t, handles, offsets)

        def get_graph_buffer_ipc_meta(fa: int) -> Tuple[torch.Tensor, List[int]]:
            return aiter_custom_ar.get_graph_buffer_ipc_meta(fa)

        def register_graph_buffers(
            fa: int, handles: List[str], offsets: List[List[int]]
        ) -> None:
            aiter_custom_ar.register_graph_buffers(fa, handles, offsets)

        def allocate_meta_buffer(size: int) -> torch.Tensor:
            return aiter_custom_ar.allocate_meta_buffer(size)

        def get_meta_buffer_ipc_handle(inp: torch.Tensor) -> torch.Tensor:
            return aiter_custom_ar.get_meta_buffer_ipc_handle(inp)

```

## Performance Reproduction for AITER-Optimized in SGLang

This guide outlines the steps to reproduce the performance results of
the AITER-optimized solution integrated into SGLang v0.4.4, with support
for MoE, GEMM, Attention, and All-Reduce operations.

<!-- markdownlint-disable -->
**Step 1:**  Build and Run the Docker Container
<!-- markdownlint-restore -->

Use the provided Dockerfile to set up your environment:
[Dockerfile](https://github.com/BruceXcluding/sglang/blob/aiter_customer_v0.4.4/docker/Dockerfile.rocm)

```bash

docker build -f Dockerfile -t customer_sglang_v0.4.4 .

```

Then run the container with the following command:

```bash

volume=$PWD
docker run -it --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $volume:/workspace \
  --name customer_sglang customer_sglang_v0.4.4:latest

```

<!-- markdownlint-disable -->
**Step 2**: Run the AITER-Optimized Benchmark
<!-- markdownlint-restore -->

Enable AITER by setting the AITER_MOE flag and run the benchmark script:

```bash

 AITER_MOE=1 python -m sglang.bench_one_batch \
  --batch-size 64 \
  --input 512 \
  --output 32 \
  --model deepseek-ai/DeepSeek-R1/ \
  --tp 8 \
  --trust-remote-code \
  --quantization fp8
```

**Note:** *AITER uses Just-In-Time (JIT) compilation. The initial run
may fail with an error like "Child process unexpectedly failed with an
exit code." Simply rerun the script. Upon success, JIT-compiled
artifacts (models_xxx.so) will be generated in aiter/aiter/jit.*

### Results

These results highlight the effectiveness of AITER's kernel-level
optimizations and modular acceleration strategies within the SGLang
pipeline.

<!-- markdownlint-disable -->
-   **Prefill Latency:** ↓ **52%**

-   **Decode Latency:** ↓ **47%**

-   **Total Throughput:** ↑ **100%**

**MI300X Before**

Benchmark \...

Prefill. latency: 3.12678 s, throughput: 10479.79 token/s

Decode. latency: 0.04589 s, throughput: 1394.62 token/s

Decode. latency: 0.04661 s, throughput: 1373.05 token/s

Decode. latency: 0.04756 s, throughput: 1345.72 token/s

Decode. latency: 0.04905 s, throughput: 1304.76 token/s

Decode. latency: 0.05025 s, throughput: 1273.53 token/s

Decode. median latency: 0.05319 s, median throughput: 1203.31 token/s

Total. latency: 4.748 s, throughput: 7332.02 token/s

**MI300X After**

Benchmark \...

Prefill. latency: 1.50748 s, throughput: 21736.97 token/s

Decode. latency: 0.02673 s, throughput: 2394.61 token/s

Decode. latency: 0.02656 s, throughput: 2409.26 token/s

Decode. latency: 0.02643 s, throughput: 2421.57 token/s

Decode. latency: 0.02710 s, throughput: 2361.97 token/s

Decode. latency: 0.02714 s, throughput: 2358.01 token/s

Decode. median latency: 0.02837 s, median throughput: 2255.85 token/s

Total. latency: 2.379 s, throughput: 14635.98 token/s

<!-- markdownlint-restore -->

## Summary

This blog presents a step-by-step guide for integrating AITER—AMD high-performance AI operator library—into SGLang to optimize DeepSeek-R1 inference. By replacing key components such as MoE layers, attention mechanisms, GEMM kernels, and all-reduce operations with AITER-optimized implementations, developers can significantly boost throughput and reduce latency on AMD MI300X GPUs. The AITER library provides a robust set of optimized operators with
well-documented APIs, making it easy to integrate into various
frameworks. It offers clear operator APIs, tensor layout specifications,
and defined integration points, facilitating smooth adoption. AITER also
supports both JIT compilation and pre-built binaries---JIT compiles
operators during the first run and stores dynamic libraries under
aiter/jit/ for reuse in subsequent executions. The invocation process
follows a structured hierarchy: from operator initialization through
Python-bound tensor interfaces down to backend kernel implementations
(ASM, CK, HIP), each featuring architecture-specific optimizations. This
ensures seamless framework integration and delivers high runtime
performance.

## Acknowledgement

We would like to express our special thanks for the support from the AMD AITER & CK & Framework Core team members.

## Additional Resources

<!-- markdownlint-disable -->

1.  [AITER: AI Tensor Engine For
    ROCm](https://rocm.blogs.amd.com/software-tools-optimization/aiter:-ai-tensor-engine-for-rocm%E2%84%A2/README.html)

2.  [SGLang PR: \[AMD\] Add Optimization for DeepSeek-R1/V3 based on
    AITER Backend
    #4344](https://github.com/sgl-project/sglang/pull/4344) : Kindly
    note that our framework team is currently developing AiterBackend,
    this PR is a draft PR for reference.

3.  [Composable kernel
    github](https://github.com/ROCm/composable_kernel/tree/develop)

<!-- markdownlint-restore -->

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
