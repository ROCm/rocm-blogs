---
blogpost: true
blog_title: "Enabling DeepSeek-V4-Flash Training on AMD Instinct MI355X GPUs with Primus"
date: "03 Sep 2026"
author: "Lihuan Zhang, Wen Xie, Yanyuan Qin, Zhen Huang, Ruibin Zhang, Kyle Zhao, Cheng Yao, Xiaoming Peng, Xiaobo Chen, Anshu Raina, Yao Fu, Zhaodong Bing, Fuwei Yang, Zhenyu Gu"
thumbnail: 'deepseek-v4-flash-thumbnail.png'
tags: "Optimization, AI/ML, Performance, PyTorch, LLM"
category: "Software tools & optimizations"
target_audience: "AI Training Developers"
key_value_propositions: "DeepSeek-V4-Flash Training, Primus Training, AMD GPUs"
language: English
myst:
    html_meta:
        "author": "Lihuan Zhang, Wen Xie, Yanyuan Qin, Zhen Huang, Ruibin Zhang, Kyle Zhao, Cheng Yao, Xiaoming Peng, Xiaobo Chen, Anshu Raina, Yao Fu, Zhaodong Bing, Fuwei Yang, Zhenyu Gu"
        "description lang=en": "DeepSeek-V4-Flash training on AMD Instinct GPUs with Primus: model architecture introduction, performance projection, kernel optimizations, and how to reproduce."
        "keywords": "DeepSeek-V4-Flash, Primus, Training, LLM, AMD GPUs"
        "vertical": "AI, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Lihuan Zhang, Wen Xie, Yanyuan Qin, Zhen Huang, Ruibin Zhang, Kyle Zhao, Cheng Yao, Xiaoming Peng, Xiaobo Chen, Anshu Raina, Yao Fu, Zhaodong Bing, Fuwei Yang, Zhenyu Gu"
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

# Enabling DeepSeek-V4-Flash Training on AMD Instinct MI355X GPUs with Primus

DeepSeek-AI released the [DeepSeek-V4 series](https://arxiv.org/abs/2606.19348) on
April 24, 2026: a preview pair of MIT-licensed Mixture-of-Experts models, with
[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) at 284B
total parameters (13B activated) and a one-million-token context window. Flash
pushes sparse attention further than any open-weight model before it: its 43
transformer layers interleave three different attention types, each layer sits
inside a hyper-connection block rather than a plain residual, and every token is
routed through 256 experts. Each of those choices breaks an assumption baked into
stock Megatron-LM training code.

This blog walks you through enabling end-to-end DeepSeek-V4-Flash pretraining in
Primus on AMD Instinct™ MI355X GPUs. You will learn what the architecture looks
like layer by layer, which knobs Primus exposes to configure it, and — where most
of the engineering went — the kernel work that took the model from *it runs* to
*it runs fast*. All of it is BF16 training — see [the endnotes](#endnotes) for
what that leaves out. Every config, launch script, and benchmark referenced here
ships in the open-source Primus repository, so you can reproduce the run
yourself.

## The DeepSeek-V4-Flash architecture

DeepSeek-V4-Flash keeps the skeleton you already know from DeepSeek-V3 — a
Transformer stack with DeepSeekMoE feed-forward layers and a Multi-Token
Prediction head. What changed sits underneath: how attention reads the KV
cache, and how residual connections carry signal between blocks. Pretraining
also moves to the Muon optimizer for most parameters, keeping AdamW for the
embedding, the prediction head, and the RMSNorm weights.

```{figure} ./images/deepseek-v4-flash-architecture.png
:align: center
:width: 100%
:alt: DeepSeek-V4-Flash architecture showing one transformer block with its four mHC mixers, plus detail panels for the three attention types, the MoE layer, and manifold-constrained hyper-connections

Figure 1: DeepSeek-V4-Flash architecture — one transformer block, with detail panels for the three attention types, the MoE layer, and mHC.
```

Figure 1 shows a single block. The overall shape is familiar: 43 layers, hidden
size 4,096, a 129,280-token vocabulary, and a MoE layer in every block with one
shared expert alongside 256 routed experts, six of which activate per token.
Two details already depart from V3 — every sub-layer is wrapped in
manifold-constrained hyper-connections (mHC) rather than a plain residual, and
the first three MoE layers route tokens by hash instead of through the learned
router.

The attention module is where most of the change lives. All 64 query heads read
a single 512-dimensional latent that serves as both key and value, making the
layer multi-query rather than multi-head. Queries arrive through a low-rank
path of rank 1,024 that the sparse-selection indexer shares. And because 64
heads of 512 dimensions is a wide tensor to project back down to 4,096, the
output projection splits into 8 groups that each pass through a
1,024-dimensional bottleneck.

### Three attention types, interleaved

V4's headline change is that not every layer attends the same way. A per-layer
compression ratio picks one of three paths, fixed when the model is defined.

```{figure} ./images/deepseek-v4-flash-cr-schedule.png
:align: center
:width: 100%
:alt: Per-layer attention type in DeepSeek-V4-Flash, showing the compress_ratio schedule across 43 decoder layers plus the MTP layer, and the resulting KV entry counts at a one-million-token context

Figure 2: Per-layer attention type across the 43 decoder layers and the MTP layer, and the KV entries each type reads at a 1M-token context.
```

The first two layers run dense attention over a 128-token sliding window, a
local warm-up before any compression kicks in. The remaining 41 layers
alternate between Compressed Sparse Attention (CSA) and Heavily Compressed
Attention (HCA), which works out to 21 CSA layers and 20 HCA layers. The MTP
layer reuses the dense type.

The payoff shows up in the lower half of Figure 2. At a one-million-token
context, a query in an HCA layer reads roughly 7,900 KV entries and a query in
a CSA layer reads 640 — against a million for dense attention. Aggregated over
the model, that is what lets DeepSeek report V4-Flash at about 10% of
DeepSeek-V3.2's single-token inference FLOPs with 7% of the KV cache.

```{figure} ./images/deepseek-v4-flash-attention-kv.png
:align: center
:width: 100%
:alt: How each DeepSeek-V4-Flash layer type builds its KV, comparing the sliding-window, CSA, and HCA paths and the learned pooling operator shared by the two compressed types

Figure 3: How each layer type builds its KV, and the learned pooling operator that CSA and HCA share.
```

Figure 3 shows how each path gets there. CSA pools every 4 tokens into one KV
entry, then a lightweight "lightning indexer" scores every pooled entry and
keeps the best 512 for the attention itself. HCA pools far more aggressively —
128 tokens per entry — but skips selection and attends densely over everything
it produced. Both add the same 128-token sliding-window branch so a query can
still see recent tokens at full resolution, and both share one learned pooling
operator: a softmax over the group, biased by a learnable per-position term,
used to weight the sum.

The asymmetry worth remembering is that CSA's groups overlap. Each compressed
entry pools its own four tokens plus the previous four, which is why CSA needs
four KV-side projections where HCA needs two. That extra projection work
reappears later when we break down kernel time.

### Hyper-connections in place of the residual

Instead of `x + F(x)`, each sub-layer sits between a pair of mHC mixers
operating on four parallel residual streams. The first mixer collapses those
four streams into the single tensor the sub-layer consumes; the second expands
the result back out and combines it with the streams coming in. That
combination matrix is projected onto the doubly-stochastic manifold by 20
Sinkhorn-Knopp iterations, which bounds its spectral norm at 1 and keeps signal
propagation non-expansive across all 43 layers.

One ordering detail matters if you are porting this: the RMSNorm sits *after*
the collapse, not before it. Several published diagrams of V4 get this
backwards.

Taken one at a time, none of these changes is exotic. Taken together they mean
you cannot train V4 by pointing stock Megatron-LM at a new config file, which
is where Primus comes in.

## Enabling DeepSeek-V4 in Primus

Primus describes a model as a chain of YAML files, each overriding the one
below it. For V4-Flash that chain has three links:

```text
primus/configs/models/megatron/llama_base.yaml     generic decoder defaults
  └─ deepseek_v4_base.yaml                         everything the V4 family shares
       └─ deepseek_v4_flash.yaml                   Flash-specific shapes
```

`deepseek_v4_base.yaml` is where the V4 vocabulary enters Primus. These are the
knobs that have no equivalent in a V3 config:

| Field | Flash value | What it controls |
| --- | --- | --- |
| `compress_ratios` | `[0, 0, 4, 128, …, 4, 0]` | Per-layer attention type; 43 decoder entries plus one for MTP |
| `index_topk` | `512` | How many compressed entries the lightning indexer keeps |
| `index_n_heads` / `index_head_dim` | `64` / `128` | Indexer scoring shape |
| `attn_sliding_window` / `attn_sink` | `128` / `true` | The local branch every layer type carries |
| `hc_mult` / `hc_sinkhorn_iters` | `4` / `20` | mHC residual streams and Sinkhorn-Knopp iterations |
| `o_groups` / `o_lora_rank` | `8` / `1024` | Grouped low-rank output projection |
| `num_hash_layers` | `3` | How many leading MoE layers use hash routing |
| `moe_router_score_function` | `sqrtsoftplus` | V4's router scoring |
| `swiglu_limit` | `10.0` | Clamped SwiGLU, for FP8 and FP4 stability |

One field does more than configure: `model_type: deepseek_v4` is what routes
the build away from the standard GPT path and into
`primus/backends/megatron/core/models/deepseek_v4/deepseek_v4_builders.py`,
which assembles the per-layer specs from `compress_ratios`. Change nothing else
and Primus would happily build a V3-shaped model with V4 numbers in it; this
line is what makes the hybrid attention stack real.

On top of the model config sits an experiment config carrying the training
hyperparameters, the parallelism, and the kernel selection:
`examples/megatron/configs/MI355X/deepseek_v4_flash-BF16-pretrain.yaml`, with
an FP8 variant beside it. The parts specific to V4 are short:

```yaml
modules:
  pre_trainer:
    framework: megatron
    model: ${PRIMUS_MODEL:deepseek_v4_flash}.yaml
    overrides:
      tensor_model_parallel_size: ${PRIMUS_TP:1}
      pipeline_model_parallel_size: ${PRIMUS_PP:4}
      expert_model_parallel_size: ${PRIMUS_EP:8}

      # attention kernels, chosen per path
      use_v4_attention_backend: ${PRIMUS_USE_V4_ATTENTION_BACKEND:turbo}
      use_v4_csa_attention_backend: ${PRIMUS_USE_V4_CSA_ATTENTION_BACKEND:turbo}
      use_v4_fp8_indexer: ${PRIMUS_USE_V4_FP8_INDEXER:false}
      use_v4_compiled_sinkhorn: ${PRIMUS_USE_V4_COMPILED_SINKHORN:false}

      # MoE acceleration
      use_turbo_grouped_gemm: true
      use_turbo_deepep: ${PRIMUS_USE_TURBO_DEEPEP:true}
```

Three things there are worth pointing out. The attention backend is selected
separately for the dense and HCA layers (`use_v4_attention_backend`) and for
the CSA layers (`use_v4_csa_attention_backend`), because CSA's indexer and
top-k selection make it a different kernel problem — we come back to that when
we look at performance. The MoE lines shown here are one of two alternatives:
setting `USE_TURBO_MEGA_MOE=True` swaps in MegaMoE, which fuses the
communication into the grouped GEMM and turns DeepEP off automatically, since
the two patch the same layer. And every V4 knob reads through an environment
variable with a default, which is what makes it practical to sweep one
optimization at a time.

## Memory and performance projection

Before booking cluster time it helps to know where the parameter budget and the
memory budget actually go. Primus Projection derives both from the model config
without running a training step, and for a model shaped like V4-Flash the
answers are lopsided in ways that are worth seeing before you start tuning.

### Where the parameters go

```{figure} ./images/deepseek-v4-flash-param-breakdown.png
:align: center
:width: 100%
:alt: DeepSeek-V4-Flash parameter breakdown showing MoE layers holding 95.7% of the model, a zoom into the non-MoE remainder, a zoom into a single MoE layer, and every component on a log scale

Figure 4: Where the 290.80B parameters live, at three levels of zoom and on a log scale.
```

Two things stand out in Figure 4:

- **MoE holds 95.7% of the model.** All 43 attention modules together come to
  1.70% — 4.94B parameters against the MoE stack's 278.15B. Inside a single MoE
  layer the concentration is sharper still: the 256 routed experts are 99.59%
  of it, the shared expert 0.39%, and the router gate 0.02%.
- **The total is 290.80B, not the 284B on the model card.** The difference is
  the 6.61B MTP module, which the published figure leaves out.

### Where the memory goes

```{figure} ./images/deepseek-v4-flash-memory-pp4-ep8.png
:align: center
:width: 100%
:alt: Per-rank memory breakdown for DeepSeek-V4-Flash at PP=4 and EP=8, showing BF16 weights, activations, and FP32 gradient and optimizer state against the MI355X HBM3E capacity

Figure 5: Per-rank memory at PP=4, EP=8, against the memory an MI355X provides.
```

Figure 5 shows rank 0 of the PP=4, EP=8 configuration at sequence length 4,096,
assuming 11 transformer layers on the rank plus the embedding. Three takeaways:

- **It fits, with about 4% to spare.** 257.62 GiB per rank against the 268.2
  GiB an MI355X exposes from its 288 GB of HBM3E, leaving roughly 10.6 GiB of
  headroom.
- **FP32 state, not activations, is the largest bucket.** The gradient buffer,
  the optimizer's main parameter copy, and the two moments come to 142.67 GiB,
  55.4% of the total — seven times the 20.38 GiB of BF16 weights they shadow,
  and more than the 94.56 GiB of activations.
- **Activations are the only bucket you can trade against compute.** Recompute
  buys memory back by paying for a second forward pass, and how much of it a run
  needs depends on everything else that run is doing. The shipped four-node
  configuration ends up needing none — but only because the kernel work below
  frees the memory first. Where the layers land and how much to recompute are
  tuned together, and every layout measured below puts ten layers on stage 0,
  not the eleven assumed here; [that section](#pipeline-layout-and-recompute)
  measures both.

These numbers are projections derived from the config rather than measurements,
and they are an upper bound: the measured peak on the shipped four-node
configuration is 242.98 GiB — 260.9 GB in the units the sections below use —
roughly 15 GiB under the projection.

## Performance optimizations

The sections below follow the order in which we switched these on, and
[the ladder at the end](#stacking-the-optimizations) measures what each one is
worth on a four-node run.

### Kernel fusions

DeepSeek-V4 brings in a lot of new machinery — mHC on every sub-layer, a
compressor and an indexer on every compressed layer, two new routers. Written
the obvious way, each of those is a chain of small elementwise operations, and
PyTorch dispatches every one as its own kernel with a full HBM round trip. None
of them is expensive on paper. Together they dominated our first working build,
and each intermediate they materialize is memory you do not get back.

So we fused them. The table below is what ships today; each row replaces an
eager chain with a single forward kernel and, where a backward is needed, a
single backward kernel.

| Fusion | What the eager path does | Fused into | Written in |
| --- | --- | --- | --- |
| SWA / CSA / HCA attention | Separate K and V paths, a split CSA pool kernel, and a sliding-window branch joined afterwards — see [the backend comparison below](#attention-kernels-for-the-three-layer-types) | One single-latent sparse-MLA kernel per layer type, gathering the selected KV entries in-kernel and folding in the window branch and the softmax sink | Triton, Gluon, FlyDSL |
| RMSNorm | Cast to fp32, square, mean, add eps, rsqrt, scale, cast back, optional weight multiply — an 8-op chain | One kernel pair, covering every non-TE RMSNorm site in the model body | Triton |
| Interleaved partial RoPE | A 9-op chain ending in a `torch.cat` that copies the whole tensor to rejoin the rotated and untouched halves | One kernel pair | Triton |
| Sinkhorn-Knopp | 39 fp32 reductions over a 4×4 matrix — one priming column normalization plus 19 row/column pairs — each its own launch | One kernel pair that keeps the entire trajectory in registers | Triton, after a `torch.compile` version |
| Hyper-connection glue | Three slices, three fused multiply-adds, two sigmoids, a softmax and two eps adds — about 8 launches | One kernel | Triton |
| Hyper-connection collapse | A broadcast multiply that materializes a full `[…, K, D]` temporary, then a reduction over it | One kernel that contracts `K` in registers and writes only the result | Triton |
| Hyper-connection expand | An outer product, a contraction over `K`, and an add | One kernel | Triton |
| Compressor pooling | Add the positional bias, cast, softmax over the window, cast back, multiply, reduce — about 5 launches | One forward kernel that reduces in fp32 and handles both the CSA window of 8 and the HCA window of 128 | Triton |
| Indexer scoring tail | ReLU, per-head multiply, sum over heads, mask allocation, mask add, cast — about 5 ATen launches | One kernel that materializes the causal mask inline, with no `[S, P]` mask tensor | Triton |
| MoE router tail | Score function, gather, sum, clamp, divide, scale, then two scatters | One kernel | Triton |
| Grouped expert weight stack | `torch.stack` then `transpose` then `contiguous` — two full passes over the per-expert weights | One kernel, single pass | Triton |

None of these is a headline optimization on its own, which is exactly why they
are easy to leave on the table. Switching all of them on at once is the single
largest step in the whole ladder: it nearly doubles end-to-end throughput and
frees 22 GB of memory at the same time, because every intermediate that no
longer gets written is also memory that no longer gets allocated.

### Attention kernels for the three layer types

V4-Flash runs [three different kinds of attention](#three-attention-types-interleaved),
picked per layer by `compress_ratio`, and each one hands the kernel a different
problem. Primus implements all three as fused kernels in several backends and
has tuned each of them:

- **eager** — a plain PyTorch path. Slow, but it is the reference the parity
  tests compare against.
- **Triton** — the first production backend, and the portable one.
- **Gluon** — Triton's experimental Gluon dialect, gfx950 only. It exposes the
  warp-level pipeline, so the kernel can be scheduled explicitly instead of
  leaving the decision to the compiler.
- **FlyDSL** — the fastest of the four. FlyDSL gives fine-grained control over
  instruction scheduling and software pipelining on MI355X, which is exactly
  what the compressed layer types need: their inner loop is a gather over a
  sparse set of KV entries, and hiding that latency behind MFMA issue is a
  scheduling problem more than a math problem. The FlyDSL DeepSeek-V4 attention
  kernels live in Primus-Turbo.

The tables below are single-GPU MI355X measurements at sequence length 4,096,
micro-batch 1, BF16, attention sink on, 128-token sliding window. Each cell is
median latency in milliseconds and the achieved TFLOP/s.<sup>[1]</sup>

#### Forward

| Model | Layer type | Triton | Gluon | FlyDSL |
| --- | --- | ---: | ---: | ---: |
| V4-Flash | SWA (cr = 0) | 0.30 \| 230.0 | 0.28 \| 248.3 | **0.20 \| 335.5** |
| V4-Flash | CSA (cr = 4) | 0.87 \| 397.1 | 0.66 \| 523.6 | **0.53 \| 651.7** |
| V4-Flash | HCA (cr = 128) | 0.38 \| 223.9 | 0.33 \| 263.2 | **0.22 \| 384.2** |
| V4-Pro | SWA (cr = 0) | 0.58 \| 236.2 | 0.51 \| 269.0 | **0.38 \| 357.9** |
| V4-Pro | CSA (cr = 4) | 2.78 \| 444.3 | 1.92 \| 645.1 | **1.41 \| 878.1** |
| V4-Pro | HCA (cr = 128) | 0.72 \| 238.6 | 0.61 \| 280.9 | **0.43 \| 395.5** |

#### Backward

| Model | Layer type | Triton | Gluon | FlyDSL |
| --- | --- | ---: | ---: | ---: |
| V4-Flash | SWA (cr = 0) | 1.16 \| 148.4 | 1.13 \| 152.0 | **0.67 \| 257.8** |
| V4-Flash | CSA (cr = 4) | 5.93 \| 144.9 | 3.99 \| 215.1 | **2.55 \| 336.9** |
| V4-Flash | HCA (cr = 128) | 1.67 \| 128.8 | 1.54 \| 139.2 | **0.78 \| 274.9** |
| V4-Pro | SWA (cr = 0) | 1.81 \| 190.1 | 1.70 \| 202.2 | **1.29 \| 267.2** |
| V4-Pro | CSA (cr = 4) | 10.74 \| 287.8 | 8.52 \| 362.9 | **6.32 \| 489.3** |
| V4-Pro | HCA (cr = 128) | 2.47 \| 174.2 | 2.27 \| 189.4 | **1.49 \| 288.2** |

To reproduce any column, set the two selectors in the experiment config. The
dense and HCA layers read the first field, the CSA layers the second:

| Config field | Triton | Gluon | FlyDSL |
| --- | --- | --- | --- |
| `use_v4_attention_backend` | `triton_v2` | `gluon_v3` | `turbo` |
| `use_v4_csa_attention_backend` | `triton_v2` | `gluon_v3` | `turbo` |

Getting the FlyDSL column to those numbers took work in both directions.
Sparse-MLA is one of the harder attention kernels to make fast: the KV cache is
a single 512-dimensional latent serving as both key and value, and each query
reads only a sparse top-k subset of it, so on top of the usual attention math
the kernel pays a gather/scatter tax — two passes over an intermediate tensor
that dense flash attention never touches.

**The forward pass is latency-bound rather than throughput-bound.** With `exp2`
and MFMA issue roughly balanced there is no occupancy left to buy, so every gain
came from shortening or overlapping the serial `QK → softmax → PV` chain:

- Batching two adjacent tiles into `K=32` doubles MFMA depth and halves the
  read-after-write chain.
- Moving to one work-group per token at `BLOCK_H=128`. Under a shared latent,
  two work-groups per token each store their own copy of it — pure redundancy.
  Storing it once cuts roughly a quarter of the work.
- Exploiting softmax's shift invariance: take the first key pair's maximum as a
  fixed bound, and the rescale factor becomes a constant 1 that the compiler
  folds away. That buys no-max speed at pure-accumulation precision, worth about
  13% on its own. It is now the default path.

**The backward pass splits into three kernels**, each with a different bound and
a different fix:

- **dQ** takes the largest share and is pinned to single-wave occupancy by
  register pressure. Its bottleneck is HBM latency on the KV gather, and the
  only way to hide it is to keep the per-tile `QK → softmax → PV` interleaving.
  It is the healthiest of the three, running 1.4–2.4× faster than the Triton
  version.
- **interm** is a head-dimension contraction GEMM. Replacing its LDS staging
  with a hand-rolled 16×16 in-register transpose through `ds_bpermute`, then
  moving to `K=32` MFMA, halves the instruction count.
- **delta** — the `rowsum(O·dO)` reduction — was a standalone, fully serial
  micro-kernel. Inlining it into dQ removes an entire launch. Batching kv blocks
  then let dQ and interm fuse as well, which also drops the HBM round trip for
  the intermediate tensor.

All six shapes are numerically correct in both directions at BF16.

### Expert parallelism: DeepEP and the grouped GEMM

With 256 experts and six of them active per token, the MoE layer is where both
the FLOPs and the communication live. Two independent optimizations sit on that
path, and they are worth separating because they are almost always enabled
together and then reported as one.

**DeepEP** replaces the token dispatch and combine. Expert parallelism has to
send every token to whichever rank owns its experts and bring the results back;
the stock path does that as a pair of all-to-all collectives with the
permutation and its inverse done in PyTorch around them. DeepEP does the
permutation, the transfer and the reverse in dedicated kernels, producing the
token layout the GEMM wants rather than assembling it afterwards.

**The Turbo grouped GEMM** replaces the per-expert loop. At EP=8 every rank
holds 32 of the 256 experts, and each of them multiplies a different number of
tokens by its own weights. Issued as 32 separate GEMMs most are far too small to
fill the GPU, and the launch overhead alone is comparable to the math. The
grouped GEMM issues all of them as one kernel over a ragged batch.

Both live in the experiment config:

```yaml
enable_primus_turbo: true
use_turbo_deepep: true
use_turbo_grouped_gemm: true
```

Switched on one after the other rather than in one step, the grouped GEMM is
worth about six times what DeepEP is worth on this model: DeepEP adds 1.3%,
and the grouped GEMM on top of it another 7.9%
([rungs 4 and 5 of the ladder below](#stacking-the-optimizations)). Enabling
them together, as most configurations do, would put that gain on the wrong
feature.

### MegaMoE: fusing communication into the grouped GEMM

Both optimizations above make one half of the expert path faster, but they leave
it in two halves: the communication still runs next to the GEMM rather than
inside it. The usual way to hide it is to put the transfer on its own stream and
overlap it with the math. That is awkward to orchestrate, and the two streams
then compete for the same compute units and memory bandwidth — the overlap gives
back part of what it saves.

Primus overlaps inside the kernel instead. Data movement and math interleave at
the instruction level rather than racing as separate streams, which is what
FlyDSL's fine-grained control over the pipeline makes possible. This ships today
as single-node fusion on MI355X; the same approach extends to much larger EP
degrees on the next generation of rack-scale systems.

```{figure} ./images/deepseek-v4-flash-moe-fusion.png
:align: center
:width: 100%
:alt: Fusing the MoE all-to-all into the grouped GEMM, showing the five-kernel dispatch-GEMM-act-GEMM-combine chain collapsing into two fused FlyDSL kernels

Figure 6: MegaMoE fuses the expert-parallel all-to-all into the grouped GEMM, turning five kernels into two.
```

As Figure 6 shows, MegaMoE — the FlyDSL layer that replaces the native
`MoELayer` — collapses that chain into two kernels: `dispatch_grouped_gemm`
fuses the token dispatch all-to-all into the first grouped GEMM, and
`grouped_gemm_combine` fuses the second grouped GEMM into the combine and the
weighted reduce. With a fused router in front and SwiGLU in between, the whole
expert path becomes `dispatch_grouped_gemm → SwiGLU → grouped_gemm_combine`.

**Two stages, so the DDP collectives still overlap.** Primus-Turbo does expose
all of this as a single fused op. Primus deliberately drives it as two stages
instead, each owning one weight and each wrapped in a tiny weight module that
computes nothing:

```text
MegaMoEExperts
├── fc1_weight : MegaMoEWeightModule   # w1 [g, 2I, H]  gate + up
└── fc2_weight : MegaMoEWeightModule   # w2 [g, H, I]   down

FORWARD (in order)                   BACKWARD (in order)
─────────────────────────────        ─────────────────────────────
w1 = fc1_weight()                    stage2.backward -> dW2
  hook: all-gather(w1), wait           hook: reduce-scatter(dW2) ─┐
stage1: dispatch + GEMM1     ─┐                                   │ overlap
w2 = fc2_weight()             │ ovl  stage1.backward -> dW1 ───────┘
  hook: all-gather(w2) ───────┘        hook: reduce-scatter(dW1)
stage2: SwiGLU + GEMM2 + combine         (overlaps the next layer)
```

Those modules exist to be hook sites. The distributed optimizer overlaps
two collectives at module and parameter granularity, and neither can overlap
anything if the expert path is one opaque call:

- `overlap_param_gather` rides the forward pre-hook, which fires per module. A
  single call site taking both weights means both all-gathers have to land before
  any compute starts. Split, `w2`'s gather is issued at `fc2_weight` and hides
  under stage 1.
- `overlap_grad_reduce` rides the grad hook, which fires when a parameter's
  `.grad` appears. One fused autograd node emits `dW1` and `dW2` together at the
  end of the layer backward. Split, `dW2` lands early and its reduce-scatter
  hides under stage 1's backward.

The split is purely at the Python and autograd level — the kernels themselves are
unchanged.

**Configuration.** Two flags turn it on, and MegaMoE is EP-only:

```yaml
enable_primus_turbo: true
use_turbo_mega_moe: true      # EP-only, TP=1, BF16
tensor_model_parallel_size: 1
add_bias_linear: false
```

The replacement is applied only when `enable_primus_turbo` and
`use_turbo_mega_moe` are both set, `tensor_model_parallel_size == 1`,
`params_dtype == bf16`, and an EP process group exists. Anything else asserts.
Sequence-level and global aux loss, z-loss, sinkhorn and input jitter are
unsupported — only the standard `aux_loss` — and aux-loss-free expert bias raises
`NotImplementedError`.

**What it buys.** The expert-parallel intra-node all-to-all is fused into the
FlyDSL grouped-GEMM kernel, so the ideal cost becomes `max(comm, gemm)` rather
than their sum. In practice the fused kernel holds at least 85% of that
perfect-overlap roofline, 90% or better in most cases, with only 0.3–0.5 ms of
overhead left over.

Figure 7 measures what that is worth. These are times for the MoE module on its
own.<sup>[1]</sup>

```{figure} ./images/deepseek-v4-flash-moe-speedup.png
:align: center
:width: 100%
:alt: MegaMoE against the unfused MoE path, showing 1.63x on the forward pass, 1.33x on the backward pass, and 1.43x on the two combined

Figure 7: MegaMoE against the unfused MoE path, measured on the MoE module alone.
```

### Pipeline layout and recompute

The last two knobs are not kernels. They decide how the 43 transformer layers,
the embedding, the MTP module and the loss are spread across the four pipeline
stages, and how much activation memory is traded back for recompute.

The default split — 10 layers on stage 0, which also carries the embedding, and
11 on each of the others — looks fair and is not. The last stage also carries
the MTP module and the loss, while 1F1B leaves stage 0 holding four microbatches
in flight where the last stage holds one. Moving two layers off the last stage
onto the middle two, `Et*10|t*12|t*12|t*9mL`, evens out the time per stage and
shortens the pipeline bubble; stage 0 keeps its 10 layers either way, because it
is the one under activation pressure. Recompute is the other half — the
conservative starting point checkpoints the first three layers of every stage —
and the optimizations above free enough memory to stop paying that tax.

Measured one at a time against the same reference:<sup>[2]</sup>

| Change | Layout | Recompute | TFLOP/s | Gain | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: |
| reference | `Et*10\|t*11\|t*11\|t*11mL` | 3 | 1167.2 | — | 217.2 GB |
| layout only | `Et*10\|t*12\|t*12\|t*9mL` | 3 | 1273.4 | +9.1% | 225.9 GB |
| recompute only | `Et*10\|t*11\|t*11\|t*11mL` | 0 | 1255.7 | +7.6% | 260.9 GB |
| both | `Et*10\|t*12\|t*12\|t*9mL` | 0 | 1378.8 | +18.1% | 260.9 GB |

They contribute almost equally, and doing both is worth 1.5 points more than the
sum of doing each alone: dropping recompute frees time that an unbalanced
pipeline would partly give back as bubble, and rebalancing the pipeline has
little to fill unless recompute stops taking the time. The memory bill is
dominated by recompute: dropping it costs 35 to 44 GB depending on the layout,
against the 8.7 GB the rebalance adds on its own. Once recompute is off, both
layouts peak at the same 260.9 GB of the 288 GB an MI355X provides, since
stage 0 holds 10 layers and the most microbatches in flight either way. That is
also why recompute cannot be dropped first: it only fits once the fusions and
the MoE work have given the memory back.

**Neither knob is a one-time decision.** Those four rows isolate what each one is
worth at a single point in the project; they are not the method that produced the
shipped values. Layout and recompute were retuned continuously throughout V4
development, because every kernel that landed moved the target: a fusion that
frees 8 GB changes which layout balances best, and a faster attention kernel
changes which stage sits on the critical path. Recompute is not a switch either.
Primus exposes the granularity, the per-stage layer count, an explicit list of
global layer ids, and a per-module selection, so how much to recompute is a
search over a space rather than a boolean. Zero is where that search happens to
land for four nodes with everything else on; eight nodes, or a different set of
kernels, land elsewhere.

Running that search by hand does not scale past a handful of configurations, and
it has to be redone every time the kernels change. We are building an auto-tuner
that chooses the pipeline layout and a fine-grained recompute plan together, and
will open-source it in Primus as it matures.

### Stacking the optimizations

Every section above reports what one optimization is worth in isolation. The
number that decides whether a run is practical is what they are worth together,
and that is not the same thing — each one changes the balance the next one sees.

So we measured the whole ladder end to end on four nodes: start from a build
with every optimization switched off, turn on exactly one thing per rung, keep
everything already on, and hold the shapes fixed at global batch 256,
micro-batch 1, sequence length 4,096, all in BF16, with router load balancing
forced to uniform so the expert GEMM shapes do not drift between rungs. Ten
iterations per rung, averaged over iterations 4 to 10.<sup>[2]</sup>

```{figure} ./images/deepseek-v4-flash-optimization-ladder.png
:align: center
:width: 100%
:alt: Throughput of DeepSeek-V4-Flash pretraining as seven optimizations are switched on one at a time, rising from 439.5 to 1378.8 TFLOP/s per GPU

Figure 8: Throughput as each optimization is added to the ones above it, on 4 nodes × 8 MI355X.
```

Figure 8 plots that climb, and the table below gives the exact number each rung
lands on:

| Stage | Optimization | What it changes | TFLOP/s/GPU | This step | Cumulative |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | Baseline | Every optimization off: unfused elementwise chains, first-generation Triton attention, the native MoE layer, an even pipeline split with three recomputed layers per stage | 439.5 | — | — |
| 1 | Kernel fusions | The fusions listed above, plus Megatron's permutation, cross-entropy and gradient-accumulation fusions | 875.4 | +99.2% | +99.2% |
| 2 | Gluon attention | Sparse-MLA moves to the Gluon dialect (`gluon_v3`), scheduled explicitly for gfx950 | 917.0 | +4.8% | +108.6% |
| 3 | FlyDSL attention | Sparse-MLA moves again, to the FlyDSL kernels in Primus-Turbo | 954.3 | +4.1% | +117.1% |
| 4 | DeepEP | Token dispatch and combine become dedicated kernels instead of PyTorch permutation around two all-to-all collectives | 966.5 | +1.3% | +119.9% |
| 5 | Turbo grouped GEMM | The 32 local expert GEMMs issue as one ragged-batch kernel | 1042.6 | +7.9% | +137.2% |
| 6 | MegaMoE | Replaces both of the above: the all-to-all is fused into the grouped GEMM rather than sitting next to it | 1167.2 | +12.0% | +165.6% |
| 7 | Pipeline layout and recompute | Layers rebalanced to 10/12/12/9, recompute dropped to zero | 1378.8 | +18.1% | +213.7% |

## Reproduce: training DeepSeek-V4-Flash

Everything above ships in the open-source
[Primus](https://github.com/AMD-AGI/Primus/tree/8e24522d3ccf9be38411385a38bb881261378eb9)
repository, and the four-node configuration in this blog is the default — you do
not have to reassemble the optimizations by hand.

The runs here pin Primus at commit
[`8e24522`](https://github.com/AMD-AGI/Primus/commit/8e24522d3ccf9be38411385a38bb881261378eb9),
which is where this launcher landed on `main`. Build the container from the
Dockerfile at that same commit —
[`.github/workflows/docker/Dockerfile`](https://github.com/AMD-AGI/Primus/blob/8e24522d3ccf9be38411385a38bb881261378eb9/.github/workflows/docker/Dockerfile),
which puts PyTorch, Megatron-LM, Primus-Turbo and the FlyDSL kernels on a ROCm
base — and point `DOCKER_IMAGE` at it; the launcher requires that variable.

One thing the image does not settle is the code. Any Primus image — including
one built from that Dockerfile — ships its own snapshot of the repository under
`/workspace/Primus`, and that snapshot is not necessarily this commit. Check
`8e24522` out on the host and mount it over that path, so the image supplies
the environment and your checkout supplies the code.

Everything else is one launcher:
[`examples/deepseek-v4/run_deepseek_v4_flash.sh`](https://github.com/AMD-AGI/Primus/blob/8e24522d3ccf9be38411385a38bb881261378eb9/examples/deepseek-v4/run_deepseek_v4_flash.sh).
Run it with no flags for rung 7. Its header documents the rest: one switch per
optimization family, so any rung of the ladder is a single variable away, and a
dry-run mode that resolves a combination and prints what it means before you
spend an allocation on it.

A healthy four-node run settles at roughly 8.5 s per iteration and 1,370–1,385
TFLOP/s per GPU, with peak memory around 261 GB of the 288 GB on each MI355X.<sup>[2]</sup>
The launcher is tuned for four nodes; another node count needs its own
`PRIMUS_PP` and `PRIMUS_PP_LAYOUT`.

## Summary

In this blog you explored what it takes to train DeepSeek-V4-Flash end to end in
Primus on AMD Instinct MI355X GPUs. You read the architecture layer by layer —
three interleaved attention types, manifold-constrained hyper-connections in
place of the plain residual, and a 256-expert MoE in every block — and saw why
none of it drops into stock Megatron-LM unchanged. You saw which knobs Primus
exposes to describe that shape in YAML, what Primus Projection says about the
parameter and memory budget before you book a single node, and then — where most
of the engineering went — the kernel work that took the model from *it runs* to
*it runs fast*.

No single change got it there. Fusing the small operations V4 introduces — mHC on
every sub-layer, a compressor and an indexer on every compressed layer, two new
routers — was the largest single step, nearly doubling throughput and freeing
22 GB at once. Moving the three attention types from Triton to Gluon and then to
the FlyDSL sparse-MLA kernels added another 9% to end-to-end throughput and
delivered up to a 2.3× speedup on the CSA backward pass alone, through
scheduling and pipelining rather than new math. On the expert path, DeepEP and
the Turbo grouped GEMM each accelerate one half of it, until MegaMoE replaces
both by fusing the expert-parallel all-to-all into the GEMM itself. And the last
18% was not a kernel at all: rebalancing the pipeline to 10/12/12/9 layers and
switching recompute off, which only fits because the kernel work freed the
memory first.

Together they take a four-node run from 439.5 to 1,378.8 TFLOP/s per GPU — 3.1× —
with the model holding at 261 GB of the 288 GB each GPU provides. All of it is in
the [Primus repository](https://github.com/AMD-AGI/Primus) and on by default:
attach a four-node allocation and run the launcher.

Three threads continue from here, and we will cover them in future posts as they
land. FP8 is the nearest — the experiment config already sits beside the BF16 one
and the numerical pieces it leans on are in place, so what is left is coverage
and stability rather than enablement. The pipeline-layout and recompute search we
ran by hand for this blog is becoming an auto-tuner that plans both together, and
we will open-source it in Primus as it matures. And MegaMoE's in-kernel overlap,
which today supports single-node fusion, is the piece that extends to much larger
expert-parallel degrees on the next generation of rack-scale systems. Each of
these will land in Primus before it appears in a blog, so the repository is the
place to watch.

## Acknowledgments

We would like to express our sincere gratitude to the following teams and individuals for their invaluable contributions and collaboration, their expertise and support have been instrumental in advancing the progress of this project: Felix Li from the FlyDSL Team, and Wen Chen and Ye Wang from the TE Team.

## Additional Resources

1. [AMD Instinct™ MI355X GPUs](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html): Product page for the accelerators every measurement in this blog runs on.

2. [DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348): The architecture this enablement follows, including the compressed-attention and hyper-connection definitions.

3. [DeepSeek-V4-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash): Published weights and reference configuration for the model.

4. [Primus](https://github.com/AMD-AGI/Primus): The training framework holding the V4 model definition, the experiment configs, and the launcher used here.

5. [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo): AMD's operator library, where the FlyDSL sparse-MLA attention and MegaMoE kernels live.

6. [Megatron-LM](https://github.com/NVIDIA/Megatron-LM): The backend Primus builds on, and the stock code path the V4 builders replace.

7. [DeepEP](https://github.com/deepseek-ai/DeepEP): Expert-parallel dispatch and combine library, measured at rung 4 of the ladder.

8. [Triton](https://github.com/triton-lang/triton): Compiler for the portable kernel backend, and home of the experimental Gluon dialect used for the gfx950 attention kernels.

9. [Primus Projection: Estimate Memory and Performance Before You Train](https://rocm.blogs.amd.com/software-tools-optimization/primus-projection/README.html): The tool behind the parameter and memory projections in this blog.

10. [MoE Training Best Practices on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html): Broader MoE training guidance that complements the V4-specific work here.

11. [Porting High-Performance HIP Kernels to FlyDSL](https://rocm.blogs.amd.com/software-tools-optimization/porting-hip-flydsl/README.html): Background on the FlyDSL programming model behind the fastest attention backend.

## Endnotes

[1] Test Environment

Single-GPU kernel latency — the attention backend tables and the MegaMoE
MoE-module times — was measured on one AMD Instinct MI355X GPU of an 8-GPU node
with BF16 precision, sequence length 4,096 and micro-batch 1. Server
manufacturers may vary configurations, which can yield different results.
Performance may also vary based on the use of the latest drivers and
optimizations.

AMD system configuration:

- Dual AMD EPYC 9575F 64-core processor
- 8× AMD Instinct MI355X GPUs, 288 GB HBM3E per GPU
- 1 NUMA node per socket
- System model: Supermicro AS-4126GS-NMR-LCC, system BIOS 1.4a
- Host OS: Ubuntu 22.04.5 LTS with Linux kernel 6.8.0-107-generic
- Host GPU driver: ROCm 7.0.1 + amdgpu 6.14.14
- VBIOS version: 113-M355-01-1K1-010C
- PyTorch 2.12.0
- AMD ROCm 7.14 software in the container
- Primus-Turbo 0.3.2, FlyDSL 0.2.4, Triton 3.7.0, Transformer Engine 2.14.0

[2] Test Environment

End-to-end pretraining throughput (TFLOP/s per GPU) — the optimization ladder,
the pipeline layout and recompute comparison, and the four-node figures in the
reproduce section — was measured on 4 MI355X nodes (32 GPUs total) with BF16
precision, TP=1, PP=4, EP=8, global batch 256, micro-batch 1 and sequence
length 4,096, averaged over iterations 4 to 10 of a 10-iteration run. Server
manufacturers may vary configurations, which can yield different results.
Performance may also vary based on the use of the latest drivers and
optimizations.

AMD system configuration:

- Dual AMD EPYC 9575F 64-core processor per node
- 32× AMD Instinct MI355X GPUs across 4 nodes, 288 GB HBM3E per GPU
- 1 NUMA node per socket
- System model: Supermicro AS-4126GS-NMR-LCC, system BIOS 1.4a
- Host OS: Ubuntu 22.04.5 LTS with Linux kernel 6.8.0-107-generic
- Host GPU driver: ROCm 7.0.1 + amdgpu 6.14.14
- VBIOS version: 113-M355-01-1K1-010C
- PyTorch 2.12.0
- AMD ROCm 7.14 software in the container
- Primus-Turbo 0.3.2, FlyDSL 0.2.4, Triton 3.7.0, Transformer Engine 2.14.0

[3] Scope of these measurements

Everything measured here is one configuration, and it is worth being explicit
about where its edges are. The optimizer is AdamW in BF16, not the Muon that
DeepSeek used for V4 pretraining — Primus wires Muon in behind `OPTIMIZER=muon`,
but that is not what these numbers measure. The indexer distillation loss that
trains CSA's selector is off, which also leaves the indexer parameters frozen:
the right setting for loading an already-trained indexer, or for measuring what
the kernels cost, and the wrong one for pretraining from scratch, where it has
to be on. FP8 is not measured here either, though the pieces that path leans on
already ship — an `E4M3` path for the indexer QK, and the clamped SwiGLU that
gives FP8 and FP4 their numerical headroom.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
Illustrations may have been created using generative AI and reviewed by AMD.
AMD, the AMD Arrow logo, AMD Instinct, AMD ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. PyTorch is a registered trademark of Meta Platforms, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
