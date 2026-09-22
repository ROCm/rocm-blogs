---
blogpost: true
blog_title: "Zebra-HyLo: Upcycling Transformers into Long-Context Hybrid LLMs on AMD Instinct™ GPUs"
date: "22 Sep 2026"
author: "Parsa Ashrafi Fashi, Utkarsh Saxena, Mehdi Rezagholizadeh, Vansh Bhatia, Aref Jafari, Akash Haridas, Mingyu Yang, Guihong Li, Vikram Appia, Emad Barsoum"
thumbnail: 'hylo-thumbnail.png'
tags: "LLM, PyTorch, AI/ML, Fine-Tuning"
category: "Applications & models"
target_audience: "AI researchers, AI developers, AI engineers working on efficient long-context LLMs"
key_value_propositions: "Zebra-HyLo converts pretrained Transformers into hybrid MLA plus linear-block models that extend usable context up to 32x and cut KV-cache memory by more than 90%, using post-training alone on 8 AMD Instinct MI300X GPUs"
language: English
myst:
    html_meta:
        "author": "Parsa Ashrafi Fashi, Utkarsh Saxena, Mehdi Rezagholizadeh, Vansh Bhatia, Aref Jafari, Akash Haridas, Mingyu Yang, Guihong Li, Vikram Appia, Emad Barsoum"
        "description lang=en": "Upcycle pretrained Transformers into long-context hybrid MLA + linear models on AMD Instinct MI300X GPUs, with 14 open checkpoints and training code."
        "keywords": "rocm, hybrid models, MLA, mamba-2, gated deltanet, long context, knowledge distillation, KV cache, Zebra-HyLo, HyLo, Zebra-Llama, MI300X"
        "property=og:locale": "en_US"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Parsa Ashrafi Fashi, Utkarsh Saxena, Mehdi Rezagholizadeh, Vansh Bhatia, Aref Jafari, Akash Haridas, Mingyu Yang, Guihong Li, Vikram Appia, Emad Barsoum"
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

# Zebra-HyLo: Upcycling Transformers into Long-Context Hybrid LLMs on AMD Instinct™ GPUs

Hybrid language models that interleave attention with linear sequence-modeling blocks have become
the default answer to the cost of long context. Jamba, Samba, Qwen3-Next and Kimi-Linear all take
this shape, and they all share one property that makes them expensive to adopt: they are pretrained
from scratch. Every one of them pays the full cost of building a foundation model again, which means
the enormous investment already sunk into existing Transformer checkpoints is thrown away.

Upcycling is the alternative. Instead of pretraining a hybrid, you take a pretrained Transformer,
replace its attention layers with cheaper block types, and spend a small amount of post-training
teaching the new blocks to behave like the old ones. Prior work in this direction — MambaInLlama [1],
Llamba [4], MOHAWK, and our own [Zebra-Llama](https://arxiv.org/abs/2505.17272) [3] — showed this works, but
measured success almost entirely on short-context benchmarks like perplexity and commonsense
reasoning accuracy. That leaves the obvious question unanswered. Hybrid architectures are motivated
by long sequences, so does an upcycled hybrid actually inherit long-context ability from its
Transformer parent?

The short answer we found is no, not unless you train for it explicitly. A model upcycled at 8K
context scores 55.1 on RULER at 8K and 0.8 at 64K — it has essentially no usable long-context
behaviour, despite an architecture built for exactly that.

**Zebra-HyLo** (hybrid long-context) is our recipe for fixing this. It treats long-context
preservation as a first-class training objective rather than something the architecture provides for
free, and it delivers models that extend usable context by up to 32x, cut KV-cache memory by more
than 90%, and serve up to 2M tokens in vLLM on eight AMD Instinct™ MI300X GPUs — all through
post-training, on roughly 10B tokens, with no pretraining from scratch. Figure 1 below previews the
headline result across context lengths.

In this blog you will learn how to turn a pretrained Transformer into a long-context hybrid model on
AMD Instinct™ MI300X GPUs. You will see how to initialize MLA and linear blocks from existing
weights, how to stage distillation from 2K out to 64K context, how to make 64K teacher-guided
distillation fit in GPU memory, and how to serve the resulting model in vLLM. Along the way you get
the 14 released checkpoints, the exact training configs behind every one of them, and the evaluation
numbers to compare against. Ready to upcycle your own checkpoint? Clone the repository and follow
along — the whole recipe runs on a single eight-GPU node.

```{figure} ./images/figure1-short-vs-long.png
:align: center
:alt: Short-context math performance and average RULER accuracy across context lengths
Figure 1: Short-context math performance and average RULER accuracy across 8K, 16K, 32K and 64K
context lengths. Zebra-HyLo models stay competitive on short context while clearly outperforming
upcycling baselines on long context, within a limited upcycling data budget. Source: Figure 1 from
the [HyLo paper](https://arxiv.org/abs/2604.24715).
```

## Takeaways

- **Long-context-aware upcycling.** We extend the Zebra-Llama recipe so that long-context ability
  survives conversion, instead of optimizing short-context accuracy alone.
- **Staged context expansion.** Prior upcycling work trains to roughly 24K context. We stage
  training from 2K through 8K to 64K and show how the training sequence length, not the
  architecture, determines long-context generalization.
- **Teacher-guided long-context distillation.** Chunk-wise KL supervision from an 8B teacher at 64K
  context, which required a stack of memory optimizations to be feasible at all.
- **Two backbones, two linear blocks.** Llama-3.2-1B/3B and Qwen3-1.7B, each with Gated DeltaNet
  (GDN) or Mamba-2 (M2), showing the recipe is not tied to one architecture family.
- **Deployment-oriented.** A vLLM integration that interleaves Mamba/GDN state with MLA KV cache,
  serving contexts up to 2M tokens where the Llama-3.2-3B baseline runs out of memory past 64K.
- **Everything is open.** Training code, the exact configs behind all 14 released checkpoints, and
  the models themselves.

## The Released Models

We release 14 checkpoints, all of them in the
[HyLo collection on Hugging Face](https://huggingface.co/collections/amd/hylo-long-context-aware-upcycling-for-hybrid-llms-6aa40540e1398545571f476a).
Names encode the layer layout: `4MLA12GDN` is 4 MLA layers plus 12 Gated DeltaNet layers. "KV cache
reduction" is how much of the backbone's KV-cache footprint the upcycled model gives back at the
same sequence length — every checkpoint frees at least 92% of it.

| Model | Backbone | Layer layout | Linear block | Trained context | KV cache reduction |
| --- | --- | --- | --- | --- | --- |
| [`HyLo-Llama-4MLA12GDN-8K-SFT`](https://huggingface.co/amd/HyLo-Llama-4MLA12GDN-8K-SFT) | Llama-3.2-1B-Instruct | 4 MLA + 12 GDN | Gated DeltaNet | 8K | **96.1%** |
| [`HyLo-Llama-4MLA12GDN-64K-SFT`](https://huggingface.co/amd/HyLo-Llama-4MLA12GDN-64K-SFT) | Llama-3.2-1B-Instruct | 4 MLA + 12 GDN | Gated DeltaNet | 64K | **96.1%** |
| [`HyLo-Llama-8MLA8GDN-8K-SFT`](https://huggingface.co/amd/HyLo-Llama-8MLA8GDN-8K-SFT) | Llama-3.2-1B-Instruct | 8 MLA + 8 GDN | Gated DeltaNet | 8K | **92.2%** |
| [`HyLo-Llama-8MLA8GDN-64K-SFT`](https://huggingface.co/amd/HyLo-Llama-8MLA8GDN-64K-SFT) | Llama-3.2-1B-Instruct | 8 MLA + 8 GDN | Gated DeltaNet | 64K | **92.2%** |
| [`HyLo-Llama-6MLA22GDN-8K-SFT`](https://huggingface.co/amd/HyLo-Llama-6MLA22GDN-8K-SFT) | Llama-3.2-3B-Instruct | 6 MLA + 22 GDN | Gated DeltaNet | 8K | **98.0%** |
| [`HyLo-Llama-6MLA22GDN-64K-SFT`](https://huggingface.co/amd/HyLo-Llama-6MLA22GDN-64K-SFT) | Llama-3.2-3B-Instruct | 6 MLA + 22 GDN | Gated DeltaNet | 64K | **98.0%** |
| [`HyLo-Llama-14MLA14GDN-8K-SFT`](https://huggingface.co/amd/HyLo-Llama-14MLA14GDN-8K-SFT) | Llama-3.2-3B-Instruct | 14 MLA + 14 GDN | Gated DeltaNet | 8K | **95.3%** |
| [`HyLo-Llama-14MLA14GDN-64K-SFT`](https://huggingface.co/amd/HyLo-Llama-14MLA14GDN-64K-SFT) | Llama-3.2-3B-Instruct | 14 MLA + 14 GDN | Gated DeltaNet | 64K | **95.3%** |
| [`HyLo-Qwen-7MLA21GDN-8K-SFT`](https://huggingface.co/amd/HyLo-Qwen-7MLA21GDN-8K-SFT) | Qwen3-1.7B | 7 MLA + 21 GDN | Gated DeltaNet | 8K | **96.1%** |
| [`HyLo-Qwen-7MLA21GDN-64K-SFT`](https://huggingface.co/amd/HyLo-Qwen-7MLA21GDN-64K-SFT) | Qwen3-1.7B | 7 MLA + 21 GDN | Gated DeltaNet | 64K | **96.1%** |
| [`HyLo-Qwen-14MLA14GDN-8K-SFT`](https://huggingface.co/amd/HyLo-Qwen-14MLA14GDN-8K-SFT) | Qwen3-1.7B | 14 MLA + 14 GDN | Gated DeltaNet | 8K | **92.2%** |
| [`HyLo-Qwen-14MLA14GDN-64K-SFT`](https://huggingface.co/amd/HyLo-Qwen-14MLA14GDN-64K-SFT) | Qwen3-1.7B | 14 MLA + 14 GDN | Gated DeltaNet | 64K | **92.2%** |
| [`HyLo-Qwen-14MLA14M2-8K-SFT`](https://huggingface.co/amd/HyLo-Qwen-14MLA14M2-8K-SFT) | Qwen3-1.7B | 14 MLA + 14 M2 | Mamba-2 | 8K | **92.2%** |
| [`HyLo-Qwen-14MLA14M2-64K-SFT`](https://huggingface.co/amd/HyLo-Qwen-14MLA14M2-64K-SFT) | Qwen3-1.7B | 14 MLA + 14 M2 | Mamba-2 | 64K | **92.2%** |

Table 1: The HyLo model family.

Every checkpoint is Apache-2.0 licensed and ships as a single `model.safetensors` alongside a
`hybrid_config.json` that records its layer plan, so the loader can rebuild the right mix of MLA and
linear layers without being told. Each model card carries the full per-task numbers and the exact
training recipe for that checkpoint. One licensing caveat worth reading before you build on the
64K models: the long-context portion of the training mixture derives from
[`nvidia/ChatQA2-Long-SFT-data`](https://huggingface.co/datasets/nvidia/ChatQA2-Long-SFT-data),
which is non-commercial, and that constrains models you train with the full mixture regardless of
the Apache-2.0 licence on the code.

## Why This Architecture

The KV cache is what makes long context expensive. A standard Transformer stores keys and values for
every token in every layer, so memory grows linearly with sequence length and multiplies by depth.
Llama-3.2-3B has 28 attention layers, and their combined cache is what makes it run out of memory
past 64K in our serving benchmarks.

Zebra-HyLo attacks this from two directions at once, which is the reason for the hybrid design:

- **Multi-head Latent Attention (MLA)**, following DeepSeek-V3 and the MLA design introduced in
  DeepSeek-V2 [7], keeps real attention but caches a
  low-rank *latent* instead of full keys and values. Per token, the cache shrinks from
  `2 · H_kv · d_h` to `r_kv + d_rope`. It preserves precise token-to-token retrieval at a fraction
  of the memory.
- **Linear blocks** — Mamba-2 [5] or Gated DeltaNet [6] — carry a fixed-size recurrent state and cache
  *nothing*. Their memory is constant in sequence length.

The ratio between the two sets the quality-efficiency trade-off. More MLA layers means more
attention capacity and more cache; more linear layers means less memory and weaker exact retrieval.
Neither extreme works well: a pure linear model loses the retrieval precision that long-context
tasks demand, which is visible in the baselines below, where pure-Mamba Llamba scores 0.0 on RULER
beyond 8K.

An important detail is what we *keep*. In both MLA and GDN layers, the SwiGLU MLP and the RMSNorm
sublayers are copied verbatim from the original Transformer block. Only the token mixer is replaced.
Most of the pretrained computation survives conversion untouched, which is why so little
post-training is needed.

## How Zebra-HyLo Works

### Step 1: Initialize the New Blocks from Pretrained Weights

We first build a pure MLA model and a pure linear model by replacing *every* attention block, each
initialized from the original pretrained weights.

For MLA, we follow [X-EcoMLA](https://arxiv.org/abs/2503.11132) [2] and factor the teacher's attention
projections with a truncated SVD. Given the query projection `W_Q = U Σ Vᵀ`, the down-projection is
initialized from `Σ[:r_q] V[:r_q, :]ᵀ` and the up-projection from the corresponding block of `U`,
keeping only the `nope` and `rope` dimensions MLA actually uses. Keys and values are decomposed
jointly, as `[W_K, W_V]`, because MLA shares one latent between them.

For GDN, the mismatch is dimensional rather than spectral, and the transfer is direct:

1. **GQA expansion.** When the backbone uses fewer KV heads than query heads (8 vs 32 in
   Llama-3.2-1B), each KV head is repeated `H_q / H_kv` times.
2. **Truncation.** GDN's key dimension is smaller than the model dimension and its value dimension
   is larger, so we copy the overlapping submatrices of `W_Q`, `W_K`, `W_V` and `W_O`.

GDN-specific parameters — the gate projection, decay parameters, beta projection and short
convolution kernels — keep their default random initialization. There is nothing in the Transformer
that corresponds to them.

### Step 2: Enhanced-ILD, Our Stage-1 Refinement

Initialization alone leaves the new blocks approximately right. Stage 1 is a short Intermediate
Layer Distillation (ILD) pass that aligns them with the teacher layer by layer, at 2K context, on
20% of the data.

Zebra-Llama aligned hidden states. Zebra-HyLo's change is small and matters more than we expected: we add
a term on the **token-mixer outputs** themselves, so the loss constrains both what each layer
outputs and what its attention replacement computes internally:

```text
L_ILD = Σ_ℓ [ ‖h_ℓ(s) − h_ℓ(t)‖₂ + ‖a_ℓ(s) − a_ℓ(t)‖₂ ]
```

where `h` are per-layer hidden states and `a` are attention/token-mixer outputs, for student (s) and
teacher (t). Supervising the mixer directly is what tells a GDN or Mamba-2 block to imitate
attention, rather than merely producing a hidden state that happens to match downstream.

The payoff shows up disproportionately in math reasoning:

| Model and setting | Commonsense avg | GSM8K |
| --- | --- | --- |
| 1B-4MLA12M2 | 51.8 | 37.2 |
| 1B-4MLA12M2 + Enhanced-ILD | **52.8** | **43.5** |
| 1B-8MLA8M2 | 53.1 | 43.4 |
| 1B-8MLA8M2 + Enhanced-ILD | **53.4** | **48.8** |
| 8B-8MLA24M2 | 62.1 | 66.3 |
| 8B-8MLA24M2 + Enhanced-ILD | **62.3** | **72.4** |

Table 2: Enhanced-ILD adds around a point of commonsense accuracy but 5-6 points of GSM8K, at every
scale we tried.

### Step 3: Long-Context SFT with a Teacher

Stage 2 assembles the hybrid from the stage-1 MLA and linear checkpoints and trains it end to end
against the teacher's next-token distribution, using pure KL divergence (`kl_weight: 1.0`,
`ce_weight: 0.0`), with YaRN [9] scaling on the MLA layers' RoPE. Linear layers need no positional
adjustment, since they have no positional embeddings to scale.

This is where the context length gets extended, from the 2K of stage 1 out to 8K or 64K. And this
turns out to be the single most consequential choice in the recipe.

## Making 64K Distillation Fit in Memory

Extending distillation from 2K to 64K is not a matter of changing one number. The blocker is the
logit tensor: KL divergence needs both student and teacher logits of shape `(T, V)`. At `T=65,536`
and `V=128,256` for Llama-3, each of those is about 16 GB in bfloat16 — before any activations, and
with an 8B teacher resident alongside the student.

The honest version of the story is that most configurations simply do not run:

| Configuration | Memory |
| --- | --- |
| No teacher | OOM |
| No teacher + activation checkpointing | 131 GiB |
| No teacher + FusedLinearCE + activation checkpointing | 29.6 GiB |
| 8B teacher | OOM |
| 8B teacher + Fused KL | OOM |
| 8B teacher + activation checkpointing | OOM |
| 8B teacher + Fused KL Hidden | 158.8 GiB |
| 8B teacher + Fused KL + activation checkpointing | 144.8 GiB |
| 8B teacher + Fused KL Hidden + activation checkpointing | **54.2 GiB** |

Table 3: Training memory for a Llama-1B 4MLA12M2 model at 64K context, with and without the teacher.

The technique that unlocks it is **fused hidden-state KL**, a logit-free distillation path. The
teacher's forward pass skips its LM head entirely and returns only final hidden states. A fused
Triton kernel from [FLA](https://github.com/fla-org/flash-linear-attention) then computes the KL
directly from hidden states and LM-head weights, tiling over the vocabulary with online softmax so
neither logit matrix is ever materialized. That removes roughly 32 GB at 64K. The teacher's LM-head
weight is reached through FSDP's `summon_full_params` so no sharded parameters get duplicated.

Combined with FSDP full sharding across 8 MI300X GPUs, a frozen no-grad teacher, bfloat16, and a
per-device batch size dropped from 4 to 1, this is what makes a 32x increase in training context
length possible while still finishing in a single epoch.

## Get Started

The code lives in the [AMD-Hybrid-Models](https://github.com/AMD-AGI/AMD-Hybrid-Models) repository,
under [`HyLo/`](https://github.com/AMD-AGI/AMD-Hybrid-Models/tree/main/HyLo). It is deliberately
small — the hybrid model definition (`hybrid/`), one distillation training entry point
(`train_hybrid/`), the trainer subclass (`trainer/`), and `configs/`, which holds nothing but the
recipes behind the release: one stage-1 ILD recipe per backbone and block type, and the stage-2 SFT
recipe for each of the 14 published checkpoints. The sibling directories `X-EcoMLA/` and
`Zebra-Llama/` in the same repository hold the predecessor projects this recipe builds on.

### Environment

The released checkpoints were trained in a container built from the public
`rocm/pytorch-training:v26.1` image plus the repository's `install.sh`. That image supplies the whole
GPU stack — ROCm 7.1, `torch` 2.10.0, Triton built from `ROCm/triton`, and FlashAttention 2.8.3 for
gfx942/gfx950 — so `install.sh` never touches it.

```bash
git clone https://github.com/AMD-AGI/AMD-Hybrid-Models.git
cd AMD-Hybrid-Models/HyLo

docker run -it \
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  --network host --ipc host --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  -v $HOME:$HOME --shm-size 64G --name hylo \
  rocm/pytorch-training:v26.1

# inside the container, from this directory
bash install.sh
```

`install.sh` builds the linear-block kernels from source — `mamba-ssm` 2.3.1 and `causal-conv1d`
1.6.1 for the Mamba-2 models, flash-linear-attention for the GDN models — and pins the Python
training stack. Both kernel families need a small AMD patch to train on MI300-class GPUs, which
`install.sh` applies for you; `patches/README.md` documents what each one changes and how to apply
it by hand. The script ends by printing a table comparing what you installed against the reference
environment.

Two details in it are load-bearing rather than stylistic, and worth keeping if you adapt it. Every
`pip install` is resolved against a constraints file generated from the `torch` and Triton already
present, so the resolver cannot quietly swap the ROCm build for a PyPI CUDA wheel. And the fused
kernels are built with `--no-build-isolation`, since an isolated build environment would fetch its
own `torch` to compile against. Both failure modes look like a clean install and then break at the
first kernel launch.

### Running the Two Stages

Stage 1 runs twice, once per block type, because the stage-2 config initializes its MLA and linear
layers from two separate ILD checkpoints:

```bash
# Stage 1a - Enhanced-ILD for the linear (Gated DeltaNet) layers
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file configs/fsdp_GDN_ILD.yaml \
  train_hybrid/train_distill.py \
  configs/llama3.2_1B/zebra_GDN_ILD.yaml

# Stage 1b - Enhanced-ILD for the MLA layers
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file configs/fsdp_MLA_ILD.yaml \
  train_hybrid/train_distill.py \
  configs/llama3.2_1B/zebra_MLA_ILD.yaml

# Stage 2 - long-context SFT with teacher-guided distillation
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file configs/fsdp_GDN.yaml \
  train_hybrid/train_distill.py \
  configs/llama3.2_1B/zebra_4MLA12GDN_8bT_SFT_8k_combined.yaml
```

The two stages chain without path edits: stage 1 writes `output_dir: checkpoints/Zebra-1B-ILD-GDN`
and `checkpoints/Zebra-1B-ILD-MLA`, which are exactly the paths the 1B stage-2 configs read through
`linear_ILD_path` and `mla_ILD_path`. `train.sh` is a worked end-to-end example, and
`multinode.slurm` covers multi-node launches. After training, `accelerate merge-weights` collapses
the FSDP shards into a single `model.safetensors`.

### Reading a Config

The recipe lives entirely in YAML, so switching architecture, block type or context length is a
config change rather than a code change. The important fields, from the released 1B GDN recipe at
64K:

```yaml
model_name_or_path: meta-llama/Llama-3.2-1B-Instruct
teacher_model_name_or_path: meta-llama/Llama-3.1-8B-Instruct
linear_type: gdn                 # 'gdn' or 'mamba'
linear_ILD_path: checkpoints/Zebra-1B-ILD-GDN
mla_ILD_path: checkpoints/Zebra-1B-ILD-MLA
mla_layers: [1, 5, 10, 14]       # every other layer becomes a linear block

max_seq_length: 65536
factor: 32.0                     # YaRN scaling from the 2048 original window
original_max_position_embeddings: 2048

kl_weight: 1.0                    # pure teacher KL, no cross-entropy term
ce_weight: 0.0
fused_kl_hidden: true             # the logit-free path from the memory section
per_device_train_batch_size: 1

# MLA geometry: this is what sets the KV-cache footprint
q_lora_rank: 1344
kv_lora_rank: 128
qk_rope_head_dim: 32
qk_nope_head_dim: 32
v_head_dim: 64
```

Switching this run to Mamba-2 means setting `linear_type: mamba`, pointing `linear_ILD_path` at an
M2 stage-1 checkpoint, and using `configs/fsdp.yaml` instead of `configs/fsdp_GDN.yaml` so FSDP
wraps `Mamba2DecoderLayer` rather than `GDNDecoderLayer`. Going from 64K to 8K means
`max_seq_length: 8192` with `factor: 4.0`.

Note on data: the released checkpoints were trained on a mixture that included four datasets AMD
preprocessed internally, which are not part of the release. The shipped configs enable only the
public general-SFT set, and each one carries the full upstream list in a comment next to the mixer
block, with the licence terms you inherit if you rebuild it.

### Loading a Checkpoint

```python
import torch
from transformers import AutoTokenizer
from hybrid.hybrid_wrapper import HybridModelWrapper

model_id = "amd/HyLo-Llama-4MLA12GDN-64K-SFT"

model = HybridModelWrapper.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Explain KV-cache compression in two sentences."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).cuda()

out = model.generate(inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

`HybridModelWrapper` reads the layer plan from `hybrid_config.json` and builds the matching MLA and
linear layers, so a checkpoint loads without you having to describe its architecture. Load in
bfloat16 — that is the precision every number here was measured at.

## Results

All models were trained on 8 AMD Instinct MI300X GPUs with FSDP full sharding in bfloat16. Stage 1
uses 2K context, learning rate 2e-4, and 20% of the SFT data; stage 2 trains one epoch at 8K or 64K
with learning rate 6e-5 for the 1B and Qwen models and 4e-5 for the 3B models. Commonsense is the
0-shot average over ARC-Challenge, ARC-Easy, HellaSwag, OpenBookQA, PIQA, RACE and WinoGrande;
RULER [8] is the average over all 13 tasks; GSM8K is 0-shot. Short-context tasks are scored with the
LM Evaluation Harness [12].

### Llama-3.2-1B

| Model and setting | KV cache | Commonsense avg | GSM8K | RULER 8K | 16K | 32K | 64K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MambaInLlama-1B-50% | 50% | 52.6 | 16.2 | 18.9 | 3.0 | 1.0 | 0.0 |
| Llamba-1B | 0% | 53.2 | 12.5 | 2.9 | 0.0 | 0.0 | 0.0 |
| Zebra-Llama-1B (4MLA-12M2) | 4% | 51.8 | 37.2 | 12.3 | 6.8 | 3.7 | 0.1 |
| **HyLo-Llama-4MLA12GDN** (8K) | 3.9% | 53.1 | 51.9 | 55.1 | 11.9 | 2.4 | 0.8 |
| **HyLo-Llama-8MLA8GDN** (8K) | 7.8% | 53.4 | 54.6 | 60.3 | 0.5 | 0.1 | 0.1 |
| **HyLo-Llama-4MLA12GDN** (64K) | 3.9% | 51.2 | 37.5 | 52.5 | 48.3 | 44.5 | **40.8** |
| **HyLo-Llama-8MLA8GDN** (64K) | 7.8% | 51.8 | 39.4 | 61.5 | 53.7 | 48.1 | **41.6** |

Table 4: Llama-3.2-1B backbone. The 64K-trained models are the only ones with usable accuracy past
16K.

### Llama-3.2-3B

| Model and setting | KV cache | Commonsense avg | GSM8K | RULER 8K | 16K | 32K | 64K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MambaInLlama-3B-50% | 50% | 58.7 | 56.8 | 37.0 | 1.0 | 0.0 | 0.0 |
| Llamba-3B | 0% | 60.5 | 47.8 | 3.5 | 0.0 | 0.0 | 0.0 |
| M1 | 21.4% | 56.2 | 62.5 | 63.5 | 43.6 | 30.3 | 17.4 |
| Zebra-Llama-3B (14MLA-14M2) | 4.7% | 58.0 | 66.2 | 35.1 | 13.3 | 6.3 | 4.2 |
| **HyLo-Llama-14MLA14GDN** (8K) | 4.7% | 59.2 | 68.2 | 71.1 | 45.6 | 19.2 | 0.2 |
| **HyLo-Llama-6MLA22GDN** (64K) | 2.0% | 57.2 | 56.0 | 68.2 | 62.1 | 55.7 | 46.3 |
| **HyLo-Llama-14MLA14GDN** (64K) | 4.7% | 57.9 | 58.9 | 73.2 | 69.7 | 62.9 | **52.0** |

Table 5: Llama-3.2-3B backbone. `14MLA14GDN` at 64K holds 52.0 on RULER-64K with under 5% of the
backbone's KV cache.

### Qwen3-1.7B

| Model and setting | KV cache | Commonsense avg | GSM8K | RULER 8K | 16K | 32K | 64K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jet-Nemotron-2B | 2.1% | 52.7 | 19.4 | 71.3 | 60.1 | 43.9 | 14.1 |
| HypeNet (7FA21LA) | 25% | 53.2 | 1.1 | 36.4 | 31.3 | 23.8 | 16.4 |
| **HyLo-Qwen-14MLA14GDN** (8K) | 7.8% | 56.5 | 76.1 | 71.1 | 45.6 | 19.2 | 0.2 |
| **HyLo-Qwen-7MLA21GDN** (64K) | 3.9% | 55.4 | 73.3 | 59.8 | 53.8 | 42.5 | 30.5 |
| **HyLo-Qwen-14MLA14M2** (64K) | 7.8% | 55.7 | 73.5 | 73.9 | 62.6 | 46.2 | **33.1** |

Table 6: Qwen3-1.7B backbone. The comparison worth dwelling on is Jet-Nemotron-2B [11], pretrained from
scratch on a far larger token budget: HyLo-Qwen-14MLA14M2 beats it by 3 points of commonsense
accuracy, 54 points of GSM8K and 19 points of RULER-64K, on roughly 10B tokens of post-training.

### Training Length Beats Position Interpolation

The cheap way to get long context is to train short and extrapolate, so we tested it. We trained at
several sequence lengths with the token budget held constant, then applied YaRN interpolation to the
MLA layers' RoPE.

YaRN works, and it is not enough. For 1B-4MLA12M2 trained at 8K, RULER-64K goes from 0.5 to 31.3
after YaRN scaling, while short-context accuracy slips from 50.7 to 49.0. But training directly at
64K reaches 37.9 on RULER-64K with comparable short-context accuracy. Interpolation recovers most of
the gap for free; actually training long closes it. Figure 2 below plots this comparison across
training sequence lengths.

```{figure} ./images/figure3-yarn.png
:align: center
:alt: Impact of training sequence length and YaRN position interpolation
Figure 2: Training sequence length versus YaRN interpolation. YaRN improves long context with a
slight short-context cost, but training at longer context preserves long-context ability better.
Source: Figure 3 from the [HyLo paper](https://arxiv.org/abs/2604.24715).
```

### The Teacher Matters More at Long Context

Distillation was known to help short-context upcycling. Its effect on long context is much larger.
For 1B-4MLA12M2 trained at 64K, moving from no teacher to an 8B teacher improves commonsense
reasoning by about 6 points — and RULER-64K by 22 points, from 15.4 to 37.9. Larger teachers help
monotonically, and the effect persists when long context comes from YaRN rather than training:
RULER-64K still improves by 14 points. Figure 3 below plots this effect of teacher size.

```{figure} ./images/figure4-teacher.png
:align: center
:alt: Impact of teacher size on long-context knowledge distillation
Figure 3: Teacher size at long-context distillation. A larger teacher improves both short-context
commonsense reasoning and long-context ability. Source: Figure 4 from the
[HyLo paper](https://arxiv.org/abs/2604.24715).
```

### Inference: 2M Tokens on Eight MI300X GPUs

We integrated Zebra-HyLo into vLLM [10], which required extending the serving stack to handle interleaved
Mamba/GDN state and MLA KV cache in one engine — the scheduler has to manage a fixed-size recurrent
state and a growing attention cache simultaneously, MLA's KV compression needs its own cache
allocation logic, and Zebra-HyLo's compressed latent head dimensions are not supported by existing fused
attention kernels.

With TP=8 and batch size 1 on a single node of 8 MI300X GPUs, we swept the context length from 8K to 2M:

- **Prefill.** Between 8K and 64K, Zebra-HyLo and Llama-3.2-3B are comparable. Past 64K, Llama-3B runs out
  of memory, since its 28 attention layers each hold a full KV cache. Both Zebra-HyLo variants complete
  the sweep to 2M, a 30x context extension over the baseline.
- **Decode.** Llama-3B has lower per-token latency at 8K-32K, but its latency grows linearly with
  context and it OOMs at 128K. HyLo-Llama-6MLA22M2 holds flat per-token latency from 8K through 64K,
  because the linear layers use a fixed-size state instead of a growing cache. Past 64K it rises
  only sub-linearly as the MLA cache grows.
- **Layer budget shows up directly.** At 2M, `6MLA22M2` is about 2x faster than `14MLA14M2`,
  reflecting the quadratic cost of 14 attention layers versus 6.

Figure 4 below shows the time-to-first-token and per-token latency curves behind these observations.

```{figure} ./images/figure5-latency.png
:align: center
:alt: TTFT and TPOT comparison for 3B models on vLLM
Figure 4: Time-to-first-token and time-per-output-token for 3B models on vLLM, MI300X, TP=8.
Source: Figure 5 from the [HyLo paper](https://arxiv.org/abs/2604.24715).
```

## What We Learned

- **Long context is a training objective, not an architectural gift.** The same architecture scores
  0.8 or 40.8 on RULER-64K depending only on whether it was trained at 8K or 64K.
- **Supervise the token mixer, not just the layer output.** One extra ILD term on mixer outputs was
  worth 5-6 GSM8K points at every scale.
- **Pick the variant that matches your sequence length.** 8K-trained models are the stronger
  short-context models; 64K-trained models trade a little commonsense and GSM8K accuracy for
  long-context behaviour that actually holds up. This is a real trade-off, not a strict improvement.
- **Memory engineering is what makes the science possible.** Long-context distillation with an 8B
  teacher is not a hyperparameter change; without the logit-free fused KL path it simply does not
  run.

## Summary

In this blog you explored Zebra-HyLo, a post-training recipe that turns pretrained Transformers into
hybrid models built from MLA and linear blocks, with long-context ability treated as a first-class
objective rather than a hoped-for side effect of the architecture. You walked through the complete
pipeline: initializing MLA and linear blocks from pretrained weights, sharpening them with
Enhanced-ILD, extending context through teacher-guided long-context SFT, fitting 64K distillation
into GPU memory with a logit-free fused KL path, and serving the finished model in vLLM. Across
Llama-3.2-1B/3B and Qwen3-1.7B backbones, with either Gated
DeltaNet or Mamba-2 linear blocks, it extends usable context up to 32x, cuts KV-cache memory by more than 90%, and serves up to 2M tokens in vLLM on eight AMD Instinct MI300X GPUs—with roughly 10B tokens of post-training and no pretraining from scratch. The training code, the exact configs for all 14
released checkpoints, and the models are all available.

Now put the recipe to work on your own models. Pull a checkpoint from the
[HyLo collection on Hugging Face](https://huggingface.co/collections/amd/hylo-long-context-aware-upcycling-for-hybrid-llms-6aa40540e1398545571f476a),
run the released configs against your backbone of choice, and compare the numbers against the
baselines in this post. Our team is continuing this line of work: scaling long-context upcycling to
larger backbones, improving distillation efficiency so the recipe fits a smaller compute budget, and
broadening the hybrid block library beyond Mamba-2 and Gated DeltaNet. Watch for follow-up posts on
[ROCm Blogs](https://rocm.blogs.amd.com/), and check the
[AMD-Hybrid-Models](https://github.com/AMD-AGI/AMD-Hybrid-Models) repository for the latest recipes.

## References

[1] Wang, Junxiong, Daniele Paliotta, Avner May, Alexander M. Rush, and Tri Dao. "The mamba in the llama: Distilling and accelerating hybrid models." NeurIPS 37 (2024).

[2] Li, Guihong, Mehdi Rezagholizadeh, Mingyu Yang, Vikram Appia, and Emad Barsoum. "X-EcoMLA: Upcycling pre-trained attention into MLA for efficient and extreme KV compression." arXiv:2503.11132 (2025).

[3] Yang, Mingyu, Mehdi Rezagholizadeh, Guihong Li, Vikram Appia, and Emad Barsoum. "Zebra-Llama: Towards extremely efficient hybrid models." arXiv:2505.17272 (2025).

[4] Bick, Aviv, Tobias Katsch, Nimit Sohoni, Arjun Desai, and Albert Gu. "Llamba: Scaling distilled recurrent models for efficient language processing." arXiv:2502.14458 (2025).

[5] Dao, Tri, and Albert Gu. "Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality." arXiv:2405.21060 (2024).

[6] Yang, Songlin, Jan Kautz, and Ali Hatamizadeh. "Gated delta networks: Improving Mamba2 with delta rule." arXiv:2412.06464 (2024).

[7] Liu, Aixin, et al. "DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model." arXiv:2405.04434 (2024).

[8] Hsieh, Cheng-Ping, et al. "RULER: What's the real context size of your long-context language models?" arXiv:2404.06654 (2024).

[9] Peng, Bowen, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. "YaRN: Efficient context window extension of large language models." arXiv:2309.00071 (2023).

[10] Kwon, Woosuk, et al. "Efficient memory management for large language model serving with PagedAttention." arXiv:2309.06180 (2023).

[11] Gu, Yuxian, et al. "Jet-Nemotron: Efficient language model with post neural architecture search." arXiv:2508.15884 (2025).

[12] Gao, Leo, et al. "A framework for few-shot language model evaluation." (2023).

## Additional Resources

- **Paper:** [Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling](https://arxiv.org/abs/2604.24715) (arXiv:2604.24715)
- **Code:** [AMD-Hybrid-Models](https://github.com/AMD-AGI/AMD-Hybrid-Models), HyLo recipe in [`HyLo/`](https://github.com/AMD-AGI/AMD-Hybrid-Models/tree/main/HyLo)
- **Weights:** [HyLo collection on Hugging Face](https://huggingface.co/collections/amd/hylo-long-context-aware-upcycling-for-hybrid-llms-6aa40540e1398545571f476a) — 14 checkpoints, Apache-2.0
- **Predecessors:** [Zebra-Llama](https://arxiv.org/abs/2505.17272) and [X-EcoMLA](https://arxiv.org/abs/2503.11132), and the earlier [AMD-HybridLM blog](https://rocm.blogs.amd.com/artificial-intelligence/hybrid-models,-mla,/README.html)
- **Base image:** [`rocm/pytorch-training:v26.1`](https://hub.docker.com/r/rocm/pytorch-training) on Docker Hub

## Citations

```bibtex
@article{fashi2026hylo,
  title={Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling},
  author={Ashrafi Fashi, Parsa and Saxena, Utkarsh and Rezagholizadeh, Mehdi and Jafari, Aref
          and Haridas, Akash and Yang, Mingyu and Bhatia, Vansh and Li, Guihong
          and Appia, Vikram and Barsoum, Emad},
  journal={arXiv preprint arXiv:2604.24715},
  year={2026},
  url={https://arxiv.org/abs/2604.24715}
}

@article{yang2025zebra,
  title={Zebra-Llama: Towards Extremely Efficient Hybrid Models},
  author={Yang, Mingyu and Rezagholizadeh, Mehdi and Li, Guihong and Appia, Vikram and Barsoum, Emad},
  journal={arXiv preprint arXiv:2505.17272},
  year={2025}
}

@article{li2025x_ecomla,
  title={{X-EcoMLA}: Upcycling Pre-Trained Attention into {MLA} for Efficient and Extreme {KV} Compression},
  author={Li, Guihong and Rezagholizadeh, Mehdi and Yang, Mingyu and Appia, Vikram and Barsoum, Emad},
  journal={arXiv preprint arXiv:2503.11132},
  year={2025}
}
```

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
