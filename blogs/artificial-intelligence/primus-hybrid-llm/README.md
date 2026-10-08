---
blogpost: true
blog_title: "Pre-Training Hybrid LLMs from Scratch in Primus on AMD Instinct™ GPUs"
date: "08 Oct 2026"
author: "Vansh Bhatia, Claire Lee, Aref Jafari, Mehdi Rezagholizadeh, Vikram Appia, Yao Fu, Zhenyu Gu, Emad Barsoum"
thumbnail: 'primus-hybrid-llm-thumbnail.png'
tags: "LLM, PyTorch, AI/ML, Performance"
category: "Applications & models"
target_audience: "AI researchers, AI developers, AI engineers pre-training hybrid or linear-attention LLMs"
key_value_propositions: "Primus pre-trains hybrid models that interleave MLA attention with Gated DeltaNet, Kimi Delta Attention or Mamba2 mixers from a single configuration string, with released recipes and reproducible relative performance on AMD Instinct MI355X GPUs"
language: English
myst:
    html_meta:
        "author": "Vansh Bhatia, Claire Lee, Aref Jafari, Mehdi Rezagholizadeh, Vikram Appia, Yao Fu, Zhenyu Gu, Emad Barsoum"
        "description lang=en": "Pre-train hybrid LLMs mixing MLA attention with Gated DeltaNet, Kimi Delta Attention and Mamba2 mixers on AMD Instinct MI355X GPUs, with open recipes."
        "keywords": "rocm, primus, hybrid models, MLA, mamba2, gated deltanet, kimi delta attention, pretraining, megatron-lm, MI355X"
        "property=og:locale": "en_US"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Vansh Bhatia, Claire Lee, Aref Jafari, Mehdi Rezagholizadeh, Vikram Appia, Yao Fu, Zhenyu Gu, Emad Barsoum"
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

# Pre-Training Hybrid LLMs from Scratch in Primus on AMD Instinct™ GPUs

In this blog you will learn how to pre-train hybrid LLMs from scratch with Primus on AMD Instinct™
MI355X GPUs. In particular, we will cover how to declare a hybrid layer stack with a single configuration
string, how three linear-recurrent mixers (Mamba2, Gated DeltaNet, and Kimi Delta Attention) trade off
throughput with training quality, which recipe settings determine whether your run fits in memory in the first place,
and which environment variables can quietly change your results. Along the way you will receive the released
configurations behind every number in this post, so that you can reproduce the runs yourself. Ready to train
your own hybrid? Pull the Primus image and follow along — everything here runs on a single eight-GPU node.

## Why Train Hybrid Models on Primus?

Language model architecture is moving beyond the pure Transformer. Standard self-attention remains the primary bottleneck for long-context windows:
its compute cost scales quadratically $O(N^2)$ with sequence length, and its Key-Value (KV) cache grows linearly with sequence length.
Conversely, linear-recurrent blocks offer the opposite trade-off—processing sequences through a fixed-size state at linear cost $O(N)$.
However, on their own, linear-recurrent layers struggle with precise, non-local retrieval. Hybrid models bridge the gap between these
trade-offs by combining both types of layers. Recent designs such as Jamba, Samba, Qwen3-Next, and Kimi Linear[^1] interleave a small number of
attention layers within a stack dominated by linear-recurrent blocks. The attention layers give direct access to earlier tokens, which a fixed-size recurrent state cannot guarantee.
Since many of those layers can be linear-recurrent, hybrid architectures reduce memory and compute overhead while retaining retrieval where it matters.

Supporting these architectures in a training framework presents a distinct engineering problem. A hybrid model is a heterogenous stack assembled from distinct sublayer types,
each with its own custom kernels, memory characteristics, and numerical stability demands. A production-grade framework must address three critical requirements:

- **Flexible placement**, so that sublayers can be ordered arbitrarily without code changes
- **Kernel fusion**, so that each sublayer type retains the fused kernels that make it efficient
- **Numerical stability**, maintained across tens of thousands of consecutive iterations

[Primus](https://github.com/AMD-AGI/Primus) AMD's unified, backend-agnostic training framework for
foundation models on AMD Instinct GPUs, is a platform that addresses the aforementioned requirements, supporting backends such as
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)[^2] and [TorchTitan](https://github.com/pytorch/torchtitan)[^3].
This article describes the design and implementation choices made to enable hybrid model pre-training from scratch on AMD GPUs.
Furthermore, we will expand on the architectures Primus supports, the cost associated with each design choice, and the results we
obtained by training five models to completion on a single node of eight MI355X GPUs.

## The Models We Trained

We trained five models to completion on one node of eight MI355X GPUs, all at sequence length 2048 in
bfloat16 on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)[^4]. The label "75%
hybrid" refers to the `*-M-M-M-*-M-M-M-*-M-M-M-` pattern, in which three MLA attention blocks[^5] are
distributed among twelve mixers.

| Model             | Size | Tokens | Final loss |
| ----------------- | ---- | ------ | ---------- |
| Pure GDN          | 1B   | 100B   | **2.487**  |
| Pure GDN          | 300M | 10B    | 3.411      |
| Pure KDA          | 300M | 10B    | 3.407      |
| 75% Hybrid Mamba2 | 300M | 10B    | 3.385      |
| 75% Hybrid GDN    | 300M | 10B    | **3.382**  |

Table 1: The five pre-training runs that produced the results in this article. Each completed with no
NaN values and no skipped iterations, and the configurations used are included in the repository.

## How Hybrid Models Are Defined in Primus

A hybrid model in Primus is a single layer stack constructed from three kinds of sublayer, which may
be interleaved in any order.

| Symbol | Sublayer                                                                 | Cost                                                         |
| ------ | ------------------------------------------------------------------------ | ------------------------------------------------------------ |
| `*`    | Attention block (MLA, multi-head latent attention)                       | Quadratic in sequence length, attends over the full sequence |
| `M`    | Linear-recurrent mixer (Mamba2, Gated DeltaNet or Kimi Delta Attention)  | Linear in sequence length, constant-size state               |
| `-`    | MLP / SwiGLU feed-forward block                                          | Standard                                                     |

Table 2: The three sublayer types from which a hybrid stack is assembled.

The arrangement is declared directly in the configuration, and no other part of the model definition
changes:

```yaml
model_type: mamba          # required for every hybrid model
is_hybrid_model: true
hybrid_override_pattern: "*-M-M-M-*-M-M-M-*-M-M-M-"
```

This 24-character pattern describes twelve mixer blocks alternating with twelve MLP blocks, with
attention placed at mixer positions 0, 4 and 8 and a linear mixer at every other position. A purely
linear model is simply a pattern containing no `*` characters, which may also be expressed as
`hybrid_attention_ratio: 0.0`.

For all mixers, the surrounding SwiGLU MLP and RMSNorm sublayers are identical, as is the training recipe.
Because only the token mixer changes, swapping one for another takes just a single-line configuration change.
The three mixers differ fundamentally in how they determine what to retain in their fixed-size state.

- **Mamba2**[^6] is a selective state-space model (SSM) that decays its state using one scalar gate per
head.
- **Gated DeltaNet (GDN)**[^7] builds on Mamba2 by adding the delta rule, a form of associative memory that
updates the stored association for the current key rather than only accumulating into the state, while keeping one scalar forget gate per head.
- **Kimi Delta Attention (KDA)**[^1] extends GDN by refining that gate to one value per key dimension,
computed through a low-rank bottleneck projection. This optimization provides finer control over what the
state retains, at a small cost in throughput.

| Mixer                      | Sizes            | Pure config                        | Hybrid config (with MLA)                          |
| -------------------------- | ---------------- | ---------------------------------- | ------------------------------------------------- |
| Mamba2                     | 370M, 1B, 3B, 8B | `mamba_370M-pretrain.yaml`         | `zebra_llama_mamba_{1B,3B,8B}_BF16-pretrain.yaml` |
| Gated DeltaNet (GDN)       | 300M, 1B         | `gdn_{300M,1B}_BF16-pretrain.yaml` | `zebra_llama_gdn_1B_BF16-pretrain.yaml`           |
| Kimi Delta Attention (KDA) | 300M, 1B         | `kda_{300M,1B}_BF16-pretrain.yaml` | `zebra_llama_kda_1B_BF16-pretrain.yaml`           |

Table 3: Released configurations, located in `examples/megatron/configs/MI355X/`. Hybrid
configurations are also provided for MI300X, and the Mamba2 configurations for MI325X.

In these filenames the `zebra_llama_` prefix records the lineage of the hybrid model, and the suffix names the linear
mixer and size, as in `zebra_llama_gdn_1B`. The particular combination of MLA attention with linear blocks
used here follows the layer plans established by [Zebra-Llama](https://arxiv.org/abs/2505.17272)[^8] and extended by
[HyLo](https://arxiv.org/abs/2604.24715)[^9]. Note that although both papers train hybrid models by upcycling existing Transformer checkpoints through post-training, the work described in this article pre-trains the Zebra-Llama architectures from scratch through random initialization.

## The Recipe

The default released configurations include all of the settings below with the exception of the batch size and
iteration count, which are deliberately set small so that an initial run completes quickly.

| Setting                     | 300M                                                     | 1B                                                       |
| --------------------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| Sequence length             | 2048                                                     | 2048                                                     |
| Precision                   | bfloat16                                                 | bfloat16                                                 |
| Micro-batch x GPUs          | 128 x 8                                                  | 64 x 8                                                   |
| Global batch                | 1024 (2.10M tokens/iter)                                 | 512 (1.05M tokens/iter)                                  |
| Iterations                  | 4,768 (10B tokens)                                       | 95,368 (100B tokens)                                     |
| Learning rate               | 2e-4 to 2e-5, cosine                                     | 3e-4 to 3e-5, cosine                                     |
| Warmup                      | 200 iterations                                           | 2,000 iterations                                         |
| AdamW                       | beta = (0.9, 0.95), weight decay 0.01, gradient clip 1.0 | beta = (0.9, 0.95), weight decay 0.01, gradient clip 1.0 |
| Parallelism                 | TP = PP = EP = 1 (pure data parallel)                    | TP = PP = EP = 1 (pure data parallel)                    |
| Optimizer state             | replicated                                               | ZeRO-1 sharded (`use_distributed_optimizer`)             |
| Cross-entropy               | `fused_ce_mode: 2`, 8 chunks                             | `fused_ce_mode: 2`, 8 chunks                             |
| FLA fusions                 | fused SwiGLU, RMSNorm, gated norm, short conv            | fused SwiGLU, RMSNorm, gated norm, short conv            |
| Dropout / LayerNorm epsilon | 0.0 / 1e-6                                               | 0.0 / 1e-6                                               |
| Seed                        | 42                                                       | 42                                                       |

Table 4: The complete pre-training recipe. Each column is a self-contained setup.

Two entries in this table are operationally critical, because they determine whether a run executes at
all rather than merely how it converges.

- **Chunked fused cross-entropy** prevents the logits tensor from dominating activation memory at a
micro-batch of this size. At these shapes it is the difference between a run that fits and one that
raises an out-of-memory error.
- **The four `use_fla_*` flags** replace the elementwise SwiGLU, normalization and short-convolution
paths with Triton kernels from [FLA](https://github.com/fla-org/flash-linear-attention). Disabling
them reduces both throughput and available headroom.

The remaining settings constitute a conventional GPT-style pre-training recipe and may be treated as
such.

## Getting Started

### Environment

```bash
docker pull rocm/primus:v26.7
docker run -it \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --network host --ipc host \
    --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged \
    --shm-size 128G \
    rocm/primus:v26.7
```

Inside the container, clone Primus from the `main` branch, where the Zebra-Llama configurations used in
this post live, together with its submodules:

```bash
git clone --recurse-submodules https://github.com/AMD-AGI/Primus.git
cd Primus
pip install -r requirements.txt
```

Four settings should be established before the first run:

```bash
export HF_TOKEN=<hf_token>                              # gated tokenizer repository
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # see discussion below
export PRIMUS_FLA_MLA_ATTN=0                            # hybrid models only
git config --global --add safe.directory '*'            # if mounting a host clone
```

Access to the tokenizer repository is a strict requirement. The released configurations tokenize with
`meta-llama/Llama-3.2-1B`, a gated Hugging Face repository, and without valid credentials the
launcher terminates with an HTTP 401 error during setup. This occurs even when training on mock data,
which does not otherwise require a tokenizer.

The allocator setting has a greater effect than its brevity suggests. These models allocate a small
number of very large tensors, and without expandable segments the caching allocator becomes fragmented
around them. Reserved memory then grows substantially beyond what is actually in use, and a run with
genuine headroom remaining can still fail because no single contiguous block is available. With the
setting enabled, reserved memory tracks allocated memory closely. It costs approximately 4% in
throughput and is a prerequisite for the larger micro-batch sizes.

`PRIMUS_FLA_MLA_ATTN` applies only to the hybrid models. Setting it to `0` keeps the MLA blocks on the
native attention path, which delivers the highest throughput at the shapes used here. The setting must
be supplied as an environment variable, because the ambient value takes precedence over the YAML key
of the same name.

Finally, if a repository clone is mounted from the host rather than created inside the image, git will
decline to inspect a tree owned by another user, which causes the editable installations to fail
during setup. The `safe.directory` entry addresses this.

### Model Configuration and Dataset

Primus separates configuration into two layers, so that the training recipe can be modified without
altering the architecture. The model definition, under `primus/configs/models/megatron/`, specifies
layer count, hidden size, the hybrid pattern and the mixer hyperparameters. The experiment
configuration references a model definition and supplies everything else:

```yaml
modules:
  pre_trainer:
    model: zebra_gdn_1B_hybrid.yaml
    overrides:
      micro_batch_size: 128
      global_batch_size: 1024
      seq_length: 2048
      lr: 2.0e-4
      tensor_model_parallel_size: 1
      pipeline_model_parallel_size: 1
```

The released configurations sets `mock_data: true`, allowing a run to be launched immediately in
order to validate the software stack without any dataset. For training on real data, prepare a
Megatron-format dataset once, then set `mock_data: false` and point `train_data_path` at the result:

```bash
python3 examples/megatron/prepare_fineweb_edu_megatron_dataset.py \
    --out-dir ./data/fineweb-edu --sample-size 10BT

python3 examples/megatron/preprocess_data.py \
    --input ./data/fineweb-edu/fineweb_edu_10BT_megatron.json \
    --output-prefix ./data/fineweb_edu_10BT \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Llama-3.2-1B \
    --append-eod --workers 32
```

### Running a Job

Primus launches every run through a single entry point, `primus-cli`. From the Primus repo inside
the container, start a single-node run in direct mode and pass the experiment configuration.
`GPUS_PER_NODE` defaults to 8, and logs and checkpoints are written under `./output`:

```bash
export EXP=examples/megatron/configs/MI355X/gdn_300M_BF16-pretrain.yaml
./primus-cli direct -- train pretrain --config "$EXP"
```

Changing the above run from GDN to KDA requires a single edit: point `EXP` at
`kda_300M_BF16-pretrain.yaml` instead. On a Slurm cluster, launch from a login node with the Primus
checkout on a shared filesystem. `primus-cli` starts a container on each node, mounts the checkout into
it and forwards `HF_TOKEN`; pass the other two settings from the environment section with `--env`.
Without Slurm, run the direct-mode command inside the container on every node with the rendezvous
variables set:

```bash
# Slurm
export EXP=examples/megatron/configs/MI355X/zebra_llama_mamba_8B_BF16-pretrain.yaml
./primus-cli slurm srun -N 4 -- container \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env PRIMUS_FLA_MLA_ATTN=0 \
    -- train pretrain --config "$EXP"

# without Slurm, on each node (NODE_RANK=1 on the second node)
export NNODES=2 NODE_RANK=0 MASTER_ADDR=<master-ip>
./primus-cli direct -- train pretrain \
    --config examples/megatron/configs/MI355X/zebra_llama_mamba_8B_BF16-pretrain.yaml
```

The 300M configurations ship with a deliberately small recipe, consisting of micro-batch 2, global
batch 16, and 50 iterations, so that training itself completes promptly and confirms that the
software stack is functioning. Be aware that the first launch is expected to take longer because
the launcher's setup hooks install the training dependencies before training begins as these dependencies
not cached.

To reproduce the results reported below, increase `micro_batch_size` to 128 and `global_batch_size` to
1024, which corresponds to eight GPUs at a micro-batch of 128 with no gradient accumulation. Each
configuration documents the largest micro-batch that has been verified to fit on a single MI355X.

## Results

All results were obtained on a single node of eight AMD Instinct MI355X GPUs at sequence length 2048
in bfloat16, using the released configurations together with the environment settings described above.

Reproducing them depends on two runtime settings. The hybrid configurations ship with the chunked
`fused_ce_mode: 1`, which prioritizes memory efficiency; set `fused_ce_mode: 2` for the higher
throughput used in these results, and `PRIMUS_FLA_MLA_ATTN=0`.

The figure below shows we achieve the lowest loss with a hybrid model with only three attention layers compared to the other models:

```{figure} ./images/loss_300M_comparison.png
:align: center
:alt: Training loss for four 300M hybrid and pure models over 10B tokens
Figure 1: Training loss for the four 300M models over 10B tokens. The final 2B tokens are shown
separately on the right, where the curves begin to separate. The 75% hybrid GDN model reaches the
lowest final loss.
```

### Loss Curves

The figures below show the training loss of each of the five runs individually, starting with the
1B model:

````{grid} 3
:gutter: 2

```{figure} ./images/loss_gdn_pure_1B.png
:align: center
:alt: Pure GDN 1B training loss over 100B tokens
Figure 2: Pure GDN 1B over 100B tokens, covering 95,368 iterations from a loss of 12.17 to 2.487 and
ending at a gradient norm of 0.102.
```

```{figure} ./images/loss_gdn_pure_300M.png
:align: center
:alt: Pure GDN 300M training loss over 10B tokens
Figure 3: Pure GDN 300M over 10B tokens.
```

```{figure} ./images/loss_kda_pure_300M.png
:align: center
:alt: Pure KDA 300M training loss over 10B tokens
Figure 4: Pure KDA 300M over 10B tokens.
```

```{figure} ./images/loss_gdn_hybrid_300M.png
:align: center
:alt: 75% hybrid GDN 300M training loss over 10B tokens
Figure 5: 75% Hybrid GDN 300M over 10B tokens, which achieved the lowest final loss of the four 300M
runs.
```

```{figure} ./images/loss_mamba_hybrid_300M.png
:align: center
:alt: 75% hybrid Mamba2 300M training loss over 10B tokens
Figure 6: 75% Hybrid Mamba2 300M over 10B tokens.
```
````

All five curves descend smoothly, with no loss spikes, no divergence and no recovery events. The
largest single-step increase anywhere in the 1B run is 0.03.

### Downstream Evaluation

The table below reports zero-shot accuracy from
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)[^10]. The purely linear models
are evaluated after conversion to Hugging Face format using the converters in `tools/hybrid/`, and the
hybrid models are evaluated directly from the Megatron checkpoint.

| Task          | Metric   | GDN 1B (100B) | GDN 300M | KDA 300M | Hybrid GDN 300M | Hybrid Mamba2 300M |
| ------------- | -------- | ------------- | -------- | -------- | --------------- | ------------------ |
| arc_easy      | acc      | **0.6869**    | 0.4764   | 0.4823   | 0.4684          | 0.4769             |
| arc_easy      | acc_norm | **0.6183**    | 0.4285   | 0.4335   | 0.4263          | 0.4293             |
| arc_challenge | acc      | **0.3396**    | 0.1937   | 0.1980   | 0.1817          | 0.1937             |
| arc_challenge | acc_norm | **0.3618**    | 0.2287   | 0.2406   | 0.2287          | 0.2321             |
| hellaswag     | acc      | **0.4091**    | 0.2750   | 0.2770   | 0.2752          | 0.2749             |
| hellaswag     | acc_norm | **0.5212**    | 0.2826   | 0.2876   | 0.2875          | 0.2864             |
| openbookqa    | acc      | **0.2640**    | 0.1580   | 0.1660   | 0.1720          | 0.1600             |
| openbookqa    | acc_norm | **0.3900**    | 0.3000   | 0.2940   | 0.2920          | 0.3120             |
| piqa          | acc      | **0.7198**    | 0.6066   | 0.6077   | 0.6121          | 0.6192             |
| piqa          | acc_norm | **0.7220**    | 0.6050   | 0.6121   | 0.5958          | 0.6072             |
| winogrande    | acc      | **0.5462**    | 0.4807   | 0.5028   | 0.5209          | 0.4909             |
| race          | acc      | **0.3311**    | 0.2498   | 0.2545   | 0.2660          | 0.2651             |

Table 5: Zero-shot accuracy for all five models.

These results summarize the practical conclusion of the study. At a fixed budget of 10B tokens, the
choice of mixer has little effect on downstream accuracy while driving as much as a 1.75x spread in
throughput, which makes it primarily a performance decision rather than a quality one. What does
improve quality is the token budget: the 1B run at 100B tokens exceeds the 300M models by roughly ten
to twenty points on most tasks. The strategy this suggests is to select the mixer that trains most
efficiently on the available hardware, then reinvest the savings in additional tokens.

## What We Learned

The first lesson is that, in Primus, a hybrid layer stack is simply a configuration string.
Attention, linear-recurrent and MLP sublayers are interleaved through a single
`hybrid_override_pattern` field, so trying a new architecture means editing one line rather than
writing new model code.

That flexibility matters because, at a fixed token budget, the choice of mixer turned out to be a
performance decision rather than a quality one. The four 300M models finished within 0.03 loss of one
another and within two points on nearly every downstream task, yet differed by as much as 1.75x in
throughput. What did improve quality was the token budget. The 1B run at 100B tokens exceeds the 300M
models by roughly ten to twenty points on most tasks, considerably more than any architectural choice
made at 10B tokens.

Adding a small amount of attention also proved inexpensive and worthwhile. Placing three MLA blocks
among twelve mixers reduced throughput by less than 3%, and that model produced the lowest loss of
the 300M runs.

The recipe held up at scale as well. The 1B run descended from a loss of 12.17 to 2.487 across 95,368
iterations, with a largest single-step increase of 0.03, no NaN values and no skipped iterations.

Finally, the environment is part of the recipe. The allocator setting and the MLA wrapper variable
each affected throughput or capacity more than most hyperparameters, and neither appears in a
configuration file. When you reproduce these runs, treat those settings with the same care as the
YAML itself.

## Summary

In this blog you explored how Primus pre-trains hybrid language models, understood as arbitrary
interleavings of MLA attention, linear-recurrent mixers, and MLP blocks, from a single configuration
string while retaining the fused kernels that make each sublayer type efficient. You walked through
the recipe, the environment settings that silently shape results, and five completed runs on one node
of eight AMD Instinct MI355X GPUs. Across those runs, the choice of mixer changed throughput by as much
as 1.75x while having little effect on loss or downstream accuracy. A 75% hybrid introduced three MLA
blocks for less than 3% in throughput and achieved the lowest loss of the group, and a 1B model
completed 100B tokens with no NaN values and no skipped iterations. The configurations behind every
reported number are included in the repository.

Now train a hybrid of your own. Start from one of the
[released configurations](https://github.com/AMD-AGI/Primus/tree/main/examples/megatron/configs), swap
the mixer with a single-line edit, and compare your throughput and loss against the numbers in this
post. For the upcycling counterpart to this work, which converts existing Transformer checkpoints
into the same hybrid architectures, see the companion blog
[Zebra-HyLo: Upcycling Transformers into Long-Context Hybrid LLMs on AMD Instinct™ GPUs](https://rocm.blogs.amd.com/artificial-intelligence/hylo-long-context/README.html).
Our team is continuing this line of work: researching efficient model architectures, advancing
hybrid-model and upcycling research, and making Primus even faster on AMD Instinct GPUs. Watch for
follow-up posts on [ROCm Blogs](https://rocm.blogs.amd.com/).

## Additional Resources

- **Code:** [Primus](https://github.com/AMD-AGI/Primus), with hybrid configurations in [`examples/megatron/configs/`](https://github.com/AMD-AGI/Primus/tree/main/examples/megatron/configs)
- **Hybrid models:** [AMD-Hybrid-Models](https://github.com/AMD-AGI/AMD-Hybrid-Models), AMD's collection of hybrid model recipes and code, including Zebra-Llama and HyLo
- **Operators:** [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo)
- **Linear-attention kernels:** [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- **Base image:** [`rocm/primus:v26.7`](https://hub.docker.com/r/rocm/primus) on Docker Hub
- **Documentation:** [Training a model with Primus and Megatron-LM](https://rocm.docs.amd.com/projects/primus/en/latest/02-user-guide/megatron-lm-training.html) and [Pretraining workflows](https://rocm.docs.amd.com/projects/primus/en/latest/02-user-guide/pretraining.html)
- **Related work on upcycling into the same architectures:** [HyLo](https://arxiv.org/abs/2604.24715) and [Zebra-Llama](https://arxiv.org/abs/2505.17272)
- **Companion blog:** [Zebra-HyLo: Upcycling Transformers into Long-Context Hybrid LLMs on AMD Instinct™ GPUs](https://rocm.blogs.amd.com/artificial-intelligence/hylo-long-context/README.html), which covers the upcycling side of the same hybrid architectures

[^1]: Kimi Team. "Kimi Linear: An expressive, efficient attention architecture." [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) (2025).

[^2]: Shoeybi, Mohammad, et al. "Megatron-LM: Training multi-billion parameter language models using model parallelism." [arXiv:1909.08053](https://arxiv.org/abs/1909.08053) (2019).

[^3]: Liang, Tianyu, et al. "TorchTitan: One-stop PyTorch Native Solution for Production Ready LLM Pre-training." [arXiv:2410.06511](https://arxiv.org/pdf/2410.06511) (2024).

[^4]: Penedo, Guilherme, et al. "The FineWeb datasets: Decanting the web for the finest text data at scale." [arXiv:2406.17557](https://arxiv.org/abs/2406.17557) (2024).

[^5]: Liu, Aixin, et al. "DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model." [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) (2024).

[^6]: Dao, Tri, and Albert Gu. "Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality." [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) (2024).

[^7]: Yang, Songlin, Jan Kautz, and Ali Hatamizadeh. "Gated delta networks: Improving Mamba2 with delta rule." [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) (2024).

[^8]: Yang, Mingyu, Mehdi Rezagholizadeh, Guihong Li, Vikram Appia, and Emad Barsoum. "Zebra-Llama: Towards extremely efficient hybrid models." [arXiv:2505.17272](https://arxiv.org/abs/2505.17272) (2025).

[^9]: Ashrafi Fashi, Parsa, Utkarsh Saxena, Mehdi Rezagholizadeh, Aref Jafari, Akash Haridas, Mingyu Yang, Vansh Bhatia, Guihong Li, Vikram Appia, and Emad Barsoum. "Long-context aware upcycling: A new frontier for hybrid LLM scaling." [arXiv:2604.24715](https://arxiv.org/abs/2604.24715) (2026).

[^10]: Gao, Leo, et al. "A framework for few-shot language model evaluation." (2023).

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
