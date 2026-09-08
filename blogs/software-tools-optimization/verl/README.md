---
blogpost: true
blog_title: "veRL on AMD: Production-Ready RL Post-Training on ROCm"
date: 08 Sep 2026
author: 'Fuwei Yang, Zhaodong Bing, Mingjie Lu, Xiaohong Kou, Yuhan Yang, Wei Cai, Liz Li, Yuankai Chen, Yao Fu, Dong Li, Zhenyu Gu'
thumbnail: 'verl-rocm-thumbnail.png'
tags: Reinforcement Learning, AI/ML, Fine-Tuning, LLM
category: Software tools & optimizations
target_audience: ML engineers, AI researchers, RL post-training practitioners
key_value_propositions: Stand up veRL on AMD Instinct GPUs with a turnkey ROCm container, AITER-accelerated vLLM and SGLang rollout, and accuracy validated on MI300 and MI355.
language: English
myst:
    html_meta:
        "author": "Fuwei Yang, Zhaodong Bing, Mingjie Lu, Xiaohong Kou, Yuhan Yang, Wei Cai, Liz Li, Yuankai Chen, Yao Fu, Dong Li, Zhenyu Gu"
        "description lang=en": "Run veRL RL post-training on AMD Instinct GPUs with a turnkey ROCm container, AITER-accelerated rollout, and validated accuracy on MI300 and MI355."
        "keywords": "veRL, reinforcement learning, ROCm, AMD Instinct, GRPO, DAPO, vLLM, SGLang, AITER"
        "vertical": "AI, Systems"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Fuwei Yang, Zhaodong Bing, Mingjie Lu, Xiaohong Kou, Yuhan Yang, Wei Cai, Liz Li, Yuankai Chen, Yao Fu, Dong Li, Zhenyu Gu"
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

# veRL on AMD: Production-Ready RL Post-Training on ROCm

Reinforcement learning post-training on AMD Instinct GPUs is here — with a turnkey container, AITER-accelerated vLLM and SGLang rollout, and accuracy validated on both MI300 and MI355.

[veRL](https://github.com/verl-project/verl) — the open-source implementation of the RL controller — has quickly become one of the most widely adopted RL post-training frameworks, powering PPO, GRPO, and DAPO pipelines with a flexible mix of rollout and training engines. Today, that same power runs on **AMD Instinct** GPUs. Whether you're on **MI300** or the latest **MI355**, you can stand up veRL on ROCm in minutes, scale it across your rollout and training engines, and trust the numbers.

**In the blog we will cover:**

- The **turnkey veRL ROCm container** and running it for GRPO, DAPO, or PPO post-training without assembling dependencies by hand
- **vLLM and SGLang** as two AITER-accelerated rollout options
- **Training accuracy validation** on Instinct GPUs — with case studies on MI300 and MI355 for both **colocated** and **fully-async** execution modes

## Zero-Friction Setup: One Container, No Dependency Headaches

Getting a modern RL stack running usually means fighting a maze of dependencies — matching CUDA/ROCm versions, building attention kernels from source, and praying the wheels line up. On AMD, that pain is gone.

veRL ships a **turnkey ROCm container** [verlai/verl:rocm7.14_torch2.12_release_0724](https://hub.docker.com/r/verlai/verl) based on [docker/rocm/Dockerfile.rocm](https://github.com/verl-project/verl/tree/main/docker/rocm) so you can go from zero to training without hand-assembling a single dependency. The image bundles the entire runtime stack — **ROCm, PyTorch, Triton, vLLM, SGLang, AITER, TransformerEngine and Megatron-core** — all pinned to a set of versions verified to work together. Pull it, mount your data, and run — RL post-training on AMD is a `docker run` away, so you can spend your time on reward design and experiments, not on environment setup.

- **No dependency installation.** Everything RL post-training needs is baked in — no extra source builds, no extra version detective work. There's an [end-to-end AMD tutorial](https://github.com/verl-project/verl/tree/main/docs/amd_tutorial) covering build, run, and example PPO/GRPO commands.
- **Built for multiple architectures.** The image targets `gfx942` **(MI300 series — MI300X / MI300A / MI308 / MI325X)** and `gfx950` **(MI350 series — MI350X / MI355X)** out of the box.
- **Customizable for your setup.** Teams that want to tailor the image can do so — target a specific GPU architecture, pin component versions, and tune the build to their environment — all documented in a dedicated [ROCm README](https://github.com/verl-project/verl/blob/main/docker/rocm/README.md).

This is all backed by CI on real AMD hardware: end-to-end tests now run on a community-hosted runner, so changes to the veRL repo are validated on AMD hardware before they are merged. Coverage is a work in progress and actively expanding toward parity with the other platforms.

## Two Rollout Backends — vLLM and SGLang, Both AITER-Accelerated

Rollout is the key component of RL post-training, and on AMD you're not locked into a single engine: veRL supports **both vLLM and SGLang** on ROCm. Both are accelerated by AMD's **AITER** kernel library — powering attention, RMSNorm / RoPE, MoE, and quantization — with everything fully overridable if you want to experiment.

- **vLLM** — the default, battle-tested rollout engine; enable the AITER path with a few environment variables.
- **SGLang** — a first-class alternative that runs on ROCm with AITER kernels enabled by default.

Both engines work across **both** of veRL's integration modes — **colocated** (rollout and training share GPUs with fast weight transfer) and **fully async** (rollout and training run concurrently for maximum GPU utilization, with live parameter sync back to the rollout workers). Whichever engine and mode fit your workload, AMD-accelerated rollout is ready to go.

## Validated Accuracy — MI300 and MI355, Colocated and Fully-Async

Enablement is only half the story. What ultimately matters is whether the model you train on AMD is as good as the model you'd train anywhere else. It is — and we've measured it.

**Training accuracy has been validated on both AMD MI300 and MI355**, across **both colocated and fully-async** modes. RL post-training on Instinct GPUs converges to the quality you expect, so you can move real workloads onto AMD with confidence rather than crossing your fingers.

The MI350-series runs below were measured on MI350. MI350X and MI355X are the same gfx950 silicon and differ in board power and cooling rather than in numerics, so the accuracy results carry over between them unchanged — only throughput tracks the power envelope.

### Case study: Qwen3-8B GRPO with SGLang rollout on MI350

Start with the **SGLang rollout backend** — the first-class alternative to vLLM introduced in the previous section — to show it delivers the same accuracy on AMD.

Key run parameters:

| Parameter | Value |
| --- | --- |
| Algorithm | GRPO (`use_kl_loss = True`, `kl_loss_coef = 0.001`) |
| Training backend | FSDP (PyTorch Fully Sharded Data Parallel) |
| Rollout engine | SGLang, async mode, AITER-accelerated, group size `n = 5` |
| Task | GSM8K + MATH (train and validate on both) |
| Max prompt / response length | 1,024 / 2,048 tokens |
| Train batch / PPO mini-batch | 1,024 / 256 |
| Learning rate | 1e-6 |
| Execution mode | Colocated, single node — 8 GPUs |

The training curves show textbook convergence:

```{figure} ./images/gsm8k-sglang-grpo.png
:align: center
:alt: Qwen3-8B GRPO with SGLang rollout on AMD
Qwen3-8B GRPO with SGLang rollout on AMD
```

**1. Strong accuracy on both benchmarks.** GSM8K validation accuracy climbs from **0.82 to ~0.955** (peak 0.956) within the first ~30 steps and holds there, while the harder MATH benchmark rises steadily from **0.37 to ~0.835** (peak 0.839). Mean training reward tracks them in lockstep, from **0.62 to ~0.90** — no reward-hacking, no collapse. An 8B model reaching ~95% on GSM8K and ~84% on MATH is exactly the quality bar you'd expect from a well-behaved GRPO run.

**2. SGLang and the FSDP trainer stay numerically locked.** The rollout↔actor mismatch is negligible: `actor/ppo_kl` hovers around **1e-4** during the entire run. In other words, the tokens that the AITER-accelerated SGLang samples during rollout are assigned essentially identical probabilities when the FSDP actor recomputes them.

### Case study: Qwen2.5-Math-7B DAPO, fully-async on MI300

We reproduced veRL's fully-async experiments on AMD MI300: **DAPO post-training of Qwen2.5-Math-7B on the MATH task**. Key parameters are defined as follows:

| Parameter | Value |
| --- | --- |
| Rollout engine | vLLM, AITER enabled, group size `n = 16` |
| Training backend | FSDP2 |
| Max prompt / response length | 2,048 / 28,672 (28k) tokens |
| PPO mini-batch size | 32 |
| Learning rate | 1e-6 |
| Execution mode | Fully-async — 16 rollout GPUs + 16 training GPUs (32 total) |
| `staleness_threshold` | 0.5 |
| `trigger_parameter_sync_step` | 4 |
| `partial_rollout` | True |

The training curves tell the story:

```{figure} ./images/dapo-fully-async-mi300.png
:align: center
:alt: Qwen2.5-Math-7B DAPO fully-async on AMD MI300
Qwen2.5-Math-7B DAPO, fully-async on AMD MI300
```

**1. Convergence is clean, and the final score matches the reference.** Validation accuracy climbs from **0.13 to a peak of 0.34** and holds around **0.32**, while mean training reward rises from **−0.98 to +0.17** in lockstep — no reward-hacking, no collapse. Critically, these numbers land right on top of veRL's published reference for the same configuration (`staleness_threshold = 0.5`, partial rollout): the upstream [fully-async-policy results](https://github.com/verl-project/verl/tree/main/verl/experimental/fully_async_policy) report max accuracy of **0.33** in a 4-node training experiment, while on AMD we can reproduce the result with the same training settings and get **0.34** in our measurement.

**2. The KL curve confirms stable off-policy training.** Rollout↔actor KL (`actor/ppo_kl`) sits at **~0.0007 for the first ~100 logged steps**, then rises and settles onto a controlled **0.15–0.25 plateau** once staleness kicks in. The run is stable end to end, with no explosive divergence despite training on deliberately stale samples.

**3. Fully-async is dramatically faster per training step.** Against a colocated DAPO run of the same model on the same 32-GPU footprint, we compare wall-clock time per training step over the first 64 steps:

| | Fully-async | Colocated |
| --- | --- | --- |
| Step time (median) | ~219 s | ~394 s |
| Step time (mean) | ~249 s | ~400 s |
| Speed-up per step | ~1.8× | 1.0× (baseline) |

A fully-async training step is **~1.8× faster** than the colocated baseline.

In this experiment, we verified on the AMD platform that fully-async mode achieves the expected accuracy and performance speedup originally reported on H20.

One honest caveat visible in the step-time plot: per-step time is flat at **~190–200 s for the first ~260 steps**, then climbs (into the 600–1,200 s range) toward the end of training. This isn't a regression or instability — it's the expected **response-length growth** of reasoning RL: as the model capacity saturates, the model only learns to increase the response length without improving the accuracy. We observe that its average response length roughly doubles (from ~890 to ~1,850 tokens) during the late stage of the experiment. The pattern can also be observed in [the reported H20 experiment data](https://wandb.ai/hou-zg-meituan/fully-async-policy-colocate_async?nw=nwuserhouzg).

### Case study: Qwen3.5-35B-A3B MoE on MI350

Now we present a more complex case — we ran **GRPO post-training of Qwen3.5-35B-A3B** on the **geometry3k** reasoning task on **MI350**, with the **Megatron** backend — and we validated it in both colocated and fully-async modes:

| Parameter | Colocated | Fully-async |
| --- | --- | --- |
| Rollout / training placement | Shared 8 GPUs | 4 GPUs for rollout and 4 GPUs for training |
| Megatron parallelism | TP 2, expert-parallel 8 | TP 2, expert-parallel 4 |
| Async knobs | N/A | `staleness_threshold=0.5`, sync every 4 steps |
| Rollout engine | vLLM + AITER, group size `n = 5` | vLLM + AITER, group size `n = 5` |
| Max response length | 2,048 tokens | 2,048 tokens |

Both modes deliver clean, stable convergence:

```{figure} ./images/moe-mi350-grpo.png
:align: center
:alt: Qwen3.5-35B-A3B GRPO on geometry3k, AMD MI350
Qwen3.5-35B-A3B GRPO on geometry3k, AMD MI350 — colocated vs fully-async
```

- **Validation accuracy climbs steadily.** Both modes rise from a ~0.37–0.48 start to a **peak of ~0.77–0.78** within the first ~75 steps and hold there.
- **Training reward rises in lockstep**, from ~0.34 to ~0.82 in both modes, tracking the validation gains rather than reward-hacking.
- **Rollout↔actor KL stays tiny** — **≈0.001 for fully-async and ≈0 for colocated** — meaning the AITER-accelerated vLLM rollout and the Megatron training policy stay numerically almost identical.
- **Fully-async sustains ~310 tok/s/GPU versus colocated's ~216** — roughly **1.4× higher per-GPU throughput**. That's exactly the payoff of decoupling.

In this section we validated that a 35B-parameter MoE model trained end-to-end with GRPO on AMD MI350 converges smoothly to strong accuracy in both colocated and fully-async modes.

## Summary

In this blog we covered production-ready RL post-training with veRL on AMD Instinct GPUs:

1. **The turnkey veRL ROCm container** — how to pull and run it for GRPO, DAPO, or PPO post-training without assembling dependencies by hand, with the full stack (ROCm, PyTorch, vLLM, SGLang, AITER, Megatron-core) baked in and CI validation on real MI300 hardware.
2. **vLLM and SGLang as two AITER-accelerated rollout options**, each available in colocated and fully-async execution modes.
3. **Training accuracy validation on Instinct GPUs** — case studies on MI300 and MI355 showing convergence across various RL workloads.

Looking ahead, AMD will continue working closely with the veRL community to support emerging models and capabilities, including architectures such as DeepSeek-V4, agentic RL workloads, and integration with platforms such as VeOmni. We will also improve rollout efficiency through technologies such as speculative decoding and FP4 inference.

Our goal is simple: make the latest veRL models and features work out of the box on AMD Instinct GPUs, with strong performance, validated accuracy, and a reliable experience for customers.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

AMD, the AMD Arrow logo, AMD Instinct, ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
