---
blogpost: true
blog_title: "Verl 0.9.0 on ROCm™ 10.0: Next-Generation RL Post-Training on AMD Instinct™ GPUs"
date: "28 Sep 2026"
author: "Tiffany Mintz, Yao Liu, Phani Vaddadi, Vish Vadlamani"
thumbnail: 'verl_0.9.0_thumbnail.jpg'
tags: "AI/ML, Reinforcement Learning"
category: "Applications & models"
target_audience: "AI Developers, Engineers, Hobbyists"
key_value_propositions: "Enable Verl 0.9.0 on ROCm™ 10.0 with a production Docker recipe, native PlatformROCm support, vLLM and SGLang rollout in one image, and the same fully async GRPO/DAPO workflows previously demonstrated on MI355X."
language: English
myst:
    html_meta:
        "author": "Tiffany Mintz, Yao Liu, Phani Vaddadi, Vish Vadlamani"
        "description lang=en": "Enable Verl 0.9.0 on ROCm™ 10. Native AMD platform support, vLLM 0.27 and SGLang in one image, and updated async GRPO/DAPO recipes on AMD Instinct™ GPUs."
        "keywords": "Verl, ROCm™ 10, 0.9.0, RLHF, Reinforcement Learning, AMD Instinct™, vLLM, SGLang, Fully Async, GRPO, DAPO, MI300X, MI325X, MI355X"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Deploying AI at Scale"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Tiffany Mintz, Yao Liu, Phani Vaddadi, Vish Vadlamani"
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

# Verl 0.9.0 on ROCm™ 10.0: Next-Generation RL Post-Training on AMD Instinct™ GPUs

In [Scaling RL with verl on AMD Instinct™ MI355X](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html), we walked through verl’s Fully Async Policy trainer on AMD Instinct™ MI355X GPUs. That post used **verl 0.7.1.amd0** on **ROCm™ 7.0.2**, with a 4+4 GPU split for GRPO on Qwen2.5-VL-7B (Megatron) and DAPO on Qwen2.5-Math-7B (FSDP2), plus a separate synchronous throughput comparison against NVIDIA B300.

This follow-up is about the stack that comes next. AMD has enabled **verl 0.9.0** on **ROCm™ 10.0**, published as the [`release/0.9.0.amd0`](https://github.com/AMD-Ecosystem/verl/tree/release/0.9.0.amd0) branch of [AMD-Ecosystem/verl](https://github.com/AMD-Ecosystem/verl). The same async recipes still run, but the runtime is now a first-class ROCm™ platform: native `PlatformROCm` detection, a dedicated `Dockerfile.rocm` that ships **vLLM 0.27.0 and SGLang** in one image, Megatron-Core 0.18, and the upstream 0.9.0 trainer/rollout features that landed after 0.7.1.

If you already followed the MI355X walkthrough, you can keep the GRPO and DAPO scripts. What changes is the container, the host ROCm™ driver, and a set of AMD-specific bring-up fixes that make 0.9.0 production-ready on AMD Instinct™ GPUs.

## What Changed Since the Previous Blog

| Item | Previous blog (Aug 2026) | This release |
| --- | --- | --- |
| verl | 0.7.1.amd0 | **0.9.0.amd0** ([`release/0.9.0.amd0`](https://github.com/AMD-Ecosystem/verl/tree/release/0.9.0.amd0)) |
| ROCm™ | 7.0.2 | **10.0.0** |
| Docker | `rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2` | Build from [`docker/rocm/Dockerfile.rocm`](https://github.com/AMD-Ecosystem/verl/blob/release/0.9.0.amd0/docker/rocm/Dockerfile.rocm) on `rocm/primus:v26.7` |
| Hardware | Validated on 8× MI355X (`gfx950`) | Targets **MI300 series (`gfx942`)** and **MI350 series (`gfx950`)** |
| PyTorch | PyTorch 2.9.1, Megatron-Core 0.16.0 | Primus v26.7 training stack with **PyTorch 2.12.0** + **Megatron-Core 0.18.0** |
| Rollout | vLLM 0.20.2 | **vLLM 0.27.0** and **SGLang** from source in the same image |
| Orchestration | Ray in the 0.7.1 image | **Ray 2.58.0**, with `amdsmi` wired so Ray sees AMD Instinct™ GPUs |
| Trainer | Experimental fully async policy | Upstream **V1 trainer is the default**, plus the same fully async GRPO/DAPO scripts |
| Platform layer | ROCm™ patches around CUDA-shaped APIs | First-class **`PlatformROCm`** backend |

The [previous post](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html) explained *when* to use sync versus async and how to run those two MI355X examples. This post is the enablement note: how 0.9.0.amd0 is built for ROCm™ 10.0, what upstream 0.9.0 brings to AMD GPUs, and how to bring the container up.

## Why 0.9.0 Matters on AMD Instinct™ GPUs

Upstream [verl v0.9.0](https://github.com/verl-project/verl/releases/tag/v0.9.0) (14 Aug 2026) is a large drop: a unified V1 trainer, Megatron-Bridge as the default Megatron path, vLLM ≥ 0.18 only, delta-sharded weight sync, and a hardware plugin layer. The AMD fork takes that drop and makes it runnable on ROCm™ 10.0 instead of leaving teams on the 0.7.1 / ROCm™ 7.0.2 image from the last blog.

Three themes matter most for ROCm™ users.

### 1. ROCm™ is a First-Class Platform, Not a CUDA Lookalike

0.9.0 introduces `verl.plugin.platform`: device APIs, Ray resource env vars, and rollout engine defaults go through a registry instead of scattered `torch.cuda` calls. On AMD, that backend is [`PlatformROCm`](https://github.com/AMD-Ecosystem/verl/blob/release/0.9.0.amd0/verl/plugin/platform/platform_rocm.py).

It subclasses the CUDA platform (PyTorch on ROCm™ still exposes `torch.cuda` via hipify) and overrides only what differs:

- Availability checks `torch.version.hip` and can fall back to `rocm-smi` inside CPU-only Ray actors.
- Rollout workers can set `SGLANG_USE_AITER` so SGLang non-attention kernels (RMSNorm, RoPE, MoE, quantization) go through AITER. `PlatformROCm` still defaults that flag on; **`Dockerfile.rocm` overrides it to `SGLANG_USE_AITER=0`** and uses a patched non-AITER `fused_add_rms_norm` path that is stable on this Primus stack. Attention stays on Triton (`SGLANG_ATTENTION_BACKEND=triton`). Set `SGLANG_USE_AITER=1` at runtime if you want the AITER kernels instead.
- Ray is told not to overwrite `HIP_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES` — the same class of issue we originally fixed in Ray for AMD Instinct™, now expressed in the platform layer rather than as a one-off patch.

You can force the backend with `VERL_PLATFORM=amd`, but auto-detection is the intended path.

SGLang is no longer “in progress” on the AMD quick-start matrix, and it is no longer a separate install. Upstream 0.9.0 landed ROCm™ SGLang support through platform defaults and Ray init env vars. [`Dockerfile.rocm`](https://github.com/AMD-Ecosystem/verl/blob/release/0.9.0.amd0/docker/rocm/Dockerfile.rocm) now **builds SGLang from source next to vLLM**, so colocated and fully async runs pick the engine with `actor_rollout_ref.rollout.name=vllm` or `sglang`.

### 2. The Trainer and Rollout Stack Caught Up to Production RL

From upstream 0.9.0, now in the AMD branch:

- **V1 PPO trainer is default.** `sync`, `colocate_async`, and `separate_async` share one control flow, replay buffer, and metric surface. The experimental Fully Async Policy scripts from the previous blog still exist under `verl/experimental/fully_async_policy/`.
- **Megatron-Bridge is the Megatron default** (vanilla mBridge is deprecated upstream). The AMD geo3k Megatron recipe still has a small  ROCm™-side adjustment (vanilla mBridge, dynamic batch size off, micro-batch raised to 2) so the MI355X 4+4 walkthrough keeps running.
- **vLLM older than 0.18 is dropped.** The ROCm™ 10.0 image builds **vLLM 0.27.0**, which is past that cutoff and includes the sleep/wake and MoE fixes needed for hybrid RL. The same Dockerfile also builds **SGLang** (pinned commit, ROCm™ `sgl-kernel`, `--no-deps` so it does not replace Primus PyTorch or vLLM).
- **`delta_sharded` checkpoint engine** diffs each actor shard against a pinned CPU snapshot and ships only changed pairs to rollout. That is the path for cheaper weight sync in disaggregated async training.
- **DeepSeek-V4-Flash GRPO on AMD.** Upstream enabled this on ROCm™, with a pure-PyTorch `fast_hadamard_transform` fallback for DSA so the kernel path does not depend on a CUDA-only library.
- **MI300 e2e PPO CI** is in the tree, so the AMD platform is covered by an AMD Instinct™ regression workflow rather than “works on our cluster.”

### 3. A ROCm™ 10.0 Image That Ray, vLLM, SGLang, and Megatron Can Actually Boot

The previous image was a full `rocm/verl:…0.7.1…` tag. 0.9.0.amd0 adds [`docker/rocm/Dockerfile.rocm`](https://github.com/AMD-Ecosystem/verl/blob/release/0.9.0.amd0/docker/rocm/Dockerfile.rocm), which starts from **`rocm/primus:v26.7`** (training stack: PyTorch, TransformerEngine, AITER) and layers the RL pieces Primus does not ship.

The Dockerfile is doing real enablement work, not just `pip install verl`:

| Concern | What the 10.0 Recipe Does |
| --- | --- |
| Ray sees 0 GPUs | Primus’s TheRock SDK ships `amd_smi` but not the `amdsmi` Python package. Ray’s AMD accelerator manager then reports **0 GPUs** even when `torch` and `rocm-smi` are healthy. The image installs `amdsmi` from the bundled SDK and switches `rocm_sdk` library preload from `RTLD_GLOBAL` to `RTLD_LOCAL` so `torch.cuda.device_count()` does not collapse to 0. |
| No rollout engine | Primus is training-only. The image builds **vLLM 0.27.0** from source with `VLLM_TARGET_DEVICE=rocm` **and SGLang from source** in the same image. verl selects the engine at runtime (`rollout.name=vllm` or `sglang`). |
| SGLang on ROCm™ | Pins an SGLang commit validated on Primus + vLLM ROCm™, applies ROCm™ layernorm / Qwen3-ASR registration patches, builds `sgl-kernel` for `gfx942` and `gfx950`, and sets image defaults `SGLANG_USE_AITER=0` and `SGLANG_ATTENTION_BACKEND=triton`. Trainer `rollout.yaml` gets `disable_custom_all_reduce: True` for both vLLM and SGLang, plus `attention_backend: triton` for SGLang. |
| AMD Instinct™ kernels | `PYTORCH_ROCM_ARCH` / `AMDGPU_TARGETS` are `gfx942;gfx950`. vLLM AITER flags (`VLLM_ROCM_USE_AITER`, FP8 padding) are image defaults. |
| Qwen3.5 / GDN | Installs flash-linear-attention, megatron-core 0.18.0, mbridge, transformers ≥ 5.12, and a `libz3.so.4.15` symlink so tilelang/TVM can `dlopen` (required for Gated Delta Net). |
| Checkpoint engine | Installs `cupy-rocm-7-0` so the NCCL/RCCL checkpoint backend registers. |
| Reproducible verl | Clones **`https://github.com/AMD-Ecosystem/verl.git`** at **`release/0.9.0.amd0`**. |
| Ray | Pins **`ray[default,serve]==2.58.0`**. |

RCCL-related env defaults are set in the image (`NCCL_MIN_NCHANNELS=112`, `HSA_NO_SCRATCH_RECLAIM=1`, `HIP_FORCE_DEV_KERNARG=1`) and can be overridden for multi-node.

## Software Baseline

Below is the 0.9.0.amd0 / ROCm™ 10.0 bring-up matrix:

| Component | Version / PIN |
| --- | --- |
| Host driver | ROCm™ **10.0.0** |
| Base image | `rocm/primus:v26.7` |
| verl | AMD-Ecosystem `release/0.9.0.amd0` |
| Python | 3.12 |
| GPU arch | `gfx942` (MI300X / MI300A / MI325X), `gfx950` (MI350X / MI355X) |
| vLLM | 0.27.0 (source) |
| SGLang | Built from source (pinned commit in `Dockerfile.rocm`; coexists with vLLM) |
| Ray | 2.58.0 |
| Megatron-Core | 0.18.0 |
| CuPy | `cupy-rocm-7-0` |
| Rollout env | vLLM: `VLLM_ROCM_USE_AITER=1`, `VLLM_USE_V1=1` (in example scripts). SGLang: `SGLANG_USE_AITER=0`, `SGLANG_ATTENTION_BACKEND=triton` (image defaults) |

Feature support on this branch:

| Category | Status |
| --- | --- |
| Runtime mode | Fully async, colocate (V1 `sync` / `colocate_async` / `separate_async`) |
| Inference engine | vLLM **and** SGLang (both installed in `Dockerfile.rocm`) |
| Trainer backend | FSDP, FSDP2, Megatron |
| Hardware | MI300 series (`gfx942`), MI350 series (`gfx950`) |

## Build and Launch the ROCm™ 10.0 Container

Host prerequisites:

1. ROCm™ 10.0 host driver stack installed and healthy.
2. Docker can access `/dev/kfd` and `/dev/dri`.
3. Dataset and model storage paths ready (Hugging Face token if you will download models).

Clone the AMD release branch and build. BuildKit is required.

```bash
git clone --recursive -b release/0.9.0.amd0 https://github.com/AMD-Ecosystem/verl.git
cd verl

DOCKER_BUILDKIT=1 docker build \
  -f docker/rocm/Dockerfile.rocm \
  --build-arg GPU_ARCH="gfx942;gfx950" \
  --build-arg VLLM_TAG=v0.27.0 \
  --build-arg VERL_BRANCH=release/0.9.0.amd0 \
  -t verl-rocm:0.9.0.amd0-rocm \
  .
```

The same Dockerfile clones and installs SGLang after vLLM (pinned `SGLANG_TAG`; override with `--build-arg SGLANG_TAG=…` if you re-validate a newer commit). To cut compile time on a single architecture, set `--build-arg GPU_ARCH=gfx950` (MI355X) or `gfx942` (MI300X). Lower `--build-arg MAX_JOBS=64` if the vLLM or `sgl-kernel` build is memory-bound.

Launch (same device, shm, and ulimit pattern as the previous blog):

```bash
NAME=verl_0_9_0_amd0
DOCKER=verl-rocm:0.9.0.amd0-rocm

docker run -it --name $NAME \
  --device /dev/kfd --device /dev/dri \
  --privileged --network=host \
  --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size=2048g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -w /workspace \
  $DOCKER \
  /bin/bash
```

Inside the container:

```bash
python - <<'PY'
import torch, verl
print("verl :", verl.__file__)
print("torch:", torch.__version__)
print("hip  :", torch.version.hip)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu_count:", torch.cuda.device_count())
    print("device_0:", torch.cuda.get_device_name(0))
try:
    import sglang
    print("sglang:", sglang.__version__)
except Exception as e:
    print("sglang import failed:", e)
try:
    import vllm
    print("vllm :", vllm.__version__)
except Exception as e:
    print("vllm import failed:", e)
PY

rocminfo | grep -E "gfx942|gfx950" || true
rocm-smi --showproductname
```

`gpu_count` must match the GPUs you expect. If it prints `0` while `rocm-smi` lists cards, Ray will fail later with “Total available GPUs 0 is less than total desired GPUs …”. That is the `amdsmi` / `RTLD_GLOBAL` issue the  Dockerfile is written to prevent.

## Repeat the Previous Async Walkthroughs on 0.9.0.amd0

The two Fully Async Policy examples from the MI355X post are still the fastest way to validate the new stack. On this branch they live in the AMD fork with data-prep scripts added under `verl/experimental/fully_async_policy/shell/data_model_preparation/`.

The intended 8-GPU split is unchanged: **4 GPUs training + 4 GPUs rollout**. The walkthrough scripts below still use **vLLM**; the same 4+4 split works with **SGLang** after you switch `rollout.name`.

Async knobs are unchanged in meaning:

- `async_training.staleness_threshold` — how stale a sample may be
- `async_training.trigger_parameter_sync_step` — how often weights are pushed to rollout
- `async_training.partial_rollout` — pause/resume in-flight generation across RCCL weight sync instead of dropping it

### GRPO: Qwen2.5-VL-7B on Geometry3k (Megatron)

```bash
export HF_TOKEN=your_token
cd /workspace/verl

bash verl/experimental/fully_async_policy/shell/data_model_preparation/prepare_geo3k_qwen25vl_7b_megatron_4_4.sh

export HF_MODEL_PATH=${HOME}/models/Qwen2.5-VL-7B-Instruct
export WANDB_MODE=offline
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN
export TE_HIPBLASLT_ALGO_SELECTION=1   # if fails, try 2, 3, ...
unset TE_HIPBLASLT_TUNING_RUN_COUNT TE_HIPBLASLT_ALGO_SAVE
mkdir -p /workspace/logs
bash verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh \
  2>&1 | tee /workspace/logs/grpo_$(date +%Y%m%d_%H%M%S).log
```

On 0.9.0.amd0 this Megatron recipe uses vanilla mBridge, turns dynamic batch size off, and uses a micro-batch of 2 — ROCm™-side stability for the same 4+4 geo3k path, not a change in the algorithm.

### DAPO: Qwen2.5-Math-7B (FSDP2)

```bash
pip install --upgrade 'accelerate>=1.14.0'

export HF_TOKEN=your_token
cd /workspace/verl

bash verl/experimental/fully_async_policy/shell/data_model_preparation/prepare_dapo_7b_math_fsdp2_4_4.sh

mkdir -p /workspace/logs
bash verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh \
  2>&1 | tee /workspace/logs/dapo_$(date +%Y%m%d_%H%M%S).log
```

Qwen2.5-Math-7B still needs `max_position_embeddings` raised to 32768 in `config.json` after download; the prepare script does that.

Healthy startup looks the same as before: rollout HTTP servers capture HIP graphs, and then `FullyAsyncTrainer` starts collecting samples from the queue. A `rocm-smi` snapshot during the run should show the training GPUs compute-bound and the rollout GPUs holding KV-cache VRAM.

For the algorithm knobs (clip-higher, KL-free DAPO, overlong buffer, GRPO group size) see the [previous blog](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html). They were not redesigned for 10.0; they run on the new image.

## What Else You Can Run from the AMD Quick-Start Matrix

Beyond the two async demos, 0.9.0.amd0 documents this colocated / fully async matrix (vLLM or SGLang):

| Runtime | Engine | Trainer | Example |
| --- | --- | --- | --- |
| Colocate | vLLM | FSDP | `examples/grpo_trainer/run_qwen3_8b_fsdp.sh` |
| Colocate | vLLM | Megatron | `examples/grpo_trainer/run_qwen3_5_35b_megatron.sh` |
| Fully async | vLLM | FSDP2 | `verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh` |
| Fully async | vLLM | Megatron | `verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh` |

SGLang is already in the 10.0 image. Use the same scripts with the rollout name switched from vLLM to SGLang (`actor_rollout_ref.rollout.name=sglang` or `rollout_name="sglang"` in the launch scripts). Keep `attention_backend=triton` on ROCm™ — the image env and trainer `rollout.yaml` already set that.

The Qwen3.5-35B Megatron GRPO example is a 0.9-era workload the 10.0 Dockerfile is explicitly provisioned for (GDN / flash-linear-attention / transformers 5.x). Treat it as the “new model” path; the VL-7B and Math-7B scripts remain the regression path from the last post.

## Known Issues (Carried Forward, Still Apply)

1. For Qwen2.5-Math-7B, `max_position_embeddings` must be 32768 after download.
2. `PYTORCH_ALLOC_CONF=expandable_segments:True` is used to reduce OOM risk and can conflict with vLLM custom all-reduce. Default configs set `vllm.disable_custom_all_reduce=True` until that ROCm™ conflict is gone.
3. SGLang `attention_backend` must be `triton`.
4. Some vLLM/SGLang ROCm™ fixes are still applied in Dockerfiles rather than only in released wheels. The 10.0 recipe pins vLLM **v0.27.0** instead of `main` because newer trees require a `torch::stable::Tensor` API. SGLang is pinned to a commit validated on this Primus + vLLM ROCm™ stack (and patched in-tree for `fused_add_rms_norm` and idempotent Qwen3-ASR config registration). Re-validate `SGLANG_TAG` if you bump `PRIMUS_TAG` or `VLLM_TAG`.

## Takeaways

- **verl 0.9.0.amd0 is enabled on ROCm™ 10.0** via [`Dockerfile.rocm`](https://github.com/AMD-Ecosystem/verl/blob/release/0.9.0.amd0/docker/rocm/Dockerfile.rocm) and the [`release/0.9.0.amd0`](https://github.com/AMD-Ecosystem/verl/tree/release/0.9.0.amd0) branch. That image now includes **both vLLM 0.27.0 and SGLang**.
- The MI355X fully async GRPO and DAPO walkthroughs are the compatibility target: same 4+4 split, same staleness / sync / partial-rollout knobs, new container and driver.
- **`PlatformROCm`** makes HIP visibility, Ray GPU discovery, and SGLang/vLLM ROCm™ defaults part of verl rather than out-of-tree patches. The 10.0 image then pins SGLang to Triton attention and a patched non-AITER RMSNorm path.
- Upstream 0.9.0 is in the AMD tree: V1 trainer default, Megatron-Bridge, vLLM 0.27, delta-sharded weight sync, DeepSeek-V4-Flash GRPO on AMD, MI300 PPO CI.
- Ray reporting zero GPUs on TheRock-based Primus images is a solved bring-up bug in this Dockerfile (`amdsmi` + `RTLD_LOCAL`). Check `torch.cuda.device_count()` before you debug verl resource pools.

## Summary

In this blog, you explored how verl 0.9.0.amd0 runs on ROCm™ 10.0 and AMD Instinct™ GPUs. You saw what changed since the 0.7.1.amd0 / ROCm™ 7.0.2 stack, and how the new `PlatformROCm` backend makes ROCm™ a first-class verl platform instead of a set of CUDA-shaped workarounds. You built a single `Dockerfile.rocm` image that ships vLLM 0.27.0 and SGLang side by side, confirmed that Ray, PyTorch, and both rollout engines can see your GPUs, and reran the fully async GRPO (Qwen2.5-VL-7B, Megatron) and DAPO (Qwen2.5-Math-7B, FSDP2) walkthroughs on the new stack.

In [Scaling RL with verl on AMD Instinct™ MI355X](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html), we showed that fully async RL on AMD Instinct™ GPUs is practical: separate rollout and training fleets, RCCL weight sync, and high trainer utilization on GRPO and DAPO. This post carries that work forward. You can keep the same 4+4 scripts and async knobs, and pick up upstream 0.9.0 features: the V1 trainer, Megatron-Bridge, delta-sharded weight sync, and new model paths such as Qwen3.5 GDN and DeepSeek-V4-Flash. You also avoid the bring-up problems that used to cost time on new images, such as Ray reporting zero GPUs or a missing rollout engine. Build from `docker/rocm/Dockerfile.rocm`, point it at `release/0.9.0.amd0`, and you have a reproducible starting point for RL post-training on MI300 and MI350 series GPUs.

This is not the last stop. Our team plans to continue optimizing and extending ROCm™ support. Our priority for future posts is to continue to provide performance insights and walkthroughs as new features are explored and existing features are optimized in Verl. Follow the [ROCm™ Blogs](https://rocm.blogs.amd.com/) and watch the [AMD-Ecosystem/verl](https://github.com/AMD-Ecosystem/verl) repository for new releases, and try the recipes on your own AMD Instinct™ GPUs in the meantime.

## Acknowledgements

Thanks to the AMD ROCm™, verl, and Primus engineers who landed `PlatformROCm`, the ROCm™ Dockerfiles, vLLM/SGLang AITER integration, and AMD Instinct™ CI, and to the verl community for the 0.9.0 release this AMD branch is based on.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

The information contained herein is provided "AS IS" and for informational purposes only and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein. No license, including implied or arising by estoppel, to any intellectual property rights is granted by this document. Terms and limitations applicable to the purchase or use of AMD products are as set forth in a signed agreement between the parties or in AMD’s Standard Terms and Conditions of Sale. GD-18u.

©2026 Advanced Micro Devices, Inc. All rights reserved. AMD, the AMD Arrow logo, AMD Instinct™,  ROCm™, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective owners. Certain AMD technologies may require third-party enablement or activation. Supported features may vary by operating system. Please confirm with the system manufacturer for specific features. No technology or product can be completely secure.
