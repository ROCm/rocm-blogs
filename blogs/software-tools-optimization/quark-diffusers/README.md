---
blogpost: true
blog_title: "Quark Support for HuggingFace Diffusers and SVDQuant"
date: "06 Aug 2026"
author: "Inesh Chakrabarti, Bowen Bao, Spandan Tiwari, Ashish Sirasao"
thumbnail: 'svdquant_flux_nvfp4.png'
tags:  AI/ML, GenAI, Diffusion Model, Performance
category: Applications & models
target_audience: "AI developers and researchers optimizing image generation inference"
key_value_propositions: "AMD Quark now supports Diffusers through HuggingFace Diffusers integration as well as SVDQuant Support."
language: English
myst:
    html_meta:
        "author": "Inesh Chakrabarti, Bowen Bao, Spandan Tiwari, Ashish Sirasao"
        "description lang=en": "Learn how to quantize, save, and reload diffusion models in Quark using its new SVDQuant and HuggingFace Diffusers support."
        "keywords": "Quark, Diffusers, SVDQuant, HuggingFace"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Inesh Chakrabarti, Bowen Bao, Spandan Tiwari, Ashish Sirasao"
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

# Quark Support for HuggingFace Diffusers and SVDQuant

Diffusion models are heavy on memory and compute: a single text-to-image call runs a large transformer or UNet dozens of times. Quantization — storing weights (and sometimes activations) in low precision — is one of the most effective ways to cut both the memory footprint and the latency of these models.

In a previous post [[1]](#references) we showed how **AMD Quark** [[2]](#references) brings **MXFP4** quantization to Diffusers [[3]](#references) and xDiT FLUX.1-dev [[4]](#references) image generation, reaching up to **1.92× speedup** over the BF16 eager baseline on a single AMD Instinct™ MI350 GPU while preserving quality. This post covers two new capabilities for diffusion models: **SVDQuant** [[5]](#references), for accurate low-bit (4-bit) quantization, and Quark's **native Hugging Face Diffusers support**. Quark now allows quantized diffusion models to be saved and reloaded through the standard Diffusers `save_pretrained` / `from_pretrained` APIs.

---

## SVDQuant Support in Quark

Pushing diffusion models to 4-bit — quantizing not just weights but **activations** too — is much harder than the FP8 or INT8 case. Both the weights and the activations in diffusion transformers contain **outlier channels** whose magnitudes dwarf the rest of the tensor. A naive 4-bit grid has to stretch to cover those outliers, which crushes the resolution available for the bulk of the values and shows up as visible artifacts in generated images.

SVDQuant addresses this with two ideas working together:

- **Smoothing** migrates part of the activation dynamic range into the weights (as in SmoothQuant [[6]](#references)), so neither side has to absorb the full outlier magnitude alone.
- A **high-precision low-rank correction branch** captures the residual outliers that low-bit quantization cannot represent. The weight matrix is decomposed via SVD; at inference, the output is the result of the low-bit residual GEMM **plus** a small rank-16-to-32 correction. That correction is cheap in compute and memory, yet recovers most of the accuracy lost at 4 bits.

This is what makes 4-bit **activation** quantization viable, not just weight-only compression: the `w4a4`, `mxfp4`, and `nvfp4` modes below quantize activations as well as weights. In Quark, SVDQuant is configured through `SVDQuantConfig`, with ready-made schemes from `build_quant_layer_config`:

| Mode      | Weights          | Activations             | Notes |
|-----------|------------------|-------------------------|-------|
| `w4a16`   | INT4 per-group   | fp16 / bf16             | weight-only 4-bit |
| `w4a4`    | INT4 per-group   | INT4 per-group dynamic  | fully 4-bit |
| `mxfp4`   | MXFP4            | MXFP4 (dynamic)         | OCP microscaling FP4 [[7]](#references), [[8]](#references), native on MI350 |
| `nvfp4`   | FP4 block-16     | FP4 block-16 dynamic    | FP4 with FP8 block scales [[9]](#references) |

**Native inference.** On MI300 / MI350 GPUs, `quark.torch.enable_native_inference` runs the low-bit residual GEMM on AMD AITER [[10]](#references) matrix-core kernels. For SVDQuant, the low-rank correction runs as a *separate* branch alongside that GEMM (optionally overlapped on a second CUDA stream via `RuntimeOptions(svdquant_overlap_streams=True)`); it is **not yet fused into the GEMM kernel** — a fused SVDQuant kernel is planned and will further reduce its latency. This path is part of AMD Quark's diffusion support.

The table below shows Quark quantization of FLUX.1-dev on a gfx950 / MI350 GPU, reported relative to the BF16 baseline:

| Config (FLUX.1-dev)    | Speedup vs BF16 ↑ | Model mem vs BF16 ↓ | Peak mem vs BF16 ↓ |
|------------------------|:-----------------:|:-------------------:|:------------------:|
| BF16                   | 1.00×             | 1.00×               | 1.00×              |
| Native FP4 (plain RTN) | 1.12×             | 0.67×               | 0.69×              |
| Native SVDQuant        | 0.92×             | 0.73×               | 0.75×              |

Plain native FP4 runs about **1.12× faster** than BF16 while cutting model memory by roughly a third. Native SVDQuant trades a little speed (currently ~**0.92×** of BF16, due to the extra correction branch that is not yet fused) for the accuracy that makes 4-bit *activations* viable, while still trimming model memory by ~27%. Once the correction branch is fused into the GEMM, SVDQuant's latency should move toward the plain FP4 path.

The images below are FLUX.1-dev generations produced with Quark SVDQuant across all four low-bit formats, using the same prompt and seed. Quality holds across INT4 and FP4:

| SVDQuant W4A16 | SVDQuant W4A4 |
|:---:|:---:|
| ![FLUX.1-dev generated with SVDQuant W4A16](./images/svdquant_flux_w4a16.png) | ![FLUX.1-dev generated with SVDQuant W4A4](./images/svdquant_flux_w4a4.png) |
| **SVDQuant MXFP4** | **SVDQuant NVFP4** |
| ![FLUX.1-dev generated with SVDQuant MXFP4](./images/svdquant_flux_mxfp4.png) | ![FLUX.1-dev generated with SVDQuant NVFP4](./images/svdquant_flux_nvfp4.png) |

*Figure 1. FLUX.1-dev generated with Quark SVDQuant across four low-bit formats (INT4 weight-only, INT4 W4A4, MXFP4, NVFP4), same prompt and seed — image quality holds across all four.*

These samples were produced with `examples/torch/diffusers/testSVDQuant.py` (multi-mode SVDQuant for SDXL / FLUX / SD3). On FLUX.1-dev, the SVDQuant W4A4, MXFP4, and NVFP4 variants all maintain CLIP scores [[11]](#references) within about half a point of the FP16 reference:

| SVDQuant (FLUX.1-dev) | Residual rounding | CLIP ↑ |
|-----------------------|-------------------|:------:|
| FP16 (reference)      | —                 | 27.82  |
| W4A4                  | RTN               | 27.50  |
| W4A4                  | GPTQ              | 27.40  |
| MXFP4                 | RTN               | 27.41  |
| MXFP4                 | GPTQ              | 27.36  |
| NVFP4                 | RTN               | 27.72  |
| NVFP4                 | GPTQ              | 28.01  |

*CLIP score with `openai/clip-vit-large-patch14` on 1,000 MJHQ [[12]](#references) prompts (higher is better). "RTN" and "GPTQ" [[13]](#references) denote how the 4-bit residual weights are rounded.*

---

## Diffusers Support

A single `import quark.integrations.diffusers` statement registers Quark as a Hugging Face Diffusers quantizer, so quantized diffusion models behave like any other Diffusers checkpoint. There are two ways to use it.

**Offline quantization — quantize once, reload anywhere.** Quantize a pipeline submodule with `ModelQuantizer`, save it with `export_safetensors` (which routes through `save_pretrained` and embeds the serialized Quark `QConfig` under `quantization_config` in `config.json`), and reload later with plain `from_pretrained`. The checkpoint is self-describing: the loader reads the config, rebuilds the quantized layers (with meta-device and `low_cpu_mem_usage` loading supported), loads the weights, and freezes the model for inference — no `QConfig` needed at the call site.

**Native online quantization — quantize at load time.** Pass `quantization_config=...` to `from_pretrained` against a plain fp16/bf16 checkpoint, and Quark applies weight-only quantization in-process, with no export/reload round-trip. We already offered online quantization through xDiT [[1]](#references) — where Quark replaces a transformer's linear layers with FP8 or MXFP4 implementations at load time and routes them to AITER kernels — and that same online path is now available directly through Diffusers.

**Availability.** Today this is enabled by importing `quark.integrations.diffusers`, which self-registers (monkeypatches) the `"quark"` method into the Diffusers registries at runtime. We also have a PR to add Quark to Diffusers upstream ([huggingface/diffusers#14077](https://github.com/huggingface/diffusers/pull/14077)); until it merges, the one-line import is all that's needed — nothing else in your code changes.

---

## Tutorial: Quantize, Save, and Reload a Diffusion Model

This tutorial is adapted from the scripts in `examples/torch/diffusers` (`quantize_diffusers.py`, `testSVDQuant.py`). It uses SDXL [[14]](#references) and FLUX.1-dev [[4]](#references) as running examples; the same pattern applies to SD1.5 [[15]](#references), SD3 [[16]](#references), and PixArt [[17]](#references) by swapping the pipeline class and the target submodule.

### Environment

The setup mirrors the earlier xDiT blog. On an AMD Instinct GPU with a recent ROCm PyTorch image:

```bash
docker run -it \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --network host --shm-size 128G \
    -v /shareddata/:/data -w /workspace \
    rocm/pytorch-xdit:v26.5
```

Inside the container, install Diffusers and Quark:

```bash
pip install diffusers transformers accelerate
# Install Quark from source (public repo, main branch)
git clone https://github.com/amd/quark.git Quark && cd Quark && pip install -e .
```

> Which submodule do you quantize? For **SDXL / SD1.5** it is `pipe.unet`; for **FLUX / SD3 / PixArt** it is `pipe.transformer`. Quantization is applied to that submodule, not the VAE or text encoders.

### 1. Quantize, Save, and Reload through Diffusers

This is the headline workflow. Weight-only quantization needs no calibration data, so it is the simplest way to see the full round-trip: quantize the target submodule with `ModelQuantizer`, save it with `export_safetensors` (which routes through `save_pretrained` and embeds the Quark `QConfig` in `config.json`), and reload it later with plain `from_pretrained`.

```python
import torch
from diffusers import DiffusionPipeline
from quark.torch import ModelQuantizer, export_safetensors
from quark.torch.quantization.config.config import Int8PerTensorSpec, QConfig, QLayerConfig

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16",
).to("cuda")

# INT8 weight-only. dataloader=None is allowed because there are no activation quantizers.
weight_spec = Int8PerTensorSpec(
    observer_method="min_max", symmetric=True, scale_type="float",
    round_method="half_even", is_dynamic=False,
).to_quantization_spec()
qconfig = QConfig(global_quant_config=QLayerConfig(weight=weight_spec))

pipe.unet = ModelQuantizer(qconfig).quantize_model(pipe.unet, dataloader=None)

# Save through the standard Diffusers API: writes diffusion_pytorch_model.safetensors
# plus a config.json carrying the serialized QConfig under `quantization_config`.
export_safetensors(pipe.unet, "./sdxl-unet-quark-int8")
```

Reloading is a two-liner. Importing the integration self-registers the `"quark"` method; `from_pretrained` then reads `quantization_config`, reconstructs the quantized layers via `process_model_transformation`, loads the checkpoint, and freezes the model for inference — no `QConfig` needed at the call site:

```python
import quark.integrations.diffusers  # registers "quark" into the Diffusers registries
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("./sdxl-unet-quark-int8")
pipe.unet = unet  # drop the reloaded, quantized module back into the pipeline

image = pipe("A cat on a windowsill", num_inference_steps=30, guidance_scale=8.0).images[0]
image.save("sdxl_int8.png")
```

The same `config.json` mechanism works when the quantized submodule lives inside a full pipeline directory: `DiffusionPipeline.from_pretrained("<org>/<checkpoint>")` reloads the whole quantized pipeline in one call. FP8 and other standard PTQ schemes follow the identical export/reload flow (activation-quantized schemes additionally need the calibration step described next).

### 2. Going to 4 Bits with SVDQuant

For true 4-bit quantization you want SVDQuant, and activation-aware schemes need calibration data. The key insight is that a diffusion submodule's inputs are intermediate latents, timestep embeddings, and conditioning — so calibration means **running the pipeline** and capturing those inputs. `get_calib_dataloader` does exactly that: each prompt triggers one pipeline run, and with `n_steps` denoising steps the submodule is called `n_steps` times per prompt, yielding `len(prompts) * n_steps` calibration samples.

```python
import torch
from diffusers import FluxPipeline
from quark.torch import ModelQuantizer, save_params
from quark.torch.quantization.config.config import QConfig, SVDQuantConfig
from quark.torch.algorithm.svdquant import build_quant_layer_config
from quark.torch.utils.diffusers import get_calib_dataloader

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)

prompts = [
    "A serene lake reflecting mountains at sunset",
    "A futuristic city with flying cars at night",
    "A close-up portrait with dramatic lighting",
    "A golden retriever playing in autumn leaves",
    "An astronaut floating above Earth",
]

# FLUX-specific pipe kwargs are forwarded straight to pipe(...).
dataloader = get_calib_dataloader(
    pipe, pipe.transformer, prompts, n_steps=20,
    height=1024, width=1024, guidance_scale=3.5, max_sequence_length=512,
)
```

`build_quant_layer_config("mxfp4")` selects the MXFP4 scheme (swap in `"w4a16"`, `"w4a4"`, or `"nvfp4"` as needed), and `SVDQuantConfig` adds the low-rank correction branch. Note the two exclude lists: sensitive embedding/normalization layers are skipped, and `*correction*` must always be excluded from quantization so the high-precision low-rank branch is protected.

```python
SVD_EXCLUDE = [
    "*x_embedder*", "*context_embedder*", "*time_text_embed*",
    "*norm_out*", "*proj_out*", "*norm1.linear*", "*norm1_context.linear*",
]

qconfig = QConfig(
    global_quant_config=build_quant_layer_config("mxfp4"),   # or "w4a16", "w4a4", "nvfp4"
    exclude=[*SVD_EXCLUDE, "*correction*"],
    algo_config=[SVDQuantConfig(
        svd_rank=32,
        search_alpha=False,          # set True to search per-layer smoothing alpha
        min_layer_size=256,
        exclude_patterns=SVD_EXCLUDE,
    )],
)

pipe.transformer = ModelQuantizer(qconfig).quantize_model(pipe.transformer, dataloader)

image = pipe(
    "A cat on a windowsill", num_inference_steps=50,
    height=1024, width=1024, guidance_scale=3.5, max_sequence_length=512,
).images[0]
image.save("flux_svdquant_mxfp4.png")
```

An SVDQuant checkpoint is an `ErrorCorrectedModule` (a quantized residual plus the low-rank correction and smooth factors), so it is persisted with Quark's `save_params` rather than the QDQ Diffusers export above:

```python
frozen = ModelQuantizer.freeze(pipe.transformer)
save_params(frozen, model_type="transformer", export_dir="./flux-svdquant-mxfp4")
```

**Tuning the smoothing strength with the calibration data.** SVDQuant's `smooth_alpha` sets how much of the activation range is migrated into the weights before the SVD split. A single global value (default `0.5`) is a fine starting point, but the best alpha varies per layer. Setting `search_alpha=True` turns on a per-layer search that **reuses the calibration activations you already collected**: for each layer, the search tries several candidate alpha values and keeps the one that minimizes post-SVD reconstruction error, using only a handful of cached activations so it stays cheap.

```python
algo_config=[SVDQuantConfig(
    svd_rank=32,
    search_alpha=True,                          # per-layer alpha search, driven by the calib activations
    alpha_candidates=[0.1, 0.3, 0.5, 0.7, 0.9], # optional; defaults to a ~0.05-0.95 sweep
    alpha_search_max_samples=8,                 # cached activations per layer used for the search
    min_layer_size=256,
    exclude_patterns=SVD_EXCLUDE,
)]
```

The search runs inside the same `ModelQuantizer(qconfig).quantize_model(pipe.transformer, dataloader)` call — the dataloader from `get_calib_dataloader` supplies the activations, so no extra setup is needed. To sweep more broadly, `examples/torch/diffusers/svdquant_calibrate.py` automates a grid search over the smoothing alpha, GPTQ on/off, and calibration-sample count, scores each configuration against a high-precision reference, and reports the best; `testSVDQuant.py` covers SDXL / FLUX / SD3 across `w4a16`, `w4a4`, and `mxfp4`.

### 3. Native FP4 Inference on AMD Instinct

Quantizing the model reduces its memory footprint immediately, but to turn low precision into *latency* wins you want real low-bit GEMM kernels rather than the emulation (QDQ) path. On MI300 / MI350 GPUs, `enable_native_inference` moves the low-bit residual GEMM (and, for SVDQuant, the low-rank correction branch) to AMD AITER matrix-core kernels:

```python
from quark.torch import enable_native_inference, RuntimeOptions

n = enable_native_inference(
    pipe.transformer,
    runtime_options=RuntimeOptions(native_linear_mode="mxfp4"),
)
print(f"Native inference enabled for {n} layers")

image = pipe(
    "A cat on a windowsill", num_inference_steps=50,
    height=1024, width=1024, guidance_scale=3.5, max_sequence_length=512,
).images[0]
```

Native inference is a **runtime mode re-enabled after loading**, not a save format — the on-disk checkpoint is identical whether or not you later run natively, so you save once and choose the execution path per deployment GPU. The SVDQuant native path — the AITER MXFP4 residual GEMM plus the separate low-rank correction branch — is part of AMD Quark's native-inference support.

---

## Summary

In this blog, you learned how to take a diffusion model down to 4 bits with AMD Quark and run it on AMD Instinct GPUs without leaving the Hugging Face Diffusers workflow you already use. Specifically, you explored:

- **Why 4-bit activation diffusion is hard, and how SVDQuant answers it.** Outlier channels in both weights and activations force a naive 4-bit grid to stretch until the bulk of the values lose resolution. SVDQuant pairs SmoothQuant-style smoothing with a high-precision, low-rank correction branch, and that combination is what makes 4-bit *activation* quantization viable rather than weight-only compression. On FLUX.1-dev you saw the W4A4, MXFP4, and NVFP4 variants all maintain CLIP scores within about half a point of the FP16 reference.
- **How to quantize, save, and reload through the standard Diffusers APIs.** A single `import quark.integrations.diffusers` statement registers Quark as a Diffusers quantizer, which makes a quantized checkpoint self-describing: `export_safetensors` embeds the `QConfig` in `config.json`, and a plain `from_pretrained` rebuilds the quantized layers with no `QConfig` needed at the call site. You can also skip the round-trip entirely and quantize online at load time — the same capability we previously offered through xDiT, now available directly in Diffusers.
- **How to calibrate an activation-aware scheme.** Because a diffusion submodule's inputs are intermediate latents and conditioning rather than text or images, calibration means running the pipeline. `get_calib_dataloader` captures those inputs for you, and `search_alpha=True` reuses the very same activations to tune the smoothing strength layer by layer.
- **How to turn low precision into real speed.** `enable_native_inference` moves the low-bit GEMM onto AMD AITER matrix-core kernels, where plain native FP4 reaches 1.12× the BF16 baseline on MI350 while cutting model memory by roughly a third.

**What's next.** The SVDQuant low-rank correction currently runs as a separate branch alongside the residual GEMM, which is why it trades a little speed today at ~0.92× of BF16. A fused SVDQuant kernel is planned, and it should move that latency toward the plain FP4 path while keeping the accuracy that 4-bit activations depend on. On the integration side, [huggingface/diffusers#14077](https://github.com/huggingface/diffusers/pull/14077) will add Quark to Diffusers upstream, at which point even the one-line import goes away.

Try it on your own pipeline: quantize `pipe.transformer` (or `pipe.unet`), export the model with `export_safetensors`, and reload it with `from_pretrained`. The full, runnable scripts live in `examples/torch/diffusers` in the public [AMD Quark](https://github.com/amd/quark) repository.

## Acknowledgements

The authors thank the AMD Quark and AITER teams for their guidance and support in enabling FP4 GEMM kernels and the Diffusers integration on AMD Instinct™ GPUs.

## References

[1] Yu, X., Bao, B., Niu, X., Luo, W., Wang, K., Tiwari, S., and Sirasao, A. "Accelerating Diffusers and xDiT Image Generation with MXFP4 using AMD Quark on AMD Instinct MI350 GPUs." *AMD ROCm Blogs*, July 2026. <https://rocm.blogs.amd.com/artificial-intelligence/quark-xdit/README.html>.

[2] AMD Quark: cross-platform deep learning quantization toolkit. GitHub: <https://github.com/amd/quark>. Documentation: <https://quark.docs.amd.com/latest/>.

[3] Hugging Face. "Diffusers: state-of-the-art diffusion models for image, video, and audio generation in PyTorch." GitHub: <https://github.com/huggingface/diffusers>.

[4] Black Forest Labs. "FLUX.1-dev." Hugging Face model card. <https://huggingface.co/black-forest-labs/FLUX.1-dev>.

[5] Li, M., Lin, Y., Zhang, Z., Cai, T., Li, X., Guo, J., Xie, E., Meng, C., Zhu, J.-Y., and Han, S. "SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models." *ICLR 2025.* arXiv:2411.05007. <https://arxiv.org/abs/2411.05007>.

[6] Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." *ICML 2023.* arXiv:2211.10438. <https://arxiv.org/abs/2211.10438>.

[7] Open Compute Project. "OCP Microscaling Formats (MX) Specification, Version 1.0." September 2023. <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>.

[8] Darvish Rouhani, B., Zhao, R., More, A., Hall, M., Khodamoradi, A., Deng, S., Choudhary, D., Cornea, M., Dellinger, E., Denolf, K., et al. "Microscaling Data Formats for Deep Learning." arXiv:2310.10537, 2023. <https://arxiv.org/abs/2310.10537>.

[9] NVIDIA. "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference." *NVIDIA Technical Blog*, June 2025. <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>.

[10] AITER: AI Tensor Engine for ROCm. GitHub: <https://github.com/ROCm/aiter>.

[11] Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. "Learning Transferable Visual Models From Natural Language Supervision." *ICML 2021.* arXiv:2103.00020. <https://arxiv.org/abs/2103.00020>.

[12] Li, D., Kamko, A., Akhgari, E., Sabet, A., Xu, L., and Doshi, S. "Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation." arXiv:2402.17245, 2024. <https://arxiv.org/abs/2402.17245>. MJHQ-30K benchmark: <https://huggingface.co/datasets/playgroundai/MJHQ-30K>.

[13] Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023.* arXiv:2210.17323. <https://arxiv.org/abs/2210.17323>.

[14] Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., and Rombach, R. "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis." *ICLR 2024.* arXiv:2307.01952. <https://arxiv.org/abs/2307.01952>.

[15] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022.* arXiv:2112.10752. <https://arxiv.org/abs/2112.10752>.

[16] Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." *ICML 2024.* arXiv:2403.03206. <https://arxiv.org/abs/2403.03206>.

[17] Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., and Li, Z. "PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis." *ICLR 2024.* arXiv:2310.00426. <https://arxiv.org/abs/2310.00426>.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
