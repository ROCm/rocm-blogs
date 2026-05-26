---
blogpost: true
blog_title: "Accelerating xDiT Image Generation with FP8 & MXFP4 using AMD Quark on AMD Instinct™ MI350 GPUs"
date: 26 May 2026
author: 'Xiao Yu, Bowen Bao, Spandan Tiwari, Ashish Sirasao'
thumbnail: 'quark-xdit-thumbnail.jpg'
tags: AI/ML, Quantization, Diffusion Models
category: Applications & models
target_audience: AI DEVELOPERS AND ENTHUSIAST
key_value_propositions: Diffusion models such as FLUX.1-dev deliver stunning image quality but are compute- and memory-bandwidth bound at inference time. In this blog, we show how AMD Quark enables FP8 and MXFP4 quantization for xDiT FLUX.1-dev on AMD Instinct™ MI350 GPUs, delivering up to 1.43× speedup over the BF16 torch.compile baseline with near-lossless image quality.
language: English
myst:
    html_meta:
        "author": "Xiao Yu, Bowen Bao, Xinjun Niu, Wei Luo, Ke Wang, Spandan Tiwari, Ashish Sirasao"
        "description lang=en": "Accelerate xDiT FLUX.1-dev image generation on AMD Instinct MI350 GPUs using AMD Quark FP8 and MXFP4 quantization. Achieve up to 1.43x speedup with near-lossless image quality."
        "keywords": "Quark, xDiT, FLUX.1-dev, FP8, MXFP4, MI350, ROCm, Aiter, diffusion, quantization"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Xiao Yu, Bowen Bao, Spandan Tiwari, Ashish Sirasao"
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

# Accelerating xDiT Image Generation with FP8 & MXFP4 using AMD Quark on AMD Instinct™ MI350 GPUs

Diffusion models such as Black Forest Labs' FLUX.1-dev [[1]](#references) deliver stunning image quality but demand significant compute and memory bandwidth at inference time. To reduce inference cost without sacrificing image quality, precision-aware quantization techniques have become a critical optimization strategy.

In this blog, we walk through how to accelerate xDiT [[2]](#references) FLUX.1-dev image generation on AMD Instinct™ MI350 GPUs using **AMD Quark** [[3]](#references) — a high-performance quantization library tightly integrated with ROCm™ and optimized for the FP4/FP8 matrix-core capabilities of MI300/MI350-class GPUs. Starting from a BF16 `torch.compile` baseline, we apply **FP8 (E4M3)** and **MXFP4** quantization, integrate the quantized models with the Diffusers pipeline through Aiter GEMM kernels, and benchmark both latency and image quality. The result: up to **1.43× speedup** over the BF16 `torch.compile` baseline with near-lossless image quality.

---

## Quantization with Quark

AMD Quark is a comprehensive cross-platform deep learning toolkit designed to simplify and enhance the quantization of deep learning models. For diffusion model quantization, Quark is tightly integrated with ROCm™ and optimized for matrix-core acceleration. It provides:

- **Multiple numeric formats** — FP8, MXFP4, MXFP6, INT8, INT4, and more
- **Modular quantization flows** — Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), and others
- **Support for diffusion models** — FLUX, Stable Diffusion, and large decoder-only LLMs
- **Native inference acceleration** — via AMD Aiter [[4]](#references) GEMM kernels on MI300/MI350 GPUs
- **Seamless integration** — with inference pipelines such as Diffusers [[5]](#references), vLLM, and SGLang

With Quark, users can configure per-layer quantization schemes based on layer sensitivity, enabling both uniform and mixed-precision flows. By combining this flexibility with the native FP4/FP8 matrix-core capabilities of MI350 GPUs, Quark achieves near-lossless image quality while significantly improving inference efficiency.

---

## Preparation

### Environment

| Component | Version / Image |
| :--- | :--- |
| Hardware | AMD Instinct™ MI350 (gfx950) |
| Docker | `rocm/pytorch-xdit:v26.4` |
| Quark Branch | `xiaoyu/aiter_gemm_support` |
| Model | `black-forest-labs/FLUX.1-dev` |
| Resolution | 1024 × 768, 20 inference steps, `guidance_scale=3.5` |

### Docker Setup

```bash
docker run -it \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --user root \
    --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    --group-add video \
    --ipc=host --network host --privileged \
    --shm-size 128G \
    --name flux_benchmark \
    -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v /shareddata/:/data \
    -v /home/$USER:/workspace \
    -w /workspace \
    rocm/pytorch-xdit:v26.4
```

### Install Dependencies (inside the container)

```bash
apt-get update && apt-get install -y nano python3-tk

pip install torchmetrics transformers hpsv2 open_clip_torch pycocotools

# Install Quark from source
cd /workspace
git clone https://github.com/amd/quark.git Quark && cd Quark
git checkout xiaoyu/aiter_gemm_support
pip install -e .

# Verify GPU & Aiter
rocm-smi --showproductname
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
python3 -c "import aiter; print('Aiter OK')"
```

---

## BF16 Baseline

BF16 (bfloat16) serves as the unquantized baseline — it loads and runs the FLUX.1-dev model as-is in 16-bit floating point. No quantization is applied.

### Benchmark Script

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "/data/black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

prompt = "A photo of a cat sitting on a windowsill at sunset"
for _ in range(3):  # warmup
    pipe(prompt, height=768, width=1024, num_inference_steps=20, guidance_scale=3.5)

import time
latencies = []
for _ in range(10):
    t0 = time.time()
    pipe(prompt, height=768, width=1024, num_inference_steps=20, guidance_scale=3.5)
    latencies.append(time.time() - t0)
print(f"BF16 Mean latency: {sum(latencies)/len(latencies):.3f} s/img")
```

---

## FP8 Quantization

FP8 is an 8-bit floating-point format (E4M3) that offers an excellent balance between precision and dynamic range while significantly reducing memory bandwidth for activations and weights. AMD Instinct™ MI300 and newer GPUs provide native matrix-core support for FP8 operations.

### FP8 Quantization & Benchmark Script

```python
import torch
from diffusers import FluxPipeline
from quark.torch.quantization.api import ModelQuantizer, RuntimeOptions
from quark.torch.quantization.config.config import (
    QConfig, QLayerConfig, FP8E4M3PerTensorSpec,
)

pipe = FluxPipeline.from_pretrained(
    "/data/black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Configure FP8 E4M3 per-tensor dynamic quantization
quant_spec = FP8E4M3PerTensorSpec(
    observer_method="min_max",
    is_dynamic=True,
).to_quantization_spec()

layer_config = QLayerConfig(weight=quant_spec, input_tensors=quant_spec)
quantizer = ModelQuantizer(QConfig(global_quant_config=layer_config))
pipe.transformer = quantizer.quantize_model(pipe.transformer)

# Freeze model with native inference (Aiter FP8 GEMM kernels)
runtime_opts = RuntimeOptions(
    native_linear_mode="fp8_per_tensor",
    use_preshuffle=True,
)
pipe.transformer = ModelQuantizer.freeze(
    pipe.transformer, runtime_options=runtime_opts,
)

# Benchmark
prompt = "A photo of a cat sitting on a windowsill at sunset"
for _ in range(3):
    pipe(prompt, height=768, width=1024, num_inference_steps=20, guidance_scale=3.5)

import time
latencies = []
for _ in range(10):
    t0 = time.time()
    pipe(prompt, height=768, width=1024, num_inference_steps=20, guidance_scale=3.5)
    latencies.append(time.time() - t0)
print(f"FP8 Mean latency: {sum(latencies)/len(latencies):.3f} s/img")
```

---

## MXFP4 Quantization

MXFP4, defined as part of the OCP Microscaling (MX) specification [[6]](#references), groups 32 elements of 4-bit floating-point values to share a common E8M0 scaling exponent. Because it uses block-level scaling with FP4 elements, MXFP4 enables substantial model compression while retaining sufficient dynamic range for diffusion-model inference.

Supported natively on AMD Instinct™ MI350 and newer GPUs, MXFP4 delivers the strongest efficiency gains among all formats tested in this blog — achieving **up to 1.43× speedup** over the BF16 `torch.compile` baseline.

### MXFP4 Quantization & Benchmark Script

```python
import torch
from diffusers import FluxPipeline
from quark.torch.quantization.api import ModelQuantizer, RuntimeOptions
from quark.torch.quantization.config.config import (
    QConfig, QLayerConfig, OCP_MXFP4Spec,
)

pipe = FluxPipeline.from_pretrained(
    "/data/black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Configure MXFP4 quantization with 32-element block scaling
weight_spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
input_spec  = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=True).to_quantization_spec()
layer_config = QLayerConfig(weight=weight_spec, input_tensors=input_spec)

quantizer = ModelQuantizer(QConfig(global_quant_config=layer_config))
pipe.transformer = quantizer.quantize_model(pipe.transformer)

# Freeze model with MXFP4 native inference (Aiter FP4 GEMM kernels)
runtime_opts = RuntimeOptions(native_linear_mode="mxfp4")
pipe.transformer = ModelQuantizer.freeze(
    pipe.transformer, runtime_options=runtime_opts,
)

# Benchmark
prompt = "A photo of a cat sitting on a windowsill at sunset"
for _ in range(3):
    pipe(prompt, height=768, width=1024, num_inference_steps=20, guidance_scale=3.5)

import time
latencies = []
for _ in range(10):
    t0 = time.time()
    pipe(prompt, height=768, width=1024, num_inference_steps=20, guidance_scale=3.5)
    latencies.append(time.time() - t0)
print(f"MXFP4 Mean latency: {sum(latencies)/len(latencies):.3f} s/img")
```

### MXFP4 Kernel Paths

Quark supports two MXFP4 GEMM kernel paths via Aiter, selectable at freeze time:

| Path | Activation Quantization | GEMM Kernel | Eager Latency | `torch.compile` |
| :--- | :--- | :--- | :---: | :---: |
| **Triton** (default) | `dynamic_mxfp4_quant` | `gemm_afp4wfp4` | 1.806 s | Supported (`default` mode) |
| **ASM** | `per_1x32_f4_quant_hip` | `gemm_a4w4` | **1.553 s** | Not yet supported |

To use the ASM path (MI350 only, ~14% faster than the Triton eager path):

```python
runtime_opts = RuntimeOptions(native_linear_mode="mxfp4", use_asm_gemm=True)
```

---

## Quality Evaluation

Image quality is evaluated using two complementary metrics on 500 COCO 2017 [[7]](#references) caption prompts:

- **CLIP Score** [[8]](#references) — measures text-image alignment (higher is better)
- **HPS v2** [[9]](#references) — measures human-preference alignment (higher is better)

### Evaluation Script

```bash
python3 evaluate.py \
    --image_dir <OUTPUT_DIR>/images \
    --output_dir <OUTPUT_DIR> \
    --coco_annotations /data/coco/annotations/captions_val2017.json \
    --num_prompts 500
```

### Quality Results

| Configuration | CLIP Score ↑ | HPS v2 ↑ | Quality Recovery |
| :--- | :---: | :---: | :---: |
| **BF16 Baseline** | 25.92 | 0.2898 | — |
| **FP8 Per-Tensor** | 25.85 | 0.2894 | 99.7% / 99.9% |
| **FP8 + Preshuffle** | 25.92 | 0.2895 | 100% / 99.9% |
| **MXFP4 (Triton)** | 26.48 | 0.2843 | 102.2% / 98.1% |

> ↑ = indicates higher is better

Both FP8 and MXFP4 preserve image quality extremely well. FP8 achieves near-identical scores to BF16. MXFP4 shows a small HPS v2 drop (~1.9%) which is expected for 4-bit quantization, while maintaining strong CLIP scores.

> **Note:** BF16 and FP8 were evaluated on 2-GPU xDiT Ulysses runs; MXFP4 was evaluated on a single GPU. Minor score variations may reflect GPU configuration differences rather than quantization impact.

---

## Performance Uplift

All latency measurements were collected on a single AMD Instinct™ MI350 (gfx950) GPU running FLUX.1-dev at 1024×768, 20 inference steps, `guidance_scale=3.5`.

### Latency Comparison

| Configuration | Latency (s/img) ↓ | Speedup ↑ |
| :--- | :---: | :---: |
| BF16 `torch.compile` | 2.228 | 1.00× |
| **FP8 Per-Tensor Eager** | 2.270 | 0.98× |
| **MXFP4 Triton Eager** | 1.806 | **1.23×** |
| **MXFP4 Triton** `torch.compile` | 1.720 | **1.30×** |
| **FP8 Per-Tensor** `torch.compile` | 1.593 | **1.40×** |
| **MXFP4 ASM Eager** | **1.553** | **1.43×** |

> ↓ = indicates lower is better, ↑ = indicates higher is better

### Performance Chart

```text
Speedup vs BF16 torch.compile (higher is better)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BF16 torch.compile      ████████████████████████████            1.00×  (2.228s)
FP8 Per-Tensor Eager    ███████████████████████████             0.98×  (2.270s)
MXFP4 Triton Eager      ██████████████████████████████████      1.23×  (1.806s)
MXFP4 Triton Compiled   ████████████████████████████████████    1.30×  (1.720s)
FP8 Per-Tensor Compiled ██████████████████████████████████████  1.40×  (1.593s)
MXFP4 ASM Eager         ███████████████████████████████████████ 1.43×  (1.553s)
```

### Key Takeaways

1. **MXFP4 ASM achieves 1.43× speedup** over BF16 `torch.compile` — the fastest configuration tested, delivering 1.553 s/img on MI350.
2. **FP8 Per-Tensor with `torch.compile`** reaches 1.40× speedup (1.593 s/img), nearly matching MXFP4 ASM while retaining higher precision.
3. **MXFP4 Triton with `torch.compile`** reaches 1.30× speedup (1.720 s/img), combining FP4 compute with compiler-level optimizations.
4. **`torch.compile` dramatically accelerates FP8** — from 0.98× (eager) to 1.40× (compiled), a 42% latency reduction, showing strong synergy between the compiler and FP8 GEMM kernels.

---

## Reproducing the Results

### Step-by-Step

1. **Launch the Docker container** following the [Docker Setup](#docker-setup) instructions above.

2. **Run the BF16 baseline benchmark:**

   ```bash
   CUDA_VISIBLE_DEVICES=0 python3 /workspace/xdit-eval/benchmark_bf16_cudagraph.py
   ```

3. **Run the MXFP4 Triton benchmark (eager + `torch.compile`):**

   ```bash
   export PYTHONPATH="/app/external/aiter:${PYTHONPATH}"
   CUDA_VISIBLE_DEVICES=0 python3 /workspace/xdit-eval/benchmark_mxfp4_dynamic_quant.py
   ```

4. **Run the MXFP4 ASM benchmark (eager):**

   ```bash
   export PYTHONPATH="/app/external/aiter:${PYTHONPATH}"
   CUDA_VISIBLE_DEVICES=0 python3 /workspace/xdit-eval/benchmark_mxfp4_asm_gemm.py
   ```

5. **Run the quality evaluation (500 images):**

   ```bash
   python3 /workspace/xdit-eval/evaluate.py \
       --image_dir <OUTPUT_DIR>/images \
       --output_dir <OUTPUT_DIR> \
       --num_prompts 500
   ```

### Available Benchmark Scripts

| Script | Description |
| :--- | :--- |
| `benchmark_bf16_cudagraph.py` | BF16 baseline: eager + `torch.compile` (`default` and `reduce-overhead` modes) |
| `benchmark_mxfp4_dynamic_quant.py` | MXFP4 Triton path: eager + `torch.compile` |
| `benchmark_mxfp4_asm_gemm.py` | MXFP4 ASM path: eager (fastest) |
| `benchmark_flux_quark_mxfp4.py` | MXFP4 full evaluation (500 images, CLIP / HPS v2) |
| `evaluate.py` | Standalone CLIP Score + HPS v2 quality evaluation |

---

## Summary

In this blog, we demonstrated a practical, step-by-step workflow for quantizing and accelerating xDiT FLUX.1-dev image generation using AMD Quark on AMD Instinct™ MI350 GPUs:

1. **MXFP4 (4-bit)** delivers up to **1.43× speedup** over the BF16 `torch.compile` baseline with only ~1.9% HPS v2 quality loss, leveraging the native FP4 matrix-core hardware on MI350.
2. **FP8 (8-bit)** with `torch.compile` delivers **1.40× speedup** (1.593 s/img) with virtually no quality loss (99.7%+ recovery across both CLIP and HPS v2 metrics).
3. **`torch.compile`** further improves MXFP4 Triton throughput to **1.30×** over BF16 compiled by enabling graph-level optimizations and reducing CPU-side dispatch overhead.

By combining Quark's flexible quantization workflows with the native FP4/FP8 matrix-core capabilities of MI350-class GPUs, developers can efficiently deploy diffusion models with significantly lower latency and reduced memory footprint — while maintaining near-lossless image quality.

Looking ahead, the MXFP4 ASM kernel path will gain `torch.compile` support, and we plan to extend Quark's diffusion-model coverage to additional architectures and mixed-precision flows. The optimizations demonstrated here will be progressively merged into upstream Quark and Aiter, making these performance gains available to the broader community.

## Acknowledgements

The authors wish to thank the AMD Quark and Aiter teams for their invaluable guidance and support in enabling FP8 and MXFP4 GEMM kernels on AMD Instinct™ MI350 GPUs.

## References

[1] [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) — Black Forest Labs' open-weights diffusion transformer for text-to-image generation

[2] [xDiT](https://github.com/xdit-project/xDiT) — Inference engine for diffusion transformers with parallelism support

[3] [AMD Quark](https://github.com/amd/quark) — A cross-platform deep learning quantization toolkit

[4] [AITER](https://github.com/ROCm/aiter) — AI Tensor Engine for ROCm

[5] [Diffusers](https://github.com/huggingface/diffusers) — Hugging Face library for state-of-the-art diffusion models

[6] [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) — Open Compute Project standard for block-scaled low-precision formats

[7] [COCO 2017](https://cocodataset.org/) — Common Objects in Context image-caption dataset

[8] [CLIP Score](https://github.com/openai/CLIP) — Text-image alignment metric based on OpenAI's CLIP model

[9] [HPS v2](https://github.com/tgxs002/HPSv2) — Human Preference Score v2 for text-to-image generation

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
