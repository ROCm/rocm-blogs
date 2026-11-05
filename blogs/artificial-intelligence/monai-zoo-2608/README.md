---
blogpost: true
blog_title: "Bringing the MONAI Model Zoo to AMD Instinct™ GPUs: Validated Inference on ROCm"
date: "30 Oct 2026"
author: "Soumitra Chatterjee, Vikas C Sajjan, Nilaykumar K Patel, Prateek Chokse, Anik Chaudhuri, Marco Grond, Vish Vadlamani, Phani Vaddadi"
thumbnail: 'hero-model-zoo.jpeg'
tags: "AI/ML, Computer Vision, Performance, PyTorch"
category: "Applications & models"
target_audience: "ML engineers and clinical researchers deploying validated medical imaging models on AMD Instinct GPUs"
key_value_propositions: "Five MONAI Model Zoo bundles validated for inference on AMD Instinct MI355X (gfx950) with ROCm 10.0. ROCm overlay applied via an explicit configuration list. All Dice thresholds passing."
language: English
myst:
    html_meta:
        "author": "Soumitra Chatterjee, Vikas C Sajjan, Nilaykumar K Patel, Prateek Chokse, Anik Chaudhuri, Marco Grond, Vish Vadlamani, Phani Vaddadi"
        "description lang=en": "Learn how to run validated inference for five MONAI Model Zoo bundles on AMD Instinct™ GPUs using a ROCm overlay."
        "keywords": "MONAI Model Zoo, AMD Instinct, ROCm, medical imaging, inference, Vista3D, SwinUNETR, torch.compile, BF16, channels-last"
        "vertical": "AI, Developers, Data Science"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Inference, Computer Vision, AI Training"
        "amd_blog_topic_categories": "Industry Applications & Use Cases"
        "amd_blog_authors": "Soumitra Chatterjee, Vikas C Sajjan, Nilaykumar K Patel, Prateek Chokse, Anik Chaudhuri, Marco Grond, Vish Vadlamani, Phani Vaddadi"
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

# Bringing the MONAI Model Zoo to AMD Instinct™ GPUs: Validated Inference on ROCm

The Medical Open Network for AI (MONAI) Model Zoo is often the first stop when a researcher is
building a medical imaging application: a shared shelf of ready-to-use, peer-reviewed models, where
you can find a whole-body computed tomography (CT) segmenter or an interactive annotation network
already trained, packaged, and ready to run.

In this blog you will learn how five of those bundles run validated, optimized inference on AMD
Instinct™ GPUs, what the ROCm™ overlay does to get them there, and how to run them yourself. This
post is one of a series on the AMD Life Science 26.08 medical-imaging stack. The companion posts
cover [the hipCIM image-I/O library beneath it](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-26-08/README.html) and
[MONAI data loading on top of it](https://rocm.blogs.amd.com/artificial-intelligence/monai-hipcim-2608/README.html), with interactive annotation still to come.

Each entry in the Model Zoo is a self-contained MONAI bundle with the model
definition, the pre-trained weights, and a configuration-driven inference pipeline that can be launched with a single command.

With [AMD Life Science 26.08](https://rocm.docs.amd.com/projects/lifescience/en/latest/), AMD is releasing validated, optimized
inference support for five MONAI Model Zoo bundles on AMD Instinct™ GPUs. This is an Early Access release focused on inference.

## Optimized Inference Through a ROCm Overlay

Five Model Zoo bundles run optimized inference on AMD Instinct GPUs
using ordinary PyTorch, with no offline engine compilation required. The optimization arrives through a ROCm™ overlay. The overlay applies channels-last 3D, 16-bit brain floating point (BF16), `torch.compile()`, and targeted fixes on top of the stock bundle configuration.

Table 1 lists the five bundles.

| Bundle | Task | Architecture | Anatomy |
| --- | --- | --- | --- |
| Vista3D | Interactive 3D segmentation | SegResNet-based encoder + SAM-style interactive prompts | Universal (133 structures) |
| SwinUNETR (BTCV) | Multi-organ CT segmentation | Swin Transformer U-Net | 13 abdominal organs |
| wholeBody_ct_segmentation | Whole-body CT segmentation | SegResNet | 104 anatomical structures |
| pancreas_ct_dints_segmentation | Pancreas CT segmentation | DiNTS (differentiable neural architecture search, or NAS) | Pancreas |
| spleen_deepedit_annotation | Interactive spleen annotation | DynUNet | Spleen |

*Table 1. The five validated bundles.*

Every bundle runs through the standard `monai.bundle` runner and clears its upstream Dice
thresholds (the Sørensen–Dice overlap score used to score segmentation accuracy) with no changes to model weights and no changes to inference logic.

Per-bundle compiled-PyTorch latencies range from 3.60 ms (spleen_deepedit) to 3,287 ms (pancreas_dints) on AMD Instinct MI355X (gfx950) with ROCm 10.0, measured on a spleen CT volume (spleen_10.nii.gz). All measurements in this post were taken on AMD Instinct MI355X.

## The Overlay: Optimization Without a Fork

MONAI Model Zoo on ROCm uses MONAI's native configuration-overlay mechanism. Each validated bundle ships a companion
`inference_rocm.json` or `inference_rocm.yaml` file alongside the standard `inference.json`. To enable AMD optimizations, pass both configuration files as a JSON list:

```bash
# MIOpen: route channels-last convolutions through NHWC kernels and tune
export PYTORCH_MIOPEN_SUGGEST_NHWC=1
export MIOPEN_FIND_MODE=1
export MIOPEN_FIND_ENFORCE=4

# TorchInductor: autotune torch.compile() kernels
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
export TORCHINDUCTOR_EPILOGUE_FUSION=1
export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=ATEN,TRITON

# Bundles load custom code from the bundle's scripts/ package, which must be on
# the Python path when the configuration is initialized. Run from inside the bundle.
cd models/BUNDLE_NAME

python -m monai.bundle run \
    --config_file "['configs/inference.json',
                    'configs/inference_rocm.json']" \
    --bundle_root . \
    --dataset_dir /path/to/input_dir \
    --output_dir /path/to/output_dir
```

The overlay merges into the base configuration at load time, and no upstream files are modified. Configuration files are either in JSON or YAML format, depending on the bundle.

Run the command from inside the bundle directory. MONAI adds the bundle root to the Python path only once `run()` starts, while `initialize()` resolves the bundle's own `scripts.*` classes before that. A bundle with custom code fails to import when launched from the repository root.

Pass `--dataset_dir` and `--output_dir` as absolute paths.

### Three Techniques, Applied Consistently

Every `inference_rocm.*` overlay applies the same three optimizations across all five bundles:

1. Channels-last 3D memory layout (`memory_format=torch.channels_last_3d`): aligns tensor memory to the layout the HIP 3D convolution kernels expect, eliminating the redundant transposes that would otherwise wrap every 3D convolution layer.
2. BF16 automatic mixed precision (`amp_kwargs={'dtype': torch.bfloat16}`): uses native
   BF16 on AMD GPUs. Activations and weights move as 16-bit values instead of 32-bit, and BF16 keeps
   a wide exponent range.
3. `torch.compile()`: uses PyTorch's Inductor backend, which on ROCm triggers hipBLASLt kernel
   selection and graph-level fusion. It is applied at the network level as `torch.compile(@network)`,
   so the full inference graph is compiled once and cached.

The Vista3D overlay is representative of the pattern:

```json
{
    "+imports": ["$import torch"],
    "network": "$@network_def.to(@device).to(memory_format=torch.channels_last_3d)",
    "initialize": [
        "$monai.utils.set_determinism(seed=123)",
        "$setattr(torch.backends.cudnn, 'deterministic', False)",
        "$setattr(@evaluator, 'amp_kwargs', {'dtype': torch.bfloat16})",
        "$@checkpointloader(@evaluator)",
        "$setattr(@evaluator, 'network', torch.compile(@network))"
    ]
}
```

The `+imports` prefix is MONAI bundle syntax for appending to an existing list. Every other key
overrides its upstream counterpart, and the base `inference.json` stays untouched.

Four of the five bundles need nothing beyond those keys. Only `spleen_deepedit` adds a line to its
own overlay:

```json
"$@network_def.enable_gemm_transpose(True) if hasattr(@network_def, 'enable_gemm_transpose') else None"
```

## Performance on AMD Instinct MI355X (gfx950)

Inference latency was measured with the overlay on AMD Instinct, using ordinary PyTorch with no
offline engine-compilation step.

Table 3 in the System Configuration section lists the hardware, software, and timing method behind
these latency numbers. Table 2 below shows the p50 latency and Dice result for each bundle. All five
bundles clear the upstream MONAI Model Zoo Dice thresholds.

| Bundle | MI355X (gfx950) inference latency (ms, p50) | Dice / correctness |
| --- | --- | --- |
| Vista3D | 268 | PASS |
| SwinUNETR (BTCV) | 293 | PASS |
| wholeBody_ct | 386 | PASS |
| pancreas_ct_dints | 3,287 | PASS |
| spleen_deepedit | 3.60 | PASS |

*Table 2. Per-bundle inference latency. p50 is the median: half the timed iterations completed
faster than the value shown. Measured over 20 timed iterations after 5 warmup iterations.*

## Memory Capacity for Large Volumes

Sliding-window inference works through a volume in overlapping windows, and the cost is dominated by
moving data rather than by arithmetic. That makes available memory a practical constraint: the more
of the volume and its activations that stay resident on the GPU, the less the pipeline has to shuttle
between host and device between windows. AMD Instinct MI355X carries 288 GB of HBM3E, which is why
the largest bundles here, Vista3D at 133 structures and wholeBody_ct at 104, run without the working
set spilling.

## Getting Started

The validated bundles are in the AMD fork of the MONAI Model Zoo in
[`AMD-Ecosystem/monai-model-zoo`](https://github.com/AMD-Ecosystem/monai-model-zoo).
You'll need:

- An AMD Instinct GPU. The measurements in this post were taken on MI355X; MI300X and MI325X are supported by the release
- ROCm 10.0
- Python 3.12
- MONAI 1.6.0, from the [AMD ROCm package index](https://pypi.amd.com/rocm-10.0.0/simple/) or a [ROCm PyTorch Docker image](https://hub.docker.com/r/rocm/pytorch). The [MONAI on ROCm installation guide](https://rocm.docs.amd.com/projects/monai/en/docs-26.08/install/installation.html) has the full steps

```bash
git clone https://github.com/AMD-Ecosystem/monai-model-zoo

# MIOpen: route channels-last convolutions through NHWC kernels and tune
export PYTORCH_MIOPEN_SUGGEST_NHWC=1
export MIOPEN_FIND_MODE=1
export MIOPEN_FIND_ENFORCE=4

# TorchInductor: autotune torch.compile() kernels
export TORCHINDUCTOR_MAX_AUTOTUNE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
export TORCHINDUCTOR_EPILOGUE_FUSION=1
export TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS=ATEN,TRITON

# Run from inside the bundle directory so its scripts/ package is importable:
cd models/BUNDLE_NAME

# Pass both the base configuration and the ROCm overlay as a JSON list:
python -m monai.bundle run \
    --config_file "['configs/inference.json',
                    'configs/inference_rocm.json']" \
    --bundle_root . \
    --dataset_dir /path/to/input_dir \
    --output_dir /path/to/output_dir
```

Check the bundle's own `docs/` directory for its expected input format: the input key varies by
bundle, and `vista3d` takes `--input_dict` rather than `--dataset_dir`.

## Summary

In this blog you explored how five MONAI Model Zoo bundles spanning universal segmentation,
whole-body anatomy, pancreas, and spleen run optimized inference on AMD Instinct™ GPUs under
AMD Life Science 26.08. It showed how the ROCm overlay makes this work without forking a single
bundle: channels-last 3D, BF16, and `torch.compile()`, applied consistently and layered on top of
each bundle's own configuration. All five pass their upstream Dice thresholds, so the speed comes
without an accuracy trade-off.

In practice, you can run these five existing Model Zoo bundles on an Instinct GPU
without a rewrite. Keep the upstream bundle, add the overlay configuration, and measure on your own
hardware.

This is an Early Access release focused on inference, and it is one layer of the AMD Life Science
medical-imaging stack. Companion posts cover [the hipCIM I/O library underneath it](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-26-08/README.html) and
[MONAI data loading](https://rocm.blogs.amd.com/artificial-intelligence/monai-hipcim-2608/README.html), with interactive annotation still to come. Work is continuing on training validation across all
five bundles, on additional Model Zoo bundles, and on MIOpen 3D kernel work for the most
convolution-heavy architectures, wholeBody_ct and spleen_deepedit, using AMD-optimized MIOpen
Composable Kernel (CK) 3D kernels. Follow the AMD ROCm Blogs for the rest of the series.

## Additional Resources

- [hipCIM 26.06: GPU-Accelerated Whole-Slide Image I/O on AMD Instinct™ GPUs](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-26-08/README.html): the image I/O layer beneath this work.
- [Accelerating MONAI Digital-Pathology Pipelines on AMD Instinct™ GPUs with hipCIM](https://rocm.blogs.amd.com/artificial-intelligence/monai-hipcim-2608/README.html): hipCIM as the MONAI `WSIReader` backend.
- [Project MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo): the upstream bundle collection, licensed under the Apache License 2.0.
- [MONAI bundle configuration documentation](https://docs.monai.io/en/stable/config_syntax.html): the configuration and overlay mechanism used here.
- [MONAI Model Zoo on GitHub](https://github.com/AMD-Ecosystem/monai-model-zoo)
- [MONAI on GitHub](https://github.com/AMD-Ecosystem/MONAI)
- [AMD ROCm documentation](https://rocm.docs.amd.com/)
- [AMD ROCm Blogs](https://rocm.blogs.amd.com/)

## System Configuration

All five bundle latencies were measured on the configuration below, on **21 September 2026**.

| Component | Configuration |
| --- | --- |
| GPU | AMD Instinct™ MI355X (gfx950), bare metal |
| Host CPU | AMD EPYC™ 9575F, 64 cores |
| ROCm | 10.0 |
| PyTorch | 2.13.0+rocm10.0.0 (HIP runtime 7.15.26333) |
| Triton | 3.8.0+rocm10.0.0 |
| MONAI | 1.6.0 |
| NumPy | 2.4.6 |
| Python | 3.12 |
| PyTorch path | `torch.compile()`, BF16, channels-last 3D |
| Inferer | MONAI `SlidingWindowInferer` |
| Timing | 5 warmup iterations, then 20 timed iterations, reported as p50 |
| Input | BTCV spleen CT volume, `spleen_10.nii.gz`, for all bundles |
| Correctness | Dice against the upstream MONAI Model Zoo thresholds |

*Table 3. Hardware, software, and timing method behind the latency numbers.*

The operating system, amdgpu driver version, and container image digest were not recorded for this
run, so they are omitted rather than inferred. Testing conducted by AMD.

## Disclaimers

Performance may vary by hardware configuration, workload, and software version.

## Third-Party Content Disclaimer

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

- MONAI and the MONAI Model Zoo bundles are © Project MONAI, licensed under the Apache License 2.0.
- The spleen CT volume `spleen_10.nii.gz` is from the Medical Segmentation Decathlon (Task09 Spleen), licensed CC-BY-SA 4.0.

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
