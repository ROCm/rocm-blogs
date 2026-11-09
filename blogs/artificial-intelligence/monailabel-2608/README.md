---
blogpost: true
blog_title: "MONAILabel 0.8.5 on AMD Instinct™ GPUs: Interactive Medical-Image Annotation"
date: "09 Nov 2026"
author: "Soumitra Chatterjee, Vikas C Sajjan, Nilaykumar K Patel, Prateek Chokse, Anik Chaudhuri, Marco Grond, Vish Vadlamani, Phani Vaddadi"
thumbnail: 'hero-monailabel.jpeg'
tags: "AI/ML, Computer Vision, PyTorch"
category: "Applications & models"
target_audience: "Clinical annotation teams, radiologists, and ML researchers bringing interactive active-learning annotation workflows to AMD Instinct GPUs"
key_value_propositions: "MONAILabel 0.8.5 on AMD Instinct™ - 131× faster model inference (like-for-like fp32) via a minimal 3-file, zero-API-change port; all 57 offline tests pass"
language: English
myst:
    html_meta:
        "author": "Soumitra Chatterjee, Vikas C Sajjan, Nilaykumar K Patel, Prateek Chokse, Anik Chaudhuri, Marco Grond, Vish Vadlamani, Phani Vaddadi"
        "description lang=en": "Learn how to run MONAILabel 0.8.5 for interactive medical-image annotation on AMD Instinct™ GPUs."
        "keywords": "MONAILabel, AMD Instinct, ROCm, medical imaging, annotation, active learning, 3D Slicer, interactive segmentation, spleen, CT"
        "vertical": "AI, Developers, Data Science"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Inference, AI Training, Computer Vision"
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

# MONAILabel 0.8.5 on AMD Instinct™ GPUs: Interactive Medical-Image Annotation

Behind every well-trained medical imaging model is a painstakingly labeled dataset, and behind that
dataset is a human expert drawing contours one slice at a time. MONAILabel exists to make that human
faster. Developed by Project MONAI (Medical Open Network for AI), it is a widely used open-source active-learning annotation server
for medical imaging. It bridges 3D Slicer, the Open Health Imaging Foundation (OHIF) Viewer, and QuPath to
MONAI-powered deep learning models, so radiologists and researchers can label 3D computed tomography (CT) and magnetic resonance imaging (MRI) volumes and
whole-slide images interactively, with the model suggesting and the human refining.

With [**AMD Life Science 26.08**](https://rocm.docs.amd.com/projects/lifescience/en/latest/), which
supports **ROCm™ 10.0**, **MONAILabel 0.8.5** comes to AMD Instinct™ GPUs as an Early Access release.

In this blog you will learn how the port works, explore its performance on a real spleen CT
segmentation workload, and see how you can try it yourself. This post is part of a series on the
AMD Life Science medical-imaging stack. The companion posts cover
[the hipCIM image-I/O library](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-26-08/README.html), [MONAI data loading](https://rocm.blogs.amd.com/artificial-intelligence/monai-hipcim-2608/README.html), and
[validated Model Zoo inference](https://rocm.blogs.amd.com/artificial-intelligence/monai-zoo-2608/README.html).

## At a Glance

The table below summarizes the headline results, all measured on AMD Instinct™ MI355X with
ROCm 10.0.

| Metric | Result | Hardware | Baseline / workload |
| --- | --- | --- | --- |
| Model inference vs CPU (like-for-like, fp32) | **131×** (median; range 115–133×) | MI355X with ROCm 10.0 | host CPU, fp32 |
| Per-volume inference (as deployed, bf16) | **~0.05–0.12 s** | MI355X with ROCm 10.0 | 4 spleen CT volumes |
| Single 96³ patch inference, GPU | **~5.5–7.8 ms** | MI355X with ROCm 10.0 | — |
| CPU–GPU numerical parity | maximum absolute error ~4.6×10⁻⁶ | MI355X with ROCm 10.0 | float32; tolerance 5×10⁻² |

*Table 1. Summary of the results in this post.*

## Why This Matters for Annotation

Annotation is a conversation. Each cycle runs
the same loop: load the volume, run inference to propose a segmentation, let the annotator correct it,
and feed the correction back. For that loop to stay productive, every inference has to return in
seconds. On a CPU, sliding-window inference over a full CT volume can drag on for minutes, and the
annotator sits waiting. On an AMD Instinct™ GPU, the same pipeline finishes in seconds, and the
conversation keeps flowing.

With AMD Life Science, teams get this speedup without changing
their workflows. The port is framework-transparent: all GPU compute runs through PyTorch-ROCm,
where HIP presents itself as `torch.cuda.*`. Existing model configs, sample apps, and MONAI bundles
run without modification. The annotator's experience hasn't changed; it's only been made faster.

## A Deliberately Small Port

One of the strengths of this port is its small footprint. The entire functional
ROCm enablement lives in **three files**, listed in Table 2 below, with no changes to the Python API and no new GPU kernels:

| File | Change |
| --- | --- |
| `monailabel/utils/others/generic.py` | `gpu_memory_map()` gains a `rocm-smi --showmeminfo vram --csv` fallback, so AMD VRAM is monitored correctly when `nvidia-smi` is absent |
| `monailabel/endpoints/logs.py` | the `/gpu` REST endpoint becomes backend-agnostic, dispatching to whichever GPU tool is present, `nvidia-smi` or `rocm-smi` |
| `Dockerfile` | documents the AMD ROCm base image and AMD PyPI wheel index alongside the existing CUDA instructions |

*Table 2. The three files changed by the port.*

Because the port works at the framework level rather than by rewriting GPU
kernels, every future upstream MONAI update inherits AMD GPU support automatically. There is no fork
to maintain and no kernel to keep in sync.

## Performance on AMD Instinct™ MI355X

The headline inference figures below come from the configuration in
[System Configuration](#system-configuration). Model inference is timed over four spleen CT volumes
from the Medical Segmentation Decathlon (Task09 Spleen), and the CPU baseline is the same model at
the same fp32 precision on the host CPU, an AMD Ryzen™ Threadripper™ PRO 7955WX with **32 threads
pinned**. CPU–GPU parity is checked against the CPU output at fp32 (tolerance 5×10⁻²). The offline
unit suite (57/57 PASS) was run on the same MONAILabel 0.8.5 wheel.

### The Core Operation: Model Inference

Every annotation cycle runs the same segmentation model over the volume the annotator is working on,
so model-inference latency is what the annotator experiences. On MI355X with ROCm 10.0, that inference
runs **131× faster than the same model on a host CPU at the same fp32 precision** (median across
four spleen CT volumes from the Medical Segmentation Decathlon Task09; range 115–133×). As deployed, with bf16
autocast, each volume's inference completes in roughly **0.05–0.12 s**. At fp32, it completes in about 0.13–0.28 s.

The smallest unit of that work, a single 96³-voxel sliding-window patch through the
BasicUNet-3D backbone (5.7M parameters), runs in single-digit milliseconds on MI355X. The range of
**~5.5–7.8 ms** brackets three independent measurement runs of the same patch rather than variation
within one run.

The headline figure is the like-for-like fp32 **volume** ratio rather than a **patch-level** speedup,
and that is deliberate. The two behave differently. A patch-level ratio depends entirely on which CPU
baseline you pick: on one machine it moves from 178× to 356× by changing the thread count alone. The
volume-level, same-precision ratio does not move that way, which is why 131× is the number quoted
here and why the baseline part and thread count are stated with it.

Speed doesn't come at the cost of correctness: the maximum absolute error between CPU and GPU outputs
is **~4.6×10⁻⁶**. This is within float32 accumulation rounding and far tighter than the 5×10⁻²
tolerance, so parity passes.

### What the Annotator Waits For

Unlike a single inference, a complete annotation request loads and resamples
the volume, runs sliding-window inference, and writes the result back. With model inference now taking
a fraction of a second, that step is no longer the bottleneck. Across the spleen volumes,
**pre-processing and file I/O account for 94–99% of a warm request**, meaning one where the server
and model are already loaded so no start-up cost is counted, and GPU inference is the small
remainder. The GPU has, in effect, made model inference free within the request. What is left is data
movement, the cost of which scales with volume size and host page-cache state rather than with the GPU. The
practical takeaway for an annotation team is simple: the model is no longer what you wait on, and the
interactive loop stays responsive. Every request produces a valid segmentation mask entirely on the GPU
(`device: cuda` throughout).

## MONAILabel and hipCIM Together

MONAILabel doesn't need [hipCIM](https://rocm.docs.amd.com/projects/hipCIM/en/docs-26.08/), AMD's GPU-accelerated image I/O library, to run. However, MONAILabel runs faster when hipCIM is present. hipCIM plugs
into the radiology sample app through MONAI's optional cuCIM integration. When both `cucim` (provided
by hipCIM) and CuPy are installed, MONAI moves the connected-component post-processing step from CPU to
GPU. With `amd-hipcim` installed, MONAI reports `has_cucim=True`. Adding `amd-cupy` then
activates the GPU connected-component path, which takes the post-processing and connected-component stage off the CPU.
hipCIM's GPU image I/O is also available independently of CuPy for other stages of the pipeline. This is a
preview of how the AMD Life Science components reinforce one another.

## Getting Started

MONAILabel on ROCm needs an AMD Instinct™ GPU with ROCm 10.0 for the 26.08 release, Python 3.12,
and a PyTorch-ROCm wheel. The recommended setup pulls ROCm-validated packages from the [AMD ROCm package index](https://pypi.amd.com/rocm-10.0.0/simple/),
or starts from a [ROCm PyTorch Docker image](https://hub.docker.com/r/rocm/pytorch):

```bash
# From the AMD PyPI index (ROCm-validated packages)
pip install monailabel --extra-index-url https://pypi.amd.com/rocm-10.0.0/simple/
```

```bash
# Start the radiology sample app and run spleen segmentation
monailabel start_server --app radiology --studies <dataset_dir> --conf models segmentation_spleen
```

The server preloads the model onto the AMD GPU at startup, and `device: cuda` in the server log
confirms GPU execution. Because the HIP/ROCm runtime presents as CUDA, no client-side changes are
needed at all.

## Summary

In this blog you explored how MONAILabel 0.8.5 brings the full active-learning annotation workflow to
AMD Instinct™ GPUs through a minimal, framework-transparent port: three files changed, zero API
changes, and zero new GPU kernels. Model inference ran **131× faster than a host CPU at the same fp32
precision** (median across four spleen CT volumes on MI355X with ROCm 10.0), fast enough that a warm
annotation request is now dominated by pre-processing and I/O rather than the model. You also saw how
pairing MONAILabel with hipCIM moves connected-component post-processing onto the GPU.

For annotation teams, the practical benefit is that the migration from a CUDA environment is close to
transparent: the same workflow, the same APIs, the same sample apps, running on Instinct hardware. If you
already run MONAILabel, there is no new interface to learn.

As an Early Access release, this is a beginning rather than a destination. Future releases will broaden
validated coverage and features, and companion posts cover the rest of the AMD Life Science
medical-imaging stack: [hipCIM for image I/O](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-26-08/README.html), [MONAI data loading](https://rocm.blogs.amd.com/artificial-intelligence/monai-hipcim-2608/README.html), and
[validated Model Zoo inference](https://rocm.blogs.amd.com/artificial-intelligence/monai-zoo-2608/README.html).
Follow the AMD ROCm Blogs for the rest of the series. Annotation teams are invited to try this release
and to say what they need next.

## Additional Resources

- [hipCIM 26.06: GPU-Accelerated Whole-Slide Image I/O on AMD Instinct™ GPUs](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-26-08/README.html): the image I/O layer beneath this work.
- [Accelerating MONAI Digital-Pathology Pipelines on AMD Instinct™ GPUs with hipCIM](https://rocm.blogs.amd.com/artificial-intelligence/monai-hipcim-2608/README.html): hipCIM as the MONAI `WSIReader` backend.
- [Bringing the MONAI Model Zoo to AMD Instinct™ GPUs: Validated Inference on ROCm](https://rocm.blogs.amd.com/artificial-intelligence/monai-zoo-2608/README.html): the validated inference bundles that run on top of this stack.
- [MONAILabel on GitHub](https://github.com/AMD-Ecosystem/MONAILabel)
- [MONAI on GitHub](https://github.com/AMD-Ecosystem/MONAI)
- [hipCIM on GitHub](https://github.com/AMD-Ecosystem/hipCIM)
- [Project MONAILabel documentation](https://docs.monai.io/projects/label/en/latest/): upstream MONAILabel docs, licensed under the Apache License 2.0.
- [AMD ROCm documentation](https://rocm.docs.amd.com/)
- [AMD ROCm Blogs](https://rocm.blogs.amd.com/)

## System Configuration

All figures in this post were measured on the configuration below, on **10 September 2026**.

| Component | Configuration |
| --- | --- |
| GPU | AMD Instinct™ MI355X (gfx950), bare metal |
| Host CPU (the baseline) | AMD Ryzen™ Threadripper™ PRO 7955WX, 32 threads pinned |
| ROCm | 10.0 |
| PyTorch | 2.13.0+rocm10.0.0 (HIP runtime 7.15.26333) |
| MONAI | 1.6.0 |
| MONAILabel | 0.8.5 |
| Python | 3.12 |
| Wheels | AMD internal ROCm CI builds, which produce the released artifacts |

*Table 3. Hardware and software behind the figures in this post.*

The operating system, amdgpu driver version, and container image digest were not recorded for this
run, so they are omitted rather than inferred.

**Measurements.** Model-inference speedup (131× median, range 115–133×), per-volume inference
latency (~0.05–0.12 s bf16 as deployed, ~0.13–0.28 s fp32), single-patch latency (~5.5–7.8 ms) and
CPU–GPU parity (~4.6×10⁻⁶, tolerance 5×10⁻²) were measured over four spleen CT volumes from the Medical
Segmentation Decathlon (Task09 Spleen). The like-for-like speedup compares the same model at fp32 on
the GPU against the host CPU above; the baseline part and thread count are what set the ratio.
Offline unit tests (57/57 PASS) were run on the MONAILabel 0.8.5 wheel. Testing conducted by AMD.

## Disclaimers

Performance may vary by hardware configuration, workload, and software version. Not all features validated in this release (FP16, `torch.distributed`, pathology app). AMD Instinct™ is a trademark of
Advanced Micro Devices, Inc.

## Third-Party Content Disclaimer

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

- MONAILabel and MONAI are © Project MONAI, licensed under the Apache License 2.0.
- 3D Slicer, the OHIF Viewer, and QuPath are third-party client applications named as supported integrations; each is governed by its own license (3D Slicer: BSD-style; OHIF: MIT; QuPath: GPL v3).
- Spleen CT volumes are from the Medical Segmentation Decathlon (Task09 Spleen), licensed CC-BY-SA 4.0.

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
