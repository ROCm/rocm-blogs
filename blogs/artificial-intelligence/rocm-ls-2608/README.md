---
blogpost: true
blog_title: "AMD Life Science 26.08: A Complete Medical Imaging Stack on AMD Instinct™ GPUs"
date: "12 Nov 2026"
author: "Soumitra Chatterjee, Vikas C Sajjan, Nilaykumar K Patel, Prateek Chokse, Anik Chaudhuri, Marco Grond, Vish Vadlamani, Phani Vaddadi"
thumbnail: 'hero-umbrella.jpeg'
tags: "AI/ML, Computer Vision, Performance, PyTorch"
category: "Applications & models"
target_audience: "ML engineers, data scientists, and clinical researchers building end-to-end medical imaging pipelines on AMD Instinct GPUs"
key_value_propositions: "AMD Life Science 26.08 provides a complete GPU-accelerated medical imaging stack on AMD Instinct: hipCIM, MONAI 1.6.0, Model Zoo, and MONAILabel, all on ROCm 10.0, keeping the same open-source APIs"
language: English
myst:
    html_meta:
        "author": "Soumitra Chatterjee, Vikas C Sajjan, Nilaykumar K Patel, Prateek Chokse, Anik Chaudhuri, Marco Grond, Vish Vadlamani, Phani Vaddadi"
        "description lang=en": "Explore the AMD Life Science 26.08 medical-imaging stack on AMD Instinct™ GPUs, from image I/O through inference to annotation."
        "keywords": "AMD Life Science, AMD Instinct, medical imaging, hipCIM, MONAI, MONAILabel, Model Zoo, digital pathology, ROCm 10.0, whole-slide imaging"
        "vertical": "AI, Developers, Data Science"
        "amd_category": "Developers"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Inference, Computer Vision"
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

# AMD Life Science 26.08: A Complete Medical Imaging Stack on AMD Instinct™ GPUs

Deep learning and medical imaging are reshaping healthcare,
from tumor detection in whole-slide pathology to whole-body segmentation in computed tomography (CT). But a medical
imaging workflow isn't a single step. A slide or a scan has to be read off disk and decoded, fed
through a model, and often handed to a clinician to review and refine. To serve this work well, AMD Instinct™ has to accelerate every step, not just one of them.

That is what [**AMD Life Science 26.08**](https://rocm.docs.amd.com/projects/lifescience/en/latest/)
delivers. Built on [**ROCm™ 10.0**](https://rocm.docs.amd.com/en/latest/), this release assembles a
complete, GPU-accelerated medical imaging stack that reads gigapixel images, runs
validated model inference, and drives interactive annotation, all on AMD Instinct™ GPUs. It does this
with the same open-source tools practitioners already use.

In this blog you will explore how AMD Life Science 26.08 takes a medical image from disk to a
clinician's annotation on AMD Instinct™ GPUs, learn what each layer of the stack contributes, and
find out how to try it yourself.

This post is the capstone of a five-part series on the AMD Life Science 26.08 medical-imaging stack.
The four companion posts, all now published, cover the layers in turn:
[hipCIM 26.06: GPU-Accelerated Whole-Slide Image I/O on AMD Instinct™ GPUs](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-26-08/README.html) for image I/O,
[Accelerating MONAI Digital-Pathology Pipelines on AMD Instinct™ GPUs with hipCIM](https://rocm.blogs.amd.com/artificial-intelligence/monai-hipcim-2608/README.html) for data
loading, [Bringing the MONAI Model Zoo to AMD Instinct™ GPUs: Validated Inference on ROCm](https://rocm.blogs.amd.com/artificial-intelligence/monai-zoo-2608/README.html) for
validated inference, and
[MONAILabel 0.8.5 on AMD Instinct™ GPUs: Interactive Medical-Image Annotation](https://rocm.blogs.amd.com/artificial-intelligence/monailabel-2608/README.html) for interactive
annotation. This post ties them together.

A life science researcher or practitioner can run a whole-slide image or a 3D CT volume through an
end-to-end inference pipeline on
AMD hardware. They keep the same `cucim` import, the same MONAI (Medical Open Network for AI) bundle checkpoints, and the same MONAILabel API. Table 1 lists the four components of the stack.

| Component | Version | Maturity | What it delivers |
| --- | --- | --- | --- |
| **hipCIM** | 26.06.00 | Generally Available (GA) | GPU-accelerated whole-slide & 3D medical image I/O (cuCIM-compatible) |
| **MONAI Core** | 1.6.0 | GA | ROCm-validated 3D medical imaging inference |
| **MONAI Model Zoo** | 26.08 | Early Access (EA) | Validated, optimized inference for 5 bundles |
| **MONAILabel** | 0.8.5 | EA | Interactive active-learning annotation server |

*Table 1. The components of the AMD Life Science 26.08 medical-imaging stack.*

## The Numbers That Tell the Story

Each of these results, shown in Table 2 below, is drawn from a companion blog that carries the full
system under test (SUT) and methodology. Taken together, they describe a complete pipeline getting
faster at every stage on AMD Instinct™:

| Result | AMD Instinct™ measured | vs baseline | SUT |
| --- | --- | --- | --- |
| **Whole-slide image (WSI) GPU patch read throughput** (hipCIM) | 9,661–9,930 p/s | **~10×** over OpenSlide (~950 p/s) | MI355X with ROCm 10.0 |
| **MONAI digital-pathology data loading** (hipCIM vs OpenSlide CPU) | ~2,750 p/s | **~4.2×** over OpenSlide | MI355X with ROCm 10.0 |
| **End-to-end pathology inference** (3 slides, `num_workers=8`) | ~0.13 s/batch (stable) | **1.6–3.8×** over OpenSlide CPU | MI355X with ROCm 10.0 |
| **MONAI Model Zoo** (5 bundles, PyTorch) | 3.60–3,287 ms | validated, all Dice PASS | MI355X with ROCm 10.0 |
| **MONAILabel model inference** | **131×** (like-for-like fp32) | median of 4 spleen CT volumes | MI355X with ROCm 10.0 |

*Table 2. Headline results. Each is measured and detailed in its companion post, on the SUT shown.*

The rest of this post walks the stack from the layer that gets pixels onto the
GPU, through the model, to the clinician.

## hipCIM 26.06.00: Getting Pixels onto the GPU

A significant amount of time in digital-pathology pipelines is spent
reading gigapixel slides into tiles, and if that step is slow, the GPU waits.
[hipCIM](https://github.com/AMD-Ecosystem/hipCIM) brings GPU-accelerated digital-pathology I/O to AMD
Instinct™. It keeps the same `from cucim import CuImage` API, so existing pipelines move across
with no code changes, and it turns "a slide on disk" into GPU-resident tensors.

On AMD Instinct™ MI355X with ROCm 10.0, hipCIM reads whole-slide images **~10× faster than OpenSlide**
on the GPU path (9,661–9,930 patches/s vs ~950 p/s). The CPU path reaches 7–11× on the same node.
hipCIM also widens
format coverage dramatically. The OME-TIFF (Open Microscopy Environment TIFF) corpus goes from 0.8% to 90.3% and adds 3D NIfTI and
DICOM reading through the identical `cucim.CuImage` API. Just as important, hipCIM 26.06.00 adds
GPU decode of real JPEG-compressed Aperio SVS slides, the format written by Leica Aperio scanners.
The [hipCIM companion post](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-26-08/README.html) covers the full story, including where the CPU path still wins.

## MONAI 1.6.0 + hipCIM: The Win, Carried End to End

A fast reader only matters if its speed reaches the model. Using hipCIM as MONAI's `WSIReader`
backend improves run time. On the canonical `pathology_tumor_detection` workload on MI355X with ROCm 10.0,
MONAI data loading runs **~4.2×** faster than MONAI's stock OpenSlide reader (~2,750 p/s vs ~660 p/s).
End-to-end, across three whole-slide images at `num_workers=8`, the pipeline runs **1.6–3.8× faster**
with hipCIM. hipCIM's read cost holds steady at ~0.13 s/batch
while the OpenSlide baseline varies by slide, from ~0.21 to ~0.51 s/batch. On the Aperio SVS slides, the decoded output stays faithful to
the CPU reference (max absolute difference of 1 on 8-bit values, the off-by-one that rounding
produces). The [MONAI + hipCIM companion post](https://rocm.blogs.amd.com/artificial-intelligence/monai-hipcim-2608/README.html) walks through each stage, including a
prototype that keeps tiles GPU-resident for even more.

## MONAI Model Zoo: Validated Inference on ROCm

With a fast data path in place, attention turns to the models. The [MONAI Model Zoo](https://github.com/AMD-Ecosystem/monai-model-zoo), the community's shelf of
ready-to-use models, comes to AMD Instinct™ with five bundles validated
for inference. The optimization ships as an **`inference_rocm.json`** / **`inference_rocm.yaml`**
overlay passed alongside the stock bundle configuration, so it applies the AMD ROCm optimizations without
modifying the upstream bundle, leaving other runtimes untouched.

The inference runs were measured on AMD Instinct™ MI355X (gfx950) with ROCm 10.0 and PyTorch 2.13.0+rocm10.0.0, using
`torch.compile` and BF16 (16-bit brain floating point) precision. These runs use standard PyTorch
with no offline engine-compilation step, and a single spleen CT volume (`spleen_10.nii.gz`) as input:

| Bundle | MI355X (gfx950) inference (ms, p50) | Dice / correctness |
| --- | --- | --- |
| Vista3D | 268 | PASS |
| SwinUNETR (BTCV) | 293 | PASS |
| wholeBody_ct | 386 | PASS |
| pancreas_ct_dints | 3,287 | PASS |
| spleen_deepedit | 3.60 | PASS |

*Table 3. Model Zoo inference latency (p50, milliseconds) and correctness on AMD Instinct™ MI355X.*

All five clear their upstream Dice correctness thresholds. Dice measures how closely a predicted
segmentation overlaps the reference one, on a scale from 0 to 1, so clearing the upstream threshold
means the AMD results match what the bundle authors expect. The
[Model Zoo companion post](https://rocm.blogs.amd.com/artificial-intelligence/monai-zoo-2608/README.html) covers the three-technique overlay and why the large models benefit
from the large HBM3e memory capacity.

## MONAILabel 0.8.5: Closing the Loop with the Clinician

[MONAILabel](https://github.com/AMD-Ecosystem/MONAILabel), the open-source active-learning annotation server,
now runs on AMD Instinct™. This is a framework-transparent port: three files changed, with no API
changes and no new GPU kernels. The payoff for annotators is immediate. On AMD Instinct™ MI355X with
ROCm 10.0, model inference runs **131× faster than the same model on a host CPU at the same fp32
precision** (median across four spleen CT volumes, range 115–133×). That is fast enough (~0.05–0.12 s
per volume, bf16 as deployed) that a warm annotation request is now dominated by pre-processing and
I/O rather than the model. 57 of 57 offline tests pass, with CPU–GPU parity to **~4.6×10⁻⁶**. A
companion post has the install steps and the full breakdown.

## Why This Stack, and Why Now

Read the four companion blogs together and a theme emerges: this is not four separate ports, but one
coherent stack where each layer makes the next one faster. hipCIM gets pixels onto the GPU; MONAI
keeps them there through the model; the Model Zoo delivers validated inference on top; and MONAILabel
closes the loop with the clinician. The large HBM3 and HBM3e memory on AMD Instinct™ runs through all
of it. It's what allows large volumes and slide batches to stay resident on the GPU rather than
shuttling to host memory and back.

## Get Started

- **hipCIM:** `pip install amd-hipcim --extra-index-url https://pypi.amd.com/rocm-10.0.0/simple/`
- **MONAI / Model Zoo / MONAILabel:** available from the AMD ROCm PyPI index and ROCm PyTorch Docker
  base images.
- **Repositories:** [hipCIM](https://github.com/AMD-Ecosystem/hipCIM),
  [MONAI](https://github.com/AMD-Ecosystem/MONAI),
  [monai-model-zoo](https://github.com/AMD-Ecosystem/monai-model-zoo), and
  [MONAILabel](https://github.com/AMD-Ecosystem/MONAILabel) under `AMD-Ecosystem`.
- **Documentation:** [AMD ROCm documentation](https://rocm.docs.amd.com/) and
  [AMD ROCm Blogs](https://rocm.blogs.amd.com/).

Researchers, clinicians, and developers are invited to bring their medical imaging workloads to AMD
Instinct™ and explore the stack. Questions, results and requests for what to support next are
welcome as comments on this post or as issues on the repositories linked above.

## Summary

In this blog, you explored the AMD Life Science 26.08 medical-imaging stack end to end: four layers
that together take a slide from disk to a clinician's annotation on AMD Instinct™ GPUs. hipCIM
26.06.00 reads whole-slide images about 10 times faster than OpenSlide on the GPU path. MONAI 1.6.0
carries that speed into the model, with data loading about 4.2 times faster than the stock OpenSlide
reader and end-to-end pathology inference 1.6 to 3.8 times faster. Five MONAI Model Zoo bundles are
validated for inference, all clearing their upstream Dice thresholds. MONAILabel 0.8.5 closes the loop
with the clinician, running model inference 131 times faster than a host CPU at the same fp32
precision.

The reason this matters for your work is continuity: each layer uses the same open-source APIs
practitioners already know, so most workloads move across without code changes. You are not adopting
an AMD fork of the medical-imaging ecosystem; you are running the ecosystem, on AMD hardware.

This post closes the five-part series on the stack. Each layer has its own companion post, mapped to
its measurement in Table 4, with the full methodology, the caveats, and the install steps:
[hipCIM 26.06: GPU-Accelerated Whole-Slide Image I/O on AMD Instinct™ GPUs](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-26-08/README.html),
[Accelerating MONAI Digital-Pathology Pipelines on AMD Instinct™ GPUs with hipCIM](https://rocm.blogs.amd.com/artificial-intelligence/monai-hipcim-2608/README.html),
[Bringing the MONAI Model Zoo to AMD Instinct™ GPUs: Validated Inference on ROCm](https://rocm.blogs.amd.com/artificial-intelligence/monai-zoo-2608/README.html), and
[MONAILabel 0.8.5 on AMD Instinct™ GPUs: Interactive Medical-Image Annotation](https://rocm.blogs.amd.com/artificial-intelligence/monailabel-2608/README.html).
Updates will keep being published as the genomics stack, training support, and the GPU-resident radiology
path mature. Follow the AMD ROCm Blogs for what comes next, and share what you need.

## Additional Resources

- [hipCIM 26.06: GPU-Accelerated Whole-Slide Image I/O on AMD Instinct™ GPUs](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-26-08/README.html): the image I/O layer beneath this work.
- [Accelerating MONAI Digital-Pathology Pipelines on AMD Instinct™ GPUs with hipCIM](https://rocm.blogs.amd.com/artificial-intelligence/monai-hipcim-2608/README.html): hipCIM as the MONAI `WSIReader` backend.
- [Bringing the MONAI Model Zoo to AMD Instinct™ GPUs: Validated Inference on ROCm](https://rocm.blogs.amd.com/artificial-intelligence/monai-zoo-2608/README.html): the validated inference bundles that run on top of this stack.
- [hipCIM on GitHub](https://github.com/AMD-Ecosystem/hipCIM)
- [MONAI on GitHub](https://github.com/AMD-Ecosystem/MONAI)
- [MONAI Model Zoo on GitHub](https://github.com/AMD-Ecosystem/monai-model-zoo)
- [MONAILabel on GitHub](https://github.com/AMD-Ecosystem/MONAILabel)
- [Project MONAI](https://github.com/Project-MONAI/MONAI): the upstream project, licensed under the Apache License 2.0.
- [AMD ROCm documentation](https://rocm.docs.amd.com/)
- [AMD ROCm Blogs](https://rocm.blogs.amd.com/)

## System Configuration

Each benchmark result presented in this post is measured, and documented in full in its companion
post. All of them run
on **AMD Instinct™ MI355X (gfx950), bare metal, with ROCm 10.0**. The host CPU differs by workload,
because the measurements were taken on different machines, so it is stated per result rather than
once for the whole post.

| Result | Measured on | Full configuration |
| --- | --- | --- |
| hipCIM WSI patch read | MI355X, ROCm 10.0, hipCIM 26.06.00, rocJPEG VA-API decode | [hipCIM companion post](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-26-08/README.html) |
| MONAI loader and end-to-end inference | MI355X, ROCm 10.0, PyTorch 2.13.0+rocm10.0.0, MONAI 1.6.0; host AMD Ryzen™ Threadripper™ PRO 7955WX | [MONAI + hipCIM companion post](https://rocm.blogs.amd.com/artificial-intelligence/monai-hipcim-2608/README.html) |
| MONAI Model Zoo bundle latencies | MI355X, ROCm 10.0, PyTorch 2.13.0+rocm10.0.0, MONAI 1.6.0, input `spleen_10.nii.gz`; host AMD EPYC™ 9575F | [Model Zoo companion post](https://rocm.blogs.amd.com/artificial-intelligence/monai-zoo-2608/README.html) |
| MONAILabel model inference | MI355X, ROCm 10.0, MONAILabel 0.8.5; CPU baseline AMD Ryzen™ Threadripper™ PRO 7955WX, 32 threads | MONAILabel companion post |

*Table 4. Where each result was measured. The companion posts carry the full configuration, the
benchmark method and the caveats.*

Operating system, amdgpu driver version, and container image digests were not recorded for these
runs, so they are omitted rather than inferred.

## Disclaimers

Performance may vary by hardware configuration, workload, and software version.

## Third-Party Content Disclaimer

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
