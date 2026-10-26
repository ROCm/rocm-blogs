---
blogpost: true
blog_title: "Accelerating MONAI Digital-Pathology Pipelines on AMD Instinct™ GPUs with hipCIM"
date: "26 Oct 2026"
author: "Soumitra Chatterjee, Vikas C Sajjan, Nilaykumar K Patel, Prateek Chokse, Anik Chaudhuri, Marco Grond, Vish Vadlamani, Phani Vaddadi"
thumbnail: 'hero-monai-hipcim.jpeg'
tags: "AI/ML, Computer Vision, Performance, PyTorch"
category: "Applications & models"
target_audience: "ML engineers and digital-pathology researchers integrating GPU-accelerated WSI reading into MONAI pipelines on AMD Instinct GPUs"
key_value_propositions: "Drop hipCIM into MONAI's WSIReader and get ~4.2× faster data loading on CMU-1.svs and 1.6-3.8× faster end-to-end inference - no model changes, same cucim API"
language: English
myst:
    html_meta:
        "author": "Soumitra Chatterjee, Vikas C Sajjan, Nilaykumar K Patel, Prateek Chokse, Anik Chaudhuri, Marco Grond, Vish Vadlamani, Phani Vaddadi"
        "description lang=en": "In this blog you will measure a MONAI digital-pathology pipeline on AMD Instinct™ GPUs using hipCIM as the WSIReader backend."
        "keywords": "MONAI, hipCIM, AMD Instinct, ROCm, digital pathology, whole-slide imaging, WSI, data loading, cuCIM, pathology pipeline, PatchWSIDataset"
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

# Accelerating MONAI Digital-Pathology Pipelines on AMD Instinct™ GPUs with hipCIM

A digital-pathology model is only as fast as the pipeline that feeds it. That's simple to overlook when
the conversation turns to network architectures and FLOPs, but on real whole-slide workloads the
truth is stubborn: a great deal of the wall-clock time goes to reading gigapixel slides off
disk and decoding them into the small tiles the model consumes. Speed up the model and leave the
reader untouched, and the pipeline barely moves.

A companion post introduces hipCIM, AMD's GPU-accelerated whole-slide image (WSI)
I/O library. This post answers the natural follow-on question: what happens when you drop hipCIM into a
real Medical Open Network for AI (MONAI) pipeline? With **AMD Life Science 26.08**, which supports **ROCm™ 10.0**, MONAI's `WSIReader` can use
hipCIM through its familiar `from cucim import CuImage` API. The gains measured here come from that
reader path, which still returns patches to the host; MONAI's GPU-resident path is not yet wired up. On the canonical MONAI digital-pathology workload, running on AMD Instinct™
MI355X with ROCm 10.0, hipCIM GPU patch reading reaches **~10× over OpenSlide**, the MONAI data
loader runs **~4.2× faster** than MONAI's stock OpenSlide reader on the CMU-1.svs slide (from Carnegie Mellon University), and end-to-end inference across three
whole-slide images is **1.6–3.8× faster**.

## Why the Reader Decides the Pace

Consider the anatomy of the workload. A whole-slide image is a pyramid of small JPEG-compressed tiles,
each a few hundred pixels on a side; a single slide can be tens of thousands of pixels across. A
pathology model samples these tiles by the thousand. The component that turns "a slide on disk"
into "tensors on the accelerator" does an enormous amount of repetitive work, and it sits directly on
the critical path. If it stalls, the GPU idles.

MONAI's `WSIReader` already supports a `cuCIM` backend, and on AMD ROCm that backend is hipCIM,
exposing the identical `from cucim import CuImage` API. The question is a
concrete one: measured on real MI355X hardware, exactly how much does a MONAI pipeline gain when its
reader is hipCIM 26.06.00 rather than the standard CPU reader?

## At a Glance

| Metric | hipCIM 26.06.00 | Vanilla MONAI (OpenSlide CPU) | Speedup | System under test |
| --- | --- | --- | --- | --- |
| WSI GPU patch read (CMU-1/2/3.svs) | 9,661–9,930 patches/s | ~943–951 patches/s | **~10×** | MI355X with ROCm 10.0 |
| MONAI `PatchWSIDataset` loader, CMU-1.svs (eight workers) | ~2,750 patches/s | ~660 patches/s | **~4.2×** | MI355X with ROCm 10.0 |
| End-to-end inference (pathology, `num_workers=8`) | ~0.13 s/batch (stable) | 0.21–0.51 s/batch | **1.6–3.8×** (3 slides) | MI355X with ROCm 10.0 |
| 3-D NIfTI and DICOM volume reading | Supported (GPU-resident via `cumed`) | Not previously available | New capability | Capability, not a timed benchmark |

*Table 1. Summary of the results in this post. Each row is detailed in its own section below.*

Table 1 summarizes the results. The GPU output is faithful to the CPU reference, too: on the Aperio SVS slides benchmarked here,
hipCIM's rocJPEG decode agrees with the CPU libjpeg-turbo path to a maximum absolute difference of 1 on
8-bit values, the kind of off-by-one that rounding, not error, produces. The result is faster and correct.

## What Was Benchmarked

The system under test was an AMD Instinct™ MI355X (gfx950), bare-metal, with
ROCm 10.0, Python 3.12, and a ROCm build of PyTorch. The hipCIM and MONAI wheels came from AMD's
internal ROCm CI, which is what produces the released artifacts: the published equivalents are
[`amd-hipcim`](https://pypi.amd.com/rocm-10.0.0/simple/amd-hipcim/) 26.06.00 and
[`amd-monai`](https://pypi.amd.com/rocm-10.0.0/simple/amd-monai/) 1.6.0 on the AMD ROCm package
index. The MONAI side is AMD's fork of upstream MONAI,
[AMD-Ecosystem/MONAI](https://github.com/AMD-Ecosystem/MONAI), which carries the AMD-specific
changes and feeds them back to [Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI) as
upstream pull requests. Patch-read, loader,
and end-to-end inference throughputs were all measured on that system. The loader comparison was
additionally re-measured on AMD Instinct™ MI350X (gfx950) with ROCm 10.0.

The workload was chosen deliberately. The `pathology_tumor_detection` bundle from the MONAI Model Zoo
(metastasis detection on Camelyon16 H&E, or hematoxylin and eosin, stained whole-slide images) is the one Model Zoo bundle that reads
source WSIs through `WSIReader(backend="cuCIM")` today, via `monai.data.PatchWSIDataset` and
`MaskedPatchWSIDataset`. That makes it the most direct, representative consumer of hipCIM available.
Its network is the bundle's TorchVision ResNet-18 fully-convolutional classifier, run for inference
with the pretrained `model.pt`, and its access pattern (224×224 patches at pyramid level 0, read as
768×768 regions split 3×3) is a loop of many small GPU patch reads typical of WSI training and
inference.

To keep the comparison clean, only one thing changes between variants: the installed hipCIM wheel. A
single Docker image per node means MONAI and PyTorch are byte-identical across every comparison, so
any measured difference is attributable to the reader alone. The WSI micro-benchmark uses Aperio
CMU-1, CMU-2, and CMU-3 `.svs` files (CC0); the loader and end-to-end runs use Camelyon16 tiled-TIFF;
the 3-D coverage test uses MSD spleen CT volumes.

## Benchmark Results

### Patch Read Throughput and Where the Win Originates

Everything downstream traces back to how fast hipCIM can read patches. The GPU
path is roughly an order of magnitude ahead of OpenSlide. Table 2 shows the per-slide figures on MI355X with ROCm 10.0:

| Slide | hipCIM-GPU (patches/s) | OpenSlide (patches/s) | GPU vs OpenSlide (ratio of patches/s) |
| --- | --- | --- | --- |
| CMU-1.svs | 9,930 | 943 | **10.5×** |
| CMU-2.svs | 9,661 | 951 | **10.2×** |
| CMU-3.svs | 9,751 | 951 | **10.3×** |
| **Average** | **9,781** | **~948** | **~10.3×** |

*Table 2. Patch-read throughput in patches per second, all columns measured on MI355X with ROCm 10.0.
The MI355X OpenSlide baseline is ~950 patches/s on this node (bare-metal, ROCm 10.0 manylinux
container); the GPU versus OpenSlide ratio is therefore ~10×, consistent with the headline. The
companion hipCIM post covers CPU path throughput (7–11× over OpenSlide on MI355X with ROCm 10.0).*

There is an important enabler in this table. GPU reads of JPEG-compressed Aperio SVS slides are new
in hipCIM 26.06.00, which is what makes the GPU path functional, correct, and fast on real slides.
That capability is the foundation everything else in this blog stands on.

One clarification before the tables that follow, because it explains an apparent oddity. The SVS GPU
read path is new in 26.06.00. The MONAI loader and end-to-end runs below use the Camelyon16
**tiled-TIFF** WSIs, a format both releases read, which is why 25.10.00 still produces numbers there.
Those loader and end-to-end numbers come out equal between the two releases for a different reason
entirely: both are bottlenecked on MONAI's host-copy path, as explained next, rather than reflecting
equivalent GPU capability.

### From Raw Reads to the MONAI Loader

Raw read throughput is the ceiling. What a user feels is the loader. Table 3 reports the bundle's
exact `PatchWSIDataset` pipeline at eight workers:

| Variant (reader) | MI355X with ROCm 10.0 (patches/s) | MI350X with ROCm 10.0 (patches/s) |
| --- | --- | --- |
| hipCIM 26.06.00 (cuCIM backend) | ~2,750 | ~3,103 |
| Vanilla (OpenSlide) | ~660 | ~671 |
| **hipCIM vs OpenSlide** (ratio) | **~4.2×** | **~4.6×** |

*Table 3. MONAI `PatchWSIDataset` loader throughput. Both columns use the Aperio `CMU-1.svs` slide, 576 patches at level 0, `num_workers=8`, median of five
fresh-process repeats. The ratio depends on the slide: OpenSlide reads tiled-TIFF considerably faster
than Aperio SVS, so quoting a loader ratio without naming the slide is misleading.*

There is a nuance worth surfacing. MONAI's stock `CuCIMWSIReader` requests host output and copies each
patch back to the CPU, which means it does not yet exercise hipCIM's GPU-resident path. That is why
the 26.06.00 and 25.10.00 loader numbers are essentially equal here: both are bottlenecked on the same
host copy. Even so, hipCIM's reader is ~4.2× faster than MONAI's stock OpenSlide reader. Wiring MONAI's
reader to hipCIM's GPU-resident path so tiles never round-trip to the host is the next opportunity, and
it stacks on top of the gain shown here.

### End-to-End Inference

All of this matters only if it moves the number a practitioner cares about: total wall time. It does,
and the size of the win depends on the slide. Across three Camelyon16 slides on MI355X with ROCm 10.0, at
`num_workers=8`, the `pathology_tumor_detection` bundle runs **1.6–3.8× faster** with hipCIM than with
the OpenSlide reader, as Table 4 shows:

| Slide | hipCIM 26.06.00 (s) | Vanilla OpenSlide (s) | hipCIM vs vanilla (ratio) |
| --- | --- | --- | --- |
| test_001 (715 batches) | 96 | 153 | **1.6×** |
| test_004 (986 batches) | 128 | 225 | **1.8×** |
| test_002 (893 batches) | 122 | 459 | **3.8×** |

*Table 4. End-to-end `pathology_tumor_detection` inference, median of three repeats per slide, on
MI355X with ROCm 10.0. Rows are ordered by speedup, not by slide number.*

This post quotes the range rather than a single slide on purpose. Normalized per batch, hipCIM's read cost is
effectively constant, at ~0.13 s/batch on every slide, while the OpenSlide baseline swings from 0.21 to
0.51 s/batch. The ratio moves because the *baseline* moves, not hipCIM: the advantage is a
pipeline-throughput effect, hipCIM scaling cleanly across the eight DataLoader workers where OpenSlide
does not (on the widest slide, OpenSlide is slower at eight workers than single-threaded). A
single `read_region` call, single-threaded, isn't where this win lives. It lives in the parallel loader.
These figures are for inference. Training is far more compute-bound, so data loading is a smaller
slice of the total and the reader matters correspondingly less there. Those distinctions matter, so
they sit alongside the headline rather than behind it.

### 3-D Volumes Through the Same API

| | hipCIM 26.06.00 (`cumed`) | hipCIM 25.10.00 | nibabel (reference) |
| --- | --- | --- | --- |
| Open + read `spleen_*.nii.gz` | Yes, 1.0–1.7 s, GPU-resident | No plugin | 0.7–1.0 s |

*Table 5. 3-D volume reading through the `cumed` reader, compared with the previous hipCIM release
and with nibabel as a CPU reference.*

Beyond speed, 26.06.00 adds a new capability. Its `cumed` reader opens NIfTI and uncompressed DICOM
through the same `cucim.CuImage` API used for whole-slide images. For MONAI, this opens a path to
route 3-D CT and MRI pipelines through hipCIM for GPU-resident volume loading, so a single I/O layer
can eventually serve both the 2-D digital-pathology and 3-D radiology sides of a workflow. (Compressed
DICOM JPEG and JP2K require a build with `-DCUMED_DICOM_COMPRESSED=ON`.)

This breadth is a differentiator: one GPU-resident I/O path, reached through one API, spanning
whole-slide pathology and volumetric radiology on AMD Instinct™.

## How to Adopt hipCIM in Your MONAI Pipeline

**Install the stack.** hipCIM and MONAI on ROCm are installed from the AMD ROCm package index. The
commands are maintained in the official documentation rather than reproduced here, so they stay
current with the release: see the
[MONAI on ROCm installation guide](https://rocm.docs.amd.com/projects/monai/en/docs-26.08/install/installation.html),
which covers PyTorch, hipCIM and MONAI together, and the
[hipCIM documentation](https://rocm.docs.amd.com/projects/hipCIM/en/docs-26.08/). Both are part of
[AMD Life Science](https://rocm.docs.amd.com/projects/lifescience/en/latest/); runnable examples live
in [AMD-Ecosystem/lifescience-examples](https://github.com/AMD-Ecosystem/lifescience-examples).

**Point MONAI at hipCIM.** This is the whole change. MONAI's `WSIReader` already supports a `cuCIM`
backend, and on ROCm that backend is hipCIM, so no model code changes are required:

```python
from monai.data import PatchWSIDataset, WSIReader

# On ROCm, backend="cuCIM" resolves to hipCIM
reader = WSIReader(backend="cuCIM")

dataset = PatchWSIDataset(
    data=[{"image": "CMU-1.svs", "location": [0, 0], "size": [224, 224]}],
    patch_size=224,
    reader=reader,
)
```

In a bundle, the same swap is a configuration override: set the reader on the dataset to
`backend="cuCIM"` and leave the rest of `configs/inference.json` untouched. That is the path the
numbers above were measured on.

**Not available yet.** Two further steps are on the roadmap rather than in your hands today.
Having MONAI's reader request `device="cuda"` so decoded tiles stay GPU-resident is not yet wired
through `CuCIMWSIReader`, which is why the loader figures here come from the host-copy path. Routing
3-D NIfTI and DICOM loading through the `cumed` reader is new in 26.06.00 and not yet integrated into
MONAI's radiology pipelines.

## Summary

This blog showed what happens when hipCIM replaces the stock reader in a MONAI
digital-pathology pipeline: the bottleneck moves off the reader. On AMD Instinct™ MI355X with
ROCm 10.0, hipCIM reads whole-slide patches about 10 times faster than OpenSlide, the MONAI data
loader runs about 4.2 times faster than the stock OpenSlide reader, and end-to-end inference over
three Camelyon16 slides is 1.6 to 3.8 times faster, with hipCIM holding a stable 0.13 seconds per
batch. It also showed where the win doesn't come from, a single `read_region` call, and why a
compute-bound workload has less to gain from a faster reader.

The practical takeaway: because MONAI's `WSIReader` already speaks the
`cuCIM` backend API, adopting hipCIM needs no model changes and no rewrite of your transforms. Point
the reader at the `cuCIM` backend and your existing pipeline runs faster on the same code.

This is one layer of the AMD Life Science medical-imaging stack. Companion posts cover the rest of it:
the hipCIM library underneath, validated Model Zoo inference above it, and
interactive annotation on top. Work is continuing on the 3-D `cumed` radiology path and
GPU-resident loading. Follow the AMD ROCm Blogs for the rest of the series.

## Additional Resources

- [Project MONAI](https://github.com/Project-MONAI/MONAI): `WSIReader`, `PatchWSIDataset`, and the [`pathology_tumor_detection` bundle](https://github.com/Project-MONAI/model-zoo/tree/dev/models/pathology_tumor_detection).
- [OpenSlide test data (Carnegie Mellon University)](https://openslide.cs.cmu.edu/download/openslide-testdata/): CMU-1, CMU-2, and CMU-3 slides, licensed CC0-1.0.
- [hipCIM on GitHub](https://github.com/AMD-Ecosystem/hipCIM)
- [MONAI on GitHub](https://github.com/AMD-Ecosystem/MONAI)
- [AMD ROCm documentation](https://rocm.docs.amd.com/)
- [AMD ROCm Blogs](https://rocm.blogs.amd.com/)

## System Configuration

All performance figures in this post were measured on the configuration below, on **10 September
2026**.

| Component | Configuration |
| --- | --- |
| GPU | AMD Instinct™ MI355X (gfx950), bare metal |
| Host CPU | AMD Ryzen™ Threadripper™ PRO 7955WX, 32 threads pinned |
| ROCm | 10.0 |
| PyTorch | 2.13.0+rocm10.0.0 (HIP runtime 7.15.26333) |
| torchvision / torchaudio | 0.28.0+rocm10.0.0 / 2.11.0.2+rocm10.0.0 |
| MONAI | 1.6.0 |
| hipCIM | 26.06.00 |
| Python | 3.12 |
| Package index | AMD ROCm `whl-next` index |

The amdgpu driver version and the container image digest were not recorded for this run, so they
are omitted rather than inferred. Each variant runs in an isolated container with GPU passthrough
(`--device=/dev/kfd --device=/dev/dri`, render group, `--shm-size=16g`); GPU JPEG decode requires
bare-metal Instinct with a working DRM render node.

**Workloads.** The end-to-end figures use the `pathology_tumor_detection` bundle over three
Camelyon16 slides at `num_workers=8` and `batch_size=400`, three repeats per slide, median reported.
The slides are `test_001`, `test_002` and `test_004` from the Camelyon16 testing set, obtained from
the GigaDB mirror and size-verified against the remote copies. They are a sample of that set rather
than the whole of it, which is why the numbering is not contiguous. The loader comparison was additionally re-measured on
AMD Instinct™ MI350X (gfx950) with ROCm 10.0, using the Aperio `CMU-1.svs` slide, 576 patches at
level 0 and the median of five fresh-process repeats. The hipCIM baseline for comparison is
`amd-hipcim==25.10.0` from the AMD ROCm package index.

## Disclaimers

Performance may vary by hardware configuration, workload, and software version.

## Third-Party Content Disclaimer

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
