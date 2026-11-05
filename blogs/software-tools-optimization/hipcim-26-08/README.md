---
blogpost: true
blog_title: "hipCIM 26.06: GPU-Accelerated Whole-Slide Image I/O on AMD Instinct™ GPUs"
date: "20 Oct 2026"
author: "Soumitra Chatterjee, Vikas C Sajjan, Nilaykumar K Patel, Prateek Chokse, Anik Chaudhuri, Marco Grond, Vish Vadlamani, Phani Vaddadi"
thumbnail: 'hero-hipcim.jpeg'
tags: "AI/ML, Computer Vision, Performance"
category: "Software tools & optimizations"
target_audience: "ML engineers and data scientists building digital pathology or medical imaging pipelines on AMD Instinct GPUs"
key_value_propositions: "hipCIM 26.06 reads whole-slide images at about 10× the OpenSlide throughput in these tests, zero-copy GPU-resident tiles, 90% OME-TIFF coverage, and Aperio SVS GPU decode on ROCm 10.0"
language: English
myst:
    html_meta:
        "author": "Soumitra Chatterjee, Vikas C Sajjan, Nilaykumar K Patel, Prateek Chokse, Anik Chaudhuri, Marco Grond, Vish Vadlamani, Phani Vaddadi"
        "description lang=en": "In this blog you will see how hipCIM 26.06 reads whole-slide images on AMD Instinct™ GPUs, and how fast and faithfully it decodes them."
        "keywords": "hipCIM, whole-slide imaging, WSI, digital pathology, AMD Instinct, ROCm, GPU-accelerated, OpenSlide, OME-TIFF, Aperio SVS, cuCIM"
        "vertical": "AI, Developers, Data Science"
        "amd_category": "Developers"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
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

# hipCIM 26.06: GPU-Accelerated Whole-Slide Image I/O on AMD Instinct™ GPUs

Every whole-slide image tells a story in pixels. A whole-slide image is a glass pathology slide scanned
into a single very large digital image, stored as a pyramid of small tiles so a viewer can pan and zoom
it. A single biopsy digitized at 20× magnification can
span tens of thousands of pixels on each side and weigh in at several gigabytes. Before a model can
learn anything from that slide, something has to read it off disk, decode its thousands of small
JPEG-compressed tiles, and turn them into tensors the accelerator can work on. In digital pathology,
that unglamorous first step is often where the wall-clock time quietly disappears.

This is the problem hipCIM was built to solve. With **AMD Life Science 26.08**, arriving alongside the
**ROCm™ 10.0 launch**, AMD is introducing **hipCIM 26.06.00**, a substantial step forward for
GPU-accelerated whole-slide image (WSI) I/O on AMD Instinct™ GPUs. hipCIM is the AMD port of
cuCIM (RAPIDS) and a drop-in replacement for it: it speaks the same `from cucim import CuImage` API,
so existing image-I/O code moves across unchanged, with no porting required, while decoded pixels land
resident on the AMD GPU, ready for whatever comes next. This release targets ROCm 10.0: the
`amd-hipcim` wheel is published against the ROCm 10.0 package index, so a ROCm 10.0 installation is
what it expects.

If this is your first encounter with the library, an earlier post,
[Announcing hipCIM: A Cutting-Edge Solution for Accelerated Multidimensional Image Processing](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-intro/README.html)
from July 2025 covers the fundamentals: supported formats, the Python and C++ APIs, and the initial
release. This post covers what is new in
26.06.00: how fast it reads, how faithfully it decodes, and the broad set of real-world slide formats
it can now open.

## The Short Version

On AMD Instinct™ MI355X with ROCm 10.0, hipCIM's GPU path reads tiled whole-slide images at about **10× the
OpenSlide throughput**, and its CPU path at 7–11×, both measured on the same node. Speed is only part of
the story. This release also hands back decoded pixels that stay resident in GPU memory, and it hands them
to ROCm array libraries (CuPy, PyTorch-ROCm) with zero-copy interop: the downstream library reads the same
GPU buffer directly, with no host round-trip and no second copy. It keeps GPU decode faithful
to the CPU reference for JPEG-compressed slides, with a maximum absolute difference of 1 on 8-bit
values for Aperio SVS, the slide format written by Leica Aperio scanners and one of the most common in
digital pathology. It widens OME-TIFF (the Open Microscopy Environment's multi-page TIFF format for microscopy images) coverage across the test corpus from 0.8% to 90.3%, and it
hardens the trickier corners of the Aperio SVS and JPEG2000 formats. Table 1 summarizes the results.

| Metric | hipCIM | Baseline | Speedup (ratio) |
| --- | --- | --- | --- |
| WSI GPU patch read, Aperio Carnegie Mellon University (CMU)-1/2/3.svs (MI355X, ROCm 10.0) | 9,661–9,930 p/s | OpenSlide ~950 p/s | **~10×** |
| WSI CPU patch read, Aperio CMU-1/2/3.svs (MI355X, ROCm 10.0) | 6,948–7,440 p/s | OpenSlide ~950 p/s | **7–8×** |
| WSI CPU patch read, GenericTIFF CMU-1 (MI355X, ROCm 10.0) | 10,326 p/s | OpenSlide ~950 p/s | **11×** |
| GPU decode fidelity vs CPU reference, Aperio SVS (JPEG) | max abs diff = 1 (8-bit) | libjpeg-turbo | faithful |
| GPU decode fidelity, GenericTIFF CMU-1 | max abs diff = 32, mean 0.13 | libjpeg-turbo | bounded (98% ±1) |
| OME-TIFF corpus coverage | 90.3% | prior hipCIM 0.8% | new |

*Table 1. Summary of the results in this post. Each row is detailed in the sections below.*

## Why Whole-Slide I/O Sits on the Critical Path

It's tempting to assume that a deep-learning pipeline spends most of its time in the model. In
digital pathology, that assumption often breaks down. A whole-slide image is stored as a pyramid of
small tiles, typically 256×256 pixels, each independently JPEG-compressed, and training or
inference samples these tiles by the thousand. The library responsible for turning "a slide on disk"
into "tensors on the accelerator" therefore sits squarely on the critical path: if it stalls, the
GPU waits.

hipCIM's job is to keep that path clear on AMD Instinct™, and with 26.06.00 it does so on the
ROCm 10.0 platform, the software foundation for AMD Instinct™ MI355X.

## Measuring Against OpenSlide

To characterize I/O performance, the benchmark reads every non-overlapping 256×256 patch from a whole-slide image and compares against OpenSlide, the de-facto CPU reference for WSI reading. This workload is the one the field has standardized on: it's reproducible, runs on any hardware without special provisioning, and gives a transparent, portable way to state throughput.

The test setup ran on an AMD Instinct™ MI355X (gfx950), bare-metal, with ROCm 10.0 and the hipCIM 26.06.00 wheel. The GPU path uses `read_region(..., device="cuda")` with rocJPEG hardware decode; the CPU path uses `read_region(..., device="cpu")` with the libjpeg-turbo backend. For each slide the benchmark reads every non-overlapping 256×256 patch at level 0, a 24×24 grid of 576 patches, with a warm cache, and reports the median of five trials. GPU, CPU, and the OpenSlide baseline were all measured in the same run.

## hipCIM on AMD Instinct™ MI355X

Table 2 reports patch-read throughput in patches per second, where higher is better, across the
standard CMU test slides.

| Slide | hipCIM-GPU (patches/s) | hipCIM-CPU (patches/s) | OpenSlide (patches/s) | GPU vs OpenSlide (ratio) | CPU vs OpenSlide (ratio) |
| --- | --- | --- | --- | --- | --- |
| Aperio CMU-1.svs | 9,930 | 6,948 | 943 | **10.5×** | 7.4× |
| CMU-2.svs | 9,661 | 7,362 | 951 | **10.2×** | 7.7× |
| CMU-3.svs | 9,751 | 7,440 | 951 | **10.3×** | 7.8× |
| GenericTIFF CMU-1 | 9,578 | 10,326 | 943 | **10.2×** | 10.9× |

*Table 2. Patch-read throughput in patches per second. All GPU, CPU, and OpenSlide numbers come from the same MI355X with ROCm 10.0 run (5-trial median). The three
Aperio SVS slides cluster tightly at 9,661–9,930 p/s GPU, a consistent ~10× over OpenSlide.
GenericTIFF shows slightly higher CPU than GPU throughput on this 256×256 small-tile workload; this
is expected, as explained in the next section.*

The CPU path, at 7–11× over OpenSlide on the same MI355X node running ROCm 10.0, remains a meaningful win
for workloads that do not require GPU-resident output.

Speed only counts if the pixels are right. For JPEG-compressed Aperio SVS slides, hipCIM's rocJPEG
hardware decode agrees with libjpeg-turbo to a maximum absolute difference of 1 on 8-bit values, the
kind of off-by-one that rounding, not error, produces. For GenericTIFF, the distribution is tight: 98% of pixels are identical or differ by only ±1, with a mean difference of 0.13. The maximum absolute difference of 32 reflects a thin outlier tail. That bounded difference is reported plainly, and work continues to characterize GPU decode accuracy on the tiled-TIFF path.

## GPU Versus CPU Throughput: What to Expect

A careful reader will notice that hipCIM's GPU and single-thread CPU paths sit much closer to each other than either does to OpenSlide. The GPU path is faster on the three Aperio SVS slides (≈9,900 vs ≈7,000 p/s); on the small-tile GenericTIFF case the CPU edges ahead. This is expected and inherent in the problem. GPU JPEG decode shines when it has a few large images to chew through; a 256×256 tile is tiny, and libjpeg-turbo decodes one with very little per-image overhead, so a CPU thread gets through them quickly. An independent benchmark from the hi-ml project (Microsoft Research) saw the same pattern on a full tiling workload at 224×224 tiles. This closeness on raw patches-per-second is a property of the many-small-tile nature of WSI reading itself, not of any one implementation.

So why report the GPU path at all? Because raw patches-per-second on this micro-benchmark is not where
its advantage lies. That advantage is architectural, and the next section turns to it.

## The Advantage That Matters: Zero-Copy, On-Device Pixels

When you read with `device="cuda"`, hipCIM hands back decoded pixels that are already resident in GPU
memory, exposed through the standard `__cuda_array_interface__`. A downstream GPU consumer, whether CuPy, PyTorch-ROCm, or a model already sitting on the device, can pick them up with a zero-copy handoff, with no host round-trip. The CPU path, by contrast, must copy every tile from host to device before any GPU
compute can begin.

For an end-to-end pipeline that feeds a GPU-resident model, this is the difference between decoded
tiles that never leave the accelerator and tiles that shuttle back and forth across the PCIe bus. On AMD Instinct™ MI355X, with its large HBM3e memory, far more tiles can stay resident at once, which translates
directly into higher sustained throughput for the GPU compute that follows. This is the architecture
hipCIM's GPU path is built around, and its payoff compounds across the rest of the AMD Life Science
medical-imaging stack.

In practice this is a two-line change: read the region on the device, then hand the result to a GPU array library. The example below does exactly that.

```python
from cucim import CuImage
import cupy as cp

img = CuImage("CMU-1.svs")

# Decode a tile straight onto the GPU, with no host round-trip
region = img.read_region(location=[50000, 40000], size=(512, 512), device="cuda")

# Zero-copy into CuPy via __cuda_array_interface__
tile = cp.asarray(region)
```

## Reading More of the Real World

Throughput means nothing if the library can't open the file in the first place, and in practice,
real slide archives are full of format quirks. A major thread of work in 26.06.00 was widening what
hipCIM can read on AMD:

- **Multi-IFD OME-TIFF.** Multi-IFD layouts used to be rejected outright. A single such layout
  triggered an exception that failed the whole file. That's fixed: the complete 90-file OME-TIFF
  reference corpus now loads end to end. The impact on real archives is even more striking. Measured
  across a separate, much larger 2,703-file test set, readable coverage climbed from **0.8% to
  90.3%**, turning a format hipCIM could barely touch into one it reads across most of that set.
- **Aperio SVS abbreviated-JPEG.** SVS tiles store their JPEG quantization and Huffman tables once
  per IFD, a format known as abbreviated JPEG (ITU-T T.81 §B.5). hipCIM now splices those tables
  back into each tile before handing it to rocJPEG so rocJPEG can decode the tile in hardware, and a single corrupt
  tile no longer takes down the whole host process.
- **Aperio JPEG2000 SVS.** The `device="cuda"` path no longer aborts on JP2K slides; it decodes them
  on the CPU and stages the result to the device.
- **Hardened batched GPU reads.** A class of host-process crashes on spatially-scattered batched
  reads, exactly the access pattern random patch samplers use during training, is fixed. The read
  path now gathers tiles per batch, which eliminates whole-slide-span over-reads, works through the page
  cache, and overlaps I/O latency with parallel per-tile reads.
- **Correct 64-bit integer handling.** An int64/uint64 correctness bug could silently corrupt values
  above 2⁵³, the sort of defect that produces wrong results with no error at all. It's fixed, so
  correctness holds even at the extremes of the integer range.

This release isn't a marginal gain. GPU decode of JPEG-compressed Aperio SVS slides is new in
26.06.00, and every GPU number in this blog rests on it: this is the release that makes the GPU path
functional, correct, and fast on real-world slides.

## What Is Not Here Yet

For completeness, a few limitations remain in this release:

- JPEG2000 whole-slide decode is handled on the CPU for now; DICOM JP2K pixel data is decoded on the CPU
  and staged to the device.
- GPUDirect Storage (GDS/cuFile) is stubbed and disabled in this release.
- The `amd-cupy` sparse path has gaps: `spsolve()` raises `NotImplementedError`, and `random_walker()`'s
  bf mode is unavailable.
- rocJPEG decode resolution is bounded to 64–16384 px.

## Try It

```python
from cucim import CuImage
img = CuImage("CMU-1.svs")
print("Dimensions:", img.size())
region = img.read_region(location=[0, 0], size=(1024, 1024), level=0, device="cuda")
```

Install: `python3 -m pip install amd-hipcim --extra-index-url https://pypi.amd.com/rocm-10.0.0/simple/`

## Where This Fits

hipCIM is the foundation of the AMD Life Science medical-imaging stack, the layer that gets pixels onto the
GPU quickly and correctly. Everything above it, from MONAI's data loaders to the Model Zoo's
validated inference bundles, benefits from this fast foundation layer. If you're moving a digital
pathology or radiology workload to AMD Instinct™, hipCIM is where that journey begins.

## Summary

This blog showed how hipCIM 26.06.00 turns whole-slide image I/O into a GPU-resident operation on
AMD Instinct™ GPUs. On MI355X with ROCm 10.0 it reads tiled slides about 10 times faster than OpenSlide
on the GPU path and 7 to 11 times faster on the CPU path, decodes Aperio SVS faithfully to the CPU
reference, widens OME-TIFF coverage from 0.8% to 90.3% of the test corpus, and hands tiles to ROCm array
libraries with no host round-trip. Because it keeps the `from cucim import CuImage` API, your existing
image-I/O code moves across unchanged. The win is faster, GPU-resident pixels for the same code you
already run.

For digital pathology and radiology teams, that means the read step stops being the bottleneck and the
accelerator spends its time on the model instead of waiting on I/O. This is the foundation layer of the
AMD Life Science medical-imaging stack, and the layers above it are MONAI data loading,
validated Model Zoo inference, and interactive annotation. Companion posts will cover each of those
as they publish. Follow the AMD ROCm Blogs for the rest of the series and for the
GPU-resident features still planned, including JPEG2000 GPU decode and GPUDirect Storage.

## Additional Resources

- [Announcing hipCIM: A Cutting-Edge Solution for Accelerated Multidimensional Image Processing](https://rocm.blogs.amd.com/software-tools-optimization/hipcim-intro/README.html): the introductory post covering hipCIM 1.0.0, July 2025.
- [hi-ml project (Microsoft Research)](https://github.com/microsoft/hi-ml): an independent benchmark of whole-slide image readers on a full tiling workload.
- [OpenSlide](https://openslide.org/): the CPU reference library used as the comparison baseline.
- [OpenSlide test data (Carnegie Mellon University)](https://openslide.cs.cmu.edu/download/openslide-testdata/): CMU-1/2/3 and GenericTIFF slides, licensed CC0-1.0.
- [hipCIM on GitHub](https://github.com/AMD-Ecosystem/hipCIM)
- [AMD ROCm documentation](https://rocm.docs.amd.com/)
- [AMD ROCm Blogs](https://rocm.blogs.amd.com/)

## System Configuration

All results in this post, covering the GPU path, the CPU path and the OpenSlide baseline, come from
a single run on the configuration below.

| Component | Configuration |
| --- | --- |
| GPU | AMD Instinct™ MI355X (gfx950), bare metal |
| ROCm | 10.0 |
| hipCIM | 26.06.00 |
| GPU JPEG decode | rocJPEG, VA-API hardware decode |
| CPU decode backend | libjpeg-turbo |
| OpenSlide baseline | run on the same node |

The host CPU, operating system, amdgpu driver version and container image digest were not recorded
for this run, so they are omitted rather than inferred.

**Benchmark.** 576 non-overlapping 256×256 patches per slide, a 24×24 grid at level 0, read
sequentially per region with a warm cache, reporting the median of five trials. Aperio CMU-1/2/3.svs
and GenericTIFF CMU-1 come from the OpenSlide test collection, licensed CC0-1.0. The GPU result was
independently reproduced on 2 September 2026 on a separate MI355X node, at 9,411–10,450 patches/s.

## Disclaimers

Performance may vary by hardware configuration, workload, and software version.

## Third-Party Content Disclaimer

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
