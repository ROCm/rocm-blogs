---
title: ROCm Blogs
myst:
  html_meta:
    "description lang=en": "AMD ROCm™ software blogs"
    "keywords": "AMD GPU, MI300, MI250, ROCm, blog"
    "property=og:locale": "en_US"
---

<!--
Updated July 8 2024
-->

<h1><a href="blog/atom.xml"><i class="fa fa-rss fa-rotate-270"></i></a> AMD ROCm™ Blogs</h1>

<script>
  const buttonWrapper = document.getElementById('buttonWrapper');

  const observer = new MutationObserver((mutationsList) => {
    for (const mutation of mutationsList) {
      if (mutation.type === 'attributes' && mutation.attributeName === 'data-mode') {
        console.log(`Data mode changed to: ${newMode}`);
        if (newMode === 'light') {
          buttonWrapper.style.setProperty('--original-background', 'white');
          buttonWrapper.style.setProperty('--hover-background-colour', 'white');
        } else {
          buttonWrapper.style.setProperty('--original-background', 'black');
          buttonWrapper.style.setProperty('--hover-background-colour', 'black');
        }
      }
    }
  });

</script>

<style>
  #buttonWrapper:hover {
    border-color: hsla(231, 99%, 66%, 1);
    transform: scale(1.05);
    background-color: var(--hover-background-colour);
  }
  #buttonWrapper {
    border-color: #A9A9A9;
    background-color: var(--original-background)
    text-align: center;
    font-size: 12px;
    border-radius: 1px;
    transition: transform 0.2s, border-color 0.2s;
  }
  h2 {
    margin: 0;
    font-size: 1.5em;
  }
  .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px;
    box-sizing: border-box;
    width: 100%;
  }
</style>

<div class="container">
  <h2>Ecosystems and partners</h2>
  <a href="blog/category/ecosystems-and-partners.html">
    <button id="buttonWrapper">
      See All >>
    </button>
  </a>
</div>

::::{grid} 1 2 3 3
:margin 1

:::{grid-item-card} Stone Ridge Expands Reservoir Simulation Options with AMD Instinct™ Accelerators
:padding: 1
:link: ./ecosystems-and-partners/stone-ridge/README
:link-type: doc

Stone Ridge Technology latest development effort was to port ECHELON from CUDA to the AMD HIP platform, enabling ECHELON to use AMD Instinct GPUs like the MI210, MI250X, and the upcoming MI300 Series
:::

:::{grid-item-card} University of Michigan, AMD collaboration
:padding: 1
:link: ./ecosystems-and-partners/university-of-michigan/README
:link-type: doc

AMD Collaboration with the University of Michigan offers
High Performance Open-Source Solutions to the Bioinformatics Community
:::

:::{grid-item-card} Siemens and AMD partnership
:padding: 1
:link: ./ecosystems-and-partners/Siemens/README
:link-type: doc

Siemens taps AMD Instinct™ GPUs to expand high-performance hardware options for Simcenter STAR-CCM+
:::
::::

<div class="container">
  <h2>Applications and models</h2>
  <a href="blog/category/applications-models.html">
    <button id="buttonWrapper">
      See All >>
    </button>
  </a>
</div>

::::{grid} 1 2 3 3
:margin: 1

:::{grid-item-card} Accelerating models on ROCm using PyTorch TunableOp
:padding: 1
:link: ./artificial-intelligence/pytorch-tunableop/README
:link-type: doc

In this blog, we will show how to leverage PyTorch TunableOp to accelerate models using ROCm on AMD GPUs.
:::

:::{grid-item-card} A Guide to Implementing and Training Generative Pre-trained Transformers (GPT) in JAX on AMD GPUs
:padding: 1
:link: ./artificial-intelligence/nanoGPT-JAX/README
:link-type: doc

In this blog, we illustrate the process of implementing and training a Generative Pre-trained Transformer (GPT) model in JAX.
:::

:::{grid-item-card} Mamba on AMD GPUs with ROCm
:padding: 1
:link: ./artificial-intelligence/mamba/README
:link-type: doc

In this blog, we delve into the Mamba architecture and demonstrate how to use Mamba on AMD GPUs with the ROCm platform.
:::

:::{grid-item-card} Deep Learning Recommendation Model on AMD GPU
:padding: 1
:link: ./artificial-intelligence/dlrm/README
:link-type: doc

In this blog, we demonstrate how to build a simple Deep Learning Recommendation Model (DLRM) with PyTorch on a ROCm-capable AMD GPU.
:::

:::{grid-item-card} Segment Anything with AMD GPUs
:padding: 1
:link: ./artificial-intelligence/segment-anything/README.html
:link-type: url

The Segment Anything Model (SAM) is a cutting-edge image segmentation model that democratizes promptable segmentation.
:::

:::{grid-item-card} Detectron2 on AMD GPUs
:padding: 1
:link: ./artificial-intelligence/detectron2/README
:link-type: doc

Panoptic segmentation and instance segmentation with Detectron2 on AMD GPUs.
:::

::::

<div class="container">
  <h2>Software tools & optimizations</h2>
  <a href="blog/category/software-tools-optimizations.html">
    <button id="buttonWrapper">
      See All >>
    </button>
  </a>
</div>

::::{grid} 1 2 3 3
:margin: 1

:::{grid-item-card} SmoothQuant model inference on AMD Instinct MI300X using Composable Kernel
:padding: 1
:link: ./software-tools-optimization/ck-int8-gemm-sq/README.html
:link-type: url

The AMD ROCm™ Composable Kernel (CK) library provides a programming model for writing performance-critical kernels for machine learning workloads.
:::

:::{grid-item-card} AMD in Action: Unveiling the Power of Application Tracing and Profiling
:padding: 1
:link: ./software-tools-optimization/roc-profiling/README
:link-type: doc

Rocprof is a robust tool designed to analyze and optimize the performance of HIP programs on AMD ROCm platforms
:::

:::{grid-item-card} Reading AMD GPU ISA
:padding: 1
:link: ./software-tools-optimization/amdgcn-isa/README
:link-type: doc

In this blog post, we will discuss how to read and understand the ISA for AMD’s Graphics Core Next (AMDGCN) architecture
:::

:::{grid-item-card} Application portability with HIP
:padding: 1
:link: ./software-tools-optimization/hipify/README
:link-type: doc

HIP enables these High-Performance Computing (HPC) facilities to transition their CUDA codes to run and take advantage of the latest AMD GPUs
:::

:::{grid-item-card} C++17 parallel algorithms and HIPSTDPAR
:padding: 1
:link: ./software-tools-optimization/hipstdpar/README
:link-type: doc

The C++17 standard added the concept of parallel algorithms to the pre-existing C++ Standard Library
:::

:::{grid-item-card} Affinity part 1 - Affinity, placement, and order
:padding: 1
:link: ./software-tools-optimization/affinity/part-1/README
:link-type: doc

Affinity is a way for processes to indicate preference of hardware components so that a given process is always scheduled to the same set of compute cores and is able to access data from local memory efficiently
:::

::::

<h2> Stay informed</h2>
<ul>
  <li><a href="blog/atom.xml"> Subscribe to our <i class="fa fa-rss fa-rotate-270"></i> RSS feed</a></li>
  <li><a href="https://github.com/ROCm/rocm-blogs"> Watch our GitHub repo </a></li>
</ul>
