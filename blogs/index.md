---
title: ROCm Blogs
myst:
  html_meta:
    "description lang=en": "AMD ROCm™ software blogs"
    "keywords": "AMD GPU, MI300, MI250, ROCm, blog"
    "property=og:locale": "en_US"
---

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

:::{grid-item-card} University of Michigan, AMD collaboration
:padding: 0
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

:::{grid-item-card} Programming AMD GPUs with Julia
:padding: 1
:link: ./software-tools-optimization/julia-amdgpu/README
:link-type: doc

Julia is a high-level, general-purpose dynamic programming language
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

:::{grid-item-card} Detectron2 on AMD GPUs
:padding: 1
:link: ./artificial-intelligence/detectron2/README
:link-type: doc

Panoptic segmentation and instance segmentation with Detectron2 on AMD GPUs
:::

:::{grid-item-card} Accelerating Large Language Models with Flash Attention on AMD GPUs
:padding: 1
:link: ./artificial-intelligence/flash-attention/README
:link-type: doc

In this blog post, we will guide you through the process of installing Flash Attention on AMD GPUs
:::

:::{grid-item-card} Step-by-Step Guide to Use OpenLLM on AMD GPUs
:padding: 1
:link: ./artificial-intelligence/openllm/README
:link-type: doc

OpenLLM is an open-source platform designed to facilitate the deployment and utilization of large language models (LLMs)
:::

:::{grid-item-card} Inferencing with Mixtral 8x22B on AMD GPUs
:padding: 1
:link: ./artificial-intelligence/moe/README
:link-type: doc

Mixtral 8x22B is a sparse MoE decoder-only transformer model, get it working on AMD GPUs
:::

:::{grid-item-card} Training a Neural Collaborative Filtering (NCF) Recommender
:padding: 1
:link: ./artificial-intelligence/ncf/README
:link-type: doc

Collaborative Filtering is a type of item recommendation where new items are recommended to the user based on their past interactions.
:::

:::{grid-item-card} Table Question-Answering with TaPas
:padding: 1
:link: ./blog/category/applications-models.html
:link-type: url

The TaPas (Table Parser) model is a BERT-based, weakly supervised question answering model that has been designed and pretrained for answering questions about tabular data
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

:::{grid-item-card} Reading AMD GPU ISA
:padding: 1
:link: ./software-tools-optimization/amdgcn-isa/README
:link-type: doc

In this blog post, we will discuss how to read and understand the ISA for AMD’s Graphics Core Next (AMDGCN) architecture
:::

:::{grid-item-card} AMD in Action: Unveiling the Power of Application Tracing and Profiling
:padding: 1
:link: ./high-performance-computing/roc-profiling/README
:link-type: doc

Rocprof is a robust tool designed to analyze and optimize the performance of HIP programs on AMD ROCm platforms
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

:::{grid-item-card} Affinity part 2 - System topology and controlling affinity
:padding: 1
:link: ./software-tools-optimization/affinity/part-2/README
:link-type: url

Discuss ways to verify if affinity is set up correctly for your run. Point to tools that can help you understand the topology of your system. Show you how to set affinity for different types of applications.
:::
::::

<h2> Stay informed</h2>
<ul>
  <li><a href="blog/atom.xml"> Subscribe to our <i class="fa fa-rss fa-rotate-270"></i> RSS feed</a></li>
  <li><a href="https://github.com/ROCm/rocm-blogs"> Watch our GitHub repo </a></li>
</ul>
