---
title: ROCm Blogs
myst:
  html_meta:
    "description lang=en": "AMD ROCm™ software blogs"
    "keywords": "AMD GPU, MI300, MI250, ROCm, blog"
    "property=og:locale": "en_US"
---

<!--
Updated August 29 2024
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
  .bd-main .bd-content .bd-article-container {
    max-width: 100%;
  }
  .bd-sidebar-secondary {
    display: none;
  }
  .sd-card-large.sd-card {}
  #buttonWrapper:hover {
    border-color: hsla(231, 99%, 66%, 1);
    transform: scale(1.05);
    background-color: var(--hover-background-colour);
  }
  .small-sd-card-large.sd-card {}
  #buttonWrapper:hover {
    border-color: hsla(231, 99%, 66%, 1);
    transform: scale(1.05);
    background-color: var(--hover-background-colour);
  }
  #buttonWrapper {
    border-color: #A9A9A9;
    background-color: var(--original-background)
    text-align: center;
    font-weight: bold;
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
  .read-more-btn {
    font-size: 20px;
    padding: 10px;
    font-weight: bold;
    cursor: pointer;
    display: inline-block;
    align-items: center;
    text-decoration: none;
    overflow: hidden;
    gap: 7px;
    display: block;
    text-align: left;
    margin-left: 0;
    margin-top: 10px;
  }
  .read-more-btn-small {
    font-size: 15px;
    padding: 10px;
    font-weight: bold;
    cursor: pointer;
    display: inline-block;
    align-items: center;
    text-decoration: none;
    overflow: hidden;
    gap: 7px;
    display: block;
    text-align: left;
    margin-left: 0;
    margin-top: 10px;
  }
  .arrows {
    font-size: 20px;
    display: inline-block;
    font-weight: bold;
    transition: transform 0.3s ease, color 0.3s ease, font-size 0.3s ease;
  }
  .read-more-btn:hover .arrows {
    transform: translateX(8px);
  }
  .arrows-small {
    font-size: 15px;
    display: inline-block;
    font-weight: bold;
    transition: transform 0.3s ease, color 0.3s ease, font-size 0.3s ease;
  }
  .read-more-btn-small:hover .arrows-small {
    transform: translateX(10px);
  }
  .date {
    font-size: 13px;
    font-weight: 300;
    line-height: 22.5px;
    text-transform: none;
    margin-bottom: 10px;
  }
  .paragraph {
    font-size: 16px;
    line-height: 24px;
    margin-bottom: 10px;
  }
  .large-sd-card-img-top.sd-card-img-top {
    width: 100%;
    height: 21vw;
    object-fit: cover;
  }
  .small-sd-card-img-top.sd-card-img-top {
    width: 100%;
    height: 15vw;
    object-fit: cover;
  }
  .large-sd-card.sd-card-body {
    width: 100%;
    height: 15%;
  }
  .small-sd-card.sd-card-body {
    width: 100%;
    height: 15%;
  }
  .bd-content .sd-card .sd-card-footer {
    border-top: none;
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

::::{grid} 1 2 2 2
:margin 2

:::{grid-item-card}
:padding: 1
:img-top: ./images/stone-ridge.jpg
:class-img-top: large-sd-card-img-top
:class-body: large-sd-card

<a href=".\ecosystems-and-partners\stone-ridge\README.html" class="card-header-link">
  <h2 class="card-header">Stone Ridge Expands Reservoir Simulation Options with AMD Instinct™ Accelerators</h2>
</a>
<div class="date">June 10, 2024</div>

<p class="paragraph">Stone Ridge Technology (SRT) pioneered the use of GPUs for high performance reservoir simulation (HPC) nearly a decade ago with ECHELON, its flagship... </p>
+++
<a href="./ecosystems-and-partners/stone-ridge/README.html" class="read-more-btn">Read More <span class="arrows">>></span></a>
:::

:::{grid-item-card}
:padding: 1
:img-top: ./images/university-of-michigan-bioinformatics.jpg
:class-img-top: large-sd-card-img-top
:class-body: large-sd-card

<a href="./ecosystems-and-partners/university-of-michigan/README.html" class="card-header-link">
  <h2 class="card-header">AMD Collaboration with the University of Michigan offers High Performance Open-Source Solutions to the Bioinformatics Community</h2>
</a>
<div class="date">May 16, 2024</div>

<p class="paragraph">Long read DNA sequencing technology is revolutionizing genetic diagnostics and precision medicine by helping us discover structural variants and assem... </p>
+++
<a href="./ecosystems-and-partners/university-of-michigan/README.html" class="read-more-btn">Read More <span class="arrows">>></span></a>
:::

:::{grid-item-card}
:padding: 1
:img-top: ./images/siemens.jpg
:class-img-top: large-sd-card-img-top
:class-body: large-sd-card

<a href="./ecosystems-and-partners/Siemens/README.html" class="card-header-link">
  <h2 class="card-header">Siemens taps AMD Instinct™ GPUs to expand high-performance hardware options for Simcenter STAR-CCM+</h2>
</a>
<div class="date">May 16, 2024</div>

<p class="paragraph">Siemens recently announced that its Simcenter STAR-CCM+ multi-physics computational fluid dynamics (CFD) software now supports AMD Instinct™ GPUs for... </p>
+++
<a href="./ecosystems-and-partners/Siemens/README.html" class="read-more-btn">Read More <span class="arrows">>></span></a>
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

::::{grid} 1 3 3 3
:margin: 1

:::{grid-item-card}
:padding: 1
:img-top: ./images/2024-08-28-mlperf.jpeg
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card

<a href="./artificial-intelligence/mlperf-inf-4-1/README.html" class="small-card-header-link">
    <h2 class="card-header">Benchmarking Machine Learning using ROCm and AMD GPUs: Reproducing Our MLPerf Inference Submission</h2>
</a>
<div class="date">August 28, 2024 by Meena Arunachalam
  ,
  Miro Hodak
  ,
  Jeremy Arnold
  ,
  <a href="https://rocm.blogs.amd.com/authors/eliot-li.html">Eliot Li</a>

</div>

Measuring the performance of new technologies is as old as human history, and often as intriguing. The AMD MLPerf Inference v4.1 submission has three entries for Llama 2 70B. The submission used a fully open-source software stack based on the ROCm platform and vLLM inference engine.
<a href="./artificial-intelligence/mlperf-inf-4-1/README.html" class="read-more-btn-small">Read More <span class="arrows-small">></span></a>
:::

:::{grid-item-card}
:padding: 1
:img-top: ./images/nlp.jpg
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card

<a href="./artificial-intelligence/llm-tasks/README.html" class="small-card-header-link">
    <h2 class="card-header">Performing natural language processing tasks with LLMs on ROCm running on AMD GPUs</h2>
</a>
<div class="date">August 21, 2024 by <a href="https://rocm.blogs.amd.com/authors/eliot-li.html">Eliot Li</a></div>

In this blog you will learn how to use ROCm, running on AMD’s Instinct GPUs, for a range of popular and useful natural language processing (NLP) tasks, using different large language models (LLMs).
+++
<a href="./artificial-intelligence/llm-tasks/README.html" class="read-more-btn-small">Read More <span class="arrows-small">></span></a>
:::

:::{grid-item-card}
:padding: 1
:img-top: ./images/times-series.jpeg
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card

<a href="./artificial-intelligence/timeseries_transformers/README.html" class="small-card-header-link">
    <h2 class="card-header">Times series transformers</h2>
</a>
<div class="date">August 19, 2024 by <a href="https://rocm.blogs.amd.com/authors/fabricio-flores.html">Fabricio Flores</a></div>

Time series forecasting (TSF) is a key concept in fields such as signal processing, data science, and machine learning (ML).
+++
<a href="./artificial-intelligence/timeseries_transformers/README.html" class="read-more-btn-small">Read More <span class="arrows-small">></span></a>
:::

:::{grid-item-card}
:padding: 1
:img-top: ./images/inference.jpeg
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card

<a href="./artificial-intelligence/grok1/README.html" class="small-card-header-link">
    <h2 class="card-header">Inferencing with Grok-1 on AMD GPUs</h2>
</a>
<div class="date">August 9, 2024 by <a href="https://rocm.blogs.amd.com/authors/eliot-li.html">Eliot Li</a>
  ,
  Luise Chen
  ,
  Lei Shao
</div>

We demonstrate that the massive Grok-1 model from xAI can run seamlessly on the AMD MI300X GPU accelerator by leveraging the ROCm software platform.
+++
<a href="./artificial-intelligence/grok1/README.html" class="read-more-btn-small">Read More <span class="arrows-small">></span></a>
:::

:::{grid-item-card}
:padding: 1
:img-top: ./images/2024-07-29-gunrocks.jpg
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card

<a href="./high-performance-computing/graphs/README.html" class="small-card-header-link">
    <h2 class="card-header">Graph analytics on AMD GPUs using Gunrock</h2>
</a>
<div class="date">July 29, 2024 by <a href="https://rocm.blogs.amd.com/authors/author/thomas-gibson.html">Thomas Gibson</a>,Muhammad Osama</div>

Can AMD GPUs help with graph analytic operations? We will show some cases where GPUs can improve the performance of these valuable algorithms.
+++
<a href="./high-performance-computing/graphs/README.html" class="read-more-btn-small">Read More <span class="arrows-small">></span></a>
:::

:::{grid-item-card}
:padding: 1
:img-top: ./images/2024-07-29-roberta.jpg
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card

<a href="./artificial-intelligence/roberta_amp/README.html" class="small-card-header-link">
    <h2 class="card-header">Optimizing RoBERTa: Fine-Tuning with Mixed Precision on AMD</h2>
</a>
<div class="date">July 29, 2024 by <a href="https://rocm.blogs.amd.com/authors/fabricio-flores.html">Fabricio Flores</a></div>

In this blog we explore how to fine-tune the Robustly Optimized BERT Pretraining Approach (RoBERTa) large language model, with emphasis on PyTorch’s mixed precision capabilities.
+++
<a href="./artificial-intelligence/roberta_amp/README.html" class="read-more-btn-small">Read More <span class="arrows-small">></span></a>
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

::::{grid} 1 2 2 2
:margin: 1

:::{grid-item-card}
:padding: 1
:img-top: ./images/2024-06-18-tensorflow.jpg
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card

<a href="./software-tools-optimization/tf_profiler/README.html" class="small-card-header-link">
    <h2 class="card-header">TensorFlow Profiler in practice: Optimizing TensorFlow models on AMD GPUs</h2>
</a>
<div class="date">June 18, 2024 by <a href="https://rocm.blogs.amd.com/authors/fabricio-flores.html">Fabricio Flores</a></div>

TensorFlow Profiler consists of a set of tools designed to measure resource utilization and performance during the execution of TensorFlow models....
+++
<a href="./software-tools-optimization/tf_profiler/README.html" class="read-more-btn-small">Read More <span class="arrows-small">></span></a>
:::

:::{grid-item-card}
:padding: 1
:img-top: ./images/2024-05-31-mi300x.png
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card

<a href="./software-tools-optimization/ck-int8-gemm-sq/README.html" class="small-card-header-link">
    <h2 class="card-header">SmoothQuant model inference on AMD Instinct MI300X using Composable Kernel</h2>
</a>
<div class="date">May 31, 2024 by <a href="https://rocm.blogs.amd.com/authors/cheng-ling.html">Cheng Ling</a></div>

The AMD ROCm™ Composable Kernel (CK) library provides a programming model for writing performance-critical kernels...
+++
<a href="./software-tools-optimization/ck-int8-gemm-sq/README.html" class="read-more-btn-small">Read More <span class="arrows-small">></span></a>
:::

:::{grid-item-card}
:padding: 1
:img-top: ./images/2024-05-13-hip.jpeg
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card

<a href="./software-tools-optimization/amdgcn-isa/README.html" class="small-card-header-link">
    <h2 class="card-header">Reading AMD GPU ISA</h2>
</a>
<div class="date">May 13, 2024 by
  <a href="https://rocm.blogs.amd.com/authors/asitav-mishra.html">Asitav Mishra</a>
  ,
  <a href="https://rocm.blogs.amd.com/authors/corbin-robeck.html">Corbin Robeck</a>
</div>

Rocprof is a robust tool designed to analyze and optimize the performance of HIP programs on AMD ROCm platforms...
+++
<a href="./software-tools-optimization/amdgcn-isa/README.html" class="read-more-btn-small">Read More <span class="arrows-small">></span></a>
:::

:::{grid-item-card}
:padding: 1
:img-top: ./images/2024-05-07-tracing.jpeg
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card

<a href="./software-tools-optimization/roc-profiling/README.html" class="small-card-header-link">
    <h2 class="card-header">AMD in Action: Unveiling the Power of Application Tracing and Profiling</h2>
</a>
<div class="date">May 7, 2024 by <a href="https://rocm.blogs.amd.com/authors/fabricio-flores.html">Fabricio Flores</a></div>

Rocprof is a robust tool designed to analyze and optimize the performance of HIP programs on AMD ROCm platforms...
+++
<a href="./software-tools-optimization/roc-profiling/README.html" class="read-more-btn-small">Read More <span class="arrows-small">></span></a>
:::

:::{grid-item-card}
:padding: 1
:img-top: ./images/2024-04-26-hip.jpeg
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card

<a href="./software-tools-optimization/hipify/README.html" class="small-card-header-link">
    <h2 class="card-header">Application portability with HIP</h2>
</a>
<div class="date">April 26, 2024 by
  <a href="https://rocm.blogs.amd.com/authors/suyash-tandon.html">Suyash Tandon</a>
  ,
  <a href="https://rocm.blogs.amd.com/authors/maria-ruiz-varela.html">Maria Ruiz Varela</a>
  ,
  <a href="https://rocm.blogs.amd.com/authors/gina-sitaraman.html">Gina Sitaraman</a>
  ,
  <a href="https://rocm.blogs.amd.com/authors/bob-robey.html">Bob Robey</a>
</div>

Many scientific applications run on AMD-equipped computing platforms and supercomputers, including Frontier...
+++
<a href="./software-tools-optimization/hipify/README.html" class="read-more-btn-small">Read More <span class="arrows-small">></span></a>
:::

:::{grid-item-card}
:padding: 1
:img-top: ./images/2024-04-18-cpp.jpeg
:class-img-top: small-sd-card-img-top
:class-body: small-sd-card

<a href="./software-tools-optimization/hipstdpar/README.html" class="small-card-header-link">
    <h2 class="card-header">C++17 parallel algorithms and HIPSTDPAR</h2>
</a>
<div class="date">April 18, 2024 by
  <a href="https://rocm.blogs.amd.com/authors/alessandro-fanfarillo.html">Alessandro Fanfarillo</a>
  ,
  <a href="https://rocm.blogs.amd.com/authors/alex-voicu.html">Alex Voicu</a>
</div>

The C++17 standard added the concept of parallel algorithms to the pre-existing C++ Standard Library. The parallel version of algorithms like...
+++
<a href="./software-tools-optimization/hipstdpar/README.html" class="read-more-btn-small">Read More <span class="arrows-small">></span></a>
:::
::::

<h2> Stay informed</h2>
<ul>
  <li><a href="blog/atom.xml"> Subscribe to our <i class="fa fa-rss fa-rotate-270"></i> RSS feed</a></li>
  <li><a href="https://github.com/ROCm/rocm-blogs"> Watch our GitHub repo </a></li>
</ul>
