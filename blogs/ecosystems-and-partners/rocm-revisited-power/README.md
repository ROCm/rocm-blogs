---
blogpost: true
blog_title: "AMD ROCm: Powering the World's Fastest Supercomputers"
date: Jun 10 2025
author: 'Mohammed Faraaz Mustafa, Saad Rahim'
thumbnail: 'supercomputing.png'
tags: AI/ML, Scientific Computing, HPC, Performance
category: Ecosystems and Partners
target_audience: AI/ML Developers, System Administrators
key_value_propositions: This blog serves as an educational piece on AMD success in the supercomputing space
language: English
myst:
    html_meta:
        "author": "Mohammed Faraaz Mustafa"
        "description lang=en": "Discover how ROCm drives the world’s top supercomputers, from El Capitan to Frontier, and why its shaping the future of scalable, open and sustainable HPC"
        "keywords": "rocm, supercomputing, Frontier, top500, exascale, HPC"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Ecosystem and Partners"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "Generative AI, AI Inference, AI Training, Data Science, Design, Simulation & Modeling"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Mohammed Faraaz Mustafa"
---

<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the right
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

# AMD ROCm: Powering the World's Fastest Supercomputers

From breaking the exaFLOP barrier with Frontier to setting new performance records with El Capitan, AMD is transforming what’s possible in high-performance computing (HPC). But the story goes beyond hardware. At the core of these world-class systems is ROCm, AMD’s open, high-performance software platform enabling new levels of scientific discovery and AI advancement.

This blog will cover how ROCm powers the world’s most advanced supercomputers, including El Capitan and Frontier, by enabling exceptional performance, scalability, and energy efficiency. It will explore ROCm’s open-source approach, its role in orchestrating massive multi-GPU systems, and the groundbreaking scientific and AI advancements made possible through this technology. Additionally, the blog will highlight AMD’s global impact on sustainable supercomputing and its commitment to precision computing for critical research and engineering applications.

This post is part of a four-part blog series. The ROCm Revisited series explores core concepts of the AMD ROCm software platform, along with its tools and optimizations—designed specifically for beginner to intermediate developers.
Explore the full series here: [Introducing the ROCm Revisited Series](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-revisited/README.html)

## Why Open-Source Software is the Key to Supercomputing Success

Proprietary software ecosystems often restrict collaboration and hinder flexibility. In contrast, ROCm provides a fundamentally different model—one built on openness, portability, transparency and sustainable innovation.

Key benefits of ROCm’s open-source approach include:

* Freedom to inspect, extend, and optimize the full stack — from compiler to kernel
* Portability of code between AMD, Nvidia, and future architectures via [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html). HIP is a C++ runtime API and kernel language that allows you to create portable applications for AMD and NVIDIA GPUs from a single source code.
* Faster innovation from a growing developer community contributing upstream
* Compatibility with major open-source frameworks like PyTorch, TensorFlow, and Kokkos

An example of this open ecosystem in action is the collaboration around the [LUMI supercomputer and the AI2 OLMo project](https://www.lumi-supercomputer.eu/ai2-olmo-an-open-language-model-made-by-scientists-for-scientists/). The Allen Institute for AI recently announced AI2 OLMo (Open Language Model), a fully open, state-of-the-art generative language model developed specifically for scientific research. Using LUMI, one of the world’s fastest and greenest pre-exascale AMD supercomputers, OLMo aims to democratize access to large language models by openly sharing all aspects of model creation—from data and training code to evaluation benchmarks and ethical considerations.

This openness has ripple effects beyond supercomputers. Developers at national labs, universities, and startups can use the same ROCm stack powering the world’s fastest supercomputers and ensure long-term sustainability of scientific software.

Learn more about ROCm open-source ecosystem by reading: [The High-Performance Computing Ecosystem](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-revisited-ecosy/README.html).

## Breaking the Exascale Barrier

Breaking the exascale barrier marks a major achievement in computing, unlocking the ability to simulate and model complex phenomena such as climate systems, nuclear reactions, and galaxy formation at unprecedented speed and scale. AMD-powered supercomputers Frontier and El Capitan define this new era, each capable of performing over one quintillion ($10^{18}$) calculations per second. Both systems operate using AMD’s ROCm software stack, which coordinates tens of thousands of GPUs to deliver performance that is scalable, programmable, and ready for real-world scientific and AI advancements.

### El Capitan: The Pinnacle of Performance

El Capitan, located at Lawrence Livermore National Laboratory, stands as the world's fastest supercomputer. Launched and operational on November 18th, 2024, it achieved a staggering 1.742 exaflops on the High Performance Linpack (HPL) benchmark, surpassing its predecessors in both speed and efficiency [^1]. This monumental performance is powered by AMD's Instinct MI300A Accelerated Processing Units (APUs), which integrate custom AMD 4th-Gen EPYC 24-core CPUs and GPUs, supported by ROCm 6.0.

Engineered primarily for national security simulations at LLNL, El Capitan is expected to revolutionize stockpile stewardship with AI-accelerated modeling.

El Capitan has secured the #1 spot for both HPL-MxP and HPCG benchmarks at the [ISC 2025](https://insidehpc.com/2025/06/top500-el-capitan-stays-on-top-us-holds-top-3-supercomputers-europe-expands-in-leadership-hpc/), showcasing its versatility beyond raw speed.

* HPL-MxP (High-Performance Linpack Mixed Precision) is a benchmark that measures a system’s ability to perform lower-precision floating-point calculations, which are critical for AI and machine learning workloads. El Capitan achieved a staggering 16.7 exaFLOPS on HPL-MxP, running on 10,952 nodes, highlighting its strength in accelerating AI training and inference tasks efficiently [^9].
* HPCG (High-Performance Conjugate Gradient) is a benchmark designed to evaluate computing performance on more memory-intensive, realistic scientific workloads. It reflects the system's capability for high-precision computations important in scientific simulations and engineering applications. El Capitan reached 17.1 petaflops on HPCG, running on 10,880 nodes, demonstrating exceptional performance in these critical domains [^9].

These top rankings underscore El Capitan’s balanced architecture, optimized to deliver peak performance across diverse workloads — from AI-driven innovation to traditional high-performance computing (HPC).

```{figure} ./images/El-Capitan.jpg
:align: center
:width: 800px
:alt: El-Capitan Visual Depiction
Figure 1. El-Capitan Situated in Lawrence Livermore National Laboratory.
```

### Frontier: A Trailblazer in Exascale Computing

Frontier, located at Oak Ridge National Laboratory, was the first supercomputer to officially break the exascale threshold, achieving a performance of 1.353 exaflops on May 27th, 2022 [^1]. This groundbreaking system currently ranks as the world’s second most powerful supercomputer.

Powered by AMD’s 3rd-generation EPYC CPUs and Instinct MI250X GPUs, Frontier comprises 9,408 CPUs and 37,632 GPUs — totaling over 50 million cores. Each compute node links one CPU to four GPUs, enabling highly optimized parallel processing. Running on the ROCm platform, Frontier delivers exceptional performance while prioritizing energy efficiency.

Frontier not only leads in raw computing power but also debuted at the top of the Green500 list as the world’s most energy-efficient supercomputer [^2]. It’s designed to drive groundbreaking research in fusion energy, climate science, and artificial intelligence.

```{figure} ./images/Frontier-Supercomputer.png
:align: center
:width: 800px
:alt: Frontier Visual Depiction
Figure 2. The Frontier supercomputer hosted at the Oak Ridge Leadership Computing Facility (OLCF).
```

### AMD in the TOP500 and Green500 List

AMD's commitment to performance and efficiency is evident in its presence on the TOP500 and Green500 lists:

* AMD's Frontier debuted top of the Green500 list for most efficient supercomputers [^2].
* As of November 2024, AMD powers 156 supercomputers on the TOP500 list, marking a 29% increase from the previous year [^3].
* On the Green500 list, which ranks supercomputers based on energy efficiency, AMD holds 157 entries, underscoring its dedication to sustainable computing [^3].

```{figure} ./images/Performance(Exaflops).png
:align: center
:width: 800px
:alt: AMD's Supercomputers Performance
Figure 5: AMD leads the top supercomputers with impressive performance.
```

## Powering Global HPC and Scientific Discovery with ROCm

AMD’s ROCm software platform plays a critical role in enabling high-performance computing across the globe, powering a growing network of supercomputers that drive scientific research. From Europe’s fastest systems to Australia’s leading HPC resources, AMD’s hardware and ROCm deliver scalable, efficient, and open-source solutions.

### Global Impact of AMD-Powered Supercomputers

AMD's influence extends beyond the United States, with several international supercomputers leveraging its technology:

* LUMI: Situated in Finland, LUMI is Europe's fastest supercomputer, boasting a peak performance of over 550 petaflops. It employs AMD EPYC CPUs and Instinct MI250X GPUs, all managed via the ROCm ecosystem.
* HPC6: Located in Bologna, Italy, HPC6 delivers 477.9 petaflops of performance, ranking it among the top supercomputers globally. It relies on AMD's hardware and ROCm software for its operations.
* Tuolumne: As a precursor to El Capitan, Tuolumne achieved 208.1 petaflops, showcasing the capabilities of AMD's MI300A APUs and ROCm integration.
* Setonix: Based in Western Australia at the Pawsey Supercomputing Research Centre, Setonix is the country’s most powerful and energy-efficient supercomputer. Leveraging AMD EPYC CPUs and Instinct GPUs, Setonix supports scientific workloads—from radio astronomy to life sciences.

### Frontier’s Scientific Wins Powered by ROCm

Frontier is not just a benchmark machine — it’s already delivering breakthroughs in science and engineering. ROCm plays a pivotal role in accelerating these discoveries by enabling optimized compute kernels, mixed-precision AI workflows, and seamless scaling across GPUs.

Some real-world highlights:

* Simulating Microscopic Behavior: Modeling the behavior of up to 600,000 electrons in a magnesium alloy with the accuracy of a quantum Monte Carlo simulation [^4].
* Modeling the Universe: Using dark matter and the movement of gas and plasma—not just gravity—to simulate the observable universe on Frontier [^5].
* Jet Engine Design: Predicting airflow and noise from a fuel-efficient jet engine to optimize its design [^6].
* Nuclear Reactor Optimizations: Simulating heat transfer through the core of a modular nuclear reactor to advance clean energy solutions [^7].
* Advancing Nuclear Fission Research: Speeding up the nuclear fission process by improving laser-based electron accelerator designs—work honored with the 2022 ACM Gordon Bell Prize [^7]
* Computational Chemistry: Simulating over one million electrons on exascale systems like Frontier, this breakthrough redefined the scale and accuracy of molecular dynamics. The work, led by an international team of eight researchers—including AMD’s Dr. Jakub Kurzak—earned the [2024 ACM Gordon Bell Prize](https://www.acm.org/media-center/2024/november/gordon-bell-prize-2024#:~:text=Atlanta%2C%20GA%2C%20November%2021%2C%202024%20%E2%80%93%20ACM%2C%20the,Biomolecular-Scale%20Ab%20Initio%20Molecular%20Dynamics%20Using%20MP2%20Potentials.%E2%80%9D).

By accelerating everything from numerical solvers to AI inference pipelines, ROCm empowers researchers to do more science, faster.

## ROCm: Coordinating Exascale Performance and the Future of AMD Supercomputing

Achieving exascale performance requires more than raw computational speed—it demands seamless coordination across tens of thousands of GPUs, robust software support for scientific workloads, and a commitment to scalability and openness. AMD’s ROCm software stack is designed precisely for this purpose, powering systems like Frontier and the upcoming El Capitan to new heights of capability.

At the hardware level, systems such as Frontier utilize 37,888 AMD Instinct™ GPUs linked by HPE Slingshot interconnects, each equipped with custom 64-port switches delivering 12.8 terabits per second of bandwidth. These compute blades are organized using a dragonfly topology, ensuring that even at full system scale, data can move efficiently with a maximum of three hops between any nodes.

On the software side, ROCm provides essential infrastructure for large-scale coordination:

* RCCL (ROCm Collective Communication Library): Accelerates GPU-to-GPU communication for operations like all-reduce and broadcast.

* ROCm-aware MPI: Enables integration with MPI stacks such as OpenMPI and MPICH to support complex distributed computing.

These technologies allow ROCm to orchestrate workloads from a single GPU to tens of thousands without sacrificing performance, enabling researchers to run tightly-coupled simulations at exascale scale.

AMD maintains a strong commitment to supporting double-precision floating-point (FP64) capabilities, recognizing that true high-performance computing (HPC) capacity requires robust precision to be viable for critical applications. Delivering effective HPC solutions means designing hardware with full double-precision support and data pathways specifically tailored to the HPC ecosystem [^8]. AMD provides products that meet the exacting precision and performance needs of HPC workloads.

Importantly, ROCm's open-source foundation ensures that this power isn't locked away—it’s shared. The same stack that runs the world’s most powerful systems is available to academia, startups, and the global HPC community, driving innovation from the lab to the cloud.

## Summary

This blog explores how AMD’s ROCm open-source software platform powers the world’s fastest supercomputers, including El Capitan and Frontier, enabling breakthrough performance and scalability in high-performance computing and AI workloads. It highlights the cutting-edge hardware innovations behind these systems, the critical role ROCm plays in orchestrating multi-GPU coordination, and showcases real-world scientific achievements made possible by AMD technology. Additionally, the blog discusses the global impact of AMD-powered supercomputers, the importance of open-source software for sustainable innovation, and AMD’s commitment to precision and efficiency in shaping the future of scalable supercomputing.

Curious how ROCm can accelerate your next breakthrough? The ROCm Revisited series explores core concepts of the AMD ROCm software platform, along with its tools and optimizations—designed specifically for beginner to intermediate developers. Explore the full series here: [Introducing the ROCm Revisited Series](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-revisited/README.html)). Explore our comprehensive [ROCm documentation](https://rocm.docs.amd.com/en/latest/index.html) to dive deeper into installation guides, programming models, and best practices.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

[^1]: “El Capitan achieves top spot, Frontier and Aurora follow behind” Top500.org, 2024. [https://www.top500.org/news/el-capitan-achieves-top-spot-frontier-and-aurora-follow-behind/](https://www.top500.org/news/el-capitan-achieves-top-spot-frontier-and-aurora-follow-behind/)

[^2]: “Computer engineers at ORNL pioneer approaches to energy efficient supercomputing | ORNL,” ORNL, Sep. 10, 2024. [https://www.ornl.gov/news/computer-engineers-ornl-pioneer-approaches-energy-efficient-supercomputing](https://www.ornl.gov/news/computer-engineers-ornl-pioneer-approaches-energy-efficient-supercomputing)

[^3]: “AMD’s New Instinct MI300A APU Powers Three Supercomputers in the Latest Top500 Rankings,” Sahm, 2024. [https://www.sahmcapital.com/news/content/amds-new-instinct-mi300a-apu-powers-three-supercomputers-in-the-latest-top500-rankings-2024-05-13](https://www.sahmcapital.com/news/content/amds-new-instinct-mi300a-apu-powers-three-supercomputers-in-the-latest-top500-rankings-2024-05-13)

[^4]: “Exascale Supercomputers With 10^18 FLOPS To Revolutionise Material Simulations. Driving The Development Of Fuel-Efficient Cars And Novel Superconductors,” Quantum Zeitgeist, Nov. 25, 2023. [https://quantumzeitgeist.com/exascale-supercomputers-revolutionise-material-simulations-paving-way-for-fuel-efficient-cars-and-novel-superconductors/](https://quantumzeitgeist.com/exascale-supercomputers-revolutionise-material-simulations-paving-way-for-fuel-efficient-cars-and-novel-superconductors/)

[^5]: D. Turney, “Supercomputer runs largest and most complicated simulation of the universe ever,” livescience.com, Feb. 13, 2025. [https://www.livescience.com/technology/computing/supercomputer-runs-largest-and-most-complicated-simulation-of-the-universe-ever](https://www.livescience.com/technology/computing/supercomputer-runs-largest-and-most-complicated-simulation-of-the-universe-ever)

[^6]: “GE Aerospace Runs New Engine Architecture Simulations on Frontier Exascale Supercomputer - High-Performance Computing News Analysis | insideHPC,” High-Performance Computing News Analysis | insideHPC, Jun. 19, 2023. [https://insidehpc.com/2023/06/ge-aerospace-runs-one-of-the-worlds-largest-supercomputer-simulations-to-test-revolutionary-new-open-fan-engine-architecture/](https://insidehpc.com/2023/06/ge-aerospace-runs-one-of-the-worlds-largest-supercomputer-simulations-to-test-revolutionary-new-open-fan-engine-architecture/)

[^7]: “Early Frontier users seize exascale advantage, grapple with grand scientific challenges | ORNL,” www.ornl.gov. [https://www.ornl.gov/news/early-frontier-users-seize-exascale-advantage-grapple-grand-scientific-challenges](https://www.ornl.gov/news/early-frontier-users-seize-exascale-advantage-grapple-grand-scientific-challenges)

[^8]: T. P. Morgan, “Brad McCredie Is The Pedal To AMD’s Datacenter GPU Metal,” The Next Platform, Jan. 24, 2025. [https://www.nextplatform.com/2025/01/24/brad-mccredie-is-the-pedal-to-amds-datacenter-gpu-metal/](https://www.nextplatform.com/2025/01/24/brad-mccredie-is-the-pedal-to-amds-datacenter-gpu-metal/)

[^9]: El Capitan Stays on Top, US Holds Top 3 Supercomputers, Europe Expands in Leadership HPC” Top500.org, 2025. [https://insidehpc.com/2025/06/top500-el-capitan-stays-on-top-us-holds-top-3-supercomputers-europe-expands-in-leadership-hpc/](https://insidehpc.com/2025/06/top500-el-capitan-stays-on-top-us-holds-top-3-supercomputers-europe-expands-in-leadership-hpc/)
