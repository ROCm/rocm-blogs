---
blogpost: true
date: 16 May 2024
tags: Partner Applications
category: Ecosystems and Partners
language: English

myst:
  html_meta:
    "description lang=en": "Siemens taps AMD Instinct™ GPUs to expand high-performance hardware options for Simcenter STAR-CCM+"
    "keywords": "Siemens, MI200, Instinct GPU, STAR-CCM+, ROCm, AMDMI200, MI250, HPC, AI, High Performance Computing, Deep Learning, (Computational Fluid Dynamics) CFD"
    "property=og:locale": "en_US"
---

# Siemens taps AMD Instinct™ GPUs to expand high-performance hardware options for Simcenter STAR-CCM+

Siemens recently announced that its Simcenter [STAR-CCM+](https://plm.sw.siemens.com/en-US/simcenter/fluids-thermal-simulation/star-ccm/) multi-physics computational fluid dynamics (CFD) software now supports [AMD Instinct™ GPUs](https://www.amd.com/en/products/accelerators/instinct.htmlhttps://www.amd.com/en/products/accelerators/instinct.html) for GPU-native computation. This move addresses its users' needs for computational efficiency, reduced simulation costs and energy usage, and greater hardware choice.

Liam McManus, Technical Product Manager for Simcenter STAR-CCM+, said, "Our customers want to design faster, evaluate more designs further upstream, and accelerate their overall design cycle. To do that, they need to increase the throughput of their simulations."

The Simcenter STAR-CCM+ team was naturally interested in the [AMD Instinct MI200 series](https://www.amd.com/en/products/accelerators/instinct/mi200.html), including the MI210, MI250, and MI250X. McManus said, "We had a lot of familiarity, experience, and success with AMD CPUs. This made us comfortable exploring what we could achieve with AMD GPUs."

## GPUs accelerate bottom lines

The computational intensity of CFD historically burdens traditional CPU-based systems. Whether predicting the airflow around a new car model or optimizing the cooling systems for cutting-edge electronics, there is always a desire for faster design cycles—a challenge for industries where time-to-market and product performance are crucial. McManus added, "Today, it's not just about simulating a component once. A simulation might be run a hundred times to optimize it and get the most efficient product possible."

Siemens found that with AMD Instinct GPUs, CFD simulations that once took days can be completed in hours or even minutes without compromising the depth or accuracy of the analysis. McManus pointed out, "GPU hardware allows us to run more designs at the same hardware cost or start to look at higher fidelity simulations within the same timeframe as before." This newfound speed enables a more exploratory approach to design, allowing engineers to test and refine multiple hypotheses in the time it once took to evaluate a single concept.

## AMD Instinct MI200 GPUs stand apart

The AMD MI200 series innovative [CDNA2 architecture offers high](https://www.amd.com/en/technologies/cdna.html) processing speeds and optimizes energy consumption, allowing for efficient handling of large datasets and complex calculations. Advanced features such as high-bandwidth memory (HBM), scalable multi-GPU connectivity, and enhanced computational precision collectively enhance the GPUs' performance and efficiency across diverse computational tasks. The MI250 is further optimized for the highest performance levels in demanding tasks, including large-scale simulations (HPC), Deep Learning, and complex scientific calculations. Engineered for scalability and massive parallel processing (MPP) abilities, the MI250 excels in High Performance Computing and artificial intelligence (AI) workloads due to its   exceptional computational throughput, memory bandwidth, core count, fast memory, and memory capacity.

"Just one AMD Instinct GPU card can provide the computational equivalent of 100 to 200 CPU cores," said McManus. Of course, we can use multiple GPUs, meaning that we can offer customers significantly reduced per-simulation costs."

Michael Kuron, a Siemens senior software engineer who led the port, emphasized, "One thing that makes AMD GPUs great is their high memory bandwidth. For CFD, we're not really limited by pure numerical performance but by how fast the GPU can shuffle the data. AMD GPUs offer some of the highest memory bandwidth out there, making them an excellent platform for CFD applications." He added, "Some of the world's fastest supercomputers these days use AMD GPUs, so being able to run on them certainly doesn't hurt."

## AMD ROCm and HIP smooth the transition

Of course, hardware was only part of the consideration. McManus said, "The [AMD ROCm platform](https://www.amd.com/en/products/software/rocm.htmlhttps://www.amd.com/en/products/software/rocm.html) has been critical in ensuring that our software could fully leverage the computational power of AMD GPUs. Its open-source nature and comprehensive toolset have significantly eased the development and optimization of our applications."

Kuron added, "Because the entire ROCm stack is open-source, I can look under the hood and fix things without waiting for any technical support." Kuron continued, "In the ROCm ecosystem, all the runtime and math libraries, plus all the stuff built on top of those, are open source. We have excellent insight when new features and capabilities come in."

[ROCm™ software's HIP programing language](https://github.com/ROCm/HIP) enabled a smooth transition of Simcenter STAR-CCM+'s existing codebase. Kuron explained, "Our existing CUDA software translates almost one-to-one to HIP, so the porting effort was much lower than rewriting it in another programming model like SYCL or OpenMP offloading. The actual change between CUDA and HIP was just a couple of hundred lines of code. Probably 95% of the change from CUDA to HIP was achieved using little more than find and replace, and the rest wasn't difficult either."

Kuron said, "Achieving one-to-one parity was a significant milestone that ensures our software delivers precise and reliable results consistently, whether running on AMD or any other hardware."

## Collaborating to serve the customer

Collaboration was pivotal to the project's success. "AMD was very responsive to our feedback, working closely with us to refine the integration," noted Kuron. "The opportunity to communicate directly with the AMD team members who implement these solutions and understand the technical details has been incredibly valuable."

McManus said, "It's great to collaborate with AMD. They're developing the GPU solutions and we can work closely with them to ensure our software runs on it. Siemens and AMD have the same objective: to get the customer to the answer as fast as possible."

## Looking ahead to MI300 and beyond

Looking to the  new AMD Instinct MI300 series, Kuron said, "We're looking forward to the increase in memory bandwidth on the [MI300 platform](https://www.amd.com/en/products/accelerators/instinct/mi300.html). The tighter coupling between CPU and GPU of the MI300A platform could help eliminate bottlenecks and speed up simulations that require some parts to run on the CPU."

McManus adds, "The increase in memory capacity, up to 192 gigabytes for the MI300X, will reduce constraints on simulation complexity and allow larger problem sizes to be addressed more effectively.

We're also exploring hybrid computational strategies for some CPU-bound simulation challenges and we're particularly intrigued by the possibilities offered by the unified memory of the MI300A."

Together, Siemens and AMD are addressing the evolving needs for quicker, more cost-effective design processes. Integrating Simcenter STAR-CCM+ with AMD Instinct GPUs broadens the range of tools available for computational fluid dynamics challenges, offering high simulation speed and cost efficiency and offering engineers a wider array of hardware options. The AMD MI300 series promises to expand these capabilities further, catering to an increasingly diverse and complex array of simulations, and very dynamic markets.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical
inaccuracies, omissions, and typographical errors. The information contained herein is subject to change
and may be rendered inaccurate for many reasons, including but not limited to product and roadmap
changes, component and motherboard version changes, new model and/or product releases, product
differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the
like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or
mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to
the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH
RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES,
ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY
IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR
PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT,
SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION
CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, [insert all other AMD trademarks used in the material here per AMD
Trademarks] and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product
names used in this publication are for identification purposes only and may be trademarks of their
respective companies. [Insert any third party trademark attribution here per AMD's Third Party
Trademark List.]

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD.
ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS
DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

© 2024 Advanced Micro Devices, Inc. All rights reserved.
