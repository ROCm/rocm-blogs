---
blogpost: true
date: 10 June 2024
tags: Partner Applications
category: Ecosystems and Partners
language: English
myst:
  html_meta:
    "description lang=en": "Stone Ridge Expands Reservoir Simulation Options with AMD Instinct™ Accelerators"
    "keywords": "Stone Ridge, SRT, ECHELON, HIP, Instinct GPU, HPC, MI210, MI250, MI300X, ROCm"
    "property=og:locale": "en_US"
---

# Stone Ridge Expands Reservoir Simulation Options with AMD Instinct™ Accelerators

Stone Ridge Technology (SRT) pioneered the use of GPUs for high performance reservoir simulation (HPC) nearly a decade ago with ECHELON, its flagship software product. ECHELON, the first of its kind, engineered from the outset to harness the full potential of massively parallel GPUs, stands apart in the industry for its power, efficiency, and accuracy. Now, ECHELON has added support for AMDInstinct accelerators into its simulation engine, offering new flexibility and optionality to its clients.

Reservoir simulators model the flow of hydrocarbons and water in the subsurface of the earth in the presence of wells. Energy companies use them to create and assess field development strategies. Emily Fox, SRT Director of Communications remarks, "SRT pioneered the integration of GPU accelerators in high-performance reservoir simulations and continues to lead by now offering AMD Instinct as a computational platform." SRT’s CTO and ECHELON developer Ken Esler adds, "Many in the field were skeptical about the efficacy of GPUs, however, ECHELON's GPU-native design overcame these doubts and marked a significant leap in simulation speed and performance, firmly establishing SRT's position in the industry.”

Esler explains, "Over a decade ago we set out to supercharge the capability of a CPU-based in-house simulator at a major oil company. We could see the enormous potential of GPUs with their high memory-bandwidth and floating-point performance. We also realized that to unlock the true potential of GPUs, we needed to build a new simulator from scratch, specifically designed for GPU acceleration. This was the genesis of ECHELON, around 2013. Since then, we've continuously enhanced its features, robustness, and performance, first alone, and since 2018 together with our consortium partner Eni S.p.A"

## Expanding GPU Options to Meet Customer Demands

SRT's latest development effort was to port ECHELON from CUDA to the AMD HIP platform, enabling ECHELON to use AMD Instinct GPUs like the MI210, MI250X, and the
upcoming MI300 Series. This strategic decision broadens ECHELON's hardware compatibility and offers clients increased flexibility and choice for their high-performance
computing needs. Vincent Natoli, SRT’s CEO says, "Companies need flexibility in selecting their hardware technology and should not be locked into a single vendor.
We want our clients to have convenient access to the hardware of their choice when it comes to implementing business critical workflows."

The timing of SRT's decision in Ken Esler’s words: "The impressive specifications of AMD's Instinct processors, particularly the MI210 and MI300, presented an increasingly compelling, competitive solution that caught our attention. The MI210's memory bandwidth at 1.6 terabytes per second was quite competitive, and the MI300's subsequent leap to over five terabytes per second is even more exciting. AMD's innovative approach in processor packaging and its use of chiplets have allowed for larger GPUs without compromising yields, resulting in highly competitive products. The engineering behind these developments is quite impressive."

## The Right Software for a Smooth Port

SRT had been contemplating the idea of adapting ECHELON for use on AMD platforms for a while. "The maturation of the ROCm and HIP ecosystem significantly lowered the barriers to adopting AMD GPUs," says Esler. The integration with ROCm, AMD's open software platform, was essential in ensuring that ECHELON could fully leverage AMD's GPU capabilities. "I appreciated AMD's strategy with ROCm," says Esler. "Instead of creating a new, proprietary and incompatible language for accelerated computing, AMD embraced existing frameworks. That significantly reduced the effort needed to adapt our existing code, allowing us to avoid more complex alternatives that are even further removed from ECHELON."

"We develop most of our code internally, but we do rely on Thrust. AMD’s rocThrust turned out to be an effective drop-in replacement. We were also impressed with the ROCm LLVM compiler and Clang in combination with HIP extensions, which enhanced productivity. The support for debuggers and profilers in ROCm has been beneficial. Overall, we are seeing impressive progress in AMD tool development."

## Collaboration Yields Fast Results

The project began in earnest in spring 2023. Erik Greenwald of SRT described porting from CUDA to HIP as initially straightforward. "We created a wrapper to retarget each build. The initial version was created relatively quickly," he says. Although there were a few challenges, like adjusting from CUDA's 32 warp to AMD's 64 warp, Greenwald found these issues manageable. "It was quite painless and quick to achieve initial results," he reflects. Esler adds, "With AMD's support, we steadily enhanced ECHELON's performance and progress in a very acceptable timeframe. We were pleased and satisfied with the performance."

## ECHELON Gets to Work on AMD Instinct accelerators

ECHELON is developed in a Consortium framework, with charter members SRT and Eni, S.p.A, the Italian integrated energy company. The ECHELON Consortium, is a collaboration of industry partners committed to advancing high-performance subsurface flow simulation and is currently open to participation by new member organizations that would like a role in shaping the development of ECHELON.

Eni S.p.A recently announced its new HPC6 high-performance computing (HPC) system at its Green Data Center. Each of the 3472 computing nodes in HPC6 comprises a 64-core AMD EPYC™ CPU (AMDEPYC) and four high-performance AMD Instinct™ MI250X GPUs, offering unmatched computational efficiency and versatility for a wide array of applications.

“With a peak computing power of over 600 PetaFlop/s, HPC6 reaffirms Eni's leadership position in the field of supercomputing among industrial entities,” says Sergio Zazzera, Head of Technical Computing for Geosciences & Subsurface Operations, Eni. “It enables highly complex simulations with enormous volumes of data, such as those needed for studying new geological basins, forecasting sub-surface flows in complex geologies, researching new materials for CO2 capture, and ensuring plasma stability in the field of magnetically confined fusion. Additionally, HPC6 will take advantage of specialized Generative AI solutions in the energy sector.”

So, what's ahead for the world's fastest GPU-powered reservoir simulation software? According to Esler, "We still have the lead in performance relative to our competitors by a good margin, but we are always looking to enhance ECHELON's performance and extend its features." There are also exciting prospects in fields beyond traditional hydrocarbon recovery, such as CO₂ and hydrogen storage, emerging areas that represent new frontiers for ECHELON as the energy sector moves towards greener technologies.
