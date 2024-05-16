---
blogpost: true
date: 16 May 2024
tags: Partner Applications
category: Ecosystems and Partners
language: English

myst:
  html_meta:
    "description lang=en": "AMD Collaboration with the University of Michigan offers High Performance Open-Source Solutions to the Bioinformatics Community"
    "keywords": "University of Michigan, HPC, Bioinformatics, Minimap2, DCGPU, MI210, "
    "property=og:locale": "en_US"
---

# AMD Collaboration with the University of Michigan offers High Performance Open-Source Solutions to the Bioinformatics Community

## The Beginning

Long read DNA sequencing technology is revolutionizing genetic diagnostics and precision medicine by helping us discover structural variants and assemble whole genomes.
It also helps us study evolutionary relationships. Lower sequencing costs and high-throughput portable long read sequencers are revolutionizing precision medicine today.
Long read sequencers from the top manufacturers including Oxford Nanopore (ONT) and PacBio, can produce reads that are much longer than previous generations of sequencers.  However, long reads vary in length and are significantly more error prone than short reads. Sequence alignment (on CPUs) is one of the main bottlenecks in long read processing workflows.

We are thrilled to share the success story of a 1.5-year collaboration between AMD and the University of Michigan, Ann Arbor where we used the AMD Instinct™ GPUs and
ROCm™ software stack to optimize the sequence alignment bottleneck in long read processing workflows. This partnership began in 2022 when Dr. Gina Sitaraman, AMD
reached out to Prof. Satish Narayanasamy from the University of Michigan to collaborate on accelerating and open-sourcing Minimap2, the state-of-the-art aligner
used in long-read DNA sequencing. Minimap2 is slow on the CPU and the best performing accelerated version of Minimap2 is closed-sourced which makes it difficult
to tweak it to work on various hardware platforms. They formed a team including current UM PhD students Juechu “Joy” Dong, Xueshen Liu and brought in Dr. Harisankar
Sadasivan (now at AMD and previously a PhD student of Prof. Narayanasamy) to co-advice the student team to optimize and accelerate Minimap2 on AMD GPUs.

## Advancing together

The collaboration began with Dr. Sitaraman and Dr. Sadasivan mentoring the student team as they ported Minimap2 to AMD GPUs to satisfy their undergraduate
program’s major design project requirement. Various stakeholders within AMD and the University of Michigan joined hands to make this collaboration successful.
The project was funded by the AMD Data Center GPU (DCGPU) business unit in the Summer of 2022 with the help of the HPC Covid Fund team associated with AMD Research.
To facilitate the project, the AMD DCGPU HPC Application Solutions team provided a workstation with an AMD Radeon™ GPU to the students’ lab.
This enabled the students to port the code using ROCm, the AMD open-source software platform for accelerated computing. AMD also provided access to AMD Instinct
MI210 GPUs via the AMD Accelerator Cloud cluster for further tuning and optimization. The students continued to work on the project in their free time after the
semester, demonstrating their dedication and commitment to the project.

Minimap2’s workload is highly irregular in size (long reads of varying lengths) and control flow, and is memory bound. Chaining is the bottleneck step in Minimap2 and
the project found takes up to 68% of the time on a CPU. GPUs are not traditionally built to deliver high performance gains right out of the box for such HPC workloads.
Therefore, several iterations of profiling, strategizing, and optimizing are needed. While Dr. Sitaraman patiently advised the students to understand the software stack,
profile and investigate the performance bottlenecks on the GPU as the designated industry mentor, Dr. Sadasivan helped them understand and implement various strategies
from prior works to further regularize the workload. Joy and Xueshen, under the guidance of their advisor, Prof. Narayanasamy, conducted a thorough investigation and
developed a new “segmentation” method to better balance the workload by breaking down long reads into smaller pieces.
This innovative approach, along with other techniques, effectively balances the irregular workload in the chaining step on the hundreds of compute units in the GPU.

Additionally, the team’s findings show that “chaining scores generated per second” is a better metric than bases per second for evaluating the performance of chaining.
In tests performed by the students, mm2-gb consistently delivered a speedup of 2.57-5.33x for chaining of long nanopore reads (10kb-100kb) and 1.87x on ultra-long reads
(100kb-300kb) on AMD Instinct MI210 GPUs. This was in comparison to mm2-fast running on 32 Intel®<sup>1</sup> Icelake cores with AVX-512.
Importantly, these speedups were achieved without compromising accuracy, making it easy for Minimap2 users to adopt mm2-gb.

<sup>1</sup> Intel is a trademark of Intel Corporation and it's subsidiaries

## Driving scientific progress

The team’s work has been recognized by the broader scientific community, with their paper accepted for publication at the BioSys’24 workshop,
part of the ACM ASPLOS conference. This is a testament to the quality and impact of their work. A copy of the paper can be obtained at
[https://doi.org/10.1101/2024.03.23.586366](https://doi.org/10.1101/2024.03.23.586366). The team is now committed to focusing on end-to-end performance of Minimap2.

The mm2-gb application represents a significant advancement in the field of genetic diagnostics, where long-read DNA sequencing is becoming increasingly popular.
This project supports life sciences research for cancer diagnosis and studying evolutionary relationships between two genome sequences.
mm2-gb accelerates the chaining step of minimap2 on GPU without compromising mapping accuracy, offering a valuable tool to the bioinformatics community.
This successful collaboration highlights the AMD commitment to fostering relationships with universities and building the next generation of scientists who code directly on our GPUs. AMD is also committed to advancing state-of-the-art solutions through contributions to open-source software.
We are proud to be part of initiatives that drive scientific progress and look forward to future collaborations.
The mm2-gb software is open-sourced and available on GitHub at [https://github.com/Minimap2onGPU/minimap2](https://github.com/Minimap2onGPU/minimap2).

![GLM schematic](images/uom-pic-1.jpg)

## Disclaimers

Disclaimer
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

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED
“AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR
ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY
DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

© [2024] Advanced Micro Devices, Inc. All rights reserved.
