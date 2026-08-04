---
blogpost: true
blog_title: "Introducing AMD CDNA 5 and the AMD Helios Rackscale Solution"
date: 04 Aug 2026
author: 'Michael Roy, Maanasa Mohanambal Sathianarayanan, Evan Groenke, Mark Chubb, Vamsi Alla, Taniya Siddiqua, Abhishek Vashisth'
thumbnail: 'helios-thumbnail_v3.png'
tags: AI/ML, Hardware, Performance, GenAI
category: Ecosystems and Partners
target_audience: All / Any
key_value_propositions: This blog introduces the AMD CDNA 5 architecture, the AMD Instinct MI455X GPU, and the AMD Helios rackscale solution as an open, integrated platform for rack-scale AI infrastructure, including the fabric redundancy, Virtual Pods, and tray-level serviceability that keep a 72-GPU rack running at scale.
language: English
myst:
    html_meta:
        "author": "Michael Roy, Maanasa Mohanambal Sathianarayanan, Evan Groenke, Mark Chubb, Vamsi Alla, Taniya Siddiqua, Abhishek Vashisth"
        "description lang=en": "Introducing AMD CDNA 5, the AMD Instinct MI455X GPU, and the AMD Helios rackscale solution: an open, integrated platform for rack-scale AI."
        "keywords": "CDNA 5, Instinct, MI455X, Helios, EPYC Venice, Pensando, ROCm, rackscale, AI infrastructure, HBM4, UALink, UALoE, Ethernet, resiliency, Virtual Pods, serviceability"
        "property=og:locale": "en_US"
        "vertical": "AI, Systems, Developers"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Ecosystem and Partners"
        "amd_blog_hardware_platforms": "Instinct GPUs, EPYC Server Processors, Pensando Network Infrastructure"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, AI Inference, Deploying AI at Scale"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
        "amd_blog_authors": "Michael Roy, Maanasa Mohanambal Sathianarayanan, Evan Groenke, Mark Chubb, Vamsi Alla, Taniya Siddiqua, Abhishek Vashisth"
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

# Introducing AMD CDNA™ 5 and the AMD Helios™ Rackscale Solution

AI infrastructure is being redefined at the scale of the entire rack. This blog will walk you through how the AMD CDNA™ 5 architecture, the AMD Instinct™ MI455X GPU, and the AMD Helios rackscale solution come together as a single open platform for large-scale AI training and inference. You will see how AMD balances compute, memory, and networking across 72 GPUs, how open standards keep the platform flexible, and how the AMD ROCm™ software stack ties it all into a complete, deployable system, from silicon to rack. By the end, you will understand what changes when AI infrastructure is designed as a unified rack rather than a collection of discrete GPU servers, and where AMD is taking it next.

## AI Infrastructure Must Evolve Beyond the GPU

Artificial intelligence has become one of the largest drivers of computing innovation in history. Foundation models are rapidly growing, context windows continue to expand, and AI factories are driving the next AI infrastructure shift.

As models have grown, simply building faster GPUs is no longer enough. Performance now depends on the entire system: compute, memory, networking, software, power, cooling, and serviceability all working together as a *unified platform*.

AMD addresses this need with an integrated AI solution that combines the AMD CDNA™ 5 architecture, AMD Instinct™ MI455X GPUs, AMD EPYC™ "Venice" processors, AMD Pensando™ networking, ROCm™ software, and the AMD Helios rackscale platform.

Together they deliver an open, standards-based platform designed for rack-scale AI infrastructure.

## Reimagining AI Compute

At the heart of AMD Helios is the AMD Instinct™ MI455X GPU, designed specifically for the next generation of AI training and inference.

Built on the AMD CDNA™ 5 architecture (shown in Figure 1, below), the MI455X GPU advances nearly every part of the GPU — compute, memory, and utilization alike.

- Up to **4x greater AI compute throughput** for key low-precision formats <sup>MI400-006</sup>
- **432 GB HBM4** memory
- **2.91x memory bandwidth** <sup>MI400-008</sup>
- Larger caches and local memories
- Improved utilization through architectural enhancements

The AMD architectural choices are built around what accelerates real workloads best, today and as models continue to scale. As part of the architectural enhancements made, CDNA™ 5 adds support for the E5M3 floating-point format for scale, the unused sign bit is used to extend the range with an extra exponent bit that improves accuracy in real workloads. Additionally, CDNA™ 5 introduces a 4-bit Tensor Lookup Table (LUT) instruction for low-precision compute. Alongside the new floating-point scale format, the 4-bit LUT instructions efficiently convert compressed 4-bit tensor values into FP4, FP6, or FP8 MXFP compute formats. These instructions execute on the Vector ALU (VALU) and perform dequantization and optional data preprocessing immediately before matrix multiplication, reducing preprocessing overhead and improving Tensor Core utilization.

Taken together, these architectural choices reflect a broader principle behind CDNA™ 5: improving the balance between compute, memory, and communication — the three resources that increasingly determine AI performance.

<!-- markdownlint-disable -->
```{figure} ./images/mi455x-die.png
:align: center
:width: 800px
:alt: AMD Instinct MI455X GPU built on the AMD CDNA 5 architecture
Figure 1. The AMD Instinct™ MI455X GPU, built on the AMD CDNA™ 5 architecture
```
<!-- markdownlint-restore -->

## Memory Designed for Large Models

Modern AI workloads increasingly rely on high-bandwidth memory and massively parallel compute, in addition to traditional CPU-based compute, to deliver new capabilities and greater model efficiency.

CDNA™ 5 expands this horizon with a substantially redesigned memory hierarchy.

HBM4 increases both capacity and bandwidth to support the largest models and context windows, while serving more concurrent inference sessions.

HBM4 memory brings both a capacity and bandwidth step-up, and AMD has continued to work closely with its memory partners to improve qualification and quality across generations. CDNA™ 5 also adds refined page-replacement that builds on what shipped in prior generations, giving the platform more proactive protection against memory-level faults as they are detected in the field.

Inside the GPU, larger L2 caches, Tensor Data Movers, and expanded register files optimize data movement, keeping more data closer to the compute engines that need it.

Together, these innovations improve memory subsystem efficiency, extracting more effective performance from available memory bandwidth to accelerate memory-bound workloads such as LLM decode.

## From Eight GPUs to an Entire Rack

Perhaps the largest architectural shift is how AMD approaches scaling.

Earlier AI systems were typically built by connecting servers containing eight GPUs. AMD Helios shifts the architectural boundary from the server to the rack.

Built on Open Compute Project (OCP) Open Rack Wide (ORW) standards, AMD Helios transforms 72 GPUs into a unified AI system while simplifying deployment, servicing, and future infrastructure evolution, as shown in Figure 2, below.

ORW was developed through the Open Compute Project in partnership with Meta as an open industry standard, with AMD Helios as its first adopter — a rack-scale form factor purpose-built for the space, power, and cooling demands of 72-GPU liquid-cooled density. The fundamental unit of AI compute is no longer the chip or even the server — it is the rack.

<!-- markdownlint-disable -->
```{figure} ./images/helios-rack.png
:align: center
:width: 400px
:alt: AMD Helios rackscale solution
Figure 2. The AMD Helios rackscale solution unifies 72 AMD Instinct™ MI455X GPUs into a single AI system
```
<!-- markdownlint-restore -->

## Built on Open Networking

As AI factories continue to grow, scale-up and scale-out networking become just as important as GPU performance.

AMD Helios embraces an open networking strategy based on industry-standard Ethernet, delivering up to 43 TB/s scale-out bandwidth.

AMD Pensando™ AI-NICs and an open Ethernet ecosystem allow customers to scale efficiently from a single rack to large AI clusters.

Within the rack, Ultra Accelerator Link™ over Ethernet (UALoE) delivers up to **260 TB/s** of scale-up bandwidth for GPU communication. Across racks, Ultra Ethernet Consortium (UEC) technologies and the Open ESUN framework provide an open, standards-based approach to scale-out networking without requiring proprietary fabrics.

Rather than locking customers into a single vendor, AMD enables an open ecosystem of switches, NICs, and management software.

## AI Infrastructure Beyond Silicon

AMD Helios is more than a collection of GPUs.

Each rack integrates:

- 72 AMD Instinct™ MI455X GPUs
- **31 TB of HBM4** GPU memory
- **18** AMD EPYC™ "Venice" processors, each with up to 256 cores
- **Up to 36 TB of DDR5 memory**
- **260 TB/s** high-bandwidth scale-up switching
- Open Ethernet scale-out networking
- Rack-level management
- Direct liquid cooling
- Tray serviceability, observability, and telemetry
- Wave32 compute execution

The platform is designed not only for performance, but also for deployment, operation, and maintenance at hyperscale.

Integrated telemetry, health monitoring, power and thermal management, and modular serviceability help operators maintain availability while reducing operational complexity.

## Architected for Rack-Scale Resiliency

When 72 GPUs, 18 compute trays, 6 switch trays, and thousands of high-speed differential connections come together inside a single system boundary, hardware events stop being an exception — they become something the architecture must account for, end to end, at every layer of the design. AMD Helios was engineered around that principle from the start: contain faults close to where they occur, keep the rest of the rack running, and make repair itself a routine, low-disruption event rather than a scheduled outage.

AMD Helios runs a single-hop, multi-plane UALink™ over Ethernet (UALoE) scale-up fabric: all 72 GPUs reach each other through 12 UALoE switches across 6 switch trays, with multiple paths between any two endpoints by design. Losing a link, cable, or switch triggers automatic rerouting — maintaining packet delivery and communication continuity through redundant paths and automated failover mechanisms with limited bandwidth loss. AMD Fabric Manager and Network OS handle provisioning, telemetry, and event response, so any reroute is detected, logged, and fully traceable rather than becoming a silent source of long-term performance degradation.

Figures 3 through 6, below, show how the fabric behaves as failures escalate from a single link to an entire switch tray.

<!-- markdownlint-disable -->
```{figure} ./images/fabric-normal.png
:align: center
:width: 663px
:alt: Scale-up network under normal operation
Figure 3. Normal operation: up to 3.6 TB/s (bi-directional) per GPU. Transient failures are handled without packet drop
```

```{figure} ./images/fabric-link-loss.png
:align: center
:width: 663px
:alt: Scale-up network with a single link lost
Figure 4. Single link lost: DMA reroutes automatically (~3% bandwidth impact) and the fabric manager rebalances CU traffic in ~1s
```

```{figure} ./images/fabric-switch-loss.png
:align: center
:width: 663px
:alt: Scale-up network with one of twelve switches lost
Figure 5. One of 12 switches lost: DMA reroutes automatically (~8% bandwidth impact)
```

```{figure} ./images/fabric-tray-loss.png
:align: center
:width: 663px
:alt: Scale-up network with one of six switch trays lost
Figure 6. One of 6 switch trays lost: DMA reroutes automatically (~17% bandwidth impact)
```
<!-- markdownlint-restore -->

Additionally, an AMD Helios rack can be divided into isolated "Virtual Pods", or vPods, where subsets of nodes are combined to form a vPod, as shown in Figure 7, below. These nodes are isolated from other vPods so a GPU-level fault stays contained to its vPod — only the affected workload restarts from its last checkpoint, while every other vPod on the rack keeps running.

<!-- markdownlint-disable -->
```{figure} ./images/helios-vpods.png
:align: center
:width: 700px
:alt: AMD Helios Virtual Pods partition a rack into isolated node groups
Figure 7. Virtual Pods (vPods): flexible, secure, and resilient
```
<!-- markdownlint-restore -->

Servicing is designed around the same principle at the tray level. Each tray is an independent, hot-pluggable unit. For example, a switch or compute tray can be removed with no rack shutdown, no coolant drain, and no disruption to the other 23 trays — rear blind-mate, quick-disconnect liquid cooling connects and disconnects as the tray seats, and four dedicated cable cartridges carry the scale-up fabric so servicing switches never means disturbing compute, or vice versa. Telemetry-driven fault isolation and RAS diagnostics across rack, tray, fabric, and device mean a technician arrives already knowing which module, in which tray, needs attention.

None of this works without a physical layer that holds up at scale. Connecting 72 GPUs at uniform bandwidth means high-speed signals crossing connectors, boards, and cable cartridges in a rack that have to behave identically across thousands of builds. AMD Helios addresses this with blind-mate connections that seat hundreds of differential pairs in a single motion rather than by hand, flyover cabling that routes critical high-speed signals off the PCB to cut insertion loss, and retimers that provide additional signal margin to enable robust operation across manufacturing tolerances, connector variation, aging, temperature, and other environmental conditions. At rack scale, ensuring consistent, reliable operation across thousands of high-speed copper links requires balancing performance, manufacturability, yield, and long-term reliability.

Flyover cabling further strengthens the design by moving the highest-speed signals off the PCB, minimizing insertion loss and improving signal integrity. This approach simplifies high-speed channel design, improves manufacturing yield, and enhances long-term reliability for hyperscale deployments — benefits that are often more valuable to customers than minimizing component count.

Power and cooling design carry the same reliability focus as the fabric and trays. Each AMD Helios rack pairs redundant power supplies with capacitor banks that hold the system up long enough to shut down gracefully if facility power is interrupted, and the platform supports either N+1, N+N, or 4M3 (4 to make 3) power-shelf redundancy so operators can match the scheme to their facility. On the cooling side, direct liquid cooling captures roughly 89% of rack heat before it ever reaches the room, and integrated, rack-location-aware leak detection is built into the platform rather than added after the fact.

## Software Completes the Solution

Hardware alone is not enough. The AMD ROCm™ software stack enables developers to take advantage of the CDNA™ 5 architecture with minimal effort, while providing an open environment for AI frameworks, communication libraries, runtime software, and developer tools.

That software runs as a layered stack matched to the rack, not just the GPU. At the platform level, ROCm™ and the UALoE runtime handle GPU drivers, AI libraries, and framework support. At the fabric level, AMD Fabric Manager configures the UALoE scale-up network, manages Virtual Pods, and drives the routing and failover behavior described above, with its agent embedded directly in switch silicon. At the rack level, AMD Rack Infrastructure Manager gives operators unified, Redfish-based visibility into power, thermal, and health telemetry across all 24 trays. And at the cluster level, native integration with Slurm, Kubernetes, and Model-as-a-Service tooling lets AMD Helios racks be commissioned, scheduled, and managed as part of a larger AI factory rather than as standalone boxes.

Together, ROCm™, AMD Instinct™ MI455X, and AMD Helios deliver a complete AI platform, from silicon to rack. Software is never finished, and we will keep sharing how ROCm™ and the AMD Helios software stack evolve.

## Looking Ahead

As AI infrastructure evolves from individual accelerators into complete rack-scale systems, architectural innovation must extend beyond the GPU itself. The AMD CDNA™ 5 architecture, AMD Instinct™ MI455X GPU, and AMD Helios rackscale solution represent the next step by AMD toward building open, scalable AI infrastructure capable of supporting frontier-model training, large-scale inference, and the next generation of AI factories.

Expect this platform to keep expanding as the open ecosystem around it grows, with more open networking options, deeper ROCm™ integration, and additional rack-scale configurations. Future ROCm blogs will explore these innovations in greater technical depth through architecture deep dives and deployment guidance.

## Summary

In this blog, you explored how AMD unifies the AMD CDNA™ 5 architecture, the AMD Instinct™ MI455X GPU, AMD EPYC™ "Venice" processors, AMD Pensando™ networking, and the AMD ROCm™ software stack into the AMD Helios rackscale solution — a single open platform that scales from eight accelerators to 72 GPUs in one rack. You saw how CDNA™ 5 rebalances compute, memory, and communication, including a 4-bit LUT architecture and E5M3 support built around real workload performance; how HBM4 and a redesigned memory hierarchy keep the largest models from becoming memory-bound; how open, Ethernet-based scale-up and scale-out networking let customers grow without proprietary fabrics; and how single-hop fabric redundancy, isolated Virtual Pods, and tray-level field serviceability keep a 72-GPU rack running through the hardware events that come with operating at that scale. Together, these technologies deliver a complete AI platform, from silicon to rack.

To go deeper into the compute, memory, networking, software, and rack design behind AMD Helios, use the resources below, and check back on the ROCm blog for follow-on posts as the platform rolls out.

## Additional Resources

- [Introducing AMD CDNA™ 5 Architecture: Enabling the Future of Frontier AI With the AMD Helios Rackscale Solution](https://www.amd.com/content/dam/amd/en/documents/products/technologies/cdna/amd-cdna5-whitepaper.pdf) (whitepaper)
- [AMD Instinct™ CDNA™ 5 Instruction Set Architecture Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf) (ISA)

## Endnotes

The performance claims in this blog reference the AMD endnotes below.

- **MI400-006** — Based on AMD Performance Labs calculations (June 2026) using an AMD Instinct MI455X GPU, peak theoretical precision performance (FP32, FP16, BF16, MXFP6, MXFP8, FP8, MXFP4 Matrix/Vector), compared to published specifications for AMD Instinct MI355X, MI350X, MI325X, MI300X, MI250X, and MI100 GPUs. Results may vary by system configuration and datatype.
- **MI400-008** — Calculations by AMD Performance Labs in June 2026, based on the published memory capacity and memory bandwidth specifications of an AMD Instinct MI455X GPU vs the published memory capacity and memory bandwidth specifications of AMD Instinct MI355X, MI350X, MI325X, MI300X, MI250X and MI100 GPUs, respectively. System manufacturers may vary configurations, yielding different results.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.

THIS INFORMATION IS PROVIDED "AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2026 Advanced Micro Devices, Inc. All rights reserved
