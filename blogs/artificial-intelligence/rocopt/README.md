---
blogpost: true
blog_title: "Introducing rocOpt: GPU-Accelerated Decision Optimization on AMD ROCm"
date: "20 Oct 2026"
author: "Anuya Welling, Mukhil Azhagan Mallaiyan Sathiaseelan, Yao Liu, Phani Vaddadi, Vish Vadlamani"
thumbnail: 'rocopt_thumbnail.png'
tags: "AI/ML"
category: "Applications & models"
target_audience: "ai developers and enthusiast"
key_value_propositions: "rocopt brings open-source, GPU-accelerated mathematical optimization to AMD Instinct - solving MILP, LP, QP, and vehicle routing problems with millions of variables in near real-time."
language: English
myst:
    html_meta:
        "author": "Anuya Welling, Mukhil Azhagan Mallaiyan Sathiaseelan, Yao Liu, Phani Vaddadi, Vish Vadlamani"
        "description lang=en": "Run large-scale MILP, LP, QP, and vehicle routing problems on AMD Instinct GPUs - introducing rocopt, optimization built for ROCm."
        "keywords": "rocopt, ROCm, AMD Instinct, MI300, MI355, GPU optimization, mathematical optimization, linear programming, mixed integer linear programming, quadratic programming, vehicle routing, operations research, PDLP"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, Deploying AI at Scale"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Anuya Welling, Mukhil Azhagan Mallaiyan Sathiaseelan, Yao Liu, Phani Vaddadi, Vish Vadlamani"
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

# Introducing rocOpt: GPU-Accelerated Decision Optimization on AMD ROCm

rocOpt is a new open-source, GPU-accelerated engine for decision optimization
on AMD Instinct GPUs. It solves large-scale linear programming (LP),
mixed-integer linear programming (MILP), quadratic programming (QP), and
vehicle routing problems on the ROCm software stack, and is ready for early
users today on MI300X and MI355X with ROCm 7.2.3.

Decision optimization has outgrown what CPU-only solvers can deliver inside a
production time budget. rocOpt lets you run those workloads on the same AMD
Instinct GPUs and ROCm software stack you already use for AI and analytics —
open source, no proprietary license, no platform switch.

In this blog you will learn why decision optimization is moving to the GPU,
how AMD built rocOpt on the AMD Data Science stack to bring GPU-accelerated
optimization natively to ROCm, and what each of its four solver families is
good for. You will also see where rocOpt fits in real workloads — logistics,
energy dispatch, portfolio optimization, supply-chain design — and how to
get started on your own problem today.

## Why GPU-Accelerated Decision Optimization?

Decision optimization is the hidden engine behind complex operational
decisions, from routing delivery fleets and dispatching power generation to
balancing portfolios and scheduling semiconductor production. The common
foundation is **mathematical programming**: expressing a problem as
objectives and constraints, then leveraging powerful solvers to find the
optimal feasible outcome.

Modern optimization problems are massive in scale. A nationwide logistics
model may contain **millions of variables and constraints**, energy dispatch
systems often require a new solution every few minutes, and portfolio
rebalancing must be completed within tight market windows. As these problems
grow, even the best CPU-based solvers encounter runtime limitations: their
core computations — sparse matrix operations, matrix factorizations, and
large-scale reductions — don't parallelize well across a handful of CPU cores
at this scale. These are precisely the types of workloads that GPUs, with
thousands of concurrent threads, excel at accelerating.

rocOpt brings GPU acceleration for this class of work to AMD Instinct
GPUs. It is open source, ships solvers for LP, MILP, QP, and
vehicle-routing problems, and slots into existing ROCm AI/ML pipelines without
leaving the platform.

## Get Started: Running rocOpt on ROCm

The [rocOpt GitHub repository](https://github.com/AMD-Ecosystem/rocOpt/tree/amd-integration) is the starting
point for new users. It contains the source code, examples, and instructions
for getting started and building rocOpt from source on AMD Instinct GPUs.

If you prefer not to compile locally, use the prebuilt containers on [Docker
Hub](https://hub.docker.com/r/rocm/rocopt/tags?name=rocopt-1.0.0.amd0_rocm7.2.3.ubuntu24.04_py3.13) to run rocOpt in a container instead of
building from source.

You can also install rocOpt by following the official [installation guide](https://rocm.docs.amd.com/projects/rocopt/en/latest/install/install_rocopt.html). Before you begin, check
the rocOpt [compatibility guide](https://rocm.docs.amd.com/projects/rocopt/en/latest/install/install_rocopt.html#compatibility-matrix) for supported
platforms, ROCm versions, and hardware.

## Built on the AMD Data Science Stack

rocOpt extends the AMD ROCm platform with GPU-accelerated decision
optimization, built on the [**AMD Data Science**](https://rocm.docs.amd.com/projects/rocm-ds/en/latest/index.html)
ecosystem. That foundation is what makes rocOpt feel native on ROCm — the
same dispatch model, the same memory pool, the same build conventions you
already use for RAPIDS-style workloads.

| Capability             | rocOpt uses        |
| ---------------------- | ------------------- |
| Device runtime          | HIP                 |
| Compiler                | `amdclang++`        |
| Templates / iterators   | rocThrust, hipCUB   |
| Primitive ops           | rocPRIM             |
| ML primitives           | hipRAFT             |
| Memory management       | hipMM               |
| Sparse / dense BLAS     | hipSPARSE, hipBLAS  |
| Direct linear solvers   | hipSOLVER           |

## Solvers Under the Hood

rocOpt ships four solver families, each tuned for a different problem class:

- **PDLP (Primal-Dual Hybrid Gradient for LP)** — A first-order method designed for very large, sparse LPs that runs on the GPU. This is the workhorse for
  million-variable LPs and the recommended solver for most large-scale
  problems on rocOpt.
- **Dual Simplex** — CPU-based, suited to small-to-medium LPs and used as the
  LP relaxation engine inside the MILP branch-and-bound loop.
- **MILP (mixed integer linear programming)** — hybrid GPU/CPU solver. Primal
  heuristics (local search, feasibility pump, feasibility jump) run on the GPU
  to drive the primal bound; branch-and-bound runs on the CPU to drive the
  dual bound. Integer-feasible solutions are shared between both halves.
- **Vehicle Routing (TSP / VRP / PDP)** — metaheuristic solver that generates
  an initial population of routes and iteratively improves them within a
  user-specified time budget, returning the best route found.

**Current solver coverage:** rocOpt currently supports **PDLP** and **dual
simplex** for LP workloads. The barrier (interior-point) solver is not yet
available because its upstream implementation depends on cuDSS; expanding
sparse direct-solver support represents an opportunity for future ROCm
development. For now, use **PDLP** for large LPs and **dual simplex** for
smaller ones.

## Where rocOpt Fits — Concrete Use Cases

A few classes of problems where GPU acceleration changes what is tractable:

**Last-mile logistics (VRP).** Plan deliveries for thousands of stops across
hundreds of vehicles, with time windows, vehicle capacities, and driver
constraints. rocOpt's routing engine produces near-optimal routes within the
dispatch window so you can replan when reality diverges from the morning's
plan.

**Energy dispatch and unit commitment (MILP).** Decide which generators run,
at what output, and when, while respecting ramp limits, reserve requirements,
and transmission constraints. Faster solves enable shorter market intervals
and tighter integration of variable renewables.

**Portfolio optimization (QP).** Choose asset weights to maximize expected
return subject to risk and exposure constraints — a quadratic objective over
thousands of instruments. GPU acceleration shortens the rebalance loop from
minutes to seconds.

**Supply chain network design (MILP).** Decide which facilities to open, which
products to make where, and how to route demand across a multi-echelon
network. Larger instance sizes mean fewer modeling abstractions and better
answers.

## Summary

In this blog you learned how rocOpt opens up GPU-accelerated decision
optimization to the AMD ecosystem. We covered why mathematical programming is
moving to the GPU, introduced rocOpt as a new GPU-accelerated optimization
library in the AMD Data Science stack, and explained what each
of its four solver families — PDLP, dual simplex, MILP branch-and-bound, and
the vehicle-routing metaheuristic — is good for. You also saw where rocOpt
fits in real workloads such as last-mile logistics, energy dispatch, portfolio
optimization, and supply-chain network design, and where to find the source,
docs, and containerized builds to start running it today on AMD Instinct
MI300X and MI355X with ROCm 7.2.3.

Visit the [rocOpt
repository](https://github.com/AMD-Ecosystem/rocOpt/tree/amd-integration) and try it on your next large-scale
optimization problem.

## Acknowledgements

The authors thank the following AMD contributors for their instrumental
support in enabling rocOpt: Diptorup Deb,
Lalith Narasimhan, Pankaj Gupta, David Oladepo, Randy Hartgrove, Amit Kumar,
 Ehud Sharlin, Saad Rahim and Bhavesh Lad.

## Additional resources

rocOpt's own [examples](https://github.com/AMD-Ecosystem/rocOpt/tree/amd-integration/examples_notebook)
in the GitHub repository cover LP, MILP, and vehicle routing through the
Python, CLI, and Server APIs, and are the best starting point for new users.
rocOpt also preserves API compatibility with the upstream cuOpt project, so
existing cuOpt workloads can transition to rocOpt with minimal code changes.
For full API reference material — C/C++, Python, MPS parser, and Server APIs
— see the [rocOpt documentation](https://rocm.docs.amd.com/projects/rocopt/en/latest/index.html).
The main caveat when porting existing cuOpt code is that the Barrier
(interior-point) LP solver is unavailable on AMD today, so use PDLP or dual
simplex instead.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. © 2026 Advanced Micro Devices, Inc. All rights reserved
