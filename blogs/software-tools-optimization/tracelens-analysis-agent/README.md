---
blogpost: true
blog_title: "Automating Performance Bottleneck Identification with the TraceLens Agent"
date: "30 Sep 2026"
author: "Tharun Adithya Srikrishnan, Ahmed Hasssan, Kyle Hoffmeyer, Deval Shah, Mohammad Abdul Basit, Adeem Jassani, Gabriel Weisz, Steven K. Reinhardt"
thumbnail: 'TraceLens_Agent.png'
tags: "Optimization, Performance, AI/ML"
category: "Software tools & optimizations"
target_audience: "Developers and customers looking to identify performance bottlenecks in large model training and inference workloads"
key_value_propositions: "Agentic system for automated performance bottleneck identification"
language: English
myst:
    html_meta:
        "author": "Tharun Adithya Srikrishnan, Ahmed Hasssan, Kyle Hoffmeyer, Deval Shah, Mohammad Abdul Basit, Adeem Jassani, Gabriel Weisz, Steven K. Reinhardt"
        "description lang=en": "TraceLens Agent is an agentic workflow that examines GPU traces to deliver a prioritized optimization report, detailing root causes and fixes."
        "keywords": "Performance Optimization, Agentic System, TraceLens"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training, Deploying AI at Scale"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Tharun Adithya Srikrishnan, Ahmed Hasssan, Kyle Hoffmeyer, Deval Shah, Mohammad Abdul Basit, Adeem Jassani, Gabriel Weisz, Steven K. Reinhardt"
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

# Automating Performance Bottleneck Identification with the TraceLens Agent

GPUs are valuable resources, and it is imperative to make training and inference workloads on them as efficient as possible. Bottlenecks in these workloads can arise from multiple sources, and mitigations vary depending on the source. For example, a kernel may execute well below its achievable throughput, a collective may remain exposed on the critical path, or the device may sit idle while waiting on host-side dispatch. Identifying these bottlenecks is the first step to improving end-to-end performance.

Profilers such as the one built into PyTorch provide visibility into performance at both the model and kernel levels. However, traces in a profile may contain thousands of CPU events and kernel records. Distilling this volume of information into a concise, prioritized set of optimization opportunities remains largely a laborious manual process. The TraceLens Agent is an autonomous performance bottleneck identification system designed to accelerate model bring-up and gap analysis across different deployments, configurations, and hardware platforms.

In this blog, you will learn how the TraceLens Agent turns a raw PyTorch profile into a prioritized, evidence-backed optimization report. You will explore how to run the workflow to find where your training and inference workloads are leaving performance on the table. Additionally, you will understand how the approach leverages modules in TraceLens, and what the agentic layer on top adds.

## TraceLens

[TraceLens](https://rocm.blogs.amd.com/software-tools-optimization/tracelens/README.html) is an open-source Python library for performance analysis of training and inference workloads. It processes profiler traces created by PyTorch, JAX, and AMD RocProf, analyzes each kernel, and determines that kernel's contribution to the overall runtime along with achieved FLOPs and bandwidth. TraceLens analysis starts with the Trace2Tree module, which constructs a hierarchical event tree that links each Python operation and each GPU kernel back to the CPU-side dispatch that launched it, along with the relevant tensor shapes and call stacks. This representation makes it possible to analyze performance at multiple levels of abstraction, from model operators down to individual GPU kernels.

TraceLens uses a roofline model to convert each kernel's shapes and execution time into arithmetic intensity. Above the ridge point, where execution transitions from memory-bound to compute-bound, the relevant ceiling is achievable matrix throughput. Below the ridge point, the relevant ceiling is achievable HBM bandwidth. In addition to roofline analysis, TraceLens includes tools called NcclAnalyser and TraceDiff for collective communication analysis and gap analysis, respectively.

## Automated Analysis

Interpreting the analysis provided by TraceLens's analysis capabilities requires the judgment of an engineer. In practice, this means repeating the same reasoning process for every trace.

- **Data interpretation:** Kernel duration and invocation frequency are not sufficient in isolation. Each kernel must be analyzed relative to its roofline bound to determine whether its achieved efficiency is expected or indicative of unrealized optimization headroom. Gap analysis requires attributing performance differences between two workloads to specific kernels and their characteristics.

- **Reasoning:** Performance regressions are rarely explained by a single metric. Tensor shapes, precision mode, backend selection, launch path, and call stack all shape execution behavior. Furthermore, a single logical operator may lower into multiple kernels, so the analysis must contextualize the conclusion.

- **System-level bottlenecks:** Important inefficiencies may not be attributable to any single kernel. Examples include host-side dispatch stalls, exposed communication, memory transfer overhead, poor compute-communication overlap, and sequences of short kernels that are natural fusion candidates.

- **Prioritization:** Candidates must be ranked so that engineering effort is directed toward changes with the largest impact.

While the number of models, execution configurations, and hardware backends continues to grow, expert trace-analysis capacity remains constrained. The TraceLens Agent is designed to automate the triage stage. The result is a workflow in which engineers can spend less time on diagnosis and more time on implementing optimizations.

## TraceLens Agent

### Overview

The analysis of a workload can be broken down into independent tasks that can be tackled in parallel. This covers tasks related to analyzing compute kernels, identifying system bottlenecks, and uncovering kernel fusion opportunities. The entire analysis is orchestrated through a skill file. A skill file is a structured prompt that an agent harness loads into context when the corresponding capability is requested. The TraceLens Agent skill pre-processes the trace, dispatches expert sub-agents for analysis tasks, and aggregates analyses to return a structured report. This workflow supports both standalone analysis on a single trace and comparative analysis on a baseline-target pair of traces.

A core principle in the agent is to keep numerical computation wherever possible in deterministic code and use the agent harness for interpretation. Trace processing is expensive, so the orchestrator processes the trace at the beginning of the workflow and precomputes the values that the sub-agents will reason over. These precomputed artifacts include the TraceLens performance report, per-category kernel metrics with efficiencies, and cross-kernel signals such as the time split across compute, idle, communication, and memory copy, and kernel fusion candidates. [Fig. 1](#figure-1) shows this orchestration flow end to end, from trace pre-processing through sub-agent dispatch to the aggregated report.

```{figure} ./images/image1.png
:alt: Fig. 1: TraceLens Agent orchestration workflow.
:width: 60%
:align: center

Figure 1: TraceLens Agent orchestration workflow.
```

### Sub-Agents

The workflow fans out into specialized sub-agents, each with its own prompt and isolated context window. Each sub-agent includes a category-specific knowledge base describing common inefficiency patterns and the fixes typically associated with them. That knowledge base is used to help the agent explain the metrics and propose actions. The agent reasons only over observable quantities, and it avoids speculative claims. Sub-agents effectively allow for parallelized execution while limiting context growth, as shown in [Fig. 2](#figure-2), where the compute and system analysis tasks fan out independently.

```{figure} ./images/image2.png
:alt: Fig. 2: Parallelizability across the compute and system analysis tasks.
:width: 45%
:align: center

Figure 2: Parallelizability.
```

The **compute tier** assigns one analyzer to each major kernel class, such as GEMM, attention, Mixture-of-Experts, elementwise, reduction, normalization, convolution, and Triton-generated kernels, along with a generic analyzer for uncategorized cases. Each analyzer scores kernels by their gap to the achievable peak. Compute findings are intended for kernel engineering teams and automated kernel-tuning systems.

The **system tier** focuses on cross-kernel optimization opportunities. An idle CPU analyzer characterizes host-side stalls and device idle gaps. A multi-kernel analyzer examines transfer patterns and the degree of overlap between communication and compute. A kernel-fusion analyzer identifies short sequences of kernels inside a module that could potentially collapse into a single launch. They are activated conditionally, only when relevant conditions are detected, such as significant idle time, materially exposed communication or memory-copy overhead, or the presence of at least one kernel-fusion candidate. System findings target model owners, serving-infrastructure owners, and any automated component that updates model code or runtime configuration.

The orchestrator and its sub-agents exchange data through disk using JSON, CSV, and intermediate markdown files with strictly enforced schemas. This disk-backed exchange serves as persistent workflow state, which is useful in longer runs where an agent may lose context established in earlier steps.

### System Reliability

The TraceLens Agent includes multiple guardrails to improve robustness. The first guardrail is the use of templates. The orchestrator and the sub-agents fill out a fixed set of sections in templates rather than generating unconstrained free-form text. As a result, each run produces the same sections in the same order. This makes the report predictable both for human readers and for downstream tools. The second guardrail is a set of feedback loops with validation at both the sub-agent and orchestrator levels. The orchestrator and sub-agents validate their own outputs and correct local errors. At the prompt level, an evidence guardrail confines each agent to quantities supported by trace data. [Fig. 3](#figure-3) summarizes how these guardrails and the evaluation harness fit together to keep the output robust.

```{figure} ./images/image3.png
:alt: Fig. 3: Robustness through evals and validation harness.
:width: 60%
:align: center

Figure 3: Robustness
```

Finally, the third guardrail is an end-to-end evaluation harness that captures regressions and reports both consistent and flaky issues before each release. Our evaluation framework performs fully automated end-to-end testing to ensure workflow adherence, structural correctness, semantic accuracy, and reproducibility across multiple runs. We execute both unit and end-to-end real-time test cases and assess the agent's outputs using a combination of scripted checks and LLM-based evaluators. The resulting evaluation reports are aggregated and classified to highlight actionable issues. This automated evaluation framework reduces uncertainty, improves reliability, and lowers the agentic token cost required to generate complete and comprehensive reports.

### Analysis Report

The final output of the agent is a single Markdown report. It begins with an executive summary that describes the workload and highlights top-level metrics, then presents findings in three tiers: Compute Kernel Optimizations, Kernel Fusion Opportunities, and System-Level Optimizations. Every finding includes three components: Insight, Action, and Impact, and an impact score ranks it relative to the others. A Detailed Analysis section follows in the same order, with supporting tables and the reasoning behind each recommendation. The report is intentionally both human-readable and machine-consumable. Structured HTML comment markers are emitted alongside ranked items and impact scores. An engineer can thus act on the report directly, or an automated optimization system can process the same file, select the flagged shapes or operators, apply an optimization, and profile the workload again.

The following is an example of a single compute-tier finding from the Detailed Analysis section.

> #### P1: Compute-bound BF16 GEMMs underrunning roofline (Tensile)
>
> **Identification:** 18 BF16 `aten::mm` shapes were flagged as the dominant GEMM cluster in the trace, accounting for 45.7 s of GPU kernel time (~80.6% of compute time). All flagged operations dispatch through the **Tensile** backend with `bound_type = compute`. The cluster spans the large MLP shapes (M=24576, K=8192, N=28672), the gate/up shapes (N=10240), the QKV/output shapes (N=8192), and the LM-head shapes (N=128256). (source: `gemm_metrics.json` &rarr; `category_findings[0].members[]`, `operations[].library`, `operations[].efficiency.bound_type`)
>
> **Data:**
>
> | Operation | Args | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
> | --------- | ---- | --------- | ---- | ----- | ---------- | ---------- | ----- |
> | aten::mm | (24576,8192) x (8192,28672) bf16 | 7607.463 | 13.42 | 320 | 5059.76 | 68.74% of 708 TFLOPS | compute-bound |
> | aten::mm | (24576,8192) x (8192,28672) bf16 | 6636.191 | 11.70 | 320 | 5059.76 | 79.04% of 708 TFLOPS | compute-bound |
> | aten::mm | (28672,24576) x (24576,8192) bf16 | 6313.337 | 11.13 | 320 | 5059.76 | 82.70% of 708 TFLOPS | compute-bound |
> | aten::mm | (24576,28672) x (28672,8192) bf16 | 6071.557 | 10.71 | 320 | 5059.76 | 85.99% of 708 TFLOPS | compute-bound |
>
> **Reasoning for Slowdown:** Every flagged shape has very high arithmetic intensity (FLOPS/Byte 3510-5863), so each GEMM is correctly compute-bound against the BF16 matrix-FP roofline (708 TFLOPS) rather than HBM-bound. The achieved TFLOPS/s sit between 486.7 and 613.5, which is 68.7%-86.7% of peak. The heaviest shape (`(24576,8192) x (8192,28672)`, 7.6 s of kernel time, count = 320) lands at only 68.7%, and a second 6.6 s instance of the same shape only reaches 79.0%, indicating the same logical GEMM is hitting more than one Tensile kernel and at least one is sub-optimal.
>
> **Resolution:** Tile / wave-occupancy tuning targets the specific Tensile kernels selected for these shapes so the same FLOPs are issued under a better-utilized MFMA pipeline, raising achieved TFLOPS/s without changing the algorithm. Narrowing precision (BF16 to FP8/FP4 where the model tolerates it) doubles or quadruples the matrix roofline (708 to 1273 TFLOPS at FP8, higher at FP4), directly lowering the compute floor for these compute-bound shapes. Generate reproducers for the kernel team for the lowest-efficiency shapes (68.7%, 69.7%, 74.0%) so the Tensile selector can be revisited for those tile/M-N-K combinations.
>
> **Impact estimate:**
>
> - Low end impact_score: 12.97
> - High end impact_score: 17.27

## Getting Started

You can use the agent in two ways: directly through TraceLens or through the AMD Skills registry. The skill can run on the agent harness you already use, such as Claude Code, Cursor, or Codex, and can be installed using NPX or the agent harness's skills marketplace.

The first option is through TraceLens directly. The orchestrator and sub-agents live in `TraceLens/Agent/Analysis/skills/analysis-orchestrator/`, and the TraceLens Agent README walks through the full setup, including trace collection and roofline calibration for your hardware. Install TraceLens:

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

The second option is through the AMD Skills registry. When using TraceLens through the AMD Skills registry, TraceLens will be installed into a Python virtual environment, as the skill needs TraceLens to operate.

A standalone analysis of a single trace looks like this:

```text
Follow the analysis orchestrator installed with TraceLens and run the
full agentic analysis workflow on /path/to/trace.json with platform
<platform> and output results to /path/to/output_dir
```

A comparative analysis on two traces looks like this:

```text
Follow the analysis orchestrator installed with TraceLens and run the
full agentic analysis workflow on /path/to/trace1.json with platform
<platform1> and /path/to/trace2.json with platform <platform2> and
output results to /path/to/output_dir
```

## Summary

In this blog you explored the TraceLens Agent, an agentic workflow that turns a raw profiler trace into a prioritized, evidence-backed optimization report. The TraceLens Agent streamlines performance engineering workflows by leveraging TraceLens's performance analysis capabilities, augmented with domain-specific interpretations provided by specialized sub-agents. This tool produces a comprehensive, structured report that ensures evidence-based consistency across executions and provides engineers with immediately actionable recommendations. Additionally, the report's machine-consumable format integrates directly with automated workload optimization systems such as [Hyperloom](https://github.com/AMD-AGI/Hyperloom), an agentic optimization engine that carries these findings through an end-to-end profiling, analysis, optimization, and validation loop. By automating trace interpretation, the TraceLens Agent overcomes the scalability limitations of traditional manual methods, which are increasingly inadequate as models, configurations, and platforms proliferate. For detailed setup instructions and deployment guidance, visit the [TraceLens](https://github.com/AMD-AGI/TraceLens) repository.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
