---
blogpost: true
blog_title: "AMD Device Metrics Exporter v1.4.2: Enhanced Observability, Deeper RAS Insights, and Smarter GPU Telemetry for Modern HPC & AI Clusters"
date: 23 Mar 2026
author: 'Akhila Yeruva'
thumbnail: 'device_metrics_exporter_thumbnail.png'
tags: Kubernetes, Optimization, AI/ML, HPC
category: Software tools & optimizations
target_audience: Public customers, anyone interested in using Operators
key_value_propositions: Workload management
language: English
myst:
    html_meta:
        "author": "Akhila Yeruva"
        "description lang=en": "Struggling with GPU bottlenecks? Learn how AMD DME v1.4.2 uncovers power, thermal, and RAS issues with actionable, production-ready telemetry."
        "keywords": "kubernetes, device metrics exporter, violation metrics"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "Open-Source Tools"
        "amd_blog_applications": "Deploying AI at Scale"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Akhila Yeruva"
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

# AMD Device Metrics Exporter v1.4.2: Enhanced Observability, Deeper RAS Insights, and Smarter GPU Telemetry for Modern HPC & AI Clusters

Modern GPU‑accelerated systems—whether powering massive AI training workloads or tightly scheduled HPC environments—depend heavily on high‑quality telemetry. Understanding how each GPU behaves under load, how often it hits power or thermal boundaries, and how reliably the hardware performs is central to maintaining performance, diagnosing failures, and tuning systems at scale.

With AMD Device Metrics Exporter (DME) v1.4.2, we are introducing a set of enhancements focused on operational clarity, reliability, and deeper hardware‑level insight. This release continues our effort to equip platform engineers, cluster operators, and developers with the tools they need to see what's happening inside the GPU, with observability aligned to real production challenges.

In this blog, you will explore the key enhancements introduced in AMD Device Metrics Exporter v1.4.2, including process-level visibility with `KFD_PROCESS_ID`, structured RAS insights using AFID, and deep performance diagnostics through violation metrics and clock telemetry. You’ll learn how to interpret these metrics in real-world scenarios, correlate them with workload behavior, and use them to diagnose bottlenecks, improve reliability, as well as optimize GPU performance across HPC and AI clusters.

## Key Enhancements in v1.4.2

### 1. New Runtime Context Labels for Debugging

#### KFD_PROCESS_ID

The exporter now reports `KFD_PROCESS_ID`, exposing the actual Linux process ID that is using a given GPU at that time.

This solves a longstanding gap for bare‑metal systems—particularly Debian‑based environments—where there is no scheduler like Slurm or PBS attaching job metadata to device activity. In those setups, operators often had no straightforward way to answer, _“Which process was using GPU 2 when utilization spiked?”_
With DME v1.4.2, you can now correlate GPU utilization, throttling, and RAS events back to the exact process without guessing or reverse‑mapping command lines.
This addition significantly improves:

- On‑call debugging
- Forensic analysis after failed jobs
- Profiling of ad‑hoc or researcher‑run workloads

This seemingly simple label provides enormous operational value for environments without cluster‑scheduler context.

### 2. AFID‑aware RAS: `GPU_AFID_ERRORS` field

Reliability and fault analysis often require visibility into hardware‑level event signatures. This latest release introduces `GPU_AFID_ERRORS`, a direct mapping to AMD Field Identifier (AFID) events.

#### What is AFID?

AFID is AMD’s standardized list of numeric IDs that categorize GPU reliability events by Category (e.g., Off‑Package Link, HBM, Device Internal), Type (e.g., XGMI, On‑die ECC, Watchdog Timeout), and Severity (Corrected, Uncorrected non‑fatal, Fatal). Think of it as a consistent “error codebook” for AMD Instinct GPUs that maps directly to service actions.
When hardware reports an error, it is often encoded as a CPER (Common Platform Error Record) entry, a UEFI‑standard error format that carries details like source, type, severity, and timestamps. AMD’s management stack (`amd-smi`) can list and decode CPER records and extract AFIDs from them. This makes AFID a machine-parsable bridge between low-level error telemetry and higher-level operations.
The exporter now exposes `GPU_AFID_ERRORS` so Prometheus can count and label these events by AFID, letting you build alerts and runbooks that say _what happened_—not just that a RAS error occurred.

AFIDs provide a structured way of identifying and categorizing GPU reliability issues. Exposing AFID‑aligned metrics means operators can:

- Detect early-warning indicators before a card becomes unhealthy
- Track repeated error patterns tied to specific AFIDs
- Automate hardware retirement or scheduling restrictions
- Improve root-cause analysis for workload failures

This is particularly important for organizations running large clusters of MI3xx series GPUs or operating environments where uptime and predictability matter as much as raw throughput.

AFID visibility brings GPU fleet health monitoring to a much more mature and actionable level.  [AFID Event List](https://docs.amd.com/r/en-US/AMD_Field_ID_70122_v1.0/Introduction)

### 3. Violation Metrics — Practical Insight Into Real Bottlenecks

GPU violation metrics often get overlooked, but they are among the most valuable signals for understanding why a workload isn’t performing the way it should. Violation Metrics fields significantly expand GPU observability across three key metric types – counter/accumulated values, violation status (active/not active), and violation activity (percentage) – covering multiple categories such as:

- Processor hot residency
- PPT (power) limit residency
- Socket thermal residency
- VR (voltage regulator) thermal residency
- HBM thermal residency
- Low utilization patterns
- GFX clock reduction due to power, thermal, or host-defined limits

Per‑compute‑core insights are available for newer violation metrics only – specifically the GFX and low‑utilization fields starting with `amd-smi` 1.8 and later. Earlier violation metrics remain device‑level, while these newer fields let you see constraint behavior at a finer per‑core granularity where supported.

#### What is a Violation Metric?

A “violation” occurs when firmware enforces a constraint that prevents the GPU from operating at its requested performance state. These constraints stem from power limits, thermal protections, voltage regulator conditions, or reliability safeguards. When any of these boundaries are reached, the firmware dynamically reduces clocks or modifies behavior to maintain system stability.

Violation metrics represent residency—the percentage of time during the sampling interval that the GPU was operating under one of these enforced conditions. These are not instantaneous throttle flags; they are time-weighted indicators of constraint enforcement. This distinction is important because it provides causal insight into performance behavior rather than just surface symptoms.

#### Why Residency-Based Metrics Matter

Traditional monitoring often relies on observing clocks or temperatures and inferring throttling behavior. However, clocks alone do not explain why performance was reduced. Violation residency metrics close that gap.

For example, if a power violation residency metric reports 30%, that means the firmware enforced a power limit for 30% of the sampling window. Even if clocks appear nominal at certain instants, the GPU may still be spending a meaningful portion of time under constraint. Residency metrics therefore provide direct visibility into firmware-enforced decisions, transforming performance debugging from inference to data-driven analysis.

#### Real-World Diagnostic Scenarios

##### Scenario 1: Multi-GPU Server Underperforming

- **Symptom:** Training throughput 12% below baseline  
- **Signal:** PPT residency at 42% across all GPUs  
- **Cause:** Power envelope exceeded  
- **Fix:** Adjust rack-level power provisioning  

##### Scenario 2: Intermittent LLM Training Slowdowns

- **Symptom:** Step time spikes  
- **Signal:** HBM thermal residency during attention ops  
- **Cause:** Memory thermal saturation  
- **Fix:** Cooling optimization  

##### Scenario 3: “Slow GPU” Reports with No RAS Errors

- **Symptom:** Customer suspects faulty GPU  
- **Signal:** >70% low‑util residency  
- **Cause:** Data ingestion bottleneck  
- **Fix:** CPU pipeline tuning  

For teams analyzing performance regressions, validating platform images, or tuning cooling profiles, these metrics offer clear, actionable signals rather than guesswork, making them invaluable.

### 4. New Clock Metrics: `GPU_MIN_CLOCK` and `GPU_MAX_CLOCK`

Clock variation is one of the most common indicators of performance issues, yet many systems only expose instantaneous clocks—not the boundaries within a workload window.
v1.4.2 introduces:

- `GPU_MIN_CLOCK`
- `GPU_MAX_CLOCK`

These metrics allow operators to correlate workload phases or throttling events with clock ranges. For example, a training step that intermittently dips to a low frequency may indicate thermal imbalance, NUMA misplacement, or input starvation.

It’s a small addition, but incredibly useful for performance tracing and tuning.

## How This Release Helps Operators and Developers

- Better Debugging in Bare-Metal Environments:
    `KFD_PROCESS_ID` finally provides native workload-to-GPU correlation without relying on an external scheduler.

- More Predictive Fleet Health Management:
    AFID‑based RAS insights surface early indicators of failing GPUs.

- Improved Performance Tuning and Troubleshooting:
    Violation metrics highlight power, thermal, and utilization constraints with fine-grained clarity.

- Cleaner and More Uniform Dashboards:
    Normalized clock labels and new metrics reduce the overhead of maintaining custom dashboards.

This release advances the state of GPU fleet monitoring. As HPC and AI workloads scale in complexity, the ability to understand what’s happening _inside_ the GPU becomes mission‑critical. DME v1.4.2 brings operators one step closer to that level of clarity.

## Summary

This blog highlights how AMD Device Metrics Exporter (DME) focuses on making GPU fleets easier to understand, debug, and tune in real-world HPC and AI environments. It introduces new runtime context labels like KFD_PROCESS_ID to directly correlate GPU activity with the Linux processes driving it, especially valuable on bare-metal systems without a job scheduler. It adds AFID-aware RAS metrics via GPU_AFID_ERRORS so operators can see exactly which hardware reliability events occurred and build targeted alerts and runbooks around them. Violation metrics are significantly expanded to show when and why GPUs are constrained by power, thermal, voltage, or utilization limits, turning vague “slow GPU” reports into data-driven diagnoses. Finally, new GPU_MIN_CLOCK and GPU_MAX_CLOCK metrics provide clearer visibility into clock behavior over time, helping pinpoint performance dips and throttling events.

Together, these enhancements give platform engineers and developers deeper, more actionable insight into GPU health, reliability, and performance across modern clusters.

## Get Started

To learn more about Device Metrics Exporter, please visit our site:

- [Release Notes](https://instinct.docs.amd.com/projects/device-metrics-exporter/en/latest/releasenotes.html)  
- [Quick Start Guide](https://instinct.docs.amd.com/projects/device-metrics-exporter/en/latest/installation/docker.html)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
