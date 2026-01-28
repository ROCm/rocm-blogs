---
blogpost: true
blog_title: "Introducing ROCprofiler SDK - The Latest Toolkit for Performance Profiling."
date: 25 Mar 2025
author: 'Jayacharan Kolla, Ammar Elwazir, Gina Sitaraman'
thumbnail: 'roc-profiler-intro-thumbnail.png'
tags: Profiling, Performance, AI/ML, System-Tuning, HPC
category: Software tools & optimizations
target_audience: AI/ML and HPC Developers.
key_value_propositions: ROCprofiler SDK is a new profiling toolkit that solves multiple issues that exist with legacy profiler interfaces. All the third-party tools and ML Framework Profilers are built on top of legacy profilers, this blog will motivate more third-party tools developers to adopt ROCprofiler SDK profiling interfaces.
language: English
myst:
    html_meta:
        "author": "Jayacharan Kolla, Ammar Elwazir, Gina Sitaraman"
        "description lang=en": "Discover ROCprofiler SDK – ROCm’s next-generation, unified, scalable, and high-performance profiling toolkit for AI and HPC workloads on AMD GPUs."
        "keywords": "ROCprofiler, Performance Analysis, Profiling, Developer Tools"
        "property=og:locale": "en_US"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blogs"
        "amd_blog_type": "Technical Articles & Blogs"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_developer_type": "ML/AI Developer, Application Developer, HPC Developer"
        "amd_deployment": "Servers"
        "amd_product_type": "Development Tools, Software & Applications"
        "amd_developer_tool": "ROCm Software, Open-Source Tools"
        "amd_applications": "High Performance Computing"
        "amd_industries": "Data Center"
        "amd_blog_releasedate": Thursday Mar 27, 12:00:00 PST 2025
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training, Computer Vision, Data Science, Design, Simulation & Modeling"
        "amd_blog_topic_categories": "Software & Ecosystem"
---
<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

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

# Introducing ROCprofiler SDK - The Latest Toolkit for Performance Profiling

Profiling is the backbone of performance optimization in AI and HPC
workloads, enabling developers to extract maximum efficiency from AMD
Instinct&trade; GPUs. With ROCm's rapid evolution, the need for a unified,
scalable, and extensible profiling framework has never been more
critical. The new **ROCprofiler-SDK** framework represents a significant
step forward in profiling capabilities, offering enhanced features,
streamlined integration, and a better user experience while also solving
past limitations with former profiler interface versions. This guide
aims to help users seamlessly transition from legacy profiling tools
to the ROCprofiler-SDK infrastructure. We will explore new features,
highlight key differences from previous tools, and provide actionable
steps for a smooth migration.

## Why ROCprofiler-SDK

- **Efficient & Reliable Initialization:** Improved tool
    initialization, reducing setup complexities and overhead.

- **Multi-Tool Support** -- Enables multiple tools to simultaneously
    use the same profiling services without interference.

- **Simplified Data Collection Control** -- Manage multiple profiling
    services easily with unified API control.

- **Enhanced Error Handling & Logging** -- Better diagnostics with
    improved error checking and logging mechanisms.

- **PC Sampling (Beta)** -- Advanced sampling capabilities for deeper
    insights into where time is spent in GPU kernels.

- **Backward ABI Compatibility** -- Ensures seamless migration for
    existing tools using former ROCm profiling interfaces.

### Key value proposition

Let's look at the key highlights in the ROCprofiler SDK and why you should migrate:

- **Collect GPU Performance Metrics**
  - Access GPU hardware counters to monitor execution time, memory usage, cache hits/misses, and more.
  - Identify GPU bottlenecks and optimize compute, memory, and bandwidth utilization.

- **Trace API Calls & GPU Workload Execution**
  - **HIP API Tracing** - Capture HIP runtime API calls, their
      execution order, and overhead for application debugging and
      performance tuning.
  - **Kernel Tracing** - Track kernel launches and execution to
      debug and optimize your application.
  - **HSA API Tracing** - Monitor low-level HSA API calls for a
      deeper understanding of how workloads interact with the ROCm
      runtime.
  - **Memory Copy Tracing** - Trace device to host (D2H) or Host to
      Device (H2D) copies.
  - **Marker Tracing:** Tracing user-instrumented `roctx` markers in
      application code.
  - and many more expanded capabilities like tracking Page Migration
      events, tracing Scratch Memory operations etc.

With **ROCprofiler SDK**, developers get a stable, future-proof
profiling interface that already integrates with ROCm performance profiling tools like [ROCm
Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/what-is-rocprof-sys.html) & [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/what-is-rocprof-compute.html).

### Evolving ROCprofiler: A more efficient and flexible profiling SDK

Earlier versions of ROCm profiling interfaces included:

1. **ROCprofiler & ROCTracer (v1)** -- Separate libraries for hardware
    counter collection and device activity tracing.

2. **ROCprofiler v2** -- A bundled version combining the above libraries,
    but still inheriting limitations as described below.

The major pain points in these implementations were:

- **Unnecessary Overhead & Lack of Service Awareness**

  - In previous versions, tools accessed profiling services through `roctracer_init()`, which initialized **all available services by default**, regardless of their need.
  - This led to **unnecessary runtime overhead** since every tool had to prepare for unused services.

- **Thread-Safety Challenges & Indirection Overhead**

  - Because services were always enabled, runtime API calls had to **go through wrappers** that checked service configurations dynamically. This introduced synchronization complexity, requiring multiple threads to coordinate access to profiling data leading to race conditions and increased contention.
  - This **increased latency** and made managing data across multiple threads difficult.

- **Limited Multi-Tool Support**

  - There was no structured way for multiple profiling tools to use the same services at the same time.
  - This restricted developers from running multiple profiling tools in parallel.

- **Poor Testing Coverage & Lack of Validation**

  - Code coverage was extremely low, making profiling results less
    reliable.
  - Many tests only checked the exit code of `rocprof` execution and
    missed result verification tests.

**ROCprofiler SDK** introduces *contexts*, a powerful mechanism that
resolves the inefficiencies of previous profiling approaches.

- **Context-Based Service Configuration**

  - Tools now can **declare** which profiling services they need in
    advance by creating **contexts**.
  - Only the requested services are initialized, reducing
    unnecessary setup overhead.

- **More Efficient & Thread-Safe Profiling**

  - Since services are only initialized when explicitly requested,
    ROCprofiler SDK **removes unnecessary API wrappers**,
    eliminating unnecessary overhead.

- **True Multi-Tool Support**

  - Multiple profiling tools can now **use the same services
    simultaneously** without interference.
  - This makes it easier to run **different profiling tools in
    parallel.**

- **Cross-Architecture Testing & Support**

  - ROCprofiler SDK undergoes rigorous testing with comprehensive
    test suites ensuring compatibility across multiple GPU
    architectures and robust performance in diverse environments.

- **Enhanced Adaptability and API improvements**

  - Frequent API improvements make it easier to identify newly added
    features and adapt tools accordingly.
  - Consistent naming conventions in the API provide clarity and
    improve usability.
  - Extensive documentation ensures that every feature is
    well-documented, making integration smoother for developers.

## What is in ROCprofiler-SDK?

- **ROCprofiler API (referred to as `rocprofiler-sdk` on Github)**

  - The core API is used for tracing and profiling workloads running
    on AMD GPUs.
  - Provides direct access to GPU performance counters, kernel
    execution tracing, and communication profiling.
  - Used by third-party profiling tools and ROCm-based profilers
    like ROCm Systems Profiler, ROCm Compute Profiler, and rocprofv3
    CLI tool.

- **`rocprofv3` Tool**

  - A **command-line interface** (CLI) for executing profiling
    workflows.
  - Allows users to **collect, analyze, and export profiling data**
    for performance tuning.
  - Simplifies profiling tasks for developers who need quick
    insights into GPU execution.

- **`roctx`**

  - A marker library for code annotations, enabling developers to
    insert markers in their code for performance analysis on
    specific code sections.

One of the most noticeable improvements in **ROCprofiler SDK** is the
refined and more intuitive command-line interface. The transition from
**rocprof v1/v2 to rocprof v3** brings a cleaner, more structured
command-line set, along with powerful new features that enhance
usability, flexibility, and profiling efficiency.\
\
The **ROCprofiler-SDK** also consists of a suite of API services
designed to help developers profile and optimize GPU applications on
the **ROCm** platform. These services provide deep insights into
application performance and execution behavior, enable
precise profiling, and allow developers to fine-tune their GPU workloads.

- **Buffered Services** -- Efficiently manage and store profiling data
    to prevent system overload.

- **Callback Tracing** -- Monitor and log runtime API calls for
    in-depth execution analysis.

- **Counter Collection** -- Gather hardware performance metrics such
    as memory usage and compute unit activity.

- **Runtime Intercept Tables** -- Capture function calls and their
    parameters to analyze application interactions.

- **PC Sampling** -- Periodically record program counters to identify
    performance hotspots.

For hands-on examples of these APIs in action, check out
the **ROCprofiler-SDK Samples Repository** on **GitHub**: [Explore the
Samples
Here](https://github.com/ROCm/rocprofiler-sdk/tree/amd-staging/samples).\
Instructions for building and running your samples are given here:
[Instruction to Build & Run ROCprofiler SDK
Samples](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/samples.html#rocprofiler-sdk-samples)

## State of Command-Line Profiling Tools

| Feature                 | rocprof (v1)                                      | rocprofv2                                      | rocprofv3 (ROCprofiler SDK)                                    |
|-------------------------|--------------------------------------------------|------------------------------------------------|--------------------------------------------------------------|
| **Brief Description**   | Legacy tool for tracing and performance counter collection | Added functionalities, AMDGPU support, and more output formats | A more efficient and flexible tool with advanced features, emphasizing stability and robustness |
| **Underlying Libraries** | ROCprofiler, ROCTracer                          | ROCprofilerV2 API                              | rocprofiler-sdk                                             |
| **Output Formats**      | CSV, JSON                                       | CSV, JSON, Ptrace                             | CSV, JSON, Ptrace, OTF2                                     |
| **Visualization Formats** | JSON (Perfetto)                               | Ptrace (Perfetto)                             | Ptrace (Perfetto), OTF2 (Vampir)                            |
| **Documentation**     | [ROCprofilerV1 User Manual](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/rocprofv1.html#rocprofilerv1-user-manual) | [ROCprofilerV2 User Manual](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/reference/rocprofilerv2-api.html) | [rocprofiler-sdk](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/_doxygen/html/md__2home_2docs_2checkouts_2readthedocs_8org_2user__builds_2advanced-micro-devices-rocprofiler-sd40c0902389b09edf3210254f2740a88.html) |
| **Status**             | **Only critical bug fixes**                      | **Not maintained anymore**                    | **Under active development**                               |

For a detailed comparison of command-line changes, refer to:
🔗 [Comparing Command-Line Tool Options: rocprof v1/v2 vs. rocprof
v3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/conceptual/comparing-with-legacy-tools.html#comparing-command-line-tool-options-rocprofiler-rocprof-rocprofv2-and-rocprofiler-sdk-rocprofv3).

## Third-Party Tools Adopting ROCprofiler SDK

The adoption of **ROCprofiler SDK** by industry-leading third-party
profiling tools underscores its robust capabilities and improved
performance with refined design.

Some leading HPC performance analysis tools are actively
participating in hackathons and have begun integrating ROCprofiler SDK
into their development:

- ✅ **[TAU](https://www.cs.uoregon.edu/research/tau/home.php)** – Comprehensive profiling tool suite for multi-threaded applications.
- ✅ **[PAPI](https://icl.utk.edu/papi/)** – Universal Interface for accessing hardware performance counters.
- ✅ **[HPCToolkit](https://hpctoolkit.org)** – Integrated suite of tools for measurement and analysis of program performance.
- ✅ **[Score-P](https://www.vi-hps.org/projects/score-p/)** – Highly scalable suite of tools for profiling and event tracing of HPC applications.
- ✅ **[Linaro MAP and Performance Reports](https://www.linaroforge.com/linaro-map/)** – Comprehensive tool enabling joint CPU-GPU profiling and providing comprehensive performance reports.

These tools provide critical profiling, tracing, and performance
monitoring for developers optimizing AI and HPC workloads at scale.
Their migration to **ROCprofiler SDK** ensures improved profiling
accuracy, enhanced thread safety, and better API extensibility for the
next generation of ROCm-powered applications.

## Join the Future of ROCm Profiling

With industry adoption gaining momentum, now is the time for other
developers and tool vendors to migrate to ROCprofiler SDK. Experience a
better and an efficient way to profile workloads on AMD GPUs.

- Explore the ROCprofiler SDK GitHub Repository:[https://github.com/ROCm/rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk)
- Access Documentation here: [ROCprofiler SDK Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/index.html)
- To conduct a hackathon or assist with your tool migration reach out to us: 📧 <dl.ROCm-Profiler.support@amd.com>

## End of Life Notice: Transition to ROCprofiler SDK

We are phasing out development and support for ROCTracer, ROCprofiler,
rocprof, and rocprofv2 in favor of ROCprofiler-SDK in upcoming ROCm
releases. Starting with the ROCm 6.4 release, only critical defect fixes
will be addressed for older versions of the profiling tools and
libraries. We encourage all users to upgrade to the latest version of
the ROCprofiler-SDK library and the rocprofv3 tool to ensure continued
support and access to new features. ROCprofiler-SDK is still in beta
today and will be production-ready by ROCm 6.5.

> We anticipate end of life for ROCprofiler V1/V2 and ROCTracer within nine months after ROCm 6.5 release, aligning with Q1 2026.

## Summary

In this blog, we explored how ROCprofiler SDK is revolutionizing ROCm
profiling by replacing legacy tools like ROCprofiler v1/v2 and ROCTracer
with a unified, scalable, and efficient framework. We discussed the
challenges of older profiling interfaces and how ROCprofiler SDK solves
those challenges. The blog highlighted key capabilities, including GPU
performance metrics collection, API/kernel tracing, memory analysis, and
communication profiling, and showcased the migration of leading
third-party tools to ROCprofiler SDK. Additionally, we provided a
detailed comparison of command-line changes from rocprof v1/v2 to
rocprof v3, outlined the deprecation timeline for older tools, and
emphasized the need for developers to transition to ROCprofiler SDK.
As ROCprofiler SDK continues to evolve, AMD is committed to expanding
its capabilities to meet the growing demands of AI and HPC workloads.
Here’s a glimpse into some exciting upcoming features that will make profiling even more powerful and insightful:

- **Virtualization Support** – Enables profiling in virtualized environments, making it easier to analyze performance in cloud workloads.
- **Support for Interconnect Metrics** – Adds xGMI and PCIe monitoring, allowing developers to track bandwidth utilization and data transfer efficiency.
- **Post-Processing Capabilities** – Introduces data processing features that let users analyze and manipulate profiling data after collection for deeper insights.
- **More Granular Performance Insights** – Future updates will bring fine-grained profiling capabilities, providing detailed visibility into memory operations and execution behavior.

Upgrade today and future-proof your performance analysis workflows with **ROCprofiler SDK**!
