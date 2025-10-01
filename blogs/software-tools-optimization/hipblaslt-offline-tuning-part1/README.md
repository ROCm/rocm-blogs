---
blogpost: true
blog_title: "GEMM Tuning within hipBLASLt- Part 1"
date: 05 Sep 2025
author: 'YangWen Huang, Carson Liao'
thumbnail: 'hipblaslt_tuning_header.png'
tags: AI/ML, Developers
category: Software tools & optimizations
target_audience: ai enthusiasts and ai developers
key_value_propositions: hipblaslt tuning
language: English
myst:
    html_meta:
        "author": "YangWen Huang, Carson Liao"
        "description lang=en": "We introduce a hipBLASLt tuning tool that lets developers optimize GEMM problem sizes and integrate them into the library."
        "keywords": "hipblaslt, tuning"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
        "amd_blog_authors": "YangWen Huang, Carson Liao"
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

# GEMM Tuning within hipBLASLt - Part 1

When optimizing matrix operations on AMD GPUs using the ROCm platform, tuning specific problem sizes is essential for achieving maximum performance. The [`hipBLASLt`](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt) library supports two official tuning mechanisms:

- **[Offline Tuning Utility](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to-use-hipblaslt-tuning-utility.html)**: This method uses tools like [`find_exact.py`](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/utilities/find_exact.py) to generate optimized kernel solutions for specific GEMM problem sizes. These solutions are then merged back into the library source and require recompilation.
  - *Pro*: Developers retain control of tuned solutions and can rebuild them with every ROCm upgrade.
  - *Con*: Requires recompilation after each library update.

- **[`hipblaslt-bench` offline tuning](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to-use-hipblaslt-offline-tuning.html)**: This approach generates a runtime configuration file that overrides the default kernel selection without needing to recompile the library.
  - *Pro*: Easy deployment—no rebuild needed.
  - *Con*: Solution indices may change with each ROCm version, requiring re-tuning after upgrades.

In this post, we explore the **offline Tuning Utility workflow using `find_exact.py`**, which allows developers to generate and retain their own optimal GEMM solutions across builds. In the next post, we'll introduce the **`hipblaslt-bench` offline tuning** which allows developers to tune their own optimal GEMM solutions without recompiling the entire library. The first method offers more fine-grained controls while the latter one offers an easy-to-use workflow.

## What is `find_exact.py`?

`find_exact.py` is a utility that leverages the hipBLASLt tuning engine to:

- Search for the optimal solution (kernel) for specific GEMM problems.
- Measure and report performance (execution time).
- Output the solutions in logical YAML format to be merged back into the library source, requiring recompilation for deployment.

This is particularly valuable when your workload involves fixed matrix shapes—common in deep learning, graphics, or HPC workloads—where runtime tuning isn't feasible or desirable.

---

## Defining Problems in `template.yaml`

The input to `find_exact.py` is a YAML configuration file that defines the tuning workflow and problem sizes. The latest format organizes tuning into two main steps:

- `Bench`: benchmark and tune specified problem sizes.
- `CreateLogic` / `UpdateGrid`: store the results in the logic YAML format suitable for integration with the hipBLASLt library.

Here is an example from the updated [`template.yaml`](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/utilities/template.yaml):

```yaml
# Can comment out Bench, CreateLogic, or UpdateGrid if you want to disable.
Bench:
  ProblemType:
    ComputeDataType: s
    ComputeInputDataType: s
    DataTypeA: s
    DataTypeB: s
    DataTypeC: s
    DataTypeD: s
    TransposeA: 0
    TransposeB: 0
    UseBias: False
  TestConfig:
    ColdIter: 20
    Iter: 100
    AlgoMethod: "all"
    RotatingBuffer: 512
  TuningParameters:
    # SplitK list control parameter example
    SplitK: [0, 4, 8]
  ProblemSizes:
  - [128, 128, 1, 128]
  - [128, 128, 1, 128]

# Create equality
CreateLogic: {}

# Update existing grid
UpdateGrid: {}
```

### Key Sections Explained

- **`Bench`**: Defines the set of problem sizes to benchmark and tune. It includes:
  - `ProblemType`: Describes the data types, transpose flags, and bias usage for the GEMM operation.
  - `TestConfig`: Controls benchmarking behavior, such as iteration count and algorithm selection.
  - `TuningParameters`: Optional fine-tuning controls (e.g., `SplitK` values).
  - `ProblemSizes`: A list of problem size tuples in the form `[M, N, Batch, K]`.

- **`CreateLogic`**: When enabled, this step generates *equality logic* entries, where the exact problem size maps to a specific tuned solution. These are written into logic YAMLs, which can be merged into the hipBLASLt source and require recompilation.

- **`UpdateGrid`**: If enabled, the tool compares tuned results to existing **grid-point** solutions and replaces them only if a better-performing solution is found.

---

### Grid and Equality in hipBLASLt

When the input size does not match equality, a grid heuristic algorithm will select the optimal kernel for it.

This hybrid design balances coverage and precision, enabling developers to focus tuning on their most critical problem sizes while still benefiting from generalized performance elsewhere.

---

### The Tuning Parameters

Currently, `SplitK` is the only supported parameter under `TuningParameters`, but more tuning controls are expected in future releases.

#### What is `SplitK`?

`SplitK` enables **splitting the `K` dimension** of the GEMM problem across multiple Compute Units (CUs) in the GPU, allowing for greater parallelism.

#### Why is `SplitK` useful?

Consider the following case:

- Target GPU: MI300 with **304 Compute Units**
- GEMM shape: `M = 1280`, `N = 1280`, `K = 9000`
- Kernel MacroTile: `128 x 128`

Without `SplitK`, the number of active tiles is:

```text
(1280 / 128) * (1280 / 128) = 10 * 10 = 100 CUs utilized
```

That's **only 100 out of 304 CUs**, leaving much of the GPU underutilized.

With `SplitK = 3`, the `K` dimension is divided into three slices:

- 0–2999
- 3000–5999
- 6000–8999

Each slice allows the same 100-tile workload to run in **parallel across different `K` partitions**, resulting in approximately **300 CUs active** simultaneously.

This dramatically increases performance, especially for problems where `K` is large and `M`/`N` are small to moderate.

This is why `SplitK` is an essential tuning knob provided to developers.

---

## Running the Script

Once you have a ROCm-compatible system and hipBLASLt installed, run the script as follows:

```bash
python3 find_exact.py template.yaml build_folder output_folder
```

### Argument Descriptions

- template.yaml: Path to the YAML file describing the problem sizes and tuning configuration.
- build_folder: Path to the compiled hipBLASLt build directory.
- output_folder: Path to the directory where the tuned solution YAMLs and timing results will be written.

The output YAML will be either an equality logic yaml or a grid logic yaml based on the configuration.

---

## Example Use Cases

Here’s when you should consider using `find_exact.py`:

- **Offline tuning**: Pre-benchmarking exact workloads to hardcode optimal solutions.
- **Performance regression testing**: Compare kernel choices and timings across ROCm versions or hardware.
- **Static kernel selection**: For fixed-shape inference in AI models or scientific computing loops.

---

## Summary

`find_exact.py` is an officially supported and actively maintained tool in the hipBLASLt tuning ecosystem. It plays a key role in the offline tuning workflow, allowing developers to precisely tune performance-critical GEMM problem sizes and integrate those results directly into the library.

For use cases involving fixed problem sizes—common in deep learning inference, embedded workloads, and HPC kernels—offline tuning provides stable, repeatable performance optimization that persists across builds. While this approach requires recompilation after each ROCm upgrade, it gives developers full control over kernel selection and performance consistency.

Whether you're preparing for deployment, optimizing for specific hardware, or comparing kernel performance across releases, `find_exact.py` is a valuable tool to include in your tuning pipeline. In our next post, we'll introduce another method to tune GEMM problems without recompiling the library.

---

## Additional Resources

- [hipBLASLt GitHub Repository](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt)
- [find_exact.py script](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/utilities/find_exact.py)
- [template.yaml example](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/utilities/template.yaml)
