---
blogpost: true
blog_title: "GEMM Tuning within hipBLASLt- Part 2"
date: 09 Oct 2025
author: 'Chia Hung, YangWen Huang, Carson Liao'
thumbnail: 'image/thumbnail.png'
tags: AI/ML, Developers
category: Software tools & optimizations
target_audience: ai enthusiasts and ai developers
key_value_propositions: hipblaslt tuning
language: English
myst:
    html_meta:
        "author": "Chia Hung, YangWen Huang, Carson Liao"
        "description lang=en": "Learn how to use hipblaslt-bench for offline GEMM tuning in hipBLASLt—benchmark, save, and apply custom-tuned kernels at runtime."
        "keywords": "hipblaslt, tuning"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "Enterprise & Data Center Trends"
        "amd_blog_authors": "Chia Hung, YangWen Huang, Carson Liao"
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

# GEMM Tuning within hipBLASLt– Part 2

This post continues from [Part 1](https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html) where we introduced GEMM tuning concepts in hipBLASLt and explored the basics of solution search. In Part 2, we focus on offline tuning with the hipblaslt-bench tool. This workflow allows developers to benchmark candidate GEMM kernels for specific problem shapes, capture the best-performing solutions, and reuse them at runtime without rebuilding or modifying the hipBLASLt library.

---

## Using hipBLASLt Offline Tuning with `hipblaslt-bench`

The `hipblaslt-bench` tool enables developers to find the best-performing GEMM kernel for specific problem sizes. The output—known as the **solution index**—can be used in future GEMM calls via the `HIPBLASLT_TUNING_OVERRIDE_FILE` mechanism.

> These solution indices are **not guaranteed to remain valid across ROCm versions**. You must re-run tuning whenever you upgrade.

---

### Workflow Overview

1. **Enable logging** to capture GEMM shapes:

   ```bash
   export HIPBLASLT_LOG_MASK=32
   ```

2. Run your GEMM operation or app. The log will emit a `hipblaslt-bench` command like:

   ```bash
   hipblaslt-bench --api_method c -m 1024 -n 512 -k 1024 \
     --lda 1024 --ldb 1024 --ldc 1024 --alpha 1.0 --beta 1.0 \
     --transA N --transB N --batch_count 1 \
     --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r \
     --scale_type f32_r --bias_type f32_r \
     --compute_type f32_r --algo_method index \
     --solution_index <<<INDEX>>>
   ```

3. Set the following environment variable before running the hipblaslt-bench command from your GEMM operation or application. This enables tuning mode and saves the best solution index:

   ```bash
   export HIPBLASLT_TUNING_FILE=tuning.txt
   ```

   This will generate a `tuning.txt` file containing the tuned solution index after the benchmark run.

4. To apply the tuned result at runtime, unset the tuning variable and set the override variable:

   ```bash
   unset HIPBLASLT_TUNING_FILE
   export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt
   ```

   This allows hipBLASLt to override the default solution with the one stored in `tuning.txt`.
   If `--algo_method heuristic` was used during benchmarking, the runtime will override the default heuristic result with the pre-selected solution index found in the file.
   This also affects runtime behavior when using the C API `hipblasLtMatmulAlgoGetHeuristic` or the C++ API `algoGetHeuristic`—these functions will return the tuned solution if a matching entry exists in the override file.

---

### Example Summary

```bash
# Step 1: Enable logging
export HIPBLASLT_LOG_MASK=32

# Step 2: Run the benchmarked GEMM
./my_gemm_app

# Step 3: Set to tuning mode
export HIPBLASLT_TUNING_FILE=tuning.txt

# Step 4: At runtime, override the default logic
unset HIPBLASLT_TUNING_FILE
export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt
```

Once enabled, your GEMM calls will use custom-tuned kernels without changing library binaries.

---

## Advantages & Limitations

| Feature                             | Description                                                                |
|------------------------------------|-----------------------------------------------------------------------------|
| Easy deployment                 | No need to rebuild the library; just load a tuning file at runtime         |
| Re-tuning required after upgrade    | Solution indices may change between ROCm versions                         |

---

## Summary

The `hipblaslt-bench` offline tuning approach is optimal when you want runtime flexibility without modifying or recompiling the library. It supports easy deployment of tuned kernels for stable GEMM workloads and allows you to update your tuning results independently of the library release. This makes it a practical choice when you need quick performance gains with minimal setup effort.

However, for maximum long-term performance and consistency—especially if you're managing multiple library versions or hardware generations—Part 1’s `find_exact.py` workflow may offer more control.

---

**References:**

- [hipBLASLt GitHub Repository](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt)
- [`hipblaslt-bench` offline tuning](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to-use-hipblaslt-offline-tuning.html)
