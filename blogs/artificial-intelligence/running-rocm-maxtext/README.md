---
blogpost: true
blog_title: "Running ROCm/MaxText Unit Tests (Decoupled and GCloud-Dependent)"
date: 06 Jan 2026
author: 'Gulsum Gudukbay Akbulut, Jehandad Khan'
thumbnail: 'rocm-maxtext-unittests.jpeg'
tags: AI/ML, GenAI, JAX
category: Applications & models
target_audience: AI/ML engineers, AI/ML enthusiasts, LLM enthusiasts and developers
key_value_propositions: This blog offers a clear and practical approach for running MaxText unit tests on ROCm GPUs in both decoupled (offline) and cloud-dependent modes, enabling fast and reliable validation, as well as streamlined development.
language: English
myst:
    html_meta:
        "author": "Gulsum Gudukbay Akbulut, Jehandad Khan"
        "description lang=en": "Learn how to run MaxText unit tests on AMD ROCm GPUs in offline and cloud modes for fast validation, clear reports, and reproducible workflows."
        "keywords": "MaxText, LLM, AI/ML, Decoupling GCP, Unit Testing, JAX, Transformer Engine"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Gulsum Gudukbay Akbulut, Jehandad Khan"
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

# ROCm MaxText Testing — Decoupled (Offline) and Cloud-Integrated Modes

In this blog, you will learn how to run MaxText unit tests on AMD ROCm GPUs in two complementary modes: offline (decoupled) and fully cloud-integrated. By the end, you will know when to use each mode, how to interpret the results, and how to fold them into your CI and debugging workflows.

You will:

- set up ROCm, JAX, and Transformer Engine for MaxText testing on Instinct GPUs;
- build and reuse a Transformer Engine (TE) JAX wheel for your GPU architecture;
- configure fast, decoupled unit tests that run without Google Cloud access;
- run full cloud-dependent tests that exercise storage, diagnostics, and remote workflows;
- generate HTML/CSV reports and logs you can archive or compare across runs;
- troubleshoot common ROCm/MaxText pitfalls in both modes.

This post is part of a MaxText on ROCm series and pairs with a companion blog that focuses on the structure of the ROCm/MaxText Repository: [ROCm Fork of MaxText: Structure and Strategy](https://rocm.blogs.amd.com/artificial-intelligence/rocm-fork-of/README.html). For a broader, performance-oriented view of MaxText workloads on AMD Instinct MI355X GPUs with ROCm 7, see the ROCm blog [ROCm 7: Training performance on AMD Instinct MI355X GPUs](https://rocm.blogs.amd.com/artificial-intelligence/ROCm7-MI355X-training-performance/README.html), which evaluates large-scale MaxText training runs.

## Understanding the Two Modes

MaxText testing supports two execution modes:

- Decoupled (offline): set `DECOUPLE_GCLOUD=TRUE`. This mode runs tests locally, uses synthetic or minimal datasets, and skips external integrations — ideal when cloud access is restricted or you want fast, network-free feedback.
- Cloud-dependent: set `DECOUPLE_GCLOUD=FALSE`. This enables cloud-oriented tests and diagnostics and validates cloud workflows.

Pro tip: if you don’t have TPU hardware, set `JAX_PLATFORMS=rocm` to avoid metadata probing delays.

## Getting Ready: Prerequisites

Make sure your environment meets these requirements before starting:

- The ROCm stack (HIP and runtime libraries) is installed.
- Python 3.12 or newer.
- A Transformer Engine (TE) JAX wheel built for your GPU architecture (for example `gfx950` or `gfx942`).
- Export the correct architecture environment variables (for example `PYTORCH_ROCM_ARCH`, `NVTE_ROCM_ARCH`, etc.).

## Building the Transformer Engine (TE) Wheel

Building the TE wheel is a one-time process per update. In summary:

1. Install `cmake` and clone the TransformerEngine repository.
2. Initialize submodules.
3. Export the ROCm and build environment variables, then build the wheel.

Example commands:

```bash
git clone https://github.com/ROCm/TransformerEngine.git
cd TransformerEngine
git submodule update --init --recursive
export USE_ROCM=1
export HIP_PATH=/opt/rocm
export NVTE_FRAMEWORK=jax
export CMAKE_BUILD_PARALLEL_LEVEL=64
export PYTORCH_ROCM_ARCH=gfx950 # UPDATE YOUR ARCH - VERY IMPORTANT
export NVTE_ROCM_ARCH=gfx950
export NVTE_USE_ROCM=1
export NVTE_FUSED_ATTN_AOTRITON=0
export PYTHONPATH=${PWD}/3rdparty/hipify_torch
export NVTE_BUILD_MAX_JOBS=200 # ADJUST THIS ACCORDING TO NUMBER OF CPUs (USE lscpu)

# If you are building for gfx942 variants, also specify the number of Compute Units
# export CU_NUM=304

python3 setup.py bdist_wheel
```

When complete, the wheel will appear under `TransformerEngine/dist/transformer_engine-*.whl`.

## Cloning the ROCm Fork of MaxText

To get the ROCm integration and decoupling logic, you need the ROCm-maintained MaxText fork. The `rocm-main` branch tracks upstream MaxText while adding decoupling logic, decoupled test markers, and configuration defaults that match the instructions in this blog.

If you already have a MaxText clone from another workflow, you can reuse it as long as it is on the `rocm-main` branch and up to date.

```bash
git clone https://github.com/ROCm/maxtext.git -b rocm-main
```

## Decoupled Unit Tests (Offline)

The decoupled mode validates core model logic, data ingestion, and kernel paths without external services. It’s fast, reproducible, and ideal for iterative development.

Environment setup:

```bash
export JAX_PLATFORMS=rocm
export DECOUPLE_GCLOUD=TRUE
```

Recommended: use a dedicated virtual environment (for example `.venv_decoupled`) and install the required packages, the TE wheel, and MaxText itself.

Full commands:

```bash
git clone https://github.com/ROCm/maxtext.git -b rocm-main
cd maxtext
export DECOUPLE_GCLOUD=TRUE
python -m venv .venv_decoupled
source .venv_decoupled/bin/activate
pip install -r dependencies/requirements/requirements_decoupled_rocm_jax_0_7.1.txt
pip install ../TransformerEngine/dist/transformer_engine*.whl
pip install . --no-deps
pip install pytest pytest-html pytest-csv
export MAXTEXT_REPO_ROOT=$PWD
export MAXTEXT_ASSETS_ROOT=$PWD/src/MaxText/assets/
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Run tests:

```bash
pytest -m decoupled -v tests --csv=decoupled-tests-report.csv --html=decoupled-tests-report.html --self-contained-html | tee maxtext_decoupled_UT.log
```

What happens internally

- Only tests safe for offline execution are included; anything requiring external services or TPUs is excluded.
- Minimal or synthetic datasets keep runs fast and reproducible.
- You’ll get CSV and HTML reports plus detailed logs for review.

### How Tests Are Selected

Tests are considered decoupled when `DECOUPLE_GCLOUD=TRUE` and they are not tagged with markers such as `external_serving`, `external_training`, or `tpu_only`. Included tests typically cover shape/dtype checks, attention mechanism tests, synthetic training loops, minimal dataset ingestion, and local checkpoint tests. Excluded tests include Vertex AI entrypoints, remote diagnostics, cloud-based checkpointing, and downloads that require external access.

### Common Issues (Symptoms / Causes / Fixes)

- TPU metadata timeout: forgot to set `JAX_PLATFORMS=rocm` → export it correctly before running.
- Grain ArrayRecord performance warning: `group_size` not equal to 1 → regenerate ArrayRecord shards with `group_size=1`.
- Import errors for stubs: wrong branch checked out → ensure `rocm-main` is used.
- Missing minimal dataset: regenerate using `get_minimal_c4_en_dataset.py` or verify the dataset path under `datasets/c4_en_dataset_minimal`.

## Why Decoupling Matters

Running tests offline provides faster feedback, reduces flakiness, and supports development in restricted environments. Decoupled testing helps you:

- iterate on kernel and model changes without waiting on cloud setups;
- reproduce failures on any ROCm/NVIDIA machine or CI runner with the same GPU generation, without the need for Google Cloud environment setup;
- separate infrastructure issues (IAM, networking, storage) from pure model or kernel issues;
- define a pipeline where changes must pass offline tests before consuming cloud resources.

Decoupling provides reproducibility, portability, and a smoother development experience. Note that it does not validate remote storage or cloud diagnostics — those require cloud-dependent mode. Passing offline tests does not guarantee integration success, so schedule periodic full (cloud-enabled) runs and clearly mark tests that rely on external services.

## Cloud-Dependent Unit Tests

For full integration and diagnostics, use the cloud-dependent mode.

Environment setup:

```bash
export JAX_PLATFORMS=rocm
export DECOUPLE_GCLOUD=FALSE
```

Set up a separate virtual environment (for example `.venv_gce`), install the standard requirements, and run:

```bash
git clone https://github.com/ROCm/maxtext.git -b rocm-main
cd maxtext
export DECOUPLE_GCLOUD=FALSE
python -m venv .venv_gce
source .venv_gce/bin/activate
pip install -r dependencies/requirements/requirements_rocm_jax_0_7.1.txt
pip install ../TransformerEngine/dist/transformer_engine*.whl
pip install . --no-deps
pip install pytest pytest-html pytest-csv
export MAXTEXT_REPO_ROOT=$PWD
export MAXTEXT_ASSETS_ROOT=$PWD/src/MaxText/assets/
export PYTHONPATH=$(pwd):$PYTHONPATH
```

Run all tests (cloud-enabled):

```bash
pytest -v tests --csv=tests-report.csv --html=tests-report.html --self-contained-html | tee maxtext_UT.log
```

This mode exercises cloud-oriented code paths and includes a broader range of tests, but may assume network connectivity and cloud resources.

## Minimal Synthetic Training Example

A fast validation run to exercise kernels and basic training flow:

```bash
python -m MaxText.train MaxText/configs/base.yml \
  run_name=test hardware=gpu steps=5 model_name=llama2-7b \
  attention=cudnn_flash_te enable_checkpointing=False \
  ici_expert_parallelism=1 ici_fsdp_parallelism=-1 ici_data_parallelism=1 \
  remat_policy=minimal scan_layers=True dataset_type=synthetic \
  logits_dot_in_fp32=False dtype=bfloat16 weight_dtype=bfloat16 \
  per_device_batch_size=1 max_target_length=2048 shardy=False
```

Notes for this run:

- `attention=cudnn_flash_te` validates ROCm fast kernels.
- `remat_policy=minimal` and `scan_layers=True` help check gradient memory patterns.
- `dataset_type=synthetic` keeps the run fast.
- `bfloat16` for `dtype` and `weight_dtype` reflects production-like precision.

## Docker-Based Flow

Prefer containers? Use a ROCm-enabled Docker image, mapping devices and increasing shared memory. Bind-mount the MaxText repo for development.

Example (alias + run):

```bash
alias drun='sudo docker run --name jax_maxtext -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -w /root -v $HOME/MaxText:/MaxText'
drun rocm/dev-ubuntu-24.04:7.1-complete
```

## Artifacts and Test Reporting

After running tests, you’ll have CSV and HTML reports for comparison and publishing.

Typical artifacts include:

- `decoupled-tests-report.csv` and `decoupled-tests-report.html` for offline runs;
- `tests-report.csv` and `tests-report.html` for full cloud-enabled runs;
- `maxtext_decoupled_UT.log` and `maxtext_UT.log` capturing detailed pytest output.

You can archive these artifacts in CI, attach them to pull requests, or compare them across ROCm, JAX, or configuration versions to spot regressions. For longer-term tracking, these key metrics (for example, pass rate, test duration, selected performance counters) could be integrated into dashboards.

## Troubleshooting at a Glance

- TE wheel import fails: check architecture flags and rebuild the wheel.
- Slow first test: JAX compilation overhead. Rerun to warm caches for faster throughput.
- Missing datasets: regenerate or verify dataset paths.
- Hangs: double-check virtual environment and package versions.

## Summary

This blog explains how to validate ROCm-based MaxText setups in two complementary ways: an offline, decoupled path and a fully cloud-dependent path. You learn how to obtain the JAX Transformer Engine (TE) wheel for your ROCm GPU architecture, clone the ROCm MaxText fork, and configure separate virtual environments for each testing mode.

In the decoupled mode (DECOUPLE_GCLOUD=TRUE), you run a curated subset of offline-safe unit tests that rely on synthetic or minimal datasets and avoid TPUs and external services. You see how tests are selected, which ones are included/excluded, and how to interpret logs and artifacts, along with a compact troubleshooting guide for common ROCm/MaxText issues.

In the cloud-dependent mode (DECOUPLE_GCLOUD=FALSE), you expand coverage to cloud-integrated diagnostics, storage, and remote workflows. The blog also walks through a minimal synthetic training run to quickly validate ROCm-based kernels and memory behavior, plus a Docker-based workflow for containerized setups. By the end, you understand how and when to use each mode, how to generate HTML/CSV reports for unit testing, and why decoupling is essential for reproducibility while full cloud runs remain critical for integration validation.

Once you have a reliable MaxText test workflow on ROCm, you can connect it with performance characterization from [ROCm 7: Training performance on AMD Instinct MI355X GPUs](https://rocm.blogs.amd.com/artificial-intelligence/ROCm7-MI355X-training-performance/README.html), which uses MaxText workloads to demonstrate end-to-end training throughput on AMD Instinct MI355X GPUs.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
