---
blogpost: true
blog_title: "Running ROCm/MaxText Unit Tests (Decoupled and GCloud-Dependent)"
date: 13 November 2025
author: 'Gulsum Gudukbay Akbulut'
thumbnail: 'maxtext-testing.png'
tags: MaxText, LLM, AI/ML, Unit Testing, JAX, Transformer Engine
target_audience: AI/ML engineers, AI/ML enthusiasts, LLM enthusiasts and developers
key_value_propositions: This blog provides a clear, practical approach for running MaxText unit tests on ROCm GPUs in both decoupled (offline) and cloud-dependent modes, enabling fast, reliable validation and streamlined development.
category: Software tools & optimizations
language: English
myst:
  html_meta:
    "author": "Gulsum Gudukbay Akbulut"
    "description lang=en": "This guide details how to run MaxText unit tests on AMD ROCm GPUs, supporting both decoupled (offline) and cloud-dependent workflows. It explains the rationale and setup for each mode: decoupled mode enables fast, reproducible local testing without cloud dependencies, ideal for environments that do not have Google Cloud setup, while cloud-dependent mode enables full integration and diagnostic coverage. The document walks through prerequisites, environment setup, building the Transformer Engine wheel, and running tests in both modes, including Docker-based options. It clarifies which tests are included or excluded in each scenario, provides troubleshooting advice, and emphasizes the importance of decoupling for reproducibility and development speed. However, it also cautions that offline success doesn’t guarantee cloud integration, recommending periodic full test runs. This comprehensive approach empowers ML practitioners to validate their ROCm-based MaxText setups efficiently, ensuring robust model development and deployment pipelines."
    "keywords": "MaxText, LLM, AI/ML, Unit Testing, JAX, Transformer Engine"
    "property=og:locale": "en_US"
    "amd_category": "Developer Resources"
    "amd_asset_type": "Blogs"
    "amd_blog_type": "Technical Articles & Blogs"
    "amd_technical_blog_type": "Applications and models"
    "amd_developer_type": "ML/AI Developer"
    "amd_deployment": "Servers"
    "amd_product_type": "Development Tools"
    "amd_developer_tool": "ROCm Software, Open-Source Tools"
    "amd_applications": "Large Language Model (LLM)"
    "amd_industries": "Data Center"
    "amd_blog_releasedate": Thu Nov 13, 12:00:00 PST 2025
---
<!---
Copyright (c) 2025 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
--->

# ROCm MaxText Testing — Decoupled (Offline) and Cloud-Integrated Modes

When working with MaxText on ROCm GPUs, you’ll often need to validate your setup in two distinct ways: offline (decoupled) and fully cloud-integrated. This guide walks you through both approaches, explaining the why as well as the how.

## Understanding the two modes

MaxText testing supports two execution modes:

- Decoupled (offline): set `DECOUPLE_GCLOUD=TRUE`. This mode runs tests locally, uses synthetic or minimal datasets, and skips external integrations — ideal when cloud access is restricted or you want fast, network-free feedback.
- Cloud-dependent: set `DECOUPLE_GCLOUD=FALSE`. This enables cloud-oriented tests and diagnostics and validates cloud workflows.

Pro tip: if you don’t have TPU hardware, set `JAX_PLATFORMS=rocm` to avoid metadata probing delays.

## Getting ready: prerequisites

Make sure your environment meets these requirements before starting:

- ROCm stack (HIP and runtime libraries) is installed.
- Python 3.12 or newer.
- A Transformer Engine (TE) JAX wheel built for your GPU architecture (for example `gfx950` or `gfx942`).
- Export the correct architecture environment variables (for example `PYTORCH_ROCM_ARCH`, `NVTE_ROCM_ARCH`, etc.).

## Building the Transformer Engine (TE) wheel

Building the TE wheel is a one-time process per update. In summary:

1. Install `cmake` and clone the TransformerEngine repository.
2. Initialize submodules.
3. Export ROCm and build environment variables and then build the wheel.

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
export NVTE_BUILD_MAX_JOBS=200

# If you are building for gfx942 variants, also specify the number of Compute Units
# export CU_NUM=304

python3 setup.py bdist_wheel
```

When complete, the wheel will appear under `TransformerEngine/dist/transformer_engine-*.whl`.

## Cloning the ROCm fork of MaxText

To get the ROCm integration and decoupling logic, clone the `rocm-main` branch of the MaxText fork:

```bash
git clone https://github.com/ROCm/maxtext.git -b rocm-main
```

## Decoupled unit tests (offline)

The decoupled mode validates core model logic, data ingestion, and kernel paths without external services. It’s fast, reproducible, and ideal for iterative development.

Environment setup:

```bash
export JAX_PLATFORMS=rocm
export DECOUPLE_GCLOUD=TRUE
```

Recommended: use a dedicated virtual environment (for example `.venv_decoupled`) and install required packages, the TE wheel, and MaxText itself.

Full commands:

```bash
git clone https://github.com/ROCm/maxtext.git -b rocm-main
cd maxtext
export JAX_PLATFORMS=rocm # if you do not have TPU
export DECOUPLE_GCLOUD=TRUE
python -m venv .venv_decoupled
source .venv_decoupled/bin/activate
pip install -r requirements_decoupled_rocm_jax_0_7_1.txt
pip install ../TransformerEngine/dist/transformer_engine*.whl
pip install .
pip install pytest pytest-html pytest-csv
export PYTHONPATH=$(pwd)/maxtext:$PYTHONPATH
```

Run tests:

```bash
pytest -m decoupled -v tests --csv=decoupled-tests-report.csv --html=decoupled-tests-report.html --self-contained-html | tee maxtext_decoupled_UT.log
```

What happens internally

- Only tests safe for offline execution are included; anything requiring external services or TPUs is excluded.
- Minimal or synthetic datasets keep runs fast and reproducible.
- You’ll get CSV and HTML reports plus detailed logs for review.

### How tests are selected

Tests are considered decoupled when `DECOUPLE_GCLOUD=TRUE` and they are not tagged with markers such as `external_serving`, `external_training`, or `tpu_only`. Included tests typically cover shape/dtype checks, attention mechanism tests, synthetic training loops, minimal dataset ingestion, and local checkpoint tests. Excluded tests include Vertex AI entrypoints, remote diagnostics, cloud-based checkpointing, and downloads that require external access.

### Common issues (symptoms / causes / fixes)

- TPU metadata timeout: forgot `JAX_PLATFORMS=rocm` → export it correctly before running.
- Grain ArrayRecord performance warning: `group_size` not equal to 1 → regenerate ArrayRecord shards with `group_size=1`.
- Import errors for stubs: wrong branch checked out → ensure `rocm-main` is used.
- Missing minimal dataset: regenerate using `get_minimal_c4_en_dataset.py` or verify the dataset path under `datasets/c4_en_dataset_minimal`.

## Why decoupling matters

Running tests offline provides faster feedback, reduces flakiness, and supports development in restricted environments. Note that it does not validate remote storage, IAM, or multi-region flows — those require cloud-dependent mode.

## Cloud-dependent unit tests

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
pip install -r requirements_rocm_jax_0_7_1.txt
pip install ../TransformerEngine/dist/transformer_engine*.whl
pip install .
pip install pytest pytest-html pytest-csv
export PYTHONPATH=$(pwd)/maxtext:$PYTHONPATH
```

Run all tests (cloud-enabled):

```bash
pytest -v tests --csv=tests-report.csv --html=tests-report.html --self-contained-html | tee maxtext_UT.log
```

This mode exercises cloud-oriented code paths and includes a broader range of tests, but may assume network connectivity and cloud resources.

## Minimal synthetic training example

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

## Docker-based flow

Prefer containers? Use a ROCm-enabled Docker image, mapping devices and increasing shared memory. Bind-mount the MaxText repo for development.

Example (alias + run):

```bash
alias drun='sudo docker run --name jax_maxtext -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -w /root -v $HOME/MaxText:/MaxText'
drun rocm/dev-ubuntu-24.04:7.1-complete
```

## Artifacts and test reporting

After running tests you’ll have CSV and HTML reports for comparison and publishing, plus logs for performance tracing.

## Troubleshooting at a glance

- TE wheel import fails: check architecture flags and rebuild the wheel.
- Slow first test: JAX compilation overhead. Rerun to warm caches for faster throughput.
- Missing datasets: regenerate or verify dataset paths.
- PRNG mismatch: ensure the override is present in `conftest.py`.
- Hangs: double-check virtual environment and package versions.

## The value of decoupling

Decoupling provides reproducibility, portability, and a smoother development experience. Passing offline tests doesn't guarantee integration success — schedule periodic full (cloud-enabled) runs and mark tests clearly when they have external dependencies.
