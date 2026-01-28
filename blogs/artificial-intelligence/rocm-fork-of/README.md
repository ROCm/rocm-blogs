---
blogpost: true
blog_title: "ROCm Fork of MaxText: Structure and Strategy"
date: 06 Jan 2026
author: 'Gulsum Gudukbay Akbulut'
thumbnail: 'rocm-maxtext-structure.jpeg'
tags: AI/ML, GenAI, JAX
category: Applications & models
target_audience: AI/ML engineers, AI/ML enthusiasts, LLM enthusiasts and developers
key_value_propositions: The ROCm fork of MaxText keeps a clean, automated sync with upstream while isolating ROCm-specific work, so ROCm changes remain easy to maintain and upstream. It also enables fast, fully offline “decoupled” workflows with minimal local datasets and smart test selection that benefit ROCm, NVIDIA, and any non-GCE platform.
language: English
myst:
    html_meta:
        "author": "Gulsum Gudukbay Akbulut"
        "description lang=en": "Learn how the ROCm fork of MaxText mirrors upstream while enabling offline testing, minimal datasets, and platform-agnostic, decoupled workflows."
        "keywords": "MaxText, LLM, AI/ML, Decoupling, GCP, Unit Testing, JAX, Transformer Engine"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Training, AI Inference, Generative AI"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Gulsum Gudukbay Akbulut"
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

# ROCm Fork of MaxText: Structure and Strategy

In this blog you will explore how the ROCm fork of MaxText is structured and how that structure supports ROCm and fully offline, decoupled workflows across platforms.

You will learn how the fork tracks upstream AI-Hypercomputer/MaxText, how minimal datasets and decoupled tests fit into your day-to-day development, and what to expect from future ROCm-tagged releases.

Specifically, you will:

- understand the dual-branch model (`main` as an upstream mirror and `rocm-main` for ROCm and decoupling work);
- see how `local_datasets/` enables fast, deterministic local data loading for CI and debugging;
- learn how decoupled pytest markers, gcloud stubs, and test selection patterns make offline mode safe and predictable;
- follow a typical ROCm developer workflow, from building TE wheels to running decoupled tests;
- understand how automated sync mechanics keep the ROCm fork closely aligned with upstream while minimizing conflicts;
- get a preview of planned improvements such as tagged releases, dashboards, and richer offline assets.

This post is part of a MaxText on ROCm series and pairs with the companion testing-focused blog, [Running ROCm/MaxText Unit Tests (Decoupled and GCloud-Dependent)](https://rocm.blogs.amd.com/artificial-intelligence/running-rocm-maxtext/README.html). For a broader, performance-oriented view of MaxText workloads on AMD Instinct MI355X GPUs with ROCm 7, see the ROCm blog [ROCm 7: Training performance on AMD Instinct MI355X GPUs](https://rocm.blogs.amd.com/artificial-intelligence/ROCm7-MI355X-training-performance/README.html), which evaluates large-scale MaxText training runs.

## High-Level Goals

At their core, the ROCm/MaxText repository and the decoupling support are about keeping up with the upstream repository (AI-Hypercomputer/MaxText) as closely as possible. However, on the rocm-main branch, we want to make sure everything runs smoothly on ROCm, across different GPUs like gfx950 and gfx942. An important note is that while we focus on ROCm, the improvements we’re making for offline, decoupled mode aren’t just for ROCm. They’re for any platform that doesn’t want to be tied to Google Cloud. So, if you’re running on NVIDIA or any other setup that doesn’t have GCE dependencies, you’ll benefit from this too.

Another big priority is making sure you can run tests and small training loops completely offline—no Google Cloud dependency required. To make life easier for developers, we’ve also included some mini, checked-in datasets so you can validate things quickly and deterministically.

## How We Handle Branches

We keep things tidy with two main branches. The “main” branch is basically a mirror of the upstream google/maxtext main—no experimental ROCm work happens here, and it only gets updated through automated merges or fast-forwards. All the ROCm-specific development, like kernel tuning and decoupling features, happens in “rocm-main.” This separation helps us keep upstream comparisons clean and reduces the risk of introducing issues before we’re ready to contribute changes back upstream.

## Releases and Tags

Right now, we don’t have formal release tags. But the plan is to start using annotated tags like vX.Y.Z-rocm once we have stable, reproducible container and wheel builds. Changelogs will focus on what’s different for ROCm, TE compatibility, offline testing improvements, and so on. Before we tag a release, we’ll make sure decoupled tests pass and that TE wheels are reproducible. There’s a bit of a dependency dance here: we can’t use TE wheels directly yet because of version mismatches between flax and JAX, but that should get sorted out in upcoming releases.

## Minimal Datasets for Local Testing

Inside the `local_datasets/` directory, you’ll find mini shards and helper scripts designed for rapid local testing. These scripts let you create tiny C4 subsets in different formats (ArrayRecord, TFRecord, Parquet), convert between formats, and generate local TFDS metadata. The idea is to make loading data super fast—sub-second, ideally—so you can get to training or testing right away, with deterministic results and a minimal footprint and without setting up Google Cloud Storage access to read datasets remotely.

**Local Datasets directory overview:**

- `get_minimal_c4_en_dataset.py`: Produce a tiny C4 subset in ArrayRecord and TFRecord.
- `get_minimal_hf_c4_parquet.py`: Convert TFRecord subset to HuggingFace-style Parquet.
- `convert_arrayrecord_to_tfrecord.py`: Reshape existing ArrayRecord to TFRecord.
- `generate_tfds_metadata.py`: Create local TFDS metadata (e.g., versions 3.0.1 and 3.1.0).
- `c4_en_dataset_minimal/`: Tiny dataset layout (ArrayRecord shards, Parquet, TFDS metadata).
- `gcloud_decoupled_test_logs`/: Artifacts from offline test runs.

## Decoupling: Offline Mode (for Everyone!)

One of the coolest features is the ability to run everything offline, without any Google Cloud dependencies. And again, this isn’t just for ROCm users—if you’re on NVIDIA or any other platform, you can take advantage of this decoupled mode too. It’s very suitable for validating core model logic, data ingestion, training loops, and kernel behaviors, all without the need to touch the cloud. That’s why we call it offline mode "for everyone": the same mechanisms (markers, stubs, and minimal datasets) work across platforms as long as you can run the MaxText/JAX stack.

You just set a couple of environment variables (like DECOUPLE_GCLOUD=TRUE and JAX_PLATFORMS=rocm or whatever platform you’re on), and you’re good to go. There are stubs and selective test filtering to make sure only the right tests run in offline mode.

### Functional Components

- **gcloud_stub.py:** Provides is_decoupled(), stub cloud_diagnostics(), and placeholders libraries like jetstream/tunix currently unused in offline mode.
- **pytest marker `decoupled`:** Automatically applied only to tests confirmed safe in offline mode via logic in tests/conftest.py.
- **Minimal datasets:** Local mini C4 shards (ArrayRecord / Parquet / TFDS metadata) used when dataset_type=tfds, grain, or HF-style ingestion is invoked.

**The test suite is smart about what it runs:** it skips anything that needs TPUs, external serving or integration tests, diagnostics (that rely on GCS paths), metrics uploads or cloud-managed checkpoint orchestrators. That means you can run quick, offline-safe tests with commands like pytest -m decoupled, or go for the full suite if you want to include cloud-dependent stuff. If a test accidentally depends on the cloud, it just won’t get marked as decoupled, so it is filtered.

### Test Selection Patterns

- Offline-only quick loop: `pytest -m decoupled -q`
- Full test suite: `pytest -v tests`
- Excluding slow integration: `pytest -m "decoupled and not slow"`
- Targeting a specific module offline: `pytest -m decoupled model_test.py`

Tests that are included in decoupled (offline) mode are those that don’t rely on any cloud infrastructure. For example, this covers things like model shape and dtype tests—such as checking attention or activation metrics—along with synthetic training loops that only run for a few steps, data pipeline transformations that use local minimal TFDS or ArrayRecord shards, and checkpoint tests that stick to the local filesystem (as long as they don’t try to sync with the cloud).

On the flip side, anything that depends on cloud services is excluded from decoupled mode. This means integrations like Vertex tensorboard manager, tests that require external model serving handshakes, checkpoint replication logic that relies on Google Cloud Storage (especially for multi-host or multi-region setups), and uploading metrics to remote destinations are all left out. The idea is to keep offline testing fast, reliable, and totally independent of cloud dependencies.

### Why Bother With All This?

Running tests offline is just faster—no waiting for network calls or dealing with remote authentication. It also makes test timing more predictable, which is great for profiling performance. Plus, you get fewer flaky tests since you’re not at the mercy of cloud hiccups. It’s just a smoother experience, especially when you’re focused on improving kernels or debugging on any platform.

### Failure & Edge Cases in Decoupled Mode

If a test needs a cloud path and you’re in offline mode, it simply won’t be marked as decoupled. If the TE wheel isn’t installed or you’re on the wrong architecture, you’ll get an early error or fallback. And if you’re missing a minimal dataset, some tests will skip themselves or throw a clear “dataset not found” error—just re-run the generation scripts to fix it.

### Quick Commands

**Decoupled run:**
`pytest -m decoupled -v tests --csv=tests-report.csv --html=decoupled.html --self-contained-html`

**Full run (including cloud-dependent):**
`pytest -v tests --csv=tests-report.csv --html=full.html --self-contained-html`

**Just offline model tests:**
`pytest -m decoupled model_test.py -q`

## Mechanics

This section explains how the ROCm fork stays aligned with upstream MaxText in practice, and what a typical day looks like for a developer working in this repository. You will see how automated sync jobs keep branches up-to-date, how you can iterate efficiently on ROCm changes, and where to look for quick answers to common questions.

### Syncing

We use an automated workflow (upstream_sync.yml) to merge upstream main into our local main. Then,
rocm-main gets reconciled to keep everything up to date and avoid large conflicts down the road.

### Typical Developer Flow

1. Checkout rocm-main.
2. Build a TE wheel for ROCm target GPU.
3. Run decoupled unit tests using minimal or synthetic datasets.
4. Develop (e.g. optimize kernels/training loop).

### FAQ

- **Why not commit ROCm work directly to main?**
Separation keeps upstream diff clean.
- **Are minimal datasets for production?**
No, only for CI and quick validation.
- **Will releases bundle dataset snapshots?**
Probably not; regeneration scripts are preferred. However, minimal datasets always live in the repository.

## Summary

In this blog, you explored how the ROCm fork of MaxText is organized so that you can stay closely aligned with upstream AI-Hypercomputer/MaxText while still taking advantage of ROCm-specific and fully offline workflows. You saw how the two-branch structure (`main` as an upstream mirror and `rocm-main` for ROCm and decoupling work) keeps changes manageable and upstream-friendly, and how automated sync mechanics reduce long‑term merge pain.

You also learned how minimal datasets in `local_datasets/` and the decoupled (offline) mode let you validate model logic, data ingestion, training loops, and checkpoints without any Google Cloud dependency, on ROCm or other platforms. The combination of gcloud stubs, the `decoupled` pytest marker, and targeted test-selection patterns keeps these runs fast and deterministic, making them ideal for CI and iterative kernel or model work.

Looking ahead, the roadmap includes minimal testing checkpoints for local runs, tokenizers for local execution, ROCm-tagged releases once containers and TE wheels are reproducible, performance dashboards, and upstreaming as many decoupling features as possible. The default experience will remain cloud-enabled unless you flip the flag, but you can increasingly rely on decoupled mode when you want speed and reproducibility.

To put this structure into practice and actually run the tests, read the companion blog [Running ROCm/MaxText Unit Tests (Decoupled and GCloud-Dependent)](https://rocm.blogs.amd.com/artificial-intelligence/running-rocm-maxtext/README.html), which walks you through concrete decoupled and cloud-integrated workflows on AMD Instinct GPUs.

For a broader, performance-oriented view of MaxText workloads on AMD Instinct MI355X GPUs with ROCm 7, see the ROCm blog [ROCm 7: Training performance on AMD Instinct MI355X GPUs](https://rocm.blogs.amd.com/artificial-intelligence/ROCm7-MI355X-training-performance/README.html), which evaluates large-scale MaxText training runs.

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
