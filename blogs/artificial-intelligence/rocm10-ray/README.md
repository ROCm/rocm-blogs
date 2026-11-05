---
blogpost: true
blog_title: "ROCm 10.0.0 Support for Ray 2.58.0: Enabling Scalable AI Workflows"
date: "27 Oct 2026"
author: "Vicky Tsang, Yao Liu, Phani Vaddadi, Vish Vadlamani"
thumbnail: 'thumbnail_rocm10.0_ray.png'
tags: "AI/ML, Serving, LLM"
category: "Applications & models"
target_audience: "Distributed training/inference developers"
key_value_propositions: "ROCm 10.0.0 with Ray 2.58.0 continues support against PyTorch 2.12 and vLLM on ROCm"
language: English
myst:
    html_meta:
        "author": "Vicky Tsang"
        "description lang=en": "ROCm 10.0.0 with Ray 2.58.0 continues PyTorch 2.12 and vLLM support."
        "keywords": "Ray, AI, ML, Distributed, Training, Inference, LLM"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, AI Training, Deploying AI at Scale"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Vicky Tsang, Yao Liu, Phani Vaddadi, Vish Vadlamani"
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

# ROCm 10.0.0 Support for Ray 2.58.0: Enabling Scalable AI Workflows

Distributed AI teams usually run out of GPUs before they run out of ideas. Training and inference capacity is the scarce, expensive resource, and much of the tooling wrapped around it assumes one vendor stack, so putting a different accelerator to work means rewriting whatever touches the hardware.

[Ray](https://github.com/ray-project/ray/) is the layer worth building on precisely because it already hides that detail: placement, scaling, and fault tolerance live behind framework APIs rather than in device code. That has been true on AMD Instinct™ GPUs for several years. ROCm™ 10.0.0 with Ray 2.58.0 continues the same contract: existing Ray code runs on Instinct unchanged, from data loading through distributed training to large language model (LLM) serving, with scaling decisions staying in Ray APIs and placement groups rather than in vendor-specific glue. This release adds a validated stack around that contract: PyTorch 2.12, vLLM on ROCm, and current Instinct GPU models (SKUs), including MI355X.

The payoff is concrete, and it is the second half of this post: the [distributed serving example](#example-distributed-llm-serving-with-ray-serve-and-vllm) brings up an OpenAI-compatible LLM endpoint on Instinct GPUs, with vLLM sharding the model across devices by tensor parallelism, in about one screen of ordinary Ray Serve code. Exactly one line in it names AMD hardware.

## What is Ray?

Ray is an open-source framework that simplifies the development of distributed applications. It is widely used for scaling Python workloads, including machine learning (ML) training, hyperparameter tuning, and reinforcement learning, across clusters of CPUs and GPUs.

## What ROCm 10.0.0 + Ray 2.58.0 Means for Ray Users

ROCm is AMD's open software platform for GPU-accelerated computing. ROCm 10.0.0 and Ray 2.58.0 continue to support the same programming model on Instinct, requiring no vendor-specific glue. What this release adds is a newly validated combination of APIs with the current ROCm, PyTorch, vLLM, and Instinct SKUs.

### Existing Ray Code Runs Unchanged

Ray discovers Instinct GPUs automatically, and `torch.cuda` maps to ROCm and the Heterogeneous-compute Interface for Portability (HIP), so `@ray.remote(num_gpus=1)`, `ScalingConfig`, placement groups, and `LLMConfig` behave the way they do on any other accelerator. There is no ROCm code path to maintain alongside an existing one, and no porting step for the Python that defines a pipeline. Migrating a workload is a scheduling and configuration exercise, not a rewrite.

### Instinct GPUs are First-Class in Ray's Scheduler

Ray does not treat Instinct hardware as a generic GPU. MI300X, MI325X, and MI355X are native accelerator types, so a deployment can request a specific SKU the same way it would request any native accelerator types. That makes heterogeneous fleets tractable: capacity from several Instinct generations can sit in one cluster while individual workloads still land on the SKU they were tuned for. See [Pinning workloads to an Instinct SKU](#pinning-workloads-to-an-instinct-sku) for when to use that constraint and when to leave it off.

### Validated End to End, Through Serving

Compatibility claims are only useful if they extend to the whole path a real workload takes. The evidence in this post is the [distributed serving example](#example-distributed-llm-serving-with-ray-serve-and-vllm): Ray 2.58.0 on ROCm 10.0.0, with vLLM on ROCm, serving a tensor-parallel LLM behind an OpenAI-compatible endpoint. The ROCm Communication Collectives Library (RCCL) carries the tensor-parallel traffic between vLLM workers. Kernel and driver optimizations in this release reduce per-operation overhead and speed up host-to-device transfers, which is where continuously streaming data pipelines spend their time.

```{seealso}
For reinforcement learning from human feedback (RLHF) training that uses Ray on ROCm 10.0.0 with [verl](https://verl.readthedocs.io/en/latest/index.html), see [verl on ROCm 10.0.0](https://rocm.blogs.amd.com/artificial-intelligence/verl_0.9.0_rocm7.15/README.html).
```

## Use Cases: Scalable AI with Ray and ROCm

Ray can be applied to many use cases for scaling ML applications, such as:

* LLMs and Generative AI​
* Batch inference​
* Model serving — see the [distributed LLM serving example](#example-distributed-llm-serving-with-ray-serve-and-vllm)
* Hyperparameter tuning​
* Distributed training​
* Reinforcement learning — see the [Verl 0.9.0 on ROCm 10.0: Next-Generation RL Post-Training on AMD Instinct GPUs](https://rocm.blogs.amd.com/artificial-intelligence/verl_0.9.0_rocm7.15/README.html) post for RLHF with Ray on ROCm 10.0.0
* ML platform​
* End-to-end ML workflows​
* Large-scale workload orchestration​

Refer to the [upstream Ray use cases documentation](https://docs.ray.io/en/latest/ray-overview/use-cases.html) for
detailed tutorials using AMD GPUs and ROCm.

## Installing Ray with ROCm Support

Install Ray 2.58.0 on ROCm 10.0.0 first, then confirm that Instinct GPUs are visible to Ray.

Ray can be installed on a single node or across a cluster. The steps below are the same either way: meet the host requirements, follow the ROCm Ray install guide, then run the cluster confirmation snippet.

### Prerequisites

Confirm the host meets these requirements before you install Ray:

* A node that meets the [ROCm system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for AMD Instinct GPUs (this blog targets Instinct MI300X, MI325X, MI350X, and MI355X)
* A supported [Linux operating system](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems)
* A [ROCm quick start installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)

### ROCm Ray

Install Ray 2.58.0 from the official guide, which covers building a Docker image, installing on bare metal or a custom container, and building from source.

* [Install Ray on ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/ray-install.html)
* [Ray 2.58.0 release branch](https://github.com/AMD-Ecosystem/ray/tree/release/2.58.0)

### Setup

After installation, Ray discovers AMD Instinct GPUs automatically, and `torch.cuda` maps to ROCm and HIP, so the standard Ray APIs work unchanged. Confirm that the cluster is ready for general Ray usage: resources are visible, and a GPU task runs on an Instinct device.

```python
import ray
import torch

ray.init()
print("Ray cluster resources:", ray.cluster_resources())


@ray.remote
def ping():
    return "ok"


@ray.remote(num_gpus=1)
def device_name():
    return torch.cuda.get_device_name(0)


print("Ray task:", ray.get(ping.remote()))
print("Ray GPU worker device:", ray.get(device_name.remote()))
```

A healthy cluster reports CPU and GPU counts, runs the CPU task, and names the Instinct device the GPU worker was placed on:

```text
Ray cluster resources: {'CPU': 192.0, 'GPU': 8.0, 'accelerator_type:AMD-Instinct-MI355X-OAM': 1.0, ...}
Ray task: ok
Ray GPU worker device: AMD Instinct MI355X
```

If Ray reports `0` GPUs, stop here and fix device access (`/dev/kfd`, `/dev/dri`, and the `video` and `render` groups) before running GPU workloads.

## Pinning Workloads to an Instinct SKU

Notice the `accelerator_type:AMD-Instinct-MI355X-OAM` entry alongside the GPU count in that output. That key is how Instinct hardware enters Ray's scheduler, and workloads can require it. Ray 2.58.0 recognizes the current Instinct parts natively:

| Instinct GPU | Ray accelerator type |
| --- | --- |
| MI300X | `AMD-Instinct-MI300X-OAM` |
| MI325X | `AMD-Instinct-MI325X-OAM` |
| MI355X | `AMD-Instinct-MI355X-OAM` |

Ray Core exposes this as `@ray.remote(accelerator_type=...)`; Ray Serve LLM exposes it as the `accelerator_type` field on `LLMConfig`. Either way it is a **placement constraint, not a ROCm enable flag**. ROCm works whether or not the constraint is set.

**When to pin:** Pin the SKU when a cluster contains several Instinct generations, when the cluster also has NVIDIA GPUs, or when a workload is tuned for the memory capacity of a specific Instinct SKU. Pinning also lets Ray infer the SKU-specific `accelerator_config` (memory class and placement-group hints) that it derives from the accelerator type.

**When not to pin:** On a homogeneous single-SKU node, leaving `accelerator_type=None` is correct and simpler. Replicas then request generic `GPU` capacity plus the tensor-parallel degree, which is all the scheduler needs when every device is identical. Pinning a SKU in that case only creates a value to update when the workload moves to different hardware.

The failure modes are worth knowing before choosing. A wrong or unrecognized string **fails closed**: Ray cannot place the deployment, and says so. `None` **fails open**: the workload runs somewhere, which on a mixed cluster may be the wrong somewhere. Neither is universally safer, so pick based on how uniform the cluster is.

## Example: Distributed LLM Serving with Ray Serve and vLLM

This is the payoff: a production-shaped, OpenAI-compatible LLM endpoint running on Instinct GPUs, in roughly one screen of code. Ray Serve LLM (`ray.serve.llm`) handles the HTTP surface and replica placement, vLLM is the engine, and the model is sharded across devices by tensor parallelism with RCCL (exposed through the NCCL backend) carrying the collectives.

Everything below is ordinary Ray Serve LLM code. The same script would run on any accelerator Ray supports. The one line specific to AMD hardware is the `accelerator_type` string, and even that is optional on a homogeneous node.

The example exercises the whole path end to end: Ray discovers Instinct GPUs, Serve places a replica, vLLM shards the model across GPUs, and a real chat-completions request returns a non-empty response, so a successful run means the serving stack works, not just that the import succeeded.

Beyond the install above, this serving example also needs:

* At least two AMD Instinct GPUs visible to Ray (default tensor-parallel size is 2)
* Ray with Serve extras (`ray[default,serve]`), vLLM built for ROCm, and the `openai` Python client
* A large `/dev/shm` (around 10 GB or more). The default 64 MB shared-memory size is too small for RCCL among tensor-parallel workers
* Network access to download the Hugging Face model on first run

In the same verified environment, create `ray_serve_vllm.py` with the following contents:

```python
#!/usr/bin/env python3
#
# Distributed vLLM serving with Ray Serve on ROCm.
#
# Brings up a Ray Serve OpenAI-compatible app (ray.serve.llm) that runs a small
# model with tensor parallelism across multiple AMD GPUs, then sends a real
# request through the HTTP endpoint and checks that you get a non-empty completion.
# This exercises the whole distributed serving path: Ray Serve placement across
# GPUs, the vLLM (ROCm) engine, and NCCL and RCCL for tensor-parallel comms.
#
# Prints "RAY SERVE VLLM OK" on success. Tunable via environment variables:
#   SERVE_MODEL_ID         Hugging Face model id to serve (default Qwen/Qwen2.5-0.5B-Instruct)
#   SERVE_TP_SIZE          Number of GPUs to shard across (default 2)
#   SERVE_READY_TIMEOUT_S how long to wait for the endpoint to come up (default 600)
#   SERVE_PORT            Serve HTTP port (default 8000)
#
import os
import sys
import time

import ray
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

MODEL_ID = os.environ.get("SERVE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
TP_SIZE = int(os.environ.get("SERVE_TP_SIZE", "2"))
READY_TIMEOUT_S = float(os.environ.get("SERVE_READY_TIMEOUT_S", "600"))
PORT = int(os.environ.get("SERVE_PORT", "8000"))


def main() -> int:
    ray.init(logging_level="warning")

    # Fail fast if the node can't satisfy the tensor-parallel degree -- otherwise
    # the Serve deployment would sit UNHEALTHY waiting for GPUs that don't exist.
    gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f">> visible GPUs: {gpus}, tensor_parallel_size: {TP_SIZE}")
    if gpus < TP_SIZE:
        print(
            f"!! need >= {TP_SIZE} GPUs for distributed serving (TP={TP_SIZE}), found {gpus}",
            file=sys.stderr,
        )
        return 1

    llm_config = LLMConfig(
        model_loading_config=dict(model_id=MODEL_ID, model_source=MODEL_ID),
        # Pin the Instinct SKU for SKU-aware placement. Use None on a homogeneous node.
        accelerator_type="AMD-Instinct-MI355X-OAM",
        deployment_config=dict(autoscaling_config=dict(min_replicas=1, max_replicas=1)),
        engine_kwargs=dict(
            tensor_parallel_size=TP_SIZE,
            max_model_len=2048,
            enforce_eager=True,
        ),
    )

    print(f">> Deploying {MODEL_ID} with tensor_parallel_size={TP_SIZE}")
    app = build_openai_app({"llm_configs": [llm_config]})
    serve.start(http_options={"host": "0.0.0.0", "port": PORT})
    serve.run(app, blocking=False)

    from openai import OpenAI

    client = OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="dummy")

    # The endpoint isn't ready until the model is downloaded, sharded across GPUs,
    # and the replica is healthy -- poll the real request until it succeeds.
    deadline = time.time() + READY_TIMEOUT_S
    completion = None
    last_err = None
    while time.time() < deadline:
        try:
            completion = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": "Reply with the single word: pong"}],
                max_tokens=16,
                temperature=0.0,
            )
            break
        except Exception as e:  # not ready yet / transient startup error
            last_err = e
            time.sleep(5)

    if completion is None:
        print(
            f"!! serve endpoint not ready within {READY_TIMEOUT_S:.0f}s: {last_err}",
            file=sys.stderr,
        )
        return 1

    text = completion.choices[0].message.content
    print(f">> distributed vLLM serve (TP={TP_SIZE}) response: {text!r}")
    if not (text and text.strip()):
        print("!! served model returned an empty completion", file=sys.stderr)
        return 1

    print("RAY SERVE VLLM OK")
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        try:
            serve.shutdown()
        except Exception:
            pass
        try:
            ray.shutdown()
        except Exception:
            pass
    raise SystemExit(rc)
```

Run it on the Instinct node:

```bash
python ray_serve_vllm.py
```

The default configuration serves `Qwen/Qwen2.5-0.5B-Instruct` with `tensor_parallel_size=2`. Override the model, GPU shard count, HTTP port, or startup timeout with environment variables, where `hf_model_id` is the Hugging Face model ID, `tensor_parallel_size` is the number of GPUs to shard across, `ready_timeout_s` is the endpoint wait time in seconds, and `serve_port` is the Serve HTTP port:

```bash
SERVE_MODEL_ID=hf_model_id SERVE_TP_SIZE=tensor_parallel_size SERVE_READY_TIMEOUT_S=ready_timeout_s SERVE_PORT=serve_port python ray_serve_vllm.py

# Two-GPU example with the 0.5B instruct model
SERVE_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct SERVE_TP_SIZE=2 python ray_serve_vllm.py

# Example: raise tensor_parallel_size to shard a larger model across more GPUs
# SERVE_MODEL_ID=Qwen/Qwen2.5-14B-Instruct \
# SERVE_TP_SIZE=8 \
# SERVE_READY_TIMEOUT_S=1800 \
# python ray_serve_vllm.py
```

The attention and key-value (KV) head counts of the model must be divisible by `SERVE_TP_SIZE`. Increase `SERVE_READY_TIMEOUT_S` on the first run so the replica can finish downloading weights and become ready.

Three choices in the script are worth calling out:

* `accelerator_type="AMD-Instinct-MI355X-OAM"` pins the replica to MI355X. Swap in `AMD-Instinct-MI300X-OAM` or `AMD-Instinct-MI325X-OAM` for those parts, or drop it to `None` on a homogeneous node. See [Pinning workloads to an Instinct SKU](#pinning-workloads-to-an-instinct-sku).
* A fail-fast check against `ray.cluster_resources()["GPU"]` so Serve does not sit `UNHEALTHY` waiting for GPUs that are not on the node.
* `enforce_eager=True` and a modest `max_model_len` keep the example quick to start; both are ordinary vLLM engine arguments to tune for a real deployment.

Expected output includes the GPU count Ray sees, the completion text, and the success marker:

```text
>> visible GPUs: 8, tensor_parallel_size: 2
>> Deploying Qwen/Qwen2.5-0.5B-Instruct with tensor_parallel_size=2
>> distributed vLLM serve (TP=2) response: 'pong'
RAY SERVE VLLM OK
```

If Ray reports fewer GPUs than `SERVE_TP_SIZE`, confirm ROCm device access (`/dev/kfd` and `/dev/dri`) and that the process is in the `video` and `render` groups. If the HTTP client times out, wait for the first Hugging Face download to finish or raise `SERVE_READY_TIMEOUT_S`.

## Summary

In this blog you explored what ROCm 10.0.0 and Ray 2.58.0 deliver for Ray workloads on Instinct. You saw existing Ray code run unchanged: Ray discovers Instinct GPUs automatically, `torch.cuda` maps to ROCm and HIP, and placement stays in `@ray.remote`, `ScalingConfig`, placement groups, and `LLMConfig`. You walked through a tensor-parallel vLLM deployment behind an OpenAI-compatible endpoint, written as ordinary Ray Serve LLM code, on a stack validated with PyTorch 2.12 and vLLM on ROCm. One line in that example names the hardware, and on a uniform node you can leave it unset.

AMD remains committed to ongoing zero-day support for ROCm and Ray releases, so Instinct stays aligned with each new version as it ships. Keep checking the [Install Ray on ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/ray-install.html) guide for the latest supported combination. The guide is updated as new ROCm and Ray versions are validated, and it is the reference to confirm before you install or upgrade.

Our team plans to keep publishing walkthroughs as that support moves forward, covering more of the Ray workloads this stack already runs, including distributed training and batch inference. For reinforcement learning from human feedback on ROCm 10.0.0, see the [verl on ROCm 10.0.0](https://rocm.blogs.amd.com/artificial-intelligence/verl_0.9.0_rocm7.15/README.html) post. Follow the [ROCm blogs](https://rocm.blogs.amd.com/) for the next installments. Install Ray 2.58.0 from the guide above, confirm the cluster, and run the serving example so an existing Ray workload can land on Instinct today.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, AMD Instinct, ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved
