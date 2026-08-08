---
blogpost: true
blog_title: "Benchmarking Kimi-K3 Across vLLM, SGLang, and ATOM on MI350X"
date: "02 Aug 2026"
author: "Yu Shao, Tej Kiran, Gurpreet Dhami, Chaitanya Sri Krishna Lolla, Aswin Mathews, Rahul Garg, Peng Sun"
thumbnail: ''
tags: "AI/ML, LLM, Performance, HPC, Serving"
category: "Applications & models"
target_audience: "AI developers, LLM serving/benchmarking engineers, ROCm users, AMD Instinct customers, MLOps/performance engineers evaluating day-0 model enablement"
key_value_propositions: "Shows how MAD's declarative model registry and madengine runner turn day-0 Kimi-K3 (2.8T-parameter MoE) enablement across vLLM, SGLang, and ATOM into a single reproducible command per engine, with one shared benchmark sweep and a common core CSV schema that align the three engines' out-of-box configurations."
language: English
myst:
    html_meta:
        "author": "Yu Shao, Tej Kiran, Gurpreet Dhami, Chaitanya Sri Krishna Lolla, Aswin Mathews, Rahul Garg, Peng Sun"
        "description lang=en": "One declarative madengine command benchmarks day-0 Kimi-K3 across vLLM, SGLang, and ATOM on AMD Instinct MI350X, with a shared harness for extending the sweep to your own workload."
        "keywords": "Kimi-K3, MAD, madengine, vLLM, SGLang, ATOM, MI350X, MI355X, benchmarking, MXFP4, reproducibility"
        "vertical": "AI, HPC"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Applications and Models"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference"
        "amd_blog_topic_categories": "AI & Intelligent Systems"
        "amd_blog_authors": "Yu Shao, Tej Kiran, Gurpreet Dhami, Chaitanya Sri Krishna Lolla, Aswin Mathews, Rahul Garg, Peng Sun"
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

# Benchmarking Kimi-K3 Across vLLM, SGLang, and ATOM on MI350X

When Moonshot AI released the weights for **Kimi-K3** — a 2.8-trillion-parameter,
1M-context, natively-MXFP4 Mixture-of-Experts model — AMD Instinct™ support was
available on day 0 across **three independent serving frameworks**: vLLM, SGLang, and
ATOM. The recipes target the gfx950 generation (MI350X and MI355X); the measurements
in this post were taken on an 8× MI350X node.

Day-0 announcements usually establish that a model runs. This post covers the next
step: benchmarking a 1.56 TB model across three engines on new-silicon (gfx950)
kernels, and knowing what the resulting numbers do and do not support. Three engines
mean three container images, three server launchers, three benchmark clients, and
three result formats — and three chances for an unnoticed difference in workload to
turn a comparison into an artifact.

MAD's answer is **madengine**, an automation layer that reduces each engine's run to a
single declarative command with a versioned config. This post covers that flow, the
day-0 results it produced, and the limits of what those results establish.

By the end, you will know how to run the same day-0 Kimi-K3 benchmark on all three
engines with one command each, how to read the CSV they share, how to retarget the
sweep to your own input/output lengths and concurrency, and where the measurement's
boundaries lie.

---

## Key Takeaways

- **One command per engine.** `madengine run --tags pyt_vllm_kimi-k3` (or `pyt_sglang_kimi-k3`,
  `pyt_atom_kimi-k3`) builds the image, launches the server, drives the benchmark, and
  emits a normalized `perf_Kimi-K3.csv` — no manual container plumbing.
- **One shared sweep, three engines.** All three frameworks run the same primary workload
  axes — 8192 input / 1024 output, concurrency `1·4·8·16·32·64·128`, TP8 — using each
  engine's out-of-box configuration. Secondary settings still differ (model dtype,
  KV-cache dtype, prompt-length sampling, EOS handling); see Table 4.
- **Declarative configs, not shell scripts.** Every server flag, environment variable,
  and sweep axis lives in a versioned YAML. The recipe *is* the config; reproducing a run
  is re-running the file.
- **Your sweep, same harness.** Copy the config, set your own ISL/OSL and concurrency, and
  point a run at it with `--additional-context`. Same containers, same CSV schema.
- **Hardware-aware guardrails.** `skip_gpu_arch: "gfx942"` keeps today's gfx950 recipe
  from silently mis-running on MI300X/MI325X, which it hasn't been validated for;
  `arch_overrides` is the same mechanism a future MI300X wideEP recipe would use once one
  is validated.
- **Reliability by design.** Automatic server-health gating, unbuffered logging, model-cache
  reuse, and a common core CSV schema make a run on your cluster reproduce a run on ours.
- **An out-of-box snapshot.** A default `madengine run` on 8× MI350X shows all three
  engines within 0.5% of each other through concurrency 32, then separating as batching
  policy starts to dominate. Read it as a launch-day picture of three enablement efforts
  at different stages, not as an engine comparison. See Figure 3.

---

## Test Configuration and Day-0 Evidence

### Day-0 enablement, per engine

Kimi-K3 weights were published by Moonshot AI on 2026-07-27. All three engines shipped
K3-capable containers dated the same day, and the MAD recipes benchmarked here landed
two days later.

| Engine | Container image (tag as run) | Day-0 evidence |
|---|---|---|
| vLLM | `vllm/vllm-openai-rocm:kimi-k3` | [vLLM day-0 blog, 2026-07-27](https://vllm.ai/blog/2026-07-27-k3); [MI355X recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3?hardware=mi355x) |
| SGLang | `lmsysorg/sglang-rocm:rocm720-mi35x-k3-20260727` | [day-0 tracking issue #32548](https://github.com/sgl-project/sglang/issues/32548) (support PR #32541); [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) |
| ATOM | `rocm/atom-dev:rocm7.2.4_ubuntu24.04_py3.12_pytorch2.10.0_20260727_kimi_k3` | [Kimi-K3 on AMD Instinct GPUs](https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-k3-on-amd-instinct-gpus.html) |
| MAD | — | Kimi-K3 support merged in [MAD PR #186](https://github.com/ROCm/MAD/pull/186), 2026-07-29 |

Table 1: Day-0 enablement evidence per engine. The `20260727` tag suffix on the SGLang
and ATOM images is the build date of the K3-enabled container.

### System configuration

| Component | Value |
|---|---|
| System | Supermicro AS-8126GS-TNMR (H14DSG-OD baseboard) |
| GPUs | 8× AMD Instinct™ MI350X (gfx950), TP8 |
| CPU | 2× AMD EPYC 9575F, 64 cores each |
| Host memory | 3 TiB (24× 128 GiB DDR5-6400 RDIMM) |
| Storage | 2× Micron 7450 3.84 TB NVMe (PCIe Gen 5) |
| Host networking | 2× Broadcom BCM57508 (up to 200 GbE), 2× Intel X710, 2× Intel X550 |
| GPU interconnect | Infinity Fabric, single node; no multi-node fabric used |
| OS | Ubuntu 24.04 |
| ROCm version | 7.2.3 (`rocm-core 7.2.3.70203-90`) |
| amdgpu driver | 6.16.13 (`amdgpu-dkms 1:6.16.13.30300100-2303411.24.04`) |
| VBIOS / firmware | `113-M350-01-1K1-030A` (identical across all 8 GPUs) |
| GPU details | gfx950, device ID `0x75a0`, 256 CUs, HSA runtime 1.18 |

Table 2: Host and accelerator configuration for every measurement in this post. The
ROCm, driver, and VBIOS values were captured by madengine's environment collection
during the benchmark runs themselves, not reconstructed afterward.

### Pinned versions

Reproducing these numbers exactly requires pinning the software, not tracking `main`:

| Component | Pin |
|---|---|
| MAD | commit `a20c885` (PR #186) |
| madengine | tag `v2.1.2` |
| Kimi-K3 model revision | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` |
| vLLM image digest | `sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b` |
| SGLang image digest | `sha256:3c01f73fe23aebf4a8853de0899a70b75c2af6c0409d2331353847aac4d3f906` |
| ATOM image digest | `sha256:04ce312d4124e3c7f8a62a321bbd2d3f07328855f362f8e6374bdc5f51afc233` |

Table 3: Version pins. The image digests are the ones the builds actually resolved and
baked in, read back from the build logs, so they identify what ran rather than what the
tag points at today. That distinction matters here: Docker Hub tags are mutable, and for
both the vLLM and SGLang tags a `docker manifest inspect` run during the same build
session already returned a *different* digest than the build had pulled
(`a8798d4a…` and `c75ce7a3…` respectively). Pull by digest, not by tag, to reproduce
these runs. The ATOM Dockerfile is the only one of the three that already pins by digest
in-repo.

---

## The Day-0 Benchmarking Problem

Kimi-K3 is not a bigger Kimi-K2. As [vLLM's day-0 announcement](https://vllm.ai/blog/2026-07-27-k3)
notes, it changes the serving problem along many axes at once — hybrid KDA (Kimi Delta
Attention) + full attention, Attention Residuals, 896 routed experts (16 active), MXFP4
weights with the SiTU activation, and native vision. Each axis lands somewhere different
in each engine's stack.

Now multiply that by three frameworks, each with its own conventions:

| Concern | vLLM | SGLang | ATOM |
|---|---|---|---|
| Container image | `vllm/vllm-openai-rocm:kimi-k3` | `lmsysorg/sglang-rocm:...-k3-20260727` | `rocm/atom-dev:...20260727_kimi_k3` |
| Server entrypoint | `vllm serve` | `sglang serve` | `python -m atom.entrypoints.openai_server` |
| MoE selector env | `AITER_SITUV2_A8W4=1` | `AITER_FLYDSL_FORCE=1` + `SGLANG_AITER_K3_OPT=1` | `AITER_FLYDSL_FORCE=1` + `ATOM_USE_TRITON_MOE=0` |
| Attention flag | (engine default) | `--attention-backend triton` | `ATOM_USE_UNIFIED_ATTN=1` |
| Reasoning parser | `--reasoning-parser kimi_k3` | `--reasoning-parser kimi_k3` | (not set) |
| Benchmark client | `vllm bench serve` | `sglang.benchmark.serving` | `atom.benchmarks.benchmark_serving` |
| Result JSON schema | `total_token_throughput`, `median_ttft_ms`… | SGLang JSONL | ATOM `median_*_ms` |

Doing this by hand means three sets of `docker run` invocations, three server launch
sequences, three health-check loops, and three JSON parsers — and then a fourth,
error-prone step of hand-reconciling the outputs. Every one of those steps can
introduce a divergence that is invisible in the final number: a mismatched input
length, a different concurrency point, a missing environment flag that selects the
slow MoE path.

MAD automates those steps.

---

## The MAD Automation Flow

MAD is built around a **declarative model registry** (`models.json`) and the
**madengine** runner. A single entry fully describes how to build, run, and score a
workload — and one command executes the whole pipeline.

![Figure 1: The madengine execution pipeline](images/kimi-k3-madengine-pipeline.png)

Figure 1: The madengine execution pipeline. One registry entry drives all five stages;
the only thing that changes between engines is which row of `models.json` you select.

For every model, madengine performs the same five steps — **Build → Start → Resolve →
Execute → Report** — regardless of which engine sits underneath. That uniformity is the
whole point: the *operator experience* is identical across vLLM, SGLang, and ATOM, even
though the internals could not be more different.

### The registry entry is the contract

Here are the fields that make Kimi-K3-on-vLLM a first-class, one-command benchmark
(the real entry also carries bookkeeping fields — `url`, `owner`,
`training_precision`, `timeout` — omitted here for readability):

```json
{
  "name": "pyt_vllm_kimi-k3",
  "dockerfile": "docker/pyt_vllm_kimi_k3",
  "scripts": "scripts/vllm/run.sh",
  "data": "huggingface",
  "n_gpus": "-1",
  "multiple_results": "perf_Kimi-K3.csv",
  "tags": ["pyt", "vllm", "inference"],
  "skip_gpu_arch": "gfx942",
  "args": "--model_repo moonshotai/Kimi-K3 --config configs/default.yaml"
}
```

Three engines, three near-identical entries — differing only in `dockerfile`,
`scripts`, and `config`. The SGLang entry even ships **two variants** (`nospec` and
`dspark` for speculative decoding) from the same script by passing `--variant`, and the
same `perf_Kimi-K3.csv` collects them all.

```json
{ "name": "pyt_sglang_kimi-k3",
  "scripts": "scripts/sglang/run_kimi_k3.sh",
  "args": "--model_repo moonshotai/Kimi-K3 --config configs/kimi_k3.yaml --variant nospec" }

{ "name": "pyt_sglang_kimi-k3_dspark",
  "scripts": "scripts/sglang/run_kimi_k3.sh",
  "args": "--model_repo moonshotai/Kimi-K3 --config configs/kimi_k3.yaml --variant dspark" }
```

### What's actually running under those five stages

Figure 1 is the operator's view. Figure 2 shows the same pipeline internally:

![Figure 2: madengine's internal call chain for a Kimi-K3 run](images/kimi-k3-madengine-architecture.png)

Figure 2: madengine's internal call chain for a Kimi-K3 run — the same classes handle
every model in the registry.

`madengine run` resolves the registry entry, builds an image from the entry's
`dockerfile`, reads the host GPU architecture from `rocminfo` (`gfx950` on
MI350X/MI355X, `gfx942` on MI300X/MI325X) and checks it against `skip_gpu_arch`, then
resolves `MAD_DATAHOME` for the `"data": "huggingface"` entry, launches the container,
and runs the entry's `scripts` inside it. The script writes `perf_Kimi-K3.csv`
(madengine passes the `multiple_results` value in as `MAD_OUTPUT_CSV`), which madengine
folds into the run-level `perf.csv`. None of this is Kimi-K3-specific: enabling K3 meant
adding `models.json` rows, Dockerfiles, and run scripts, with no changes to madengine
itself. See the [madengine repository](https://github.com/ROCm/madengine) for the
implementation.

---

## The Config *is* the Recipe

The benchmark recipe lives in version-controlled YAML, not in a person's terminal
history. Every server flag, every environment toggle that selects a kernel path, and
every sweep axis is declarative and auditable.

Here is the Kimi-K3 block of the vLLM config (`scripts/vllm/configs/default.yaml`),
lightly abridged — the comments are condensed and a trailing `bench_args` block that
disables the GSM8K accuracy run is omitted:

```yaml
- benchmark: serving
  model: moonshotai/Kimi-K3
  tp: 8
  inp: 8192
  out: 1024
  dtype: auto
  max_concurrency: 1 4 8 16 32 64 128 256      # the shared K3 sweep
  env:
    VLLM_ROCM_USE_AITER: 1
    SAFETENSORS_FAST_GPU: 1
    AITER_SITUV2_A8W4: 1                        # selects the aiter a8w4 MoE path
    AITER_BF16_FP8_MOE_BOUND: 0
    VLLM_USE_BREAKABLE_CUDAGRAPH: 0
  extra_args:
    --moe-backend: auto
    --load-format: auto
    --gpu-memory-utilization: 0.95
    --mm-encoder-tp-mode: data                  # MoonViT-V2 is 401M; TP is pure overhead
    --max-num-seqs: 256
    --max-num-batched-tokens: 4096
    --reasoning-parser: kimi_k3                  # K3 always thinks
    --language-model-only: true                 # text-only bench frees VRAM for KV
```

Notice that the comments encode *why* each knob is set — `AITER_SITUV2_A8W4: 1`
"selects the AITER a8w4 MoE path"; `--mm-encoder-tp-mode: data` because "MoonViT-V2
is only 401M params; TP on it is pure comm overhead." The recipe is self-documenting,
and re-running it a month later on a different cluster reproduces the same run, because
there is no hidden state.

### One shared sweep across three engines

The primary reliability decision is that **all three engines run the same workload
axes**: `inp=8192, out=1024, TP8`, swept over concurrency. The shipped configs list
concurrency points `1·4·8·16·32·64·128·256`; the results in this post were measured
through 128, and the 256 point has not been run on any engine.

The sweep shape was chosen, not inherited. The SGLang config file documents how it was
reverse-engineered from the framework's public day-0 tracking issue so the numbers stay
comparable:

> *"(E2EL − TTFT) / TPOT + 1 lands on ~1024 output tokens for every row, and
> concurrency × (inp + out) / E2EL reproduces the reported total throughput only at
> inp=8192 (e.g. concurrency 8: 8 × 9216 / 31.297 s = 2355.8 vs 2356.21 reported).
> MAD's usual 1024/1024 would produce numbers that cannot be compared against the
> tracking issue."*

The workload was chosen so that MAD's output can be checked against the framework
authors' own published figures, which gives the harness a cross-check against
measurement error.

| Axis | vLLM | SGLang | ATOM |
|---|---|---|---|
| Tensor parallel | 8 | 8 | 8 |
| Input length | 8192 | 8192 | 8192 |
| Output length | 1024 | 1024 | 1024 |
| Concurrency measured | 1→128 | 1→128 | 1→128 |
| Model dtype | `auto` | `bfloat16` | — |
| KV cache dtype | — | — | `fp8` |
| Prefix caching | off | off (`--disable-radix-cache`) | off (`--no-enable_prefix_caching`) |
| Prompts per point | 10 × concurrency | 10 × concurrency | 10 × concurrency |
| `--random-range-ratio` | not passed (client default) | `1.0` | `0.8` |
| `--ignore-eos` | yes | not passed | yes |

Table 4: The shared K3 sweep. The primary axes the config controls — TP, ISL, OSL,
concurrency, prompt count — are aligned across engines. The remaining rows are each
engine's out-of-box defaults, which differ and are not normalized by the runners. Note
`dtype` and KV cache dtype are distinct settings: vLLM and SGLang expose only a general
model/activation `dtype` flag for this recipe, while ATOM's config sets a genuine
`kv_cache_dtype`; neither vLLM nor SGLang override their (bf16) KV cache dtype here.

The last two rows are the caveat: each engine's benchmark client has its own defaults,
and the runners do not currently normalize them. vLLM omits `--random-range-ratio`
entirely, whose client default of `0.0` pins every prompt at exactly 8192 tokens.
SGLang passes `1.0`, which under its own `[input_len × ratio, input_len + 1]` sampling
convention is likewise effectively exact. **ATOM passes `0.8`, so its prompt lengths
are sampled over a range rather than pinned** — ATOM is measuring a nearby but not
identical workload. Separately, SGLang does not pass `--ignore-eos`, so a request that
emits an EOS token can finish before 1024 output tokens, where vLLM and ATOM force the
full output length.

These differences are real and unquantified: no controlled A/B run has isolated their
effect on the reported throughput. Aligning the three bench invocations is tracked as a
follow-up. Until then, treat cross-engine gaps as indicative rather than as measured
engine differences.

The sweep expansion itself is handled generically by the runner — `max_concurrency`
is a space-separated list that the runner takes a Cartesian product over, so adding a
concurrency point is a one-token edit, not a code change:

```python
# scripts/vllm/run_vllm.py
SUPPORTED_LIST_ARGS = ['model', 'tp', 'inp', 'out', 'bs', 'num_prompts', 'max_concurrency']
# each space-separated value is expanded via itertools.product into one run per combination
```

---

## Reliability Engineering: The Guardrails

Automation that produces wrong numbers quickly is worse than no automation. The MAD flow
includes several guardrails so that a run that reports success is a run that measured
what it claims to measure.

### 1. Server-health gating before measurement

Every serving runner launches the server as a subprocess and **polls it to readiness**
before sending a single benchmark request — so a slow 1.56 TB load never gets
mis-measured as high latency:

```sh
# the server is polled until healthy; only then does the benchmark client start
until curl -s http://localhost:8000/v1/models; do sleep 30; done
```

vLLM's runner allows 30 minutes for that poll. Both SGLang and ATOM raise it to 5400
seconds, appropriate for a multi-terabyte checkpoint — SGLang polls `/health` rather
than `/v1/models`, and ATOM's `_wait_for_server()` also watches the server process
itself, returning as soon as it exits so an OOM during initialization fails fast
instead of looking like a hang until the timeout expires.

### 2. Common core output schema

Every engine — no matter its native JSON format — is parsed into a **common CSV core**,
so downstream dashboards and regression checks never special-case the engine:

```text
model, benchmark, tp, inp, out, num_prompts,
max_concurrency, cmd, performance, metric, unit
```

Each runner adds a few engine-native columns on top of that shared core — vLLM adds
`dtype` and `bs`; SGLang adds `variant` (for the `nospec`/`dspark` split) and `dtype`;
ATOM adds `kv_cache_dtype`, `hf_pipeline_tag`, and `bs`. Each engine's run produces its
own `perf_Kimi-K3.csv`, and `update_perf_csv` merges each into the run-level `perf.csv`,
carrying over any columns the base file doesn't already have — so no column is silently
dropped even though the three engines don't emit byte-identical headers.

The runner records not just throughput but a set of common latency metrics —
`median_ttft`, `median_tpot`, `median_itl`, `median_e2el` — plus the exact `cmd` that
produced the row, so any number in the CSV can be traced back to the precise invocation
that generated it. These are medians; the CSV does not carry the full latency
distribution, though each engine's raw result JSON retains the percentiles requested via
`--percentile-metrics`.

| Metric | Meaning | Unit |
|---|---|---|
| `throughput_tot` | Total token throughput | tok/sec |
| `throughput_gen` | Output (generation) throughput | tok/sec |
| `median_ttft` | Time to first token | ms |
| `median_tpot` | Time per output token | ms |
| `median_itl` | Inter-token latency | ms |
| `median_e2el` | End-to-end latency | ms |

Table 5: The common core metric schema emitted for every engine and every concurrency
point. Shared columns make cross-engine and cross-run comparison mechanical; each engine
adds its own extra columns on top.

### 3. Reproducible weights, cached once

The `data: "huggingface"` field wires in weight resolution. By default weights come
from the Hub (with `hf-transfer` for speed and `MAD_SECRETS_HFTOKEN` for gated repos),
but `MAD_DATAHOME` transparently redirects to a pre-downloaded local copy — so the
same 1.56 TB checkpoint is fetched once and reused across every engine and every rerun:

```sh
madengine run --tags pyt_vllm_kimi-k3 --keep-model-dir --live-output \
  --additional-context '{"docker_mounts": {"/model_weights": "/path/to/Kimi-K3"},
                         "docker_env_vars": {"MAD_DATAHOME": "/model_weights"}}'
```

`--keep-model-dir` preserves that cache between runs; `--live-output` streams the
unbuffered logs so a long sweep is observable in real time rather than a black box.

### 4. Hardware-aware gating — a default, not a hard law

Today's registry entries mark Kimi-K3 `skip_gpu_arch: gfx942`, because the day-0
recipes on all three engines assume the model's native MXFP4 weights sit on the gfx950
generation (MI350X and MI355X) and run a dense TP8 layout:

```json
"skip_gpu_arch": "gfx942"
```

madengine reads the host's `MAD_SYSTEM_GPU_ARCHITECTURE` (the ROCm architecture
string `rocminfo` reports — `gfx942` on MI300X/MI325X) and skips the workload rather
than silently producing a result under the wrong assumptions.
That gate is a property of *this recipe*, though, not of the model itself —
MI300X/MI325X lack gfx950's native MXFP4 support, but a config
built around wide expert-parallel (wideEP) sharding and a matched concurrency
profile could still place Kimi-K3's 896 experts across enough MI300X GPUs to serve
it, just with different quantization and a different parallelism shape than the
TP8 recipe this post benchmarks. The same `arch_overrides` block the shipped configs
already use elsewhere — e.g. `scripts/vllm/configs/default.yaml` forcing TP8 on gfx942
where TP4 would OOM for other MoE models — is exactly the hook a future MI300X Kimi-K3
config would use, so `skip_gpu_arch` here should be read as "no validated recipe
yet," not "impossible."

---

## Bring Your Own Sweep: Custom ISL/OSL and Settings

The shared 8192/1024 sweep exists to align the three engines with each other and with
the framework authors' published figures. It is almost certainly not *your* workload. A
summarization service runs long-in/short-out; a code assistant runs the reverse; an
agentic loop runs neither. And Kimi-K3's headline spec is a 1M-token context, which 8192
barely touches. This post contains no long-context data: `inp` has not been pointed at
32k, 128k, or beyond on any of the three engines. That is the natural next data point,
and the harness below is what you would use to generate it.

### 1. Copy the config, change the shape

In your clone of MAD, copy the shipped K3 block into a new file next to it — the
`configs/` directory alongside the runner is the path the container will look in:

```sh
cp scripts/vllm/configs/default.yaml scripts/vllm/configs/custom.yaml
```

Then trim it to the single block you care about and change the workload axes:

```yaml
# scripts/vllm/configs/custom.yaml — a 2k/2k sweep instead of the shared 8k/1k
- benchmark: serving
  model: moonshotai/Kimi-K3
  tp: 8
  inp: 2048          # your input sequence length
  out: 2048          # your output sequence length
  dtype: auto
  max_concurrency: 1 8 32 64        # your concurrency points
  env:
    VLLM_ROCM_USE_AITER: 1
    SAFETENSORS_FAST_GPU: 1
    AITER_SITUV2_A8W4: 1            # keep this — dropping it silently
    AITER_BF16_FP8_MOE_BOUND: 0     # falls back to the slower a16w4 MoE path
    VLLM_USE_BREAKABLE_CUDAGRAPH: 0
  extra_args:
    --moe-backend: auto
    --gpu-memory-utilization: 0.95
    --max-num-seqs: 256
    --max-num-batched-tokens: 4096
    --reasoning-parser: kimi_k3
    --language-model-only: true
```

The `env` and `extra_args` blocks are the tuned part of the recipe — carry them over
verbatim unless you are deliberately measuring one of those knobs. `AITER_SITUV2_A8W4`
in particular selects the fast MoE kernel path; a "custom config" that quietly omits it
will produce numbers that look like a regression but are really a misconfiguration.

### 2. Point a run at it

`--additional-context` overrides the registry entry for a single invocation:

```sh
madengine run --tags pyt_vllm_kimi-k3 --keep-model-dir --live-output \
  --additional-context '{"model_args": "--model_repo moonshotai/Kimi-K3 --config configs/custom.yaml",
                         "docker_mounts": {"/model_weights": "/shareddata/Kimi-K3"},
                         "docker_env_vars": {"MAD_DATAHOME": "/model_weights"}}'
```

Two things are easy to get wrong here:

- **`model_args` replaces the registry's `args` string — it does not merge with it.**
  Whatever you pass is the *complete* argument list handed to the run script, so
  `--model_repo moonshotai/Kimi-K3` has to be restated alongside your `--config`.
  Passing only `--config configs/custom.yaml` leaves the model repo empty and the run
  script exits on a missing argument.
- **The whole thing is one JSON object.** All three keys — `model_args`, `docker_mounts`,
  `docker_env_vars` — live inside a single pair of braces in a single pair of quotes.

There is also a shorthand. All three run scripts accept `CONFIG` as an environment
variable, so you can select a config without restating the model repo at all:

```sh
madengine run --tags pyt_vllm_kimi-k3 --keep-model-dir --live-output \
  --additional-context '{"docker_env_vars": {"CONFIG": "configs/custom.yaml",
                                             "MAD_DATAHOME": "/model_weights"},
                         "docker_mounts": {"/model_weights": "/shareddata/Kimi-K3"}}'
```

### 3. Where the file has to live

The config path is resolved *inside the container*, relative to the scripts directory
that madengine copies in — so `configs/custom.yaml` means
`scripts/vllm/configs/custom.yaml` in your checkout. A YAML sitting in `/tmp` on the
host will not be found. If you would rather not put the file in the repo, mount it and
pass an absolute container path instead:

```sh
--additional-context '{"docker_mounts": {"/cfg": "/home/me/sweeps"},
                       "docker_env_vars": {"CONFIG": "/cfg/custom.yaml"}}'
```

The same three flags work for the other two engines; the only difference is which
config the entry starts from — `scripts/sglang/configs/kimi_k3.yaml` for SGLang
(which also takes `--variant nospec|dspark`) and `scripts/atom/configs/default.yaml`
for ATOM.

### What you give up

A custom sweep is no longer comparable to Figure 3, Table 7, or the framework tracking
issue — those numbers are only meaningful at 8192/1024. That is a fair trade when the
question is "how does K3 serve *my* traffic on this node," and the wrong tool when the
question is "is this engine faster than that one." Keep the shared sweep for the second
question; because the core schema is shared, both sets of numbers land in the same
`perf_Kimi-K3.csv` shape, so you can carry both.

| Knob | What it changes | Watch out for |
|---|---|---|
| `inp` / `out` | Input / output sequence length | Long `inp` raises KV pressure; may need lower `max_concurrency` |
| `max_concurrency` | Sweep points (space-separated) | Each value is a full server-side run — cost scales linearly |
| `tp` | Tensor parallel degree | K3 needs ~1680 GB; TP8 is the only fit on one 8-GPU node |
| `extra_args` | vLLM server flags | Passed through verbatim to `vllm serve` |
| `env` | Kernel-path selection | Dropping `AITER_SITUV2_A8W4` costs real throughput |

Table 6: The knobs most worth editing in a custom config.

---

## Results: Out-of-Box, All Three Engines

Reproducing the day-0, three-engine benchmark is three commands:

```sh
# vLLM — 8k/1k serving sweep, TP8
madengine run --tags pyt_vllm_kimi-k3   --keep-model-dir --live-output

# SGLang — same sweep; add the _dspark tag for speculative decoding
madengine run --tags pyt_sglang_kimi-k3 --keep-model-dir --live-output

# ATOM — same sweep, fp8 KV cache
madengine run --tags pyt_atom_kimi-k3   --keep-model-dir --live-output
```

Each produces a `perf_Kimi-K3.csv` sharing the same core columns, so stacking the three
CSVs lines the concurrency axis up row-for-row.

Below is an out-of-box `madengine run` on 8× MI350X, all three engines, no tuning beyond
the shared config in this post.

![Kimi-K3 day-0 OOB serving throughput: vLLM vs SGLang vs ATOM on MI350X](images/kimi-k3-vllm-sglang-throughput.png)

Figure 3: **Out-of-box snapshot.** Total token throughput vs. max concurrency, 8192 in /
1024 out, TP8, 8× MI350X, measured 2026-07-29. All three engines from the same madengine
sweep, each using its day-0 out-of-box configuration; one run per concurrency point, no
repetitions. This is not a tuned comparison and not a leaderboard — it is a picture of
where each stack's K3-specific work stood on launch day, and it is the thing most likely
to have changed by the time you re-run the command.

Two properties of the shape matter more than the ordering. First, this is a *functional*
result before it is a performance one: a 2.8T-parameter MoE with a new attention design
serves correctly on a single 8× MI350X node, on three separate engines, on day 0.
Second, through concurrency 32 all three land on essentially the same curve — 4,676 /
4,694 / 4,692 tok/s, a spread under 0.5%. That convergence is an observation about this
configuration; a single throughput point does not isolate which component — scheduler,
kernels, precision, benchmark client, or hardware — sets the ceiling there.

The curves separate past 32, and that is where engine-specific maturity is visible:
continuous-batching policy, MoE dispatch at large batch, KV-cache layout. vLLM keeps
climbing to 8,228 tok/s at concurrency 128; SGLang and ATOM flatten. ATOM's curve in
particular reflects a generic serving path rather than a K3-tuned one — its day-0 recipe
carries no model-specific kernel or batching work, so the high-concurrency numbers
measure an un-optimized baseline, not a ceiling. The useful reading is the *gap between
the low- and high-concurrency regimes* for each engine, as an indication of how much
batching work remains. Because the primary sweep axes are aligned, those gaps are
comparable across engines subject to the out-of-box differences in Table 4, whose effect
on these numbers has not been measured.

| Concurrency | vLLM (tok/s) | SGLang (tok/s) | ATOM (tok/s) |
|---:|---:|---:|---:|
| 1 | 287.28 | 422.75 | 346.12 |
| 4 | 1,000.47 | 1,388.58 | 1,187.52 |
| 8 | 1,744.78 | 2,263.67 | 2,056.96 |
| 16 | 2,985.02 | 3,451.65 | 3,297.54 |
| 32 | 4,675.92 | 4,693.86 | 4,691.54 |
| 64 | 6,567.25 | 5,994.97 | 5,024.71 |
| 128 | 8,228.15 | 6,293.32 | 5,136.26 |

Table 7: **Out-of-box snapshot.** Raw total-token-throughput values behind Figure 3,
straight out of each engine's `perf_Kimi-K3.csv`. Measured 2026-07-29 on 8× MI350X, one
run per concurrency point, no repetitions — so these values carry no run-to-run variance
estimate. The shipped configs list concurrency 256, but the runs reported here were taken
through 128 only; 256 has not been measured on any of the three engines.

---

## Limitations and Follow-Ups

The measurements above are a day-0 snapshot, and the following bound what they support:

- **MI350X only.** The recipes target gfx950 (MI350X and MI355X), but every number here
  comes from an 8× MI350X node. MI355X has not been measured.
- **Single run per point.** Each concurrency point is one measurement over
  `10 × concurrency` prompts, with no repetitions and no warm-up iterations on the
  serving path. There is no run-to-run variance estimate, so small cross-engine
  differences should not be over-read.
- **Concurrency 256 unmeasured.** The shipped configs list it; no engine has run it.
  Aligning the configs with what was actually measured is a follow-up in the MAD repo.
- **Bench-client defaults not normalized.** The prompt-length sampling and EOS
  differences in Table 4 are unquantified; no A/B run has isolated their effect.
- **No long-context data.** Kimi-K3 supports a 1M-token context; the sweep uses 8192.
  Long-context behavior on all three engines is unmeasured.
- **Mutable image tags.** The digests in Table 3 identify exactly what ran, but the vLLM
  and SGLang Dockerfiles reference their base images *by tag*, and both tags were
  observed serving a different digest within the same build session. Anyone rebuilding
  from the tag today may not get the image benchmarked here. The upstream framework
  commits inside those containers are also not recorded.

## Summary

The same registry-plus-runner pattern used for Kimi-K3 already spans the AMD MAD catalog
— vLLM, SGLang, ATOM, Primus/Megatron training, JAX MaxText, xDiT diffusion, and
disaggregated P/D serving — all driven by the same `madengine run --tags …` interface and
the same declarative configs.

That uniformity makes day-0 support repeatable rather than a one-off:

- **For model launches:** enabling a new model on a new engine is a registry entry, a
  Dockerfile, and a YAML — reviewable in a PR, not lost in a shell session.
- **For CI and regression:** the shared core schema and shared sweeps mean a nightly job
  can diff today's `perf_Kimi-K3.csv` against a reference and flag drift automatically.
- **For the community:** anyone with an 8× MI350X node can reproduce this benchmark from
  the configs in the repo, subject to the version pins in Table 3.

As Moonshot AI and the framework teams extend agentic serving — longer horizons, deeper
tool use, larger context — the same harness is what makes each step measurable on AMD
Instinct™ hardware.

---

## Get Started

To reproduce the runs in this post, pin both repositories to the versions in Table 3
rather than tracking `main`:

```sh
pip install git+https://github.com/ROCm/madengine.git@v2.1.2
git clone https://github.com/ROCm/MAD.git && cd MAD
git checkout a20c885

# pick your engine
madengine run --tags pyt_vllm_kimi-k3   --keep-model-dir --live-output
madengine run --tags pyt_sglang_kimi-k3 --keep-model-dir --live-output
madengine run --tags pyt_atom_kimi-k3   --keep-model-dir --live-output
```

To pin the checkpoint to the exact revision benchmarked here, fetch it once by revision
and point `MAD_DATAHOME` at the result:

```sh
hf download moonshotai/Kimi-K3 \
  --revision 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 \
  --local-dir /shareddata/Kimi-K3
```

- **madengine:** [github.com/ROCm/madengine](https://github.com/ROCm/madengine)
- **Model:** [moonshotai/Kimi-K3 on HuggingFace](https://huggingface.co/moonshotai/Kimi-K3)

## References

[1] [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) — Moonshot AI's 2.8T-parameter Mixture-of-Experts LLM

[2] [MAD](https://github.com/ROCm/MAD) — Model Automation and Dashboarding for AMD Instinct GPUs

[3] [madengine](https://github.com/ROCm/madengine) — The MAD execution engine and CLI

[4] [vLLM](https://github.com/vllm-project/vllm) — High-throughput serving engine for large language models

[5] [SGLang](https://github.com/sgl-project/sglang) — Fast serving framework for large language models

[6] [AITER](https://github.com/ROCm/aiter) — AI Tensor Engine for ROCm

## Disclaimers

Testing conducted by AMD on 2026-07-29. Hardware configuration: Supermicro
AS-8126GS-TNMR with 8× AMD Instinct™ MI350X (gfx950), 2× AMD EPYC 9575F, 3 TiB DDR5
system memory, Ubuntu 24.04, ROCm 7.2.3, amdgpu driver 6.16.13; TP8. Kimi-K3 checkpoint
≈ 1.56 TB. The recipes target the
gfx950 generation (MI350X and MI355X); MI355X was not measured. Results reflect
out-of-box engine configurations with one run per data point and no repetitions; see
Tables 2 and 3 for the full system and version configuration.

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED "AS IS" WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

Results shown are from specific test configurations and may vary based on workload,
model, and system configuration.

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED 'AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, AMD Instinct, AMD ROCm, CDNA, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies. Linux is the registered trademark of Linus Torvalds in the U.S. and other countries. PyTorch, the PyTorch logo and any related marks are trademarks of The Linux Foundation. vLLM is a trademark of vLLM Project. All other trademarks and product names referenced in this publication, including Kimi-K3, Moonshot AI, SGLang, ATOM, and HuggingFace, are the property of their respective owners.
© 2026 Advanced Micro Devices, Inc. All rights reserved
