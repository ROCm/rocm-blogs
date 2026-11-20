---
blogpost: true
blog_title: "FlashInfer on ROCm 10.1: The FlashInfer API with Automatic Kernel Routing"
date: "20 Nov 2026"
author: "Debasis Mandal, Mukhil Azhagan Mallaiyan Sathiaseelan, Yao Liu, Phani Vaddadi, Vish Vadlamani"
thumbnail: 'FlashInfer-thumbnail3.png'
tags: "Serving, AI/ML, GenAI, LLM, Optimization, PyTorch"
category: "Applications & models"
target_audience: "AI practitioners"
key_value_propositions: "FlashInfer on ROCm brings the upstream v0.6.18 API to CDNA3 and CDNA4 on ROCm 10.1.0, with the same wrapper signatures as upstream, and backend=\"auto\" routing each call to an in-tree or AITER kernel."
language: English
myst:
    html_meta:
        "author": "Debasis Mandal, Mukhil Azhagan Mallaiyan Sathiaseelan, Yao Liu, Phani Vaddadi, Vish Vadlamani"
        "description lang=en": "Explore the new attention and fused ops in FlashInfer on ROCm: the upstream API, with each call routed to optimized AITER or in-tree HIP kernels."
        "keywords": "FlashInfer, AITER, ROCm, CDNA3, CDNA4, MI300X, MI355X, attention, prefill, decode, FP8, KV cache, LLM, serving"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI, Data Science"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Debasis Mandal, Mukhil Azhagan Mallaiyan Sathiaseelan, Yao Liu, Phani Vaddadi, Vish Vadlamani"
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

# FlashInfer on ROCm 10.1: The FlashInfer API with Automatic Kernel Routing

FlashInfer<sup><a href="#references">1</a></sup> is an open-source library of attention and large language model (LLM) serving kernels. FlashInfer on ROCm™ brings its API to AMD Instinct™ GPUs, and this release continues from the [prefill-attention release](https://rocm.blogs.amd.com/artificial-intelligence/flashinfer-release2/README.html). Cascade attention, multi-head latent attention (MLA), and the standalone rotary position embedding (RoPE) ops, all in progress then, have now landed. New in this release are 8-bit floating-point (FP8) prefill, POD-Attention (prefill and decode overlapped in one kernel), fused mixture-of-experts (MoE), and AI Tensor Engine for ROCm (AITER) decode under CUDA graph capture. All of it runs on CDNA3 (gfx942: MI300X and MI325X)<sup><a href="#references">2</a></sup> and CDNA4 (gfx950: MI350X and MI355X)<sup><a href="#references">3</a></sup> GPUs.

The point of FlashInfer on ROCm is the FlashInfer API itself: the same wrappers and the same `plan()` and `run()` model as on CUDA. The decode, prefill, and MLA wrappers keep the signatures of upstream [FlashInfer v0.6.18](https://github.com/flashinfer-ai/flashinfer/tree/v0.6.18), argument order included, so a call written for FlashInfer on CUDA binds the same arguments on ROCm. What does not carry over is listed in the [support matrix](#support-matrix) below and in the repository's [backend notes](https://github.com/AMD-Ecosystem/flashinfer/blob/release/0.6.18+amd.2/docs/rocm/backends.md).

Behind that API, FlashInfer on ROCm combines its own in-tree Heterogeneous-computing Interface for Portability (HIP) kernels with specialized libraries from the ROCm ecosystem where they serve a call well. Today that is AITER<sup><a href="#references">4</a></sup>, AMD's library of tuned kernels for Instinct GPUs, built on Composable Kernel (CK Tile)<sup><a href="#references">5</a></sup>, hand-written assembly, and HIP. Which library serves a call is a backend detail: the API, and the code that calls it, stay the same.

The default `backend="auto"` makes the choice per call, following routing rules set by AMD's own testing and benchmarking on both CDNA3 and CDNA4, so callers get AMD's routing choice without changing any code. In this blog, you will explore what FlashInfer on ROCm supports, how `backend="auto"` routes each call, how an in-tree kernel uses a CDNA4 instruction, and how to build FlashInfer on ROCm and use it to serve a batch of chat requests.

## What's New in This Release

### One API, Routed by `backend="auto"`

Single, batch paged, and batch ragged prefill, and batch decode each have two backends, AITER and the in-tree HIP kernels, and `auto` picks one per call. MLA and fused MoE have only the AITER backend. Every other op has an in-tree HIP kernel; some norm, activation, RoPE, and KV-append (key-value cache append) entry points can also opt in to AITER. Where an entry point takes a `backend=` argument, it chooses between them. It defaults to `"auto"`, and what `auto` does depends on the op:

![Flowchart of how backend="auto" picks a kernel. Attention calls (single and batch prefill, batch decode) first check whether AITER can serve the call; if not, for example an FP8 KV cache with a 16-bit query, a custom mask, the HND layout, tensor-core decode, or graph capture without max_seq_len, they run on the in-tree HIP kernel. If AITER can serve the call, per-architecture rules pick AITER or keep the in-tree kernel, for example for a short query. Norm, activation, RoPE and KV-append ops always run on the in-tree HIP kernel, with AITER opt-in where offered. Sampling, the logits processor, block-sparse, POD, Gemma norm, single decode and the cascade merge run only on the in-tree HIP kernel. MLA and fused MoE run only on AITER.](images/auto-routing.svg)

For the attention ops, `auto` first checks that AITER can serve the call at all (dtypes, KV layout, masks, and head dimensions), and then applies per-architecture routing rules measured on both gfx942 and gfx950. The rules live in one table, [`flashinfer/rocm/arch_caps.py`](https://github.com/AMD-Ecosystem/flashinfer/blob/release/0.6.18+amd.2/flashinfer/rocm/arch_caps.py), from which the repository's support matrix is generated. After `plan()`, the batch prefill and batch decode wrappers report the choice in `wrapper.backend` (`"aiter"`, or `"fa2"` for the in-tree kernel) and, for most AITER declines, the reason in `wrapper.backend_fallback_reason`.

AITER is integrated at the C++ level. When a call runs on AITER, FlashInfer launches the kernel from its own C++ code, either by loading AITER's compiled attention kernels directly or by linking AITER's C++ entry points for MoE, norm, activation, RoPE and KV append, rather than through AITER's Python API. AITER's Python side is used only to build, locate or probe a kernel variant, at `plan()` or, for ops that have no `plan()`, on the first call; MLA is the one op that still runs AITER through Python.

### Where the In-Tree HIP Kernels Run

The in-tree kernels are not only a fallback for machines without AITER. As the figure shows, they serve the root mean square normalization (RMSNorm), activation, RoPE, KV-append, and sampling calls a model makes by default, and the attention calls AITER cannot serve, including an FP8 KV cache read by a 16-bit query, and batch decode with `use_tensor_cores=True`, which speculative-decode verify uses. These kernels are optimized for CDNA3 and CDNA4 in their own right rather than carried over unchanged from the CUDA sources. The RoPE kernels and the activation kernels for the sigmoid linear unit (SiLU) and Gaussian error linear unit (GELU) were retuned on CDNA3 (MI300X), and the norm and prefill kernels size their shared-memory use from the local data share (LDS) capacity of the GPU they run on, which is larger on CDNA4. The example below shows the kind of work involved.

#### Example: Transposing V in One CDNA4 Instruction

The in-tree prefill kernel is a FlashAttention-2 design built on the matrix fused multiply-add (MFMA) instructions of CDNA. Each tile does two matrix multiplies: `Q·Kᵀ` for the attention scores, then `P·V` to apply them. K fragments are read from LDS, the on-chip shared memory, in the layout the MFMA expects, but V is needed in the other orientation. So on CDNA3 the kernel loads each V fragment and then transposes it in registers with six cross-lane permutes (`ds_bpermute_b32`) before it feeds the `P·V` MFMAs.

CDNA4 adds a transposing LDS read, `ds_read_b64_tr_b16` (`__builtin_amdgcn_ds_read_tr16_b64_v4f16` in HIP), which returns the fragment already transposed in one instruction, replacing both the load and the shuffles.

![Diagram: on CDNA3 the V fragment is read from LDS with load_fragment and then transposed in registers by transpose_mma_tile, six ds_bpermute_b32 cross-lane shuffles, for every V fragment. On CDNA4 load_transposed_fragment reads the same LDS tile already transposed with one ds_read_b64_tr_b16 instruction, using a new per-lane seed.](images/pv-transpose-cdna3-vs-cdna4.svg)

It is not a drop-in replacement. Within each 16-lane group the instruction distributes data differently from the software path: seeded at the old address, lane 0 receives the plausible but incorrect elements `{0, 64, 128, 192}` instead of `{0, 16, 32, 48}`. Solving for the seed that produces the required fragment gives a new index into the same swizzled LDS tile, so the shared-memory layout does not change (`col` counts 64-bit units of four 16-bit elements):

```text
row = 4 * (lane / 16) + (lane % 16) / 4
col = lane % 4
```

### Running Under `torch.compile`

Set `FLASHINFER_USE_TORCH_CUSTOM_OPS=1` before importing `flashinfer`, and the attention, KV-append and fused MoE ops in the [`torch-compile.md`](https://github.com/AMD-Ecosystem/flashinfer/blob/release/0.6.18+amd.2/docs/rocm/torch-compile.md) table compile under `torch.compile`; keep `plan()` outside the compiled region, and set the flag only for compiled runs, since it adds host-side dispatch to every eager call. PyTorch 2.12 can silently miscompile an op that writes into a view of a larger tensor, so pass `append_paged_kv_cache` separately allocated K and V caches rather than one `[pages, 2, ...]` cache. The document covers the other layouts to avoid and what compiles with `fullgraph=True`.

### Support Matrix

The table summarizes what runs on both architectures; the [support matrix](https://github.com/AMD-Ecosystem/flashinfer/blob/release/0.6.18+amd.2/README.md#support-matrix) in the repository README lists every op with its backends, per architecture.

| Area | What runs on gfx942 and gfx950 |
| :--- | :--- |
| Prefill attention | Single, batch paged, batch ragged; multi-head attention (MHA), grouped-query attention (GQA), and multi-query attention (MQA); sliding window; logits soft-cap; custom masks (in-tree); FP8 KV cache |
| Decode attention | Single and batch paged; FP8 KV cache; CUDA graph capture; speculative-decode verify (chain drafts) |
| Other attention | Cascade (shared prefix), POD, block-sparse, MLA (DeepSeek V2/V3 layout) |
| Fused ops | RoPE (Llama and Llama 3.1 scaling, fused RoPE + FP8 quant + paged-KV append), RMSNorm and fused add-RMSNorm, Gemma RMSNorm, SiLU-and-mul and GELU-and-mul |
| MoE | `aiter_fused_moe` in bf16/fp16, and with FP8 expert weights (activations are quantized per token inside the call); every expert weight must first be pre-shuffled with `shuffle_moe_weight`, after `quantize_moe_weight` for FP8 |
| Sampling | Top-K, Top-P, Min-P, sampling from logits, the composable logits-processor pipeline |

FP8 KV caches on the in-tree kernels use the FNUZ encodings (`float8_e4m3fnuz`, `float8_e5m2fnuz`) on both architectures; the OCP `float8_e4m3fn` and `float8_e5m2` are refused, because the in-tree kernels would read them under the FNUZ encoding and return silently wrong values: half-scale, and NaN for negative zero. The AITER FP8 paths, an FP8 query and KV cache on batch paged prefill and FP8 MoE weights, instead take the GPU's own encoding, `float8_e4m3fnuz` on gfx942 and `float8_e4m3fn` on gfx950.

On gfx950, a `float8_e4m3fn` KV cache written for AITER FP8 prefill cannot also be read by decode. The in-tree decode refuses the OCP encoding, and AITER decode takes only 16-bit KV caches.

Upstream v0.6.18 is larger. What is absent splits by cause. **Not supported on ROCm** are modules that wrap a CUDA-only library, `PTX`, an SM-gated kernel, or CuTe DSL, so a ROCm equivalent is a rewrite rather than a port (for example `flashinfer.gemm`, the upstream CUTLASS `fused_moe`, `flashinfer.cudnn`, `flashinfer.xqa`). Where the import would already break, it is gated with a clear `ImportError`; otherwise the module imports and the call fails, so probe the symbol you need rather than trusting a successful import. **Without a ROCm implementation** are ordinary kernels that use none of those (for example `flashinfer.topk`, `flashinfer.mamba`'s selective state update). Separately, `flashinfer.norm.layernorm` and the quantized norm ops (`layernorm_quant`, `rmsnorm_quant`, `fused_add_rmsnorm_quant`) import but fail when called: their source builds, but the ROCm norm binding does not export them. The repository [README](https://github.com/AMD-Ecosystem/flashinfer/blob/release/0.6.18+amd.2/README.md#what-upstream-has-that-this-does-not) and backend notes have the complete lists.

## Getting Started

FlashInfer on ROCm is built from source in the configuration in the table below.

### Supported Configuration

One software configuration is supported: the one the repository's `docker/Dockerfile.rocm` builds.

| | Supported |
| :--- | :--- |
| GPUs | gfx942 (CDNA3: MI300X, MI325X), gfx950 (CDNA4: MI350X, MI355X) |
| ROCm | 10.1.0 |
| PyTorch | 2.12.0 |
| Python | 3.12 |
| OS | Ubuntu 24.04 |
| AITER | `amd-aiter` 0.1.21.post2, built from source at its tag |

No wheel or prebuilt image is published for this release; FlashInfer on ROCm is built from the repository. The development image pins a matched ROCm, PyTorch, Python and AITER set, so it is the shortest path to a working environment. Most of the image build is spent compiling AITER kernels ahead of time. The build refuses `USER_UID=0` (root) or a UID another account in the image still holds (the base image's `ubuntu` user at 1000 is removed first), so build the image as a regular user:

```bash
git clone --branch release/0.6.18+amd.2 https://github.com/AMD-Ecosystem/flashinfer.git
cd flashinfer
docker build -t flashinfer-dev:rocm10.1.0 -f docker/Dockerfile.rocm . \
  --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) \
  --build-arg USER_GID=$(id -g)
docker run -it --privileged --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size=64G \
  -v "$PWD":/workspace -w /workspace flashinfer-dev:rocm10.1.0
```

Inside the container:

```bash
python -m pip install --no-build-isolation -ve .
python -c "import flashinfer; print(flashinfer.__version__)"
```

In-tree kernels compile just in time on first use. The image compiles every AITER prefill variant ahead of time, but AITER batch decode compiles at `plan()`, and AITER fused MoE and the opt-in AITER ops build on first use, so `plan()` or the first call can take several minutes while the kernels compile.

The repository [README](https://github.com/AMD-Ecosystem/flashinfer/blob/release/0.6.18+amd.2/README.md) covers the test suite, the benchmark runner and the environment variables.

## Example: Serving a Batch of Chat Requests

An inference server's model runner is where the FlashInfer calls come together. Requests arrive with prompts of different lengths; the server batches them, runs one prefill over every prompt token to fill the KV cache, and then generates with a decode loop, one new token per request per step. The example below is that loop for a Llama-3.1-8B-shaped model. To run quickly on one GPU it keeps two of the 32 layers and uses random weights, so the shapes and the sequence of calls are the real model's, while the generated tokens are not meaningful. The five code blocks run in order as one script. Each layer keeps K and V in one tensor, which is fine in eager mode; under `torch.compile` they would need separate allocations.

No call passes `backend=`. Attention routes to AITER, while RMSNorm, RoPE, KV append, SiLU-and-mul and sampling run on the in-tree HIP kernels, as in the routing figure above.

### 1. The Model and Its Paged KV Cache

The shapes are those of Llama 3.1 8B: 32 query heads sharing 8 KV heads of dimension 128, a 14,336-wide MLP and a 128,256-token vocabulary. The KV cache is paged, 16 tokens per page, with one tensor per layer, and each request owns enough pages for its prompt plus the tokens it will generate.

```python
import torch
import flashinfer

# 4 chat requests on a Llama-3.1-8B-shaped model (2 of 32 layers, random weights).
torch.manual_seed(0)
dev, dt = "cuda", torch.bfloat16
hidden, inter, vocab, n_layers = 4096, 14336, 128256, 2
n_q, n_kv, hd = 32, 8, 128
page_size, new_tokens = 16, 32
prompt_lens = torch.tensor([300, 1200, 640, 2048], dtype=torch.int32)
batch = len(prompt_lens)

def weight(rows, cols):
    return torch.randn(rows, cols, dtype=dt, device=dev) / cols**0.5

ones = torch.ones(hidden, dtype=dt, device=dev)
embed, lm_head = weight(vocab, hidden), weight(vocab, hidden)
layers = [dict(qkv=weight((n_q + 2 * n_kv) * hd, hidden), o=weight(hidden, n_q * hd),
               gate_up=weight(2 * inter, hidden), down=weight(hidden, inter))
          for _ in range(n_layers)]

# Paged KV cache, one tensor per layer: each request owns the pages it will need.
pages = (prompt_lens + new_tokens + page_size - 1) // page_size
first_page = torch.cumsum(pages, 0) - pages
kv_caches = [torch.zeros(int(pages.sum()), 2, page_size, n_kv, hd, dtype=dt, device=dev)
             for _ in range(n_layers)]
```

### 2. Paged-KV Bookkeeping

The paged APIs describe the cache with three index tensors: where each request's pages start in a flat page list, the page IDs themselves, and how full each request's last page is. `kv_layout()` derives them from each request's current length, and is called again every step as the sequences grow.

```python
def kv_layout(seq_lens):
    """indptr / indices / last_page_len for the pages in use at these lengths."""
    used = (seq_lens + page_size - 1) // page_size
    indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(used, 0)])
    indices = torch.cat([torch.arange(f, f + u) for f, u in zip(first_page.tolist(), used.tolist())])
    last = (seq_lens - 1) % page_size + 1
    return [t.to(dev, torch.int32) for t in (indptr, indices, last)]
```

### 3. One Forward Pass

One function serves both phases. It takes the new tokens of every request packed together, plus an attention wrapper that has already been planned. `get_batch_indices_positions()` gives each new token its request and position, which RoPE and the KV append both need, and from the second norm on, `fused_add_rmsnorm` folds the residual add into the norm.

```python
def forward(tokens, qo_indptr, seq_lens, layout, attn):
    """All requests' new tokens, packed: [num_tokens] -> last-token logits per request."""
    kv_indptr, kv_indices, kv_last = layout
    batch_idx, pos = flashinfer.get_batch_indices_positions(
        qo_indptr, seq_lens.to(dev), tokens.numel())
    x, residual = embed[tokens], None
    for L, cache in zip(layers, kv_caches):
        if residual is None:
            residual, x = x, flashinfer.rmsnorm(x, ones)
        else:
            flashinfer.fused_add_rmsnorm(x, residual, ones)  # residual += x; x = norm(residual)
        q, k, v = (x @ L["qkv"].T).split([n_q * hd, n_kv * hd, n_kv * hd], dim=-1)
        q, k, v = (t.reshape(-1, h, hd).contiguous() for t, h in ((q, n_q), (k, n_kv), (v, n_kv)))
        flashinfer.apply_llama31_rope_pos_ids_inplace(q, k, pos)
        flashinfer.append_paged_kv_cache(k, v, batch_idx, pos, cache,
                                         kv_indices, kv_indptr, kv_last)
        x = attn.run(q, cache).reshape(-1, n_q * hd) @ L["o"].T
        flashinfer.fused_add_rmsnorm(x, residual, ones)
        x = flashinfer.silu_and_mul(x @ L["gate_up"].T) @ L["down"].T
    flashinfer.fused_add_rmsnorm(x, residual, ones)
    return x[qo_indptr[1:] - 1] @ lm_head.T

def sample(logits):
    return flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, top_k=50, top_p=0.9)
```

### 4. Prefill

Prefill runs every prompt token of every request through one causal attention call, writes their K and V into the cache, and samples each request's first output token from its last prompt position.

```python
workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

# Prefill: every prompt token of every request in one batched, causal call.
seq_lens = prompt_lens.clone()
qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(prompt_lens, 0)]).int().to(dev)
layout = kv_layout(seq_lens)
prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
prefill.plan(qo_indptr, *layout, n_q, n_kv, hd, page_size, causal=True, q_data_type=dt)
prompts = torch.randint(0, vocab, (int(prompt_lens.sum()),), device=dev)
next_tokens = sample(forward(prompts, qo_indptr, seq_lens, layout, prefill))
```

### 5. Decode Loop

Decode generates one token per request per step. `plan()` runs once per step and every layer reuses it; `run()` is the per-layer call.

```python
# Decode: one token per request per step; plan() once per step, shared by all layers.
decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
one_each = torch.arange(batch + 1, dtype=torch.int32, device=dev)
generated = [next_tokens]
for _ in range(new_tokens - 1):
    seq_lens += 1
    layout = kv_layout(seq_lens)
    decode.plan(*layout, n_q, n_kv, hd, page_size, q_data_type=dt)
    next_tokens = sample(forward(next_tokens, one_each, seq_lens, layout, decode))
    generated.append(next_tokens)

print(f"prefill ran on {prefill.backend}, decode on {decode.backend}")
print("generated token ids:", tuple(torch.stack(generated, 1).shape))  # (batch, new_tokens)
```

The first run compiles the kernels the image did not build ahead of time; later runs in the same container reuse them. On an MI355X it prints `prefill ran on aiter, decode on aiter` and `generated token ids: (4, 32)`.

## Summary

In this blog, you learned how FlashInfer on ROCm brings the upstream v0.6.18 API to ROCm 10.1.0 on AMD Instinct CDNA3 and CDNA4 GPUs, how `backend="auto"` routes each call between the in-tree HIP kernels and AITER, and how to build it and run a serving loop on it. The main points:

- **The FlashInfer API on AMD:** the decode, prefill, and MLA wrappers keep the upstream signatures.
- **`backend="auto"` routing:** each call goes to the in-tree HIP kernels or to AITER by compatibility and by per-architecture rules, and the wrapper reports which one it chose. RMSNorm, activations, RoPE, KV append and sampling run on the in-tree kernels by default.
- **FP8 and `torch.compile`:** FP8 KV caches and FP8 prefill are supported, and the ops in the table of the repository's `torch-compile.md` compile under `torch.compile` with `FLASHINFER_USE_TORCH_CUSTOM_OPS=1`.

## Acknowledgements

The authors wish to acknowledge the AMD teams that supported this work, whose contributions were instrumental in enabling FlashInfer: Aditya Bhattacharji, Pankaj Gupta, Anisha Sankar, Amit Kumar, Ram Seenivasan, Eliot Li, Ian Dass, Kiran Thumma, Aakash Sudhanwa, Ehud Sharlin, Saad Rahim, Lucia Cao, Jacky Zhao, Zhen Han, Junhua (Richard) Hou, Carlus Huang, Hai Xiao, Wenbo Shao, Marco Grond, Sriranjani Ramasubramanian.

<a id="references"></a>

## References

[1] Ye, Z., et al. (2025). FlashInfer: Efficient and customizable attention engine for LLM inference serving. [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)

[2] AMD CDNA 3 Architecture. [AMD CDNA 3 architecture white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)

[3] AMD CDNA 4 Architecture. [AMD CDNA 4 architecture white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)

[4] Pandey, S., et al. (2025). AITER: AI Tensor Engine for ROCm. [AITER: AI Tensor Engine for ROCm](https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html)

[5] Composable Kernel: programming model for performance-critical kernels for machine learning workloads on AMD GPUs. [Composable Kernel documentation](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/index.html)

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information.
However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.
THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
AMD, the AMD Arrow logo, AMD Instinct, ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
© 2026 Advanced Micro Devices, Inc. All rights reserved.

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
