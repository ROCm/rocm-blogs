---
blogpost: true
blog_title: "FlashInfer on ROCm: High‑Throughput Prefill Attention via AITER"
date: 06 Apr 2026
author: 'Debasis Mandal, Diptorup Deb, Rishi Madduri, Anuya Welling, Mukhil Azhagan Mallaiyan Sathiaseelan, Yao Liu, Phani Vaddadi, Vish Vadlamani'
thumbnail: 'flashinfer-thumbnail2.jpg'
tags: AI/ML, GenAI, Optimization, Serving
category: Applications & models
target_audience: AI practitioners
key_value_propositions: Adds support for features like prefill, sampling, logits processor, and AITER backend. Based on upstream version 0.5.3 and supports gfx950.
language: English
myst:
    html_meta:
        "author": "Debasis Mandal, Diptorup Deb, Rishi Madduri, Anuya Welling, Mukhil Azhagan Mallaiyan Sathiaseelan, Yao Liu, Phani Vaddadi, Vish Vadlamani"
        "description lang=en": "FlashInfer is an open-source library for accelerating LLM serving that is now supported by ROCm."
        "keywords": "Flashinfer, transformer, inference, prefill, decode, flash attention, LLM, serving"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software"
        "amd_blog_applications": "AI Inference, Generative AI, Data Science"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Debasis Mandal, Diptorup Deb, Rishi Madduri, Anuya Welling, Mukhil Azhagan Mallaiyan Sathiaseelan, Yao Liu, Phani Vaddadi, Vish Vadlamani"
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

# FlashInfer on ROCm: High‑Throughput Prefill Attention via AITER

The explosive growth of large language models (LLMs) like DeepSeek-R1, Llama 3, and Qwen 3 has created an urgent need for efficient inference solutions. As these models scale to billions of parameters and context lengths extend to hundreds of thousands of tokens, the attention mechanism becomes a critical bottleneck, consuming substantial memory for key-value (KV) caches and requiring significant compute for each token generated.

FlashInfer<sup><a href="#references">1</a></sup> addresses these challenges with a high-performance library that optimizes the attention computation at the heart of transformer inference. Originally developed for NVIDIA GPUs, [FlashInfer on ROCm](https://github.com/ROCm/flashinfer) brings accelerated LLM serving to AMD Instinct GPUs. Building on the [earlier release of decode kernels](https://rocm.blogs.amd.com/artificial-intelligence/flashinfer/README.html), this release adds prefill kernels to complete the core attention operations needed for production LLM serving.

This blog covers the porting effort to AMD's modern CDNA architectures (CDNA3<sup><a href="#references">2</a></sup> and CDNA4<sup><a href="#references">3</a></sup>) and demonstrates how to use FlashInfer<sup><a href="#references">1</a></sup> for prefill operations on AMD hardware.

## What is FlashInfer?

FlashInfer is a kernel library specifically designed for LLM serving workloads. Unlike general-purpose attention implementations, FlashInfer optimizes for the unique characteristics of inference:

- **Prefill phase:** Processing the initial prompt requires computing attention over the entire input sequence, a *compute-intensive* operation that benefits from *high throughput*.
- **Decode phase:** Generating tokens one-by-one requires *low-latency* attention against a growing KV cache, a *memory-bound* operation that benefits from efficient memory access patterns.

FlashInfer provides specialized kernels for each phase:

| Capability | Description |
| ---------- | ----------- |
| **Paged KV-Cache** | Manages memory like virtual memory pages, eliminating fragmentation and enabling efficient memory utilization across variable-length sequences |
| **Grouped Query Attention (GQA)** | Native support for Grouped Query Attention (GQA) and Multi-Query Attention (MQA), reducing KV cache memory requirements |
| **Ragged Tensors** | Efficiently handles batches of sequences with different lengths without padding overhead |

## FlashInfer on ROCm

- [Installation guide](https://rocm.docs.amd.com/projects/flashinfer/en/docs-26.03/install/flashinfer-install.html)
- [Docker images](https://hub.docker.com/r/rocm/flashinfer/tags)
- [GitHub repository](https://github.com/ROCm/flashinfer)

The ROCm port of FlashInfer brings the above optimizations to AMD Instinct GPUs. This release updates FlashInfer on ROCm from version 0.2.5 to 0.5.3, adding the FlashAttention-2 based prefill kernels, including single-request, batched, and ragged variants, to AMD's CDNA3 and CDNA4 architectures, complementing the [decode kernels ported in the earlier release](https://rocm.blogs.amd.com/artificial-intelligence/flashinfer/README.html).

### Feature Support Matrix

| Kernel Type | FP16 / BF16 | FP8 | Notes |
| ----------- | :---------: | :-: | ----- |
| Decode Attention | ✅ | ✅ | Supports MHA, GQA, and MQA |
| Prefill Attention | ✅ | WIP | Supports MHA, GQA, and MQA |

Additional features from upstream FlashInfer, including FP8 support for prefill, cascade attention, Multi-Head Latent Attention (MLA), and positional encodings (RoPE/ALiBi), are actively under development. Refer to the Feature Support Matrix in the `README` of the [GitHub repository](https://github.com/ROCm/flashinfer) for the latest status.

### Porting to Modern CDNA Architectures

Adapting FlashInfer to AMD GPUs required fundamental changes to its kernel architecture. The port involved restructuring four core computational stages: loading query matrices into shared memory, streaming key/value data, computing query-key dot products, and performing the softmax-value multiplication.

The primary changes centered on replacing NVIDIA's warp matrix operations (wmma) with CDNA3/CDNA4 Matrix Fused Multiply-Add (MFMA) instructions. This required:

- Restructuring thread layouts from 32-thread warps to 64-thread wavefronts
- Modifying shared memory access patterns to accommodate different bank conflict and coalescing requirements
- Updating indexing logic to align with MFMA's 16×16 matrix tile geometry

These optimizations enable FlashAttention-2's online softmax algorithm and block-sparse attention patterns to run efficiently on modern AMD CDNA GPUs.

### Experimental AITER Backend Support

This release of FlashInfer on ROCm also provides experimental support for using AITER<sup><a href="#references">4</a></sup> as a backend, in addition to the above HIP<sup><a href="#references">5</a></sup>-based implementation. The AITER backend is currently enabled for the `single_prefill` and `batch_prefill` kernels.

#### Known Limitations

The AITER backend supports the `NHD` kv_layout; other kv_layout values are unsupported. Additionally, when using CK<sup><a href="#references">6</a></sup> (Composable Kernel) FMHA kernels for AITER Multi-Head Attention (MHA), batch prefill is limited to page sizes 1, 16, and 1024.

## Getting Started

Please look at the [installation guide](https://rocm.docs.amd.com/projects/flashinfer/en/docs-26.03/install/flashinfer-install.html) for detailed instructions on how to install FlashInfer on ROCm.

The quickest way to get started is by using a prebuilt Docker image that includes FlashInfer, PyTorch, AITER, and all other dependencies.

First, ensure that the following requirements are met:

- **Linux:** Ubuntu 24.04 (see [supported distributions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems))
- **Hardware**: AMD Instinct MI300X or MI325X (gfx942 / CDNA3 architecture), or MI355X (gfx950 / CDNA4 architecture)

Next, pull the FlashInfer image from Docker Hub.

```bash
docker pull rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1
```

Finally, start a container with GPU access.

```bash
docker run -it --rm \
--privileged \
--network=host --device=/dev/kfd \
--device=/dev/dri --group-add video \
--name=my_flashinfer --cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--ipc=host --shm-size 16G \
rocm/flashinfer:flashinfer-0.5.3.amd1_rocm7.2_ubuntu24.04_py3.12_pytorch2.9.1
```

[Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) is pre-configured inside the container and will automatically start the `base` environment.

Verify the installation by running the following command. It should output `0.5.3+amd.1`.

```bash
python -c "import flashinfer; print(flashinfer.__version__)"
```

To see FlashInfer in action, save the following examples to a file `prefill.py` and run `python prefill.py`.

### Single-request Prefill Attention with AITER Backend

The prefill phase processes the initial prompt by computing attention over the entire input sequence. This is compute-bound and benefits from high-throughput kernels. To use AITER as the backend for these kernels, set `backend="aiter"` as a keyword argument when invoking the kernels, as can be seen below.

```python
import torch
import flashinfer

# Configuration
seq_len = 1024        # Prompt length
num_qo_heads = 32     # Number of query/output heads
num_kv_heads = 8      # Number of KV heads (GQA with 4:1 ratio)
head_dim = 128

# Create Q, K, V tensors (NHD layout: sequence, heads, dimension)
q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda")
k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")

# Run single prefill attention with causal masking
output = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True,  backend="aiter")
```

### Batched Prefill with Paged KV Cache

For production serving with multiple concurrent requests of various sequence lengths, FlashInfer supports batched prefill with paged KV caches. This enables efficient memory management across requests with varying context lengths, similar to virtual memory paging.

```python
import torch
import flashinfer

batch_size = 4
seq_len = 512         # Sequence length per request
page_size = 16        # KV cache page size
num_qo_heads = 32     # Number of query/output heads
num_kv_heads = 8      # Number of KV heads (GQA)
head_dim = 128

# Calculate paging parameters
num_pages_per_seq = (seq_len + page_size - 1) // page_size
total_pages = num_pages_per_seq * batch_size

# Flattened query tensor: all sequences concatenated
q = torch.randn(batch_size * seq_len, num_qo_heads, head_dim, dtype=torch.float16, device="cuda")

# Paged KV cache: [total_pages, 2, page_size, num_kv_heads, head_dim]
kv_data = torch.randn(total_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")

# Index pointers and page table metadata
q_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * seq_len
kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * num_pages_per_seq
kv_indices = torch.arange(0, total_pages, dtype=torch.int32, device="cuda")
kv_last_page_len = torch.full((batch_size,), (seq_len - 1) % page_size + 1, dtype=torch.int32, device="cuda")

# Create workspace and wrapper
workspace = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device="cuda")
wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="aiter")

# Plan the batched operation
wrapper.plan(
    q_indptr, kv_indptr, kv_indices, kv_last_page_len,
    num_qo_heads, num_kv_heads, head_dim, page_size,
    causal=True
)

# Execute batched prefill
output = wrapper.run(q, kv_data)
```

For more examples, including batched prefill with ragged tensors, see the [examples directory](https://github.com/ROCm/flashinfer/tree/amd-integration/examples) in the FlashInfer repository.

## Summary

FlashInfer on ROCm brings high-performance LLM inference to AMD Instinct GPUs. With optimized kernels for both prefill and decode phases, support for modern attention variants like GQA and MQA, and efficient paged KV cache management, FlashInfer enables production-grade LLM serving on AMD hardware.

Key highlights of this release:

- **Prefill kernels:** for single-request, batched, and ragged attention patterns
- **Decode kernels:** with FP8 support for memory-efficient token generation
- **Paged KV cache:** support for efficient memory management in serving scenarios
- **Native GQA/MQA support:** for modern model architectures

Development continues with cascade attention, positional encodings, and sampling kernels on the roadmap. We encourage you to try FlashInfer on ROCm and share your feedback through the [GitHub repository](https://github.com/ROCm/flashinfer/issues).

## Acknowledgements

The authors wish to acknowledge the AMD teams that supported this work, whose contributions were instrumental in enabling FlashInfer: Aditya Bhattacharji, Pankaj Gupta, Radha Srimanthula, Anisha Sankar, Amit Kumar, Ram Seenivasan, Eliot Li, Ian Dass, Kiran Thumma, Aakash Sudhanwa, Ehud Sharlin, Saad Rahim, Lucia Cao, Jacky Zhao, Zhen Han, Junhua (Richard) Hou, Lin Sun, Carlus Huang, Hai Xiao.

<a id="references"></a>

## References

[1] Ye, Z., et al. (2025). FlashInfer: Efficient and customizable attention engine for LLM inference serving. [arXiv:2501.01005](https://arxiv.org/abs/2501.01005)

[2] AMD CDNA 3 Architecture. [White paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)

[3] AMD CDNA 4 Architecture. [White paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)

[4] Pandey, S., et al. (2025). AITER: AI Tensor Engine for ROCm. [ROCm technical blog](https://rocm.blogs.amd.com/software-tools-optimization/aiter-ai-tensor-engine/README.html)

[5] HIP: C++ runtime API and kernel language for AMD GPUs (Heterogeneous-computing Interface for Portability). [ROCm documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html)

[6] Composable Kernel — programming model for performance-critical kernels for machine learning workloads on AMD GPUs. [ROCm documentation](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/index.html)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the
content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS
PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT
IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO
YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE
FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.
