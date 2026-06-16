# Technical Deep-Dive into AMD MLPerf Training v6.0 Optimizations

## The MXFP4 Training Recipe

The MXFP4 (Microscaling FP4) training recipe, implemented in the [ROCm Transformer Engine](https://github.com/ROCm/TransformerEngine/tree/mpo), leverages the native FP4 support on AMD Instinct MI355X GPUs to push training throughput beyond what FP8 alone can deliver. MXFP4 uses the OCP Microscaling format with E2M1 (2-bit exponent, 1-bit mantissa) data values and E8M0 (8-bit exponent) per-block scaling factors, where each block of 32 contiguous elements shares a single E8M0 scale — providing a fine-grained balance between compression and representational fidelity.

The recipe is implemented as a first-class precision recipe in Transformer Engine, configuring the quantization pipeline for all linear layers in the transformer stack. The pipeline produces packed FP4 data (two E2M1 values per byte) alongside E8M0 scales in both rowwise and columnwise layouts to serve the forward, Dgrad, and Wgrad GEMMs. A deterministic 16-point Hadamard rotation is applied before quantization to disperse outlier energy across dimensions, reducing quantization distortion without altering the underlying computation — the orthogonality of the transform ensures exact cancellation across the forward-backward cycle. This deterministic approach proved essential: experiments showed that Wgrad quantization dominates training instability, and stochastic rounding or randomized Hadamard transforms fail to stabilize the full pipeline.

On the kernel side, the entire quantization pipeline — Hadamard rotation, scale computation, FP4 casting, byte packing, data shuffling, and scale swizzling — is fused into a single HIP kernel launch, eliminating intermediate memory round-trips. The resulting MXFP4 tensors are dispatched to AITER's hand-tuned ASM A4W4 GEMM kernels, with per-model tuned configurations loaded from pre-computed CSV files to ensure optimal tile sizes and instruction scheduling for every matrix shape encountered during training.

## Recipe-Specific Implementations: Llama2 70B and Llama3.1 8B

While both models use the same MXFP4 quantization infrastructure in Transformer Engine, their training recipes differ significantly due to the memory pressure of the 70B model.

Llama3.1 8B pretraining is the simpler case: the model trains end-to-end in MXFP4 with deterministic Hadamard enabled, without any precision transitions mid-training. A warmup phase using FP8 hybrid precision JIT-compiles all kernels before the measured MXFP4 training begins, after which the model state is fully reset for a clean start. The transpose cache is enabled for this model to keep a precomputed weight transpose alongside the primary quantized data, avoiding recomputation during backward GEMMs.

The Llama2 70B LoRA finetuning recipe is fundamentally more complex. FP4 requires both rowwise and columnwise quantized data (dual paths), consuming more VRAM than FP8's single-path approach, making a pure end-to-end MXFP4 run infeasible at 70B scale. Towards this end, a phased strategy is used in which after warmup, weights are pre-quantized to MXFP4 while FP8 copies are stashed on CPU memory. MXFP4 training proceeds until, after reaching a pivotal stage, a healing transition restores FP8 weights from the CPU stash, switches the active recipe to FP8 DelayedScaling (E4M3 forward, E5M2 backward), and disables activation recomputation for the remaining steps to convergence.

## Optimizations

During MXFP4 training, all three linear GEMMs (forward, Dgrad, Wgrad) use E2M1 precision uniformly — input activations and gradients are quantized through the fused HIP kernel, while weights use a cached MXFP4 representation that persists across forward calls to avoid redundant quantization. For LoRA finetuning where base weights are frozen, columnwise quantization is skipped entirely since no Wgrad is needed. Beyond the MXFP4 GEMM pipeline, both models benefit from CK-based Flash Attention v3 kernels via AITER for forward and backward attention, AITER's optimized ASM RoPE kernels replacing TE's native implementation, and fused SwiGLU activations through Transformer Engine.

The healing transition in Llama2 70B is carefully sequenced to avoid memory spikes: before training begins, a pre-quantization step clones all linear weights into FP8 format and pins them to CPU memory, then replaces the live GPU weights with MXFP4 tensors. At healing time, a two-pass design first unlinks all MXFP4 weights (freeing GPU memory), then transfers FP8 data from CPU to GPU via a dedicated CUDA stream — preventing both weight sets from being GPU-resident simultaneously. The precision recipe is patched from MXFP4BlockScaling to DelayedScaling, activation recomputation is disabled, and all FP4 metadata is cleared so that TE re-initializes under the new recipe.

Beyond the LLM workloads, the FLUX.1 Schnell image generation model benefits from a set of targeted optimizations. Compiling the double- and single-block stacks as whole regions with torch.compile let the compiler fuse across blocks into fewer, larger kernels, yielding a 4.1% improvement. Separately, capturing the full iteration as a local CUDA graph and replaying it removed per-kernel CPU launch latency for the many small DiT kernels for an additional 5.3% improvement. Replacing the default Triton-based RMSNorm with custom HIP kernels for both the forward and backward passes delivered an overall 28% throughput improvement, with further kernel tuning contributing another 3% gain. Finally, tuning the FP8 cast/transpose path to use the fastest available kernel—via the custom implementation in ROCm Transformer Engine—added a further 0.5% improvement.

## References

1. Cim et al., ["Pretraining Large Language Models with MXFP4 on Native FP4 Hardware"](https://arxiv.org/abs/2605.09825), arXiv:2605.09825, 2026
2. [Technical Dive into AMD MLPerf Training v5.1 Submission](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-training-v5.1/README.html)
3. [ROCm/TransformerEngine `mpo` branch](https://github.com/ROCm/TransformerEngine/tree/mpo)
4. [AMD CDNA 4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
5. OCP Microscaling Formats (MX) v1.0 Specification

