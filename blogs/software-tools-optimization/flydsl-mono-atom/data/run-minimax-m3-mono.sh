#!/usr/bin/env bash
# MiniMax-M3 on four MI355X GPUs; ATOM with PR #2419 and its dependencies.
# Adapted from recipes/MiniMax-M3.md at 2b9f3412e466fd4172a4a621fe9215a1041130c1.
# Requires access to the target and draft checkpoints.
set -euo pipefail

export ATOM_MONO_ENABLE="${ATOM_MONO_ENABLE:-1}"
export ATOM_M3_INDEXER_CP="${ATOM_M3_INDEXER_CP:-1}"
export ATOM_FORCE_ATTN_TRITON=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0

python -m atom.entrypoints.openai_server \
  --model amd/MiniMax-M3-MXFP4 \
  --tensor-parallel-size 4 \
  --host 127.0.0.1 --server-port 8000 \
  --trust-remote-code \
  --kv_cache_dtype fp8 --index_cache_dtype fp8 \
  --block-size 128 --max-model-len 1048576 \
  --max-num-seqs 4 --max-num-batched-tokens 32768 \
  --attn-prefill-chunk-size 16384 \
  --gpu-memory-utilization 0.90 \
  --cudagraph-mode FULL \
  --hf-overrides '{"use_index_cache": false}' \
  --online_quant_config '{
    "global_quant_config": "ptpc_fp8",
    "exclude_layer": [
      "lm_head", "model.embed_tokens", "vision_tower",
      "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"
    ]
  }' \
  --method eagle3 \
  --draft-model Inferact/MiniMax-M3-EAGLE3 \
  --num-speculative-tokens 3
