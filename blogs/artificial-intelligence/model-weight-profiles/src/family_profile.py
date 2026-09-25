# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Written with AI assistance.
#
# Blog checkpoints this script is meant to count:
#   google-bert/bert-base-uncased
#   unsloth/Meta-Llama-3.1-8B
#   unsloth/Meta-Llama-3.1-70B-Instruct
#   amd/Llama-3.1-8B-Instruct-FP8-KV
#   amd/Llama-3.1-70B-Instruct-FP8-KV
#   deepseek-ai/DeepSeek-R1
#   Qwen/Qwen3-VL-30B-A3B-Instruct
#   Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
# Unsloth redistributes the Llama 3.1 weights the blog figures use.
# The matching meta-llama/Llama-3.1-* repos are gated, so the example stays on Unsloth.

import sys
from collections import defaultdict

from fetch_hub_metadata import tensor_rows

DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1}


def dtype_bytes(dtype):
    d = dtype.upper().replace("-", "_")
    if "E5M2" in d:
        return 1
    if "E4M3" in d or d in ("F8", "FP8"):
        return 1
    return DTYPE_BYTES[d]


def family(name):
    name = name.lower()
    if "scale" in name:
        return "Other"
    if any(x in name for x in ("word_embeddings", "position_embeddings", "token_type_embeddings", "embed_tokens", "lm_head")):
        return "Embedding / LM Head"
    if any(x in name for x in ("attention.", "self_attn.", ".attn.")):
        return "Attention"
    if any(x in name for x in ("intermediate.dense.weight", ".output.dense.weight", ".output.layernorm", "pooler.dense", "cls.predictions.transform.dense", "cls.seq_relationship", "gate_proj", "gate_up_proj", "up_proj", "down_proj", "linear_fc", "merger")):
        return "Dense"
    return "Other"


model_id = sys.argv[1] if len(sys.argv) > 1 else "google-bert/bert-base-uncased"
totals = defaultdict(lambda: [0, 0])
for name, shape, dtype, parameters in tensor_rows(model_id):
    key = family(name), dtype
    totals[key][0] += parameters
    totals[key][1] += parameters * dtype_bytes(dtype)

print("| Family | Dtype | Parameters | Size |")
print("| --- | --- | ---: | ---: |")
for (name, dtype), (parameters, size) in sorted(totals.items(), key=lambda item: -item[1][0]):
    print(f"| {name} | {dtype} | {parameters / 1e6:.1f}M | {size / 1e6:.1f} MB |")
