# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Written with AI assistance.

import math
import os
import sys
from pathlib import Path

from huggingface_hub import get_safetensors_metadata

ca = Path.home() / ".config/amd-ca-bundle.pem"
if ca.exists():
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(ca))


def tensor_rows(model_id):
    metadata = get_safetensors_metadata(model_id)
    for shard in metadata.files_metadata.values():
        for name, tensor in shard.tensors.items():
            yield name, tensor.shape, str(tensor.dtype), math.prod(tensor.shape)


if __name__ == "__main__":
    # Default is BERT. Llama 3.1 figures use unsloth/Meta-Llama-3.1-8B (and the 70B Instruct twin)
    # so the example does not need access to the gated meta-llama/Llama-3.1-* repos.
    model_id = sys.argv[1] if len(sys.argv) > 1 else "google-bert/bert-base-uncased"
    rows = list(tensor_rows(model_id))
    for name, shape, dtype, parameters in rows[:10]:
        print(f"{name:65} {str(shape):20} {dtype:5} {parameters:,}")
    print(f"{len(rows)} tensors, {sum(row[3] for row in rows):,} parameters")
