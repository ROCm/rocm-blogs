# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Written with AI assistance.

import colorsys
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import squarify
import yaml

HUES = {"embedding": 0.98, "attention": 0.58, "dense": 0.33}
NAMES = {"embedding": "Embedding / LM Head", "attention": "Attention", "dense": "Dense"}


def color(family, dtype):
    if dtype == "F32":
        saturation, value = 0.78, 0.52
    elif dtype in ("BF16", "F16"):
        saturation, value = {
            "embedding": (0.67, 0.72),
            "attention": (0.64, 0.76),
            "dense": (0.58, 0.84),
        }[family]
    else:
        saturation, value = 0.38, 0.97
    return colorsys.hsv_to_rgb(HUES[family], saturation, value)

profile_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("bert_base_uncased.profile.yaml")
output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("profile.png")
profile = yaml.safe_load(profile_path.read_text())

tiles = []
for family, values in profile["stored_params"]["families"].items():
    if family not in HUES:
        continue
    by_dtype = defaultdict(int)
    for group in values["param_groups"].values():
        by_dtype[group.get("dtype", "")] += group.get("bytes", 0)
    for dtype, size in by_dtype.items():
        label_dtype = "FP32" if dtype == "F32" else dtype
        tiles.append((size, f"{NAMES[family]}\n{size / 1e6:.1f} MB {label_dtype}", color(family, dtype)))

tiles.sort(reverse=True)
fig, ax = plt.subplots(figsize=(7, 5))
squarify.plot(sizes=[x[0] for x in tiles], label=[x[1] for x in tiles], color=[x[2] for x in tiles], ax=ax)
ax.axis("off")
ax.set_title(profile["model_id"])
fig.tight_layout()
fig.savefig(output_path, dpi=180)
print(output_path)
