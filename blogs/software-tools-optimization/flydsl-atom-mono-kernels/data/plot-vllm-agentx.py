"""Regenerate Figure 5 from mono-benefits.csv. Requires matplotlib."""

import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
SOURCE = "https://github.com/vllm-project/vllm/pull/60397"
with (ROOT / "mono-benefits.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

points = {}
for row in rows:
    if row["source"] != SOURCE or row["metric"] != "P90 interactivity":
        continue
    match = re.search(r"concurrency (\d+);", row["configuration"])
    if (
        match is None
        or row["unit"] != "tokens/s/user"
        or "TP2;" not in row["configuration"]
    ):
        raise ValueError("Expected TP2, request concurrency, and tokens/s/user")
    concurrency = int(match[1])
    if concurrency in points:
        raise ValueError(f"Duplicate interactivity for concurrency {concurrency}")
    values = (float(row["baseline_value"]), float(row["optimized_value"]))
    if not all(math.isfinite(value) and 0 < value < 470 for value in values):
        raise ValueError("Invalid interactivity or values outside the plot range")
    points[concurrency] = values
if set(points) != {1, 2, 4, 8}:
    raise ValueError("Expected the C1, C2, C4, and C8 mono-off/on comparisons")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, ax = plt.subplots(figsize=(10.5, 6.6), facecolor="#f8fafc")
fig.subplots_adjust(left=0.13, right=0.97, top=0.77, bottom=0.25)
ax.set_facecolor("#f8fafc")
colors = ("#94a3b8", "#dc4a48")
for y, (concurrency, (off, on)) in zip([6, 4, 2, 0], sorted(points.items())):
    ax.axhspan(y - 0.8, y + 0.8, color="#edf2f7", zorder=0)
    ax.barh(y + 0.3, off, height=0.48, color=colors[0])
    ax.barh(y - 0.3, on, height=0.48, color=colors[1])
    ax.text(off + 6, y + 0.3, f"{off:.1f}", va="center", fontsize=11)
    ax.text(
        on + 6,
        y - 0.3,
        f"{on:.1f} · {on / off:.2f}×",
        va="center",
        fontsize=11,
        weight="bold",
    )

ax.legend(
    handles=[
        Patch(color=color, label=label)
        for color, label in zip(
            colors, ("FlyDSL Mono off", "FlyDSL Mono on (30 layers)")
        )
    ],
    loc="upper left",
    bbox_to_anchor=(0, 1.17),
    frameon=False,
    fontsize=10,
)
ax.set_xlim(0, 550)
ax.set_ylim(-1, 7)
ax.set_yticks([6, 4, 2, 0], ["C1", "C2", "C4", "C8"])
ax.set_xlabel("P90 interactivity (tokens/s/user) — higher is better", labelpad=12)
ax.set_ylabel("Request concurrency", labelpad=10)
ax.grid(axis="x", color="#dce2e8", linewidth=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0, pad=8)
fig.text(
    0.13, 0.94, "Higher interactivity with FlyDSL Mono", fontsize=20, weight="bold"
)
fig.text(0.13, 0.885, "DeepSeek-V4.1-Flash in vLLM · MI355X · TP2", fontsize=13)
fig.text(
    0.13,
    0.13,
    "Each group pairs FlyDSL Mono off and on at the same concurrency; labels show values and ratios.",
    fontsize=10,
)
fig.text(
    0.13,
    0.09,
    "One 3,600 s run per mode and concurrency · synthetic DSpark acceptance length 3.51",
    fontsize=10,
)
fig.text(
    0.13,
    0.05,
    "Interactivity = 1 / P90 inter-token latency. TTFT is reported separately. Source: vLLM #60397.",
    fontsize=10,
)
fig.savefig(ROOT.parent / "images" / "vllm-dsv41-interactivity.png", dpi=160)
plt.close(fig)
