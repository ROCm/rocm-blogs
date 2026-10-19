"""Regenerate Figure 4 from mono-benefits.csv. Requires matplotlib."""

import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = "https://github.com/vllm-project/vllm/pull/60397"
with (ROOT / "mono-benefits.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

points = {}
for row in rows:
    if row["source"] != SOURCE or row["metric"] != "P90 interactivity":
        continue
    match = re.search(r"concurrency (\d+);", row["configuration"])
    if match is None or row["unit"] != "tokens/s/user":
        raise ValueError("Expected request concurrency and tokens/s/user")
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
fig.subplots_adjust(left=0.13, right=0.96, top=0.77, bottom=0.30)
ax.set_facecolor("#f8fafc")
colors = ("#526274", "#d64440")
for y, (concurrency, (off, on)) in zip(range(3, -1, -1), sorted(points.items())):
    ax.annotate(
        "",
        xy=(on, y),
        xytext=(off, y),
        arrowprops={"arrowstyle": "->", "color": "#a3adb9", "lw": 1.7},
        zorder=1,
    )
    ax.scatter([off], [y], s=64, color=colors[0], zorder=3)
    ax.scatter([on], [y], s=64, color=colors[1], zorder=3)
    ax.annotate(
        f"{off:.1f}",
        (off, y),
        xytext=(0, -19),
        textcoords="offset points",
        ha="center",
        color=colors[0],
        fontsize=11,
    )
    ax.annotate(
        f"{on:.1f}",
        (on, y),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        color=colors[1],
        fontsize=11,
    )
    ax.text(510, y, f"{on / off:.2f}×", ha="right", va="center", weight="bold")

for color, label in zip(colors, ("Mono off", "FlyDSL mono: K1 + K2 on 30 layers")):
    ax.scatter([], [], s=64, color=color, label=label)
ax.legend(loc="upper left", bbox_to_anchor=(0, 1.19), frameon=False, fontsize=10)
ax.set_xlim(0, 520)
ax.set_ylim(-0.6, 3.55)
ax.set_yticks([3, 2, 1, 0], ["C1", "C2", "C4", "C8"])
ax.set_xlabel("P90 interactivity (tokens/s/user) — higher is better", labelpad=12)
ax.set_ylabel("Request concurrency", labelpad=10)
ax.grid(axis="x", color="#dce2e8", linewidth=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0, pad=8)
fig.text(
    0.13, 0.94, "Higher interactivity with mono decode", fontsize=20, weight="bold"
)
fig.text(0.13, 0.885, "DeepSeek-V4.1-Flash in vLLM · MI355X · TP2", fontsize=13)
fig.text(
    0.13,
    0.15,
    "Arrows pair mono off and on at each concurrency; labels on the right show the interactivity ratio.",
    fontsize=10,
)
fig.text(
    0.13,
    0.11,
    "One 3,600 s run per mode and concurrency · synthetic DSpark acceptance 3.51",
    fontsize=10,
)
fig.text(
    0.13,
    0.07,
    "Interactivity = 1 / P90 inter-token latency. TTFT is reported separately. Source: vLLM #60397.",
    fontsize=10,
)
fig.savefig(ROOT.parent / "images" / "vllm-dsv41-interactivity.png", dpi=160)
plt.close(fig)
