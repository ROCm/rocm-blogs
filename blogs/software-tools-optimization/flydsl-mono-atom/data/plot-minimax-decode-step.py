"""Regenerate Figure 3 from minimax-decode-step.csv. Requires matplotlib."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
with (ROOT / "minimax-decode-step.csv").open(newline="") as stream:
    rows = sorted(csv.DictReader(stream), key=lambda row: int(row["request_batch"]))
if [row["request_batch"] for row in rows] != ["1", "2", "4"]:
    raise ValueError("Expected one measurement each for batch sizes 1, 2, and 4")
keys = (
    "atom_separate_kernels_ms",
    "atom_flydsl_mono_ms",
    "atom_flydsl_mono_cp_ms",
)
values = [[float(row[key]) for key in keys] for row in rows]
if not all(math.isfinite(value) and value > 0 for group in values for value in group):
    raise ValueError("Latencies must be finite and positive")
conditions = (
    "hardware",
    "tensor_parallel_size",
    "context",
    "speculation",
    "draft_tokens",
)
if len({tuple(row[key] for key in conditions) for row in rows}) != 1:
    raise ValueError("All batch comparisons must use the same benchmark configuration")
labels = ["Separate kernels", "FlyDSL Mono", "FlyDSL Mono + indexer CP"]
colors = ["#94a3b8", "#dc4a48", "#146184"]
centers = [8, 4, 0]
offsets = [0.72, 0, -0.72]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "text.color": "#152a3c",
        "axes.labelcolor": "#465a6b",
        "savefig.facecolor": "white",
    }
)
fig, ax = plt.subplots(figsize=(10.8, 6.7))
fig.text(
    0.045, 0.93, "FlyDSL Mono: M3 decode latency in ATOM", fontsize=23, weight="bold"
)
fig.text(
    0.045,
    0.872,
    "MiniMax-M3 · MI355X · TP4 · 1M context · MXFP4 · EAGLE3 (3 tokens)",
    fontsize=11,
    color="#586b7b",
)
for group_index, (center, group) in enumerate(zip(centers, values)):
    ax.axhspan(center - 1.14, center + 1.14, color="#f7f9fc", zorder=0)
    for series, (offset, value, color) in enumerate(zip(offsets, group, colors)):
        y = center + offset
        ax.barh(
            y,
            value,
            height=0.53,
            color=color,
            label=labels[series] if group_index == 0 else None,
        )
        label = f"{value:.2f} ms"
        if series:
            label += f" · {group[0] / value:.2f}×"
        ax.text(value + 0.22, y, label, va="center", fontsize=11, weight="bold")
fig.legend(loc="upper left", bbox_to_anchor=(0.11, 0.832), ncol=3, frameon=False)
ax.set_yticks(centers, [f"Batch {row['request_batch']}" for row in rows], fontsize=12)
ax.tick_params(axis="y", length=0, pad=10)
ax.tick_params(axis="x", length=0, labelsize=10, colors="#64748b")
ax.set_xlim(0, max(max(group) for group in values) * 1.32)
ax.set_ylim(-1.45, 9.45)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_xlabel(
    "Median GPU decode-step latency (ms) · lower is better", fontsize=11, labelpad=10
)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#e5ebf0", linewidth=0.8)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.text(
    0.045,
    0.07,
    "Each group uses the separate-kernel result at the same batch size as its baseline.",
    fontsize=10,
    color="#586b7b",
)
fig.text(
    0.045,
    0.032,
    "Source: ATOM PR #2419. CP adds a long-context indexer optimization to FlyDSL Mono.",
    fontsize=9,
    color="#586b7b",
)
fig.subplots_adjust(left=0.12, right=0.97, top=0.75, bottom=0.18)
fig.savefig(ROOT.parent / "images" / "minimax-flydsl-decode-step.png", dpi=200)
plt.close(fig)
