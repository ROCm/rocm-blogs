"""Regenerate Figure 3 from minimax-decode-step.csv. Requires matplotlib."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
with (ROOT / "minimax-decode-step.csv").open(newline="") as stream:
    rows = [row for row in csv.DictReader(stream) if row["request_batch"] == "1"]
if len(rows) != 1:
    raise ValueError("Expected exactly one batch-one measurement")
row = rows[0]
values = [
    float(row[key])
    for key in (
        "atom_separate_kernels_ms",
        "atom_flydsl_mono_ms",
        "atom_flydsl_mono_cp_ms",
    )
]
if not all(math.isfinite(value) and value > 0 for value in values):
    raise ValueError("Latencies must be finite and positive")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "text.color": "#152a3c",
        "axes.labelcolor": "#465a6b",
        "savefig.facecolor": "white",
    }
)
fig, ax = plt.subplots(figsize=(10.4, 4.15))
fig.text(
    0.045, 0.9, "M3 decode in ATOM: mono, then indexer CP", fontsize=22, weight="bold"
)
fig.text(
    0.045,
    0.822,
    "MiniMax-M3 · MI355X · TP4 · batch 1 · 1M context · MXFP4 · EAGLE3 (3 tokens)",
    fontsize=10.4,
    color="#586b7b",
)
labels = ["ATOM separate kernels", "FlyDSL mono", "FlyDSL mono + indexer CP"]
ax.barh([2, 1, 0], values, height=0.53, color=["#94a3b8", "#dc4a48", "#146184"])
for index, (y, value) in enumerate(zip([2, 1, 0], values)):
    ax.text(value + 0.18, y, f"{value:.2f} ms", va="center", fontsize=13, weight="bold")
    if index:
        ax.text(
            0.27,
            y,
            f"{values[0] / value:.2f}×",
            va="center",
            fontsize=13,
            color="white",
            weight="bold",
        )
ax.set_yticks([2, 1, 0], labels, fontsize=12)
ax.tick_params(axis="y", length=0, pad=12)
ax.tick_params(axis="x", length=0, labelsize=10, colors="#64748b")
ax.set_xlim(0, round(max(values) * 1.17, 1))
ax.set_xticks([0, 3, 6, 9, 12])
ax.set_xlabel(
    "Median GPU decode-step latency (ms) · lower is better", fontsize=10, labelpad=10
)
ax.set_axisbelow(True)
ax.grid(axis="x", color="#e5ebf0", linewidth=0.8)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.text(
    0.045,
    0.033,
    "Source: ATOM PR #2419. CP adds a long-context indexer optimization to mono.",
    fontsize=9,
    color="#586b7b",
)
fig.subplots_adjust(left=0.31, right=0.96, top=0.735, bottom=0.22)
fig.savefig(ROOT.parent / "images" / "minimax-flydsl-decode-step.png", dpi=200)
plt.close(fig)
