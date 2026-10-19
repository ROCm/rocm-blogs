"""Regenerate Figure 7 from matched M3 and V4.1 comparisons. Requires matplotlib."""

import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
with (ROOT / "minimax-decode-step.csv").open(newline="") as stream:
    m3 = sorted(csv.DictReader(stream), key=lambda row: int(row["request_batch"]))
if [row["request_batch"] for row in m3] != ["1", "2", "4"]:
    raise ValueError("Expected M3 batches 1, 2, and 4")
m3_ratios = []
for row in m3:
    separate, mono, cp = (
        float(row[key])
        for key in (
            "atom_separate_kernels_ms",
            "atom_flydsl_mono_ms",
            "atom_flydsl_mono_cp_ms",
        )
    )
    if not all(math.isfinite(value) and value > 0 for value in (separate, mono, cp)):
        raise ValueError("M3 latencies must be finite and positive")
    m3_ratios.append((1.0, separate / mono, separate / cp))

v41 = {}
with (ROOT / "mono-benefits.csv").open(newline="") as stream:
    for row in csv.DictReader(stream):
        if (
            row["source"] != "https://github.com/vllm-project/vllm/pull/60397"
            or row["metric"] != "P90 interactivity"
        ):
            continue
        match = re.search(r"concurrency (\d+);", row["configuration"])
        if match is None or "TP2;" not in row["configuration"]:
            raise ValueError("Expected TP2 with an explicit concurrency")
        concurrency = int(match[1])
        if concurrency in v41:
            raise ValueError("Duplicate V4.1 concurrency")
        off, on = float(row["baseline_value"]), float(row["optimized_value"])
        if not all(math.isfinite(value) and value > 0 for value in (off, on)):
            raise ValueError("Interactivity values must be finite and positive")
        v41[concurrency] = (1.0, on / off)
if set(v41) != {1, 2, 4, 8}:
    raise ValueError("Expected V4.1 concurrency 1, 2, 4, and 8")

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#152a3c"})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), sharey=True)
colors = ["#94a3b8", "#dc4a48", "#146184"]
labels = ["Non-mono baseline", "FlyDSL mono", "FlyDSL mono + indexer CP (M3)"]
fig.text(
    0.05, 0.94, "Mono gains against a matched baseline", fontsize=23, weight="bold"
)
fig.legend(
    handles=[Patch(color=color, label=label) for color, label in zip(colors, labels)],
    loc="upper left",
    bbox_to_anchor=(0.045, 0.9),
    ncol=3,
    frameon=False,
    fontsize=11,
)

panels = [
    (axes[0], m3_ratios, ["Batch 1", "Batch 2", "Batch 4"], "M3 · ATOM · TP4"),
    (
        axes[1],
        [v41[c] for c in sorted(v41)],
        [f"C{c}" for c in sorted(v41)],
        "DeepSeek-V4.1 · vLLM · TP2",
    ),
]
for ax, groups, ticks, title in panels:
    width = 0.23 if len(groups[0]) == 3 else 0.29
    for x, group in enumerate(groups):
        for series, value in enumerate(group):
            position = x + (series - (len(group) - 1) / 2) * width
            ax.bar(position, value, width=width * 0.9, color=colors[series], zorder=3)
            ax.text(
                position,
                value + 0.035,
                f"{value:.2f}×",
                ha="center",
                fontsize=9.5,
                weight="bold" if series else "normal",
            )
    ax.set_title(title, fontsize=16, weight="bold", pad=17)
    ax.set_xticks(range(len(groups)), ticks, fontsize=11)
    ax.set_ylim(0, 2.08)
    ax.set_yticks([0, 0.5, 1, 1.5, 2], ["0", "0.5×", "1.0×", "1.5×", "2.0×"])
    ax.axhline(1, color="#64748b", linewidth=1, linestyle="--", zorder=2)
    ax.grid(axis="y", color="#e5ebf0", zorder=0)
    ax.tick_params(length=0, pad=8, colors="#465a6b")
    for spine in ax.spines.values():
        spine.set_visible(False)
axes[0].set_xlabel(
    "GPU decode-step speedup · higher is better", labelpad=12, fontsize=10
)
axes[1].set_xlabel(
    "P90 interactivity ratio · higher is better", labelpad=12, fontsize=10
)
fig.text(
    0.05,
    0.08,
    "Each batch or concurrency group has its own baseline at 1.00×. Raw values are in Figures 3 and 5.",
    fontsize=11,
    color="#586b7b",
)
fig.text(
    0.05,
    0.035,
    "Sources: ATOM #2419 and vLLM #60397. M3 includes a separate CP option; V4.1 uses K1 + K2 on 30 layers.",
    fontsize=10,
    color="#586b7b",
)
fig.subplots_adjust(left=0.06, right=0.98, top=0.72, bottom=0.23, wspace=0.16)
fig.savefig(ROOT.parent / "images" / "mono-speedup-comparison.png", dpi=200)
plt.close(fig)
