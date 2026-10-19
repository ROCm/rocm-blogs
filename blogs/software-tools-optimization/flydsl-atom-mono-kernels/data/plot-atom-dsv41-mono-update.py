"""Regenerate Figure 4 from atom-dsv41-mono-update.csv. Requires matplotlib."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
with (ROOT / "atom-dsv41-mono-update.csv").open(newline="") as stream:
    rows = sorted(csv.DictReader(stream), key=lambda row: int(row["request_batch"]))
if len(rows) != 4 or len({row["request_batch"] for row in rows}) != 4:
    raise ValueError("Expected four distinct batch sizes")
if any(row["framework"] != "ATOM" or row["tp"] != "4" for row in rows):
    raise ValueError("Expected ATOM TP4 measurements")
values = [float(row["reported_step_time_improvement_pct"]) for row in rows]
if not all(math.isfinite(value) and value > 0 for value in values):
    raise ValueError("Reported improvements must be finite and positive")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "text.color": "#152a3c",
        "axes.labelcolor": "#465a6b",
        "savefig.facecolor": "white",
    }
)
fig, ax = plt.subplots(figsize=(10.4, 4.6))
fig.text(
    0.045, 0.91, "V4.1 in ATOM: further mono improvements", fontsize=21, weight="bold"
)
fig.text(
    0.045,
    0.845,
    "DeepSeek-V4.1-Flash · TP4 · updated mono versus previous mono",
    fontsize=11,
    color="#586b7b",
)
positions = list(range(len(rows)))
ax.bar(positions, values, width=0.53, color="#146184")
for x, value in zip(positions, values):
    ax.text(
        x,
        value + 0.3,
        f"{value:g}%",
        ha="center",
        va="bottom",
        fontsize=15,
        weight="bold",
    )
ax.set_xticks(positions, [f"Batch {row['request_batch']}" for row in rows])
ax.set_yticks([0, 4, 8, 12], ["0%", "4%", "8%", "12%"])
ax.set_ylim(0, max(values) + 2)
ax.set_ylabel("Reported step-time improvement", fontsize=11, labelpad=12)
ax.tick_params(axis="both", length=0, labelsize=11, pad=8, colors="#64748b")
ax.set_axisbelow(True)
ax.grid(axis="y", color="#e5ebf0", linewidth=0.8)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.text(
    0.045,
    0.065,
    "Source: ATOM PR #2545. Both versions use mono; this is not a mono-off/on comparison.",
    fontsize=10,
    color="#586b7b",
)
fig.text(
    0.045,
    0.025,
    "Source-reported percentages; absolute step times and repeat-run statistics are not provided.",
    fontsize=9,
    color="#586b7b",
)
fig.subplots_adjust(left=0.12, right=0.965, top=0.77, bottom=0.19)
fig.savefig(ROOT.parent / "images" / "atom-dsv41-mono-update.png", dpi=200)
plt.close(fig)
