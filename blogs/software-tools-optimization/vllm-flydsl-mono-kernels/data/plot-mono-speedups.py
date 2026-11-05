"""Plot concurrency-1 FlyDSL mono decode speedups from the merged vLLM PRs."""

import csv
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images", "vllm-mono-speedups.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BAR = "#2a78d6"

with open(os.path.join(HERE, "mono-speedups.csv")) as f:
    rows = list(csv.DictReader(f))
rows.reverse()  # largest bar on top

labels = [f"{r['model']} · {r['gpu']} {r['config']}\n{r['metric']}" for r in rows]
values = [float(r["speedup"]) for r in rows]

fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

y = range(len(rows))
ax.barh(y, values, height=0.56, color=BAR, edgecolor=SURFACE, linewidth=2)
ax.axvline(1.0, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
ax.text(1.02, -0.62, "stock vLLM = 1.00×", color=INK_2, fontsize=9, ha="left", va="center")
for i, v in zip(y, values):
    ax.text(v + 0.03, i, f"{v:.2f}×", va="center", ha="left", color=INK, fontsize=10, fontweight="bold")

ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=9, color=INK)
ax.set_xlim(0, 2.6)
ax.set_xlabel("Speedup over stock vLLM at concurrency 1 (higher is better)", color=INK_2, fontsize=10)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", colors=INK_2, labelsize=9)
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.9, len(rows) - 0.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.text(
    0.01,
    0.97,
    "FlyDSL mono decode in vLLM: concurrency-1 speedups from merged PRs",
    color=INK,
    fontsize=12,
    fontweight="bold",
    va="top",
)
fig.savefig(OUT, facecolor=SURFACE)
print(OUT)
