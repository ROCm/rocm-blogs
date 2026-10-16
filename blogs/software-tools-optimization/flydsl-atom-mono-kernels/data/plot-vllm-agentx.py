"""Regenerate Figure 5 from mono-benefits.csv. Requires matplotlib."""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
SOURCE = "https://github.com/vllm-project/vllm/pull/60397"
with (ROOT / "mono-benefits.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

points = {}
for row in rows:
    if row["source"] != SOURCE or "AgentX" not in row["configuration"]:
        continue
    concurrency = int(re.search(r"concurrency (\d+);", row["configuration"])[1])
    points.setdefault(concurrency, {})[row["metric"]] = (
        float(row["baseline_value"]),
        float(row["optimized_value"]),
    )

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, ax = plt.subplots(figsize=(10.5, 6.6), facecolor="#f8fafc")
fig.subplots_adjust(left=0.13, right=0.96, top=0.78, bottom=0.30)
ax.set_facecolor("#f8fafc")
colors = ("#526274", "#d64440")
for concurrency, metrics in sorted(points.items()):
    x0, x1 = metrics["P90 interactivity"]
    y0, y1 = metrics["total throughput (input + output)"]
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops={"arrowstyle": "->", "color": "#a3adb9", "lw": 1.7},
        zorder=1,
    )
    ax.scatter([x0], [y0], s=64, color=colors[0], zorder=3)
    ax.scatter([x1], [y1], s=64, color=colors[1], zorder=3)
    ax.annotate(
        f"C{concurrency}", (x0, y0), xytext=(-9, -17),
        textcoords="offset points", ha="right", color=colors[0], fontsize=11,
    )
    ax.annotate(
        f"C{concurrency}", (x1, y1), xytext=(9, 5),
        textcoords="offset points", color=colors[1], fontsize=11,
    )

for color, label in zip(colors, ("Mono off", "FlyDSL mono: K1 + K2 on 30 layers")):
    ax.scatter([], [], s=64, color=color, label=label)
ax.legend(loc="upper right", frameon=False, fontsize=10)
ax.set_xlim(190, 465)
ax.set_ylim(17000, 67000)
ax.set_yticks([20000, 30000, 40000, 50000, 60000])
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}K"))
ax.set_xlabel("P90 interactivity (tokens/s/user) — higher is better", labelpad=12)
ax.set_ylabel("Input + output throughput (tokens/s)", labelpad=10)
ax.grid(axis="y", color="#dce2e8", linewidth=0.8)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0, pad=8)
fig.text(0.13, 0.93, "More work served, faster token delivery", fontsize=20, weight="bold")
fig.text(0.13, 0.87, "DeepSeek-V4.1-Flash in vLLM · MI355X · TP2", fontsize=13)
fig.text(0.13, 0.15, "Arrows pair the same concurrency (C1, C2, C4, C8); four measured points, not a full curve.", fontsize=10)
fig.text(0.13, 0.11, "Same-node comparison · 3,600 s/point · synthetic DSpark acceptance 3.51", fontsize=10)
fig.text(0.13, 0.07, "Throughput includes cached input context. TTFT is reported separately. Source: vLLM #60397.", fontsize=10)
fig.savefig(ROOT.parent / "images" / "vllm-dsv41-throughput-interactivity.png", dpi=160)
plt.close(fig)
