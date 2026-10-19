"""Regenerate Figure 4 from atom-agentx.csv. Requires matplotlib."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
with (ROOT / "atom-agentx.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, axes = plt.subplots(2, 1, figsize=(10.5, 10), facecolor="#f8fafc")
fig.subplots_adjust(left=0.14, right=0.94, top=0.83, bottom=0.19, hspace=0.65)

panels = (
    ("MiniMax-M3-MXFP4", (4,), (345, 510), (5000, 11000), [6000, 8000, 10000]),
    ("DeepSeek-V4.1-Flash", (2, 4), (280, 690), (4000, 39000), [10000, 20000, 30000]),
)
for ax, (model, tps, xlim, ylim, ticks) in zip(axes, panels):
    ax.set_facecolor("#f8fafc")
    for tp in tps:
        group = sorted(
            (r for r in rows if r["model"] == model and int(r["tp"]) == tp),
            key=lambda r: int(r["concurrency"]),
        )
        x = [float(r["p90_interactivity_tokens_per_s_per_user"]) for r in group]
        y = [float(r["input_output_throughput_tokens_per_s_per_gpu"]) for r in group]
        color = "#d64440" if tp == 4 else "#216b9a"
        ax.plot(x, y, "o-", color=color, lw=1.7, markersize=7, label=f"TP{tp}")
        for r, px, py in zip(group, x, y):
            c = int(r["concurrency"])
            offset = (9, 6)
            if model == "DeepSeek-V4.1-Flash" and tp == 2 and c == 2:
                offset = (9, -17)
            ax.annotate(f"C{c}", (px, py), xytext=offset,
                        textcoords="offset points", color=color, fontsize=11)
    ax.set_title(model, loc="left", fontsize=15, weight="bold", pad=12)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}K"))
    ax.set_xlabel("P90 interactivity (tokens/s/user)", labelpad=10)
    ax.set_ylabel("Input + output throughput\n(tokens/s/GPU)", labelpad=10)
    ax.grid(axis="y", color="#dce2e8", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=11)

fig.text(0.14, 0.955, "High-interactivity serving in ATOM", fontsize=21, weight="bold")
fig.text(0.14, 0.915, "InferenceX AgentX · MI355X · FlyDSL mono enabled", fontsize=13)
fig.text(0.14, 0.875, "C = request concurrency · Each panel has its own axis scales", fontsize=11)
fig.text(0.14, 0.11, "Complete serving configurations; these runs have no matched mono-off baseline.", fontsize=10)
fig.text(0.14, 0.08, "3,600 s/point · prescribed speculative acceptance · GPU-resident KV caches", fontsize=10)
fig.text(0.14, 0.05, "Throughput includes cached input context. TTFT appears in the table. Source: InferenceX.", fontsize=10)
fig.savefig(ROOT.parent / "images" / "atom-throughput-interactivity.png", dpi=160)
plt.close(fig)
