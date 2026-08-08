# Regenerates images/kimi-k3-vllm-sglang-throughput.png (Figure 3).
#
# Data are the total-token-throughput values from Table 7, taken straight from each
# engine's perf_Kimi-K3.csv for the 2026-07-29 out-of-box run on 8x MI350X.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

CONCURRENCY = [1, 4, 8, 16, 32, 64, 128]

SERIES = [
    ("vLLM", [287.28, 1000.47, 1744.78, 2985.02, 4675.92, 6567.25, 8228.15], "#1f77b4"),
    ("SGLang", [422.75, 1388.58, 2263.67, 3451.65, 4693.86, 5994.97, 6293.32], "#ff7f0e"),
    ("ATOM", [346.12, 1187.52, 2056.96, 3297.54, 4691.54, 5024.71, 5136.26], "#2ca02c"),
]

TITLE = (
    "Kimi-K3 day-0 OOB serving: vLLM vs SGLang vs ATOM on MI350X\n"
    "8192 in / 1024 out, TP8, madengine default sweep — Total TPS"
)

plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(12, 7.75), dpi=120)
fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

x = range(len(CONCURRENCY))
for label, values, color in SERIES:
    ax.plot(x, values, marker="o", markersize=7, linewidth=2,
            linestyle="--", color=color, label=label, clip_on=False)
    # end-of-line callout with the concurrency-128 value
    ax.annotate(f"{label}\n{values[-1]:,.0f} tok/s",
                xy=(x[-1], values[-1]), xytext=(12, 0),
                textcoords="offset points", va="center",
                fontsize=13, fontweight="bold", color="#222222")

ax.set_title(TITLE, fontsize=17, fontweight="bold", loc="left", pad=18, color="#111111")
ax.set_xlabel("Max concurrency", fontsize=14, color="#333333")
ax.set_ylabel("Total token throughput (tok/s)", fontsize=14, color="#333333")

ax.set_xticks(list(x))
ax.set_xticklabels([str(c) for c in CONCURRENCY], fontsize=13)
ax.tick_params(axis="y", labelsize=13)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_ylim(0, 9000)
ax.set_xlim(-0.15, len(CONCURRENCY) - 1)

ax.grid(axis="y", color="#d9d9d9", linewidth=0.9)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#bbbbbb")

ax.legend(fontsize=13, frameon=False, loc="upper left", handlelength=2.4)

fig.subplots_adjust(left=0.085, right=0.80, top=0.86, bottom=0.09)
fig.savefig("images/kimi-k3-vllm-sglang-throughput.png", facecolor=fig.get_facecolor())
