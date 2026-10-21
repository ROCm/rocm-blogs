"""Rebuild the article figures. Requires matplotlib and cairosvg.

Run: python data/render-figures.py
Performance plots read the adjacent CSV files; no GPU execution is required.
"""

import csv
from pathlib import Path

import cairosvg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATA = Path(__file__).resolve().parent
IMAGES = DATA.parent / "images"
BACKGROUND = "#f8fafc"
INK = "#17283d"
BASELINE = "#8798ad"
MONO = "#ce3e44"
EXTENDED = "#087f8c"


def read_csv(name):
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def style_axis(ax):
    ax.set_facecolor(BACKGROUND)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#dde4ec", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=8)


def paired_bars(ax, rows, before, after, colors, limit):
    positions = [6, 4, 2, 0]
    for y, row in zip(positions, rows):
        off, on = float(row[before]), float(row[after])
        ax.axhspan(y - 0.82, y + 0.82, color="#eef2f6", zorder=0)
        ax.barh(y + 0.3, off, height=0.5, color=colors[0])
        ax.barh(y - 0.3, on, height=0.5, color=colors[1])
        ax.text(off + 7, y + 0.3, f"{off:.1f}", va="center", fontsize=12)
        ax.text(
            on + 7,
            y - 0.3,
            f"{on:.1f}  ({on / off:.2f}×)",
            va="center",
            fontsize=12,
            weight="bold",
        )
    ax.set_xlim(0, limit)
    ax.set_ylim(-1, 7)
    ax.set_yticks(positions, [f"C{r['concurrency']}" for r in rows])
    style_axis(ax)


def interactivity():
    rows = sorted(
        read_csv("serving-interactivity.csv"), key=lambda row: int(row["concurrency"])
    )
    fig, ax = plt.subplots(figsize=(10, 6.8), facecolor=BACKGROUND)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.75, bottom=0.24)
    paired_bars(
        ax,
        rows,
        "baseline_interactivity",
        "flydsl_interactivity",
        (BASELINE, MONO),
        580,
    )
    ax.set_xlabel("P90 interactivity (tokens/s/user) · higher is faster", labelpad=12)
    ax.legend(
        handles=[
            Patch(color=BASELINE, label="FlyDSL Mono off"),
            Patch(color=MONO, label="FlyDSL Mono on · 30 layers"),
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        fontsize=12,
    )
    fig.text(0.11, 0.94, "Faster token generation in vLLM", fontsize=22, weight="bold")
    fig.text(0.11, 0.89, "DeepSeek-V4.1-Flash · MI355X · TP2", fontsize=15)
    fig.text(
        0.11, 0.12, "Matched off/on pairs at each request concurrency (C).", fontsize=12
    )
    fig.text(
        0.11,
        0.075,
        "One 3,600 s run per mode and C · synthetic DSpark acceptance 3.51",
        fontsize=11,
    )
    fig.text(
        0.11,
        0.03,
        "Source: vLLM #60397. TTFT is reported separately in the article.",
        fontsize=11,
    )
    fig.savefig(IMAGES / "vllm-flydsl-interactivity.png", dpi=180)
    plt.close(fig)


def coverage():
    rows = read_csv("coverage-interactivity.csv")
    fig, axes = plt.subplots(2, 1, figsize=(10, 11.3), facecolor=BACKGROUND)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.8, bottom=0.17, hspace=0.52)
    for ax, tp in zip(axes, (2, 4)):
        points = sorted(
            [row for row in rows if int(row["tp"]) == tp],
            key=lambda row: int(row["concurrency"]),
        )
        paired_bars(
            ax,
            points,
            "baseline_interactivity",
            "extended_interactivity",
            (MONO, EXTENDED),
            740,
        )
        ax.set_title(f"TP{tp}", loc="left", fontsize=16, weight="bold", pad=10)
        ax.set_xlabel(
            "P90 interactivity (tokens/s/user) · higher is faster", labelpad=8
        )
    fig.text(
        0.11,
        0.95,
        "Extending FlyDSL coverage to all 40 layers",
        fontsize=21,
        weight="bold",
    )
    fig.text(0.11, 0.915, "DeepSeek-V4.1-Flash in vLLM · MI355X", fontsize=15)
    fig.legend(
        handles=[
            Patch(color=MONO, label="30 layers: FlyDSL K1 + K2"),
            Patch(color=EXTENDED, label="40 layers: add FlyDSL FFN launch"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.10, 0.9),
        frameon=False,
        fontsize=12,
    )
    fig.text(
        0.11,
        0.105,
        "Every baseline already has FlyDSL Mono enabled on 30 layers.",
        fontsize=12,
    )
    fig.text(
        0.11,
        0.073,
        "3,600 s per point · synthetic DSpark acceptance 3.51 · TP2 uses the rerun",
        fontsize=11,
    )
    fig.text(
        0.11,
        0.041,
        "Ratios use the displayed values. Source: vLLM #60397.",
        fontsize=11,
    )
    fig.savefig(IMAGES / "vllm-flydsl-coverage.png", dpi=180)
    plt.close(fig)


def architecture():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1010" viewBox="0 0 1000 1010" role="img" aria-labelledby="title desc">
<title id="title">FlyDSL mono kernels in vLLM</title>
<desc id="desc">Thirty standard layers use FlyDSL K1 for attention preparation and K2 for attention, reductions, and the FFN. The other ten keep vLLM attention and use the same FlyDSL FFN stages. Both return the same layer tensor contract.</desc>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#64748b"/></marker>
  <style>text { font-family: DejaVu Sans, sans-serif; fill: #17283d; } .title { font-size: 32px; font-weight: 700; } .head { font-size: 25px; font-weight: 700; } .body { font-size: 22px; } .small { font-size: 20px; } .arrow { fill: none; stroke: #64748b; stroke-width: 3; marker-end: url(#arrow); }</style>
</defs>
<rect width="1000" height="1010" fill="#f8fafc"/>
<text x="44" y="56" class="title">FlyDSL mono kernels in vLLM</text>
<text x="44" y="96" class="body">DeepSeek-V4.1-Flash · eligible decode steps on MI355X</text>
<text x="44" y="158" class="head">30 standard layers</text>
<text x="536" y="158" class="head">Other 10 backbone layers</text>

<rect x="44" y="181" width="420" height="250" rx="14" fill="#fcecee" stroke="#ce3e44" stroke-width="2"/>
<text x="68" y="222" class="head">FlyDSL K1</text>
<text x="68" y="262" class="body">mHC residual mixing + norm</text>
<text x="68" y="300" class="body">Query / KV projections + norms</text>
<text x="68" y="338" class="body">KV-cache insertion</text>
<text x="68" y="376" class="body">Query projection + RoPE</text>
<path d="M254 435 L254 477" class="arrow"/>

<rect x="44" y="484" width="420" height="392" rx="14" fill="#fcecee" stroke="#ce3e44" stroke-width="2"/>
<text x="68" y="528" class="head">FlyDSL K2</text>
<text x="68" y="568" class="body">Split attention + combine</text>
<text x="68" y="606" class="body">Inverse RoPE + output projections</text>
<line x1="68" y1="629" x2="440" y2="629" stroke="#e9b6bd" stroke-width="2"/>
<text x="68" y="665" class="body">Attention TP reduction</text>
<text x="68" y="703" class="body">FFN residual mixing + norm</text>
<text x="68" y="741" class="body">Router + top-6 selection</text>
<text x="68" y="779" class="body">Shared + routed projections</text>
<text x="68" y="817" class="body">MoE TP reduction</text>

<rect x="536" y="181" width="420" height="310" rx="14" fill="#e9eef4" stroke="#8798ad" stroke-width="2"/>
<text x="560" y="222" class="head">vLLM attention path</text>
<text x="560" y="262" class="body">Model-specific residual mixing</text>
<text x="560" y="300" class="body">Compressor / indexer as needed</text>
<text x="560" y="338" class="body">Attention + output projections</text>
<text x="560" y="398" class="small">Output: unreduced TP partials</text>
<text x="560" y="446" class="small">Includes embedding / Engram layers</text>
<path d="M746 495 L746 537" class="arrow"/>

<rect x="536" y="544" width="420" height="332" rx="14" fill="#e5f3f3" stroke="#087f8c" stroke-width="2"/>
<text x="560" y="588" class="head">FlyDSL FFN launch</text>
<text x="560" y="627" class="small">Reuses K2's FFN stages</text>
<text x="560" y="665" class="body">Attention TP reduction</text>
<text x="560" y="703" class="body">FFN residual mixing + norm</text>
<text x="560" y="741" class="body">Router + top-6 selection</text>
<text x="560" y="779" class="body">Shared + routed projections</text>
<text x="560" y="817" class="body">MoE TP reduction</text>

<path d="M254 880 L254 916" class="arrow"/>
<path d="M746 880 L746 916" class="arrow"/>
<rect x="44" y="923" width="912" height="53" rx="10" fill="#17283d"/>
<text x="500" y="957" text-anchor="middle" style="fill:white;font-size:23px;font-weight:700">Same layer tensor contract · vLLM continues execution</text>
</svg>
"""
    (IMAGES / "vllm-flydsl-decode-flow.svg").write_text(svg)
    cairosvg.svg2png(
        bytestring=svg.encode(),
        write_to=str(IMAGES / "vllm-flydsl-decode-flow.png"),
        output_width=1800,
        output_height=1818,
    )


def main():
    IMAGES.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK}
    )
    architecture()
    interactivity()
    coverage()
    print(
        "Regenerated the architecture figure and both CSV-backed performance figures."
    )


if __name__ == "__main__":
    main()
