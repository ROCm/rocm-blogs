"""Draw the bit layout of the floating-point formats used for model weights.

Usage: python3 draw_number_formats.py <output.png>
"""
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

# name, exponent bits, mantissa bits, largest finite value
FORMATS = [
    ("FP32", 8, 23, "3.4e38"),
    ("FP16", 5, 10, "65,504"),
    ("BF16", 8, 7, "3.4e38"),
    ("FP8 E4M3", 4, 3, "448"),
    ("FP8 E5M2", 5, 2, "57,344"),
]

SIGN_COLOR = "#8d99ae"
EXPONENT_COLOR = "#1f7a8c"
MANTISSA_COLOR = "#e0a458"

BLOCK = 0.82        # width of one bit block; the shortfall is the gap between bits
PITCH = 1.0         # spacing between bit blocks, the same across field boundaries
FIELD_GAP = 0.0     # bits line up in one grid; color separates the fields
ROW_PITCH = 2.6     # vertical spacing between formats
BLOCK_HEIGHT = BLOCK

fig, ax = plt.subplots(figsize=(11, 3.6), facecolor="white")

for row, (name, exponent, mantissa, largest) in enumerate(FORMATS):
    y = (len(FORMATS) - row - 1) * ROW_PITCH
    x = 0.0
    for count, color in [(1, SIGN_COLOR), (exponent, EXPONENT_COLOR), (mantissa, MANTISSA_COLOR)]:
        field_start = x
        for _ in range(count):
            ax.add_patch(Rectangle((x, y), BLOCK, BLOCK_HEIGHT, facecolor=color,
                                   edgecolor="0.35", linewidth=0.6))
            x += PITCH
        field_end = x - PITCH + BLOCK
        ax.text((field_start + field_end) / 2, y + BLOCK_HEIGHT + 0.12, str(count),
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)
        x += FIELD_GAP
    total_bits = 1 + exponent + mantissa
    ax.text(-0.7, y + BLOCK_HEIGHT / 2, name, ha="right", va="center", fontsize=11, fontweight="600")
    ax.text(x - PITCH + BLOCK + 0.6, y + BLOCK_HEIGHT / 2,
            f"{total_bits} bits = {total_bits // 8} byte{'s' if total_bits > 8 else ''}"
            f"  ·  largest value {largest}", ha="left", va="center", fontsize=9.5, color="0.25")

ax.set_xlim(-8, 52)
ax.set_ylim(-1.2, len(FORMATS) * ROW_PITCH)
ax.set_aspect("equal")
ax.set_axis_off()
fig.legend(
    handles=[
        Patch(facecolor=SIGN_COLOR, label="Sign (1 bit)"),
        Patch(facecolor=EXPONENT_COLOR, label="Exponent — range"),
        Patch(facecolor=MANTISSA_COLOR, label="Mantissa — precision"),
    ],
    loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0),
)
fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.12)
fig.savefig(sys.argv[1], dpi=150, facecolor="white")
