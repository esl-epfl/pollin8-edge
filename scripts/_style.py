"""Shared figure style: Wong (2011) colour-blind-safe palette + IEEE sizing.

Single-column journal width ~88 mm (3.46 in). Large fonts for A4 print. Every
bar gets a hatch so figures survive greyscale printing and colour-blind readers.
"""
from __future__ import annotations
import matplotlib as mpl
mpl.use("Agg")          # headless (cluster / CI): write files, never open a Tk window
import matplotlib.pyplot as plt

# Wong (2011) palette (Nature Methods) — colour-blind safe.
WONG = {
    "black":  "#000000",
    "orange": "#E69F00",
    "skyblue":"#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "vermil": "#D55E00",
    "purple": "#CC79A7",
}
PALETTE = [WONG["blue"], WONG["vermil"], WONG["green"], WONG["orange"],
           WONG["skyblue"], WONG["purple"], WONG["yellow"], WONG["black"]]
HATCHES = ["//", "\\\\", "xx", "..", "++", "oo", "**", "--"]

COL_W_IN = 3.46          # 88 mm single column
GOLDEN = 0.62


def apply_rc():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42,          # editable text in vector PDF
        "ps.fonttype": 42,
    })


def styled_bars(ax, labels, values, ylabel, title=None, logy=False):
    bars = ax.bar(labels, values,
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))],
                  edgecolor="black", linewidth=0.6)
    for i, b in enumerate(bars):
        b.set_hatch(HATCHES[i % len(HATCHES)])
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    return bars
