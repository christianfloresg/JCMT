#!/usr/bin/env python3
"""Publication prototype for the hierarchical JCMT envelope classification.

Panel A shows the two concentration criteria. Panel B shows the earlier
HCO-strength/association gates. Final class is redundantly encoded by color and
marker shape. Missing concentration or peak measurements occupy labeled strips
rather than being silently omitted.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"Embedded": "#2B6F9C", "Non-embedded": "#D39C2C", "Confused": "#B05A67"}
MARKERS = {"Embedded": "o", "Non-embedded": "s", "Confused": "X"}


def num(value):
    try:
        ans = float(value)
        return ans if math.isfinite(ans) else math.nan
    except (TypeError, ValueError):
        return math.nan


def yes(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def plot(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.35), constrained_layout=True)
    ax = axes[0]
    ax.axvspan(-0.08, 0, color="#ECEFF1", zorder=0)
    ax.axhspan(-0.08, 0, color="#ECEFF1", zorder=0)
    ax.fill_between([0.6, 1.03], 0.6, 1.03, color="#2B6F9C", alpha=0.07, zorder=0)
    for category in ("Embedded", "Non-embedded", "Confused"):
        subset = [r for r in rows if r["classification"] == category]
        x = [num(r["C_HCO"]) if math.isfinite(num(r["C_HCO"])) else -0.04 for r in subset]
        y = [num(r["C_C18O"]) if math.isfinite(num(r["C_C18O"])) else -0.04 for r in subset]
        ax.scatter(x, y, s=58, marker=MARKERS[category], color=COLORS[category],
                   edgecolor="white", linewidth=0.75, alpha=0.93, label=category, zorder=3)
    ax.axvline(0.6, color="#222", linestyle="--", linewidth=1.05)
    ax.axhline(0.6, color="#222", linestyle="--", linewidth=1.05)
    ax.text(0.815, 0.985, "both concentration\ncriteria pass", ha="center", va="top",
            fontsize=8, color="#2B6F9C")
    ax.text(-0.04, 0.98, "HCO+ C\nunavailable", ha="center", va="top", fontsize=7,
            color="#68737D", rotation=90)
    ax.text(0.98, -0.04, "C18O C unavailable", ha="right", va="center", fontsize=7,
            color="#68737D")
    ax.set(xlim=(-0.08, 1.03), ylim=(-0.08, 1.03), xlabel=r"$C_{\rm HCO^+}$",
           ylabel=r"$C_{\rm C^{18}O}$", title="A. Concentration gate")

    ax = axes[1]
    # Background regions correspond to the classification precedence.
    ax.axhspan(-2.0, 1.0, color="#D39C2C", alpha=0.08, zorder=0)
    ax.fill_between([15.0, 62.0], 1.0, 100.0, color="#B05A67", alpha=0.07, zorder=0)
    ax.axvspan(-4.0, 0.0, color="#ECEFF1", zorder=0)
    for category in ("Embedded", "Non-embedded", "Confused"):
        subset = [r for r in rows if r["classification"] == category]
        xs, ys = [], []
        for r in subset:
            offsets = []
            if yes(r["HCO_peak_significant"]) and math.isfinite(num(r["HCO_peak_offset_arcsec"])):
                offsets.append(num(r["HCO_peak_offset_arcsec"]))
            if yes(r["C18O_peak_significant"]) and math.isfinite(num(r["C18O_peak_offset_arcsec"])):
                offsets.append(num(r["C18O_peak_offset_arcsec"]))
            xs.append(max(offsets) if offsets else -2.0)
            limit = num(r["W_HCO_threshold_Kkms"])
            ys.append(num(r["W_HCO_Kkms"]) / limit if limit > 0 else math.nan)
        finite = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
        ax.scatter([x for x, _ in finite], [y for _, y in finite], s=58,
                   marker=MARKERS[category], color=COLORS[category], edgecolor="white",
                   linewidth=0.75, alpha=0.93, label=category, zorder=3)
    ax.axvline(15.0, color="#222", linestyle="--", linewidth=1.05)
    ax.axhline(1.0, color="#222", linestyle="-.", linewidth=1.05)
    ax.set_yscale("symlog", linthresh=0.25, linscale=0.8)
    ax.set_xlim(-4, 62)
    ax.set_ylim(-1.2, 100)
    ax.text(-2, 65, "no significant\nmap peak", ha="center", va="top", fontsize=7,
            color="#68737D", rotation=90)
    ax.text(38, 52, "off-source gas: 5 confused", ha="center", fontsize=8,
            color="#934957")
    ax.text(37, 0.12, "weak/absent HCO+: 20\n→ non-embedded", ha="center", fontsize=8,
            color="#9A711C")
    ax.text(7.2, 25, "strong + associated: 9\n→ evaluate concentrations",
            ha="center", fontsize=8, color="#45525C")
    ax.set(xlabel="Largest significant HCO+ or C18O peak offset (arcsec)",
           ylabel=r"$W_{\rm HCO^+}/[0.4(140\,{\rm pc}/d)]$",
           title="B. HCO+ strength and association gates")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(True, color="#D7DCE0", linewidth=0.7, alpha=0.72)
        a.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("Hierarchical envelope classification diagnostics",
                 fontsize=14.5, fontweight="bold", y=1.015)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix("." + extension), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="classification_corrected_15arcsec.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path stem without extension")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    plot(rows, args.output)


if __name__ == "__main__":
    main()
