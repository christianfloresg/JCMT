#!/usr/bin/env python3
"""Classify the JCMT sample and plot legacy C versus peak offset.

This script intentionally reads a small, editable CSV with one row per YSO.
Classification precedence:

1. NON-EMBEDDED if HCO+ has no significant peak, or if W(HCO+) at the YSO
   does not exceed 0.4*(140 pc/d).  Weak/absent HCO+ is not "confused".
2. CONFUSED if that HCO+ gate passes and a significant HCO+ and/or C18O peak
   is farther than the selected offset threshold from the YSO.
3. EMBEDDED if both significant peaks are within the threshold, both C values
   exceed 0.6, and the HCO+ intensity gate passes.
4. All remaining objects are NON-EMBEDDED.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {"Embedded": "#2B6F9C", "Non-embedded": "#D39C2C", "Confused": "#B05A67"}
MARKERS = {"Embedded": "o", "Non-embedded": "s", "Confused": "X"}


def number(value):
    try:
        answer = float(value)
        return answer if math.isfinite(answer) else math.nan
    except (TypeError, ValueError):
        return math.nan


def boolean(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def classify(row, offset_threshold, confusion_require_both=False):
    distance = number(row["distance_pc"])
    c_hco, c_c18 = number(row["C_HCO"]), number(row["C_C18O"])
    w_hco = number(row["W_HCO_Kkms"])
    w_limit = 0.4 * 140.0 / distance if distance > 0 else math.nan
    h_sig, c_sig = boolean(row["HCO_peak_significant"]), boolean(row["C18O_peak_significant"])
    h_off, c_off = number(row["HCO_peak_offset_arcsec"]), number(row["C18O_peak_offset_arcsec"])
    w_pass = math.isfinite(w_hco) and math.isfinite(w_limit) and w_hco > w_limit
    h_near = h_sig and math.isfinite(h_off) and h_off <= offset_threshold
    c_near = c_sig and math.isfinite(c_off) and c_off <= offset_threshold
    off_source = ((h_sig and math.isfinite(h_off) and h_off > offset_threshold) or
                  (c_sig and math.isfinite(c_off) and c_off > offset_threshold))
    concentrated = math.isfinite(c_hco) and c_hco > 0.6 and math.isfinite(c_c18) and c_c18 > 0.6

    confusion_tracers_present = h_sig and (c_sig if confusion_require_both else True)
    eligible_confusion = off_source and confusion_tracers_present
    if not h_sig or not w_pass:
        category = "Non-embedded"
    elif eligible_confusion:
        category = "Confused"
    elif h_near and c_near and concentrated:
        category = "Embedded"
    else:
        category = "Non-embedded"

    reasons = []
    if not h_sig: reasons.append("no_significant_HCO_peak")
    if not w_pass: reasons.append("HCO_intensity_below_scaled_threshold")
    if confusion_tracers_present and h_sig and w_pass and math.isfinite(h_off) and h_off > offset_threshold:
        reasons.append("HCO_peak_beyond_offset_threshold")
    if confusion_tracers_present and h_sig and w_pass and c_sig and math.isfinite(c_off) and c_off > offset_threshold:
        reasons.append("C18O_peak_beyond_offset_threshold")
    if h_sig and w_pass and not eligible_confusion:
        if not c_sig: reasons.append("no_significant_C18O_peak")
        if off_source and confusion_require_both and not c_sig:
            reasons.append("offsource_HCO_but_confusion_rule_requires_both_tracers")
        if not (math.isfinite(c_hco) and c_hco > 0.6): reasons.append("C_HCO_not_gt0p6")
        if not (math.isfinite(c_c18) and c_c18 > 0.6): reasons.append("C_C18O_not_gt0p6")

    result = dict(row)
    result.update({
        "W_HCO_threshold_Kkms": w_limit,
        "HCO_intensity_pass": w_pass,
        "HCO_peak_within_threshold": h_near,
        "C18O_peak_within_threshold": c_near,
        "classification": category,
        "classification_reason": ";".join(reasons) or "all_embedded_criteria_pass",
        "confusion_rule": ("both_tracers_significant" if confusion_require_both
                           else "HCO_and_or_C18O_significant"),
    })
    return result


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_plot(rows, output_stem, offset_threshold, physical=False):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), constrained_layout=True)
    for ax, molecule, c_key, offset_key in [
        (axes[0], "HCO$^+$", "C_HCO", "HCO_peak_offset_arcsec"),
        (axes[1], "C$^{18}$O", "C_C18O", "C18O_peak_offset_arcsec"),
    ]:
        for category in ("Embedded", "Non-embedded", "Confused"):
            selected = []
            for row in rows:
                x, y = number(row[offset_key]), number(row[c_key])
                sig_key = "HCO_peak_significant" if offset_key.startswith("HCO") else "C18O_peak_significant"
                if row["classification"] == category and boolean(row[sig_key]) and math.isfinite(x) and math.isfinite(y):
                    if physical:
                        x *= number(row["distance_pc"])
                    selected.append((x, y))
            if selected:
                ax.scatter([x for x, _ in selected], [y for _, y in selected], s=55,
                           marker=MARKERS[category], color=COLORS[category], edgecolor="white",
                           linewidth=0.7, label=category, alpha=0.92)
        ax.axhline(0.6, color="#222222", linestyle="-.", linewidth=1.0, label="$C=0.6$")
        if physical:
            limits = [offset_threshold * number(r["distance_pc"]) for r in rows]
            ax.axvspan(min(limits), max(limits), color="#56616B", alpha=0.09,
                       label=f"Physical span of {offset_threshold:g}″ boundary")
            xlabel = "Nearest significant peak offset from YSO (AU)"
        else:
            ax.axvline(offset_threshold, color="#222222", linestyle="--", linewidth=1.1,
                       label=f"Offset boundary ({offset_threshold:g}″)")
            xlabel = "Nearest significant peak offset from YSO (arcsec)"
        ax.set(xlabel=xlabel, ylabel="Legacy $C$ (YSO-anchored)", title=molecule,
               ylim=(-0.08, 1.03))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#D7DCE0", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Legacy concentration versus selected gas-peak offset\n"
                 "Weak or absent HCO$^+$ is classified non-embedded before testing confusion",
                 fontsize=13.3, fontweight="bold")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_stem.with_suffix("." + extension), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--offset-threshold", type=float, default=15.0)
    parser.add_argument("--confusion-require-both", action="store_true",
                        help="Sensitivity option: require significant peaks in both tracers before calling confused")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline="") as handle:
        inputs = list(csv.DictReader(handle))
    results = [classify(row, args.offset_threshold, args.confusion_require_both) for row in inputs]
    save_csv(args.outdir / "classification_from_simple_input.csv", results)
    make_plot(results, args.outdir / "legacy_C_vs_offset_reclassified_arcsec",
              args.offset_threshold, physical=False)
    make_plot(results, args.outdir / "legacy_C_vs_offset_reclassified_AU",
              args.offset_threshold, physical=True)
    print(dict(Counter(row["classification"] for row in results)))


if __name__ == "__main__":
    main()
