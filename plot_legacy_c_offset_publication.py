#!/usr/bin/env python3
"""Classify and plot the editable legacy-C/nearest-peak input table.

Classification uses a continuous 15 arcsec association boundary:
  * Non-embedded first: HCO+ is not significantly detected, or its
    YSO-centered integrated intensity does not exceed 0.4*(140 pc/d).
  * Confused: after that HCO+ detection/intensity gate passes, the nearest
    significant local peak in either tracer is >15 arcsec.
  * Embedded: both nearest peaks are <=15 arcsec, both legacy C values are >0.6,
    and YSO-centered HCO+ W exceeds 0.4*(140 pc/d).
  * Non-embedded: all other sources.

Edit the input CSV (for example, selected_peak_offset_arcsec or legacy_C) and
rerun this script to regenerate the tables and figures.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


COLORS = {"Embedded": "#2B6F9C", "Non-embedded": "#D39C2C", "Confused": "#B05A67"}
MARKERS = {"Embedded": "o", "Non-embedded": "s", "Confused": "X"}
GRID = "#D7DCE0"


def fnum(value):
    try:
        ans = float(value)
        return ans if math.isfinite(ans) else math.nan
    except (TypeError, ValueError):
        return math.nan


def truth(value):
    return bool(value)


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def classify(rows, threshold_arcsec=15.0):
    by = {(r["source"], r["molecule"]): r for r in rows}
    sources = list(dict.fromkeys(r["source"] for r in rows))
    output = []
    for source in sources:
        h, c = by[(source, "HCO+")], by[(source, "C18O")]
        oh, oc = fnum(h.get("selected_peak_offset_arcsec")), fnum(c.get("selected_peak_offset_arcsec"))
        ch, cc = fnum(h.get("legacy_C")), fnum(c.get("legacy_C"))
        distance = fnum(h.get("distance_pc"))
        w, werr = fnum(h.get("W_YSO_Kkms")), fnum(h.get("W_YSO_err_Kkms"))
        wlim = 0.4 * 140.0 / distance if distance > 0 else math.nan

        h_peak = math.isfinite(oh)
        c_peak = math.isfinite(oc)
        h_assoc = h_peak and oh <= threshold_arcsec
        c_assoc = c_peak and oc <= threshold_arcsec
        h_off = h_peak and oh > threshold_arcsec
        c_off = c_peak and oc > threshold_arcsec
        w_pass = math.isfinite(w) and math.isfinite(wlim) and w > wlim
        confused = h_peak and w_pass and (h_off or c_off)
        checks = {
            "HCO_peak_within15": h_assoc,
            "C18O_peak_within15": c_assoc,
            "C_HCO_gt0p6": math.isfinite(ch) and ch > 0.6,
            "C_C18O_gt0p6": math.isfinite(cc) and cc > 0.6,
            "W_HCO_gt_scaled_threshold": w_pass,
        }
        embedded = all(checks.values())
        # Precedence is physically important: absent/weak HCO+ is evidence for
        # non-embedded, not confusion caused by unrelated C18O elsewhere.
        if not h_peak or not w_pass:
            category = "Non-embedded"
        elif confused:
            category = "Confused"
        elif embedded:
            category = "Embedded"
        else:
            category = "Non-embedded"
        reasons = []
        if not h_peak: reasons.append("no_significant_HCO_peak")
        if not w_pass: reasons.append("HCO_intensity_below_scaled_threshold")
        if h_peak and w_pass and h_off: reasons.append("HCO_nearest_significant_peak_gt15")
        if h_peak and w_pass and c_off: reasons.append("C18O_nearest_significant_peak_gt15")
        if h_peak and w_pass and not confused:
            reasons.extend(k for k, ok in checks.items() if not ok)
        out = {
            "source": source, "display_name": h["display_name"], "map_source": h["map_source"],
            "distance_pc": distance, "classification_15arcsec": category,
            "classification_reason": ";".join(reasons),
            "legacy_C_HCO": ch, "legacy_C_C18O": cc,
            "HCO_selected_peak_offset_arcsec": oh, "C18O_selected_peak_offset_arcsec": oc,
            "HCO_selected_peak_offset_au": oh * distance if h_peak else math.nan,
            "C18O_selected_peak_offset_au": oc * distance if c_peak else math.nan,
            "HCO_selected_peak_offset_pc": oh * distance / 206265.0 if h_peak else math.nan,
            "C18O_selected_peak_offset_pc": oc * distance / 206265.0 if c_peak else math.nan,
            "HCO_selected_peak_snr": fnum(h.get("selected_peak_snr")),
            "C18O_selected_peak_snr": fnum(c.get("selected_peak_snr")),
            "HCO_global_peak_offset_arcsec": fnum(h.get("global_peak_offset_arcsec")),
            "C18O_global_peak_offset_arcsec": fnum(c.get("global_peak_offset_arcsec")),
            "W_HCO_YSO_Kkms": w, "W_HCO_err_Kkms": werr,
            "W_HCO_threshold_Kkms": wlim,
            "association_threshold_arcsec": threshold_arcsec,
            "association_threshold_au": threshold_arcsec * distance,
        }
        out.update(checks)
        output.append(out)
    return output


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_text(path, rows):
    with path.open("w") as handle:
        handle.write("# source classification HCO_offset_arcsec C18O_offset_arcsec reason\n")
        for row in rows:
            handle.write(
                f"{row['source']:<24} {row['classification_15arcsec']:<12} "
                f"{fnum(row['HCO_selected_peak_offset_arcsec']):8.3f} "
                f"{fnum(row['C18O_selected_peak_offset_arcsec']):8.3f} "
                f"{row['classification_reason'] or '-'}\n"
            )


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def scatter_plot(classes, outbase, physical=False):
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.2), constrained_layout=True)
    configs = [("HCO$^+$", "legacy_C_HCO", "HCO_selected_peak_offset_"),
               ("C$^{18}$O", "legacy_C_C18O", "C18O_selected_peak_offset_")]
    for ax, (title, ckey, prefix) in zip(axes, configs):
        xkey = prefix + ("au" if physical else "arcsec")
        for category in ("Embedded", "Non-embedded", "Confused"):
            subset = [r for r in classes if r["classification_15arcsec"] == category
                      and math.isfinite(fnum(r[ckey])) and math.isfinite(fnum(r[xkey]))]
            ax.scatter([fnum(r[xkey]) for r in subset], [fnum(r[ckey]) for r in subset],
                       s=55, marker=MARKERS[category], facecolor=COLORS[category],
                       edgecolor="white", linewidth=0.7, label=category, alpha=0.92)
        ax.axhline(0.6, color="#222222", linestyle="-.", linewidth=1.0, label="$C=0.6$")
        if not physical:
            ax.axvline(15.0, color="#222222", linestyle="--", linewidth=1.1,
                       label="Association boundary (15″)")
            ax.set_xlim(-1, max(62, max((fnum(r[xkey]) for r in classes if math.isfinite(fnum(r[xkey]))), default=60) * 1.05))
            xlabel = "Nearest significant gas-peak offset from YSO (arcsec)"
        else:
            thresholds = [fnum(r["association_threshold_au"]) for r in classes]
            ax.axvspan(min(thresholds), max(thresholds), color="#56616B", alpha=0.09,
                       label="Physical span of the 15″ boundary")
            xlabel = "Nearest significant gas-peak offset from YSO (AU)"
        ax.set(xlabel=xlabel, ylabel="Legacy $C$ (YSO-anchored)", title=title, ylim=(-0.08, 1.03))
        style(ax)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    subtitle = ("Classification uses legacy C, YSO-centered HCO$^+$ intensity, and the nearest >3σ local peak"
                if not physical else
                "The 15″ decision boundary spans different physical scales; shaded range shows that span")
    fig.suptitle("Legacy concentration versus gas-peak offset\n" + subtitle,
                 fontsize=13.5, fontweight="bold")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outbase.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_semicolons(row, key, cast=float):
    text = row.get(key, "")
    if not text:
        return []
    return [cast(x) for x in text.split(";") if x != ""]


def diagnostic_figure(page_rows, class_by, repo, molecule, page_number, total_pages):
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 11.5))
    axes = axes.ravel()
    for ax, row in zip(axes, page_rows):
        source = row["source"]
        category = class_by[source]["classification_15arcsec"]
        try:
            with fits.open(repo / row["map_path"]) as hdul:
                data = np.squeeze(hdul[0].data).astype(float)
                _ = WCS(hdul[0].header).celestial
            xsrc, ysrc = fnum(row["xsrc_pix"]), fnum(row["ysrc_pix"])
            pix_x, pix_y = fnum(row["pixscale_x_arcsec"]), fnum(row["pixscale_y_arcsec"])
            yy, xx = np.mgrid[: data.shape[0], : data.shape[1]]
            dx, dy = (xx - xsrc) * pix_x, (yy - ysrc) * pix_y
            finite = np.isfinite(data)
            vmin, vmax = np.nanpercentile(data[finite], [5, 99])
            ax.pcolormesh(dx, dy, data, shading="nearest", cmap="magma", vmin=vmin, vmax=vmax)
            try:
                levels = np.nanpercentile(data[finite], [65, 80, 92])
                if len(np.unique(levels)) == 3:
                    ax.contour(dx, dy, data, levels=levels, colors="white", alpha=0.65, linewidths=0.7)
            except Exception:
                pass
            ax.scatter(0, 0, marker="+", s=115, linewidth=2.0, color="#22D3EE", label="IR YSO")
            sx, sy = fnum(row.get("selected_peak_x_pix")), fnum(row.get("selected_peak_y_pix"))
            if math.isfinite(sx) and math.isfinite(sy):
                ax.scatter((sx - xsrc) * pix_x, (sy - ysrc) * pix_y, marker="*", s=100,
                           color="#F5C84C", edgecolor="#31404A", linewidth=0.6, label="selected nearest peak")
            gx, gy = fnum(row.get("global_peak_x_pix")), fnum(row.get("global_peak_y_pix"))
            if math.isfinite(gx) and math.isfinite(gy):
                ax.scatter((gx - xsrc) * pix_x, (gy - ysrc) * pix_y, marker="D", s=42,
                           facecolor="none", edgecolor="#55DDE0", linewidth=1.2, label="global maximum")
            cxs = parse_semicolons(row, "candidate_x_pix")
            cys = parse_semicolons(row, "candidate_y_pix")
            if cxs and cys:
                ax.scatter([(x - xsrc) * pix_x for x in cxs], [(y - ysrc) * pix_y for y in cys],
                           s=28, facecolor="none", edgecolor="white", linewidth=0.65,
                           label=">3σ local maxima")
            fwhm = fnum(row.get("fwhm_radius_arcsec"))
            if math.isfinite(fwhm):
                ax.add_patch(Circle((0, 0), fwhm, fill=False, edgecolor="#55E6A5",
                                    linewidth=1.25, label="YSO-fixed Gaussian $R_{obs}=FWHM$"))
            ax.add_patch(Circle((0, 0), 15.0, fill=False, edgecolor="white", linestyle=":",
                                linewidth=1.1, label="15″ association boundary"))
            cval, off = fnum(row.get("legacy_C")), fnum(row.get("selected_peak_offset_arcsec"))
            ctext = "nan" if not math.isfinite(cval) else f"{cval:.2f}"
            otext = "no >3σ peak" if not math.isfinite(off) else f"offset={off:.1f}″"
            ax.set_title(f"{row['display_name']}  [{category}]\nC={ctext}; {otext}; $R_{{obs}}$={fwhm:.1f}″" if math.isfinite(fwhm)
                         else f"{row['display_name']}  [{category}]\nC={ctext}; {otext}; $R_{{obs}}$ unavailable",
                         fontsize=8.2, color=COLORS[category])
            ax.set(xlim=(-62, 62), ylim=(-62, 62), xlabel="ΔRA-like map x (arcsec)",
                   ylabel="ΔDec-like map y (arcsec)")
            ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        except Exception as exc:
            ax.text(0.5, 0.5, f"{row['display_name']}\n{exc}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            ax.set_axis_off()
    for ax in axes[len(page_rows):]:
        ax.set_axis_off()
    handles = [
        Line2D([], [], marker="+", linestyle="none", color="#22D3EE", markersize=9, label="IR YSO"),
        Line2D([], [], marker="*", linestyle="none", markerfacecolor="#F5C84C", markeredgecolor="#31404A", markersize=9, label="selected nearest peak"),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor="none", markeredgecolor="#55DDE0", markersize=6, label="global maximum"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="white", markersize=5, label=">3σ local maxima"),
        Line2D([], [], color="#55E6A5", label="YSO-fixed $R_{obs}=FWHM$"),
        Line2D([], [], color="black", linestyle=":", label="15″ boundary"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.957), ncol=3,
               frameon=False, fontsize=7.5)
    fig.suptitle(f"All-source {molecule} moment-zero diagnostics — page {page_number}/{total_pages}",
                 y=0.99, fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.90, bottom=0.055, hspace=0.44, wspace=0.31)
    return fig


def all_diagnostics(rows, classes, repo, outdir):
    class_by = {r["source"]: r for r in classes}
    for molecule, tag in (("HCO+", "HCOplus"), ("C18O", "C18O")):
        mol_rows = [r for r in rows if r["molecule"] == molecule]
        pages = [mol_rows[i:i + 9] for i in range(0, len(mol_rows), 9)]
        pdf_path = outdir / f"all_source_map_diagnostics_{tag}.pdf"
        with PdfPages(pdf_path) as pdf:
            for i, page in enumerate(pages, 1):
                fig = diagnostic_figure(page, class_by, repo, molecule, i, len(pages))
                png = outdir / f"all_source_map_diagnostics_{tag}_page{i}.png"
                fig.savefig(png, dpi=240, bbox_inches="tight")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--offset-threshold", type=float, default=15.0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input)
    classes = classify(rows, args.offset_threshold)
    save_csv(args.outdir / "legacy_c_offset_classification_15arcsec.csv", classes)
    save_text(args.outdir / "legacy_c_offset_classification_15arcsec.txt", classes)
    scatter_plot(classes, args.outdir / "legacy_C_vs_peak_offset_arcsec", physical=False)
    scatter_plot(classes, args.outdir / "legacy_C_vs_peak_offset_AU", physical=True)
    all_diagnostics(rows, classes, args.repo.resolve(), args.outdir)


if __name__ == "__main__":
    main()
