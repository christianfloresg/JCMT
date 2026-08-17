#!/usr/bin/env python3
"""Classify legacy and gas-centered JCMT envelope diagnostics and plot changes.

Primary revised classification uses the raw moment-zero gas-centered
concentration.  A background-subtracted classification is emitted only as a
method-sensitivity check; it is not silently substituted into the paper's
thresholds.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gas_centered_concentration import COORDS, DIST, MAP_SOURCE, REPO


DISPLAY = {
    "IRAS03260+3111": "CON 3B", "IRAS03301+3111": "IRAS 03301+3111",
    "IRAS04108+2803": "IRAS 04108+2803 B", "IRAS04113+2758S": "MHO 2",
    "FPTau": "FP Tau", "IRAS04181+2655M": "IRAS 04181+2654 B",
    "IRAS04181+2655S": "IRAS 04181+2654", "T-Tauri": "T Tauri North",
    "DG-Tau": "DG Tau", "GV_Tau": "GV Tau", "HK-Tau": "HK Tau",
    "Haro6-13": "Haro 6-13", "IRAS04295+2251": "IRAS 04295+2251",
    "Haro6-28": "Haro 6-28", "Haro6-33": "Haro 6-33", "UYAur": "UY Aur A",
    "IRAS04489+3042": "IRAS 04489+3042", "V347_Aur": "V347 Aur",
    "IRAS04591-0856": "IRAS 04591-0856", "IRAS05379-0758": "IRAS 05379-0758(2)",
    "IRAS05555-1405": "IRAS 05555-1405(S)", "DoAr25": "DoAr 25",
    "SR24": "SR 24 S", "GY92-235": "GY92 235", "WLY2-42": "WLY 2-42",
    "Elia32": "Elia 2-32", "Elia33": "Elia 2-33", "YLW58": "YLW 58",
    "DoAr43": "DoAr 43", "OphIRS63": "IRAS 16285-2355", "EC92": "EC92 92",
    "IRS5": "TS84 IRS 5", "IRAS19247+2238(1)": "IRAS 19247+2238 (1)",
    "IRAS19247+2238(2)": "IRAS 19247+2238 (2)",
}

COLORS = {
    "Embedded": "#2B6F9C",
    "Non-embedded": "#D39C2C",
    "Confused": "#B05A67",
    "neutral": "#56616B",
    "grid": "#D7DCE0",
}


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError):
        return math.nan


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_old_distances():
    out = {}
    for line in (REPO / "text_files/source_distances.txt").read_text().splitlines():
        p = line.split()
        if p and not p[0].startswith("#") and len(p) >= 2:
            try: out[p[0]] = float(p[1])
            except ValueError: pass
    return out


def read_integrated_beam(molecule="HCO+"):
    out = {}
    path = REPO / f"spectrum_parameters_{molecule}.txt"
    for line in path.read_text().splitlines():
        p = line.split()
        if p and not p[0].startswith("#") and len(p) >= 11:
            out[p[0]] = (fnum(p[9]), fnum(p[10]))
    return out


def truth(v):
    return str(v).lower() == "true"


def finite_gt(v, threshold):
    return math.isfinite(v) and v > threshold


def classify(c_hco, c_c18, hrow, crow, w_hco, threshold):
    h_on = truth(hrow.get("has_onsource_peak"))
    c_on = truth(crow.get("has_onsource_peak"))
    h_global_sig = fnum(hrow.get("global_peak_snr")) >= 3
    c_global_sig = fnum(crow.get("global_peak_snr")) >= 3
    h_conf = (not h_on and h_global_sig and fnum(hrow.get("global_offset")) > 20)
    c_conf = (not c_on and c_global_sig and fnum(crow.get("global_offset")) > 20)
    confused = h_conf or c_conf

    checks = {
        "HCO_peak_within14": h_on and fnum(hrow.get("offset")) <= 14,
        "C18O_peak_within14": c_on and fnum(crow.get("offset")) <= 14,
        "C_HCO_gt0p6": finite_gt(c_hco, 0.6),
        "C_C18O_gt0p6": finite_gt(c_c18, 0.6),
        "W_HCO_gt_scaled_threshold": finite_gt(w_hco, threshold),
    }
    embedded = all(checks.values())
    if confused:
        cls = "Confused"
    elif embedded:
        cls = "Embedded"
    else:
        cls = "Non-embedded"
    failures = [k for k, ok in checks.items() if not ok]
    reasons = []
    if h_conf: reasons.append("HCO_no_on_source_peak_and_global_peak_gt20")
    if c_conf: reasons.append("C18O_no_on_source_peak_and_global_peak_gt20")
    if not confused: reasons.extend(failures)
    return cls, checks, ";".join(reasons)


def interval_crosses(row, key, threshold):
    lo, hi = fnum(row.get(key + "_p16")), fnum(row.get(key + "_p84"))
    return math.isfinite(lo) and math.isfinite(hi) and lo <= threshold <= hi


def build_classifications(measurements):
    by = {(r["source"], r["molecule"]): r for r in measurements}
    old_d = read_old_distances()
    wint = read_integrated_beam("HCO+")
    rows = []
    for source in COORDS:
        map_source = MAP_SOURCE.get(source, source)
        hrow = by[(source, "HCO+")]
        crow = by[(source, "C18O")]
        w, werr = wint.get(map_source, (math.nan, math.nan))
        dnew, dnew_err, dbasis = DIST[source]
        dold = old_d[map_source]
        told = 0.4 * 140.0 / dold
        tnew = 0.4 * 140.0 / dnew
        tnew_err = tnew * dnew_err / dnew

        old_h, old_c = fnum(hrow.get("old_c_factor_gaussian")), fnum(crow.get("old_c_factor_gaussian"))
        new_h, new_c = fnum(hrow.get("c_validated")), fnum(crow.get("c_validated"))
        bg_h, bg_c = fnum(hrow.get("c_bgsub")), fnum(crow.get("c_bgsub"))
        # Apply the same numerical gates to the sensitivity metric.
        if not (fnum(hrow.get("peak_snr")) >= 3 and 0 <= bg_h <= 1 and fnum(hrow.get("coverage")) >= 0.8): bg_h = math.nan
        if not (fnum(crow.get("peak_snr")) >= 3 and 0 <= bg_c <= 1 and fnum(crow.get("coverage")) >= 0.8): bg_c = math.nan

        old_cls, old_checks, old_reason = classify(old_h, old_c, hrow, crow, w, told)
        new_cls, new_checks, new_reason = classify(new_h, new_c, hrow, crow, w, tnew)
        bg_cls, _, bg_reason = classify(bg_h, bg_c, hrow, crow, w, tnew)

        uncertainty_flags = []
        if interval_crosses(hrow, "c", 0.6): uncertainty_flags.append("HCO_C_interval_crosses_0p6")
        if interval_crosses(crow, "c", 0.6): uncertainty_flags.append("C18O_C_interval_crosses_0p6")
        if math.isfinite(werr) and (w - werr) <= tnew <= (w + werr): uncertainty_flags.append("HCO_W_interval_crosses_threshold")
        if fnum(hrow.get("offset_p16")) <= 14 <= fnum(hrow.get("offset_p84")): uncertainty_flags.append("HCO_offset_interval_crosses_14")
        if fnum(crow.get("offset_p16")) <= 14 <= fnum(crow.get("offset_p84")): uncertainty_flags.append("C18O_offset_interval_crosses_14")
        if "provisional" in dbasis: uncertainty_flags.append("distance_provisional")
        if hrow.get("status") in ("invalid", "fit_failed") or crow.get("status") in ("invalid", "fit_failed"):
            uncertainty_flags.append("invalid_or_failed_concentration")
        additional = []
        for r, mol in ((hrow, "HCO"), (crow, "C18O")):
            if truth(r.get("has_onsource_peak")) and fnum(r.get("global_peak_snr")) >= 3 and fnum(r.get("global_offset")) > 20:
                additional.append(mol + "_additional_offsource_peak_gt20")

        row = {
            "source": source, "display_name": DISPLAY.get(source, source), "map_source": map_source,
            "old_distance_pc": dold, "new_distance_pc": dnew, "new_distance_err_pc": dnew_err,
            "distance_basis": dbasis, "W_HCO_YSO_Kkms": w, "W_HCO_err_Kkms": werr,
            "old_W_threshold_Kkms": told, "new_W_threshold_Kkms": tnew,
            "new_W_threshold_err_Kkms": tnew_err, "old_C_HCO": old_h, "old_C_C18O": old_c,
            "new_Cgas_HCO": new_h, "new_Cgas_HCO_p16": fnum(hrow.get("c_p16")), "new_Cgas_HCO_p84": fnum(hrow.get("c_p84")),
            "new_Cgas_C18O": new_c, "new_Cgas_C18O_p16": fnum(crow.get("c_p16")), "new_Cgas_C18O_p84": fnum(crow.get("c_p84")),
            "new_Cgas_bgsub_HCO": bg_h, "new_Cgas_bgsub_C18O": bg_c,
            "HCO_peak_offset_arcsec": fnum(hrow.get("offset")), "C18O_peak_offset_arcsec": fnum(crow.get("offset")),
            "HCO_global_peak_offset_arcsec": fnum(hrow.get("global_offset")), "C18O_global_peak_offset_arcsec": fnum(crow.get("global_offset")),
            "HCO_onsource_peak_snr": fnum(hrow.get("onsource_peak_snr")), "C18O_onsource_peak_snr": fnum(crow.get("onsource_peak_snr")),
            "old_classification": old_cls, "new_classification": new_cls,
            "new_bgsub_sensitivity_classification": bg_cls,
            "classification_changed": old_cls != new_cls,
            "old_failure_reason": old_reason, "new_failure_reason": new_reason,
            "bgsub_failure_reason": bg_reason, "uncertainty_flags": ";".join(uncertainty_flags),
            "additional_peak_flags": ";".join(additional),
            "HCO_measurement_flags": hrow.get("flags", ""), "C18O_measurement_flags": crow.get("flags", ""),
        }
        for prefix, checks in (("old", old_checks), ("new", new_checks)):
            for k, v in checks.items(): row[f"{prefix}_{k}"] = v
        rows.append(row)
    return rows


def save_csv(path, rows):
    fields = list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def save_text_products(outdir, measurements, rows):
    by={(r["source"],r["molecule"]):r for r in measurements}
    for molecule,tag in (("HCO+","HCOplus"),("C18O","C18O")):
        path=outdir/f"gas_centered_concentrations_{tag}.txt"
        with open(path,"w") as f:
            f.write("# source C_legacy C_gas C_p16 C_p84 C_gas_bgsub selected_offset_arcsec global_offset_arcsec onsource_peak_snr status flags\n")
            for source in COORDS:
                r=by[(source,molecule)]
                vals=[source,r.get("old_c_factor_gaussian","nan"),r.get("c_validated","nan"),r.get("c_p16","nan"),r.get("c_p84","nan"),r.get("c_bgsub","nan"),r.get("offset","nan"),r.get("global_offset","nan"),r.get("onsource_peak_snr","nan"),r.get("status",""),r.get("flags","")]
                f.write(" ".join(str(x) if str(x) else "nan" for x in vals)+"\n")
    for version in ("old","new"):
        with open(outdir/f"envelope_classifications_{version}.txt","w") as f:
            f.write("# source classification failure_reason\n")
            for r in rows:
                f.write(f"{r['source']} {r[version+'_classification']} {r[version+'_failure_reason'] or '-'}\n")


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=COLORS["grid"], linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def plot_old_new_c(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    for ax, mol, oldk, newk, lok, hik in [
        (axes[0], "HCO$^+$", "old_C_HCO", "new_Cgas_HCO", "new_Cgas_HCO_p16", "new_Cgas_HCO_p84"),
        (axes[1], "C$^{18}$O", "old_C_C18O", "new_Cgas_C18O", "new_Cgas_C18O_p16", "new_Cgas_C18O_p84"),
    ]:
        paired = [r for r in rows if math.isfinite(fnum(r[oldk])) and math.isfinite(fnum(r[newk]))]
        for cls in COLORS:
            if cls in ("neutral", "grid"): continue
            rr = [r for r in paired if r["new_classification"] == cls]
            if not rr: continue
            x=np.array([fnum(r[oldk]) for r in rr]); y=np.array([fnum(r[newk]) for r in rr])
            lo=np.array([fnum(r[lok]) for r in rr]); hi=np.array([fnum(r[hik]) for r in rr])
            ok=np.isfinite(lo)&np.isfinite(hi)
            yerr=np.vstack([np.maximum(y[ok]-lo[ok],0),np.maximum(hi[ok]-y[ok],0)])
            ax.errorbar(x[ok],y[ok],yerr=yerr,fmt="none",ecolor=COLORS[cls],alpha=.35,lw=.8)
            ax.scatter(x,y,s=48,c=COLORS[cls],edgecolor="white",linewidth=.6,label=cls,alpha=.9)
        ax.plot([-0.05,1.02],[-0.05,1.02],"--",color=COLORS["neutral"],lw=1,label="No change")
        ax.axvline(.6,color="#222",ls=":",lw=1); ax.axhline(.6,color="#222",ls=":",lw=1)
        ax.set(xlim=(-.05,1.02),ylim=(-.05,1.02),xlabel="Legacy C (YSO-anchored)",ylabel="Revised $C_{gas}$",title=mol)
        style_ax(ax)
    axes[1].legend(frameon=False,fontsize=8,loc="lower right")
    fig.suptitle("Legacy versus revised gas-centered concentration",fontsize=14,fontweight="bold")
    fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig)


def plot_delta(rows, out):
    fig, axes = plt.subplots(1,2,figsize=(13,10),sharey=True,constrained_layout=True)
    order=sorted(rows,key=lambda r:DISPLAY.get(r["source"],r["source"]))
    y=np.arange(len(order))
    for ax,oldk,newk,title in [(axes[0],"old_C_HCO","new_Cgas_HCO","HCO$^+$"),(axes[1],"old_C_C18O","new_Cgas_C18O","C$^{18}$O")]:
        vals=[]
        for r in order:
            o,n=fnum(r[oldk]),fnum(r[newk]); vals.append(n-o if math.isfinite(o) and math.isfinite(n) else math.nan)
        vals=np.array(vals)
        ax.axvline(0,color="#222",lw=1)
        for i,v in enumerate(vals):
            if math.isfinite(v):
                ax.plot([0,v],[i,i],color=COLORS["neutral"],lw=1)
                ax.scatter(v,i,s=34,facecolor=COLORS["Embedded"] if v>=0 else "white",edgecolor=COLORS["Embedded"],linewidth=1)
        ax.set(xlabel="$C_{gas}-C_{legacy}$",title=title,xlim=(-.75,.75)); style_ax(ax)
    axes[0].set_yticks(y,labels=[r["display_name"] for r in order],fontsize=7)
    fig.suptitle("Change in concentration by source",fontsize=14,fontweight="bold")
    fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig)


def plot_criteria(rows, out):
    order=sorted(rows,key=lambda r:r["display_name"])
    criteria=[("HCO peak ≤14″","new_HCO_peak_within14"),("C18O peak ≤14″","new_C18O_peak_within14"),("C(HCO)>0.6","new_C_HCO_gt0p6"),("C(C18O)>0.6","new_C_C18O_gt0p6"),("W(HCO)>limit","new_W_HCO_gt_scaled_threshold")]
    arr=np.array([[1 if str(r[k]).lower()=="true" else 0 for _,k in criteria] for r in order])
    fig,ax=plt.subplots(figsize=(8.5,11),constrained_layout=True)
    from matplotlib.colors import ListedColormap
    ax.imshow(arr,aspect="auto",cmap=ListedColormap(["#F2D7C9","#CFE2EF"]),vmin=0,vmax=1)
    ax.set_xticks(range(len(criteria)),[x[0] for x in criteria],rotation=25,ha="right")
    ax.set_yticks(range(len(order)),[r["display_name"] for r in order],fontsize=7)
    for i,r in enumerate(order):
        ax.text(len(criteria)+.15,i,r["new_classification"],va="center",fontsize=7,color=COLORS[r["new_classification"]])
    ax.set_xlim(-.5,len(criteria)+1.8)
    ax.set_title("Revised envelope criteria by source",fontweight="bold",pad=12)
    ax.set_xlabel("Blue = criterion passed; peach = failed or unavailable")
    fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig)


def plot_transitions(rows, out):
    cats=["Embedded","Non-embedded","Confused"]
    mat=np.zeros((3,3),int)
    for r in rows: mat[cats.index(r["old_classification"]),cats.index(r["new_classification"])]+=1
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),constrained_layout=True)
    ax=axes[0]; im=ax.imshow(mat,cmap="Blues",vmin=0,vmax=max(mat.max(),1))
    for i in range(3):
        for j in range(3): ax.text(j,i,str(mat[i,j]),ha="center",va="center",fontsize=13,fontweight="bold")
    ax.set_xticks(range(3),cats,rotation=25,ha="right"); ax.set_yticks(range(3),cats)
    ax.set_xlabel("Revised classification"); ax.set_ylabel("Legacy classification"); ax.set_title("Classification transition matrix")
    counts_old=[sum(r["old_classification"]==c for r in rows) for c in cats]
    counts_new=[sum(r["new_classification"]==c for r in rows) for c in cats]
    x=np.arange(3); w=.36
    axes[1].bar(x-w/2,counts_old,w,label="Legacy",color="white",edgecolor=COLORS["neutral"],linewidth=1.3)
    axes[1].bar(x+w/2,counts_new,w,label="Revised",color=[COLORS[c] for c in cats],edgecolor="white")
    axes[1].set_xticks(x,cats,rotation=25,ha="right"); axes[1].set_ylabel("Number of sources"); axes[1].set_title("Class counts"); axes[1].legend(frameon=False)
    style_ax(axes[1]); fig.suptitle("Legacy and revised envelope classifications",fontsize=14,fontweight="bold")
    fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig)


def plot_offset_concentration(rows, out):
    fig,axes=plt.subplots(1,2,figsize=(11.5,5),constrained_layout=True)
    for ax,mol,ck,ok in [(axes[0],"HCO$^+$","new_Cgas_HCO","HCO_peak_offset_arcsec"),(axes[1],"C$^{18}$O","new_Cgas_C18O","C18O_peak_offset_arcsec")]:
        for cls in ("Embedded","Non-embedded","Confused"):
            rr=[r for r in rows if r["new_classification"]==cls and math.isfinite(fnum(r[ck])) and math.isfinite(fnum(r[ok]))]
            ax.scatter([fnum(r[ok]) for r in rr],[fnum(r[ck]) for r in rr],s=48,c=COLORS[cls],edgecolor="white",linewidth=.6,label=cls)
        ax.axvline(14,color="#222",ls="--",lw=1,label="One beam (14″)"); ax.axvline(20,color="#222",ls=":",lw=1,label="Confusion guide (20″)")
        ax.axhline(.6,color="#222",ls="-.",lw=1,label="C = 0.6")
        ax.set(xlabel="Selected gas-peak offset from YSO (arcsec)",ylabel="$C_{gas}$",title=mol,xlim=(-1,62),ylim=(-.05,1.02)); style_ax(ax)
    axes[1].legend(frameon=False,fontsize=7,loc="lower right")
    fig.suptitle("Concentration and source association carry different information",fontsize=14,fontweight="bold")
    fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig)


def plot_intensity_margin(rows, out):
    order=sorted(rows,key=lambda r:fnum(r["W_HCO_YSO_Kkms"])/fnum(r["new_W_threshold_Kkms"]) if fnum(r["new_W_threshold_Kkms"])>0 else -99)
    ratio=np.array([fnum(r["W_HCO_YSO_Kkms"])/fnum(r["new_W_threshold_Kkms"]) for r in order])
    y=np.arange(len(order)); colors=[COLORS[r["new_classification"]] for r in order]
    fig,ax=plt.subplots(figsize=(9,10.5),constrained_layout=True)
    ax.barh(y,ratio,color=colors,edgecolor="white"); ax.axvline(1,color="#222",ls="--",lw=1.2,label="Adopted threshold")
    ax.set_yticks(y,[r["display_name"] for r in order],fontsize=7); ax.set_xscale("symlog",linthresh=.1)
    ax.set_xlabel(r"$W_{HCO^+,YSO}/[0.4(140\,pc/d)]$"); ax.set_title("YSO-centered HCO$^+$ intensity relative to distance-scaled threshold",fontweight="bold")
    style_ax(ax); ax.legend(frameon=False)
    fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--measurements",type=Path,required=True); ap.add_argument("--outdir",type=Path,required=True)
    args=ap.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)
    measurements=read_csv(args.measurements); rows=build_classifications(measurements)
    save_csv(args.outdir/"envelope_classification_old_vs_new.csv",rows)
    save_text_products(args.outdir,measurements,rows)
    plot_old_new_c(rows,args.outdir/"01_old_vs_new_concentration.png")
    plot_delta(rows,args.outdir/"02_concentration_change_by_source.png")
    plot_criteria(rows,args.outdir/"03_revised_criteria_by_source.png")
    plot_transitions(rows,args.outdir/"04_classification_transitions.png")
    plot_offset_concentration(rows,args.outdir/"05_offset_vs_concentration.png")
    plot_intensity_margin(rows,args.outdir/"06_hco_intensity_threshold.png")


if __name__ == "__main__":
    main()
