import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE = "text_files/combined_yso_parameters_v2.xlsx"

POINT_SIZE = 180
ALPHA = 0.95
SCALE_GRAVITY_BY_100 = True   # gravity/100 if True

# ----------------------------
# Helpers
# ----------------------------

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _gravity(df: pd.DataFrame) -> pd.Series:
    gx = _to_num(df.get("gravity"))
    if SCALE_GRAVITY_BY_100:
        gx = gx / 100.0
    return gx

def _envelope_stat(df: pd.DataFrame) -> pd.Series:
    e_s = df.get("Envelope_status")
    return e_s


def find_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def intensity_cols_for_molecule(df: pd.DataFrame, molecule: str):
    """
    Returns (det_col, ul_col) for the integrated intensity-like values:
      det_col ~ sum_Kkms_arcsec2
      ul_col  ~ W_sum_UL_Kkms_beamsu
    Tries multiple naming conventions.
    """
    # molecule token variations you might have in your excel
    if molecule == "HCOplus":
        tokens = ["HCOplus", "HCO+", "HCOp", "HCOp"]
    elif molecule == "C18O":
        tokens = ["C18O", "C18o", "c18o"]
    else:
        tokens = [molecule]

    det_candidates = []
    ul_candidates = []

    for t in tokens:
        # Common ways this could have been appended/renamed
        det_candidates += [
            f"{t}_sum_Kkms_arcsec2",
            f"sum_Kkms_arcsec2_{t}",
            f"{t}:sum_Kkms_arcsec2",
            f"{t} sum_Kkms_arcsec2",
        ]
        ul_candidates += [
            f"{t}_W_sum_UL_Kkms_beamsu",
            f"W_sum_UL_Kkms_beamsu_{t}",
            f"{t}:W_sum_UL_Kkms_beamsu",
            f"{t} W_sum_UL_Kkms_beamsu",
        ]

    # Also support the exact column names from the Excel I generated earlier
    if molecule == "HCOplus":
        det_candidates = ["HCOp_sum_Kkms_arcsec2"] + det_candidates
        ul_candidates  = ["UL_HCOp_W_sum_UL_Kkms_beamsu"] + ul_candidates
    if molecule == "C18O":
        det_candidates = ["C18O_sum_Kkms_arcsec2"] + det_candidates
        ul_candidates  = ["UL_C18O_W_sum_UL_Kkms_beamsu"] + ul_candidates

    det_col = find_first_existing_col(df, det_candidates)
    ul_col  = find_first_existing_col(df, ul_candidates)

    return det_col, ul_col

def concentration_col_for_molecule(df: pd.DataFrame, molecule: str) -> str:
    """
    conc_{mol}__c_factor_gaussian in your existing file.
    molecule expected: 'HCOplus' or 'C18O'
    """
    c_col = f"conc_{molecule}__c_factor_gaussian"
    if c_col not in df.columns:
        raise KeyError(f"Missing concentration column: {c_col}")
    return c_col

def plot_scatter_basic(x, y, xlabel, ylabel, filename, title=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=POINT_SIZE, alpha=ALPHA, edgecolor="none")
    plt.xlabel(xlabel, size=16)
    plt.ylabel(ylabel, size=16)
    if title:
        plt.title(title)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ----------------------------
# (1) gravity vs concentration (no colors)
# ----------------------------

def plot_gravity_vs_concentration(
    df: pd.DataFrame,
    molecule: str,
    title_suffix="",
    envelope_coloring: bool = True,
):
    gx = _gravity(df)

    c_col = concentration_col_for_molecule(df, molecule)
    gy = _to_num(df[c_col])

    mask = gx.notna() & gy.notna()

    plt.figure(figsize=(8, 6))

    if envelope_coloring:
        e_s = _envelope_stat(df)
        e_s = e_s.astype(str).str.strip().str.lower()

        env_colors = {"yes": "#bebada", "no": "#ffffb3", "unclear": "#8dd3c7"}

        for status, color in env_colors.items():
            m = mask & (e_s == status)
            if not m.any():
                continue
            plt.scatter(
                gx[m], gy[m],
                s=POINT_SIZE,
                c=color,
                alpha=ALPHA,
                edgecolor="none",
                label=status,
                ec='k'
            )

        plt.legend(title="Envelope", fontsize=12, title_fontsize=12, loc="best")
    else:
        # fallback: single-color plot
        plt.scatter(
            gx[mask], gy[mask],
            s=POINT_SIZE,
            alpha=ALPHA,
            edgecolor="none",
        )

    plt.xlabel("log(g)", size=16)
    plt.ylabel(f"{title_suffix} conc factor (gaussian)", size=16)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.savefig(f"{molecule}_gravity_vs_concentration_envelopeColored.png", dpi=150)
    plt.show()


# ----------------------------
# (2) gravity vs sum_Kkms_arcsec2 with UL arrows if needed
# ----------------------------

def plot_gravity_vs_integrated_sum_with_UL(
    df: pd.DataFrame,
    molecule: str,
    title_suffix="",
    divide_by_binarity: bool = False,
    binarity_col: str = "Binarity",
):
    gx = _gravity(df)

    # Envelope status series: 'yes'/'no'/'unclear'
    e_s = _envelope_stat(df)
    e_s = e_s.astype(str).str.strip().str.lower()

    env_colors = {"yes": "#bebada", "no": "#ffffb3", "unclear": "#8dd3c7"}

    det_col, ul_col = intensity_cols_for_molecule(df, molecule)
    if det_col is None and ul_col is None:
        raise KeyError(f"Could not find intensity columns for molecule={molecule}")

    det = _to_num(df[det_col]) if det_col else pd.Series(np.nan, index=df.index)
    ul  = _to_num(df[ul_col])  if ul_col  else pd.Series(np.nan, index=df.index)

    # Divide by binarity (per-star) if requested
    if divide_by_binarity:
        b = pd.to_numeric(df[binarity_col], errors="coerce")
        valid_b = b.notna() & (b > 0)
        det = det.where(~valid_b, det / b)
        ul  = ul.where(~valid_b,  ul  / b)

    # Masks
    mask_det = gx.notna() & det.notna()
    # UL only where detection missing
    mask_ul  = gx.notna() & det.isna() & ul.notna()

    plt.figure(figsize=(8, 6))

    # --- detections, colored by envelope status ---
    for status, color in env_colors.items():
        m = mask_det & (e_s == status)
        if not m.any():
            continue

        plt.scatter(
            gx[m], det[m],
            s=POINT_SIZE,
            c=color,
            alpha=ALPHA,
            edgecolor="none",
            label=status,
            ec='k'
        )

    # --- upper limits as downward arrows, also colored ---
    for status, color in env_colors.items():
        m = mask_ul & (e_s == status)
        if not m.any():
            continue

        yerr = 0.15 * ul[m].values
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, 0.1)

        (_, caplines, _) = plt.errorbar(
            gx[m].values,
            ul[m].values,
            yerr=yerr,
            fmt="o",
            marker='v',
            color=color,
            alpha=ALPHA,
            uplims=True,
            capsize=1,
            markersize=10,
            mec='k',
            # ecolor="k",  # error bars + uplims arrows
            # elinewidth=0.1,
            # avoid duplicate legend entries if detections already added
            label=f"{status} (UL)" if not (mask_det & (e_s == status)).any() else None,
        )

    # caplines[0].set_markersize(10)
    plt.xlabel("log(g)", size=16)
    ylab = f"{title_suffix} integrated intensity" + r"$\rm (K \, km/s \, arcsec^2)$"
    # if divide_by_binarity:
    #     ylab += f" / {binarity_col}"
    plt.ylabel(ylab, size=16)

    plt.tick_params(axis="both", labelsize=14)
    plt.legend(title="Envelope", fontsize=12, title_fontsize=12, loc="best")
    plt.tight_layout()

    suffix = "_per_star" if divide_by_binarity else ""
    plt.savefig(f"{molecule}_gravity_vs_sum_with_UL{suffix}_envelopeColored.png", dpi=150)
    plt.show()


# ----------------------------
# (3) gravity vs ratio Ic18O/IHco+
#     Uses detections when available, else UL.
#     Handles limit cases:
#       - UL numerator + det denom -> upper limit on ratio (down arrow)
#       - det numerator + UL denom -> lower limit on ratio (up arrow)
#       - UL both -> ambiguous; plotted as plain points (you can choose to skip)
# ----------------------------

def plot_gravity_vs_ratio_C18O_over_HCOplus(df: pd.DataFrame):
    gx = _gravity(df)

    c_det_col, c_ul_col = intensity_cols_for_molecule(df, "C18O")
    h_det_col, h_ul_col = intensity_cols_for_molecule(df, "HCOplus")

    if (c_det_col is None and c_ul_col is None) or (h_det_col is None and h_ul_col is None):
        raise KeyError("Could not find intensity columns needed for ratio plot (C18O and HCOplus).")

    C_det = _to_num(df[c_det_col]) if c_det_col else pd.Series(np.nan, index=df.index)
    C_ul  = _to_num(df[c_ul_col])  if c_ul_col  else pd.Series(np.nan, index=df.index)
    H_det = _to_num(df[h_det_col]) if h_det_col else pd.Series(np.nan, index=df.index)
    H_ul  = _to_num(df[h_ul_col])  if h_ul_col  else pd.Series(np.nan, index=df.index)

    # Cases
    mask_dd = gx.notna() & C_det.notna() & H_det.notna()  # detection/detection
    mask_ud = gx.notna() & C_det.isna() & C_ul.notna() & H_det.notna()  # UL C18O, det HCO => ratio UL
    mask_du = gx.notna() & C_det.notna() & H_det.isna() & H_ul.notna()  # det C18O, UL HCO => ratio LL
    mask_uu = gx.notna() & C_det.isna() & C_ul.notna() & H_det.isna() & H_ul.notna()  # both UL (ambiguous)

    plt.figure(figsize=(8, 6))

    # dd
    if mask_dd.any():
        ratio = C_det[mask_dd] / H_det[mask_dd]
        plt.scatter(gx[mask_dd], ratio, s=POINT_SIZE, alpha=ALPHA, edgecolor="none")

    # UL on ratio (down arrow): (C_ul / H_det)
    if mask_ud.any():
        ratio_ul = C_ul[mask_ud] / H_det[mask_ud]
        yerr = 0.15 * ratio_ul.values
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, 0.1)
        plt.errorbar(
            gx[mask_ud].values,
            ratio_ul.values,
            yerr=yerr,
            fmt="o",
            alpha=ALPHA,
            uplims=True,
            capsize=0,
            markersize=6,
        )

    # LL on ratio (up arrow): (C_det / H_ul)
    if mask_du.any():
        ratio_ll = C_det[mask_du] / H_ul[mask_du]
        yerr = 0.15 * ratio_ll.values
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, 0.1)
        plt.errorbar(
            gx[mask_du].values,
            ratio_ll.values,
            yerr=yerr,
            fmt="o",
            alpha=ALPHA,
            lolims=True,   # upward arrow = lower limit
            capsize=0,
            markersize=6,
        )

    # both UL: plot as plain points at (C_ul/H_ul) (optional)
    if mask_uu.any():
        ratio_guess = C_ul[mask_uu] / H_ul[mask_uu]
        plt.scatter(gx[mask_uu], ratio_guess, s=POINT_SIZE, alpha=0.35, edgecolor="none")

    plt.xlabel("log(g)", size=16)
    plt.ylabel("I(C¹⁸O) / I(HCO⁺)  (det or limits)", size=16)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    plt.savefig("gravity_vs_ratio_Ic18O_over_IHCOplus.png", dpi=150)
    plt.show()


# ----------------------------
# (4) gravity vs Mgas_Msun_* with median-of-Mgas columns + asym. error bars
#     nominal = median across Mgas columns per source
#     err_low  = median - min
#     err_high = max - median
# ----------------------------


def plot_gravity_vs_mgas_median(
    df: pd.DataFrame,
    divide_by_binarity: bool = False,
    binarity_col: str = "Binarity",
):
    gx = _gravity(df)

    # Envelope status series: values 'yes'/'no'/'unclear'
    e_s = _envelope_stat(df)
    e_s = e_s.astype(str).str.strip().str.lower()

    # Color map you want
    env_colors = {"yes": "#bebada", "no": "#ffffb3", "unclear": "#8dd3c7"}

    mgas_cols = [c for c in df.columns if c.startswith("Mgas_Msun")]
    if not mgas_cols:
        raise KeyError("No columns found starting with 'Mgas_Msun'")

    ul_cols = [
        "Mgas_UL_Tex10K_Msun",
        "Mgas_UL_Tex20K_Msun",
        "Mgas_UL_Tex30K_Msun",
        "Mgas_UL_Tex40K_Msun",
    ]
    ul_cols = [c for c in ul_cols if c in df.columns]

    mg = df[mgas_cols].apply(_to_num)

    nominal = mg.median(axis=1, skipna=True)
    vmin    = mg.min(axis=1, skipna=True)
    vmax    = mg.max(axis=1, skipna=True)

    ul = df[ul_cols].apply(_to_num).max(axis=1, skipna=True) if ul_cols else pd.Series(np.nan, index=df.index)

    # Divide by binarity if requested (apply consistently to nominal, bounds, and UL)
    if divide_by_binarity:
        b = pd.to_numeric(df[binarity_col], errors="coerce")
        valid_b = b.notna() & (b > 0)

        nominal = nominal.where(~valid_b, nominal / b)
        vmin    = vmin.where(~valid_b,    vmin    / b)
        vmax    = vmax.where(~valid_b,    vmax    / b)
        ul      = ul.where(~valid_b,      ul      / b)

    mask_det = gx.notna() & nominal.notna()
    mask_ul_only = gx.notna() & nominal.isna() & ul.notna()

    plt.figure(figsize=(8, 6))

    # --- detections with asymmetric error bars, color-coded by envelope status ---
    for status, color in env_colors.items():
        m = mask_det & (e_s == status)
        if not m.any():
            continue

        err_low = (nominal - vmin)[m].values
        err_hi  = (vmax - nominal)[m].values

        plt.errorbar(
            gx[m].values,
            nominal[m].values,
            yerr=np.vstack([err_low, err_hi]),
            fmt="o",
            color=color,
            alpha=ALPHA,
            capsize=3,
            markersize=12,
            label=status,
            mec='k'
        )

    # --- UL-only as downward arrows, also color-coded ---
    for status, color in env_colors.items():
        m = mask_ul_only & (e_s == status)
        if not m.any():
            continue

        yerr = 0.15 * ul[m].values
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, 0.1)

        plt.errorbar(
            gx[m].values,
            ul[m].values,
            yerr=yerr,
            fmt="o",
            marker='v',
            color=color,
            alpha=ALPHA,
            uplims=True,
            capsize=0,
            markersize=12,
            mec='k',
            label=f"{status} (UL)" if not (mask_det & (e_s == status)).any() else None,
        )

    plt.xlabel("log(g)", size=16)
    ylab = "Gas mass within 3000 au radius "+ r"$\rm (M_\odot)$"
    # if divide_by_binarity:
    #     ylab += f" / {binarity_col}"
    # ylab += " (UL as arrows)"
    plt.ylabel(ylab, size=16)

    plt.tick_params(axis="both", labelsize=14)
    plt.legend(title="Envelope", fontsize=12, title_fontsize=12, loc="best")
    plt.tight_layout()

    suffix = "_per_star" if divide_by_binarity else ""
    plt.savefig(f"gravity_vs_Mgas_median_with_errors_and_UL{suffix}_envelopeColored.png", dpi=150)
    plt.show()


# ----------------------------
# (5) Temperature vs Mgas_Msun_* with median-of-Mgas columns + asym. error bars
#     nominal = median across Mgas columns per source
#     err_low  = median - min
#     err_high = max - median
# ----------------------------


def plot_parameter_vs_mgas_median_with_UL(
    df: pd.DataFrame,
    param_col: str = "Temperature",
    divide_by_binarity: bool = True,
    binarity_col: str = "Binarity",
    envelope_coloring: bool = True,
):
    # X axis
    tx = pd.to_numeric(df[param_col], errors="coerce")

    # Envelope status
    if envelope_coloring:
        e_s = _envelope_stat(df).astype(str).str.strip().str.lower()
        env_colors = {"yes": "#bebada", "no": "#ffffb3", "unclear": "#8dd3c7"}
    else:
        e_s = pd.Series("all", index=df.index)
        env_colors = {"all": "black"}

    # Detection mass columns
    mgas_cols = [c for c in df.columns if c.startswith("Mgas_Msun")]
    if not mgas_cols:
        raise KeyError("No detection columns found starting with 'Mgas_Msun'")

    # UL mass columns
    ul_cols = [
        "Mgas_UL_Tex10K_Msun",
        "Mgas_UL_Tex20K_Msun",
        "Mgas_UL_Tex30K_Msun",
        "Mgas_UL_Tex40K_Msun",
    ]
    ul_cols = [c for c in ul_cols if c in df.columns]

    mg = df[mgas_cols].apply(_to_num)

    nominal = mg.median(axis=1, skipna=True)
    vmin    = mg.min(axis=1, skipna=True)
    vmax    = mg.max(axis=1, skipna=True)

    ul = df[ul_cols].apply(_to_num).max(axis=1, skipna=True) if ul_cols else pd.Series(np.nan, index=df.index)

    # Divide by binarity (per star) if requested
    if divide_by_binarity:
        b = pd.to_numeric(df[binarity_col], errors="coerce")
        valid_b = b.notna() & (b > 0)

        nominal = nominal.where(~valid_b, nominal / b)
        vmin    = vmin.where(~valid_b,    vmin    / b)
        vmax    = vmax.where(~valid_b,    vmax    / b)
        ul      = ul.where(~valid_b,      ul      / b)

    # Masks: detections and UL-only
    mask_det = tx.notna() & nominal.notna()
    mask_ul_only = tx.notna() & nominal.isna() & ul.notna()

    plt.figure(figsize=(8, 6))

    # --- detections (with asymmetric error bars), colored by envelope ---
    for status, color in env_colors.items():
        m = mask_det & (e_s == status)
        if not m.any():
            continue

        err_low = (nominal - vmin)[m].values
        err_hi  = (vmax - nominal)[m].values

        plt.errorbar(
            tx[m].values,
            nominal[m].values,
            yerr=np.vstack([err_low, err_hi]),
            fmt="o",
            color=color,
            alpha=ALPHA,
            capsize=3,
            markersize=12,
            mec='k',
            label=status,
        )

    # --- UL-only (down arrows), colored by envelope ---
    for status, color in env_colors.items():
        m = mask_ul_only & (e_s == status)
        if not m.any():
            continue

        yerr = 0.15 * ul[m].values
        yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, 0.1)

        plt.errorbar(
            tx[m].values,
            ul[m].values,
            yerr=yerr,
            fmt="o",
            marker='v',
            color=color,
            alpha=ALPHA,
            uplims=True,
            capsize=0,
            markersize=12,
            mec='k',
            label=f"{status} (UL)" if not (mask_det & (e_s == status)).any() else None,
        )

    plt.xlabel(param_col, size=16)
    ylab = "Mgas (Msun) median ± max deviation"
    if divide_by_binarity:
        ylab += f" / {binarity_col}"
    ylab += " (UL as arrows)"
    plt.ylabel(ylab, size=16)

    plt.tick_params(axis="both", labelsize=14)
    if envelope_coloring:
        plt.legend(title="Envelope", fontsize=12, title_fontsize=12, loc="best")
    plt.tight_layout()

    suffix = "_per_star" if divide_by_binarity else ""
    # plt.savefig(f"temperature_vs_Mgas_median_with_UL{suffix}_envelopeColored.png", dpi=150)
    plt.show()


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    df = pd.read_excel(FILE)

    # print([c for c in df.columns if "sum_Kkms" in c or "W_sum_UL" in c or c.startswith("Mgas_Msun")])

    # (1) gravity vs concentration (no colors)
    # plot_gravity_vs_concentration(df, "C18O", title_suffix="C¹⁸O")
    # plot_gravity_vs_concentration(df, "HCOplus", title_suffix="HCO⁺")
    #
    # # (2) gravity vs sum_Kkms_arcsec2, UL arrows when sum missing
    # plot_gravity_vs_integrated_sum_with_UL(df, "C18O", title_suffix="C¹⁸O", divide_by_binarity=True)
    # plot_gravity_vs_integrated_sum_with_UL(df, "C18O", title_suffix="C¹⁸O", divide_by_binarity=False)

    # plot_gravity_vs_integrated_sum_with_UL(df, "HCOplus", title_suffix="HCO⁺", divide_by_binarity=True)
    # plot_gravity_vs_integrated_sum_with_UL(df, "HCOplus", title_suffix="HCO⁺", divide_by_binarity=False)

    #
    # # (3) gravity vs Ic18O/IHco+
    # plot_gravity_vs_ratio_C18O_over_HCOplus(df)
    #
    # # (4) gravity vs Mgas_Msun_* with median + max-deviation error bars
    # plot_gravity_vs_mgas_median(df, divide_by_binarity=True)

    # # (5) Temperature vs Mgas_Msun_* with median + max-deviation error bars

    plot_parameter_vs_mgas_median_with_UL(
        df,
        param_col="Tbol",
        divide_by_binarity=True,      # per-star
        binarity_col="Binarity",
        envelope_coloring=True
    )