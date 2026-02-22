#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import NormalDist

# ----------------------------
# Configuration
# ----------------------------
DATA_PATH = "text_files/combined_yso_parameters_v2.xlsx"

STATUS_COL = "Envelope_status"   # "yes", "no", "unclear"
X_COL = "gravity"                # stored as 100 * log g in your file

STATUS_MAP = {"yes": 1, "no": 0} # drop "unclear"

BIN_WIDTH = 0.25                  # dex
COLOR_ENV = "#bebada"


# ----------------------------
# Wilson binomial confidence interval
# ----------------------------
def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan

    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom

    lo = np.clip(center - half_width, 0.0, 1.0)
    hi = np.clip(center + half_width, 0.0, 1.0)
    return lo, hi


# ----------------------------
# Load and prepare data
# ----------------------------
def load_and_prepare(path):
    df = pd.read_excel(path)

    # normalize envelope status
    df[STATUS_COL] = (
        df[STATUS_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan})
    )

    # map to binary and drop unclear
    df["y"] = df[STATUS_COL].map(STATUS_MAP)
    df = df[df["y"].isin([0, 1])].copy()

    # convert gravity to log g
    df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce") / 1e2
    df = df[np.isfinite(df[X_COL])]

    return df


# ----------------------------
# Plot binned envelope fraction
# ----------------------------
def plot_binned_envelope_fraction(df, bin_width=BIN_WIDTH, alpha=0.05):

    x = df[X_COL].to_numpy()
    y = df["y"].to_numpy()

    bins = np.arange(x.min(), x.max() + bin_width, bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])

    frac = []
    err_lo = []
    err_hi = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x < hi)
        n = mask.sum()

        if n == 0:
            frac.append(np.nan)
            err_lo.append(np.nan)
            err_hi.append(np.nan)
            continue

        k = int(y[mask].sum())
        f = k / n
        ci_lo, ci_hi = wilson_ci(k, n, alpha=alpha)

        frac.append(f)
        err_lo.append(max(0.0, f - ci_lo))
        err_hi.append(max(0.0, ci_hi - f))

    frac = np.array(frac)
    err_lo = np.array(err_lo)
    err_hi = np.array(err_hi)

    ok = np.isfinite(frac)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        centers[ok],
        frac[ok],
        yerr=[err_lo[ok], err_hi[ok]],
        fmt="o",
        ms=10,
        lw=2,
        capsize=4,
        color=COLOR_ENV,
        label="Envelope fraction",
    )

    ax.set_xlabel(r"$\log g$", fontsize=16)
    ax.set_ylabel("Envelope fraction", fontsize=16)
    ax.set_xlim(2.65, 3.92)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(frameon=False, fontsize=12)

    fig.tight_layout()
    fig.savefig("binned_envelope_fraction_vs_gravity.png", dpi=300)
    plt.show()


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    df_model = load_and_prepare(DATA_PATH)
    plot_binned_envelope_fraction(df_model)
