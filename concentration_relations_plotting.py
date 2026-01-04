import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

FILE = "text_files/combined_yso_parameters_v2.xlsx"
POINT_SIZE = 150
ALPHA = 0.85
SCALE_GRAVITY_BY_100 = True   # set True if you want gravity/100
COLOR_RANGE = (0.3, 1.0)       # fixed color scale for concentration

def plot_gravity_vs_molecule(df: pd.DataFrame, molecule: str, title_suffix=""):
    """
    molecule: 'HCOplus' or 'C18O'
    y-axis uses spec_{molecule}__Integ_Beam
    color uses conc_{molecule}__c_factor_gaussian
    """
    mol = molecule  # exact token used in your columns
    y_col = f"spec_{mol}__Integ_Beam"
    c_col = f"conc_{mol}__c_factor_gaussian"

    # Pull columns safely
    gx = pd.to_numeric(df.get("gravity"), errors="coerce")
    gy = pd.to_numeric(df.get(y_col), errors="coerce")
    gc = pd.to_numeric(df.get(c_col), errors="coerce")

    if SCALE_GRAVITY_BY_100:
        gx = gx / 100.0

    # Valid points for plotting (x & y must exist)
    mask_xy = gx.notna() & gy.notna()

    # Split by whether we have concentration
    mask_with_c = mask_xy & gc.notna()
    mask_no_c  = mask_xy & gc.isna()

    plt.figure(figsize=(8, 6))

    # Plot points with no conc in black
    plt.scatter(
        gx[mask_no_c], gy[mask_no_c],
        s=POINT_SIZE, c="black", alpha=ALPHA, edgecolor="none",
        # label="no concentration value"
    )

    # Plot points with conc using fixed color scale
    if mask_with_c.any():
        vmin, vmax = COLOR_RANGE
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("gist_rainbow")

        sc = plt.scatter(
            gx[mask_with_c], gy[mask_with_c],
            s=POINT_SIZE, c=gc[mask_with_c], cmap=cmap, norm=norm,
            alpha=ALPHA, edgecolor="k",
            # label=f"with {mol}"
        )

        cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap))
        cbar.set_label(f"{title_suffix} concentration factor ",size=16)

    plt.xlabel("log(g)",size=16)# + (" (×0.01)" if SCALE_GRAVITY_BY_100 else ""))
    plt.ylabel(title_suffix+' integrated intensity (K km/s)',size=16)
    ttl = f"gravity vs. {y_col} colored by {c_col}"
    if title_suffix:
        ttl += f" — {title_suffix}"
    # plt.title(ttl)
    # plt.legend(loc="upper left")
    plt.tick_params(axis='both', labelsize=14)  # 14 = size in points

    plt.tight_layout()
    plt.savefig(f"{mol}_gravity_comparison.png", dpi=150)
    plt.show()

def main():
    df = pd.read_excel(FILE)

    # Choose one:
    # plot_gravity_vs_molecule(df, "HCOplus", title_suffix="HCO⁺")
    # Or:
    plot_gravity_vs_molecule(df, "C18O", title_suffix="C¹⁸O")

    # If you want BOTH in one go:
    # for mol in ["HCOplus", "C18O"]:
    #     plot_gravity_vs_molecule(df, mol, title_suffix=mol)

if __name__ == "__main__":
    main()
