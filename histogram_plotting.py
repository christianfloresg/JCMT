import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import sem, ks_2samp
import os

# ----------------------------
# Load and prepare data
# ----------------------------
def load_data(filepath):
    df = pd.read_excel(filepath)

    required_cols = [
        "SourceName",
        "gravity",
        "conc_HCOplus__c_factor_gaussian",
        "spec_HCOplus__Integ_Beam",
        "indices__spectral_index", ## These are old values from different literature sources
        "indices__corr_spectral_index_lit", ## These are old values from different literature sources
        "spectral_index_SED", ## These are values calculated by SED
        "spectral_index_corrected_SED",
        "Tbol",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df

# ----------------------------
# Classification
# ----------------------------
def classify_sources(df):
    cf_gauss = pd.to_numeric(df["conc_HCOplus__c_factor_gaussian"], errors="coerce")
    integ_beam = pd.to_numeric(df["spec_HCOplus__Integ_Beam"], errors="coerce")

    envelope_mask = (cf_gauss > 0.6) & (integ_beam > 0.3)
    group = np.where(envelope_mask, "envelope", "envelope_free")

    exceptions = {"EC92", "IRAS05379-0758"}
    df_upper = df["SourceName"].fillna("").str.upper().str.strip()
    group = np.where(df_upper.isin(exceptions), "confused", group)

    df["group"] = group
    return df


# ----------------------------
# Split into groups
# ----------------------------
def split_groups(df):
    cols_show = [
        "SourceName",
        "gravity",
        "conc_HCOplus__c_factor_gaussian",
        "spec_HCOplus__Integ_Beam",
        "indices__spectral_index",
        "indices__corr_spectral_index_lit",
        "spectral_index_SED",  ## These are values calculated by SED
        "spectral_index_corrected_SED",
        "Tbol",
    ]

    return {
        "envelope": df[df["group"] == "envelope"][cols_show],
        "envelope_free": df[df["group"] == "envelope_free"][cols_show],
        "confused": df[df["group"] == "confused"][cols_show],
    }


# ----------------------------
# Save group CSVs
# ----------------------------
def save_groups(groups, outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    for name, subdf in groups.items():
        path = outdir / f"{name}_sources.csv"
        subdf.to_csv(path, index=False)

        print(f"\n{name.upper()} sources ({len(subdf)}):")
        print(subdf["SourceName"].tolist() or f"(no {name} sources)")
        print(f"→ saved to {path}")


# ----------------------------
# Plot histogram
# ----------------------------
def plot_histogram(groups, outdir, property ="gravity"):
    # bins = np.arange(-1.4, 1.1, 0.20)
    # bins = np.arange(-1.8, 2.0, 0.20)
    # bins = np.arange(50, 1600, 100)
    bins = np.arange(260, 400, 10)

    colors = {
        "envelope": "C1",
        "envelope_free": "C0",
        "confused": "C2",
    }
    hatches = {
        "envelope": ".",
        "envelope_free": None,
        "confused": "x",
    }

    plt.figure(figsize=(7, 6))

    for name, subdf in groups.items():
        data = pd.to_numeric(
            subdf[property], errors="coerce").dropna()
        plt.hist(
            data,
            bins=bins,
            fill=False,
            hatch=hatches[name],
            label=f"{name} (n={len(data)})",
            edgecolor=colors[name],
            color=colors[name],
            lw=2,
            alpha=0.6,
        )

    plt.xlabel(property, size=14)
    plt.ylabel("count", size=14)
    plt.legend(fontsize=12)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    fig_path = os.path.join(Path(outdir),property+"_by_group.png")
    plt.savefig(fig_path, dpi=150)
    plt.show()

    print(f"\nHistogram saved to {fig_path}")


# ----------------------------
# Statistics + KS test
# ----------------------------
def compute_statistics(df,property="gravity"):
    ir_env = pd.to_numeric(
        df.loc[df["group"] == "envelope", property],
        errors="coerce",
    ).dropna()

    ir_free = pd.to_numeric(
        df.loc[df["group"] == "envelope_free", property],
        errors="coerce",
    ).dropna()

    stats_df = pd.DataFrame(
        {
            "Group": ["Envelope", "Envelope-free"],
            "N": [len(ir_env), len(ir_free)],
            "Mean IR index": [np.mean(ir_env), np.mean(ir_free)],
            "Std error": [sem(ir_env), sem(ir_free)],
        }
    )

    ks_stat, ks_p = ks_2samp(ir_env, ir_free)

    print("\n=== IR Index Statistics by Group ===")
    print(stats_df.to_string(index=False, float_format="%.4f"))

    print("\n=== KS Test: envelope vs envelope-free ===")
    print(f"KS statistic = {ks_stat:.4f}")
    print(f"p-value      = {ks_p:.4e}")


# ----------------------------
# Main runner
# ----------------------------
def main():
    FILE = "text_files/combined_yso_parameters_v2.xlsx"
    property = "gravity"
    OUTDIR = "histograms/group_output_"+property
    df = load_data(FILE)
    df = classify_sources(df)

    groups = split_groups(df)
    save_groups(groups, OUTDIR)
    plot_histogram(groups, OUTDIR,property)
    compute_statistics(df,property)


if __name__ == "__main__":
    main()
