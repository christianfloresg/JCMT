import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import sem, ks_2samp
import os
from scipy.stats import sem, ks_2samp, mannwhitneyu, anderson_ksamp
from datetime import date,datetime

today = str(date.today())
currentDateAndTime = datetime.now()
hour_now = str(currentDateAndTime.hour)

# ----------------------------
# Load and prepare data
# ----------------------------
def load_data(filepath):
    df = pd.read_excel(filepath)

    required_cols = [
        "SourceName",
        "gravity",
        "Envelope_status",
        "indices__spectral_index", ## These are old values from different literature sources
        "indices__corr_spectral_index_lit", ## These are old values from different literature sources
        "spectral_index_SED", ## These are values calculated by SED
        "spectral_index_corrected_SED",
        "Tbol",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
        if col == "gravity":
            df[col]=df[col]/100.

    return df

# ----------------------------
# Classification
# ----------------------------


def classify_sources(
    df: pd.DataFrame,
    status_col: str = "Envelope_status",
    out_col: str = "group",
) -> pd.DataFrame:
    """
    Map Envelope_status to analysis groups:
      yes     -> envelope
      no      -> envelope_free
      unclear -> confused
      anything else / missing -> confused
    """
    status = (
        df.get(status_col, pd.Series(index=df.index, dtype="object"))
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan})
    )

    mapping = {
        "yes": "envelope",
        "no": "envelope_free",
        "unclear": "confused",
    }

    df[out_col] = status.map(mapping).fillna("confused")
    return df

# ----------------------------
# Split into groups
# ----------------------------
def split_groups(df: pd.DataFrame) -> dict:
    cols_show = [
        "SourceName",
        "gravity",
        "Envelope_status",
        "indices__spectral_index",
        "indices__corr_spectral_index_lit",
        "spectral_index_SED",
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
def plot_histogram(groups, outdir, property ="gravity",save=False):
    bins = np.arange(-1.4, 1.1, 0.20)
    if property=='gravity':
        bins = np.arange(2.60, 4.00, .10)
    elif property=='spectral_index_SED':
        bins = np.arange(-1.8, 2.0, 0.20)
    elif property == 'spectral_index_corrected_SED':
        bins = np.arange(-1.8, 2.0, 0.20)

    elif property == "Tbol":
        bins = np.arange(50, 1600, 100)



    colors = {
        # "envelope": "#bebada",
        # "envelope_free": "gold",
        # "confused": "#8dd3c7",
        "envelope": "#756bb1",     # darker lavender / purple
        "envelope_free": "#e6ab02",      # mustard instead of pale yellow
        "confused": "#1b9e77"  # deeper teal-green
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
            lw=2.5,
            alpha=1.0,
        )

    plt.xlabel(property, size=14)
    plt.ylabel("count", size=14)
    plt.legend(fontsize=12)
    plt.tick_params(axis="both", labelsize=14)
    plt.tight_layout()
    fig_path = os.path.join(Path(outdir),property+"_"+today+"_by_group.png")
    if save:
        plt.savefig(fig_path, dpi=150)
    plt.show()

    print(f"\nHistogram saved to {fig_path}")



# ----------------------------
# Helper: pretty stats + robust checks
# ----------------------------
def _to_numeric_group(df, group_name, property):
    return pd.to_numeric(df.loc[df["group"] == group_name, property], errors="coerce").dropna().to_numpy()


# ----------------------------
# Statistics + KS + Mann–Whitney + Anderson–Darling (k-sample)
# ----------------------------
def compute_statistics(df, property="gravity"):
    x_env = _to_numeric_group(df, "envelope", property)
    x_free = _to_numeric_group(df, "envelope_free", property)

    stats_df = pd.DataFrame(
        {
            "Group": ["Envelope", "Envelope-free"],
            "N": [len(x_env), len(x_free)],
            f"Mean {property}": [np.mean(x_env) if len(x_env) else np.nan,
                                np.mean(x_free) if len(x_free) else np.nan],
            "Std error": [sem(x_env) if len(x_env) > 1 else np.nan,
                          sem(x_free) if len(x_free) > 1 else np.nan],
            f"Median {property}": [np.median(x_env) if len(x_env) else np.nan,
                                   np.median(x_free) if len(x_free) else np.nan],
        }
    )

    print(f"\n=== {property} Statistics by Group ===")
    print(stats_df.to_string(index=False, float_format="%.4f"))

    if len(x_env) == 0 or len(x_free) == 0:
        print("\nNot enough data in one of the groups to run 2-sample tests.")
        return

    # --- KS (kept for reference, but not ideal with ties/small N)
    ks_stat, ks_p = ks_2samp(x_env, x_free)

    # --- Mann–Whitney U (rank-based; good default alternative)
    # two-sided test; use exact when possible, else asymptotic
    try:
        mw = mannwhitneyu(x_env, x_free, alternative="two-sided", method="exact")
    except TypeError:
        # older SciPy versions don't have "method"
        mw = mannwhitneyu(x_env, x_free, alternative="two-sided")
    except ValueError:
        # exact may fail with ties/large samples; fall back
        mw = mannwhitneyu(x_env, x_free, alternative="two-sided", method="asymptotic")

    # --- Anderson–Darling (k-sample) test
    # SciPy provides anderson_ksamp for k>=2 samples; for 2-sample it's the AD 2-sample generalization.
    # Note: with many ties, p-values are approximate; still usually more informative than KS.
    ad = anderson_ksamp([x_env, x_free])

    print("\n=== 2-sample tests: envelope vs envelope-free ===")

    print("\n[KS test]")
    print(f"KS statistic = {ks_stat:.4f}")
    print(f"p-value      = {ks_p:.4e}")

    print("\n[Mann–Whitney U test]")
    # SciPy returns statistic (U) and pvalue
    print(f"U statistic  = {mw.statistic:.4f}")
    print(f"p-value      = {mw.pvalue:.4e}")

    print("\n[Anderson–Darling k-sample test (2-sample)]")
    # ad.statistic, ad.significance_level (percent), ad.critical_values
    # significance_level is the approximate p-value expressed in percent for anderson_ksamp in many SciPy versions.
    # We'll print both.
    p_approx = ad.significance_level / 100.0 if hasattr(ad, "significance_level") else np.nan
    print(f"AD statistic = {ad.statistic:.4f}")
    if np.isfinite(p_approx):
        print(f"p-value      ≈ {p_approx:.4e}  (approx; reported by SciPy as {ad.significance_level:.3f}%)")
    else:
        print("p-value      = (not available in this SciPy version)")
    # Optional: show critical values if present
    if hasattr(ad, "critical_values"):
        print(f"critical vals= {np.array(ad.critical_values)}")


# ----------------------------
# Main runner
# ----------------------------
def main():
    FILE = "text_files/combined_yso_parameters_v2.xlsx"
    property = 'spectral_index_SED'#"Tbol"#'spectral_index_corrected_SED'#'spectral_index_SED' # "gravity"
    OUTDIR = f"histograms/group_output_{property}"

    df = load_data(FILE)
    df = classify_sources(df)

    groups = split_groups(df)
    save_groups(groups, OUTDIR)
    plot_histogram(groups, OUTDIR, property,save=True)
    compute_statistics(df, property)


if __name__ == "__main__":
    main()