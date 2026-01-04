import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import sem, ks_2samp

# ----------------------------
# Load the combined table
# ----------------------------
FILE = "text_files/combined_yso_parameters_v2.xlsx"   # path to your file
df = pd.read_excel(FILE)

# Ensure needed columns exist (in case of missing data)
for col in ["SourceName", "gravity", "conc_HCOplus__c_factor_gaussian", "spec_HCOplus__Integ_Beam"]:
    if col not in df.columns:
        df[col] = np.nan

# ----------------------------
# Classification Rules
# ----------------------------
cf_gauss = pd.to_numeric(df["conc_HCOplus__c_factor_gaussian"], errors="coerce")
integ_beam = pd.to_numeric(df["spec_HCOplus__Integ_Beam"], errors="coerce")
gravity = pd.to_numeric(df["gravity"], errors="coerce")

# Base classification
envelope_mask = (cf_gauss > 0.6) & (integ_beam > 0.3)
group = np.where(envelope_mask, "envelope", "envelope_free")

# Override for the two exceptions
exceptions = {"EC92", "IRAS05379-0758"}
df_upper = df["SourceName"].fillna("").str.upper().str.strip()
group = np.where(df_upper.isin(exceptions), "confused", group)

df["group"] = group
df["gravity_num"] = gravity  # numeric gravity for histogram

# ----------------------------
# Save / print group member lists
# ----------------------------
OUTDIR = Path("group_output")
OUTDIR.mkdir(exist_ok=True)

cols_show = ["SourceName", "gravity", "conc_HCOplus__c_factor_gaussian", "spec_HCOplus__Integ_Beam"]
groups = {
    "envelope": df[df["group"] == "envelope"][cols_show],
    "envelope_free": df[df["group"] == "envelope_free"][cols_show],
    "confused": df[df["group"] == "confused"][cols_show],
}

for name, subdf in groups.items():
    csv_path = OUTDIR / f"{name}_sources.csv"
    subdf.to_csv(csv_path, index=False)
    print(f"\n{name.upper()} sources ({len(subdf)}):")
    print(subdf["SourceName"].tolist() or f"(no {name} sources)")
    print(f"→ saved to {csv_path}")

# ----------------------------
# Plot histogram
# ----------------------------
plt.figure(figsize=(7, 6))
# bins = "auto"
bins = np.arange(2.70, 4.10, 0.20)
colors = {
    "envelope": "C1",       # red
    "envelope_free": "C0",  # blue
    "confused": "C2"        # green
}

for name, subdf in groups.items():
    data = pd.to_numeric(subdf["gravity"], errors="coerce").dropna() / 100
    plt.hist(data, bins=bins, alpha=0.6, label=f"{name} (n={len(data)})", edgecolor="black",color=colors[name])

plt.xlabel("gravity",size=16)
plt.ylabel("count",size=16)
# plt.title("Gravity distribution by group")
plt.legend(fontsize=14)
plt.tight_layout()

FIG_PATH = OUTDIR / "gravity_histogram_by_group.png"
plt.tick_params(axis='both', labelsize=14)  # 14 = size in points
plt.savefig(FIG_PATH, dpi=150)
plt.show()

print(f"\nHistogram saved to {FIG_PATH}")


# ---- Extract numeric gravity values by group ----
g_env = pd.to_numeric(df.loc[df["group"] == "envelope", "gravity"], errors="coerce").dropna()
g_free = pd.to_numeric(df.loc[df["group"] == "envelope_free", "gravity"], errors="coerce").dropna()

# ---- Compute statistics ----
stats_dict = {
    "Group": ["Envelope", "Envelope-free"],
    "N": [len(g_env), len(g_free)],
    "Mean gravity": [np.mean(g_env), np.mean(g_free)],
    "Std error": [sem(g_env), sem(g_free)]
}

stats_df = pd.DataFrame(stats_dict)

# ---- KS test ----
ks_stat, ks_p = ks_2samp(g_env, g_free)

# ---- Print results ----
print("\n=== Gravity Statistics by Group ===")
print(stats_df.to_string(index=False, float_format="%.4f"))

print("\n=== KS Test: envelope vs envelope-free ===")
print(f"KS statistic = {ks_stat:.4f}")
print(f"p-value      = {ks_p:.4e}")