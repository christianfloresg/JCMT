# hist_ir_indices_by_group.py
# Usage: python hist_ir_indices_by_group.py
# Change INDEX_TO_PLOT to one of:
#   "indices__spectral_index"
#   "indices__corr_spectral_index_lit"
#   "indices__corr_spectral_index_Conn"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
FILE = "text_files/combined_yso_parameters_v2.xlsx"
INDEX_TO_PLOT = "indices__spectral_index"  # <- choose one of the three
# BINS = "auto"            # or an int, e.g. 20
BINS = np.arange(-1.3, 1.3, 0.25)

DENSITY = False          # True -> probability density; False -> counts
ALPHA = 0.65
EDGE = "black"
COLORS = {
    "envelope": "C1",       # red
    "envelope_free": "C0",  # blue
    "confused": "C2",       # green
}

# ----------------------------
# Load & classify
# ----------------------------
df = pd.read_excel(FILE)

# Pull needed columns
cf_gauss   = pd.to_numeric(df.get("conc_HCOplus__c_factor_gaussian"), errors="coerce")
integ_beam = pd.to_numeric(df.get("spec_HCOplus__Integ_Beam"), errors="coerce")
gravity    = pd.to_numeric(df.get("gravity"), errors="coerce")  # not used for the plot, but included for completeness

# Classification rules
envelope_mask = (cf_gauss > 0.6) & (integ_beam > 0.3)
group = np.where(envelope_mask, "envelope", "envelope_free")

exceptions = {"EC92", "IRAS05379-0758"}
src_upper = df["SourceName"].fillna("").astype(str).str.upper().str.strip()
group = np.where(src_upper.isin(exceptions), "confused", group)
df["group"] = group

# ----------------------------
# Select index column to plot
# ----------------------------
valid_cols = {
    "indices__spectral_index": "indices__spectral_index",
    "indices__corr_spectral_index_lit": "indices__corr_spectral_index_lit",
    "indices__corr_spectral_index_Conn": "indices__corr_spectral_index_Conn",
}
if INDEX_TO_PLOT not in valid_cols:
    raise ValueError(f"INDEX_TO_PLOT must be one of: {list(valid_cols.keys())}")

idx_col = valid_cols[INDEX_TO_PLOT]
if idx_col not in df.columns:
    raise ValueError(f"Column '{idx_col}' not found in {FILE}")

data_num = pd.to_numeric(df[idx_col], errors="coerce")

# ----------------------------
# Build group-wise arrays
# ----------------------------
mask_env   = df["group"] == "envelope"
mask_free  = df["group"] == "envelope_free"
mask_conf  = df["group"] == "confused"

arr_env  = data_num[mask_env].dropna().to_numpy()
arr_free = data_num[mask_free].dropna().to_numpy()
arr_conf = data_num[mask_conf].dropna().to_numpy()

print(f"\nData column: {idx_col}")
print(f"envelope:       N={len(arr_env)}")
print(f"envelope_free:  N={len(arr_free)}")
print(f"confused:       N={len(arr_conf)}")

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8, 6))

# # Overlaid histograms; if you prefer stacked, we can switch to a stacked call.
# if len(arr_env):
#     plt.hist(arr_env, bins=BINS, density=DENSITY, alpha=ALPHA,
#              label=f"envelope (n={len(arr_env)})", edgecolor=EDGE, color=COLORS["envelope"])
# if len(arr_free):
#     plt.hist(arr_free, bins=BINS, density=DENSITY, alpha=ALPHA,
#              label=f"envelope_free (n={len(arr_free)})", edgecolor=EDGE, color=COLORS["envelope_free"])
# if len(arr_conf):
#     plt.hist(arr_conf, bins=BINS, density=DENSITY, alpha=ALPHA,
#              label=f"confused (n={len(arr_conf)})", edgecolor=EDGE, color=COLORS["confused"])


plt.hist(arr_free, bins=BINS, alpha=ALPHA, density=DENSITY,
         label= f"envelope_free (n={len(arr_free)})",
         edgecolor=EDGE,  color=COLORS["envelope_free"])

plt.hist(arr_env, bins=BINS, alpha=ALPHA, density=DENSITY,
         label=f"envelope (n={len(arr_env)})",
         edgecolor=EDGE,  color=COLORS["envelope"])

plt.hist(arr_conf, bins=BINS, alpha=ALPHA, density=DENSITY,
         label= f"confused (n={len(arr_conf)})",
         edgecolor=EDGE,  color=COLORS["confused"])

xlabel = {
    "indices__spectral_index": "Spectral index",
    "indices__corr_spectral_index_lit": "Corrected spectral index (literature Av)",
    "indices__corr_spectral_index_Conn": "Corrected spectral index (Connelley Av)",
}[INDEX_TO_PLOT]

plt.xlabel(xlabel, fontsize=14)
plt.ylabel("density" if DENSITY else "count", fontsize=14)
plt.title(f"Histogram by group: {xlabel}", fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()
