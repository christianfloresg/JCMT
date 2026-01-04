import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------------------------------------
# Load data from the combined XLSX file
# ---------------------------------------
df = pd.read_excel("text_files/combined_yso_parameters_v2.xlsx")

# Map column names from spreadsheet to legacy variable names
logg_data     = pd.to_numeric(df["gravity"], errors="coerce")                  # main x variable
logg_unc_pos  = pd.to_numeric(df["grav_unce_up"], errors="coerce")             # + uncertainty
logg_unc_neg  = pd.to_numeric(df["grav_unce_down"], errors="coerce")            # - uncertainty
hco_plus      = pd.to_numeric(df["conc_HCOplus__c_factor_gaussian"], errors="coerce")
integ_beam    = pd.to_numeric(df["spec_HCOplus__Integ_Beam"], errors="coerce")
source_names  = df["SourceName"].astype(str)

# ---------------------------------------
# Apply classification rules
# ---------------------------------------
envelope_mask = (hco_plus > 0.6) & (integ_beam > 0.3)
group = np.where(envelope_mask, "envelope", "envelope_free")

# Override for exceptions
exceptions = {"EC92", "IRAS05379-0758"}
df_upper = source_names.str.upper().str.strip()
group = np.where(df_upper.isin(exceptions), "confused", group)

df["group"] = group

# Keep only envelope/envelope_free for logistic regression
df_model = df[df["group"].isin(["envelope", "envelope_free"])].copy()
df_model["y"] = (df_model["group"] == "envelope").astype(int)

# ---------------------------------------
# Logistic regression using gravity only
# ---------------------------------------
X = sm.add_constant(df_model["gravity"])   # no scaling for now
y = df_model["y"]


# ---------------------------------------
# Remove NaNs before regression
# ---------------------------------------
df_model = df[df["group"].isin(["envelope", "envelope_free"])].copy()

# Ensure gravity is numeric
df_model["gravity"] = pd.to_numeric(df_model["gravity"], errors="coerce")

# Remove rows with missing predictor or missing response
df_model = df_model[df_model["gravity"].notna()].copy()

df_model["y"] = (df_model["group"] == "envelope").astype(int)

# Build design matrix AFTER dropping NaNs
X = sm.add_constant(df_model["gravity"])
y = df_model["y"]

glm = sm.GLM(y, X, family=sm.families.Binomial())
res = glm.fit()

print("\n=== Logistic Model Summary ===")
print(res.summary())

# ---------------------------------------
# Prediction curve
# ---------------------------------------
x_vals = np.linspace(df_model["gravity"].min(), df_model["gravity"].max(), 200)
X_pred = sm.add_constant(x_vals)
pred = res.get_prediction(X_pred).summary_frame(alpha=0.05)

# 50% crossover point (probability = 0.5)
b0, b1 = res.params["const"], res.params["gravity"]
x50 = -b0 / b1
print(f"\nGravity at 50% envelope probability: {x50:.4f}")

# ---------------------------------------
# Plot
# ---------------------------------------
plt.figure(figsize=(8,6))

# Jitter scatter (0 and 1)
jitter = (np.random.rand(len(df_model)) - 0.5) * 0.05
# plt.scatter(df_model["gravity"], df_model["y"] + jitter, s=50, alpha=0.6)

plt.errorbar(df_model["gravity"], df_model["y"] + jitter, xerr=[df_model["grav_unce_down"],df_model["grav_unce_up"]], fmt='o', color='k',
             ecolor='gray', capsize=4, alpha=0.5, zorder=1,ms=12)

# Logistic curve
plt.plot(x_vals, pred["mean"], 'k-', lw=2, label="Logistic fit")

# Confidence band
plt.fill_between(x_vals, pred["mean_ci_lower"], pred["mean_ci_upper"],
                 alpha=0.3, label="95% CI")

# 50% vertical line
plt.axvline(x50, linestyle="--", color="red", label=f"50% point = {x50:.0f}")

plt.xlabel("Gravity",size=14)
plt.ylabel("P(envelope)",size=14)
# plt.title("Envelope Probability vs Gravity (HCO+ based)")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.tight_layout()
plt.savefig('logistic_curve.png')
plt.show()
