import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# ---------------------------
# Parameters
# ---------------------------
DATA_PATH = "text_files/combined_yso_parameters_v2.xlsx"

FIGSIZE = (7, 6)
SCATTER_SIZE = 70
SCATTER_EDGE = True
CMAP = "RdYlBu"
LEVELS = np.linspace(0, 1, 11)
ALPHA_BG = 0.6
GRID_ALPHA = 0.3
SAVE_DPI = 300

LABELS = {1: "Envelope", 0: "No envelope"}
MARKERS = {1: "o", 0: "s"}

# ---------------------------
# Load and prepare data
# ---------------------------
df = pd.read_excel(DATA_PATH)

# Normalize envelope status
df["Envelope_status"] = (
    df["Envelope_status"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Map to binary outcome
df["y"] = df["Envelope_status"].map({"yes": 1, "no": 0})

# Keep only valid rows
df = df[df["y"].isin([0, 1])].copy()

# Coerce predictors to numeric
df["gravity"] = pd.to_numeric(df["gravity"], errors="coerce") / 1e2
df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")

# Drop rows with missing predictors
df = df[np.isfinite(df["gravity"]) & np.isfinite(df["Temperature"])]

print(f"Using {len(df)} sources for 2D logistic regression")

# ---------------------------
# Fit logistic regression
# ---------------------------
X = df[["gravity", "Temperature"]].values
y = df["y"].values

clf = LogisticRegression()
clf.fit(X, y)

# ---------------------------
# Grid for decision map
# ---------------------------
x_min, x_max = df["gravity"].min() - 0.2, df["gravity"].max() + 0.2
y_min, y_max = df["Temperature"].min() - 100, df["Temperature"].max() + 100

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = clf.predict_proba(
    np.c_[xx.ravel(), yy.ravel()]
)[:, 1].reshape(xx.shape)

# ---------------------------
# Plot: decision map
# ---------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)

contour = ax.contourf(
    xx, yy, Z,
    levels=LEVELS,
    cmap=CMAP,
    alpha=ALPHA_BG
)

for cls, grp in df.groupby("y"):
    ax.scatter(
        grp["gravity"],
        grp["Temperature"],
        s=SCATTER_SIZE,
        marker=MARKERS[cls],
        edgecolor="k" if SCATTER_EDGE else None,
        label=LABELS[cls]
    )

ax.set_xlabel(r"$\log g$")
ax.set_ylabel(r"$T_{\rm eff}$ (K)")
ax.legend()
ax.grid(alpha=GRID_ALPHA)

cbar = fig.colorbar(contour, ax=ax)
cbar.set_label("P(envelope)")

fig.tight_layout()
fig.savefig("Figures/logg_temperature_boundary.pdf")
fig.savefig("Figures/logg_temperature_boundary.png", dpi=SAVE_DPI)

plt.show()
