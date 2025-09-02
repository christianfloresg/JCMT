# Reproducible Python code to generate and SAVE the figures you saw.
# This cell writes high-res PDF and PNG files you can download and tweak.
# You can copy this into your own script/notebook; everything is self-contained.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# ---------------------------
# Parameters you can tweak
# ---------------------------
FIGSIZE = (7, 6)
SCATTER_SIZE = 70
SCATTER_EDGE = True          # set to False for no marker edges
CMAP = "RdYlBu"              # decision map colormap
LEVELS = np.linspace(0, 1, 11)  # probability contour levels
ALPHA_BG = 0.6               # alpha for the decision map background
GRID_ALPHA = 0.3
SAVE_DPI = 300

# Marker/label settings for classes (kept simple for easy edits)
LABELS = {1: "Envelope", 0: "No envelope"}
MARKERS = {1: "o", 0: "s"}

# ---------------------------
# Data block (as provided)
# ---------------------------
raw = """envelope  IRS5 0.786 3.08
non-envelope  WLY2-42 0.072 3.45
envelope  OphIRS63 0.309 3.25
envelope  Elia32 0.027 2.79
envelope  Elia33 -0.012 2.94
non-envelope  YLW58 -0.761 3.1
non-envelope  GY92-235 -0.365 3.63
non-envelope  SR24 -0.666 3.57
non-envelope  DoAr43 -0.565 3.65
non-envelope  DoAr25 -1.076 3.52
envelope  IRAS03301+3111 0.321 3.32
non-envelope  IRAS19247+2238 -0.231 3.43
non-envelope  IRAS19247+2238 -0.231 3.52
envelope  IRAS03260+3111 -0.566 3.16
envelope  IRAS04113+2758S -0.393 3.14
non-envelope  IRAS05379-0758 -0.723 3.13
non-envelope  DG-Tau -0.032 3.28
non-envelope  IRAS04489+3042 0.246 3.7
envelope  IRAS04591-0856 0.435 3.26
envelope  V347_Aur -0.218 2.94
non-envelope  UYAur -0.155 3.42
non-envelope  FPTau -1.121 3.4
non-envelope  IRAS05555-1405 0.395 3.27
non-envelope  IRAS04181+2655M 0.346 3.71
envelope  IRAS04181+2655S 0.462 3.75
envelope  GV_Tau 0.97 3.26
non-envelope  Haro6-13 -0.116 3.55
non-envelope  Haro6-28 -0.923 3.86
non-envelope  IRAS04108+2803E 0.84 3.71
non-envelope  IRAS04295+2251 0.478 3.41
non-envelope  Haro6-33 -0.075 3.69
non-envelope  HK-Tau -0.427 3.62
envelope  T-Tauri 0.094 3.45
envelope  EC92 0.92 2.89"""

df = pd.DataFrame([line.split() for line in raw.splitlines()],
                  columns=["envelope", "name", "alpha", "logg"])
df["alpha"] = df["alpha"].astype(float)
df["logg"] = df["logg"].astype(float)
df["y"] = (df["envelope"] == "envelope").astype(int)

# ---------------------------
# Fit logistic regression
# ---------------------------
X = df[["logg", "alpha"]].values
y = df["y"].values
clf = LogisticRegression()
clf.fit(X, y)

# Grid for decision map
x_min, x_max = df["logg"].min() - 0.2, df["logg"].max() + 0.2
y_min, y_max = df["alpha"].min() - 0.2, df["alpha"].max() + 0.2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)

# ---------------------------
# Figure 2: Decision map (no title, colorbar matches background)
# ---------------------------
fig2, ax2 = plt.subplots(figsize=FIGSIZE)
contour = ax2.contourf(xx, yy, Z, levels=LEVELS, cmap=CMAP, alpha=ALPHA_BG)
# scatter points (default matplotlib colors; edit if desired)
for cls, grp in df.groupby("y"):
    ax2.scatter(grp["logg"], grp["alpha"],
                s=SCATTER_SIZE, marker=MARKERS[cls],
                edgecolor="k" if SCATTER_EDGE else None,
                label=LABELS[cls])
ax2.set_xlabel(r"$\log g$")
ax2.set_ylabel(r"$\alpha$ (non-corrected)")
ax2.legend()
ax2.grid(alpha=GRID_ALPHA)
cbar = fig2.colorbar(contour, ax=ax2)
cbar.set_label("P(envelope)")
fig2.tight_layout()
fig2.savefig("/mnt/data/fig_logg_alpha_boundary.pdf")
fig2.savefig("/mnt/data/fig_logg_alpha_boundary.png", dpi=SAVE_DPI)

# ---------------------------
# ROC calculations
# ---------------------------
# log g alone (lower log g => higher envelope prob): use negative log g as score
fpr_logg, tpr_logg, _ = roc_curve(y, -df["logg"])
auc_logg = auc(fpr_logg, tpr_logg)

# alpha alone
fpr_alpha, tpr_alpha, _ = roc_curve(y, df["alpha"])
auc_alpha = auc(fpr_alpha, tpr_alpha)

# combined model score
scores = clf.predict_proba(X)[:, 1]
fpr_comb, tpr_comb, _ = roc_curve(y, scores)
auc_comb = auc(fpr_comb, tpr_comb)

# ---------------------------
# Figure 3: ROC curves (no title, standard margins)
# ---------------------------
fig3, ax3 = plt.subplots(figsize=FIGSIZE)
ax3.plot(fpr_logg, tpr_logg, label=f"log g (AUC = {auc_logg:.2f})")
ax3.plot(fpr_alpha, tpr_alpha, label=f"$\\alpha$ (AUC = {auc_alpha:.2f})")
ax3.plot(fpr_comb, tpr_comb, label=f"Combined (AUC = {auc_comb:.2f})")
ax3.plot([0, 1], [0, 1], "k--", alpha=0.7)
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend(loc="lower right")
ax3.grid(alpha=GRID_ALPHA)
fig3.tight_layout()
fig3.savefig("/mnt/data/fig_roc.pdf")
fig3.savefig("/mnt/data/fig_roc.png", dpi=SAVE_DPI)

auc_logg, auc_alpha, auc_comb
