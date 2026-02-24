import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ----------------------------
# Configuration
# ----------------------------
DEFAULT_PATH = "text_files/combined_yso_parameters_v2.xlsx"
STATUS_COL = "Envelope_status"          # "yes" / "no" / "unclear"
X_COL = "gravity"
XERR_LO_COL = "grav_unce_down"          # (positive number) lower uncertainty
XERR_HI_COL = "grav_unce_up"            # (positive number) upper uncertainty
NAME_COL = "SourceName"

STATUS_MAP = {"yes": 1, "no": 0}        # "unclear" will be dropped


# ----------------------------
# Core utilities
# ----------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df


def prepare_logit_data(
    df: pd.DataFrame,
    x_col: str = X_COL,
    status_col: str = STATUS_COL,
    status_map: dict = STATUS_MAP,
    xerr_lo_col: str = XERR_LO_COL,
    xerr_hi_col: str = XERR_HI_COL,
) -> pd.DataFrame:
    out = df.copy()

    # Normalize status strings
    out[status_col] = (
        out[status_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan})
    )

    # Map to binary response (drop unclear/unknown)
    out["y"] = out[status_col].map(status_map)

    # Coerce numeric predictor + uncertainties
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce")/1.e2
    out[xerr_lo_col] = pd.to_numeric(out.get(xerr_lo_col), errors="coerce")/1.e2
    out[xerr_hi_col] = pd.to_numeric(out.get(xerr_hi_col), errors="coerce")/1.e2

    # If uncertainty columns are missing, fill with zeros
    if xerr_lo_col not in out.columns:
        out[xerr_lo_col] = 0.0
    if xerr_hi_col not in out.columns:
        out[xerr_hi_col] = 0.0

    # Clean: keep only rows with y in {0,1} and finite x
    out = out[out["y"].isin([0, 1])].copy()
    out = out[np.isfinite(out[x_col])].copy()

    # Replace missing uncertainties with 0 (so plotting and MC still work)
    out[xerr_lo_col] = out[xerr_lo_col].fillna(0.0).clip(lower=0.0)
    out[xerr_hi_col] = out[xerr_hi_col].fillna(0.0).clip(lower=0.0)

    return out


def fit_logit_glm(df_model: pd.DataFrame, x_col: str = X_COL):
    X = sm.add_constant(df_model[x_col].values)
    y = df_model["y"].values
    glm = sm.GLM(y, X, family=sm.families.Binomial())
    res = glm.fit()
    return res


def predict_curve(res, x_grid: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    X_pred = sm.add_constant(x_grid)
    pred = res.get_prediction(X_pred).summary_frame(alpha=alpha)
    # Columns include: mean, mean_se, mean_ci_lower, mean_ci_upper
    return pred


def x_at_p(res, p: float = 0.5) -> float:
    """
    Solve logit(p) = b0 + b1*x for x.
    """
    b0, b1 = res.params[0], res.params[1]
    if b1 == 0:
        return np.nan
    return (np.log(p / (1 - p)) - b0) / b1


# ----------------------------
# Monte Carlo for x-uncertainty (errors-in-x propagation)
# ----------------------------
def sample_x_with_asymmetric_uncertainty(
    x: np.ndarray, lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Simple asymmetric sampler:
    - with prob 0.5 sample below x using N(x, lo)
    - with prob 0.5 sample above x using N(x, hi)
    This isn't a perfect statistical model, but it matches typical "up/down" error bars.
    """
    lo = np.maximum(lo, 0.0)
    hi = np.maximum(hi, 0.0)

    coin = rng.random(size=x.shape) < 0.5
    sd = np.where(coin, lo, hi)
    return x + rng.normal(loc=0.0, scale=sd, size=x.shape)


def monte_carlo_band(
    df_model: pd.DataFrame,
    x_grid: np.ndarray,
    n_sims: int = 2000,
    alpha: float = 0.05,
    x_col: str = X_COL,
    xerr_lo_col: str = XERR_LO_COL,
    xerr_hi_col: str = XERR_HI_COL,
    seed: int = 0,
):
    """
    Propagate x-axis measurement uncertainty into:
    - median predicted curve
    - pointwise (1-alpha) bands (percentiles)
    - distribution of x50
    """

    percentage_of_envelope = 50

    rng = np.random.default_rng(seed)

    x_obs = df_model[x_col].to_numpy()
    lo = df_model[xerr_lo_col].to_numpy()
    hi = df_model[xerr_hi_col].to_numpy()
    y = df_model["y"].to_numpy()

    preds = np.empty((n_sims, len(x_grid)), dtype=float)
    x50s = np.empty(n_sims, dtype=float)

    for i in range(n_sims):
        x_samp = sample_x_with_asymmetric_uncertainty(x_obs, lo, hi, rng)
        X = sm.add_constant(x_samp)
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        res_i = glm.fit()

        pred_i = predict_curve(res_i, x_grid, alpha=alpha)["mean"].to_numpy()
        preds[i, :] = pred_i
        x50s[i] = x_at_p(res_i, percentage_of_envelope/100.)

    lower_q = 100 * (alpha / 2)
    upper_q = 100 * (1 - alpha / 2)

    band = {
        "mean_median": np.nanpercentile(preds, 50, axis=0),
        "mean_lo": np.nanpercentile(preds, lower_q, axis=0),
        "mean_hi": np.nanpercentile(preds, upper_q, axis=0),
        "x50_median": np.nanpercentile(x50s, 50),
        "x50_lo": np.nanpercentile(x50s, lower_q),
        "x50_hi": np.nanpercentile(x50s, upper_q),
    }
    return band


# ----------------------------
# Plotting
# ----------------------------
def plot_logit(
    df_model: pd.DataFrame,
    x_grid: np.ndarray,
    curve_mean: np.ndarray,
    band_lo: np.ndarray,
    band_hi: np.ndarray,
    x50: float,
    x_col: str = X_COL,
    xerr_lo_col: str = XERR_LO_COL,
    xerr_hi_col: str = XERR_HI_COL,
    title: str | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Colors
    col_yes = "#bebada"   # envelope
    col_no  = "#ffffb3"   # no envelope

    # Jitter y for visualization (make it index-aligned)
    rng = np.random.default_rng(42)
    jitter_vals = (rng.random(len(df_model)) - 0.5) * 0.04
    jitter = pd.Series(jitter_vals, index=df_model.index)

    # Split data
    df_yes = df_model[df_model["y"] == 1]
    df_no  = df_model[df_model["y"] == 0]

    # Plot "yes" points
    ax.errorbar(
        df_yes[x_col],
        df_yes["y"] + jitter.loc[df_yes.index],
        xerr=[df_yes[xerr_lo_col], df_yes[xerr_hi_col]],
        fmt="o",
        ms=18,
        lw=1,
        color=col_yes,
        ecolor="gray",
        alpha=0.9,
        capsize=3,
        mec='k',
        label="Envelope",
        zorder=3,
    )

    # Plot "no" points
    ax.errorbar(
        df_no[x_col],
        df_no["y"] + jitter.loc[df_no.index],
        xerr=[df_no[xerr_lo_col], df_no[xerr_hi_col]],
        fmt="o",
        ms=18,
        mec='k',
        lw=1,
        color=col_no,
        ecolor="gray",
        alpha=0.9,
        capsize=3,
        label="No envelope",
        zorder=3,
    )

    # Logistic curve and confidence band
    ax.plot(x_grid, curve_mean, color="black", lw=2, label="Logistic fit", zorder=4)
    ax.fill_between(x_grid, band_lo, band_hi, color="black", alpha=0.15, label="95% CI")

    # 50% threshold
    ax.axvline(
        x50,
        linestyle="--",
        color="red",
        lw=2,
        label=f"$P=0.5$ at $\\log g={x50:.2f}$",
    )

    # Labels / limits / ticks
    ax.set_xlabel(r"$\log g$", fontsize=16)
    ax.set_ylabel("Envelope detection", fontsize=16)
    ax.set_xlim(2.65, 3.92)
    ax.set_ylim(-0.15, 1.15)

    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=12, frameon=False)

    if title:
        ax.set_title(title, fontsize=16)

    fig.tight_layout()
    # fig.savefig("logistic_curve_envelope_vs_gravity.png", dpi=300)
    plt.show()



# ----------------------------
# Main workflow
# ----------------------------
def run(
    path: str = DEFAULT_PATH,
    use_monte_carlo: bool = True,
    n_sims: int = 2000,
    alpha: float = 0.05,
):
    df = load_data(path)
    df_model = prepare_logit_data(df)

    # Fit "naive" GLM (ignores x uncertainty, but useful baseline)
    res = fit_logit_glm(df_model)

    print("\n=== Logistic Model Summary (naive GLM) ===")
    print(res.summary())

    # Grid for plotting
    x_grid = np.linspace(df_model[X_COL].min(), df_model[X_COL].max(), 250)

    if use_monte_carlo:
        band = monte_carlo_band(
            df_model=df_model,
            x_grid=x_grid,
            n_sims=n_sims,
            alpha=alpha,
            seed=0,
        )
        curve = band["mean_median"]
        lo = band["mean_lo"]
        hi = band["mean_hi"]
        x50 = band["x50_median"]
        x50_ci = (band["x50_lo"], band["x50_hi"])

        print(f"\n50% crossover (MC, accounts for x-uncertainty): {x50:.4f}")
        print(f"{int((1-alpha)*100)}% CI for 50% point (MC): [{x50_ci[0]:.4f}, {x50_ci[1]:.4f}]")

    else:
        pred = predict_curve(res, x_grid, alpha=alpha)
        curve = pred["mean"].to_numpy()
        lo = pred["mean_ci_lower"].to_numpy()
        hi = pred["mean_ci_upper"].to_numpy()
        x50 = x_at_p(res, 0.5)

        print(f"\n50% crossover (naive GLM): {x50:.4f}")

    plot_logit(
        df_model=df_model,
        x_grid=x_grid,
        curve_mean=curve,
        band_lo=lo,
        band_hi=hi,
        x50=x50,
        # title="Envelope probability vs Gravity",
    )

    return res, df_model


if __name__ == "__main__":
    run(
        path=DEFAULT_PATH,
        use_monte_carlo=False,  # set False to use standard GLM CI instead
        n_sims=500,
        alpha=0.05,
    )
