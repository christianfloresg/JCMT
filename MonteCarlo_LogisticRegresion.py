import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ----------------------
# DATA
# ----------------------
logg = np.array([
    3.4317, 3.5184, 2.7904, 2.9387, 3.6554, 3.7958, 3.4497, 3.82, 4.04, 3.5462,
    3.858, 3.7096, 3.4131, 3.2576, 3.1449, 3.7133, 3.7512, 3.6227, 3.6887, 3.2839,
    3.631, 3.16, 3.3223, 3.7024, 3.2607, 3.1251, 3.2737, 3.4294, 3.516, 3.0842,
    3.5712, 2.9412, 3.1009, 3.4203, 3.6524, 3.4014
])

# Asymmetric uncertainties
logg_unc_pos = np.array([
    0.2345, 0.1427, 0.0976, 0.2126, 0.1179, 0.0957, 0.1707, 0.67, 0.46, 0.2005,
    0.0524, 0.1288, 0.3303, 0.1228, 0.0612, 0.1129, 0.1329, 0.0932, 0.1196, 0.1347,
    0.1713, 0.1606, 0.1315, 0.1197, 0.1788, 0.1744, 0.5329, 0.1724, 0.1699, 0.164,
    0.3244, 0.2461, 0.3715, 0.079, 0.1348, 0.2093
])
logg_unc_neg = np.array([
    0.1022, 0.0937, 0.1245, 0.1119, 0.1166, 0.1081, 0.075, 0.67, 0.46, 0.2168,
    0.0632, 0.1635, 0.222, 0.1721, 0.0531, 0.1181, 0.3584, 0.1032, 0.1425, 0.1976,
    0.3352, 0.1567, 0.1536, 0.4697, 0.2159, 0.1244, 0.2726, 0.1145, 0.1085, 0.0837,
    0.4635, 0.1539, 0.1006, 0.1083, 0.2003, 0.1014
])

# C-factor values
hco_plus = np.array([
    0.0000e+00, -8.0700e-02, 3.0780e-01, 4.3970e-01, 0.0000e+00, 0.0000e+00,
    -5.8980e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
    0.0000e+00, 7.7730e-01, 2.2100e-01, 8.1900e-02, 5.5860e-01, 0.0000e+00,
    0.0000e+00, -3.4000e-03, 8.7700e-02, 7.4120e-01, 3.3490e-01, 0.0000e+00,
    6.9980e-01, 2.4610e-01, -1.6317e+00, 0.0000e+00, 0.0000e+00, 4.5440e-01,
    0.0000e+00, 5.6870e-01, -1.9718e+00, 0.0000e+00, -3.4395e+00, 0.0000e+00
])
c18o = np.array([
    0., 0.1131, 0.326, 0.4933, 0., 0., 0.177, 0., 0., 0.,
    0., 0., 0., 0.6319, 0.4364, 0.2564, 0.6455, 0., -0.7307, 0.3944,
    0.3015, 0.29, 0.3688, -8.8003, 0.5255, -0.4915, 0., 0.3719, 0.3719, 0.3645,
    0., 0.3839, 0.0907, 0., -35.0692, 0.
])

# Envelope status: present if both tracers > 0.2, else dispersed
status_joint = ~((hco_plus > 0.2) & (c18o > 0.2))


# ----------------------
# FUNCTION: split-normal sampling
# A split-normal distribution (also called an asymmetric normal or two-piece normal)
# is basically a Gaussian curve that has different standard deviations on either side of the mean.
# It's what you get when the positive and negative measurement uncertainties are not the same.
# ----------------------
def sample_split_normal(mu, sigma_pos, sigma_neg, rng):
    """Sample one value from a split-normal distribution."""
    # Probability of sampling from left side
    p_left = sigma_neg / (sigma_pos + sigma_neg)
    if rng.random() < p_left:
        return rng.normal(mu, sigma_neg)
    else:
        return rng.normal(mu, sigma_pos)


# ----------------------
# MONTE CARLO SIMULATION
# ----------------------
n_iter = 5000
logg_range = np.linspace(min(logg) - 0.1, max(logg) + 0.1, 200)
probs_samples = np.zeros((n_iter, len(logg_range)))
logg50_samples = []

rng = np.random.default_rng(42)

for i in range(n_iter):
    # Perturb logg using asymmetric uncertainties
    logg_sampled = np.array([
        sample_split_normal(mu, sp, sn, rng)
        for mu, sp, sn in zip(logg, logg_unc_pos, logg_unc_neg)
    ])

    X_mc = sm.add_constant(logg_sampled)
    try:
        model_mc = sm.Logit(status_joint.astype(int), X_mc).fit(disp=False)
        X_pred_mc = sm.add_constant(logg_range)
        probs_samples[i, :] = model_mc.predict(X_pred_mc)

        # Store logg at 50% probability
        beta0, beta1 = model_mc.params
        logg50_samples.append(-beta0 / beta1)
    except:
        probs_samples[i, :] = np.nan

# ----------------------
# RESULTS
# ----------------------
lower = np.nanpercentile(probs_samples, 2.5, axis=0)
upper = np.nanpercentile(probs_samples, 97.5, axis=0)
median = np.nanpercentile(probs_samples, 50, axis=0)

logg50_samples = np.array(logg50_samples)
logg50_med = np.nanmedian(logg50_samples)
logg50_low = np.nanpercentile(logg50_samples, 2.5)
logg50_high = np.nanpercentile(logg50_samples, 97.5)

print(f"logg_50 median = {logg50_med:.3f}, 95% CI = [{logg50_low:.3f}, {logg50_high:.3f}]")

# ----------------------
# PLOT
# ----------------------
plt.figure(figsize=(7, 5))
# plt.scatter(logg, status_joint, color='black', alpha=0.7, label='Observed (1=dispersed)')
plt.errorbar(logg, status_joint, xerr=[logg_unc_neg,logg_unc_pos], fmt='o', color='k',
             ecolor='gray', capsize=4, alpha=0.5, zorder=1,ms=12)
    #,         label='Observed (1=dispersed) ± logg error')

plt.plot(logg_range, median, color='blue', lw=2, label='Median logistic fit')# '(MC w/ asym. logg errors)')
plt.fill_between(logg_range, lower, upper, color='blue', alpha=0.2, label='95% CI (fit + logg error)')
plt.axvline(logg50_med, color='red', ls='--', lw=1,
            label=f'50% prob at logg={logg50_med:.2f}')
plt.xlabel('log g')
plt.ylabel('Probability envelope dispersed')
plt.legend()
# plt.grid(True)
plt.xlim(2.7,4.13)
plt.show()
