import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt

# Lane-Emden equation for n=1 (isothermal)
def lane_emden(x, y):
    # y[0] = theta, y[1] = dtheta/dxi
    theta, dtheta = y
    if x == 0:
        return [dtheta, 0]  # Avoid divide by zero
    return [dtheta, -2/x*dtheta - np.exp(-theta)]

# Solve Lane-Emden from xi=0 to xi_max (dimensionless radius)
xi_max = 6.45  # Approximate critical value for Bonnor-Ebert sphere
xi = np.linspace(1e-6, xi_max, 1000)
sol = solve_ivp(lane_emden, [1e-6, xi_max], [0, 0], t_eval=xi, method='RK45')
theta = sol.y[0]
dtheta = sol.y[1]

# Column density (projected): N(x) = 2 ∫ n(s) ds along line of sight
def density(xi):
    # n(xi) = exp(-theta(xi)), by BE solution
    theta_interp = np.interp(xi, sol.t, sol.y[0])
    return np.exp(-theta_interp)

def col_density_proj(R):
    # Projected column at impact parameter R (in xi units)
    # N(R) = 2 ∫ n(s) ds from s=0 to s=sqrt(xi_max^2 - R^2)
    smax = np.sqrt(xi_max**2 - R**2)
    def integrand(s):
        r = np.sqrt(R**2 + s**2)
        return density(r)
    return 2 * quad(integrand, 0, smax)[0]

# Calculate profile
Rvals = np.linspace(0, xi_max, 100)
Nvals = np.array([col_density_proj(R) for R in Rvals])
N_peak = Nvals[0]
N_edge = Nvals[-1]

# Calculate integrated column density (total "flux")
dR = Rvals[1] - Rvals[0]
area = 2 * np.pi * np.sum(Rvals * Nvals) * dR
R_eff = xi_max  # For normalized BE sphere

# Concentration factor for BE sphere (using Johnstone+ formula)
# C = 1 - (integrated flux) / (area * peak)
flat_flux = np.pi * (R_eff**2) * N_peak
C_BE = 1 - (area / flat_flux)

# Uniform sphere for comparison
Nvals_uniform = np.ones_like(Nvals) * N_peak
area_uniform = 2 * np.pi * np.sum(Rvals * Nvals_uniform) * dR
C_uniform = 1 - (area_uniform / flat_flux)

print(f"Critical Bonnor-Ebert sphere: C = {C_BE:.2f}")
print(f"Uniform sphere: C = {C_uniform:.2f}")
print(f"Central-to-edge column density ratio (BE): {N_peak/N_edge:.2f}")

# Plotting
plt.figure(figsize=(7,5))
plt.plot(Rvals, Nvals/N_peak, label='Critical BE Sphere')
plt.plot(Rvals, Nvals_uniform/N_peak, '--', label='Uniform Sphere')
plt.xlabel("Projected Radius (normalized)")
plt.ylabel("Normalized Column Density")
plt.title("Column Density Profiles: Critical BE vs. Uniform Sphere")
plt.legend()
plt.tight_layout()
plt.show()
