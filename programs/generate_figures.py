"""
generate_figures.py
===================
Generates verification figures for:
1. Spatial convergence (log-log plot: Linf error vs dS, confirming O(dS^2))
2. Temporal convergence (log-log plot: Linf error vs dt, using same-grid
   Richardson reference M=200, N_ref=12800 to isolate temporal order without
   spatial floor interference; confirms O(dt) for Implicit and O(dt^2) for CN)
3. Numerical Greeks (Delta and Gamma profiles at T=1, plus short maturity T=0.1
   near-strike Gamma spike highlighting the payoff kink impact)
4. Non-smooth payoff & CN oscillations: Gamma under standard CN vs a corrected
   one-interval Rannacher startup (two Backward Euler half-steps)

Outputs saved to 'figures/' directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, exp, erf, pi

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------
# Analytical Functions
# ---------------------------------------------------------
def _N(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _n(x):
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)

def bs_call(S, K, T, r, sigma):
    if S <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N(d1) - K * exp(-r * T) * _N(d2)

def bs_delta(S, K, T, r, sigma):
    if S <= 0 or T <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return _N(d1)

def bs_gamma(S, K, T, r, sigma):
    if S <= 0 or T <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return _n(d1) / (S * sigma * sqrt(T))

# ---------------------------------------------------------
# Solvers
# ---------------------------------------------------------
def thomas(a, b, c, d):
    n = len(b)
    b, d = b.copy(), d.copy()
    x = np.zeros(n)
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x

def solve_implicit(K, T, r, sigma, Smax, M, N):
    dS = Smax / M
    dt = T / N
    S = np.linspace(0, Smax, M + 1)
    V = np.maximum(S - K, 0.0)
    Sj = np.arange(1, M) * dS
    al = 0.5 * sigma**2 * Sj**2 / dS**2 - 0.5 * r * Sj / dS
    be = -sigma**2 * Sj**2 / dS**2 - r
    ga = 0.5 * sigma**2 * Sj**2 / dS**2 + 0.5 * r * Sj / dS
    a = -dt * al
    b = 1.0 - dt * be
    c = -dt * ga
    for step in range(N):
        tau = (step + 1) * dt
        V0 = 0.0
        VM = (Smax - K * exp(-r * tau))
        rhs = V[1:M].copy()
        rhs[0] -= a[0] * V0
        rhs[-1] -= c[-1] * VM
        V[0] = V0
        V[1:M] = thomas(a, b, c, rhs)
        V[M] = VM
    return S, V

def solve_cn(K, T, r, sigma, Smax, M, N):
    dS = Smax / M
    dt = T / N
    S = np.linspace(0, Smax, M + 1)
    V = np.maximum(S - K, 0.0)
    Sj = np.arange(1, M) * dS
    al = 0.5 * sigma**2 * Sj**2 / dS**2 - 0.5 * r * Sj / dS
    be = -sigma**2 * Sj**2 / dS**2 - r
    ga = 0.5 * sigma**2 * Sj**2 / dS**2 + 0.5 * r * Sj / dS
    h = 0.5 * dt
    aL, bL, cL = -h * al, 1.0 - h * be, -h * ga
    aR, bR, cR = h * al, 1.0 + h * be, h * ga
    for step in range(N):
        tn = (step + 1) * dt
        to = step * dt
        V0n = 0.0; VMn = (Smax - K * exp(-r * tn))
        V0o = 0.0; VMo = (Smax - K * exp(-r * to))
        vi = V[1:M]
        rhs = bR * vi
        rhs[1:] += aR[1:] * vi[:-1]
        rhs[:-1] += cR[:-1] * vi[1:]
        rhs[0] += aR[0] * V0o - aL[0] * V0n
        rhs[-1] += cR[-1] * VMo - cL[-1] * VMn
        V[0] = V0n
        V[1:M] = thomas(aL, bL, cL, rhs)
        V[M] = VMn
    return S, V

def solve_rannacher(K, T, r, sigma, Smax, M, N, n_half_steps=2):
    dS = Smax / M
    dt = T / N
    S = np.linspace(0, Smax, M + 1)
    V = np.maximum(S - K, 0.0)
    Sj = np.arange(1, M) * dS
    al = 0.5 * sigma**2 * Sj**2 / dS**2 - 0.5 * r * Sj / dS
    be = -sigma**2 * Sj**2 / dS**2 - r
    ga = 0.5 * sigma**2 * Sj**2 / dS**2 + 0.5 * r * Sj / dS

    h = 0.5 * dt
    aL, bL, cL = -h * al, 1.0 - h * be, -h * ga
    aR, bR, cR =  h * al, 1.0 + h * be,  h * ga

    dt_half = 0.5 * dt
    a_half = -dt_half * al; b_half = 1.0 - dt_half * be; c_half = -dt_half * ga

    if n_half_steps < 0 or n_half_steps > 2 * N or n_half_steps % 2:
        raise ValueError("n_half_steps must be even and between 0 and 2*N")
    num_half = n_half_steps
    for s in range(num_half):
        tau = (s + 1) * dt_half
        V0 = 0.0
        VM = (Smax - K * exp(-r * tau))
        rhs = V[1:M].copy()
        rhs[0] -= a_half[0] * V0
        rhs[-1] -= c_half[-1] * VM
        V[0] = V0
        V[1:M] = thomas(a_half, b_half, c_half, rhs)
        V[M] = VM

    start_step = num_half // 2
    for step in range(start_step, N):
        tn = (step + 1) * dt
        to = step * dt
        V0n = 0.0; VMn = (Smax - K * exp(-r * tn))
        V0o = 0.0; VMo = (Smax - K * exp(-r * to))
        vi = V[1:M]
        rhs = bR * vi
        rhs[1:] += aR[1:] * vi[:-1]
        rhs[:-1] += cR[:-1] * vi[1:]
        rhs[0] += aR[0] * V0o - aL[0] * V0n
        rhs[-1] += cR[-1] * VMo - cL[-1] * VMn
        V[0] = V0n
        V[1:M] = thomas(aL, bL, cL, rhs)
        V[M] = VMn
    return S, V

def max_error(S, V, K, T, r, sigma):
    errs = [abs(V[j] - bs_call(S[j], K, T, r, sigma)) for j in range(1, len(S) - 1) if S[j] > 0]
    return max(errs)

# Standard benchmark parameters
K = 100.0; T = 1.0; r = 0.05; sigma = 0.20; Smax = 400.0

# ---------------------------------------------------------
# Figure 1: Spatial Convergence
# ---------------------------------------------------------
print("Generating Fig 1: Spatial Convergence...")
M_list = [25, 50, 100, 200, 400]
N_fixed = 5000
dS_vals = [Smax / m for m in M_list]
err_imp_spatial = []
err_cn_spatial = []

for m in M_list:
    Simp, Vimp = solve_implicit(K, T, r, sigma, Smax, m, N_fixed)
    Scn, Vcn = solve_cn(K, T, r, sigma, Smax, m, N_fixed)
    err_imp_spatial.append(max_error(Simp, Vimp, K, T, r, sigma))
    err_cn_spatial.append(max_error(Scn, Vcn, K, T, r, sigma))

plt.figure(figsize=(7, 5))
plt.loglog(dS_vals, err_imp_spatial, 'o-', color='#1f77b4', lw=2, ms=6, label=r'Fully Implicit ($N=5000$)')
plt.loglog(dS_vals, err_cn_spatial, 's--', color='#d62728', lw=2, ms=6, label=r'Crank-Nicolson ($N=5000$)')

# Reference slope O(dS^2) matching M=100..400
ref_x = np.array([dS_vals[2], dS_vals[-1]])
ref_y = err_cn_spatial[2] * (ref_x / ref_x[0])**2.0
plt.loglog(ref_x, ref_y, 'k:', lw=1.5, label=r'Reference $\mathcal{O}(\Delta S^2)$ Slope')

plt.xlabel(r'Spatial Step $\Delta S$ (log scale)')
plt.ylabel(r'$L_\infty$ Error vs Analytical (log scale)')
plt.title(r'Spatial Refinement (alignment-sensitive near the payoff kink)')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.tight_layout()
fig1_path = os.path.join(FIG_DIR, "fig1_spatial_convergence.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"  Saved: {fig1_path}")

# ---------------------------------------------------------
# Figure 2: Temporal Convergence (Same-Grid Richardson Reference)
# ---------------------------------------------------------
print("Generating Fig 2: Temporal Convergence (Same-Grid Reference)...")
N_list = [50, 100, 200, 400, 800]
M_temporal = 200
N_ref = 12800
dt_vals = [T / n for n in N_list]

_, V_ref_imp = solve_implicit(K, T, r, sigma, Smax, M_temporal, N_ref)
_, V_ref_cn  = solve_cn(K, T, r, sigma, Smax, M_temporal, N_ref)

err_imp_temp = []
err_cn_temp = []

for n in N_list:
    _, Vimp = solve_implicit(K, T, r, sigma, Smax, M_temporal, n)
    _, Vcn  = solve_cn(K, T, r, sigma, Smax, M_temporal, n)
    err_imp_temp.append(float(np.max(np.abs(Vimp[1:-1] - V_ref_imp[1:-1]))))
    err_cn_temp.append(float(np.max(np.abs(Vcn[1:-1] - V_ref_cn[1:-1]))))

plt.figure(figsize=(7, 5))
plt.loglog(dt_vals, err_imp_temp, 'o-', color='#1f77b4', lw=2, ms=6,
           label=r'Fully Implicit (Backward Euler, $M=200$)')
plt.loglog(dt_vals, err_cn_temp, 's--', color='#d62728', lw=2, ms=6,
           label=r'Crank-Nicolson ($\theta=1/2$, $M=200$)')

# Reference slopes
ref_x = np.array([dt_vals[0], dt_vals[-1]])
ref_y1 = err_imp_temp[0] * (ref_x / ref_x[0])**1.0
ref_y2 = err_cn_temp[0] * (ref_x / ref_x[0])**2.0
plt.loglog(ref_x, ref_y1, 'b:', lw=1.3, label=r'Theoretical $\mathcal{O}(\Delta \tau)$ Slope')
plt.loglog(ref_x, ref_y2, 'r:', lw=1.3, label=r'Theoretical $\mathcal{O}(\Delta \tau^2)$ Slope')

plt.xlabel(r'Time Step Size $\Delta \tau = T/N$ (log scale)')
plt.ylabel(r'$L_\infty$ Error vs $N_{\mathrm{ref}}=12800$ on Same Grid (log scale)')
plt.title(r'Temporal Convergence: Isolated $\mathcal{O}(\Delta \tau)$ and $\mathcal{O}(\Delta \tau^2)$ Orders')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.tight_layout()
fig2_path = os.path.join(FIG_DIR, "fig2_temporal_convergence.png")
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"  Saved: {fig2_path}")

# ---------------------------------------------------------
# Figure 3: Numerical Greeks (T=1.0 and Short Maturity T=0.1)
# ---------------------------------------------------------
print("Generating Fig 3: Numerical Greeks...")
M_g = 400; N_g = 2000
dS_g = Smax / M_g
S_g, V_cn_g = solve_cn(K, T, r, sigma, Smax, M_g, N_g)

# Greek calculations at T=1.0
delta_cn = np.zeros(M_g + 1)
gamma_cn = np.zeros(M_g + 1)
delta_ex = np.array([bs_delta(s, K, T, r, sigma) for s in S_g])
gamma_ex = np.array([bs_gamma(s, K, T, r, sigma) for s in S_g])

for j in range(1, M_g):
    delta_cn[j] = (V_cn_g[j+1] - V_cn_g[j-1]) / (2.0 * dS_g)
    gamma_cn[j] = (V_cn_g[j+1] - 2.0 * V_cn_g[j] + V_cn_g[j-1]) / (dS_g**2)
delta_cn[0] = (V_cn_g[1] - V_cn_g[0]) / dS_g
gamma_cn[0] = gamma_ex[0]
delta_cn[-1] = (V_cn_g[-1] - V_cn_g[-2]) / dS_g
gamma_cn[-1] = gamma_ex[-1]

# Short maturity T=0.1 to reveal kink behavior
T_short = 0.1
S_sh, V_cn_sh = solve_cn(K, T_short, r, sigma, Smax, M_g, N_g)
gamma_cn_sh = np.zeros(M_g + 1)
gamma_ex_sh = np.array([bs_gamma(s, K, T_short, r, sigma) for s in S_sh])
for j in range(1, M_g):
    gamma_cn_sh[j] = (V_cn_sh[j+1] - 2.0 * V_cn_sh[j] + V_cn_sh[j-1]) / (dS_g**2)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

# Panel 1: Delta Profile (T=1.0)
mask = (S_g >= 40) & (S_g <= 180)
ax1.plot(S_g[mask], delta_ex[mask], 'k-', lw=2.2, label='Analytical Delta')
ax1.plot(S_g[mask], delta_cn[mask], 'r--', lw=1.6, label='FDM Delta (CN)')
ax1.axvline(K, color='gray', ls=':', label=f'Strike K={K:.0f}')
ax1.set_xlabel('Spot Price S')
ax1.set_ylabel(r'Delta $\Delta = \partial V / \partial S$')
ax1.set_title(r'Delta Profile ($T=1.0\,\mathrm{y}$)')
ax1.legend(loc='upper left')
ax1.grid(True, ls='--', alpha=0.6)

# Panel 2: Gamma Profile (T=1.0)
ax2.plot(S_g[mask], gamma_ex[mask], 'k-', lw=2.2, label='Analytical Gamma')
ax2.plot(S_g[mask], gamma_cn[mask], 'b--', lw=1.6, label='FDM Gamma (CN)')
ax2.axvline(K, color='gray', ls=':', label=f'Strike K={K:.0f}')
ax2.set_xlabel('Spot Price S')
ax2.set_ylabel(r'Gamma $\Gamma = \partial^2 V / \partial S^2$')
ax2.set_title(r'Gamma Profile ($T=1.0\,\mathrm{y}$, Diffused Kink)')
ax2.legend(loc='upper right')
ax2.grid(True, ls='--', alpha=0.6)

# Panel 3: Short Maturity Gamma Comparison (T=0.1 vs T=1.0)
mask_kink = (S_g >= 75) & (S_g <= 125)
ax3.plot(S_g[mask_kink], gamma_ex_sh[mask_kink], 'k-', lw=2.0, label=r'Exact Gamma ($T=0.1\,\mathrm{y}$)')
ax3.plot(S_g[mask_kink], gamma_cn_sh[mask_kink], 'm--', lw=1.6, label=r'FDM Gamma ($T=0.1\,\mathrm{y}$)')
ax3.plot(S_g[mask_kink], gamma_ex[mask_kink], 'gray', ls='-', lw=1.2, label=r'Exact Gamma ($T=1.0\,\mathrm{y}$)')
ax3.axvline(K, color='gray', ls=':', label=f'Strike K={K:.0f}')
ax3.set_xlabel('Spot Price S')
ax3.set_ylabel(r'Gamma $\Gamma = \partial^2 V / \partial S^2$')
ax3.set_title(r'Short Maturity $T=0.1\,\mathrm{y}$ (Kink Concentration)')
ax3.legend(loc='upper right')
ax3.grid(True, ls='--', alpha=0.6)

plt.tight_layout()
fig3_path = os.path.join(FIG_DIR, "fig3_numerical_greeks.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"  Saved: {fig3_path}")

# ---------------------------------------------------------
# Figure 4: Payoff Kink, CN Oscillations & Rannacher Fix
# ---------------------------------------------------------
print("Generating Fig 4: Crank-Nicolson Oscillations vs Rannacher Smoothing...")
M_osc = 200; N_coarse = 5
dS_osc = Smax / M_osc

S_osc, V_pure_cn = solve_cn(K, T, r, sigma, Smax, M_osc, N_coarse)
_, V_ran = solve_rannacher(K, T, r, sigma, Smax, M_osc, N_coarse, n_half_steps=2)

gamma_pure = np.zeros(M_osc + 1)
gamma_ran  = np.zeros(M_osc + 1)
gamma_exact = np.array([bs_gamma(s, K, T, r, sigma) for s in S_osc])

for j in range(1, M_osc):
    gamma_pure[j] = (V_pure_cn[j+1] - 2.0 * V_pure_cn[j] + V_pure_cn[j-1]) / (dS_osc**2)
    gamma_ran[j]  = (V_ran[j+1]  - 2.0 * V_ran[j]  + V_ran[j-1])  / (dS_osc**2)

mask_zoom = (S_osc >= 85) & (S_osc <= 115)

plt.figure(figsize=(8, 5))
plt.plot(S_osc[mask_zoom], gamma_exact[mask_zoom], 'k-', lw=2.5, label='Analytical Gamma (Exact Benchmark)')
plt.plot(S_osc[mask_zoom], gamma_pure[mask_zoom], 'ro-', lw=1.8, ms=5, label=r'Standard Crank-Nicolson (Spurious $\Gamma$ Oscillation)')
plt.plot(S_osc[mask_zoom], gamma_ran[mask_zoom], 'gs--', lw=2.0, ms=5, label='Rannacher (2 BE half-steps over first interval)')
plt.axvline(K, color='gray', ls=':', label=f'Payoff Kink at Strike K={K:.0f}')

plt.xlabel('Spot Price S')
plt.ylabel(r'Gamma $\Gamma = \partial^2 V / \partial S^2$')
plt.title(r'Treatment of Payoff Kink: Standard CN vs Rannacher Smoothing ($M=200, N=5$)')
plt.legend()
plt.grid(True, ls='--', alpha=0.6)
plt.tight_layout()
fig4_path = os.path.join(FIG_DIR, "fig4_cn_oscillations_rannacher.png")
plt.savefig(fig4_path, dpi=300)
plt.close()
print(f"  Saved: {fig4_path}")

print("\nAll figures generated successfully.")
