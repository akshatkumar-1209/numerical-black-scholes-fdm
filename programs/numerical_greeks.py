"""
numerical_greeks.py  —  Week 9
================================
Numerical computation of Delta and Gamma from the FDM price surface,
and comparison with their analytical Black-Scholes counterparts.

Delta (Δ):  dV/dS  — finite-difference approximation from grid
Gamma (Γ):  d²V/dS²  — second-order finite-difference approximation from grid

Special focus: behaviour near the strike price (S ≈ K), where the payoff
has a kink and gradients change rapidly.

Parameters: K=100, T=1, r=0.05, sigma=0.20, Smax=400, M=400, N=2000

Author: Akshat Kumar
"""

import numpy as np
from math import log, sqrt, exp, erf, pi

# ─────────────────────────────────────────────────────────────────────────────
# Analytical Black-Scholes functions
# ─────────────────────────────────────────────────────────────────────────────

def _N(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _n(x):
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)

def bs_price(S, K, T, r, sigma, opt='call'):
    if S <= 0 or T <= 0:
        return max(S - K, 0.0) if opt == 'call' else max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    c  = S * _N(d1) - K * exp(-r * T) * _N(d2)
    return c if opt == 'call' else c - S + K * exp(-r * T)

def bs_delta(S, K, T, r, sigma, opt='call'):
    """Analytical Delta."""
    if S <= 0 or T <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return _N(d1) if opt == 'call' else _N(d1) - 1.0

def bs_gamma(S, K, T, r, sigma):
    """Analytical Gamma (same for call and put by put-call parity)."""
    if S <= 0 or T <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return _n(d1) / (S * sigma * sqrt(T))

# ─────────────────────────────────────────────────────────────────────────────
# Thomas Algorithm
# ─────────────────────────────────────────────────────────────────────────────

def thomas(a, b, c, d):
    n  = len(b)
    b  = b.copy(); d = d.copy()
    x  = np.zeros(n)
    for i in range(1, n):
        m     = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x

# ─────────────────────────────────────────────────────────────────────────────
# Solvers (reused from convergence_study.py)
# ─────────────────────────────────────────────────────────────────────────────

def solve_implicit(K, T, r, sigma, Smax, M, N, opt='call'):
    dS = Smax / M; dt = T / N
    S  = np.linspace(0, Smax, M + 1)
    V  = np.maximum(S - K, 0.0) if opt == 'call' else np.maximum(K - S, 0.0)
    Sj = np.arange(1, M) * dS
    al = 0.5*sigma**2*Sj**2/dS**2 - 0.5*r*Sj/dS
    be = -sigma**2*Sj**2/dS**2 - r
    ga = 0.5*sigma**2*Sj**2/dS**2 + 0.5*r*Sj/dS
    a  = -dt*al; b = 1.0 - dt*be; c = -dt*ga
    for step in range(N):
        tau = (step+1)*dt
        V0  = 0.0 if opt=='call' else K*exp(-r*tau)
        VM  = (Smax - K*exp(-r*tau)) if opt=='call' else 0.0
        rhs = V[1:M].copy(); rhs[0] -= a[0]*V0; rhs[-1] -= c[-1]*VM
        V[0] = V0; V[1:M] = thomas(a,b,c,rhs); V[M] = VM
    return S, V

def solve_cn(K, T, r, sigma, Smax, M, N, opt='call'):
    dS = Smax/M; dt = T/N
    S  = np.linspace(0, Smax, M+1)
    V  = np.maximum(S-K, 0.0) if opt=='call' else np.maximum(K-S, 0.0)
    Sj = np.arange(1,M)*dS
    al = 0.5*sigma**2*Sj**2/dS**2 - 0.5*r*Sj/dS
    be = -sigma**2*Sj**2/dS**2 - r
    ga = 0.5*sigma**2*Sj**2/dS**2 + 0.5*r*Sj/dS
    h  = 0.5*dt
    aL=-h*al; bL=1.0-h*be; cL=-h*ga
    aR= h*al; bR=1.0+h*be; cR= h*ga
    for step in range(N):
        tn=(step+1)*dt; to=step*dt
        V0n=0.0 if opt=='call' else K*exp(-r*tn)
        VMn=(Smax-K*exp(-r*tn)) if opt=='call' else 0.0
        V0o=0.0 if opt=='call' else K*exp(-r*to)
        VMo=(Smax-K*exp(-r*to)) if opt=='call' else 0.0
        vi=V[1:M]
        rhs=bR*vi
        rhs[1:]  +=aR[1:]*vi[:-1]; rhs[:-1]+=cR[:-1]*vi[1:]
        rhs[0]   +=aR[0]*V0o-aL[0]*V0n; rhs[-1]+=cR[-1]*VMo-cL[-1]*VMn
        V[0]=V0n; V[1:M]=thomas(aL,bL,cL,rhs); V[M]=VMn
    return S, V

# ─────────────────────────────────────────────────────────────────────────────
# Numerical Greeks from FDM price surface
# ─────────────────────────────────────────────────────────────────────────────

def fdm_greeks(S, V, M):
    """
    Compute numerical Delta and Gamma at each interior grid node
    using central finite differences on the price surface.

    Delta_j = (V_{j+1} - V_{j-1}) / (2*dS)          [central, O(dS^2)]
    Gamma_j = (V_{j+1} - 2*V_j + V_{j-1}) / dS^2    [central, O(dS^2)]
    """
    dS    = S[1] - S[0]
    delta = np.zeros(M + 1)
    gamma = np.zeros(M + 1)
    for j in range(1, M):
        delta[j] = (V[j + 1] - V[j - 1]) / (2.0 * dS)
        gamma[j] = (V[j + 1] - 2.0 * V[j] + V[j - 1]) / (dS * dS)
    # Boundary: one-sided
    delta[0]  = (V[1] - V[0]) / dS
    gamma[0]  = (V[2] - 2*V[1] + V[0]) / (dS*dS)
    delta[-1] = (V[-1] - V[-2]) / dS
    gamma[-1] = (V[-1] - 2*V[-2] + V[-3]) / (dS*dS)
    return delta, gamma

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    K     = 100.0
    T     = 1.0
    r     = 0.05
    sigma = 0.20
    Smax  = 400.0
    M     = 400
    N     = 2000
    dS    = Smax / M

    print("=" * 72)
    print("  NUMERICAL BLACK-SCHOLES FDM — WEEK 9")
    print("  Numerical Delta & Gamma vs Analytical Black-Scholes Greeks")
    print("=" * 72)
    print(f"  Parameters: K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"  Grid: M={M} (dS={dS:.2f}), N={N} (dt={T/N:.5f})")
    print()

    for opt in ['call', 'put']:
        print("=" * 72)
        print(f"  EUROPEAN {opt.upper()} OPTION")
        print("=" * 72)

        # Solve with both methods
        S_im, V_im = solve_implicit(K, T, r, sigma, Smax, M, N, opt)
        S_cn, V_cn = solve_cn     (K, T, r, sigma, Smax, M, N, opt)

        delta_im, gamma_im = fdm_greeks(S_im, V_im, M)
        delta_cn, gamma_cn = fdm_greeks(S_cn, V_cn, M)

        # Display nodes near ATM and beyond
        display_S = [60, 70, 80, 90, 95, 100, 105, 110, 120, 140, 160, 200]

        # ── Delta table ──
        print(f"\n  DELTA (Δ = ∂V/∂S)  —  central finite difference on FDM grid")
        print(f"  {'S':>6}  {'Δ Analytical':>14}  {'Δ Implicit':>12}  {'Err (Impl)':>12}  {'Δ CN':>12}  {'Err (CN)':>12}")
        print("  " + "-" * 76)
        for S_req in display_S:
            j     = min(M-1, max(1, round(S_req / dS)))
            s_act = S_im[j]
            da    = bs_delta(s_act, K, T, r, sigma, opt)
            d_im  = delta_im[j]
            d_cn  = delta_cn[j]
            print(f"  {s_act:>6.1f}  {da:>14.8f}  {d_im:>12.8f}  {abs(d_im-da):>12.4e}  {d_cn:>12.8f}  {abs(d_cn-da):>12.4e}")

        # ── Gamma table ──
        print(f"\n  GAMMA (Γ = ∂²V/∂S²)  —  second-order FD on FDM grid")
        print(f"  {'S':>6}  {'Γ Analytical':>14}  {'Γ Implicit':>12}  {'Err (Impl)':>12}  {'Γ CN':>12}  {'Err (CN)':>12}")
        print("  " + "-" * 76)
        for S_req in display_S:
            j     = min(M-1, max(1, round(S_req / dS)))
            s_act = S_im[j]
            ga    = bs_gamma(s_act, K, T, r, sigma)
            g_im  = gamma_im[j]
            g_cn  = gamma_cn[j]
            print(f"  {s_act:>6.1f}  {ga:>14.8f}  {g_im:>12.8f}  {abs(g_im-ga):>12.4e}  {g_cn:>12.8f}  {abs(g_cn-ga):>12.4e}")

        # ── Near-strike accuracy ──
        print(f"\n  NEAR-STRIKE ACCURACY (S = K±5 nodes, i.e. dS={dS:.1f} steps around K={K})")
        near_S = [K - 3*dS, K - 2*dS, K - dS, K, K + dS, K + 2*dS, K + 3*dS]
        print(f"  {'S':>7}  {'Δ Exact':>12}  {'|Err Impl|':>12}  {'|Err CN|':>12}  {'Γ Exact':>12}  {'|Err Impl|':>12}  {'|Err CN|':>12}")
        print("  " + "-" * 90)
        for s_req in near_S:
            j     = min(M-1, max(1, round(s_req / dS)))
            s_act = S_im[j]
            da    = bs_delta(s_act, K, T, r, sigma, opt)
            ga    = bs_gamma(s_act, K, T, r, sigma)
            d_im  = delta_im[j];  d_cn  = delta_cn[j]
            g_im  = gamma_im[j];  g_cn  = gamma_cn[j]
            print(f"  {s_act:>7.1f}  {da:>12.8f}  {abs(d_im-da):>12.4e}  {abs(d_cn-da):>12.4e}  {ga:>12.8f}  {abs(g_im-ga):>12.4e}  {abs(g_cn-ga):>12.4e}")

    # ── Interpretation ──
    print()
    print("=" * 72)
    print("  INTERPRETATION OF NUMERICAL GREEKS")
    print("=" * 72)
    print("""
  Delta (Δ = ∂V/∂S):
    The central-difference formula  Δ_j = (V_{j+1} - V_{j-1}) / (2·dS)
    produces O(dS²) accuracy in smooth regions.  Both Implicit and CN grids
    give Delta errors of order 1e-4 to 1e-5 away from the strike.

    Near the strike (S ≈ K): the payoff max(S-K, 0) has a kink at S=K.
    As T→0, the smeared kink concentrates and Gamma spikes.  For T=1y the
    kink is well-smoothed by diffusion, so near-strike errors remain modest.
    For short-dated options, grid refinement around K is essential.

  Gamma (Γ = ∂²V/∂S²):
    Computed as  Γ_j = (V_{j+1} - 2V_j + V_{j-1}) / dS²
    This second-difference amplifies grid noise, so Gamma errors are
    typically an order of magnitude larger than Delta errors for the same
    grid.  Gamma peaks at S=K and decays away from the strike.

    CN vs Implicit: Both methods give comparable Greek accuracy at the
    same spatial resolution M, because the Greeks depend primarily on the
    spatial discretisation (both use O(dS²) central differences in S).
    The temporal superiority of CN (O(dt²) vs O(dt)) is less visible in
    Greeks unless very coarse time steps are used.

  Recommendation: To improve Greek accuracy near the strike, use a finer
  spatial grid (larger M) or a non-uniform grid with nodes clustered near K.
  """)

    print("=" * 72)
    print("  NUMERICAL GREEKS STUDY COMPLETE")
    print("=" * 72)
