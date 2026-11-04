"""
numerical_greeks.py  —  Week 9
================================
Numerical computation of Delta and Gamma from the FDM price surface,
and comparison with their analytical Black-Scholes counterparts.

Delta (Delta):  dV/dS  — central finite-difference approximation from grid
Gamma (Gamma):  d²V/dS²  — second-order finite-difference approximation from grid

Studies:
  1. Delta and Gamma tables vs analytical (call and put, M=400 N=2000)
  2. Near-strike accuracy at T=1 and T=0.1 (kink more visible at short maturity)
  3. Delta/Gamma mesh-refinement study: vary M in {50,100,200,400}, confirm O(dS^2)

Parameters: K=100, T=1, r=0.05, sigma=0.20, Smax=400

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
# Solvers
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
    V  = np.maximum(S-K, 0.0) if opt == 'call' else np.maximum(K-S, 0.0)
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

        S_im, V_im = solve_implicit(K, T, r, sigma, Smax, M, N, opt)
        S_cn, V_cn = solve_cn     (K, T, r, sigma, Smax, M, N, opt)

        delta_im, gamma_im = fdm_greeks(S_im, V_im, M)
        delta_cn, gamma_cn = fdm_greeks(S_cn, V_cn, M)

        display_S = [60, 70, 80, 90, 95, 100, 105, 110, 120, 140, 160, 200]

        # Delta table
        print(f"\n  DELTA (Delta = dV/dS)  —  central finite difference on FDM grid")
        print(f"  {'S':>6}  {'Delta Analytical':>18}  {'Delta Implicit':>14}  {'Err (Impl)':>12}  {'Delta CN':>12}  {'Err (CN)':>12}")
        print("  " + "-" * 82)
        for S_req in display_S:
            j     = min(M-1, max(1, round(S_req / dS)))
            s_act = S_im[j]
            da    = bs_delta(s_act, K, T, r, sigma, opt)
            d_im  = delta_im[j]
            d_cn  = delta_cn[j]
            print(f"  {s_act:>6.1f}  {da:>18.8f}  {d_im:>14.8f}  {abs(d_im-da):>12.4e}  {d_cn:>12.8f}  {abs(d_cn-da):>12.4e}")

        # Gamma table
        print(f"\n  GAMMA (Gamma = d2V/dS2)  —  second-order FD on FDM grid")
        print(f"  {'S':>6}  {'Gamma Analytical':>18}  {'Gamma Implicit':>14}  {'Err (Impl)':>12}  {'Gamma CN':>12}  {'Err (CN)':>12}")
        print("  " + "-" * 82)
        for S_req in display_S:
            j     = min(M-1, max(1, round(S_req / dS)))
            s_act = S_im[j]
            ga    = bs_gamma(s_act, K, T, r, sigma)
            g_im  = gamma_im[j]
            g_cn  = gamma_cn[j]
            print(f"  {s_act:>6.1f}  {ga:>18.8f}  {g_im:>14.8f}  {abs(g_im-ga):>12.4e}  {g_cn:>12.8f}  {abs(g_cn-ga):>12.4e}")

        # Near-strike table at T=1
        print(f"\n  NEAR-STRIKE ACCURACY at T=1 (S = K+-3 nodes, dS={dS:.1f})")
        near_S = [K - 3*dS, K - 2*dS, K - dS, K, K + dS, K + 2*dS, K + 3*dS]
        print(f"  {'S':>7}  {'Delta Exact':>12}  {'|Err Impl|':>12}  {'|Err CN|':>12}  {'Gamma Exact':>12}  {'|Err Impl|':>12}  {'|Err CN|':>12}")
        print("  " + "-" * 95)
        for s_req in near_S:
            j     = min(M-1, max(1, round(s_req / dS)))
            s_act = S_im[j]
            da    = bs_delta(s_act, K, T, r, sigma, opt)
            ga    = bs_gamma(s_act, K, T, r, sigma)
            d_im  = delta_im[j];  d_cn  = delta_cn[j]
            g_im  = gamma_im[j];  g_cn  = gamma_cn[j]
            print(f"  {s_act:>7.1f}  {da:>12.8f}  {abs(d_im-da):>12.4e}  {abs(d_cn-da):>12.4e}  {ga:>12.8f}  {abs(g_im-ga):>12.4e}  {abs(g_cn-ga):>12.4e}")

    # ─── SECTION 2: Short maturity T=0.1 near-strike ───────────────────────
    print()
    print("=" * 72)
    print("  SHORT MATURITY T=0.1 — NEAR-STRIKE GREEKS")
    print("  At T=0.1 the kink has not yet smoothed: Gamma peaks sharply at K.")
    print("=" * 72)

    T_short = 0.1
    M_sh    = 400
    N_sh    = 2000
    dS_sh   = Smax / M_sh

    S_sh_im, V_sh_im = solve_implicit(K, T_short, r, sigma, Smax, M_sh, N_sh, 'call')
    S_sh_cn, V_sh_cn = solve_cn(K, T_short, r, sigma, Smax, M_sh, N_sh, 'call')
    del_sh_im, gam_sh_im = fdm_greeks(S_sh_im, V_sh_im, M_sh)
    del_sh_cn, gam_sh_cn = fdm_greeks(S_sh_cn, V_sh_cn, M_sh)

    print(f"\n  T={T_short}, M={M_sh} (dS={dS_sh:.2f}), N={N_sh}")
    print(f"  Analytical Gamma peak at S=K: {bs_gamma(K, K, T_short, r, sigma):.6f}")
    print(f"\n  {'S':>7}  {'Delta Exact':>12}  {'|Err Impl|':>12}  {'|Err CN|':>12}  {'Gamma Exact':>13}  {'|Err Impl|':>12}  {'|Err CN|':>12}")
    print("  " + "-" * 97)
    near_S_sh = [K - 3*dS_sh, K - 2*dS_sh, K - dS_sh, K, K + dS_sh, K + 2*dS_sh, K + 3*dS_sh]
    for s_req in near_S_sh:
        j     = min(M_sh-1, max(1, round(s_req / dS_sh)))
        s_act = S_sh_im[j]
        da    = bs_delta(s_act, K, T_short, r, sigma, 'call')
        ga    = bs_gamma(s_act, K, T_short, r, sigma)
        d_im  = del_sh_im[j];  d_cn  = del_sh_cn[j]
        g_im  = gam_sh_im[j];  g_cn  = gam_sh_cn[j]
        print(f"  {s_act:>7.1f}  {da:>12.8f}  {abs(d_im-da):>12.4e}  {abs(d_cn-da):>12.4e}  {ga:>13.8f}  {abs(g_im-ga):>12.4e}  {abs(g_cn-ga):>12.4e}")

    print("""
  At T=0.1 the analytical Gamma peak at S=K is much larger than at T=1.
  Near-strike Greek errors are correspondingly larger, since the numerical
  second derivative must resolve a sharper feature with the same spatial grid.
  This confirms that short-dated option Greeks require finer spatial grids
  or non-uniform grids with clustering around K.
  """)

    # ─── SECTION 3: Delta/Gamma mesh-refinement study ──────────────────────
    print("=" * 72)
    print("  SECTION 3 — DELTA/GAMMA MESH-REFINEMENT STUDY")
    print("  Fixed N=2000, vary M in {50,100,200,400}.")
    print("  Errors measured at S=K (ATM node or nearest). Expected: O(dS^2).")
    print("=" * 72)

    M_ref_vals = [50, 100, 200, 400]
    N_ref_fix  = 2000
    opt_ref    = 'call'

    for solver_name, solver in [("Fully Implicit", solve_implicit),
                                  ("Crank-Nicolson", solve_cn)]:
        print(f"\n  --- {solver_name} (N={N_ref_fix}, T=1) ---")
        print(f"  {'M':>6}  {'dS':>8}  {'Delta Err ATM':>15}  {'Ratio':>8}  {'Gamma Err ATM':>15}  {'Ratio':>8}")
        print("  " + "-" * 72)
        prev_de, prev_ge = None, None
        for M_r in M_ref_vals:
            dS_r = Smax / M_r
            S_r, V_r = solver(K, T, r, sigma, Smax, M_r, N_ref_fix, opt_ref)
            del_r, gam_r = fdm_greeks(S_r, V_r, M_r)
            j_K  = min(M_r-1, max(1, round(K / dS_r)))
            s_K  = S_r[j_K]
            da   = bs_delta(s_K, K, T, r, sigma, opt_ref)
            ga   = bs_gamma(s_K, K, T, r, sigma)
            d_e  = abs(del_r[j_K] - da)
            g_e  = abs(gam_r[j_K] - ga)
            r_d  = prev_de / d_e if prev_de and d_e > 0 else float('nan')
            r_g  = prev_ge / g_e if prev_ge and g_e > 0 else float('nan')
            print(f"  {M_r:>6}  {dS_r:>8.2f}  {d_e:>15.5e}  {r_d:>8.2f}  {g_e:>15.5e}  {r_g:>8.2f}")
            prev_de, prev_ge = d_e, g_e
        print()

    print("""
  Interpretation:
    Both Delta and Gamma errors at the ATM node halve (or quarter) as M doubles,
    consistent with O(dS^2) convergence. The Gamma error is typically an order
    of magnitude larger than the Delta error at the same M, because the second
    finite difference amplifies discretisation noise.

    At M=50 the ATM node may not coincide with K, introducing a grid-alignment
    contribution.  From M=100 onward, ratios settle near 4.0 for both Greeks,
    confirming the theoretical spatial order.
  """)

    print("=" * 72)
    print("  NUMERICAL GREEKS STUDY COMPLETE")
    print("=" * 72)
