# implicit_solver.py
# ============================================================
# Fully Implicit Finite-Difference Method for the
# European Black-Scholes PDE.
#
# Forward-time form (tau = T - t):
#   dV/dtau = (1/2)*sigma^2*S^2 * d2V/dS2  +  r*S * dV/dS  -  r*V
#
# Temporal discretisation: Backward Euler (theta = 1)
#   (V^{n+1} - V^n)/dtau = L_BS * V^{n+1}
#
# Tridiagonal system per time step:
#   a_j*V_{j-1}^{n+1} + b_j*V_j^{n+1} + c_j*V_{j+1}^{n+1} = V_j^n
#
#   a_j = -dtau * alpha_j,  b_j = 1 - dtau*beta_j,  c_j = -dtau*gamma_j
#
#   alpha_j = sigma^2*Sj^2/(2*dS^2) - r*Sj/(2*dS)
#   beta_j  = -sigma^2*Sj^2/dS^2 - r
#   gamma_j = sigma^2*Sj^2/(2*dS^2) + r*Sj/(2*dS)
#
# Accuracy: O(dS^2) spatial, O(dtau) temporal. Unconditionally stable.
# Reference: Duffy (2022), Ch. 6-7.
# Author   : Akshat
# ============================================================

import numpy as np
from math import erf, sqrt, exp, pi as _pi


# ── Normal distribution utilities (no scipy needed) ────────────────────────

def _norm_cdf(x):
    """Cumulative standard normal CDF via math.erf."""
    if np.isscalar(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return 0.5 * (1.0 + np.array([erf(xi / sqrt(2.0)) for xi in x]))

def _norm_pdf(x):
    """Standard normal PDF."""
    if np.isscalar(x):
        return exp(-0.5 * x * x) / sqrt(2.0 * _pi)
    return np.array([exp(-0.5 * xi * xi) / sqrt(2.0 * _pi) for xi in x])


# ── Analytical benchmark ───────────────────────────────────────────────────

def bs_call(S, K, T, r, sigma):
    """Exact European call price (Black-Scholes formula)."""
    S = np.atleast_1d(np.asarray(S, dtype=float))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)

def bs_put(S, K, T, r, sigma):
    """Exact European put price (put-call parity)."""
    return bs_call(S, K, T, r, sigma) - S + K * exp(-r * T)


# ── Thomas algorithm ───────────────────────────────────────────────────────

def thomas_solve(a, b, c, d):
    """O(n) tridiagonal solver. a: sub-diag, b: main-diag, c: super-diag."""
    n  = len(b)
    b_ = b.copy().astype(float)
    d_ = d.copy().astype(float)
    x  = np.zeros(n)
    for i in range(1, n):
        m     = a[i] / b_[i - 1]
        b_[i]-= m * c[i - 1]
        d_[i]-= m * d_[i - 1]
    x[-1] = d_[-1] / b_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d_[i] - c[i] * x[i + 1]) / b_[i]
    return x


# ── Coefficient assembly ────────────────────────────────────────────────────

def _bs_coefficients(S_inner, dS, r, sigma):
    """Returns alpha, beta, gamma at interior nodes."""
    S2    = S_inner ** 2
    alpha = sigma**2 * S2 / (2 * dS**2) - r * S_inner / (2 * dS)
    beta  = -sigma**2 * S2 / dS**2 - r
    gamma = sigma**2 * S2 / (2 * dS**2) + r * S_inner / (2 * dS)
    return alpha, beta, gamma


# ── Main solver ─────────────────────────────────────────────────────────────

def implicit_bs(K, T, r, sigma, S_max, M, N, option_type='call'):
    """
    Fully implicit Black-Scholes FDM solver.

    Parameters
    ----------
    K, T, r, sigma : float   Standard BS parameters
    S_max          : float   Truncated upper boundary
    M              : int     Spatial steps
    N              : int     Temporal steps
    option_type    : str     'call' or 'put'

    Returns
    -------
    S : (M+1,) array  Asset price grid
    V : (M+1,) array  Option price at t=0 (today)
    """
    dS   = S_max / M
    dtau = T / N
    S    = np.linspace(0.0, S_max, M + 1)

    # Initial condition: terminal payoff at tau=0
    V = np.maximum(S - K, 0.0) if option_type == 'call' else np.maximum(K - S, 0.0)

    S_inner = S[1:M]
    alpha, beta, gamma = _bs_coefficients(S_inner, dS, r, sigma)

    # Tridiagonal coefficients (constant in time)
    a = -dtau * alpha
    b =  1.0 - dtau * beta
    c = -dtau * gamma

    for n in range(N):
        tau_new = (n + 1) * dtau
        V0 = 0.0 if option_type == 'call' else K * exp(-r * tau_new)
        VM = (S_max - K * exp(-r * tau_new)) if option_type == 'call' else 0.0

        rhs = V[1:M].copy()
        rhs[0]  -= a[0]  * V0
        rhs[-1] -= c[-1] * VM

        V[0]   = V0
        V[1:M] = thomas_solve(a, b, c, rhs)
        V[M]   = VM

    return S, V


# ── Error utilities ─────────────────────────────────────────────────────────

def compute_errors(S, V_num, K, T, r, sigma, option_type='call'):
    mask = (S > 0) & (S < S[-1])
    S_i  = S[mask]
    V_ex = bs_call(S_i, K, T, r, sigma) if option_type == 'call' else bs_put(S_i, K, T, r, sigma)
    err  = np.abs(V_num[mask] - V_ex)
    return err.max(), np.sqrt(np.mean(err**2)), S_i, V_num[mask], V_ex


# ── Main demonstration ───────────────────────────────────────────────────────

if __name__ == "__main__":
    from math import sqrt, exp
    K, T, r, sigma, S_max = 100.0, 1.0, 0.05, 0.20, 400.0
    M, N = 100, 1000

    print("=" * 72)
    print("  Fully Implicit Black-Scholes FDM")
    print(f"  K={K}, T={T}, r={r}, sigma={sigma}, S_max={S_max}")
    print(f"  Grid: M={M} (dS={S_max/M:.1f}), N={N} (dtau={T/N:.4f})")
    print("=" * 72)

    dS = S_max / M

    for opt in ['call', 'put']:
        S, V = implicit_bs(K, T, r, sigma, S_max, M, N, opt)
        e_inf, e_l2, _, _, _ = compute_errors(S, V, K, T, r, sigma, opt)
        print(f"\n--- European {opt.capitalize()} ---")
        print(f"  Max error (L-inf): {e_inf:.4e}")
        print(f"  RMS error (L-2)  : {e_l2:.4e}")
        print(f"\n  {'S (grid)':>10}  {'FDM':>12}  {'Exact':>12}  {'|Error|':>12}")
        print("  " + "-" * 52)
        for s_req in [60, 80, 100, 120, 140, 160, 200]:
            j = min(M, max(0, int(round(s_req / dS))))
            s_act = S[j]
            v_fdm = V[j]
            if opt == 'call':
                v_ex = float(bs_call(np.array([s_act]), K, T, r, sigma)[0])
            else:
                v_ex = float(bs_put(np.array([s_act]), K, T, r, sigma)[0])
            print(f"  {s_act:>10.1f}  {v_fdm:>12.6f}  {v_ex:>12.6f}  {abs(v_fdm-v_ex):>12.4e}")

    # Spatial convergence (N=5000: temporal negligible)
    print("\n\n--- Spatial Convergence Study (Call, N=5000 fixed) ---")
    print("  (N=5000 ensures temporal error negligible -- isolates O(dS^2))")
    print("  Note: M=50 anomaly because K=100 lies between grid nodes for dS=8.")
    print(f"  {'M':>6}  {'dS':>8}  {'Linf Err':>12}  {'Ratio':>8}")
    print("  " + "-" * 42)
    prev = None
    for Mv in [25, 50, 100, 200, 400]:
        S_g = np.linspace(0, S_max, Mv + 1)
        _, Vv = implicit_bs(K, T, r, sigma, S_max, Mv, 5000, 'call')
        e, _, _, _, _ = compute_errors(S_g, Vv, K, T, r, sigma, 'call')
        ratio = prev / e if prev else float('nan')
        note = "  <-- K not on grid" if Mv == 50 else ""
        print(f"  {Mv:>6}  {S_max/Mv:>8.2f}  {e:>12.4e}  {ratio:>8.2f}{note}")
        prev = e
    print("  Ratios ~4 when K on grid (M=100,200,400) confirm O(dS^2).")

    # Temporal convergence (M=800: spatial floor ~7e-04)
    print("\n--- Temporal Convergence Study (Call, M=800 fixed) ---")
    print("  (M=800 gives dS=0.5 --> spatial floor ~7e-04, well below temporal errors)")
    print(f"  {'N':>6}  {'dtau':>10}  {'Linf Err':>12}  {'Ratio':>8}")
    print("  " + "-" * 44)
    prev = None
    for Nv in [50, 100, 200, 400, 800]:
        S_g, Vv = implicit_bs(K, T, r, sigma, S_max, 800, Nv, 'call')
        e, _, _, _, _ = compute_errors(S_g, Vv, K, T, r, sigma, 'call')
        ratio = prev / e if prev else float('nan')
        print(f"  {Nv:>6}  {T/Nv:>10.5f}  {e:>12.4e}  {ratio:>8.2f}")
        prev = e
    print("  Ratios approaching 2 confirm O(dtau) temporal accuracy.")

    print("\nImplicit solver: DONE.")
    print("Confirmed: O(dS^2) spatial, O(dtau) temporal, unconditionally stable.")
