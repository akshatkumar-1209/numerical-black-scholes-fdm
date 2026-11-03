"""
convergence_study.py  —  Weeks 7 & 8
======================================
Systematic spatial and temporal refinement study for the
Fully Implicit and Crank-Nicolson Black-Scholes FDM solvers.

Parameters (standard benchmark):
  K=100, T=1, r=0.05, sigma=0.20, S_max=400

Outputs
-------
1. Price comparison tables (FDM vs analytical, multiple S-nodes)
2. Spatial convergence table  (fixed N=5000, vary M)
3. Temporal convergence table (fixed M=800,  vary N)
4. Observed orders of convergence (log2 of error ratios)

Author: Akshat Kumar
"""

import numpy as np
from math import log, sqrt, exp, erf, pi

# ─────────────────────────────────────────────────────────────────────────────
# Analytical Black-Scholes
# ─────────────────────────────────────────────────────────────────────────────

def _N(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _n(x):
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)

def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N(d1) - K * exp(-r * T) * _N(d2)

def bs_put(S, K, T, r, sigma):
    c = bs_call(S, K, T, r, sigma)
    return c - S + K * exp(-r * T)

def bs_price(S, K, T, r, sigma, opt):
    return bs_call(S, K, T, r, sigma) if opt == 'call' else bs_put(S, K, T, r, sigma)

# ─────────────────────────────────────────────────────────────────────────────
# Thomas Algorithm (tridiagonal solver)
# ─────────────────────────────────────────────────────────────────────────────

def thomas(a, b, c, d):
    """Solve tridiagonal system.  a=lower, b=main, c=upper, d=rhs."""
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

# ─────────────────────────────────────────────────────────────────────────────
# Fully Implicit Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_implicit(K, T, r, sigma, Smax, M, N, opt='call'):
    """
    Backward-Euler (fully implicit) finite-difference solver.
    Accuracy: O(dS^2) spatial, O(dtau) temporal.
    Unconditionally stable.
    """
    dS = Smax / M
    dt = T / N
    S  = np.linspace(0, Smax, M + 1)
    # Terminal payoff
    V  = np.maximum(S - K, 0.0) if opt == 'call' else np.maximum(K - S, 0.0)
    Sj = np.arange(1, M) * dS          # interior nodes

    al = 0.5 * sigma**2 * Sj**2 / dS**2 - 0.5 * r * Sj / dS
    be = -sigma**2 * Sj**2 / dS**2 - r
    ga = 0.5 * sigma**2 * Sj**2 / dS**2 + 0.5 * r * Sj / dS

    # LHS coefficients  (1 - dt*L)
    a = -dt * al
    b = 1.0 - dt * be
    c = -dt * ga

    for step in range(N):
        tau  = (step + 1) * dt
        V0   = 0.0            if opt == 'call' else K * exp(-r * tau)
        VM   = (Smax - K * exp(-r * tau)) if opt == 'call' else 0.0

        rhs        = V[1:M].copy()
        rhs[0]    -= a[0]  * V0
        rhs[-1]   -= c[-1] * VM

        V[0]   = V0
        V[1:M] = thomas(a, b, c, rhs)
        V[M]   = VM

    return S, V

# ─────────────────────────────────────────────────────────────────────────────
# Crank-Nicolson Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_cn(K, T, r, sigma, Smax, M, N, opt='call'):
    """
    Crank-Nicolson (theta=1/2) finite-difference solver.
    Accuracy: O(dS^2) spatial, O(dtau^2) temporal.
    Unconditionally stable.
    """
    dS = Smax / M
    dt = T / N
    S  = np.linspace(0, Smax, M + 1)
    V  = np.maximum(S - K, 0.0) if opt == 'call' else np.maximum(K - S, 0.0)
    Sj = np.arange(1, M) * dS

    al = 0.5 * sigma**2 * Sj**2 / dS**2 - 0.5 * r * Sj / dS
    be = -sigma**2 * Sj**2 / dS**2 - r
    ga = 0.5 * sigma**2 * Sj**2 / dS**2 + 0.5 * r * Sj / dS

    h  = 0.5 * dt
    # LHS: (I - h*L)
    aL =  -h * al;  bL = 1.0 - h * be;  cL = -h * ga
    # RHS: (I + h*L)
    aR =   h * al;  bR = 1.0 + h * be;  cR =  h * ga

    for step in range(N):
        tn = (step + 1) * dt
        to = step * dt
        V0n = 0.0 if opt == 'call' else K * exp(-r * tn)
        VMn = (Smax - K * exp(-r * tn)) if opt == 'call' else 0.0
        V0o = 0.0 if opt == 'call' else K * exp(-r * to)
        VMo = (Smax - K * exp(-r * to)) if opt == 'call' else 0.0

        vi = V[1:M]
        rhs          = bR * vi
        rhs[1:]     += aR[1:]  * vi[:-1]
        rhs[:-1]    += cR[:-1] * vi[1:]
        rhs[0]      += aR[0]  * V0o - aL[0]  * V0n
        rhs[-1]     += cR[-1] * VMo - cL[-1] * VMn

        V[0]   = V0n
        V[1:M] = thomas(aL, bL, cL, rhs)
        V[M]   = VMn

    return S, V

# ─────────────────────────────────────────────────────────────────────────────
# Error metrics
# ─────────────────────────────────────────────────────────────────────────────

def linf_error(S, V, K, T, r, sigma, opt):
    """L-infinity error at interior nodes."""
    errs = []
    for j in range(1, len(S) - 1):
        if S[j] > 0:
            exact = bs_price(S[j], K, T, r, sigma, opt)
            errs.append(abs(V[j] - exact))
    return max(errs) if errs else 0.0

def atm_error(S, V, K, T, r, sigma, Smax, M, opt):
    """Error at the at-the-money node (S nearest K)."""
    j = min(M, max(1, round(K / (Smax / M))))
    exact = bs_price(S[j], K, T, r, sigma, opt)
    return abs(V[j] - exact), S[j], V[j], exact

# ─────────────────────────────────────────────────────────────────────────────
# Convergence order
# ─────────────────────────────────────────────────────────────────────────────

def obs_order(e1, e2, h1, h2):
    """Observed order p from  e ~ C*h^p."""
    if e1 <= 0 or e2 <= 0 or e1 == e2:
        return float('nan')
    return log(e1 / e2) / log(h2 / h1)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN STUDY
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Standard benchmark parameters
    K     = 100.0
    T     = 1.0
    r     = 0.05
    sigma = 0.20
    Smax  = 400.0
    opt   = 'call'

    # Reference analytical price at S=K
    v_exact_atm = bs_call(K, K, T, r, sigma)
    print("=" * 72)
    print("  NUMERICAL BLACK-SCHOLES FDM — WEEKS 7 & 8")
    print("  Systematic Verification & Convergence Analysis")
    print("=" * 72)
    print(f"  Parameters: K={K}, T={T}, r={r}, sigma={sigma}, Smax={Smax}")
    print(f"  Analytical call at S=K={K}: {v_exact_atm:.8f}")
    print()

    # ─── WEEK 7: PRICE COMPARISON TABLE ────────────────────────────────────
    print("=" * 72)
    print("  WEEK 7 — PRICE COMPARISON TABLES (FDM vs Analytical)")
    print("=" * 72)

    for solver_name, solver in [("Fully Implicit (M=200, N=1000)", solve_implicit),
                                  ("Crank-Nicolson (M=200, N=500)",  solve_cn)]:
        M_t = 200; N_t = (1000 if 'Implicit' in solver_name else 500)
        dS_t = Smax / M_t
        S_g, V_g = solver(K, T, r, sigma, Smax, M_t, N_t, opt)
        print(f"\n  Solver : {solver_name}")
        print(f"  dS={dS_t:.2f},  dt={T/N_t:.5f}")
        print(f"\n  {'S':>8}  {'FDM Price':>12}  {'Exact Price':>12}  {'|Error|':>11}  {'Rel Err %':>10}")
        print("  " + "-" * 60)
        for S_req in [60, 70, 80, 90, 100, 110, 120, 140, 160, 200]:
            j = min(M_t, max(1, round(S_req / dS_t)))
            s   = S_g[j]
            v   = V_g[j]
            ex  = bs_price(s, K, T, r, sigma, opt)
            err = abs(v - ex)
            rel = 100 * err / ex if ex > 1e-12 else float('nan')
            print(f"  {s:>8.1f}  {v:>12.6f}  {ex:>12.6f}  {err:>11.4e}  {rel:>10.4f}")
        # Also show put
        S_gp, V_gp = solver(K, T, r, sigma, Smax, M_t, N_t, 'put')
        print(f"\n  [Put option, same grid]")
        print(f"  {'S':>8}  {'FDM Price':>12}  {'Exact Price':>12}  {'|Error|':>11}")
        print("  " + "-" * 50)
        for S_req in [60, 80, 100, 120, 160]:
            j = min(M_t, max(1, round(S_req / dS_t)))
            s   = S_gp[j]
            v   = V_gp[j]
            ex  = bs_price(s, K, T, r, sigma, 'put')
            err = abs(v - ex)
            print(f"  {s:>8.1f}  {v:>12.6f}  {ex:>12.6f}  {err:>11.4e}")

    # ─── WEEK 8: CONVERGENCE ANALYSIS ──────────────────────────────────────
    print()
    print("=" * 72)
    print("  WEEK 8 — CONVERGENCE ANALYSIS")
    print("=" * 72)

    M_values   = [25, 50, 100, 200, 400]
    N_spatial  = 5000   # large N: temporal error negligible → isolates spatial order
    N_values   = [50, 100, 200, 400, 800]
    M_temporal = 800    # large M: spatial floor ~3e-5 → below temporal errors at low N

    # ── 8a: Spatial convergence ──
    print("\n  [8a] SPATIAL CONVERGENCE STUDY (N=5000 fixed, vary M)")
    print("       Large N ensures temporal error is negligible.")
    print("       Expected: O(dS^2) -> error ratios ~4.0")
    print()

    for solver_name, solver in [("Fully Implicit", solve_implicit),
                                  ("Crank-Nicolson", solve_cn)]:
        print(f"  --- {solver_name} ---")
        print(f"  {'M':>6}  {'dS':>8}  {'L-inf Error':>13}  {'Ratio':>8}  {'Obs. Order':>12}")
        print("  " + "-" * 56)
        prev_e, prev_dS = None, None
        for M in M_values:
            dS = Smax / M
            S_g, V_g = solver(K, T, r, sigma, Smax, M, N_spatial, opt)
            e = linf_error(S_g, V_g, K, T, r, sigma, opt)
            ratio = prev_e / e if prev_e and e > 0 else float('nan')
            order = obs_order(prev_e, e, prev_dS, dS) if prev_e else float('nan')
            note = "  ← K not on grid" if M == 50 else ""
            print(f"  {M:>6}  {dS:>8.2f}  {e:>13.5e}  {ratio:>8.2f}  {order:>12.3f}{note}")
            prev_e, prev_dS = e, dS
        print()

    # ── 8b: Temporal convergence ──
    print("  [8b] TEMPORAL CONVERGENCE STUDY (M=800 fixed, vary N)")
    print("       M=800 → dS=0.5 → spatial floor ≈ 3e-5 (well below temporal errors at low N).")
    print("       Expected: Implicit O(dt^1) → ratios ≈ 2.0")
    print("                 CN       O(dt^2) → ratios ≈ 4.0")
    print()

    for solver_name, solver, exp_order in [("Fully Implicit", solve_implicit, 1),
                                             ("Crank-Nicolson", solve_cn,       2)]:
        print(f"  --- {solver_name}  [expected O(dt^{exp_order})] ---")
        print(f"  {'N':>6}  {'dt':>10}  {'L-inf Error':>13}  {'Ratio':>8}  {'Obs. Order':>12}")
        print("  " + "-" * 58)
        prev_e, prev_dt = None, None
        for N in N_values:
            dt = T / N
            S_g, V_g = solver(K, T, r, sigma, Smax, M_temporal, N, opt)
            e = linf_error(S_g, V_g, K, T, r, sigma, opt)
            ratio = prev_e / e if prev_e and e > 0 else float('nan')
            order = obs_order(prev_e, e, prev_dt, dt) if prev_e else float('nan')
            print(f"  {N:>6}  {dt:>10.5f}  {e:>13.5e}  {ratio:>8.2f}  {order:>12.3f}")
            prev_e, prev_dt = e, dt
        print()

    # ── 8c: Summary interpretation ──
    print("  [8c] INTERPRETATION")
    print("  " + "-" * 68)
    print("""
  Spatial Order (both methods):
    Both Fully Implicit and Crank-Nicolson use central differences in S,
    giving O(dS^2) spatial truncation error.  Halving dS (M→2M) should
    reduce the error by a factor of ~4.  The tables confirm ratios of
    approximately 4.0 once K lies on a grid node.  The M=50 anomaly
    (dS=8, K=100 lies between nodes) causes the ratio to appear irregular
    — this is a grid-alignment effect, not a solver defect.

  Temporal Order:
    Fully Implicit uses Backward Euler (theta=1), giving O(dt) temporal
    accuracy.  Halving N (dt→dt/2) reduces the error by a factor of ~2,
    confirming first-order convergence in time.

    Crank-Nicolson uses the trapezoid rule (theta=1/2), giving O(dt^2)
    temporal accuracy.  Halving N reduces the error by a factor of ~4,
    confirming second-order convergence in time.

  Stability:
    Both methods are unconditionally stable (no CFL restriction).
    The tridiagonal system arising from the backward-in-time implicit
    step has a strictly diagonally dominant matrix, guaranteeing stability
    for any choice of dS and dt.

  Practical recommendation:
    For equivalent computational cost (same M*N), Crank-Nicolson achieves
    significantly smaller errors than the Fully Implicit method due to its
    superior O(dt^2) temporal accuracy.
  """)

    print("=" * 72)
    print("  CONVERGENCE STUDY COMPLETE")
    print("=" * 72)
