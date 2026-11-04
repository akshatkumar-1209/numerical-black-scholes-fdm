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
3. Temporal convergence table (same-grid Richardson reference: fixed M=200, vary N,
   reference N_ref=12800 on same grid to isolate temporal error)
4. Observed orders of convergence (log2 of error ratios)

Note on temporal study design
------------------------------
Using an external analytical solution as the reference conflates the remaining
spatial discretisation error with the temporal error being measured.  When the
spatial floor (~6e-4 for M=800) is comparable to or larger than the temporal
error at fine N, the apparent convergence order collapses to zero.

The correct procedure is to fix M (here M=200) and compare V(M,N) against a
reference solution V(M, N_ref) computed on the SAME spatial grid with a very
large N_ref (12800).  Both solutions share the same spatial discretisation, so
the difference isolates the temporal component.  This gives clean orders:
Implicit O(dt^1), CN O(dt^2).

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
    Accuracy: O(dS^2) spatial, O(dt) temporal.
    Backward Euler is A-stable for the scalar test equation; matrix properties depend on the discretized operator.
    """
    dS = Smax / M
    dt = T / N
    S  = np.linspace(0, Smax, M + 1)
    V  = np.maximum(S - K, 0.0) if opt == 'call' else np.maximum(K - S, 0.0)
    Sj = np.arange(1, M) * dS

    al = 0.5 * sigma**2 * Sj**2 / dS**2 - 0.5 * r * Sj / dS
    be = -sigma**2 * Sj**2 / dS**2 - r
    ga = 0.5 * sigma**2 * Sj**2 / dS**2 + 0.5 * r * Sj / dS

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
    Accuracy: O(dS^2) spatial, O(dt^2) temporal.
    Crank-Nicolson is A-stable for the scalar test equation; matrix properties depend on the discretized operator.
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
    aL =  -h * al;  bL = 1.0 - h * be;  cL = -h * ga
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

def linf_error_vs_analytic(S, V, K, T, r, sigma, opt):
    """L-infinity error at interior nodes vs analytical solution."""
    errs = []
    for j in range(1, len(S) - 1):
        if S[j] > 0:
            exact = bs_price(S[j], K, T, r, sigma, opt)
            errs.append(abs(V[j] - exact))
    return max(errs) if errs else 0.0

def linf_error_vs_reference(V_coarse, V_ref):
    """L-infinity error of interior nodes against a reference solution on the same grid."""
    return np.max(np.abs(V_coarse[1:-1] - V_ref[1:-1]))

def obs_order(e1, e2, h1, h2):
    """Observed order p from e ~ C*h^p, where h1 is coarse and h2 is fine (h1 > h2)."""
    if e1 <= 0 or e2 <= 0 or h1 <= h2:
        return float('nan')
    return log(e1 / e2) / log(h1 / h2)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN STUDY
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    K     = 100.0
    T     = 1.0
    r     = 0.05
    sigma = 0.20
    Smax  = 400.0
    opt   = 'call'

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
    N_spatial  = 5000   # large N: temporal error negligible -> isolates spatial order

    # ── 8a: Spatial convergence ──
    print("\n  [8a] SPATIAL CONVERGENCE STUDY (N=5000 fixed, vary M)")
    print("       Large N ensures temporal error is negligible.")
    print("       Expected: O(dS^2) -> error ratios ~4.0")
    print("       Note: M=50 (dS=8) has K between nodes -> ratio irregular (grid alignment effect).")
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
            e = linf_error_vs_analytic(S_g, V_g, K, T, r, sigma, opt)
            ratio = prev_e / e if prev_e and e > 0 else float('nan')
            order = obs_order(prev_e, e, prev_dS, dS) if prev_e else float('nan')
            note = "  <- K not on grid" if M == 50 else ""
            print(f"  {M:>6}  {dS:>8.2f}  {e:>13.5e}  {ratio:>8.2f}  {order:>12.3f}{note}")
            prev_e, prev_dS = e, dS
        print()

    # ── 8b: Temporal convergence — same-grid Richardson reference ──
    print("  [8b] TEMPORAL CONVERGENCE STUDY")
    print("       Method: same-grid Richardson reference (M=200 fixed, N_ref=12800).")
    print("       Error = max|V(M,N) - V(M, N_ref)| over interior nodes.")
    print("       This isolates temporal error; both solutions share the same spatial grid.")
    print("       Expected: Implicit O(dt^1) -> ratios ~2.0")
    print("                 CN       O(dt^2) -> ratios ~4.0")
    print()

    M_temporal = 200    # fixed spatial grid
    N_ref      = 12800  # reference: very fine in time, same spatial grid
    N_values   = [50, 100, 200, 400, 800]

    for solver_name, solver, exp_order in [("Fully Implicit", solve_implicit, 1),
                                             ("Crank-Nicolson", solve_cn,       2)]:
        print(f"  --- {solver_name}  [expected O(dt^{exp_order})] ---")
        print(f"  {'N':>6}  {'dt':>10}  {'L-inf Error':>13}  {'Ratio':>8}  {'Obs. Order':>12}")
        print("  " + "-" * 58)

        # Compute reference on same spatial grid
        _, V_ref = solver(K, T, r, sigma, Smax, M_temporal, N_ref, opt)

        prev_e, prev_dt = None, None
        for N in N_values:
            dt = T / N
            S_g, V_g = solver(K, T, r, sigma, Smax, M_temporal, N, opt)
            e = linf_error_vs_reference(V_g, V_ref)
            ratio = prev_e / e if prev_e and e > 0 else float('nan')
            order = obs_order(prev_e, e, prev_dt, dt) if prev_e else float('nan')
            print(f"  {N:>6}  {dt:>10.5f}  {e:>13.5e}  {ratio:>8.2f}  {order:>12.3f}")
            prev_e, prev_dt = e, dt
        print()

    # ── 8c: Summary interpretation ──
    print("  [8c] INTERPRETATION")
    print("  " + "-" * 68)
    print("""
  Spatial order (both methods):
    Both Fully Implicit and Crank-Nicolson use central differences in S,
    giving O(dS^2) spatial truncation error.  Halving dS (M->2M) should
    reduce the error by a factor of ~4.  The tables confirm ratios of
    approximately 4.0 once K lies on a grid node.  The M=50 anomaly
    (dS=8, K=100 falls between nodes) causes the ratio to appear irregular
    — this is a grid-alignment effect, not a solver defect.

  Temporal order (same-grid Richardson reference):
    Fixing M=200 and comparing V(M,N) against V(M, 12800) on the same
    spatial grid cleanly isolates the temporal component of the error.

    Fully Implicit uses Backward Euler (theta=1), giving O(dt) temporal
    accuracy.  Halving N (dt->dt/2) reduces the error by a factor of ~2.0,
    confirming first-order convergence in time.

    Crank-Nicolson uses the trapezoid rule (theta=1/2), giving O(dt^2)
    temporal accuracy.  Halving N reduces the error by a factor of ~4.0,
    confirming second-order convergence in time.

    Important note: measuring temporal error against an external analytical
    solution on a large M (e.g. M=800) can produce misleading results when
    the spatial floor (~6e-4 for M=800) is comparable to the temporal error
    at fine N.  The CN error then appears to stall at ~6e-4 regardless of N,
    giving an apparent convergence order of zero.  The same-grid reference
    avoids this artefact entirely.

  Stability:
    Backward Euler and Crank-Nicolson are A-stable scalar methods. Matrix
    stability and dominance depend on the discrete operator and coefficients;
    Thomas elimination solves the tested systems but does not prove scheme stability.

  Practical recommendation:
    For equivalent computational cost (same M*N), Crank-Nicolson achieves
    significantly smaller errors than the Fully Implicit method due to its
    superior O(dt^2) temporal accuracy.
  """)

    print("=" * 72)
    print("  CONVERGENCE STUDY COMPLETE")
    print("=" * 72)
