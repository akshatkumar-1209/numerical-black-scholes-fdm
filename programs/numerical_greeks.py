"""
numerical_greeks.py  --  Week 9 : numerical Delta and Gamma
===========================================================
Delta_j = (V_{j+1} - V_{j-1}) / (2 dS)            error O(dS^2)
Gamma_j = (V_{j+1} - 2 V_j + V_{j-1}) / dS^2      error O(dS^2)
computed directly from the FDM price vector and compared with the
analytical Black-Scholes Delta N(d1) and Gamma n(d1)/(S sigma sqrt(T)).

  A. Greeks on one grid (M=400, N=2000), implicit and CN, call.
  B. Mesh refinement of the Greek errors at S=K and in the strike window
     |S-K| <= 10, for T = 1 and for the short maturity T = 0.1, where the
     kink has had little time to diffuse.  CN with N = 2000 (time error
     negligible) so that only dS matters.
Writes results/output_9_numerical_greeks.txt ; returns a summary dict.
"""
import os, sys, io
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdm_core import (solve_fdm, fd_greeks, bs_delta, bs_gamma, node_index, order)

K, R, SIG, SMAX = 100.0, 0.05, 0.20, 400.0
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def exact_greeks(S, T):
    d = np.array([bs_delta(s, K, T, R, SIG) for s in S])
    g = np.array([bs_gamma(s, K, T, R, SIG) for s in S])
    return d, g

def refinement(T, Ms=(100, 200, 400, 800), N=2000, window=10.0):
    rows = []
    for M in Ms:
        S, V = solve_fdm(K, T, R, SIG, SMAX, M, N, "call", "cn")
        d, g = fd_greeks(S, V)
        de, ge = exact_greeks(S, T)
        j = node_index(S, K)
        win = (np.abs(S - K) <= window) & np.isfinite(d)
        rows.append({"M": M, "dS": SMAX / M,
                     "dK": abs(d[j] - de[j]), "gK": abs(g[j] - ge[j]),
                     "dW": float(np.abs(d - de)[win].max()),
                     "gW": float(np.abs(g - ge)[win].max())})
    return rows

def run(verbose=True):
    os.makedirs(RES, exist_ok=True)
    buf = io.StringIO()
    P = lambda *a: print(*a, file=buf)
    summary = {}
    P("=" * 78); P("  NUMERICAL BLACK-SCHOLES FDM -- WEEK 9"); P("  Numerical Delta & Gamma vs analytical Black-Scholes Greeks"); P("=" * 78)
    P(f"  K={K}, r={R}, sigma={SIG}, Smax={SMAX}   (European call)")

    # ---- A
    M, N, T = 400, 2000, 1.0
    S, Vi = solve_fdm(K, T, R, SIG, SMAX, M, N, "call", "implicit")
    _, Vc = solve_fdm(K, T, R, SIG, SMAX, M, N, "call", "cn")
    di, gi = fd_greeks(S, Vi); dc, gc = fd_greeks(S, Vc)
    de, ge = exact_greeks(S, T)
    P("\n" + "=" * 78); P(f"  A. GREEKS ON ONE GRID: T={T}, M={M} (dS={SMAX/M:g}), N={N}"); P("=" * 78)
    for name, ex, ni, nc in (("DELTA", de, di, dc), ("GAMMA", ge, gi, gc)):
        P(f"\n  {name}")
        P("        S      Analytical      Implicit   |Err Impl|            CN     |Err CN|")
        P("  " + "-" * 80)
        for s in (60, 70, 80, 90, 95, 100, 105, 110, 120, 140, 160, 200):
            j = node_index(S, float(s))
            P(f"  {s:7.1f}  {ex[j]:12.8f}  {ni[j]:12.8f}  {abs(ni[j]-ex[j]):10.3e}  {nc[j]:12.8f}  {abs(nc[j]-ex[j]):10.3e}")
    jK = node_index(S, K)
    summary["A_delta_err_K_cn"] = abs(dc[jK] - de[jK]); summary["A_gamma_err_K_cn"] = abs(gc[jK] - ge[jK])
    summary["A_delta_err_K_imp"] = abs(di[jK] - de[jK]); summary["A_gamma_err_K_imp"] = abs(gi[jK] - ge[jK])
    P("\n  Put Greeks: Delta_put = Delta_call - 1 and Gamma_put = Gamma_call hold exactly")
    P("  (put-call parity), so the put errors are identical to the call errors above.")
    P("\n  NEAR-STRIKE (CN), nodes K-3dS ... K+3dS")
    P("        S    Delta err      Gamma err     Gamma exact   Gamma FDM")
    for k in range(-3, 4):
        j = jK + k
        P(f"  {S[j]:7.1f}  {abs(dc[j]-de[j]):10.3e}  {abs(gc[j]-ge[j]):12.3e}  {ge[j]:12.8f}  {gc[j]:12.8f}")

    # ---- B
    summary["B"] = {}
    for T in (1.0, 0.1):
        rows = refinement(T)
        summary["B"][T] = rows
        P("\n" + "=" * 78)
        P(f"  B. MESH REFINEMENT OF THE GREEK ERRORS, T={T}  (CN, N=2000, window |S-K|<=10)")
        P("=" * 78)
        P(f"  Smoothing width of the kink at maturity: sigma*sqrt(T)*K = {SIG*np.sqrt(T)*K:.2f}")
        P("      M     dS | Delta err@K  order | Delta err win  order | Gamma err@K  order | Gamma err win  order")
        P("  " + "-" * 100)
        prev = None
        for r_ in rows:
            o = [order(prev[k], r_[k]) if prev else float("nan") for k in ("dK", "dW", "gK", "gW")]
            f = lambda x: f"{x:5.2f}" if x == x else "    -"
            P(f"  {r_['M']:5d} {r_['dS']:6.2f} | {r_['dK']:11.3e}  {f(o[0])} | {r_['dW']:12.3e}  {f(o[1])} | "
              f"{r_['gK']:11.3e}  {f(o[2])} | {r_['gW']:12.3e}  {f(o[3])}")
            prev = r_
        summary[f"B_orders_T{T}"] = {k: [order(rows[i-1][k], rows[i][k]) for i in range(1, len(rows))] for k in ("dK", "dW", "gK", "gW")}
    P("\n" + "=" * 78); P("  INTERPRETATION"); P("=" * 78)
    P("""
  * Both Greeks are central differences of the price vector, so their error is
    O(dS^2) once the grid resolves the smoothing scale sigma*sqrt(T)*K of the
    kink.  The tables confirm p ~ 2 for Delta and Gamma.
  * Gamma is a second difference: it divides by dS^2, so for the same grid it
    is less accurate relative to its size than Delta (relative errors at S=K,
    T=1, M=400: ~1e-4 for Delta vs ~4e-4 for Gamma; at T=0.1, M=100 the Gamma
    error at K is ~7% of its value).
  * At T=1 the kink has diffused over ~20 units of S and the errors near K are
    small.  At T=0.1 the width is only ~6.3: with dS=4 (M=100) the grid barely
    resolves it, the errors near K are large and the asymptotic order is only
    reached for finer grids.  Near the strike the Greek errors are therefore
    controlled by dS relative to sigma*sqrt(T)*K, not by dt.
  * Remedy for short maturities: refine (or cluster nodes) around K, and use
    implicit start-up steps (Rannacher, see Week 10) to avoid Gamma noise.
""")
    P("=" * 78 + "\n  NUMERICAL GREEKS STUDY COMPLETE\n" + "=" * 78)
    with open(os.path.join(RES, "output_9_numerical_greeks.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(buf.getvalue())
    for k in ("dW", "gW"):
        assert all(1.8 < o < 2.2 for o in summary["B_orders_T1.0"][k]), f"Greek order T=1 {k}"
    if verbose: print(buf.getvalue())
    return summary

if __name__ == "__main__":
    run()
