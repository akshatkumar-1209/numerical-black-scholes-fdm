"""
convergence_study.py  --  Weeks 7 & 8 : verification and convergence analysis
=============================================================================
Benchmark: K=100, T=1, r=0.05, sigma=0.2, Smax=400 (European call/put).

Design (this is what separates spatial from temporal error):

  A. Price tables            FDM vs analytical on one reasonable grid.
  B. Spatial error           Time error removed first, so only dS matters:
                               Crank-Nicolson  : N = 2000 (time error ~1e-7)
                               Fully implicit  : 2*V(N=2000) - V(N=1000)
                                                 (Richardson in time removes
                                                  the O(dt) term)
  C. Temporal error          Measured against a *time-converged solution on
                             the SAME spatial grid* (CN, N_ref = 12800), so the
                             spatial error cancels exactly.
                             Total error vs the exact price is also shown, to
                             make visible why a naive "fine M, vary N" study
                             stalls on the spatial error floor.
  D. 4x4 matrix              L-inf error vs exact price for M,N in {50..400}.

Writes results/output_8_convergence_study.txt and
       results/output_11_grid_refinement_matrix.txt ; returns a summary dict.
"""
import os, sys, io
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdm_core import (solve_fdm, richardson_time, exact_vector, error_norms,
                      node_index, order, bs_price)

K, T, R, SIG, SMAX = 100.0, 1.0, 0.05, 0.20, 400.0
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def _fmt(x, w=11):
    return f"{x:{w}.4e}" if x == x else " " * (w - 1) + "-"

def _ord(x):
    return f"{x:6.2f}" if x == x else "     -"

# ------------------------------------------------------------------ studies
def spatial_study(Ms=(25, 50, 100, 200, 400, 800), n_cn=2000, n_imp=2000):
    rows = []
    for M in Ms:
        row = {"M": M, "dS": SMAX / M}
        S, Vc = solve_fdm(K, T, R, SIG, SMAX, M, n_cn, "call", "cn")
        ex = exact_vector(S, K, T, R, SIG)
        row["cn"] = error_norms(S, Vc, ex)
        _, Vi = richardson_time(K, T, R, SIG, SMAX, M, n_imp, "call")
        row["imp"] = error_norms(S, Vi, ex)
        j = node_index(S, K)
        row["on_node"] = j is not None
        row["cn_K"] = abs(Vc[j] - ex[j]) if j is not None else None
        row["imp_K"] = abs(Vi[j] - ex[j]) if j is not None else None
        rows.append(row)
    return rows

def temporal_study(M, Ns=(25, 50, 100, 200, 400, 800), n_ref=12800):
    S, Vref = solve_fdm(K, T, R, SIG, SMAX, M, n_ref, "call", "cn")
    ex = exact_vector(S, K, T, R, SIG)
    rows = []
    for N in Ns:
        row = {"N": N, "dt": T / N}
        for tag, sch in (("imp", "implicit"), ("cn", "cn")):
            _, V = solve_fdm(K, T, R, SIG, SMAX, M, N, "call", sch)
            row[tag] = error_norms(S, V, Vref)          # temporal error only
            row[tag + "_total"] = error_norms(S, V, ex)[0]   # vs exact price
        rows.append(row)
    return rows

def refinement_matrix(vals=(50, 100, 200, 400)):
    out = {}
    for tag, sch in (("imp", "implicit"), ("cn", "cn")):
        mat = np.zeros((len(vals), len(vals)))
        for i, M in enumerate(vals):
            for j, N in enumerate(vals):
                S, V = solve_fdm(K, T, R, SIG, SMAX, M, N, "call", sch)
                mat[i, j] = error_norms(S, V, exact_vector(S, K, T, R, SIG))[0]
        out[tag] = mat
    return vals, out

def price_table(sch, M, N, opt):
    S, V = solve_fdm(K, T, R, SIG, SMAX, M, N, opt, sch)
    rows = []
    for s in (60, 70, 80, 90, 100, 110, 120, 140, 160, 200):
        j = node_index(S, float(s))
        ex = bs_price(s, K, T, R, SIG, opt)
        rows.append((s, V[j], ex, abs(V[j] - ex)))
    return rows

# --------------------------------------------------------------------- main
def run(verbose=True):
    os.makedirs(RES, exist_ok=True)
    buf = io.StringIO()
    P = lambda *a: print(*a, file=buf)
    summary = {}
    P("=" * 78)
    P("  NUMERICAL BLACK-SCHOLES FDM -- WEEKS 7 & 8")
    P("  Systematic verification and convergence analysis")
    P("=" * 78)
    P(f"  K={K}, T={T}, r={R}, sigma={SIG}, Smax={SMAX}")
    P(f"  Analytical call at S=K: {bs_price(K, K, T, R, SIG):.8f}   "
      f"put: {bs_price(K, K, T, R, SIG, 'put'):.8f}")
    P("  Errors are L-inf (max over interior nodes) unless stated; L2 is the")
    P("  discrete norm sqrt(sum e_j^2 dS).  Observed order p = log2(E_h / E_h/2).")

    # ---- A. price tables
    P("\n" + "=" * 78 + "\n  A. PRICE TABLES  (FDM vs analytical)\n" + "=" * 78)
    for sch, name, M, N in (("implicit", "Fully implicit", 200, 1000),
                            ("cn", "Crank-Nicolson", 200, 500)):
        for opt in ("call", "put"):
            P(f"\n  {name}, {opt}, M={M}, N={N}  (dS={SMAX/M:g}, dt={T/N:g})")
            P("        S    FDM price   Exact price      |Error|")
            P("  " + "-" * 50)
            tab = price_table(sch, M, N, opt)
            for s, v, ex, e in tab:
                P(f"  {s:7.1f}  {v:11.6f}  {ex:11.6f}  {e:11.4e}")
            summary[f"price_{sch}_{opt}_atm_err"] = [e for s, v, ex, e in tab if s == 100][0]
    # ---- B. spatial
    sp = spatial_study()
    P("\n" + "=" * 78)
    P("  B. SPATIAL CONVERGENCE (time error removed; see header)")
    P("=" * 78)
    P("  K is a grid node only when M is a multiple of 4 (dS = 400/M divides 100).")
    for tag, name in (("cn", "Crank-Nicolson (N=2000)"),
                      ("imp", "Fully implicit (Richardson in time, N=2000/1000)")):
        P(f"\n  --- {name} ---")
        P("      M      dS  K node?     Linf err  order       L2 err  order    err at S=K  order")
        P("  " + "-" * 86)
        prev = None
        for r_ in sp:
            linf, l2 = r_[tag]
            eK = r_[tag + "_K"]
            if prev:
                o1, o2 = order(prev[tag][0], linf), order(prev[tag][1], l2)
                o3 = order(prev[tag + "_K"], eK)
            else:
                o1 = o2 = o3 = float("nan")
            P(f"  {r_['M']:5d} {r_['dS']:7.2f}  {'yes' if r_['on_node'] else 'NO ':>6} "
              f"{_fmt(linf)} {_ord(o1)} {_fmt(l2)} {_ord(o2)} {_fmt(eK if eK else float('nan'))} {_ord(o3)}")
            prev = r_
    P("\n  Reading the table: p = 2.00 holds for every pair of grids in which K is a node")
    P("  (M=100 -> 200 -> 400 -> 800).  The pair M=50 -> 100 gives a meaningless 0.47,")
    P("  because K is NOT a node at M=50 (K lies half-way between nodes 96 and 104),")
    P("  so the kink is represented differently on the two grids.  Practical rule:")
    P("  choose M so that K is a grid node before measuring orders.")
    for tag in ("cn", "imp"):
        o_ = [order(sp[i - 1][tag][0], sp[i][tag][0]) for i in (3, 4, 5)]
        summary[f"spatial_order_{tag}"] = o_
    summary["spatial_rows"] = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r_.items()} for r_ in sp]
    # ---- C. temporal
    summary["temporal"] = {}
    for M in (100, 200):
        tp = temporal_study(M)
        summary["temporal"][M] = tp
        P("\n" + "=" * 78)
        P(f"  C. TEMPORAL CONVERGENCE, M={M}  (error vs time-converged CN solution, same grid)")
        P("=" * 78)
        P("      N         dt   Implicit Linf  order   Implicit L2  |    CN Linf  order       CN L2"
          "  |  CN total err vs exact")
        P("  " + "-" * 104)
        prev = None
        for r_ in tp:
            oi = order(prev["imp"][0], r_["imp"][0]) if prev else float("nan")
            oc = order(prev["cn"][0], r_["cn"][0]) if prev else float("nan")
            P(f"  {r_['N']:5d} {r_['dt']:10.5f}   {_fmt(r_['imp'][0])} {_ord(oi)} {_fmt(r_['imp'][1])}  |"
              f" {_fmt(r_['cn'][0])} {_ord(oc)} {_fmt(r_['cn'][1])}  |  {_fmt(r_['cn_total'])}")
            prev = r_
        summary[f"temporal_order_imp_M{M}"] = [order(tp[i - 1]["imp"][0], tp[i]["imp"][0]) for i in range(1, len(tp))]
        summary[f"temporal_order_cn_M{M}"] = [order(tp[i - 1]["cn"][0], tp[i]["cn"][0]) for i in range(1, len(tp))]
    P("\n  The last column is the error against the EXACT price.  For CN it stops")
    P("  decreasing once the temporal error drops below the spatial error of the")
    P("  grid (about 6e-4 at M=200 ... see section B).  Comparing against the exact")
    P("  price therefore hides the O(dt^2) behaviour; comparing against a")
    P("  time-converged solution on the same grid shows it cleanly.")
    # ---- D. matrix
    vals, mats = refinement_matrix()
    mbuf = io.StringIO()
    Q = lambda *a: print(*a, file=mbuf)
    Q("=" * 78); Q("4x4 REFINEMENT MATRIX: L-infinity ERRORS vs EXACT PRICE, M, N in {50, 100, 200, 400}"); Q("=" * 78)
    P("\n" + "=" * 78 + "\n  D. 4x4 REFINEMENT MATRIX (L-inf error vs exact price)\n" + "=" * 78)
    for tag, name in (("imp", "FULLY IMPLICIT (Backward Euler)"), ("cn", "CRANK-NICOLSON (theta = 1/2)")):
        for w in (P, Q):
            w(f"\n--- {name} ---")
            w("   M \\ N | " + " | ".join(f"{'N=' + str(n):>12}" for n in vals))
            w("-" * 66)
            for i, M in enumerate(vals):
                w(f"{'M=' + str(M):>8} | " + " | ".join(f"{mats[tag][i, j]:12.4e}" for j in range(len(vals))))
    P("\n  Diagonal (M=N doubled together) is a *combined* refinement: the implicit")
    P("  error is dominated by the O(dt) term, so the combined order is between 1")
    P("  and 2 and is NOT a clean measure of either order.  Use sections B and C.")
    summary["matrix"] = {k: v.tolist() for k, v in mats.items()}
    P("\n" + "=" * 78 + "\n  CONVERGENCE STUDY COMPLETE\n" + "=" * 78)

    with open(os.path.join(RES, "output_8_convergence_study.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(buf.getvalue())
    with open(os.path.join(RES, "output_11_grid_refinement_matrix.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(mbuf.getvalue())

    # ---- acceptance checks: the script fails loudly if the theory is not met
    for tag in ("cn", "imp"):
        assert all(1.9 < o < 2.1 for o in summary[f"spatial_order_{tag}"]), f"spatial order {tag}"
    for M in (100, 200):
        assert all(0.95 < o < 1.05 for o in summary[f"temporal_order_imp_M{M}"][:4]), f"implicit temporal order M={M}"
        assert all(1.95 < o < 2.05 for o in summary[f"temporal_order_cn_M{M}"][:4]), f"CN temporal order M={M}"
    if verbose:
        print(buf.getvalue())
    return summary

if __name__ == "__main__":
    run()
