"""
payoff_kink_cn_oscillation.py  --  Week 10/11 : non-smooth payoff, CN oscillations,
                                    implicit start-up (Rannacher) time stepping
==================================================================================
The call payoff max(S-K,0) has a kink at S=K (V_S jumps, V_SS = delta function).

  A. Where is K relative to the grid?  K on a node vs K midway between nodes.
  B. CN vs Rannacher (2 backward-Euler steps of size dt, then CN) for coarse
     time steps, T=1.
  C. The same comparison for the short maturity T=0.1.

Metrics (window = nodes K-5dS ... K+5dS):
  Gamma err  = max |Gamma_FDM - Gamma_exact|      Gamma min = min Gamma_FDM
  (a negative Gamma is an unphysical oscillation: the exact Gamma is > 0)
  Price err  = L-inf error vs exact price over all interior nodes.
Writes results/output_10_payoff_kink_oscillation.txt ; returns a summary dict.
"""
import os, sys, io
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fdm_core import (solve_fdm, fd_greeks, bs_gamma, bs_delta, exact_vector,
                      error_norms, node_index, order)

K, R, SIG, SMAX = 100.0, 0.05, 0.20, 400.0
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def metrics(S, V, T):
    ex = exact_vector(S, K, T, R, SIG)
    d, g = fd_greeks(S, V)
    ge = np.array([bs_gamma(s, K, T, R, SIG) for s in S])
    de = np.array([bs_delta(s, K, T, R, SIG) for s in S])
    j = node_index(S, K)
    w = slice(j - 5, j + 6)
    return {"price": error_norms(S, V, ex)[0],
            "gamma_err": float(np.abs(g - ge)[w].max()),
            "gamma_min": float(g[w].min()),
            "delta_err": float(np.abs(d - de)[w].max())}

def alignment_study(T=1.0, N=2000):
    out = {}
    for lab, Ms in (("on_node", (24, 48, 96, 192, 384)), ("midway", (26, 50, 98, 194, 386))):
        rows = []
        for M in Ms:
            S, V = solve_fdm(K, T, R, SIG, SMAX, M, N, "call", "cn")
            rows.append({"M": M, "dS": SMAX / M, "err": error_norms(S, V, exact_vector(S, K, T, R, SIG, "call"))[0]})
        out[lab] = rows
    return out

def cn_vs_rannacher(T, Ns, M=200):
    S = np.linspace(0, SMAX, M + 1)
    rows = []
    for N in Ns:
        row = {"N": N, "dt": T / N, "lam": T / N * SIG ** 2 * K ** 2 / (SMAX / M) ** 2}
        for sch in ("cn", "rannacher"):
            _, V = solve_fdm(K, T, R, SIG, SMAX, M, N, "call", sch)
            row[sch] = metrics(S, V, T)
        rows.append(row)
    return rows

def _table(P, rows):
    P("      N       dt  lambda | CN: price err  Gamma err  Gamma min | Rannacher: price err  Gamma err  Gamma min")
    P("  " + "-" * 104)
    for r_ in rows:
        c, a = r_["cn"], r_["rannacher"]
        P(f"  {r_['N']:5d} {r_['dt']:8.4f} {r_['lam']:7.1f} | {c['price']:12.3e} {c['gamma_err']:10.3e} {c['gamma_min']:10.4f} |"
          f" {a['price']:17.3e} {a['gamma_err']:10.3e} {a['gamma_min']:10.4f}")

def run(verbose=True):
    os.makedirs(RES, exist_ok=True)
    buf = io.StringIO()
    P = lambda *a: print(*a, file=buf)
    summary = {}
    P("=" * 78); P("  NUMERICAL BLACK-SCHOLES FDM -- WEEKS 10/11"); P("  Payoff kink, Crank-Nicolson oscillations, implicit start-up (Rannacher)"); P("=" * 78)
    P(f"  K={K}, r={R}, sigma={SIG}, Smax={SMAX}, European call, Gamma window = K +/- 5 nodes")
    P("  Rannacher here = 2 backward-Euler steps of size dt, then Crank-Nicolson.")
    P("  lambda = dt*sigma^2*K^2/dS^2 : dimensionless step size relative to the grid.")

    al = alignment_study()
    summary["alignment"] = al
    P("\n" + "=" * 78); P("  A. POSITION OF THE STRIKE RELATIVE TO THE GRID  (CN, N=2000, T=1)"); P("=" * 78)
    P("  'on node': M multiple of 4 (K is a node).  'midway': M = 2 mod 4 (K lies half-way")
    P("  between two nodes).  Within each family the refinement M -> 2M keeps K's position.\n")
    P("   on node:     M      dS      Linf err  order  |  midway:      M      dS      Linf err  order")
    P("  " + "-" * 84)
    for a, b in zip(al["on_node"], al["midway"]):
        ia, ib = al["on_node"].index(a), al["midway"].index(b)
        oa = order(al["on_node"][ia - 1]["err"], a["err"]) if ia else float("nan")
        ob = order(al["midway"][ib - 1]["err"], b["err"]) if ib else float("nan")
        f = lambda x: f"{x:5.2f}" if x == x else "    -"
        P(f"            {a['M']:5d} {a['dS']:7.2f} {a['err']:13.4e}  {f(oa)}  |          {b['M']:5d} {b['dS']:7.2f} {b['err']:13.4e}  {f(ob)}")
    summary["align_orders_on_node"] = [order(al["on_node"][i - 1]["err"], al["on_node"][i]["err"]) for i in range(1, 5)]
    summary["align_orders_midway"] = [order(al["midway"][i - 1]["err"], al["midway"][i]["err"]) for i in range(1, 5)]
    P("""
  Finding: BOTH positions give second-order convergence (p -> 2.0).  The kink does
  not reduce the order for a uniform grid with either alignment; it only changes
  the error constant.  For the same dS, the 'midway' grid is about 3x more
  accurate than the 'on node' grid in the L-inf price error.  This also explains
  the odd step M=50 -> 100 in the spatial study of Week 8: the two grids belong to
  different families, so that step is not a refinement of one family.""")

    for T, Ns, sec in ((1.0, (3, 5, 10, 20, 50, 200), "B"), (0.1, (3, 5, 10, 20, 50, 200), "C")):
        rows = cn_vs_rannacher(T, Ns)
        summary[f"cn_ran_T{T}"] = rows
        P("\n" + "=" * 78); P(f"  {sec}. CN vs RANNACHER, T={T}, M=200 (dS=2)"); P("=" * 78)
        _table(P, rows)
    P("""
  Findings
  (1) T=1, large dt (lambda >~ 10): standard CN produces an unphysical Gamma --
      NEGATIVE at N=3 and N=5 -- and a large price error.  The oscillation is the
      kink's high-frequency content, which CN does not damp (|R_CN(z)| -> 1 as
      z -> -inf), so it survives the first steps.  Two backward-Euler start-up steps
      (|R_BE(z)| -> 0) remove it: Gamma stays positive and the Gamma error falls
      by a factor ~170 (N=3), ~240 (N=5) and ~85 (N=10); the price error also falls.
  (2) For small dt (lambda <~ 2) CN does not oscillate and Rannacher gives no
      benefit: both Gamma errors equal the spatial error.  In the intermediate
      range (N=20-50, T=1) Rannacher's price error is slightly LARGER than CN's
      (e.g. 1.4e-2 vs 9.5e-3 at N=20), because two first-order steps are taken.
  (3) T=0.1: with these step sizes CN does not oscillate (Gamma min > 0); the
      remaining error is the spatial error of dS=2 against a kink only
      sigma*sqrt(T)*K = 6.3 wide.  Rannacher does not help and increases the
      price error for N <= 20.  It is a remedy for large dt/dS^2, not a
      universal improvement.
  (4) Practical rule: use Rannacher when dt*sigma^2*K^2/dS^2 is large (>~ 5);
      otherwise refine dS (or cluster nodes near K) if Greeks near K matter.""")
    P("\n" + "=" * 78 + "\n  PAYOFF KINK STUDY COMPLETE\n" + "=" * 78)
    with open(os.path.join(RES, "output_10_payoff_kink_oscillation.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(buf.getvalue())
    r1 = {r_["N"]: r_ for r_ in summary["cn_ran_T1.0"]}
    assert r1[5]["cn"]["gamma_min"] < 0 < r1[5]["rannacher"]["gamma_min"], "oscillation not reproduced"
    assert r1[5]["rannacher"]["gamma_err"] < r1[5]["cn"]["gamma_err"] / 50
    assert all(1.9 < o < 2.2 for o in summary["align_orders_on_node"][1:])
    assert all(1.9 < o < 2.2 for o in summary["align_orders_midway"][1:])
    if verbose: print(buf.getvalue())
    return summary

if __name__ == "__main__":
    run()
