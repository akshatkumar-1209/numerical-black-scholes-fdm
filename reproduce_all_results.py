"""
reproduce_all_results.py -- regenerate EVERY number, table and figure in the repo
================================================================================
    python reproduce_all_results.py            # results/*.txt + results/summary.json
    python reproduce_all_results.py --figures  # ... and figures/*.png

Each study script contains acceptance checks (observed orders must match theory,
the CN oscillation must be reproduced, ...).  If any check fails this script
stops with an AssertionError and a non-zero exit code -- it never prints a
success message unconditionally.  Runtime: about 30-60 seconds.
"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "programs"))
import convergence_study, numerical_greeks, payoff_kink_cn_oscillation
from fdm_core import thomas, solve_fdm, bs_price

def jsonable(o):
    if isinstance(o, dict):  return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, float) and o != o: return None
    return o

def basic_checks():
    """Thomas vs LAPACK, put-call parity of the FDM solution."""
    rng = np.random.default_rng(42); n = 100
    a = np.r_[0.0, -rng.random(n - 1)]; c = np.r_[-rng.random(n - 1), 0.0]
    b = np.abs(a) + np.abs(c) + 1.0; d = rng.random(n)
    A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    err = np.linalg.norm(thomas(a, b, c, d) - np.linalg.solve(A, d))
    assert err < 1e-12, f"Thomas vs LAPACK: {err}"
    S, Vc = solve_fdm(100, 1, .05, .2, 400, 200, 500, "call", "cn")
    _, Vp = solve_fdm(100, 1, .05, .2, 400, 200, 500, "put", "cn")
    parity = np.abs((Vc - Vp) - (S - 100 * np.exp(-0.05))).max()
    assert parity < 5e-3, f"put-call parity of FDM solution: {parity}"
    return {"thomas_vs_lapack_2norm": float(err), "fdm_put_call_parity_max": float(parity)}

def main():
    t0 = time.time()
    out = {"basic": basic_checks()}
    print("basic checks passed:", out["basic"])
    for name, mod in (("convergence", convergence_study), ("greeks", numerical_greeks), ("kink", payoff_kink_cn_oscillation)):
        out[name] = mod.run(verbose=False)
        print(f"  [{name}] done, acceptance checks passed")
    with open(os.path.join(HERE, "results", "summary.json"), "w") as f:
        json.dump(jsonable(out), f, indent=1)
    c = out["convergence"]
    print("\nKEY RESULTS")
    print("  spatial order  (CN, M=100..800)      :", [round(x, 2) for x in c["spatial_order_cn"]])
    print("  spatial order  (implicit)            :", [round(x, 2) for x in c["spatial_order_imp"]])
    print("  temporal order (implicit, M=200)     :", [round(x, 2) for x in c["temporal_order_imp_M200"]])
    print("  temporal order (CN, M=200)           :", [round(x, 2) for x in c["temporal_order_cn_M200"]])
    g = out["greeks"]
    print("  Greek order T=1 (Delta win, Gamma win):", [round(x, 2) for x in g["B_orders_T1.0"]["dW"]], [round(x, 2) for x in g["B_orders_T1.0"]["gW"]])
    if "--figures" in sys.argv:
        import generate_figures; generate_figures.run()
    print(f"\nALL ACCEPTANCE CHECKS PASSED  ({time.time() - t0:.0f} s)")

if __name__ == "__main__":
    main()
