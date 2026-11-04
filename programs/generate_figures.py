"""
generate_figures.py -- all figures, drawn from the SAME study functions that
produce the numbers in results/ (nothing is typed in by hand).

  fig1_spatial_convergence.png      L-inf error vs dS (time error removed)
  fig2_temporal_convergence.png     temporal error vs dt (same-grid reference)
  fig3_numerical_greeks.png         Delta, Gamma profiles + error near K
  fig4_cn_oscillations_rannacher.png  Gamma near K: CN vs Rannacher, coarse dt
  fig5_greeks_mesh_convergence.png  Delta/Gamma error vs dS, T=1 and T=0.1
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convergence_study as cs, numerical_greeks as ng, payoff_kink_cn_oscillation as pk
from fdm_core import solve_fdm, fd_greeks, bs_delta, bs_gamma, node_index

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
K, R, SIG, SMAX = 100.0, 0.05, 0.20, 400.0
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.4})

def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=200); plt.close()
    print("  saved", name)

def ref_line(ax, x, y0, p, label, style):
    x = np.array(x, float); ax.loglog(x, y0 * (x / x[0]) ** p, style, lw=1.2, label=label)

def run():
    os.makedirs(FIG, exist_ok=True)
    # fig1
    sp = [r for r in cs.spatial_study() if r["on_node"]]
    dS = [r["dS"] for r in sp]
    plt.figure(figsize=(7, 5))
    plt.loglog(dS, [r["imp"][0] for r in sp], "o-", label="Fully implicit (time error removed)")
    plt.loglog(dS, [r["cn"][0] for r in sp], "s--", label="Crank-Nicolson (N=2000)")
    ref_line(plt.gca(), dS, sp[0]["cn"][0], 2, r"slope 2", "k:")
    plt.gca().invert_xaxis(); plt.xlabel(r"$\Delta S$"); plt.ylabel(r"$L_\infty$ price error")
    plt.title(r"Spatial convergence, $K$ a grid node: $O(\Delta S^2)$"); plt.legend(); save("fig1_spatial_convergence.png")
    # fig2
    tp = cs.temporal_study(200)
    dt = [r["dt"] for r in tp]
    plt.figure(figsize=(7, 5))
    plt.loglog(dt, [r["imp"][0] for r in tp], "o-", label="Fully implicit")
    plt.loglog(dt, [r["cn"][0] for r in tp], "s--", label="Crank-Nicolson")
    plt.loglog(dt, [r["cn_total"] for r in tp], "x:", color="gray", label="CN error vs exact price (hits spatial floor)")
    ref_line(plt.gca(), dt, tp[0]["imp"][0], 1, "slope 1", "b:"); ref_line(plt.gca(), dt, tp[0]["cn"][0], 2, "slope 2", "r:")
    plt.xlabel(r"$\Delta\tau$"); plt.ylabel(r"$L_\infty$ error")
    plt.title("Temporal convergence, M=200\n(error vs time-converged solution on the same grid)", fontsize=11); plt.legend(fontsize=9); save("fig2_temporal_convergence.png")
    # fig3
    M, N, T = 400, 2000, 1.0
    S, V = solve_fdm(K, T, R, SIG, SMAX, M, N, "call", "cn"); d, g = fd_greeks(S, V)
    de = np.array([bs_delta(s, K, T, R, SIG) for s in S]); ge = np.array([bs_gamma(s, K, T, R, SIG) for s in S])
    m = (S > 50) & (S < 200)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(S[m], de[m], "k-", label="analytical"); ax[0, 0].plot(S[m][::6], d[m][::6], "r.", label="FDM (CN)"); ax[0, 0].set_title("Delta"); ax[0, 0].legend()
    ax[0, 1].plot(S[m], ge[m], "k-", label="analytical"); ax[0, 1].plot(S[m][::6], g[m][::6], "r.", label="FDM (CN)"); ax[0, 1].set_title("Gamma"); ax[0, 1].legend()
    w = (S > 80) & (S < 120)
    ax[1, 0].semilogy(S[w], np.abs(d - de)[w]); ax[1, 0].axvline(K, color="gray", ls=":"); ax[1, 0].set_title("|Delta error|, near K"); ax[1, 0].set_xlabel("S")
    ax[1, 1].semilogy(S[w], np.abs(g - ge)[w]); ax[1, 1].axvline(K, color="gray", ls=":"); ax[1, 1].set_title("|Gamma error|, near K"); ax[1, 1].set_xlabel("S")
    fig.suptitle("Numerical Greeks, T=1, M=400, N=2000"); save("fig3_numerical_greeks.png")
    # fig4
    M = 200; T = 1.0
    Sg = np.linspace(0, SMAX, M + 1); gex = np.array([bs_gamma(s, K, T, R, SIG) for s in Sg]); w = (Sg > 70) & (Sg < 130)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for a, Nn in zip(ax, (5, 10)):
        for sch, st, lab in (("cn", "r-o", "Crank-Nicolson"), ("rannacher", "b-s", "Rannacher (2 BE + CN)")):
            _, Vv = solve_fdm(K, T, R, SIG, SMAX, M, Nn, "call", sch); _, gg = fd_greeks(Sg, Vv)
            a.plot(Sg[w], gg[w], st, ms=3, lw=1, label=lab)
        a.plot(Sg[w], gex[w], "k-", lw=1.5, label="analytical"); a.axhline(0, color="gray", lw=0.6)
        a.set_title(f"N={Nn}  (dt={T/Nn:g}), M=200"); a.set_xlabel("S"); a.legend(fontsize=8)
    ax[0].set_ylabel("Gamma"); fig.suptitle("Gamma near the strike: CN oscillation and its removal"); save("fig4_cn_oscillations_rannacher.png")
    # fig5
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, TT in zip(ax, (1.0, 0.1)):
        rows = ng.refinement(TT); x = [r["dS"] for r in rows]
        a.loglog(x, [r["dK"] for r in rows], "o-", label="Delta error at K"); a.loglog(x, [r["gK"] for r in rows], "s-", label="Gamma error at K")
        ref_line(a, x, rows[0]["gK"], 2, "slope 2", "k:"); a.invert_xaxis(); a.set_xlabel(r"$\Delta S$"); a.set_title(f"T={TT}"); a.legend()
    fig.suptitle("Mesh convergence of numerical Greeks at S=K (CN, N=2000)"); save("fig5_greeks_mesh_convergence.png")

if __name__ == "__main__":
    run()
