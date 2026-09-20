# method_comparison.py
# ============================================================
# Robustness comparison: Implicit vs Crank-Nicolson FDM
# Tests a wide range of parameters -- as a professor might.
# Author: Akshat
# ============================================================

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from implicit_solver       import implicit_bs, bs_call, bs_put
from implicit_solver       import compute_errors as imp_err
from crank_nicolson_solver import crank_nicolson_bs
from crank_nicolson_solver import compute_errors as cn_err

M, N = 200, 1000   # shared grid

def compare(label, K, T, r, sigma):
    S_max = 4.0 * K
    S_im, V_im = implicit_bs(K, T, r, sigma, S_max, M, N, 'call')
    S_cn, V_cn = crank_nicolson_bs(K, T, r, sigma, S_max, M, N, 'call')
    e_im, _, _, _, _ = imp_err(S_im, V_im, K, T, r, sigma, 'call')
    e_cn, _, _, _, _ = cn_err(S_cn, V_cn, K, T, r, sigma, 'call')
    j_atm = round(K / (S_max / M))
    v_im_atm = V_im[j_atm]
    v_cn_atm = V_cn[j_atm]
    from math import log, sqrt, exp
    from implicit_solver import _norm_cdf
    d1 = (log(K/K) + (r + 0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    v_ex_atm = K*_norm_cdf(d1) - K*exp(-r*T)*_norm_cdf(d2)
    print(f"  {label:<38} | Im={e_im:.2e} | CN={e_cn:.2e}")
    print(f"    ATM exact={v_ex_atm:.5f}  Im={v_im_atm:.5f}  CN={v_cn_atm:.5f}")

if __name__ == "__main__":
    print("=" * 80)
    print(f"  Implicit vs Crank-Nicolson Robustness Test  [M={M}, N={N}]")
    print("=" * 80)

    print("\n--- Varying Volatility (K=100, T=1, r=0.05) ---")
    for sig in [0.10, 0.20, 0.30, 0.50, 0.80]:
        compare(f"sigma={sig:.2f}", 100, 1.0, 0.05, sig)

    print("\n--- Varying Maturity (K=100, r=0.05, sigma=0.20) ---")
    for t in [0.25, 0.5, 1.0, 2.0]:
        compare(f"T={t:.2f} yr", 100, t, 0.05, 0.20)

    print("\n--- Varying Strike (T=1, r=0.05, sigma=0.20) ---")
    for k in [80, 90, 100, 110, 120]:
        compare(f"K={k}", k, 1.0, 0.05, 0.20)

    print("\n--- Varying Risk-Free Rate (K=100, T=1, sigma=0.20) ---")
    for rate in [0.01, 0.03, 0.05, 0.08, 0.12]:
        compare(f"r={rate:.2f}", 100, 1.0, rate, 0.20)

    print("\nRobustness test complete.")
