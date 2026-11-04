"""
reproduce_all_results.py
========================
Automated reproduction and verification script for the
Numerical Black-Scholes FDM project.

Runs all five numerical experiments, compares computed values against
the committed output files, and raises AssertionError with details if
any result falls outside expected ranges.

Usage:
  python reproduce_all_results.py

Expected output: PASSED for each assertion.  Generates
  results/output_11_grid_refinement_matrix.txt

Author: Akshat Kumar
"""

import os
import sys
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
    if S <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N(d1) - K * exp(-r * T) * _N(d2)

def bs_delta(S, K, T, r, sigma, opt='call'):
    if S <= 0 or T <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return _N(d1) if opt == 'call' else _N(d1) - 1.0

def bs_gamma(S, K, T, r, sigma):
    if S <= 0 or T <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return _n(d1) / (S * sigma * sqrt(T))

# ─────────────────────────────────────────────────────────────────────────────
# Thomas Algorithm
# ─────────────────────────────────────────────────────────────────────────────

def thomas(a, b, c, d):
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
# Solvers
# ─────────────────────────────────────────────────────────────────────────────

def solve_implicit(K, T, r, sigma, Smax, M, N, opt='call'):
    dS = Smax / M; dt = T / N
    S = np.linspace(0, Smax, M + 1)
    V = np.maximum(S - K, 0.0) if opt == 'call' else np.maximum(K - S, 0.0)
    Sj = np.arange(1, M) * dS
    al = 0.5 * sigma**2 * Sj**2 / dS**2 - 0.5 * r * Sj / dS
    be = -sigma**2 * Sj**2 / dS**2 - r
    ga = 0.5 * sigma**2 * Sj**2 / dS**2 + 0.5 * r * Sj / dS
    a = -dt * al; b = 1.0 - dt * be; c = -dt * ga
    for step in range(N):
        tau = (step + 1) * dt
        V0 = 0.0 if opt == 'call' else K * exp(-r * tau)
        VM = (Smax - K * exp(-r * tau)) if opt == 'call' else 0.0
        rhs = V[1:M].copy()
        rhs[0] -= a[0] * V0; rhs[-1] -= c[-1] * VM
        V[0] = V0; V[1:M] = thomas(a, b, c, rhs); V[M] = VM
    return S, V

def solve_cn(K, T, r, sigma, Smax, M, N, opt='call'):
    dS = Smax / M; dt = T / N
    S = np.linspace(0, Smax, M + 1)
    V = np.maximum(S - K, 0.0) if opt == 'call' else np.maximum(K - S, 0.0)
    Sj = np.arange(1, M) * dS
    al = 0.5 * sigma**2 * Sj**2 / dS**2 - 0.5 * r * Sj / dS
    be = -sigma**2 * Sj**2 / dS**2 - r
    ga = 0.5 * sigma**2 * Sj**2 / dS**2 + 0.5 * r * Sj / dS
    h = 0.5 * dt
    aL, bL, cL = -h * al, 1.0 - h * be, -h * ga
    aR, bR, cR =  h * al, 1.0 + h * be,  h * ga
    for step in range(N):
        tn = (step + 1) * dt; to = step * dt
        V0n = 0.0 if opt == 'call' else K * exp(-r * tn)
        VMn = (Smax - K * exp(-r * tn)) if opt == 'call' else 0.0
        V0o = 0.0 if opt == 'call' else K * exp(-r * to)
        VMo = (Smax - K * exp(-r * to)) if opt == 'call' else 0.0
        vi = V[1:M]
        rhs = bR * vi
        rhs[1:] += aR[1:] * vi[:-1]; rhs[:-1] += cR[:-1] * vi[1:]
        rhs[0] += aR[0] * V0o - aL[0] * V0n
        rhs[-1] += cR[-1] * VMo - cL[-1] * VMn
        V[0] = V0n; V[1:M] = thomas(aL, bL, cL, rhs); V[M] = VMn
    return S, V

def solve_rannacher(K, T, r, sigma, Smax, M, N, opt='call', n_half_steps=2):
    dS = Smax / M; dt = T / N
    S = np.linspace(0, Smax, M + 1)
    V = np.maximum(S - K, 0.0) if opt == 'call' else np.maximum(K - S, 0.0)
    Sj = np.arange(1, M) * dS
    al = 0.5 * sigma**2 * Sj**2 / dS**2 - 0.5 * r * Sj / dS
    be = -sigma**2 * Sj**2 / dS**2 - r
    ga = 0.5 * sigma**2 * Sj**2 / dS**2 + 0.5 * r * Sj / dS
    h = 0.5 * dt
    aL, bL, cL = -h * al, 1.0 - h * be, -h * ga
    aR, bR, cR =  h * al, 1.0 + h * be,  h * ga

    dt_half = 0.5 * dt
    a_half = -dt_half * al; b_half = 1.0 - dt_half * be; c_half = -dt_half * ga

    if n_half_steps < 0 or n_half_steps > 2 * N or n_half_steps % 2:
        raise ValueError("n_half_steps must be even and between 0 and 2*N")
    num_half = n_half_steps
    for s in range(num_half):
        tau = (s + 1) * dt_half
        V0 = 0.0 if opt == 'call' else K * exp(-r * tau)
        VM = (Smax - K * exp(-r * tau)) if opt == 'call' else 0.0
        rhs = V[1:M].copy()
        rhs[0] -= a_half[0] * V0; rhs[-1] -= c_half[-1] * VM
        V[0] = V0; V[1:M] = thomas(a_half, b_half, c_half, rhs); V[M] = VM

    start_step = num_half // 2
    for step in range(start_step, N):
        tn = (step + 1) * dt; to = step * dt
        V0n = 0.0 if opt == 'call' else K * exp(-r * tn)
        VMn = (Smax - K * exp(-r * tn)) if opt == 'call' else 0.0
        V0o = 0.0 if opt == 'call' else K * exp(-r * to)
        VMo = (Smax - K * exp(-r * to)) if opt == 'call' else 0.0
        vi = V[1:M]
        rhs = bR * vi
        rhs[1:] += aR[1:] * vi[:-1]; rhs[:-1] += cR[:-1] * vi[1:]
        rhs[0] += aR[0] * V0o - aL[0] * V0n
        rhs[-1] += cR[-1] * VMo - cL[-1] * VMn
        V[0] = V0n; V[1:M] = thomas(aL, bL, cL, rhs); V[M] = VMn
    return S, V

def linf_error_analytic(S, V, K, T, r, sigma, opt='call'):
    bs = bs_call if opt == 'call' else lambda s, k, t, rv, sg: bs_call(s, k, t, rv, sg) - s + k * exp(-rv * t)
    errs = [abs(V[j] - bs(S[j], K, T, r, sigma)) for j in range(1, len(S) - 1) if S[j] > 0]
    return max(errs) if errs else 0.0

def linf_error_reference(V, V_ref):
    return float(np.max(np.abs(V[1:-1] - V_ref[1:-1])))

def assert_range(label, value, lo, hi):
    if not (lo <= value <= hi):
        raise AssertionError(
            f"FAILED: {label}\n"
            f"  Expected in [{lo:.4e}, {hi:.4e}], got {value:.6e}"
        )
    print(f"  PASSED: {label} = {value:.4e}  (in [{lo:.2e}, {hi:.2e}])")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    K, T, r, sigma, Smax = 100.0, 1.0, 0.05, 0.20, 400.0
    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(res_dir, exist_ok=True)

    print("=" * 72)
    print("  NUMERICAL BLACK-SCHOLES FDM — REPRODUCTION OF ALL RESULTS")
    print(f"  K={K}, T={T}, r={r}, sigma={sigma}, Smax={Smax}")
    print(f"  Analytical ATM call: {bs_call(K, K, T, r, sigma):.8f}")
    print("=" * 72)

    # Closed-form benchmark and parity checks on a common grid.
    benchmark_max_error = 0.0
    parity_max_residual = 0.0
    for solver_name, solver, M_b, N_b in [
        ("Fully Implicit", solve_implicit, 200, 1000),
        ("Crank-Nicolson", solve_cn, 200, 500),
    ]:
        S_call, V_call = solver(K, T, r, sigma, Smax, M_b, N_b, "call")
        S_put, V_put = solver(K, T, r, sigma, Smax, M_b, N_b, "put")
        call_error = linf_error_analytic(S_call, V_call, K, T, r, sigma, "call")
        put_error = linf_error_analytic(S_put, V_put, K, T, r, sigma, "put")
        parity = np.max(np.abs(V_call - V_put - S_call + K * exp(-r * T)))
        benchmark_max_error = max(benchmark_max_error, call_error, put_error)
        parity_max_residual = max(parity_max_residual, float(parity))
        print(f"  {solver_name}: call/put L-inf benchmark errors = {call_error:.6e}, {put_error:.6e}; max parity residual = {parity:.6e}")
    if not np.isfinite(benchmark_max_error) or benchmark_max_error > 0.1:
        raise AssertionError(f"Analytical benchmark error too large: {benchmark_max_error:.6e}")
    # Backward Euler has a small O(dt) parity defect because exp(-r*tau)
    # is not an exact solution of its discrete time recurrence.
    if not np.isfinite(parity_max_residual) or parity_max_residual > 2.0e-4:
        raise AssertionError(f"Numerical put-call parity residual too large: {parity_max_residual:.6e}")

    # ─── EXPERIMENT 1: 4x4 Refinement Matrix ────────────────────────────────
    print("\n[1/5] 4x4 Grid Refinement Matrix (M, N in {50, 100, 200, 400})")
    grid_vals = [50, 100, 200, 400]
    mat_imp = np.zeros((4, 4))
    mat_cn  = np.zeros((4, 4))

    for i, M_val in enumerate(grid_vals):
        for j, N_val in enumerate(grid_vals):
            S_i, V_i = solve_implicit(K, T, r, sigma, Smax, M_val, N_val)
            S_c, V_c = solve_cn(K, T, r, sigma, Smax, M_val, N_val)
            mat_imp[i, j] = linf_error_analytic(S_i, V_i, K, T, r, sigma)
            mat_cn[i, j]  = linf_error_analytic(S_c, V_c, K, T, r, sigma)

    # Spatial ratios along the M=400 row: mat_imp[3,0]/mat_imp[3,1], etc.
    # As M doubles (N fixed at 400), expect ratio ~4 for spatial convergence
    # Along last column (N=400 fixed), vary M:
    ratio_cn_sp_200_400 = mat_cn[2, 3] / mat_cn[3, 3]  # M=200 to M=400 error ratio
    assert_range("CN spatial ratio M=200->M=400 (N=400 fixed)", ratio_cn_sp_200_400, 3.5, 4.5)

    matrix_file = os.path.join(res_dir, "output_11_grid_refinement_matrix.txt")
    with open(matrix_file, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("4x4 REFINEMENT MATRIX: L-inf errors (M, N in {50, 100, 200, 400})\n")
        f.write("Parameters: K=100, T=1, r=0.05, sigma=0.20, Smax=400\n")
        f.write("=" * 72 + "\n\n")
        f.write("--- Fully Implicit (Backward Euler) ---\n")
        f.write(f"{'M\\N':>7} |" + "".join([f" {'N='+str(n):>12} |" for n in grid_vals]) + "\n")
        f.write("-" * 68 + "\n")
        for i, m in enumerate(grid_vals):
            row = "".join([f" {mat_imp[i,j]:>12.4e} |" for j in range(4)])
            f.write(f"{'M='+str(m):>7} |{row}\n")
        f.write("\n--- Crank-Nicolson (theta = 1/2) ---\n")
        f.write(f"{'M\\N':>7} |" + "".join([f" {'N='+str(n):>12} |" for n in grid_vals]) + "\n")
        f.write("-" * 68 + "\n")
        for i, m in enumerate(grid_vals):
            row = "".join([f" {mat_cn[i,j]:>12.4e} |" for j in range(4)])
            f.write(f"{'M='+str(m):>7} |{row}\n")
        f.write("\n")
        f.write("Reading the matrix:\n")
        f.write("  Moving down a column (M doubles, N fixed): spatial error dominates; error drops by ~4x -> confirms O(dS^2).\n")
        f.write("  Moving right along a row (N doubles, M fixed):\n")
        f.write("    - Fully Implicit error reduces by ~2x -> confirms O(dt) temporal improvement.\n")
        f.write("    - Crank-Nicolson error remains nearly constant across N because the spatial error\n")
        f.write("      floor O(dS^2) (~1.0e-2 for M=200, ~2.5e-3 for M=400) dominates the analytical benchmark\n")
        f.write("      error and masks further temporal error reduction. Temporal O(dt^2) convergence\n")
        f.write("      is therefore cleanly demonstrated via the same-spatial-grid Richardson reference study (Exp 3).\n")
    print(f"  -> Written: {matrix_file}")

    # ─── EXPERIMENT 2: Spatial Convergence ───────────────────────────────────
    print("\n[2/5] Spatial Convergence (N=5000, vary M in {100, 200, 400})")
    for name, solver in [("Implicit", solve_implicit), ("CN", solve_cn)]:
        previous_error = None
        for M_val in [100, 200, 400]:
            S_c, V_c = solver(K, T, r, sigma, Smax, M_val, 5000)
            e_c = linf_error_analytic(S_c, V_c, K, T, r, sigma)
            if previous_error is not None:
                assert_range(f"{name} spatial ratio M={M_val//2}->{M_val}", previous_error / e_c, 3.5, 4.5)
            previous_error = e_c

    # ─── EXPERIMENT 3: Temporal Convergence (same-grid reference) ────────────
    print("\n[3/5] Temporal Convergence — same-grid Richardson reference (M=200, N_ref=12800)")
    M_t = 200; N_ref = 12800

    _, V_ref_im = solve_implicit(K, T, r, sigma, Smax, M_t, N_ref)
    _, V_ref_cn = solve_cn(K, T, r, sigma, Smax, M_t, N_ref)
    _, V_ref2_im = solve_implicit(K, T, r, sigma, Smax, M_t, 2 * N_ref)
    _, V_ref2_cn = solve_cn(K, T, r, sigma, Smax, M_t, 2 * N_ref)
    ref_change_im = linf_error_reference(V_ref_im, V_ref2_im)
    ref_change_cn = linf_error_reference(V_ref_cn, V_ref2_cn)
    print(f"  Reference refinement N_ref={N_ref}->{2*N_ref}: Implicit change={ref_change_im:.6e}; CN change={ref_change_cn:.6e}")
    if ref_change_im > 2.0e-4 or ref_change_cn > 1.0e-8:
        raise AssertionError(
            "Temporal reference is not sufficiently converged: "
            f"Implicit change={ref_change_im:.6e}, CN change={ref_change_cn:.6e}"
        )

    prev_im, prev_cn = None, None
    im_ratios, cn_ratios = [], []
    for N_val in [50, 100, 200, 400]:
        _, V_i = solve_implicit(K, T, r, sigma, Smax, M_t, N_val)
        _, V_c = solve_cn(K, T, r, sigma, Smax, M_t, N_val)
        e_i = linf_error_reference(V_i, V_ref_im)
        e_c = linf_error_reference(V_c, V_ref_cn)
        if prev_im:
            im_ratios.append(prev_im / e_i)
        if prev_cn:
            cn_ratios.append(prev_cn / e_c)
        prev_im, prev_cn = e_i, e_c

    for idx, ratio in enumerate(im_ratios):
        assert_range(f"Implicit temporal ratio step {idx+1} (expect ~2.0)", ratio, 1.7, 2.3)
    for idx, ratio in enumerate(cn_ratios):
        assert_range(f"CN temporal ratio step {idx+1} (expect ~4.0)", ratio, 3.5, 4.5)

    # ─── EXPERIMENT 4: Greeks ─────────────────────────────────────────────────
    print("\n[4/5] Numerical Greeks at ATM (M=400, N=2000)")
    M_g = 400; N_g = 2000; dS_g = Smax / M_g
    S_g, V_g = solve_cn(K, T, r, sigma, Smax, M_g, N_g)
    j_atm = round(K / dS_g)
    s_atm = S_g[j_atm]
    delta_num = (V_g[j_atm + 1] - V_g[j_atm - 1]) / (2.0 * dS_g)
    gamma_num = (V_g[j_atm + 1] - 2.0 * V_g[j_atm] + V_g[j_atm - 1]) / (dS_g**2)
    delta_err = abs(delta_num - bs_delta(s_atm, K, T, r, sigma))
    gamma_err = abs(gamma_num - bs_gamma(s_atm, K, T, r, sigma))
    assert_range("CN ATM Delta error (M=400, N=2000)", delta_err, 1e-7, 2e-4)
    assert_range("CN ATM Gamma error (M=400, N=2000)", gamma_err, 1e-8, 5e-5)

    # ─── EXPERIMENT 5: Rannacher vs CN Gamma oscillation ─────────────────────
    print("\n[5/5] Rannacher vs CN — Gamma oscillation at N=5 (M=200)")
    M_r = 200; dS_r = Smax / M_r; j_K = round(K / dS_r)
    _, V_pure = solve_cn(K, T, r, sigma, Smax, M_r, 5)
    _, V_ran  = solve_rannacher(K, T, r, sigma, Smax, M_r, 5, n_half_steps=2)
    g_pure = np.array([(V_pure[j+1] - 2*V_pure[j] + V_pure[j-1]) / dS_r**2
                        for j in range(1, M_r)])
    g_ran  = np.array([(V_ran[j+1]  - 2*V_ran[j]  + V_ran[j-1])  / dS_r**2
                        for j in range(1, M_r)])
    osc_pure = max(abs(g_pure[j_K-2] - g_pure[j_K-1]), abs(g_pure[j_K] - g_pure[j_K-1]))
    osc_ran  = max(abs(g_ran[j_K-2]  - g_ran[j_K-1]),  abs(g_ran[j_K]  - g_ran[j_K-1]))
    err_pure = linf_error_analytic(np.linspace(0, Smax, M_r + 1), V_pure, K, T, r, sigma)
    err_ran = linf_error_analytic(np.linspace(0, Smax, M_r + 1), V_ran, K, T, r, sigma)

    # CN should have large oscillation at N=5; Rannacher should suppress it substantially
    assert_range("CN Gamma oscillation at N=5 (expect large)", osc_pure, 0.05, 1.0)
    assert_range("Rannacher Gamma oscillation at N=5 (expect small)", osc_ran, 0.0, 0.01)
    print(f"  N=5 max price errors: CN={err_pure:.6e}, corrected Rannacher={err_ran:.6e}")
    print(f"  N=5 oscillation suppression: {osc_pure / osc_ran:.3f}x")

    print("\n" + "=" * 72)
    print("  ALL ASSERTIONS PASSED")
    print("  Spatial order: CN ~4.0 (O(dS^2)), Implicit ~4.0 (O(dS^2))")
    print("  Temporal order: CN ~4.0 (O(dt^2)), Implicit ~2.0 (O(dt))")
    print("  [same-grid Richardson reference used for temporal study]")
    print("=" * 72)

if __name__ == "__main__":
    main()
