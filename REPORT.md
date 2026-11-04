# Numerical Solution of the Black–Scholes PDE by Finite Differences
### Technical report — Akshat Kumar (supervisor: Dr. Ashok Kumar Vaikuntam)

All numbers in this report are produced by `python reproduce_all_results.py` (stored in `results/`, figures in `figures/`). Derivation details are in `derivations/01–08`.

## 1. Problem

European call/put under Black–Scholes with constant r, σ. With τ = T − t the price V(S, τ) solves

  V_τ = ½σ²S² V_SS + r S V_S − r V,  S ∈ (0, S_max), τ ∈ (0, T],

with V(S, 0) = max(S−K, 0) (call) or max(K−S, 0) (put), and Dirichlet conditions
call: V(0,τ) = 0, V(S_max,τ) = S_max − K e^{−rτ}; put: V(0,τ) = K e^{−rτ}, V(S_max,τ) = 0.
The closed-form price, Delta N(d₁) and Gamma n(d₁)/(Sσ√T) serve as the benchmark.
Parameters throughout: K = 100, T = 1, r = 0.05, σ = 0.2, S_max = 400 = 4K.

## 2. Spatial discretisation and the tridiagonal structure

On S_j = jΔS (ΔS = S_max/M) the central differences V_S ≈ (V_{j+1}−V_{j−1})/(2ΔS) and V_SS ≈ (V_{j+1}−2V_j+V_{j−1})/ΔS² (truncation error O(ΔS²), Taylor expansion about S_j) give, for j = 1…M−1,

  (L_h V)_j = α_j V_{j−1} + β_j V_j + γ_j V_{j+1},
  α_j = σ²S_j²/(2ΔS²) − rS_j/(2ΔS), β_j = −σ²S_j²/ΔS² − r, γ_j = σ²S_j²/(2ΔS²) + rS_j/(2ΔS).

Each equation couples only the three neighbours j−1, j, j+1, so the matrix of any (semi-)implicit step has non-zero entries only on the main and the two adjacent diagonals: it is **tridiagonal**. The boundary values V_0 and V_M are known and are moved to the right-hand side of the first and last equation.

## 3. Time stepping

* **Fully implicit (backward Euler):** (I − Δτ L_h)V^{n+1} = V^n + boundary terms. Lower/main/upper diagonals: −Δτα_j, 1−Δτβ_j, −Δτγ_j. Error O(ΔS² + Δτ); unconditionally stable and L-stable.
* **Crank–Nicolson (θ = ½):** (I − ½Δτ L_h)V^{n+1} = (I + ½Δτ L_h)V^n + boundary terms. Two tridiagonal matrices (left: −½Δτα_j, 1−½Δτβ_j, −½Δτγ_j; right: the corresponding + signs). Error O(ΔS² + Δτ²); A-stable, not L-stable.
* **Rannacher start-up:** the first two steps are backward Euler (size Δτ), the rest CN.

Each system is solved with the **Thomas algorithm**: forward elimination m_i = a_i/b'_{i−1}, b'_i = b_i − m_i c_{i−1}, d'_i = d_i − m_i d'_{i−1}, then back substitution x_i = (d'_i − c_i x_{i+1})/b'_i; O(M) operations. Because the matrices are constant in time, the elimination is done once and reused (`fdm_core.Tridiagonal`). Verification: agreement with LAPACK to 5e-16 on a random 100×100 diagonally dominant system, and an exact 4×4 test.

## 4. Verification against the analytical solution

At M = 200 the ATM call error is 1.10e-02 (implicit, N = 1000) and 9.90e-03 (CN, N = 500); put errors are of the same size, and the put-call parity of the FDM solution holds to 4.0e-09. Price tables for S ∈ [60, 200] are in `results/output_8_convergence_study.txt`.

## 5. Convergence analysis

Total error ≈ C_S ΔS² + C_T Δτ^q. To separate the two terms:

**Spatial error** — remove the time error first (CN with N = 2000; implicit via Richardson extrapolation 2V_N − V_N/2), then refine ΔS on grids where K keeps the same position relative to the nodes:

| M | ΔS | Implicit L∞ | p | CN L∞ | p |
|--:|--:|--:|--:|--:|--:|
| 100 | 4.00 | 4.000e-02 | — | 4.000e-02 | — |
| 200 | 2.00 | 1.001e-02 | 2.00 | 1.001e-02 | 2.00 |
| 400 | 1.00 | 2.499e-03 | 2.00 | 2.499e-03 | 2.00 |
| 800 | 0.50 | 6.246e-04 | 2.00 | 6.245e-04 | 2.00 |

Observed p = 2.00 for both schemes (also in L2 and at S = K; see README).

**Temporal error** — compare with a time-converged solution on the *same* grid (CN, N_ref = 12800), so that the spatial error cancels (M = 200):

| N | Δτ | Implicit | p | CN | p | CN vs exact price |
|--:|--:|--:|--:|--:|--:|--:|
| 25 | 0.04000 | 4.287e-02 | — | 4.035e-04 | — | 9.634e-03 |
| 50 | 0.02000 | 2.147e-02 | 1.00 | 1.010e-04 | 2.00 | 9.915e-03 |
| 100 | 0.01000 | 1.074e-02 | 1.00 | 2.524e-05 | 2.00 | 9.985e-03 |
| 200 | 0.00500 | 5.372e-03 | 1.00 | 6.310e-06 | 2.00 | 1.000e-02 |
| 400 | 0.00250 | 2.686e-03 | 1.00 | 1.576e-06 | 2.00 | 1.001e-02 |
| 800 | 0.00125 | 1.343e-03 | 1.00 | 3.929e-07 | 2.00 | 1.001e-02 |

Observed q = 1.00 (implicit) and 2.00 (CN); identical for M = 100. The last column — the error against the exact price — flattens at the spatial-error floor of the grid. A naive "fix M = 800, vary N, compare with the exact price" study therefore shows CN stalling at ≈6e-4 and the implicit order drifting below 1: that is the spatial error, not a property of the scheme.

## 6. Numerical Greeks

Δ_j = (V_{j+1}−V_{j−1})/(2ΔS), Γ_j = (V_{j+1}−2V_j+V_{j−1})/ΔS² (error O(ΔS²) by Taylor expansion; Γ divides by ΔS², so it is relatively less accurate). At S = K, T = 1, M = 400: Δ error 6.2e-05, Γ error 7.2e-06.

Mesh convergence at S = K (CN, N = 2000), T = 1:

| M | ΔS | Δ error at K | p | Γ error at K | p |
|--:|--:|--:|--:|--:|--:|
| 100 | 4.00 | 1.010e-03 | — | 1.184e-04 | — |
| 200 | 2.00 | 2.500e-04 | 2.01 | 2.911e-05 | 2.02 |
| 400 | 1.00 | 6.236e-05 | 2.00 | 7.248e-06 | 2.01 |
| 800 | 0.50 | 1.558e-05 | 2.00 | 1.810e-06 | 2.00 |

T = 0.1 (kink width σ√T·K = 6.3 instead of 20):

| M | ΔS | Δ error at K | p | Γ error at K | p |
|--:|--:|--:|--:|--:|--:|
| 100 | 4.00 | 4.009e-03 | — | 4.501e-03 | — |
| 200 | 2.00 | 8.837e-04 | 2.18 | 8.530e-04 | 2.40 |
| 400 | 1.00 | 2.162e-04 | 2.03 | 2.036e-04 | 2.07 |
| 800 | 0.50 | 5.378e-05 | 2.01 | 5.035e-05 | 2.02 |

Both converge with order 2. Near the strike at short maturity the error on coarse grids is large (Γ error ≈ 7% of Γ at ΔS = 4) because ΔS is not small compared with the width of the smoothed kink; the Greek accuracy near K is governed by ΔS/(σ√T K).

## 7. Non-smooth payoff, CN oscillations, Rannacher start-up

*Strike position.* Refining grid families with K on a node, or half-way between nodes, both give p → 2; only the constant changes (half-way ≈ 3× more accurate at equal ΔS).

*Oscillations.* The CN amplification factor R(z) = (1+z/2)/(1−z/2) satisfies |R| → 1 as z → −∞, so high-frequency components of the kink are not damped; backward Euler has R = 1/(1−z) → 0 and damps them. λ = Δτσ²K²/ΔS². T = 1, M = 200:

| N | λ | CN price err | Rannacher price err | CN Γ err | Rannacher Γ err | CN min Γ |
|--:|--:|--:|--:|--:|--:|--:|
| 3 | 33.3 | 5.62e-01 | 1.96e-01 | 2.74e-01 | 1.63e-03 | -0.2550 |
| 5 | 20.0 | 2.39e-01 | 7.89e-02 | 1.58e-01 | 6.55e-04 | -0.1394 |
| 10 | 10.0 | 3.39e-02 | 2.76e-02 | 2.24e-02 | 2.64e-04 | 0.0009 |
| 20 | 5.0 | 9.45e-03 | 1.44e-02 | 4.93e-05 | 6.09e-05 | 0.0129 |
| 50 | 2.0 | 9.91e-03 | 1.07e-02 | 3.01e-05 | 3.62e-05 | 0.0129 |
| 200 | 0.5 | 1.00e-02 | 1.01e-02 | 3.13e-05 | 3.17e-05 | 0.0129 |

For λ ≳ 10 CN yields negative Gamma (unphysical); Rannacher restores it and lowers the Gamma error by ×170 (N = 3), ×240 (N = 5), ×85 (N = 10). T = 0.1:

| N | λ | CN price err | Rannacher price err | CN Γ err | Rannacher Γ err | CN min Γ |
|--:|--:|--:|--:|--:|--:|--:|
| 3 | 3.3 | 4.45e-02 | 8.44e-02 | 2.58e-02 | 5.48e-03 | 0.0159 |
| 5 | 2.0 | 2.84e-02 | 5.27e-02 | 7.42e-04 | 2.00e-03 | 0.0153 |
| 10 | 1.0 | 3.13e-02 | 3.75e-02 | 7.45e-04 | 1.21e-03 | 0.0152 |
| 20 | 0.5 | 3.19e-02 | 3.35e-02 | 8.26e-04 | 9.46e-04 | 0.0152 |
| 50 | 0.2 | 3.21e-02 | 3.24e-02 | 8.49e-04 | 8.69e-04 | 0.0152 |
| 200 | 0.1 | 3.21e-02 | 3.22e-02 | 8.53e-04 | 8.54e-04 | 0.0152 |

Here CN does not oscillate at these step sizes and Rannacher only increases the price error for N ≤ 20. Rannacher is a remedy for large Δτ/ΔS², not a universal improvement; for small Δτ plain CN suffices and refinement of ΔS (or node clustering) is what improves the Greeks (figures 4 and 5).

## 8. Conclusions

1. Both schemes are verified against the analytical solution; observed orders agree with theory (space 2; time 1 for implicit and 2 for CN) once the error sources are separated correctly.
2. Spatial and temporal errors must be separated by design (time-converged reference on the same grid; time-error removal for spatial studies).
3. Numerical Delta and Gamma converge with order 2; accuracy near the strike depends on ΔS relative to the kink width, and Gamma is the more sensitive.
4. CN oscillates for large Δτ/ΔS²; two implicit start-up steps remove this; the benefit is regime-dependent.

## 9. Limitations (to be stated openly)

Single parameter set; European options; uniform grid; Dirichlet truncation at 4K not studied separately; Rannacher with full (not half) backward-Euler steps; presentation slides not yet prepared.
