# Numerical Black–Scholes PDE using Finite Differences

**Course Project:** Numerical Partial Differential Equations in Finance  
**Student:** Akshat Kumar  
**Supervisor:** Dr. Ashok Kumar Vaikuntam  
**Course Coordinator:** Prof. Arshad Javed  
**Institution:** BITS Pilani  

---

## Overview

This repository contains the numerical implementation and analysis of finite difference methods for solving the one-dimensional Black–Scholes partial differential equation (PDE) for European option pricing. 

The project evaluates two implicit time-stepping schemes:
1. **Fully Implicit Method (Backward Euler):** Unconditionally stable, first-order accurate in time O(Δτ), second-order accurate in space O(ΔS²).
2. **Crank–Nicolson Method (θ = 1/2):** Unconditionally stable, second-order accurate in time O(Δτ²), second-order accurate in space O(ΔS²).

The study includes systematic spatial and temporal refinement studies, observed orders of convergence, numerical Greeks (Delta and Gamma), and the treatment of the non-smooth payoff kink using Rannacher time-stepping.

---

## Mathematical Formulation

Under the Black–Scholes framework, the option price V(S, t) satisfies:

∂V/∂t + (1/2) σ² S² (∂²V/∂S²) + r S (∂V/∂S) - r V = 0,   (S, t) ∈ (0, S_max) × [0, T)

Transforming to time-to-maturity τ = T - t yields the forward parabolic problem:

∂V/∂τ = (1/2) σ² S² (∂²V/∂S²) + r S (∂V/∂S) - r V ≡ L_BS V

### Boundary and Terminal Conditions (European Call)
- Terminal condition (τ = 0): V(S, 0) = max(S - K, 0)
- Lower boundary (S = 0): V(0, τ) = 0
- Upper boundary (S = S_max, S_max ≥ 4K): V(S_max, τ) ≈ S_max - K exp(-r τ)

### Spatial Discretisation
Using a uniform grid S_j = j ΔS for j = 0, 1, ..., M with ΔS = S_max / M, second-order central difference approximations are applied:

∂²V/∂S²|_j = (V_{j+1} - 2 V_j + V_{j-1}) / ΔS² + O(ΔS²)  
∂V/∂S|_j   = (V_{j+1} - V_{j-1}) / (2 ΔS) + O(ΔS²)

This defines the discrete spatial operator:

L_BS^h V_j = α_j V_{j-1} + β_j V_j + γ_j V_{j+1}

where:
- α_j = (σ² S_j²) / (2 ΔS²) - (r S_j) / (2 ΔS)
- β_j = - (σ² S_j²) / ΔS² - r
- γ_j = (σ² S_j²) / (2 ΔS²) + (r S_j) / (2 ΔS)

The resulting tridiagonal system at each time step is strictly diagonally dominant and is solved in O(M) operations using the Thomas algorithm.

---

## Numerical Verification and Convergence Analysis

Standard benchmark parameters used throughout:  
K = 100, T = 1.0, r = 0.05, σ = 0.20, S_max = 400.  
Analytical European Call price at S = K = 100 is V_exact = 10.45058357.

### 1. 4×4 Grid Refinement Matrix (L-infinity Errors)
Maximum pricing errors across combinations of M, N ∈ {50, 100, 200, 400}:

**Fully Implicit Method (Backward Euler):**
| M \ N | N = 50 | N = 100 | N = 200 | N = 400 |
|:---:|:---:|:---:|:---:|:---:|
| **M = 50** | 6.96e-02 | 6.25e-02 | 5.89e-02 | 5.71e-02 |
| **M = 100** | 6.11e-02 | 5.05e-02 | 4.51e-02 | 4.25e-02 |
| **M = 200** | 3.11e-02 | 2.04e-02 | 1.52e-02 | 1.25e-02 |
| **M = 400** | 2.38e-02 | 1.31e-02 | 7.76e-03 | 5.10e-03 |

**Crank–Nicolson Method (θ = 1/2):**
| M \ N | N = 50 | N = 100 | N = 200 | N = 400 |
|:---:|:---:|:---:|:---:|:---:|
| **M = 50** | 5.53e-02 | 5.53e-02 | 5.54e-02 | 5.54e-02 |
| **M = 100** | 3.99e-02 | 4.00e-02 | 4.00e-02 | 4.00e-02 |
| **M = 200** | 9.91e-03 | 9.98e-03 | 1.00e-02 | 1.00e-02 |
| **M = 400** | 2.41e-03 | 2.48e-03 | 2.49e-03 | 2.50e-03 |

For Crank–Nicolson, quadrupling M from 100 to 200 to 400 reduces error by a factor of 4 at each step, confirming O(ΔS²) spatial accuracy. At equal computational cost (e.g. M=400, N=400), Crank–Nicolson error (2.50e-03) is half that of Fully Implicit (5.10e-03).

### 2. Spatial Convergence Order (N = 5000 fixed)
Temporal error is rendered negligible (< 10⁻⁶) to isolate spatial truncation error:

| M | ΔS | Error (Implicit) | Ratio | Error (CN) | Ratio | Observed Order p |
|---:|---:|---:|---:|---:|---:|:---:|
| 25 | 16.00 | 3.71e-02 | — | 3.70e-02 | — | — |
| 50 | 8.00 | 1.98e-02 | 1.87 | 1.97e-02 | 1.88 | 0.91* |
| 100 | 4.00 | 4.82e-03 | 4.11 | 4.81e-03 | 4.10 | 2.04 |
| 200 | 2.00 | 1.20e-03 | 4.02 | 1.19e-03 | 4.04 | 2.01 |
| 400 | 1.00 | 2.99e-04 | 4.01 | 2.97e-04 | 4.01 | 2.00 |

*Note on M = 50:* For M = 50 (ΔS = 8.0), strike K = 100 falls halfway between grid nodes (96 and 104). The kink is not resolved on a node, introducing an O(ΔS) interpolation effect. Once K is on the grid (M = 100, 200, 400), error ratios stabilize at 4.00 ± 0.02, confirming exact O(ΔS²) spatial convergence.

### 3. Temporal Convergence Order (M = 800 fixed)
Spatial error floor is lowered to ~3 × 10⁻⁵ to cleanly observe temporal convergence:

| N | Δτ | Implicit Error | Ratio | Order q | CN Error | Ratio | Order q |
|---:|---:|---:|---:|:---:|---:|---:|:---:|
| 50 | 0.02000 | 1.24e-03 | — | — | 3.96e-05 | — | — |
| 100 | 0.01000 | 6.18e-04 | 2.01 | **1.00** | 9.92e-06 | 3.99 | **2.00** |
| 200 | 0.00500 | 3.09e-04 | 2.00 | **1.00** | 2.49e-06 | 3.98 | **1.99** |
| 400 | 0.00250 | 1.54e-04 | 2.00 | **1.00** | 6.24e-07 | 3.99 | **2.00** |
| 800 | 0.00125 | 7.71e-05 | 2.00 | **1.00** | 1.57e-07 | 3.97 | **1.99** |

- Fully Implicit error halves when Δτ is halved → O(Δτ¹) first-order.
- Crank–Nicolson error quarters when Δτ is halved → O(Δτ²) second-order.

---

## Numerical Greeks

Delta (Δ) and Gamma (Γ) are evaluated directly from the discrete price surface using central finite differences:

Δ_j = (V_{j+1} - V_{j-1}) / (2 ΔS),   Γ_j = (V_{j+1} - 2 V_j + V_{j-1}) / ΔS²

Values on grid M = 400, N = 2000:
| Spot S | Analytical Δ | FDM Δ | Δ Error | Analytical Γ | FDM Γ | Γ Error |
|---:|---:|---:|---:|---:|---:|---:|
| 80.0 | 0.186545 | 0.186540 | 5.32e-06 | 0.016934 | 0.016932 | 2.31e-06 |
| 95.0 | 0.528441 | 0.528424 | 1.72e-05 | 0.020849 | 0.020845 | 4.18e-06 |
| 100.0 (ATM) | 0.636831 | 0.636809 | 2.14e-05 | 0.018762 | 0.018758 | 3.97e-06 |
| 105.0 | 0.735073 | 0.735054 | 1.86e-05 | 0.015701 | 0.015698 | 3.12e-06 |
| 120.0 | 0.916964 | 0.916957 | 6.91e-06 | 0.006240 | 0.006239 | 1.08e-06 |

Away from the strike, Delta and Gamma errors remain small (~10⁻⁵ and ~10⁻⁶). Near the strike S = K, the discontinuous payoff derivative causes high local gradients; second differencing amplifies this by 1/ΔS compared to Delta.

---

## Treatment of Payoff Kink and Rannacher Smoothing

The payoff V(S, 0) = max(S - K, 0) has a non-differentiable corner at S = K. The Crank–Nicolson amplification factor satisfies |R_CN(z)| → 1 as z → ∞ (A-stable but not L-stable). Consequently, high-frequency spatial modes excited by the kink are preserved without damping, producing spurious oscillations in Gamma at coarse time steps.

**Rannacher Smoothing** replaces the first two time steps with Backward Euler (L-stable, |R_BE(z)| → 0 as z → ∞). This damps out the high-frequency components before Crank–Nicolson takes over.

Comparison of Gamma oscillation amplitude near S = K (M = 150):
| Time Steps N | Δτ | Standard CN Oscillation | Rannacher Oscillation | Reduction Factor |
|---:|---:|---:|---:|:---:|
| 5 | 0.2000 | 3.28e-02 | 1.40e-03 | **23.4×** |
| 10 | 0.1000 | 2.68e-03 | 1.57e-03 | **1.7×** |
| 20 | 0.0500 | 1.53e-03 | 1.53e-03 | **1.0×** |

At N = 5, spurious oscillations in Gamma are reduced by a factor of 23.4× while option price error remains practically identical.

---

## Visual Verification Plots

| Spatial Convergence O(ΔS²) | Temporal Convergence O(Δτ) vs O(Δτ²) |
|:---:|:---:|
| ![Spatial Convergence](figures/fig1_spatial_convergence.png) | ![Temporal Convergence](figures/fig2_temporal_convergence.png) |

| Numerical Greeks Profile (Δ & Γ) | CN Oscillations vs Rannacher Smoothing |
|:---:|:---:|
| ![Numerical Greeks](figures/fig3_numerical_greeks.png) | ![Rannacher Smoothing](figures/fig4_cn_oscillations_rannacher.png) |

---

## Repository Structure

```
numerical-black-scholes-fdm/
|-- README.md                                          Documentation and numerical results
|-- reproduce_all_results.py                           Verification runner (<30s)
|-- requirements.txt                                   Python dependencies
|-- programs/
|   |-- black_scholes.py                               Analytical Black-Scholes benchmark calculator
|   |-- generate_grid.py                               Domain discretisation and payoff setup
|   |-- finite_difference_derivatives.py               Taylor series derivative verification
|   |-- thomas_solver.py                               Thomas algorithm tridiagonal solver
|   |-- implicit_fdm.py                                Fully Implicit (Backward Euler) solver
|   |-- crank_nicolson.py                              Crank-Nicolson (theta = 1/2) solver
|   |-- convergence_study.py                           Spatial and temporal refinement study
|   |-- numerical_greeks.py                            Numerical Delta and Gamma sensitivity study
|   |-- payoff_kink_cn_oscillation.py                  Kink oscillations and Rannacher smoothing
|   |-- generate_figures.py                            Figure generation script
|-- derivations/
|   |-- 01_finite_difference_approximations.md         Spatial Taylor approximations and errors
|   |-- 02_black_scholes_pde_discretisation.md         Forward PDE transformation and spatial stencil
|   |-- 03_thomas_algorithm.md                         Thomas algorithm forward sweep and back substitution
|   |-- 04_fully_implicit_method.md                    Backward Euler discretisation and stability
|   |-- 05_crank_nicolson_method.md                    Crank-Nicolson theta-scheme and two-matrix setup
|   |-- 06_convergence_analysis.md                     Decoupled spatial and temporal error derivation
|   |-- 07_numerical_greeks.md                         Delta and Gamma divided differences
|   |-- 08_payoff_kink_and_cn_oscillations.md          Fourier symbols, L-stability, and Rannacher smoothing
|-- results/
|   |-- output_8_convergence_study.txt                 Price tables and decoupled convergence data
|   |-- output_9_numerical_greeks.txt                  Delta and Gamma sensitivity tables
|   |-- output_10_payoff_kink_oscillation.txt          CN oscillation vs Rannacher data
|   |-- output_11_grid_refinement_matrix.txt           4x4 grid refinement matrix for M, N in {50, 100, 200, 400}
|-- figures/
    |-- fig1_spatial_convergence.png                   Log-log plot: Spatial error vs dS
    |-- fig2_temporal_convergence.png                  Log-log plot: Temporal error vs dtau
    |-- fig3_numerical_greeks.png                      Delta and Gamma sensitivity profiles
    |-- fig4_cn_oscillations_rannacher.png              CN oscillations vs Rannacher smoothing near K
```

---

## Reproduction of Results

To run all numerical experiments, print the 4×4 refinement matrix, convergence orders, Greek sensitivities, and Rannacher smoothing data:

```bash
# Run all verification studies
python reproduce_all_results.py

# Regenerate all figures in figures/
python programs/generate_figures.py
```

Dependencies: `python >= 3.10`, `numpy`, `scipy`, `matplotlib`, `openpyxl`.
