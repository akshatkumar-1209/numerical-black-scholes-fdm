# Numerical Solution of the Black-Scholes Equation using Finite Difference Methods

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-scientific-informational.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: In Progress](https://img.shields.io/badge/Status-In%20Progress%20(Phase%201)-success.svg)]()

**Student:** Akshat Kumar  
**Supervisor:** Dr. Ashok Kumar Vaikuntam  
**Primary Reference:** Daniel J. Duffy, *Numerical Methods in Computational Finance: A Partial Differential Equation (PDE/FDM) Approach*, Wiley (2022)  

---

## 📌 Project Overview

This research project implements, verifies, and analyzes finite difference methods (FDM) for pricing financial derivatives governed by the **Black-Scholes partial differential equation (PDE)**. Rather than treating Black-Scholes pricing as a black-box formula, this work focuses on the complete, rigorous mathematical and computational pipeline:

$$\text{Reading} \longrightarrow \text{Mathematical Derivation} \longrightarrow \text{Program Development} \longrightarrow \text{Testing \& Verification} \longrightarrow \text{Numerical Results}$$

### The Black-Scholes Initial-Boundary Value Problem (IBVP)

The continuous pricing equation for a European option $V(S, t)$ with underlying asset price $S \ge 0$ and calendar time $t \in [0, T]$ is:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$$

Transforming to time-to-maturity $\tau = T - t$ yields the forward parabolic problem:

$$\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V$$

subject to:
- **Initial condition ($\\tau = 0$):** Terminal payoff $V(S, 0) = \max(S - K, 0)$ for Call, or $\max(K - S, 0)$ for Put.
- **Boundary condition at $S = 0$:** $V(0, \tau) = 0$ (Call), $V(0, \tau) = K e^{-r\tau}$ (Put).
- **Boundary condition as $S \to S_{\max}$:** $V(S_{\max}, \tau) \approx S_{\max} - K e^{-r\tau}$ (Call), $V(S_{\max}, \tau) \approx 0$ (Put).

---

## 🎯 Standard Benchmark Parameters

All numerical benchmarks, analytical baselines, and convergence experiments are evaluated using standard parameters:

| Parameter | Symbol | Value | Description |
|:---|:---:|:---:|:---|
| Strike Price | $K$ | `100.0` | Option exercise price |
| Time to Expiry | $T$ | `1.0` year | Maturity horizon |
| Risk-Free Interest Rate | $r$ | `0.05` (5%) | Continuously compounded annual rate |
| Asset Volatility | $\sigma$ | `0.20` (20%) | Annualized standard deviation of returns |
| Truncated Boundary | $S_{\max}$ | `400.0` | Truncation limit ($4 \times K$) |

---

## 📂 Repository Structure

```
numerical-black-scholes-fdm/
├── README.md                                # Comprehensive documentation
├── requirements.txt                         # Python package dependencies
├── .gitignore                               # Ignore cache & temporary files
├── derivations/                             # Mathematical foundations & derivations
│   ├── 01_finite_difference_approximations.md # Taylor expansion & truncation error analysis
│   ├── 02_black_scholes_pde_discretisation.md # Continuous-to-discrete FDM formulation
│   └── 03_thomas_algorithm.md               # Tridiagonal TDMA algorithm derivation & O(N) complexity
├── programs/                                # Python numerical source code
│   ├── black_scholes_analytic.py            # Exact Black-Scholes solver & Greek calculator
│   ├── finite_difference_derivatives.py     # Convergence verification of FD stencils
│   ├── generate_grid.py                     # Non-uniform/uniform grid & payoff generator
│   └── thomas_solver.py                     # Optimized Thomas algorithm (tridiagonal solver)
└── results/                                 # Verification outputs & benchmark logs
    ├── output_1_black_scholes_analytic.txt  # Analytical benchmark run log
    ├── output_2_finite_difference.txt       # O(h) and O(h^2) empirical convergence log
    ├── output_3_grid.txt                    # Discretization grid & boundary log
    └── output_4_thomas.txt                  # Linear solver benchmark log
```

---

## 🔬 Implemented Modules & Verification Status

### 1. Analytical Benchmark (`programs/black_scholes_analytic.py`)
- Computes exact European Call/Put prices using the cumulative normal distribution function $N(d_1), N(d_2)$.
- Calculates analytical Greeks: Delta ($\Delta = \frac{\partial V}{\partial S}$) and Gamma ($\Gamma = \frac{\partial^2 V}{\partial S^2}$).
- **Verification:** Automatically validates put-call parity:
  $$|C(S, t) - P(S, t) - S + K e^{-r(T-t)}| < 10^{-15}$$
- **Benchmark values for $S = 100$:**
  - ATM Call Price = `10.450584`
  - ATM Put Price = `5.573526`
  - ATM Delta (Call) = `0.636831`
  - ATM Gamma = `0.018762`

### 2. Finite Difference Derivative Verification (`programs/finite_difference_derivatives.py`)
- Derives and verifies forward difference, backward difference, and symmetric central difference stencils from Taylor expansions.
- Tested on benchmark function $f(x) = \sin(x) e^{-x/5}$ at $x_0 = 1.0$.
- **Empirical convergence rates:**
  - Forward Difference: Error halves as $h \to h/2$ $\implies \mathcal{O}(h)$ first-order accuracy confirmed.
  - Central First Difference: Error quarters as $h \to h/2$ (ratio $\approx 4.00$) $\implies \mathcal{O}(h^2)$ second-order accuracy confirmed.
  - Central Second Difference ($V_{SS}$ stencil): Ratio $\approx 4.00$ $\implies \mathcal{O}(h^2)$ confirmed.

### 3. Grid Generation & Boundary Setup (`programs/generate_grid.py`)
- Sets up spatial discretization $S_j = j \Delta S$ ($j = 0, \dots, M$) and temporal discretization $\tau_n = n \Delta \tau$ ($n = 0, \dots, N$).
- Generates exact terminal payoff vectors $\max(S - K, 0)$ and $\max(K - S, 0)$.
- Enforces boundary conditions at $S = 0$ and $S = S_{\max}$.

### 4. Thomas Algorithm Tridiagonal Solver (`programs/thomas_solver.py`)
- Implements the specialized Tridiagonal Matrix Algorithm (TDMA) utilizing forward elimination and backward substitution.
- Achieves optimal $\mathcal{O}(N)$ computational complexity compared to $\mathcal{O}(N^3)$ for general Gaussian elimination.
- **Verification:**
  - Test 1: $4 \times 4$ known analytical system with exact solution $[1, 1, 1, 1]^T$ $\implies$ Exact match (norm difference $= 0.00$).
  - Test 2: Random $100 \times 100$ diagonally dominant matrix compared against standard `numpy.linalg.solve` $\implies \|x_{\text{Thomas}} - x_{\text{NumPy}}\| = 4.96 \times 10^{-16}$.

---

## 📊 Numerical Verification Summary

| Test Case / Experiment | Expected / Theoretical | Computed Numerical | Error / Residual | Status |
|:---|:---:|:---:|:---:|:---:|
| **Put-Call Parity ($S=100$)** | $C - P - S + K e^{-rT} = 0$ | Exact parity | $0.00 \times 10^{-16}$ | ✅ PASSED |
| **FD First Derivative Central** | $\mathcal{O}(h^2)$ convergence | Ratio $= 3.99 - 4.00$ | Truncation error confirmed | ✅ PASSED |
| **FD Second Derivative Central ($V_{SS}$)** | $\mathcal{O}(h^2)$ convergence | Ratio $= 4.00$ | Truncation error confirmed | ✅ PASSED |
| **Boundary Condition $S=0$** | Call: $0.0$, Put: $100.0$ | $0.0000$, $100.0000$ | $0.00$ | ✅ PASSED |
| **Thomas Solver ($4 \times 4$)** | $[1, 1, 1, 1]$ | $[1.0, 1.0, 1.0, 1.0]$ | $0.00$ | ✅ PASSED |
| **Thomas Solver ($100 \times 100$)** | Matches NumPy LAPACK | $\|x_T - x_{NP}\|$ | $4.96 \times 10^{-16}$ | ✅ PASSED |

---

## 🚀 How to Run the Programs

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running Individual Modules
```bash
# 1. Exact Black-Scholes analytical benchmark and Greeks
python programs/black_scholes_analytic.py

# 2. Finite difference truncation error convergence study
python programs/finite_difference_derivatives.py

# 3. Spatial and temporal grid construction and payoff validation
python programs/generate_grid.py

# 4. Thomas algorithm tridiagonal linear system solver verification
python programs/thomas_solver.py
```

---

## 🗓️ Project Roadmap & Status (Aligned with 12-Week Tracker)

| Week | Milestone / Topic | Focus & Deliverable | Status | % Done |
|:---:|:---|:---|:---:|:---:|
| **1** | **Black-Scholes Formulation & Benchmark** | Analytical Call/Put solver, Greeks ($\Delta, \Gamma$), put-call parity check | ✅ Completed | **100%** |
| **2** | **Finite-Difference Foundations** | Taylor expansion derivations, $\mathcal{O}(h)$ vs $\mathcal{O}(h^2)$ empirical convergence | ✅ Completed | **100%** |
| **3** | **Discrete Black-Scholes Operator** | Three-point spatial stencil, $lpha_j, eta_j, \gamma_j$ coefficients derivation | 🔄 In Progress | **80%** |
| **4** | **Numerical Linear Algebra - Thomas Algorithm** | TDMA derivation, $\mathcal{O}(N)$ complexity, $4\times 4$ and $100\times 100$ verification | ✅ Completed | **100%** |
| **5** | **Fully Implicit Black-Scholes Method** | Duffy Chapter 7 reading, boundary enforcement, implicit time-stepping solver | 🔄 In Progress | **25%** |
| **6** | **Crank-Nicolson Scheme** | $\theta$-scheme derivation ($\\theta = 1/2$), matrix system assembly | ⏳ Not Started | **0%** |
| **7** | **Verification & Error Measurement** | Pointwise, $L_2$, and $L_\infty$ error norms against analytical benchmark | ⏳ Not Started | **0%** |
| **8** | **Stability & Convergence Studies** | Spatial/temporal mesh refinement, observed order of convergence | ⏳ Not Started | **0%** |
| **9** | **Numerical Greeks Computation** | Finite-difference computation of Delta ($\Delta$) and Gamma ($\Gamma$) from discrete solution | ⏳ Not Started | **0%** |
| **10** | **Monotonicity & Non-Oscillation** | Maximum principle, M-matrix property, mesh constraints | ⏳ Not Started | **0%** |
| **11** | **Non-Smooth Payoff & Improved CN** | Strike alignment, Rannacher start-up damping | ⏳ Not Started | **0%** |
| **12** | **Consolidation, Final Report & Presentation** | Codebase integration, reproduction tests, final project presentation | ⏳ Not Started | **0%** |


---

## 🔬 Week 5 — Fully Implicit FDM (`programs/implicit_solver.py`)

### Mathematical Derivation

Backward Euler temporal discretisation at the new time level $n+1$:

$$\frac{V_j^{n+1} - V_j^n}{\Delta\tau} = \mathcal{L}_{BS} V_j^{n+1}$$

Yields a tridiagonal system per step:

$$a_j V_{j-1}^{n+1} + b_j V_j^{n+1} + c_j V_{j+1}^{n+1} = V_j^n$$

$$a_j = -\Delta\tau\,\alpha_j, \quad b_j = 1 - \Delta\tau\,\beta_j, \quad c_j = -\Delta\tau\,\gamma_j$$

| Property | Value |
|:---|:---|
| Spatial accuracy | $\mathcal{O}(\Delta S^2)$ |
| Temporal accuracy | $\mathcal{O}(\Delta\tau)$ |
| Stability | **Unconditionally stable** |

**Convergence verified:** Spatial ratios ≈ 4.0 (M=100→200→400). Temporal ratios ≈ 1.9 (N=50→800 with M=800).

---

## 🔬 Week 6 — Crank-Nicolson FDM (`programs/crank_nicolson_solver.py`)

### Mathematical Derivation

The $\theta = 1/2$ member of the theta-scheme family. Averages spatial operators at both time levels:

$$\frac{V_j^{n+1} - V_j^n}{\Delta\tau} = \frac{1}{2}\mathcal{L}_{BS} V_j^{n+1} + \frac{1}{2}\mathcal{L}_{BS} V_j^n$$

Gives the two-matrix system $A \mathbf{V}^{n+1} = B \mathbf{V}^n + \mathbf{b}_{\text{bc}}$:

| Matrix | Sub-diag | Main-diag | Super-diag |
|:---:|:---:|:---:|:---:|
| $A$ (LHS) | $-\frac{\Delta\tau}{2}\alpha_j$ | $1 - \frac{\Delta\tau}{2}\beta_j$ | $-\frac{\Delta\tau}{2}\gamma_j$ |
| $B$ (RHS) | $+\frac{\Delta\tau}{2}\alpha_j$ | $1 + \frac{\Delta\tau}{2}\beta_j$ | $+\frac{\Delta\tau}{2}\gamma_j$ |

### Why O(Δτ²)?

CN is equivalent to a **central difference in time** at midpoint $\tau_{n+1/2}$. The truncation error is $\mathcal{O}(\Delta\tau^2)$ versus $\mathcal{O}(\Delta\tau)$ for Backward Euler.

| Property | Value |
|:---|:---|
| Spatial accuracy | $\mathcal{O}(\Delta S^2)$ |
| Temporal accuracy | $\mathcal{O}(\Delta\tau^2)$ ← **second-order** |
| Stability | **Unconditionally stable** |

**Convergence verified:** Spatial ratios 4.00, 4.01 (M=100→200→400, N=5000). Temporal ratio ≈ 3.6 (N=10→20, temporal-dominated).

**Robustness:** Tested across σ ∈ {0.15, 0.35, 0.50}, T ∈ {0.25, 0.5, 0.75, 2.0}, K ∈ {50, 80, 100, 120, 150}, r ∈ {3%, 4%, 5%, 8%, 10%}.

---

## 🗃️ Complete Repository Contents

```
numerical-black-scholes-fdm/
├── README.md
├── requirements.txt
├── programs/
│   ├── black_scholes_analytic.py        Week 1  Analytical benchmark, Greeks
│   ├── finite_difference_derivatives.py Week 2  FD convergence verification
│   ├── generate_grid.py                 Week 2  Grid + payoff generation
│   ├── thomas_solver.py                 Week 3  Thomas algorithm
│   ├── implicit_solver.py               Week 5  Fully Implicit FDM  ✅ NEW
│   ├── crank_nicolson_solver.py         Week 6  Crank-Nicolson FDM  ✅ NEW
│   └── method_comparison.py             Week 6  Robustness comparison ✅ NEW
├── derivations/
│   ├── 01_finite_difference_approximations.md
│   ├── 02_black_scholes_pde_discretisation.md
│   ├── 03_thomas_algorithm.md
│   ├── 04_fully_implicit_method.md      ✅ NEW
│   └── 05_crank_nicolson_method.md      ✅ NEW
└── results/
    ├── output_1_black_scholes_analytic.txt
    ├── output_2_finite_difference.txt
    ├── output_3_grid.txt
    ├── output_4_thomas.txt
    ├── output_5_implicit_solver.txt     ✅ NEW
    ├── output_6_crank_nicolson.txt      ✅ NEW
    └── output_7_method_comparison.txt   ✅ NEW
```
