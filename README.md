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

## 🗓️ Project Roadmap (12 Weeks)

- [x] **Week 1:** Mathematical foundations, continuous Black-Scholes PDE, analytical pricing, Greeks, put-call parity.
- [x] **Week 2:** Taylor expansions, finite difference discretizations ($V_S, V_{SS}$), truncation error analysis, computational mesh generation.
- [x] **Week 3:** Tridiagonal matrix systems, Thomas algorithm implementation, verification against standard linear algebra solvers.
- [ ] **Week 4:** Derivation and coefficient assembly for the Fully Implicit Finite Difference Method.
- [ ] **Week 5:** Implementation and time-stepping of the Fully Implicit scheme.
- [ ] **Week 6:** Derivation and implementation of the Crank-Nicolson Method ($\theta = 1/2$).
- [ ] **Week 7:** Comparison of Implicit vs Crank-Nicolson schemes; oscillation inspection near non-smooth payoff.
- [ ] **Week 8:** Spatial and temporal mesh-refinement studies, Richardson extrapolation, observed order of convergence.
- [ ] **Week 9:** Numerical Greek computation ($\Delta, \Gamma$) directly from discrete solution vectors.
- [ ] **Week 10:** Stability and monotonicity analysis, maximum principle, M-matrix property validation.
- [ ] **Week 11:** Performance benchmarking, runtime profiling, sparsity benefits of TDMA.
- [ ] **Week 12:** Final consolidated report, presentation, and archival.
