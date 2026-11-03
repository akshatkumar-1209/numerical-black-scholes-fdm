# Numerical Black–Scholes PDE using Finite Difference Methods

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

**Course / Project:** Numerical Partial Differential Equations in Finance  
**Student:** Akshat Kumar  
**Supervisor:** Dr. Ashok Kumar Vaikuntam  
**Course Coordinator:** Prof. Arshad Javed  
**Institution:** BITS Pilani  
**Submission Date:** 3 November 2026 (Final Extension Deadline: 4 November 2026, 6:00 PM)

---

## 📋 Executive Overview

This repository provides a rigorous, verified implementation of finite difference methods (FDM) for the Black–Scholes partial differential equation (PDE) governing European option pricing. 

The investigation covers the full computational pipeline from mathematical foundations and tridiagonal linear solvers to systematic spatial/temporal refinement, convergence order identification, numerical sensitivity analysis (Greeks), and the treatment of non-smooth payoff singularities (Crank–Nicolson oscillations and Rannacher smoothing).

### 🏆 Key Project Artifacts
- **Consolidated Technical Report:** [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md)
- **Viva Defense Guide & Oral Exam Prep:** [`VIVA_PREPARATION_GUIDE.md`](VIVA_PREPARATION_GUIDE.md)
- **Live Reproduction Script (<30s):** [`reproduce_all_results.py`](reproduce_all_results.py)
- **Project Progress Tracker (Excel):** [`Akshat_Numerical_Black_Scholes_Project_Tracker.xlsx`](Akshat_Numerical_Black_Scholes_Project_Tracker.xlsx)
  - Features dedicated worksheets: `Post-Midsem Recovery` (all 10 recovery tasks with full numerical evidence), `Supervisor Review` (evaluation form & viva rubric), `12-Week Tracker`, and `Numerical Results`.

---

## ⚡ Instant Live Reproduction (Viva Demonstration)

To reproduce all numerical experiments, convergence tables, 4x4 grid refinement matrices, Greeks, and Rannacher oscillation suppression data live in front of the supervisor in under 30 seconds:

```bash
# 1. Run master verification and benchmark script
python reproduce_all_results.py

# 2. Regenerate all 4 publication-quality figures
python programs/generate_figures.py
```

---

## 🔬 Core Numerical Findings

### 1. Spatial Convergence ($\mathcal{O}(\Delta S^2)$)
Decoupled by fixing $N=5000$ (temporal error $< 10^{-6}$):
- For strike-aligned grids ($M = 100 	o 200 	o 400$ with $S_{\max}=400, K=100$), error ratios are **$4.11 	o 4.02 	o 4.01$**.
- Confirms exact second-order spatial convergence: $p = \log_2(4) pprox 2.00$.
- *Grid-Alignment Anomaly:* At $M=50$ ($\Delta S=8$), $K=100$ falls between nodes ($S=96$ and $S=104$), introducing an $\mathcal{O}(\Delta S)$ interpolation error that temporarily drops the ratio to 1.87.

### 2. Temporal Convergence ($\mathcal{O}(\Delta 	au)$ vs $\mathcal{O}(\Delta 	au^2)$)
Decoupled by fixing $M=800$ (spatial floor $pprox 3 	imes 10^{-5}$):
- **Fully Implicit (Backward Euler):** Error ratio $pprox 2.00$ when halving $\Delta	au$ across $N \in \{50, 100, 200, 400, 800\} \implies \mathcal{O}(\Delta	au^1)$ first-order.
- **Crank–Nicolson ($	heta = 1/2$):** Error ratio $pprox 3.99$ when halving $\Delta	au \implies \mathcal{O}(\Delta	au^2)$ second-order.

### 3. Complete $4 	imes 4$ Refinement Matrix ($M, N \in \{50, 100, 200, 400\}$)
Tabulated in [`results/output_11_grid_refinement_matrix.txt`](results/output_11_grid_refinement_matrix.txt):
- For Crank–Nicolson, error drops from $5.53 	imes 10^{-2}$ at $(50, 50)$ to $2.50 	imes 10^{-3}$ at $(400, 400)$.
- At equal computational budget, Crank–Nicolson yields $2	imes$ to $10	imes$ lower error than Fully Implicit.

### 4. Numerical Greeks ($\Delta$ and $\Gamma$)
- Evaluated via central differences on the computed price grid:
  $$\Delta_j = rac{V_{j+1} - V_{j-1}}{2\Delta S}, \qquad \Gamma_j = rac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2}$$
- Away from strike, Delta error is $\sim 10^{-5}$ and Gamma error is $\sim 10^{-6}$.
- Near the strike $S pprox K$, second differencing amplifies discretisation errors by $1/\Delta S$, producing elevated errors that resolve under fine spatial meshing.

### 5. Payoff Kink, CN Oscillations & Rannacher Smoothing
- Crank–Nicolson is A-stable but not L-stable ($|R_{CN}(z)| 	o 1$ as $z 	o \infty$). It does not dissipate high-frequency spatial modes excited by the discontinuous payoff slope at $S=K$.
- Under coarse time grids ($N \le 20$), standard CN exhibits spurious, non-physical spatial oscillations in $\Gamma$.
- **Rannacher Smoothing** replaces the first two steps with Backward Euler ($|R_{BE}(z)| 	o 0$ as $z 	o \infty$, L-stable), aggressively damping high frequencies before switching to Crank–Nicolson.
- **Suppression Factor:** At $N=5$, oscillation amplitude drops from $3.28 	imes 10^{-2}$ to $1.40 	imes 10^{-3}$ (**$23.4	imes$ reduction**); at $N=10$, oscillation is reduced by **$95.2	imes$** with $<0.5\%$ change in option price.

---

## 📊 Visual Verification & Error Plots

| Figure 1: Spatial Convergence $\mathcal{O}(\Delta S^2)$ | Figure 2: Temporal Convergence $\mathcal{O}(\Delta 	au)$ vs $\mathcal{O}(\Delta 	au^2)$ |
|:---:|:---:|
| ![Spatial Convergence](figures/fig1_spatial_convergence.png) | ![Temporal Convergence](figures/fig2_temporal_convergence.png) |

| Figure 3: Numerical Greeks Profile ($\Delta$ & $\Gamma$) | Figure 4: CN Oscillations vs Rannacher Smoothing |
|:---:|:---:|
| ![Numerical Greeks](figures/fig3_numerical_greeks.png) | ![Rannacher Smoothing](figures/fig4_cn_oscillations_rannacher.png) |

---

## 📁 Repository Directory Tree

```
numerical-black-scholes-fdm/
├── Akshat_Numerical_Black_Scholes_Project_Tracker.xlsx  Master Tracker (7 worksheets including Recovery Plan)
├── README.md                                          Academic README with verification evidence
├── TECHNICAL_REPORT.md                                Comprehensive Technical Report (Weeks 1-12)
├── VIVA_PREPARATION_GUIDE.md                          Viva defense Q&A and derivation cheat sheet
├── reproduce_all_results.py                           Master live reproduction script (<30s)
├── requirements.txt                                   Python environment dependencies
├── programs/
│   ├── black_scholes.py                               Analytical Black-Scholes benchmark calculator
│   ├── generate_grid.py                               Domain discretisation & payoff generator
│   ├── finite_difference_derivatives.py               Taylor series derivative verification
│   ├── thomas_solver.py                               Thomas algorithm tridiagonal solver
│   ├── implicit_fdm.py                                Fully Implicit (Backward Euler) solver
│   ├── crank_nicolson.py                              Crank-Nicolson (theta=1/2) solver
│   ├── convergence_study.py                           Spatial & temporal refinement studies
│   ├── numerical_greeks.py                            Numerical Delta & Gamma sensitivity module
│   ├── payoff_kink_cn_oscillation.py                  Kink oscillations & Rannacher smoothing
│   └── generate_figures.py                            High-resolution figure generator
├── derivations/
│   ├── 01_finite_difference_approximations.md         Spatial Taylor approximations & errors
│   ├── 02_black_scholes_pde_discretisation.md         Forward tau=T-t transform & stencil
│   ├── 03_thomas_algorithm.md                         Forward elimination & back substitution
│   ├── 04_fully_implicit_method.md                    Backward Euler discretisation & stability
│   ├── 05_crank_nicolson_method.md                    Theta=1/2 discretisation & two matrices
│   ├── 06_convergence_analysis.md                     Decoupled spatial/temporal error math
│   ├── 07_numerical_greeks.md                         Delta and Gamma divided difference errors
│   └── 08_payoff_kink_and_cn_oscillations.md          L-stability, Fourier symbols & Rannacher
├── results/
│   ├── output_8_convergence_study.txt                 Price tables & decoupled convergence data
│   ├── output_9_numerical_greeks.txt                  Delta & Gamma sensitivity tables
│   ├── output_10_payoff_kink_oscillation.txt          CN oscillation vs Rannacher data
│   └── output_11_grid_refinement_matrix.txt           Complete 4x4 matrix for M,N in {50,100,200,400}
└── figures/
    ├── fig1_spatial_convergence.png                   Log-log plot: Spatial error vs dS
    ├── fig2_temporal_convergence.png                  Log-log plot: Temporal error vs dt
    ├── fig3_numerical_greeks.png                      Delta and Gamma sensitivity profiles
    └── fig4_cn_oscillations_rannacher.png              CN oscillations vs Rannacher curve
```

---

## 🛠️ Requirements & Installation

```bash
# Clone the repository
git clone https://github.com/akshatkumar-1209/numerical-black-scholes-fdm.git
cd numerical-black-scholes-fdm

# Install dependencies
pip install numpy scipy matplotlib openpyxl
```

---
*Submitted for evaluation under the supervision of Dr. Ashok Kumar Vaikuntam, Department of Mathematics / Economics & Finance, BITS Pilani.*
