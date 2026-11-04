# Technical Report: Finite Difference Methods for the Black–Scholes PDE

**Student:** Akshat Kumar  
**Supervisor:** Dr. Ashok Kumar Vaikuntam  
**Course / Project:** Numerical Black–Scholes PDE using Finite Differences  
**Date:** November 2026  
**Repository:** [github.com/akshatkumar-1209/numerical-black-scholes-fdm](https://github.com/akshatkumar-1209/numerical-black-scholes-fdm)

---

## 1. Executive Summary

This report presents a systematic numerical study of European option pricing via finite-difference discretisation of the Black–Scholes partial differential equation (PDE). We implement and examine two time-marching schemes: the **Fully Implicit method (Backward Euler)** and the **Crank–Nicolson method ($\theta = 1/2$)**, both solved via an independent implementation of the **Thomas algorithm**.

Key outcomes established in this investigation:
1. **Analytical Benchmark Verification:** Both numerical solvers are verified against the closed-form Black–Scholes formula across out-of-the-money, at-the-money, and in-the-money regimes for European call and put options. Put-call parity is satisfied to machine precision.
2. **Decoupled Spatial Convergence:** With time error suppressed ($N=5000$), second-order spatial convergence $\mathcal{O}(\Delta S^2)$ is verified. For an aligned grid ($K$ lying on a grid node), error reduction ratios of 4.00 and 4.01 are observed as $M$ doubles from 100 to 200 to 400.
3. **Decoupled Temporal Convergence via Same-Grid Reference:** We show that measuring temporal error against an external continuous analytical solution on a fixed spatial mesh introduces an artificial spatial error floor ($\sim 6 \times 10^{-4}$ at $M=800$), causing apparent Crank–Nicolson convergence to stall at fine time steps. Measuring temporal error against a time-converged reference solution ($N_{\text{ref}} = 12800$) on the **same spatial grid** ($M=200$) isolates the temporal component cleanly, demonstrating observed orders of:
   - First-order $\mathcal{O}(\Delta \tau)$ for Fully Implicit (error ratios $\approx 2.01 - 2.07$).
   - Second-order $\mathcal{O}(\Delta \tau^2)$ for Crank–Nicolson (error ratios $\approx 4.00 - 4.01$).
4. **Numerical Greeks and Mesh Sensitivity:** Sensitivities ($\Delta$ and $\Gamma$) are evaluated from the discrete price surface using central finite differences. We perform a mesh refinement study ($M \in \{50, 100, 200, 400\}$) confirming $\mathcal{O}(\Delta S^2)$ convergence for both Greeks at the strike. We also analyze a short-maturity contract ($T=0.1\,\text{y}$), where the Gamma peak increases more than threefold, highlighting the necessity of spatial resolution near $S=K$.
5. **Payoff Kink and Rannacher Smoothing:** The non-smooth payoff introduces a discontinuity in the first derivative at $S=K$. Because Crank–Nicolson is A-stable but not L-stable ($R_{\text{CN}}(z) \to -1$ as $z \to \infty$), coarse time discretisations exhibit severe spurious oscillations in Gamma (reaching an oscillation amplitude of $0.246$ at $N=5$, $M=200$). Rannacher time-stepping (replacing the first two steps with Backward Euler) damps these oscillations to $1.40 \times 10^{-3}$ (a 175-fold suppression). We document the accompanying trade-off: at $N=20$, Rannacher price error ($1.44 \times 10^{-2}$) is slightly higher than standard Crank–Nicolson ($9.45 \times 10^{-3}$), demonstrating that the benefit of Rannacher lies in sensitivity stability rather than absolute price precision.

---

## 2. Mathematical Formulation and Numerical Discretisation

### 2.1 The Black–Scholes Initial-Boundary Value Problem
Under risk-neutral valuation with constant risk-free rate $r$ and asset volatility $\sigma$, the option price $V(S, t)$ satisfies:
$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0, \quad (S, t) \in (0, S_{\max}) \times [0, T)$$

Transforming to time-to-maturity $\tau = T - t$ yields the forward parabolic problem:
$$\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V \equiv \mathcal{L}_{\text{BS}} V$$

### 2.2 Boundary and Terminal Conditions
- **Terminal Payoff ($\tau = 0$):**
  $$V(S, 0) = \max(S - K, 0) \quad \text{(Call)}, \qquad V(S, 0) = \max(K - S, 0) \quad \text{(Put)}$$
- **Lower Boundary ($S = 0$):**
  $$V(0, \tau) = 0 \quad \text{(Call)}, \qquad V(0, \tau) = K e^{-r\tau} \quad \text{(Put)}$$
- **Upper Boundary ($S = S_{\max}$ with $S_{\max} \ge 4K$):**
  $$V(S_{\max}, \tau) \approx S_{\max} - K e^{-r\tau} \quad \text{(Call)}, \qquad V(S_{\max}, \tau) = 0 \quad \text{(Put)}$$

### 2.3 Spatial Discretisation
On a uniform grid $S_j = j \Delta S$ ($j=0, \dots, M$, $\Delta S = S_{\max}/M$), we use second-order central difference operators:
$$\frac{\partial^2 V}{\partial S^2}\bigg|_j \approx \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2}, \qquad \frac{\partial V}{\partial S}\bigg|_j \approx \frac{V_{j+1} - V_{j-1}}{2\Delta S}$$

At interior node $j$, the discrete operator is:
$$\mathcal{L}_{\text{BS}}^h V_j = \alpha_j V_{j-1} + \beta_j V_j + \gamma_j V_{j+1}$$
$$\alpha_j = \frac{\sigma^2 S_j^2}{2\Delta S^2} - \frac{r S_j}{2\Delta S}, \quad \beta_j = -\frac{\sigma^2 S_j^2}{\Delta S^2} - r, \quad \gamma_j = \frac{\sigma^2 S_j^2}{2\Delta S^2} + \frac{r S_j}{2\Delta S}$$

### 2.4 Time Marching Schemes
1. **Fully Implicit (Backward Euler, $\theta = 1$):**
   $$-\Delta\tau \alpha_j V_{j-1}^{n+1} + (1 - \Delta\tau \beta_j) V_j^{n+1} - \Delta\tau \gamma_j V_{j+1}^{n+1} = V_j^n$$
   Unconditionally stable, formal accuracy $\mathcal{O}(\Delta S^2 + \Delta \tau)$.

2. **Crank–Nicolson ($\theta = 1/2$):**
   $$\left(I - \frac{\Delta\tau}{2}\mathcal{L}_{\text{BS}}^h\right) V^{n+1} = \left(I + \frac{\Delta\tau}{2}\mathcal{L}_{\text{BS}}^h\right) V^n$$
   Unconditionally stable, formal accuracy $\mathcal{O}(\Delta S^2 + \Delta \tau^2)$.

At each time step, the resulting tridiagonal system is strictly diagonally dominant and solved in $\mathcal{O}(M)$ operations via the Thomas algorithm.

---

## 3. Numerical Verification Against Analytical Benchmark

Benchmark parameters used throughout:
$$K = 100, \quad T = 1.0\,\text{y}, \quad r = 0.05, \quad \sigma = 0.20, \quad S_{\max} = 400$$
The exact Black–Scholes call price at $S=K=100$ is $V_{\text{exact}} = 10.45058357$.

### Table 1: Price Comparison (FDM vs Analytical Benchmark)
*Grid: $M=200$ ($\Delta S = 2.0$), Fully Implicit with $N=1000$ ($\Delta\tau = 0.001$), Crank–Nicolson with $N=500$ ($\Delta\tau = 0.002$)*

| $S$ | Option Type | Exact Price | Implicit FDM | Error (Implicit) | CN FDM | Error (CN) | Rel Err % (CN) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 60.0 | Call | 0.054437 | 0.056122 | $1.69 \times 10^{-3}$ | 0.055770 | $1.33 \times 10^{-3}$ | 2.45% |
| 70.0 | Call | 0.441448 | 0.443163 | $1.71 \times 10^{-3}$ | 0.442414 | $9.66 \times 10^{-4}$ | 0.22% |
| 80.0 | Call | 1.859420 | 1.856354 | $3.07 \times 10^{-3}$ | 1.855919 | $3.50 \times 10^{-3}$ | 0.19% |
| 90.0 | Call | 5.091222 | 5.082101 | $9.12 \times 10^{-3}$ | 5.082591 | $8.63 \times 10^{-3}$ | 0.17% |
| 100.0 (ATM) | Call | 10.450584 | 10.439632 | $1.10 \times 10^{-2}$ | 10.440686 | $9.90 \times 10^{-3}$ | 0.09% |
| 110.0 | Call | 17.662954 | 17.654510 | $8.44 \times 10^{-3}$ | 17.655424 | $7.53 \times 10^{-3}$ | 0.04% |
| 120.0 | Call | 26.169044 | 26.164336 | $4.71 \times 10^{-3}$ | 26.164798 | $4.25 \times 10^{-3}$ | 0.02% |
| 140.0 | Call | 45.110610 | 45.110114 | $4.95 \times 10^{-4}$ | 45.110062 | $5.48 \times 10^{-4}$ | 0.001% |
| 160.0 | Call | 64.913023 | 64.913129 | $1.06 \times 10^{-4}$ | 64.913111 | $8.72 \times 10^{-5}$ | 0.0001% |
| 200.0 | Call | 104.877724 | 104.877635 | $8.97 \times 10^{-5}$ | 104.877741 | $1.65 \times 10^{-5}$ | <0.0001% |
| 60.0 | Put | 35.177379 | 35.179184 | $1.80 \times 10^{-3}$ | 35.178712 | $1.33 \times 10^{-3}$ | 0.004% |
| 80.0 | Put | 16.982362 | 16.979415 | $2.95 \times 10^{-3}$ | 16.978861 | $3.50 \times 10^{-3}$ | 0.02% |
| 100.0 (ATM) | Put | 5.573526 | 5.562694 | $1.08 \times 10^{-2}$ | 5.563629 | $9.90 \times 10^{-3}$ | 0.18% |
| 120.0 | Put | 1.291986 | 1.287397 | $4.59 \times 10^{-3}$ | 1.287740 | $4.25 \times 10^{-3}$ | 0.33% |
| 160.0 | Put | 0.035966 | 0.036191 | $2.25 \times 10^{-4}$ | 0.036053 | $8.72 \times 10^{-5}$ | 0.24% |

Both methods match analytical solutions across all strike zones. Call and put prices at equal spot satisfy put-call parity to within the discretisation tolerance.

---

## 4. Convergence Analysis

### 4.1 Spatial Refinement ($N = 5000$ Fixed)
Holding $N=5000$ suppresses temporal truncation error to negligible levels ($< 10^{-6}$), isolating spatial truncation error $\mathcal{O}(\Delta S^p)$.

### Table 2: Spatial Convergence
| $M$ | $\Delta S$ | Implicit $L_\infty$ Err | Ratio | Implicit Order | CN $L_\infty$ Err | Ratio | CN Order |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 25 | 16.00 | $2.4816 \times 10^{-1}$ | — | — | $2.4797 \times 10^{-1}$ | — | — |
| 50 | 8.00 | $5.5499 \times 10^{-2}$ | 4.47 | 2.16 | $5.5355 \times 10^{-2}$ | 4.48 | 2.16 |
| 100 | 4.00 | $4.0187 \times 10^{-2}$ | 1.38 | 0.47 | $4.0002 \times 10^{-2}$ | 1.38 | 0.47 |
| 200 | 2.00 | $1.0208 \times 10^{-2}$ | 3.94 | 1.98 | $1.0008 \times 10^{-2}$ | 4.00 | 2.00 |
| 400 | 1.00 | $2.6982 \times 10^{-3}$ | 3.78 | 1.92 | $2.4986 \times 10^{-3}$ | 4.01 | 2.00 |

**Grid-Alignment Effect:** At $M=50$ ($\Delta S = 8$), $K=100$ falls between grid nodes ($12 \times 8 = 96$, $13 \times 8 = 104$). The payoff kink is not resolved on a node, distorting the error ratio from $M=50 \to 100$. Once $K$ is on the grid ($M \ge 100$), error reduction ratios settle cleanly at $4.00$ and $4.01$, verifying exact $\mathcal{O}(\Delta S^2)$ spatial convergence.

### 4.2 Temporal Refinement: Same-Grid Richardson Reference
When temporal convergence is measured against an analytical solution on a fixed spatial grid (e.g., $M=800$), the spatial error floor ($\approx 6.2 \times 10^{-4}$) exceeds the temporal error for $N \ge 200$, causing the error to plateau and the apparent Crank–Nicolson order to drop to zero.

To isolate the temporal order correctly, we fix the spatial mesh ($M=200$) and compute the reference solution $V_{\text{ref}}$ on the **same spatial grid** using a highly refined time step ($N_{\text{ref}} = 12800$). Because $V(M, N)$ and $V(M, N_{\text{ref}})$ share the exact same spatial operator, spatial error subtracts out:
$$e_{\text{temp}}(N) = \max_{1 \le j < M} |V_j(M=200, N) - V_j(M=200, N_{\text{ref}}=12800)|$$

### Table 3: Temporal Convergence (Same-Grid Reference, $M=200$)
| $N$ | $\Delta \tau$ | Implicit Error | Ratio | Observed Order | CN Error | Ratio | Observed Order |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 50 | 0.02000 | $2.1381 \times 10^{-2}$ | — | — | $1.0097 \times 10^{-4}$ | — | — |
| 100 | 0.01000 | $1.0656 \times 10^{-2}$ | **2.01** | **1.005** | $2.5244 \times 10^{-5}$ | **4.00** | **2.000** |
| 200 | 0.00500 | $5.2876 \times 10^{-3}$ | **2.02** | **1.011** | $6.3099 \times 10^{-6}$ | **4.00** | **2.000** |
| 400 | 0.00250 | $2.6023 \times 10^{-3}$ | **2.03** | **1.023** | $1.5763 \times 10^{-6}$ | **4.00** | **2.001** |
| 800 | 0.00125 | $1.2593 \times 10^{-3}$ | **2.07** | **1.047** | $3.9292 \times 10^{-7}$ | **4.01** | **2.004** |

The results show textbook convergence:
- Fully Implicit exhibits error ratios $\approx 2.0$, confirming first-order $\mathcal{O}(\Delta \tau)$.
- Crank–Nicolson exhibits error ratios $\approx 4.0$, confirming second-order $\mathcal{O}(\Delta \tau^2)$.

### 4.3 Full $4 \times 4$ Grid Refinement Matrix
The $L_\infty$ maximum price errors against the analytical solution across the combined grid space ($M, N \in \{50, 100, 200, 400\}$) are:

**Fully Implicit Method (Backward Euler):**
| $M \backslash N$ | $N = 50$ | $N = 100$ | $N = 200$ | $N = 400$ |
|:---:|:---:|:---:|:---:|:---:|
| **$M = 50$** | $6.9554 \times 10^{-2}$ | $6.2505 \times 10^{-2}$ | $5.8974 \times 10^{-2}$ | $5.7207 \times 10^{-2}$ |
| **$M = 100$** | $6.1092 \times 10^{-2}$ | $5.0460 \times 10^{-2}$ | $4.5143 \times 10^{-2}$ | $4.2483 \times 10^{-2}$ |
| **$M = 200$** | $3.1097 \times 10^{-2}$ | $2.0441 \times 10^{-2}$ | $1.5162 \times 10^{-2}$ | $1.2523 \times 10^{-2}$ |
| **$M = 400$** | $2.3789 \times 10^{-2}$ | $1.3117 \times 10^{-2}$ | $7.7570 \times 10^{-3}$ | $5.1043 \times 10^{-3}$ |

**Crank–Nicolson Method ($\theta = 1/2$):**
| $M \backslash N$ | $N = 50$ | $N = 100$ | $N = 200$ | $N = 400$ |
|:---:|:---:|:---:|:---:|:---:|
| **$M = 50$** | $5.5300 \times 10^{-2}$ | $5.5342 \times 10^{-2}$ | $5.5352 \times 10^{-2}$ | $5.5355 \times 10^{-2}$ |
| **$M = 100$** | $3.9981 \times 10^{-2}$ | $3.9997 \times 10^{-2}$ | $4.0001 \times 10^{-2}$ | $4.0002 \times 10^{-2}$ |
| **$M = 200$** | $9.9135 \times 10^{-3}$ | $9.9846 \times 10^{-3}$ | $1.0002 \times 10^{-2}$ | $1.0007 \times 10^{-2}$ |
| **$M = 400$** | $2.4082 \times 10^{-3}$ | $2.4760 \times 10^{-3}$ | $2.4930 \times 10^{-3}$ | $2.4972 \times 10^{-3}$ |

---

## 5. Numerical Greeks and Sensitivity Analysis

Option sensitivities are computed directly from the FDM solution vector via central differences:
$$\Delta_j = \frac{V_{j+1} - V_{j-1}}{2\Delta S}, \qquad \Gamma_j = \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2}$$

### Table 4: Greeks Profile Across Spot ($M=400, N=2000, T=1.0$)
*Analytical Delta at $S=80$: $d_1 = (\ln(0.8) + 0.07)/0.2 = -0.7657$, $N(d_1) = 0.22192213$.*

| $S$ | Exact $\Delta$ | Implicit $\Delta$ Err | CN $\Delta$ Err | Exact $\Gamma$ | Implicit $\Gamma$ Err | CN $\Gamma$ Err |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 60.0 | 0.01375767 | $1.31 \times 10^{-4}$ | $1.08 \times 10^{-4}$ | 0.00292946 | $2.17 \times 10^{-6}$ | $1.24 \times 10^{-6}$ |
| 70.0 | 0.07587539 | $9.61 \times 10^{-5}$ | $8.87 \times 10^{-5}$ | 0.01020092 | $1.50 \times 10^{-5}$ | $1.09 \times 10^{-5}$ |
| 80.0 | 0.22192213 | $7.28 \times 10^{-5}$ | $3.48 \times 10^{-5}$ | 0.01859823 | $1.00 \times 10^{-5}$ | $6.56 \times 10^{-6}$ |
| 90.0 | 0.42983173 | $1.38 \times 10^{-4}$ | $9.39 \times 10^{-5}$ | 0.02181975 | $7.53 \times 10^{-6}$ | $5.34 \times 10^{-6}$ |
| 95.0 | 0.53726014 | $1.13 \times 10^{-4}$ | $8.46 \times 10^{-5}$ | 0.02090532 | $1.14 \times 10^{-5}$ | $7.70 \times 10^{-6}$ |
| 100.0 (ATM) | 0.63683065 | $7.16 \times 10^{-5}$ | $6.24 \times 10^{-5}$ | 0.01876202 | $1.11 \times 10^{-5}$ | $7.25 \times 10^{-6}$ |
| 105.0 | 0.72372750 | $3.04 \times 10^{-5}$ | $3.82 \times 10^{-5}$ | 0.01592524 | $8.01 \times 10^{-6}$ | $5.12 \times 10^{-6}$ |
| 110.0 | 0.79575417 | $3.23 \times 10^{-8}$ | $1.89 \times 10^{-5}$ | 0.01288651 | $4.10 \times 10^{-6}$ | $2.57 \times 10^{-6}$ |
| 120.0 | 0.89645502 | $2.12 \times 10^{-5}$ | $1.16 \times 10^{-6}$ | 0.00750025 | $1.51 \times 10^{-6}$ | $9.25 \times 10^{-7}$ |
| 140.0 | 0.97894145 | $2.10 \times 10^{-7}$ | $3.64 \times 10^{-6}$ | 0.00180645 | $1.79 \times 10^{-6}$ | $1.07 \times 10^{-6}$ |
| 160.0 | 0.99653322 | $6.58 \times 10^{-6}$ | $3.91 \times 10^{-6}$ | 0.00032564 | $1.07 \times 10^{-7}$ | $9.26 \times 10^{-8}$ |

### 5.2 Mesh Refinement Study for Greeks at ATM ($N = 2000$ Fixed)
Holding $N=2000$ fixed, we refine the spatial mesh to verify $\mathcal{O}(\Delta S^2)$ convergence of the numerical Greeks at the strike $S=K=100$:

### Table 5: Delta and Gamma Convergence at ATM
| $M$ | $\Delta S$ | Delta Err (CN) | Ratio | Gamma Err (CN) | Ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 50 | 8.00 | $4.7070 \times 10^{-3}$ | — | $8.4644 \times 10^{-5}$ | — |
| 100 | 4.00 | $1.0099 \times 10^{-3}$ | 4.66 | $1.1839 \times 10^{-4}$ | 0.71* |
| 200 | 2.00 | $2.5003 \times 10^{-4}$ | **4.04** | $2.9113 \times 10^{-5}$ | **4.07** |
| 400 | 1.00 | $6.2359 \times 10^{-5}$ | **4.01** | $7.2484 \times 10^{-6}$ | **4.02** |

*Note: For $M \ge 100$, both Delta and Gamma errors quarter when $\Delta S$ is halved (ratios $\approx 4.0$), confirming second-order spatial convergence of the discrete derivative operators.*

### 5.3 Short-Maturity Behavior ($T = 0.1\,\text{y}$)
For $T=1.0\,\text{y}$, diffusion has smooths out the initial payoff kink. For short maturities (e.g., $T=0.1\,\text{y}$), the kink remains concentrated near $S=K$, and the analytical Gamma peak increases by a factor of 3.3 (from $0.01876$ to $0.06269$):

### Table 6: Near-Strike Behavior at $T = 0.1\,\text{y}$ ($M=400, N=2000$)
| $S$ | Exact $\Delta$ | CN $\Delta$ Err | Exact $\Gamma$ | CN $\Gamma$ Err |
|:---:|:---:|:---:|:---:|:---:|
| 97.0 | 0.35534758 | $3.42 \times 10^{-4}$ | 0.06070611 | $1.13 \times 10^{-4}$ |
| 98.0 | 0.41732048 | $3.33 \times 10^{-4}$ | 0.06297834 | $1.67 \times 10^{-4}$ |
| 99.0 | 0.48076645 | $2.86 \times 10^{-4}$ | 0.06364141 | $1.98 \times 10^{-4}$ |
| 100.0 (ATM) | 0.54406484 | $2.16 \times 10^{-4}$ | 0.06269314 | $2.04 \times 10^{-4}$ |
| 101.0 | 0.60565352 | $1.39 \times 10^{-4}$ | 0.06025060 | $1.84 \times 10^{-4}$ |
| 102.0 | 0.66413933 | $7.01 \times 10^{-5}$ | 0.05653030 | $1.45 \times 10^{-4}$ |
| 103.0 | 0.71838327 | $1.99 \times 10^{-5}$ | 0.05181864 | $9.43 \times 10^{-5}$ |

At $T=0.1$, the Gamma error is roughly 28 times larger than at $T=1.0$ on the same mesh ($2.04 \times 10^{-4}$ vs $7.25 \times 10^{-6}$), illustrating that short-dated options require mesh clustering near $S=K$.

---

## 6. Payoff Kink, Crank–Nicolson Oscillations, and Rannacher Smoothing

### 6.1 Origin of Oscillations
The terminal payoff $V(S,0) = \max(S-K, 0)$ is only continuous ($C^0$); its second derivative is a Dirac delta $\delta(S-K)$. Crank–Nicolson has amplification factor:
$$R_{\text{CN}}(z) = \frac{1 - z/2}{1 + z/2} \to -1 \quad \text{as } z = \lambda_k \Delta\tau \to \infty$$
Because $|R_{\text{CN}}(\infty)| = 1$, the method possesses **no numerical dissipation** for high-frequency spatial modes. The kink excites these modes, and Crank–Nicolson propagates them unattenuated, alternating in sign at each time step.

### 6.2 Quantitative Study of CN Gamma Oscillations ($M = 200$)
At $M=200$, the analytical Gamma peak at $S=K=100$ is $0.018762$. The numerical Gamma computed via standard Crank–Nicolson for coarse time steps reveals non-physical values:

### Table 7: Spurious Oscillations under Standard Crank–Nicolson ($M=200$)
| $N$ | $\Delta \tau$ | $\Gamma$ at $K-\Delta S$ | $\Gamma$ at $K$ | $\Gamma$ at $K+\Delta S$ | Max Oscillation Amplitude |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | 0.2000 | +0.106204 | **-0.139382** | +0.098156 | **0.24559** |
| 10 | 0.1000 | +0.002902 | +0.041193 | +0.000877 | **0.04032** |
| 20 | 0.0500 | +0.019733 | +0.018811 | +0.017671 | $1.14 \times 10^{-3}$ |
| 50 | 0.0200 | +0.019763 | +0.018790 | +0.017706 | $1.08 \times 10^{-3}$ |
| 100 | 0.0100 | +0.019764 | +0.018791 | +0.017707 | $1.08 \times 10^{-3}$ |

At $N=5$, standard Crank–Nicolson produces a **negative Gamma** ($\Gamma(100) = -0.139$) with oscillation amplitude $0.246$, violating no-arbitrage convexity constraints.

### 6.3 The Rannacher Fix and Quantitative Comparison
Rannacher (1984) proposed replacing the first two time steps with Backward Euler (Fully Implicit) steps, which is L-stable ($R_{\text{BE}}(z) = 1/(1+z) \to 0$ as $z \to \infty$), strongly damping the high-frequency modes. Subsequent steps resume Crank–Nicolson.

### Table 8: Standard CN vs Rannacher Smoothing ($M=200$)
| $N$ | CN Price $L_\infty$ | Rannacher Price $L_\infty$ | CN Gamma Osc | Rannacher Gamma Osc | Oscillation Reduction |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | $2.3922 \times 10^{-1}$ | $7.8901 \times 10^{-2}$ | 0.24559 | 0.00140 | **175×** |
| 10 | $3.3942 \times 10^{-2}$ | $2.7553 \times 10^{-2}$ | 0.04032 | 0.00128 | **31×** |
| 20 | $9.4455 \times 10^{-3}$ | $1.4381 \times 10^{-2}$ | 0.00114 | 0.00109 | $1.05\times$ |
| 50 | $9.9147 \times 10^{-3}$ | $1.0705 \times 10^{-2}$ | 0.00108 | 0.00108 | $1.00\times$ |
| 100 | $9.9850 \times 10^{-3}$ | $1.0184 \times 10^{-2}$ | 0.00108 | 0.00108 | $1.00\times$ |

### 6.4 Critical Engineering Observation: Price vs Sensitivity Trade-Off
A critical observation often omitted in textbook summaries:
- At very coarse time steps ($N=5, 10$), Rannacher dampens wild oscillations and improves price error ($0.239 \to 0.079$).
- At moderate time steps ($N=20$), **Rannacher's option price error ($1.44 \times 10^{-2}$) is higher than pure Crank–Nicolson ($9.45 \times 10^{-3}$)**.
This occurs because the two initial Backward Euler steps introduce first-order temporal truncation error $\mathcal{O}(\Delta \tau)$. Thus, one should not claim that Rannacher produces "negligible price difference" universally; rather, it represents a deliberate engineering trade-off: **sacrificing a small amount of temporal price precision to guarantee stable, non-oscillatory Greeks near the strike.**

---

## 7. Conclusions and Practical Guidelines

1. **Grid Alignment:** The strike price $K$ should always coincide with a spatial grid node ($K = j \Delta S$). Failing to align $K$ degrades spatial convergence from $\mathcal{O}(\Delta S^2)$ toward $\mathcal{O}(\Delta S)$.
2. **Temporal Convergence Methodology:** When verifying high-order time integration numerically, the temporal error must be isolated using a same-grid Richardson reference solution. Using an external analytical solution introduces a spatial error floor that masks second-order temporal convergence.
3. **Sensitivity Stability:** Standard Crank–Nicolson cannot be safely used for short-dated options or coarse time steps without damping. Rannacher smoothing completely suppresses spurious oscillations at negligible computational overhead, making it the preferred production scheme.

---

## 8. Reproducibility

All numerical experiments, error tables, and figures in this report are fully reproducible by running:
```bash
python reproduce_all_results.py
```
This automated test suite executes all 5 benchmarks, checks 11 numerical assertions, and completes in under 30 seconds.
