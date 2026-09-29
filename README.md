# Numerical Solution of the Black-Scholes Partial Differential Equation

### Computational Finance Research Project
**Author:** Akshat Kumar  
**Supervisor:** Dr. Ashok Kumar Vaikuntam  
**Topic:** Numerical Solutions of Option Pricing Models using Finite Difference Methods (FDM)

---

## 1. Executive Summary & Project Context

In derivative pricing, the Black-Scholes-Merton model provides a continuous-time framework for valuing options under geometric Brownian motion. While European options under idealized conditions (constant interest rates and volatility) yield closed-form analytical solutions, financial engineering practice routinely confronts market features that break analytical tractability:
- Early exercise features (American and Bermudan options)
- Discrete dividend schedules and barrier conditions
- Volatility surfaces (local volatility, Dupire's equation, stochastic volatility)
- Multi-asset or path-dependent contracts

For these applications, numerical partial differential equation (PDE) methods are indispensable. This project focuses on the rigorous mathematical derivation, stability analysis, discretization, and algorithmic implementation of Finite Difference Methods (FDM) applied to the Black-Scholes PDE.

The repository provides modular, human-readable Python implementations allowing interactive testing with arbitrary financial parameters and grid resolutions, backed by analytical benchmarking and mathematical derivations.

---

## 2. Mathematical Formulation

### 2.1 The Black-Scholes PDE
Under risk-neutral valuation, the price of a European contingent claim $V(S, t)$ satisfies the second-order linear parabolic PDE:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$$

where:
- $S \in [0, \infty)$ is the underlying asset price
- $t \in [0, T]$ is current calendar time
- $\sigma > 0$ is the constant volatility of the underlying asset
- $r \ge 0$ is the continuously compounded risk-free interest rate
- $T$ is the contract expiry date

### 2.2 Time Reversal Transformation
Because the terminal payoff $V(S, T)$ is prescribed at maturity $T$, the original PDE is a backward-in-time problem. To transform it into a standard forward parabolic initial-value problem (IVP), we define the time-to-maturity variable:

$$\tau = T - t, \quad \tau \in [0, T]$$

Applying the chain rule $\frac{\partial V}{\partial t} = -\frac{\partial V}{\partial \tau}$, the PDE becomes:

$$\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V$$

### 2.3 Initial and Boundary Conditions
To establish a well-posed initial-boundary value problem (IBVP), the spatial domain is truncated to $[0, S_{\max}]$, where $S_{\max}$ is chosen sufficiently far from the strike ($S_{\max} \ge 3K \text{ to } 4K$) so that boundary errors at $S \approx K$ remain negligible.

#### Initial Condition ($\tau = 0$, Expiry):
- **European Call:** $V(S, 0) = \max(S - K, 0)$
- **European Put:** $V(S, 0) = \max(K - S, 0)$

#### Boundary Conditions for European Call:
- **At $S = 0$:** The asset value remains zero permanently:
  $$V(0, \tau) = 0$$
- **At $S = S_{\max}$:** As $S \to \infty$, the call option value behaves asymptotically like $S - K e^{-r\tau}$:
  $$V(S_{\max}, \tau) \approx S_{\max} - K e^{-r\tau}$$

#### Boundary Conditions for European Put:
- **At $S = 0$:** By discounting the strike:
  $$V(0, \tau) = K e^{-r\tau}$$
- **At $S = S_{\max}$:** Deep out-of-the-money put approaches zero:
  $$V(S_{\max}, \tau) \approx 0$$

---

## 3. Discretization & Grid Construction

### 3.1 Spatial and Temporal Grids
We discretize the continuous domain $[0, S_{\max}] \times [0, T]$ onto a uniform rectangular mesh:
- Spatial increments: $S_j = j \Delta S$ for $j = 0, 1, \dots, M$, with $\Delta S = \frac{S_{\max}}{M}$
- Temporal increments: $\tau_n = n \Delta \tau$ for $n = 0, 1, \dots, N$, with $\Delta \tau = \frac{T}{N}$
- Discrete approximation: $V_j^n \approx V(S_j, \tau_n)$

### 3.2 Finite Difference Stencils
Using standard Taylor series expansions, the spatial derivatives at grid node $j$ are approximated using second-order central difference operators:

$$\frac{\partial V}{\partial S} \approx \frac{V_{j+1}^n - V_{j-1}^n}{2 \Delta S} + \mathcal{O}(\Delta S^2)$$

$$\frac{\partial^2 V}{\partial S^2} \approx \frac{V_{j+1}^n - 2V_j^n + V_{j-1}^n}{\Delta S^2} + \mathcal{O}(\Delta S^2)$$

Substituting these finite difference approximations into the spatial differential operator yields:

$$\mathcal{L} V_j = \alpha_j V_{j-1} + \beta_j V_j + \gamma_j V_{j+1}$$

where the node-dependent coefficients are derived as:

$$\alpha_j = \frac{1}{2}\sigma^2 j^2 - \frac{1}{2} r j$$

$$\beta_j = -\sigma^2 j^2 - r$$

$$\gamma_j = \frac{1}{2}\sigma^2 j^2 + \frac{1}{2} r j$$

Notice that $\alpha_j + \beta_j + \gamma_j = -r \le 0$, which ensures the spatial operator adheres to discrete maximum principles when suitable time discretizations are applied.

---

## 4. Implemented Numerical Methods

### 4.1 Exact Analytical Benchmark (`programs/black_scholes.py`)
To validate any numerical scheme without reliance on fixed lookup tables, the analytical closed-form Black-Scholes formula is implemented from scratch:

$$d_1 = \frac{\ln(S/K) + \left(r + \frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

$$C(S, K, T, r, \sigma) = S \Phi(d_1) - K e^{-rT} \Phi(d_2)$$

$$P(S, K, T, r, \sigma) = K e^{-rT} \Phi(-d_2) - S \Phi(-d_1)$$

where $\Phi(\cdot)$ denotes the standard normal cumulative distribution function (evaluated analytically via Gauss error function `math.erf`).

#### Sensitivity Greek Calculations:
The analytical module also computes all primary risk sensitivities:
- **Delta ($\Delta$):** $\frac{\partial V}{\partial S} = \Phi(d_1)$ (Call) or $\Phi(d_1) - 1$ (Put)
- **Gamma ($\Gamma$):** $\frac{\partial^2 V}{\partial S^2} = \frac{\phi(d_1)}{S \sigma \sqrt{T}}$
- **Theta ($\Theta$):** $-\frac{\partial V}{\partial \tau} = -\frac{S \phi(d_1)\sigma}{2\sqrt{T}} - r K e^{-rT}\Phi(d_2)$
- **Vega ($\mathcal{V}$):** $\frac{\partial V}{\partial \sigma} = S \sqrt{T} \phi(d_1)$
- **Rho ($\rho$):** $\frac{\partial V}{\partial r} = K T e^{-rT} \Phi(d_2)$ (Call) or $-K T e^{-rT} \Phi(-d_2)$ (Put)

---

### 4.2 Fully Implicit Finite Difference Method (`programs/implicit_fdm.py`)
The Fully Implicit scheme (Backward Euler in time) evaluates the spatial derivatives at the advanced time level $\tau_{n+1}$:

$$\frac{V_j^{n+1} - V_j^n}{\Delta \tau} = \mathcal{L} V_j^{n+1}$$

Rearranging in terms of unknowns at time level $n+1$:

$$-a_j V_{j-1}^{n+1} + (1 - b_j) V_j^{n+1} - c_j V_{j+1}^{n+1} = V_j^n, \quad j = 1, \dots, M-1$$

where:
$$a_j = \Delta \tau \alpha_j, \quad b_j = \Delta \tau \beta_j, \quad c_j = \Delta \tau \gamma_j$$

#### Matrix Formulation:
In vector notation, this produces a tridiagonal linear system for the interior spatial nodes $\mathbf{V}^{n+1} = [V_1^{n+1}, \dots, V_{M-1}^{n+1}]^T$:

$$A \mathbf{V}^{n+1} = \mathbf{V}^n + \mathbf{b}^{n+1}$$

where $A$ is a strictly diagonally dominant tridiagonal matrix:
$$A = \begin{pmatrix}
1 - b_1 & -c_1 & 0 & \dots & 0 \\
-a_2 & 1 - b_2 & -c_2 & \dots & 0 \\
\vdots & \ddots & \ddots & \ddots & \vdots \\
0 & \dots & -a_{M-2} & 1 - b_{M-2} & -c_{M-2} \\
0 & \dots & 0 & -a_{M-1} & 1 - b_{M-1}
\end{pmatrix}$$

The vector $\mathbf{b}^{n+1}$ incorporates the known boundary conditions:
$$\mathbf{b}^{n+1} = \begin{bmatrix} a_1 V_0^{n+1} \\ 0 \\ \vdots \\ 0 \\ c_{M-1} V_M^{n+1} \end{bmatrix}$$

#### Theoretical Properties:
- **Order of Accuracy:** $\mathcal{O}(\Delta \tau + \Delta S^2)$ — first-order in time, second-order in space.
- **Stability:** Unconditionally $A$-stable and $L_\infty$-stable.
- **Non-oscillatory:** Fully implicit stepping preserves monotonicity, guaranteeing no spurious oscillations near the non-smooth payoff corner at $S = K$.

---

### 4.3 Crank–Nicolson Finite Difference Method (`programs/crank_nicolson.py`)
The Crank–Nicolson scheme uses a centered difference at the temporal half-step $\tau_{n + 1/2}$ ($\theta = 1/2$ trapezoidal rule), averaging spatial derivatives at levels $n$ and $n+1$:

$$\frac{V_j^{n+1} - V_j^n}{\Delta \tau} = \frac{1}{2} \left[ \mathcal{L} V_j^{n+1} + \mathcal{L} V_j^n \right]$$

Multiplying by $\Delta \tau$ and separating knowns (time level $n$) from unknowns (time level $n+1$):

$$-\frac{a_j}{2} V_{j-1}^{n+1} + \left(1 - \frac{b_j}{2}\right) V_j^{n+1} - \frac{c_j}{2} V_{j+1}^{n+1} = \frac{a_j}{2} V_{j-1}^n + \left(1 + \frac{b_j}{2}\right) V_j^n + \frac{c_j}{2} V_{j+1}^n$$

#### Matrix Formulation:
$$A_{\text{CN}} \mathbf{V}^{n+1} = B_{\text{CN}} \mathbf{V}^n + \mathbf{b}^{n + 1/2}$$

where:
- $A_{\text{CN}}$ has subdiagonal $-\frac{a_j}{2}$, main diagonal $1 - \frac{b_j}{2}$, and superdiagonal $-\frac{c_j}{2}$.
- $B_{\text{CN}}$ has subdiagonal $\frac{a_j}{2}$, main diagonal $1 + \frac{b_j}{2}$, and superdiagonal $\frac{c_j}{2}$.
- $\mathbf{b}^{n + 1/2}$ contains boundary condition contributions evaluated at both time levels $n$ and $n+1$.

#### Theoretical Properties & Non-Smooth Payoff Behavior:
- **Order of Accuracy:** $\mathcal{O}(\Delta \tau^2 + \Delta S^2)$ — second-order accurate in both space and time.
- **Von Neumann Stability:** Unconditionally stable in the $L_2$ norm for all step sizes $\Delta \tau, \Delta S > 0$.
- **Rannacher Observation:** The symbol of the Crank-Nicolson amplification factor is $R(z) = \frac{1 - z/2}{1 + z/2}$. For high-frequency spatial modes ($z \to \infty$), $R(z) \to -1$. Because the initial payoff function $\max(S - K, 0)$ has a kink (discontinuous first derivative) at $S = K$, high-frequency Fourier components are excited. If the time step $\Delta \tau$ is chosen too large without adequate grid refinement, Crank-Nicolson can generate decaying oscillations near the strike. This behavior is analyzed in detail in `derivations/05_crank_nicolson_method.md`.

---

### 4.4 The Tridiagonal Thomas Algorithm ($LU$ Decomposition)
Direct matrix inversion of an $M \times M$ matrix scales as $\mathcal{O}(M^3)$. Because matrices $A$ and $A_{\text{CN}}$ are strictly tridiagonal, the linear system at each time step is solved using the Thomas algorithm (a specialized Gaussian elimination without pivoting) in $\mathcal{O}(M)$ operations.

Given the system:
$$\alpha_j x_{j-1} + \beta_j x_j + \gamma_j x_{j+1} = d_j, \quad j = 1, \dots, m$$

#### 1. Forward Elimination (LU Factorization):
Compute modified coefficients $c'_j$ and $d'_j$:

$$c'_1 = \frac{\gamma_1}{\beta_1}, \quad d'_1 = \frac{d_1}{\beta_1}$$

$$c'_j = \frac{\gamma_j}{\beta_j - \alpha_j c'_{j-1}}, \quad d'_j = \frac{d_j - \alpha_j d'_{j-1}}{\beta_j - \alpha_j c'_{j-1}} \quad (j = 2, \dots, m)$$

#### 2. Backward Substitution:
$$x_m = d'_m$$
$$x_j = d'_j - c'_j x_{j+1} \quad (j = m-1, \dots, 1)$$

Total computational complexity per time step: $5M$ operations, yielding total run time of $\mathcal{O}(M \cdot N)$ across the full temporal integration.

---

## 5. Comparative Method Matrix

| Dimension | Analytical (Closed-Form) | Fully Implicit (Backward Euler) | Crank–Nicolson Scheme |
| :--- | :--- | :--- | :--- |
| **Script** | `programs/black_scholes.py` | `programs/implicit_fdm.py` | `programs/crank_nicolson.py` |
| **Accuracy** | Exact (Machine precision $\epsilon_{\text{mach}}$) | $\mathcal{O}(\Delta \tau + \Delta S^2)$ | $\mathcal{O}(\Delta \tau^2 + \Delta S^2)$ |
| **Stability** | N/A | Unconditionally $A$-stable & $L_\infty$-stable | Unconditionally $A$-stable ($L_2$ norm) |
| **Oscillations** | None | Strictly monotone (No oscillations) | Possible near $S = K$ if $\Delta \tau$ is too coarse |
| **Matrix Type** | None | Tridiagonal ($A \mathbf{V}^{n+1} = \mathbf{V}^n$) | Tridiagonal ($A \mathbf{V}^{n+1} = B \mathbf{V}^n$) |
| **Complexity** | $\mathcal{O}(1)$ | $\mathcal{O}(M \cdot N)$ via Thomas solver | $\mathcal{O}(M \cdot N)$ via Thomas solver |
| **Primary Utility** | Exact benchmark for standard contracts | High stability for non-smooth payoffs | High temporal precision for smooth regimes |

---

## 6. Interactive Testing Framework

To ensure that evaluation is not restricted to fixed, pre-computed tables, all programs operate interactively from the command line. Users (and evaluators) can supply arbitrary financial parameters and mesh densities.

### 6.1 Testing Flow:
1. **Interactive Prompt:** The user enters the option contract variables ($K, T, r, \sigma$) and grid dimensions ($S_{\max}, M, N$, option type).
2. **Dynamic Exact Comparison:** The numerical solvers automatically compute the exact Black-Scholes closed-form value for the exact spot price requested by the user.
3. **Accuracy Reporting:** The programs evaluate:
   - Interpolated FDM price at the specified spot $S$
   - Exact analytical closed-form price
   - Absolute pointwise error $|V_{\text{FDM}} - V_{\text{exact}}|$
   - Relative percentage error $\frac{|V_{\text{FDM}} - V_{\text{exact}}|}{V_{\text{exact}}} \times 100\%$
4. **Full Grid Inspection:** Users can choose whether to output a slice of values across the grid (from deep in-the-money to deep out-of-the-money) or inspect specific nodes.

---

## 7. Execution Guide

### 7.1 Prerequisites
The codebase uses standard Python 3 with NumPy.
```bash
pip install numpy
```

### 7.2 Running the Programs

#### A. Exact Black-Scholes Calculator & Greeks
```bash
python programs/black_scholes.py
```
**Prompts:** Spot $S$, Strike $K$, Expiry $T$, Rate $r$, Volatility $\sigma$.  
**Outputs:** Call price, Put price, and all 5 Greeks ($\Delta, \Gamma, \Theta, \mathcal{V}, \rho$).

#### B. Fully Implicit FDM Solver
```bash
python programs/implicit_fdm.py
```
**Prompts:** $K, T, r, \sigma, S_{\max}, M, N$, option type (`call` or `put`), and test spot price $S$.  
**Outputs:** Numerical price, exact analytical benchmark, absolute error, and optional grid table.

#### C. Crank–Nicolson FDM Solver
```bash
python programs/crank_nicolson.py
```
**Prompts:** $K, T, r, \sigma, S_{\max}, M, N$, option type (`call` or `put`), and test spot price $S$.  
**Outputs:** Second-order accurate numerical price, exact analytical benchmark, and absolute error comparison.

---

## 8. Theoretical Derivations Index

Comprehensive, equation-by-equation mathematical proofs and derivations are maintained in the [`derivations/`](./derivations) directory:

1. [`derivations/01_finite_difference_approximations.md`](./derivations/01_finite_difference_approximations.md): Taylor series expansions, derivation of forward, backward, and central difference approximations, and formal truncation error bounds.
2. [`derivations/02_black_scholes_pde_discretisation.md`](./derivations/02_black_scholes_pde_discretisation.md): Transformation from backward terminal problem to forward diffusion PDE; derivation of node coefficients $\alpha_j, \beta_j, \gamma_j$.
3. [`derivations/03_thomas_algorithm.md`](./derivations/03_thomas_algorithm.md): Step-by-step derivation of Gaussian elimination for tridiagonal systems ($LU$ decomposition), operation count, and diagonal dominance criteria.
4. [`derivations/04_fully_implicit_method.md`](./derivations/04_fully_implicit_method.md): Backward Euler discretization, matrix assembly, boundary treatment, and von Neumann stability proof.
5. [`derivations/05_crank_nicolson_method.md`](./derivations/05_crank_nicolson_method.md): Crank-Nicolson derivation, $\theta$-method representation, second-order convergence proofs, and analysis of Rannacher smoothing for non-smooth initial conditions.

---

## 9. Repository Structure

```text
numerical-black-scholes-fdm/
├── README.md                                # Comprehensive academic documentation
├── requirements.txt                        # Minimal dependencies (numpy)
├── derivations/                            # Step-by-step mathematical derivations
│   ├── 01_finite_difference_approximations.md
│   ├── 02_black_scholes_pde_discretisation.md
│   ├── 03_thomas_algorithm.md
│   ├── 04_fully_implicit_method.md
│   └── 05_crank_nicolson_method.md
└── programs/                               # Interactive Python implementations
    ├── black_scholes.py                    # Exact formula & all Greeks (live input)
    ├── crank_nicolson.py                   # O(dt^2, dS^2) CN solver with Thomas solver
    ├── implicit_fdm.py                     # Fully implicit backward Euler solver
    ├── finite_difference_derivatives.py    # FD numerical derivative verification
    ├── generate_grid.py                    # Mesh generation utility
    └── thomas_solver.py                    # Modular tridiagonal LU algorithm
```

---

## 10. References & Bibliography

1. **Daniel J. Duffy** (2022). *Numerical Methods in Computational Finance: A Partial Differential Equation (PDE/FDM) Approach*. Wiley Finance.
2. **Karel in 't Hout** (2017). *Numerical Partial Differential Equations in Finance Explained: An Introduction to Computational Finance*. Financial Engineering Explained, Palgrave Macmillan.
3. **John C. Hull** (2018). *Options, Futures, and Other Derivatives*. 10th Edition, Pearson.
4. **R. Rannacher** (1984). *Finite element solution of diffusion problems with irregular data*. Numerische Mathematik, 43(2), 309–327.
