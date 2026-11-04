# Derivation 6: Convergence Analysis of FDM Schemes

## Overview

This document derives and explains the spatial and temporal convergence orders for the Fully Implicit and Crank-Nicolson finite-difference schemes applied to the Black-Scholes PDE.

The Black-Scholes PDE in forward time `tau = T - t`:

$$\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V$$

Let `L_BS` denote the spatial differential operator:

$$\mathcal{L}_{BS} V = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V$$

---

## 1. Spatial Discretisation (Both Methods)

On a uniform grid `S_j = j * dS`, `j = 0, 1, ..., M`, the spatial operator is approximated by:

$$\frac{\partial^2 V}{\partial S^2}\bigg|_j \approx \frac{V_{j+1} - 2V_j + V_{j-1}}{dS^2} + O(dS^2)$$

$$\frac{\partial V}{\partial S}\bigg|_j \approx \frac{V_{j+1} - V_{j-1}}{2\,dS} + O(dS^2)$$

Both approximations use **central differences** and carry a truncation error of `O(dS^2)`.

The discrete operator coefficients are:

$$\alpha_j = \frac{\sigma^2 S_j^2}{2\,dS^2} - \frac{r S_j}{2\,dS}, \quad \beta_j = -\frac{\sigma^2 S_j^2}{dS^2} - r, \quad \gamma_j = \frac{\sigma^2 S_j^2}{2\,dS^2} + \frac{r S_j}{2\,dS}$$

---

## 2. Fully Implicit Method (Backward Euler)

### Temporal Discretisation

The Backward Euler (fully implicit) scheme discretises time as:

$$\frac{V_j^{n+1} - V_j^n}{\Delta\tau} = \mathcal{L}^h_{BS} V_j^{n+1}$$

The **temporal truncation error** of Backward Euler is `O(dt)`, since:

$$\frac{V(\tau + \Delta\tau) - V(\tau)}{\Delta\tau} = \frac{\partial V}{\partial \tau}\bigg|_{\tau+\Delta\tau} - \frac{\Delta\tau}{2}\frac{\partial^2 V}{\partial \tau^2}\bigg|_{\tau+\Delta\tau} + O(\Delta\tau^2)$$

### Overall Accuracy

| Source | Order |
|:---|:---:|
| Spatial (central diff in S) | `O(dS^2)` |
| Temporal (backward Euler) | `O(dt)` |
| **Combined** | **`O(dS^2 + dt)`** |

### Stability

The fully implicit method is **unconditionally stable** for all `dS > 0`, `dt > 0`. The LHS matrix `(I - dt * L^h)` is strictly diagonally dominant, with all eigenvalues lying in the left half-plane. The spectral radius of the amplification factor satisfies `|g(xi)| <= 1` for all wavenumbers `xi`.


### Observed spatial convergence (from `results/output_8_convergence_study.txt`, section B)

Grids on which K is a node (M a multiple of 4).  Fully implicit (time error removed by Richardson extrapolation):

| M | dS | L-inf error | order | L2 error | error at S=K | order at S=K |
|--:|---:|---:|---:|---:|---:|---:|
| 100 | 4.00 | 4.000e-02 | — | 2.068e-01 | 3.982e-02 | — |
| 200 | 2.00 | 1.001e-02 | 2.00 | 5.157e-02 | 9.898e-03 | 2.01 |
| 400 | 1.00 | 2.499e-03 | 2.00 | 1.288e-02 | 2.471e-03 | 2.00 |
| 800 | 0.50 | 6.246e-04 | 2.00 | 3.221e-03 | 6.176e-04 | 2.00 |

Crank-Nicolson (N = 2000):

| M | dS | L-inf error | order | L2 error | error at S=K | order at S=K |
|--:|---:|---:|---:|---:|---:|---:|
| 100 | 4.00 | 4.000e-02 | — | 2.068e-01 | 3.982e-02 | — |
| 200 | 2.00 | 1.001e-02 | 2.00 | 5.157e-02 | 9.898e-03 | 2.01 |
| 400 | 1.00 | 2.499e-03 | 2.00 | 1.288e-02 | 2.471e-03 | 2.00 |
| 800 | 0.50 | 6.245e-04 | 2.00 | 3.220e-03 | 6.175e-04 | 2.00 |

Both schemes give p = 2.00.  (The pair M = 50 -> 100 is not a refinement within one alignment family -- K is not a node for M = 50 -- and is therefore excluded from the order estimate; see Derivation 8, section A.)

### Observed temporal convergence, fully implicit (M = 200, reference: CN with N = 12800 on the same grid)

| N | dt | temporal error (L-inf) | order | L2 |
|--:|---:|---:|---:|---:|
| 25 | 0.04000 | 4.287e-02 | — | 2.344e-01 |
| 50 | 0.02000 | 2.147e-02 | 1.00 | 1.178e-01 |
| 100 | 0.01000 | 1.074e-02 | 1.00 | 5.905e-02 |
| 200 | 0.00500 | 5.372e-03 | 1.00 | 2.956e-02 |
| 400 | 0.00250 | 2.686e-03 | 1.00 | 1.479e-02 |
| 800 | 0.00125 | 1.343e-03 | 1.00 | 7.398e-03 |

The temporal error halves when dt is halved: **O(dt)**, observed order 1.00.

---

## 3. Crank-Nicolson Method (Theta = 1/2)

### Temporal Discretisation

The Crank-Nicolson scheme averages the spatial operator at two time levels:

$$\frac{V_j^{n+1} - V_j^n}{\Delta\tau} = \frac{1}{2}\mathcal{L}^h_{BS} V_j^{n+1} + \frac{1}{2}\mathcal{L}^h_{BS} V_j^n$$

This is equivalent to a **central difference in time** at the midpoint `tau_{n+1/2}`. The temporal truncation error is `O(dt^2)`:

$$\frac{V(\tau + \Delta\tau) - V(\tau)}{\Delta\tau} = \frac{\partial V}{\partial \tau}\bigg|_{\tau+\Delta\tau/2} + \frac{\Delta\tau^2}{24}\frac{\partial^3 V}{\partial \tau^3} + O(\Delta\tau^4)$$

### Overall Accuracy

| Source | Order |
|:---|:---:|
| Spatial (central diff in S) | `O(dS^2)` |
| Temporal (Crank-Nicolson) | `O(dt^2)` |
| **Combined** | **`O(dS^2 + dt^2)`** |

### Stability

CN is also **unconditionally stable** (von Neumann analysis shows the amplification factor has modulus exactly 1 for the heat equation, i.e., it is non-dissipative).

### Observed temporal convergence, Crank-Nicolson (M = 200, same reference)

| N | dt | temporal error (L-inf) | order | L2 |
|--:|---:|---:|---:|---:|
| 25 | 0.04000 | 4.035e-04 | — | 2.104e-03 |
| 50 | 0.02000 | 1.010e-04 | 2.00 | 5.258e-04 |
| 100 | 0.01000 | 2.524e-05 | 2.00 | 1.314e-04 |
| 200 | 0.00500 | 6.310e-06 | 2.00 | 3.286e-05 |
| 400 | 0.00250 | 1.576e-06 | 2.00 | 8.208e-06 |
| 800 | 0.00125 | 3.929e-07 | 2.00 | 2.046e-06 |

The temporal error drops by a factor 4 when dt is halved: **O(dt^2)**, observed order 2.00.  The same orders are obtained for M = 100 (results file, section C), i.e. they do not depend on the spatial grid.

---

## 4. Comparing the Two Methods

### How the two error sources are separated (method used in this project)

Write the total error of a scheme on grid (dS, dt) as

    E(dS, dt) ~ C_S dS^2 + C_T dt^p ,    p = 1 (implicit),  p = 2 (CN).

* **Spatial error alone.**  Make the time error negligible and vary dS.
  For CN take N = 2000 (time error ~1e-7).  For the implicit scheme the O(dt)
  term is removed by Richardson extrapolation in time, V* = 2 V(N) - V(N/2),
  which is O(dt^2) accurate.  Both schemes share the same spatial operator, so
  they must give the same spatial errors; the tables below agree to 4 digits.
* **Temporal error alone.**  Do NOT compare with the exact price: the spatial
  term C_S dS^2 does not go away when only dt is refined and forms an error
  *floor*.  Instead compare with a time-converged numerical solution on the
  *same* spatial grid (CN with N_ref = 12800).  Both solutions contain the same
  spatial error, which cancels exactly in the difference, leaving C_T dt^p.
* **Why this matters.**  A study that fixes M=800 and compares with the exact
  price shows the implicit order drifting from 1.0 to 0.76 and the CN error
  stalling at ~6e-4 -- the spatial error of that grid (see the grey curve in
  figures/fig2_temporal_convergence.png).  That is a flaw of the experiment, not
  of the scheme.

### Equal cost, M = 200

Temporal errors at N = 100: implicit 1.07e-02, CN 2.52e-05 -- CN is about 425 times more accurate in time at the same dt, and CN with N = 25 (4.0e-04) beats implicit with N = 800 (1.3e-03).  Against the *exact* price both schemes saturate at the spatial error of the grid, so on a fixed grid the benefit of CN is that the same total error is reached with far fewer time steps.

### The 4x4 grid-refinement matrix

`results/output_11_grid_refinement_matrix.txt` lists the L-inf error against the exact price for M, N in {50,100,200,400}.  Doubling M and N together mixes both error sources; for the implicit scheme the O(dt) term dominates, so the combined order is between 1 and 2 and is not a clean measure of either order.  Sections 2-3 above are the proper measurement.

---

## 5. Summary

| Property | Fully Implicit | Crank-Nicolson |
|:---|:---:|:---:|
| Spatial order | `O(dS^2)` | `O(dS^2)` |
| Temporal order | `O(dt)` | `O(dt^2)` |
| Stability | Unconditional | Unconditional |
| Dissipation | Yes (L-stable) | None |
| Recommended for | Robustness with coarse dt | Accuracy with moderate dt |
