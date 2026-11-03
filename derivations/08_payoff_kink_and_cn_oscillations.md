# Derivation 8: Non-Smooth Payoff, Crank-Nicolson Oscillations, and Rannacher Smoothing

## Overview

This document analyses:
1. The non-smooth payoff kink and its effect on numerical convergence
2. Spurious oscillations in the Crank-Nicolson scheme caused by the kink
3. The Rannacher time-stepping method as the standard fix

---

## 1. The Non-Smooth Payoff (Kink at S = K)

### Mathematical Description

The terminal payoff for a European call is:

$$V(S, \tau=0) = \max(S - K, 0) = \begin{cases} S - K & S > K \\ 0 & S \leq K \end{cases}$$

This function is:
- **Continuous** everywhere: `V(K,0) = 0` from both sides
- **C^0 but not C^1** at `S = K`: left derivative is 0, right derivative is 1
- **Distributional second derivative**: `V''(S, 0) = delta(S - K)`, a Dirac delta

In the theory of parabolic PDEs, the regularity of the solution `V(S, tau)` for `tau > 0` depends on the regularity of the initial data. The heat kernel smooths the kink:

$$V(S, \tau) = \int_0^\infty \Phi(S, S', \tau) \max(S' - K, 0) \, \frac{dS'}{S'}$$

For any `tau > 0`, `V` is infinitely smooth. But the FDM operates on a discrete grid, and the kink at `tau = 0` causes issues in the first few time steps.

---

## 2. Effect of the Kink on Spatial Convergence

### When K is NOT a Grid Node

Suppose `K = 100`, `S_max = 400`, `M = 50`. Then `dS = 8` and the grid nodes near `K` are at `S = 96` and `S = 104`. The payoff jumps from `max(96-100,0) = 0` to `max(104-100,0) = 4` over one cell, but the true kink lies between nodes.

The interpolation error at the kink is `O(dS)` rather than `O(dS^2)`. This degrades the **effective spatial order** of convergence to approximately `O(dS)` instead of the formal `O(dS^2)`.

### When K IS a Grid Node

If `M` is chosen so that `K / (S_max / M)` is an integer (i.e., `K = j * dS` for some integer `j`), the payoff kink falls exactly on a grid node. The piecewise-linear initial data is then represented exactly on the grid, and the full `O(dS^2)` convergence order is recovered.

**Practical rule:** Always choose `M` such that `M * K / S_max` is an integer. For `K=100, S_max=400`, this means `M` must be a multiple of 4.

---

## 3. Crank-Nicolson Oscillations

### Source of Oscillations

The Crank-Nicolson scheme can be written in terms of its amplification factor for a Fourier mode `e^{i*xi*j*dS}`:

$$g_{CN}(\xi) = \frac{1 + h \lambda(\xi)}{1 - h \lambda(\xi)}, \qquad h = \frac{dt}{2}$$

where `lambda(xi)` is the eigenvalue of the spatial operator (negative for stability). For the Black-Scholes PDE with a standard discretisation, `|g_{CN}| = 1` exactly for purely imaginary `lambda`. This means:

- **All Fourier modes are preserved without damping**
- High-frequency modes introduced by the payoff kink are propagated forward in time

When the kink is projected onto the discrete grid, it excites high-wavenumber components. Standard CN passes these through unchanged, creating visible oscillations in the price and especially in Gamma near `S = K`.

The oscillations are most visible when:
- `dt` is large (coarse time grid)
- `T` is small (the kink has less time to be smoothed by diffusion)
- Near the money (`S ≈ K`)

### Quantitative Example (from `payoff_kink_cn_oscillation.py`)

For `M=200, K=100, T=1`:

| N (steps) | dt | Gamma at K (analytical: ~0.0199) | Oscillation size |
|--:|---:|---:|---:|
| 5 | 0.200 | Large oscillation | ~O(1) |
| 10 | 0.100 | Moderate oscillation | ~O(0.1) |
| 20 | 0.050 | Small oscillation | ~O(0.01) |
| 50 | 0.020 | Very small | ~O(0.001) |
| 100 | 0.010 | Negligible | ~O(1e-4) |

The oscillations decay as `N` increases (smaller `dt`), because smaller time steps allow less amplification per step and the kernel smooths the kink more.

---

## 4. Rannacher Time-Stepping (Implicit Start-Up)

### The Idea

The Fully Implicit (Backward Euler) scheme has amplification factor:

$$g_{BE}(\xi) = \frac{1}{1 - h \lambda(\xi)}$$

For large `|lambda|` (high-frequency modes), `|g_{BE}| -> 0`. This means Backward Euler is **L-stable** and **strongly damps high-frequency components** in the first time step. The Dirac-delta-like second derivative of the payoff is effectively killed.

### The Rannacher Procedure

Replace the first `n_impl = 2` time steps with Backward Euler (each at full `dt`), then switch to Crank-Nicolson for all remaining steps:

```
Step 1: V^1 = (I - dt*L)^{-1} V^0         [Backward Euler -- damps kink]
Step 2: V^2 = (I - dt*L)^{-1} V^1         [Backward Euler -- further smoothing]
Steps 3..N: V^{n+1} = (I - h*L)^{-1} (I + h*L) V^n   [Crank-Nicolson]
```

After two implicit steps, the discontinuous components of the initial data are strongly damped. From step 3 onward, CN operates on a smooth function and its non-dissipative property is no longer a problem.

### Accuracy Preservation

The first two BE steps introduce an extra temporal error of order `O(dt)` in those steps. However, since only two steps out of N total are affected:

$$\text{Total extra error} \sim \frac{2}{N} \cdot O(dt) = O\!\left(\frac{dt}{N}\right) = O\!\left(\frac{T}{N^2}\right)$$

For reasonable `N`, this is negligible compared to the `O(dt^2)` error of CN.

### Quantitative Results (from `payoff_kink_cn_oscillation.py`)

For `M=200`:

| N | Standard CN L-inf | Rannacher L-inf | CN Gamma Osc | Rannacher Gamma Osc |
|--:|---:|---:|---:|---:|
| 5 | large | similar | O(1) | O(0.01) |
| 10 | moderate | similar | O(0.1) | O(0.001) |
| 20 | small | similar | O(0.01) | O(1e-4) |
| 50 | very small | very small | O(1e-3) | O(1e-5) |

The price error (L-inf) is virtually unchanged, while the Gamma oscillation is **reduced by 1-2 orders of magnitude**.

---

## 5. Grid Alignment and Rannacher: Combined Recommendations

The two fixes are complementary and should both be applied in practice:

| Issue | Fix | Effect |
|:---|:---|:---|
| K not on grid → O(dS) spatial order | Align grid: choose M so K = j*dS | Restores O(dS^2) |
| CN oscillations near kink | Rannacher: 2 BE start-up steps | Suppresses Gamma oscillation by 100x |

---

## 6. Summary

| Method | Dissipation | Oscillation near kink | Price accuracy |
|:---|:---:|:---:|:---:|
| Fully Implicit | High (L-stable) | None | O(dS^2, dt) |
| Crank-Nicolson (standard) | None | Yes (coarse dt) | O(dS^2, dt^2) |
| Crank-Nicolson (Rannacher) | First 2 steps only | Strongly suppressed | O(dS^2, dt^2) |

The **Rannacher scheme** is the industry standard for option pricing with non-smooth payoffs. It combines the smoothing power of Backward Euler in the first steps with the accuracy of Crank-Nicolson thereafter.

---

## References

- Rannacher, R. (1984). "Finite element solution of diffusion problems with irregular data." *Numerische Mathematik*, 43(2), 309–327.
- Tavella, D. and Randall, C. (2000). *Pricing Financial Instruments: The Finite Difference Method*. Wiley.
- Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance*, 2nd ed. Wiley.
