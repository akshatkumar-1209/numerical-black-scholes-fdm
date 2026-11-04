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

The payoff `max(S-K,0)` is continuous but its derivative jumps at `S = K`, so the solution is not smooth there at small `tau` and the formal `O(dS^2)` truncation estimate (which needs bounded fourth derivatives) does not apply directly.  What does the experiment show?  `payoff_kink_cn_oscillation.py` (section A) refines the grid within two families -- `K` exactly on a node (`M` a multiple of 4) and `K` exactly half-way between two nodes (`M = 2 mod 4`) -- with the time error removed (CN, N = 2000):

| M (K on node) | dS | L-inf error | order | M (K midway) | dS | L-inf error | order |
|--:|---:|---:|---:|--:|---:|---:|---:|
| 24 | 16.67 | 8.248e-01 | — | 26 | 15.38 | 1.862e-01 | — |
| 48 | 8.33 | 1.777e-01 | 2.21 | 50 | 8.00 | 5.536e-02 | 1.75 |
| 96 | 4.17 | 4.337e-02 | 2.03 | 98 | 4.08 | 1.429e-02 | 1.95 |
| 192 | 2.08 | 1.086e-02 | 2.00 | 194 | 2.06 | 3.700e-03 | 1.95 |
| 384 | 1.04 | 2.712e-03 | 2.00 | 386 | 1.04 | 9.367e-04 | 1.98 |

**Finding.** Both families converge with order `p -> 2.0`.  The kink does *not* reduce the order on a uniform grid, for either alignment; it changes the error constant: at the same `dS` the half-way grid is about 3 times more accurate in `L_inf`.

**Consequence for order measurements.** An observed order is only meaningful when the grids being compared belong to the same family, i.e. the position of `K` relative to the nodes is preserved under refinement (`M -> 2M` does this).  The odd value `p = 0.47` for the step `M = 50 -> 100` in the Week 8 spatial study arises because `K` is half-way between nodes for `M = 50` but on a node for `M = 100`.

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

### Quantitative Example (from `payoff_kink_cn_oscillation.py`, section B)

`M = 200` (`dS = 2`), `T = 1`.  `lambda = dt sigma^2 K^2 / dS^2` is the time step measured in units of the grid.  Gamma window: `K +/- 5` nodes; the exact Gamma is positive everywhere (`~0.019` at `K`).

| N | dt | lambda | CN: max Gamma error | CN: min Gamma | CN: price error |
|--:|---:|---:|---:|---:|---:|
| 3 | 0.3333 | 33.3 | 2.74e-01 | -0.2550 | 5.62e-01 |
| 5 | 0.2000 | 20.0 | 1.58e-01 | -0.1394 | 2.39e-01 |
| 10 | 0.1000 | 10.0 | 2.24e-02 | 0.0009 | 3.39e-02 |
| 20 | 0.0500 | 5.0 | 4.93e-05 | 0.0129 | 9.45e-03 |
| 50 | 0.0200 | 2.0 | 3.01e-05 | 0.0129 | 9.91e-03 |
| 200 | 0.0050 | 0.5 | 3.13e-05 | 0.0129 | 1.00e-02 |

For `lambda >~ 10` (`N <= 10`) CN produces a **negative Gamma** (N = 3, 5) or a badly distorted one (N = 10); for `lambda <~ 5` the Gamma error equals the ordinary spatial error and there is no oscillation.  See `figures/fig4_cn_oscillations_rannacher.png`.

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

Each backward-Euler step has a local truncation error `O(dt^2)`; two such steps therefore add a contribution of order `dt^2` to the global error, which is the same order as the global error of CN, but with a larger constant.  Rannacher time stepping thus keeps second-order accuracy but is *not free*: in the intermediate range it can be slightly less accurate than plain CN (see the table).

### Quantitative Results (from `payoff_kink_cn_oscillation.py`, sections B and C)

`M = 200`.  Price error = `L_inf` error against the exact price; Gamma error as above.

| T | N | CN price err | Rannacher price err | CN Gamma err | Rannacher Gamma err | CN Gamma min |
|--:|--:|---:|---:|---:|---:|---:|
| 1.0 | 3 | 5.62e-01 | 1.96e-01 | 2.74e-01 | 1.63e-03 | -0.2550 |
| 1.0 | 5 | 2.39e-01 | 7.89e-02 | 1.58e-01 | 6.55e-04 | -0.1394 |
| 1.0 | 10 | 3.39e-02 | 2.76e-02 | 2.24e-02 | 2.64e-04 | 0.0009 |
| 1.0 | 20 | 9.45e-03 | 1.44e-02 | 4.93e-05 | 6.09e-05 | 0.0129 |
| 1.0 | 50 | 9.91e-03 | 1.07e-02 | 3.01e-05 | 3.62e-05 | 0.0129 |
| 1.0 | 200 | 1.00e-02 | 1.01e-02 | 3.13e-05 | 3.17e-05 | 0.0129 |
| 0.1 | 3 | 4.45e-02 | 8.44e-02 | 2.58e-02 | 5.48e-03 | 0.0159 |
| 0.1 | 5 | 2.84e-02 | 5.27e-02 | 7.42e-04 | 2.00e-03 | 0.0153 |
| 0.1 | 10 | 3.13e-02 | 3.75e-02 | 7.45e-04 | 1.21e-03 | 0.0152 |
| 0.1 | 20 | 3.19e-02 | 3.35e-02 | 8.26e-04 | 9.46e-04 | 0.0152 |
| 0.1 | 50 | 3.21e-02 | 3.24e-02 | 8.49e-04 | 8.69e-04 | 0.0152 |
| 0.1 | 200 | 3.21e-02 | 3.22e-02 | 8.53e-04 | 8.54e-04 | 0.0152 |

Observations:

1. `T = 1`, `N = 3, 5, 10`: Rannacher removes the oscillation (Gamma stays positive); the Gamma error drops by a factor of about 170, 240 and 85, and the price error also drops.
2. `T = 1`, `N = 20, 50`: no oscillation for either scheme; Rannacher's price error is slightly *larger* (1.4e-2 vs 9.4e-3 at N = 20).  For `N >= 200` the two coincide.
3. `T = 0.1`: with these step sizes CN does not oscillate and Rannacher gives no benefit; for `N <= 20` it increases the price error.  Here the dominant error is the spatial error (grid `dS = 2` against a kink of width `sigma sqrt(T) K = 6.3`).

So Rannacher start-up is a remedy for **large `dt` relative to the grid** (`lambda` large), not a universal improvement.

---

## 5. Practical Recommendations

| Situation | Recommendation | Evidence |
|:---|:---|:---|
| Measuring convergence orders | Compare only grids of the same alignment family (M -> 2M) | Section 2 |
| Large `dt` relative to grid (`lambda >~ 5-10`) with Greeks needed | Use Rannacher (2 BE start-up steps) | Gamma error reduced 85-240x |
| Small `dt` (`lambda <~ 2`) | Plain CN is sufficient | Rannacher adds no benefit, slightly more price error |
| Short maturity, Greeks near `K` | Refine `dS` (or cluster nodes near `K`); `dS` must be small compared with `sigma sqrt(T) K` | `output_9`, section B, T = 0.1 |

---

## 6. Summary

| Method | Dissipation | Oscillation near kink | Price accuracy |
|:---|:---:|:---:|:---:|
| Fully Implicit | High (L-stable) | None | O(dS^2, dt) |
| Crank-Nicolson (standard) | None | Yes (coarse dt) | O(dS^2, dt^2) |
| Crank-Nicolson (Rannacher) | First 2 steps only | Strongly suppressed | O(dS^2, dt^2) |

The **Rannacher scheme** is a standard remedy for option pricing with non-smooth payoffs when the time step is large; It combines the smoothing power of Backward Euler in the first steps with the accuracy of Crank-Nicolson thereafter.

---

## References

- Rannacher, R. (1984). "Finite element solution of diffusion problems with irregular data." *Numerische Mathematik*, 43(2), 309–327.
- Tavella, D. and Randall, C. (2000). *Pricing Financial Instruments: The Finite Difference Method*. Wiley.
- Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance*, 2nd ed. Wiley.
