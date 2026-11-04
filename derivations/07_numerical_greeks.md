# Derivation 7: Numerical Greeks — Delta and Gamma

## Overview

This document derives the finite-difference approximations for Delta and Gamma computed from the FDM price surface, and analyses their accuracy, particularly near the strike price.

---

## 1. Definitions

In the Black-Scholes framework, the **Greeks** measure sensitivities of the option price to market variables.

### Delta (Delta = dV/dS)

Delta is the rate of change of the option price with respect to the underlying spot price:

$$\Delta = \frac{\partial V}{\partial S}$$

**Analytical Black-Scholes Delta:**

$$\Delta_{\text{call}} = N(d_1), \qquad \Delta_{\text{put}} = N(d_1) - 1$$

where `d_1 = [ln(S/K) + (r + sigma^2/2)T] / (sigma*sqrt(T))` and `N(·)` is the standard normal CDF.

### Gamma (Gamma = d^2V/dS^2)

Gamma is the second derivative of option price with respect to spot:

$$\Gamma = \frac{\partial^2 V}{\partial S^2}$$

**Analytical Black-Scholes Gamma (same for call and put):**

$$\Gamma = \frac{n(d_1)}{S \sigma \sqrt{T}}$$

where `n(·)` is the standard normal PDF.

---

## 2. Numerical Approximations from the FDM Grid

Given the price vector `V_j = V(S_j)` from the FDM solver on the uniform grid `S_j = j * dS`, we compute:

### Numerical Delta

**Central finite difference** (second-order accurate, `O(dS^2)`):

$$\Delta_j^h = \frac{V_{j+1} - V_{j-1}}{2\,\Delta S}$$

This is a **direct post-processing step** — no additional PDE solve is needed. We read off the already-computed price values at adjacent grid nodes.

**Truncation error:**

By Taylor expansion:
$$V_{j+1} = V_j + \Delta S \cdot V'_j + \frac{\Delta S^2}{2} V''_j + \frac{\Delta S^3}{6} V'''_j + O(\Delta S^4)$$
$$V_{j-1} = V_j - \Delta S \cdot V'_j + \frac{\Delta S^2}{2} V''_j - \frac{\Delta S^3}{6} V'''_j + O(\Delta S^4)$$

Subtracting:
$$\frac{V_{j+1} - V_{j-1}}{2\,\Delta S} = V'_j + \frac{\Delta S^2}{6} V'''_j + O(\Delta S^4) = \Delta_j + O(\Delta S^2)$$

So the **Delta approximation is O(dS^2) accurate** in smooth regions.

### Numerical Gamma

**Second central difference** (also `O(dS^2)`):

$$\Gamma_j^h = \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2}$$

**Truncation error:**

Adding the two Taylor expansions:
$$\frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2} = V''_j + \frac{\Delta S^2}{12} V^{(4)}_j + O(\Delta S^4) = \Gamma_j + O(\Delta S^2)$$

So the **Gamma approximation is also O(dS^2)**, but the divisor `dS^2` amplifies absolute errors compared to Delta (divided by `2*dS`).

---

## 3. Accuracy Near the Strike Price

### The Kink and Its Effect

At expiry (`T=0`), the call payoff is `V(S,0) = max(S-K, 0)`, which satisfies:
- `V'(S,0) = 1` for `S > K`
- `V'(S,0) = 0` for `S < K`
- `V''(S,0) = delta(S-K)` (Dirac delta — a singularity)

For `T > 0`, the heat kernel (Black-Scholes fundamental solution) smooths this kink. The smoothed Gamma is:

$$\Gamma(S, T) = \frac{n(d_1)}{S \sigma \sqrt{T}}$$

This peaks sharply at `S = K` with peak value `~1/(sigma * sqrt(2*pi*T)) / K`. For `T=1, sigma=0.2, K=100`, Gamma at `S = K` is `0.01876` (its maximum, `~0.0218`, lies near `S = 90`).

### Why Greeks Are Less Accurate Near the Strike

Even though the exact solution is smooth for `T > 0`, the **initial condition still carries memory of the kink**. The FDM price `V_j` has a discretisation error that is largest near `S = K`, because:

1. The payoff kink means the initial data `V(S_j, 0) = max(S_j - K, 0)` is piecewise linear with a break between nodes.
2. The second-difference operator `(V_{j+1} - 2V_j + V_{j-1}) / dS^2` amplifies errors in `V_j`, especially near the kink.
3. The analytical `d^2V/dS^2` varies most rapidly near `S = K`, requiring finer resolution there.

### Quantitative Comparison (from `numerical_greeks.py`)

For `M = 400, N = 2000, T = 1, K = 100` (Crank-Nicolson): at `S = K` the Delta error is 6.2e-05 (exact value 0.6368) and the Gamma error is 7.2e-06 (exact value 0.01876), i.e. relative errors of about 1e-4 and 4e-4.  Away from the strike (`S = 60` or `160`) the errors are of comparable size (Delta ~1e-4 to 4e-6, Gamma ~1e-7 to 1e-5): at `T = 1` the kink has diffused over a width `sigma sqrt(T) K = 20`, so the strike region is well resolved by `dS = 1`.

**Mesh refinement** (`numerical_greeks.py`, section B; errors at `S = K`, CN with N = 2000):

| T | M | dS | Delta error | order | Gamma error | order |
|--:|--:|---:|---:|---:|---:|---:|
| 1 | 100 | 4.00 | 1.01e-03 | — | 1.18e-04 | — |
| 1 | 200 | 2.00 | 2.50e-04 | 2.01 | 2.91e-05 | 2.02 |
| 1 | 400 | 1.00 | 6.24e-05 | 2.00 | 7.25e-06 | 2.01 |
| 1 | 800 | 0.50 | 1.56e-05 | 2.00 | 1.81e-06 | 2.00 |
| 0.1 | 100 | 4.00 | 4.01e-03 | — | 4.50e-03 | — |
| 0.1 | 200 | 2.00 | 8.84e-04 | 2.18 | 8.53e-04 | 2.40 |
| 0.1 | 400 | 1.00 | 2.16e-04 | 2.03 | 2.04e-04 | 2.07 |
| 0.1 | 800 | 0.50 | 5.38e-05 | 2.01 | 5.03e-05 | 2.02 |

Both Greeks converge with order 2, as predicted by the Taylor analysis.  For `T = 0.1` the kink is only `sigma sqrt(T) K = 6.3` wide; with `dS = 4` the Gamma error at `K` is about 7% of its value (4.5e-3 vs 0.0627), and the asymptotic order is reached only for `dS <~ 2`.  The Greek error near the strike is therefore governed by `dS / (sigma sqrt(T) K)`.  Gamma is less accurate than Delta because it divides by `dS^2` rather than `2 dS`.

---

## 4. Improving Greek Accuracy

### Option 1: Grid Refinement

Doubling M (halving dS) reduces:
- Delta error by a factor of ~4 (O(dS^2))
- Gamma error by a factor of ~4 (O(dS^2))

### Option 2: Non-Uniform Grid

Concentrate grid nodes near the strike by using a transformation such as:

$$S_j = K + \sinh\!\left(\xi_j \cdot \text{arcsinh}\!\left(\frac{S_{max} - K}{c}\right)\right) \cdot c$$

This places more nodes near `S = K` where Gamma peaks, improving accuracy without increasing the total number of nodes.

### Option 3: Bump-and-Reprice

Compute Delta and Gamma by solving the PDE at `S+h` and `S-h` separately (external bumping) with a smaller `h` than the grid spacing. More computationally expensive but free from grid artefacts.

---

## 5. Summary

| Quantity | Formula | Accuracy |
|:---|:---|:---:|
| Delta | `(V_{j+1} - V_{j-1}) / (2*dS)` | `O(dS^2)` |
| Gamma | `(V_{j+1} - 2*V_j + V_{j-1}) / dS^2` | `O(dS^2)` |
| Near-strike Delta error | Larger due to high `d^2 Delta/dS^2` | Decreases as `M` increases |
| Near-strike Gamma error | Largest near `S=K` (peak of Gamma) | Requires fine grid or clustering |

Both CN and Fully Implicit produce the same Greek accuracy for the same spatial grid M, because both use the same O(dS^2) central-difference approximation in space.
