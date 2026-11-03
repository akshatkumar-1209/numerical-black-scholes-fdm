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

This peaks sharply at `S = K` with peak value `~1/(sigma * sqrt(2*pi*T)) / K`. For `T=1, sigma=0.2, K=100`, the peak Gamma is `~0.0199`.

### Why Greeks Are Less Accurate Near the Strike

Even though the exact solution is smooth for `T > 0`, the **initial condition still carries memory of the kink**. The FDM price `V_j` has a discretisation error that is largest near `S = K`, because:

1. The payoff kink means the initial data `V(S_j, 0) = max(S_j - K, 0)` is piecewise linear with a break between nodes.
2. The second-difference operator `(V_{j+1} - 2V_j + V_{j-1}) / dS^2` amplifies errors in `V_j`, especially near the kink.
3. The analytical `d^2V/dS^2` varies most rapidly near `S = K`, requiring finer resolution there.

### Quantitative Comparison (from `numerical_greeks.py`)

For `M=400, N=2000, T=1, K=100`:

- **Away from strike** (`S = 60` or `S = 160`): Delta error ~`1e-5`, Gamma error ~`1e-4`
- **Near strike** (`S ≈ K`): Delta error ~`2e-4` to `5e-4`, Gamma error ~`1e-3` to `5e-3`

This is expected: Gamma errors are larger than Delta errors by approximately one order of `dS` (since Gamma involves dividing by `dS^2` vs `2*dS`).

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
