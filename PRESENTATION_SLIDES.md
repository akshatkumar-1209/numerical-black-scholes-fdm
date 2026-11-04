# Numerical Black–Scholes FDM: Project Presentation Slides

**Project:** Finite Difference Methods for European Option Pricing  
**Student:** Akshat Kumar  
**Supervisor:** Dr. Ashok Kumar Vaikuntam  
**Institution / Department:** Computational Finance / Mathematics  

---

## Slide 1: Title and Objective

### Numerical Black–Scholes PDE Using Finite Difference Methods
- **Core Objective:** Design, implement from scratch, verify, and analyze finite-difference pricing algorithms for European options.
- **Implemented Schemes:**
  - Fully Implicit FDM (Backward Euler, $\theta=1$, $\mathcal{O}(\Delta S^2 + \Delta \tau)$)
  - Crank–Nicolson FDM (Trapezoidal, $\theta=1/2$, $\mathcal{O}(\Delta S^2 + \Delta \tau^2)$)
  - Tridiagonal linear solver: Thomas algorithm ($\mathcal{O}(M)$ complexity)
  - Rannacher smoothing for non-smooth payoff kink treatment
- **Core Questions Addressed:**
  1. Do the numerical schemes recover their theoretical spatial and temporal convergence orders?
  2. How do discrete Greek sensitivities ($\Delta, \Gamma$) behave near the strike $S=K$?
  3. Why does standard Crank–Nicolson fail near the payoff kink at coarse time steps, and how does Rannacher smoothing fix it?

---

## Slide 2: Mathematical Formulation

### The Black–Scholes Initial-Boundary Value Problem
- **Time Transformation:** $\tau = T - t \implies \frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V \equiv \mathcal{L}_{\text{BS}} V$
- **Boundary Conditions:**
  - Lower boundary ($S=0$): $V(0,\tau) = 0$ (Call), $V(0,\tau) = K e^{-r\tau}$ (Put)
  - Upper boundary ($S=S_{\max}$): $V(S_{\max},\tau) \approx S_{\max} - K e^{-r\tau}$ (Call), $V(S_{\max},\tau) = 0$ (Put)
  - Terminal condition ($\tau=0$): $V(S,0) = \max(S-K,0)$ (Call)
- **Spatial Central Differencing:**
  - $\alpha_j = \frac{\sigma^2 S_j^2}{2\Delta S^2} - \frac{r S_j}{2\Delta S}, \quad \beta_j = -\frac{\sigma^2 S_j^2}{\Delta S^2} - r, \quad \gamma_j = \frac{\sigma^2 S_j^2}{2\Delta S^2} + \frac{r S_j}{2\Delta S}$
  - Strictly diagonally dominant tridiagonal system $\implies$ unconditionally stable via Thomas algorithm.

---

## Slide 3: Systematic Price Verification

### Benchmark Parameters ($K=100, T=1, r=0.05, \sigma=0.20, S_{\max}=400$)
- Analytical Benchmark Call Price at $S=K=100$: **10.450584**
- **Grid Resolution:** $M=200$ ($\Delta S = 2.0$), Implicit ($N=1000$), CN ($N=500$)

| Spot $S$ | Exact Call Price | Implicit Price | Implicit Error | CN Price | CN Error | Rel Err % (CN) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 80 | 1.859420 | 1.856354 | $3.07 \times 10^{-3}$ | 1.855919 | $3.50 \times 10^{-3}$ | 0.19% |
| 100 (ATM) | 10.450584 | 10.439632 | $1.10 \times 10^{-2}$ | 10.440686 | $9.90 \times 10^{-3}$ | 0.09% |
| 120 | 26.169044 | 26.164336 | $4.71 \times 10^{-3}$ | 26.164798 | $4.25 \times 10^{-3}$ | 0.02% |

- **Takeaway:** Both solvers attain sub-cent accuracy across the entire spot domain; European puts satisfy put-call parity to within numerical precision.

---

## Slide 4: Spatial Convergence and Grid-Alignment Effect

### Decoupled Spatial Study ($N=5000$ Fixed, Vary $M \in \{25, 50, 100, 200, 400\}$)
- **The Aligned vs Unaligned Strike Effect:**
  - For $M=50$ ($\Delta S=8$), $K=100$ falls between nodes ($96$ and $104$). The kink is not resolved on a grid node, degrading convergence.
  - For $M=100, 200, 400$, $K=100$ lies exactly on a node.

| Grid $M$ | $\Delta S$ | $K$ on Grid? | Crank–Nicolson $L_\infty$ Error | Error Reduction Ratio |
|:---:|:---:|:---:|:---:|:---:|
| 50 | 8.00 | NO | $5.5355 \times 10^{-2}$ | — |
| 100 | 4.00 | YES | $4.0002 \times 10^{-2}$ | 1.38 (Alignment transition) |
| 200 | 2.00 | YES | $1.0008 \times 10^{-2}$ | **4.00** |
| 400 | 1.00 | YES | $2.4986 \times 10^{-3}$ | **4.01** |

- **Takeaway:** On aligned meshes, halving $\Delta S$ reduces error by a factor of 4.0, confirming exact $\mathcal{O}(\Delta S^2)$ spatial convergence.

---

## Slide 5: Temporal Convergence & The Same-Grid Reference

### Isolating Temporal Order Without Spatial Floor Masking
- **The Pitfall of External Analytical Baselines:**
  - Measuring temporal error against $V_{\text{exact}}$ on a fixed spatial grid (e.g., $M=800$) introduces a spatial error floor of $\approx 6.2 \times 10^{-4}$.
  - At fine time steps ($N \ge 200$), temporal error drops below $6 \times 10^{-4}$, causing apparent CN convergence to plateau and observed order to collapse to 0.
- **The Rigorous Fix: Same-Grid Richardson Reference:**
  - Fix spatial grid $M=200$; compute $V_{\text{ref}}$ with $N_{\text{ref}} = 12800$ on the **same grid**.
  - $e_{\text{temp}}(N) = \max_j |V_j(M=200, N) - V_j(M=200, 12800)|$.

| Time Steps $N$ | $\Delta \tau$ | Implicit Error | Ratio (Impl) | CN Error | Ratio (CN) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 50 | 0.02000 | $2.1381 \times 10^{-2}$ | — | $1.0097 \times 10^{-4}$ | — |
| 100 | 0.01000 | $1.0656 \times 10^{-2}$ | **2.01** ($\approx \mathcal{O}(\Delta\tau)$) | $2.5244 \times 10^{-5}$ | **4.00** ($\approx \mathcal{O}(\Delta\tau^2)$) |
| 200 | 0.00500 | $5.2876 \times 10^{-3}$ | **2.02** | $6.3099 \times 10^{-6}$ | **4.00** |
| 400 | 0.00250 | $2.6023 \times 10^{-3}$ | **2.03** | $1.5763 \times 10^{-6}$ | **4.00** |
| 800 | 0.00125 | $1.2593 \times 10^{-3}$ | **2.07** | $3.9292 \times 10^{-7}$ | **4.01** |

---

## Slide 6: Numerical Greeks & Short-Maturity Behavior

### Discrete Sensitivities via Central Differences ($M=400, N=2000$)
- $\Delta_j = \frac{V_{j+1} - V_{j-1}}{2\Delta S}, \qquad \Gamma_j = \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2}$
- **Mesh Refinement at Strike ($S=K=100$):**
  - $M=100 \implies \Delta \text{ Err} = 1.01 \times 10^{-3}, \quad \Gamma \text{ Err} = 1.18 \times 10^{-4}$
  - $M=200 \implies \Delta \text{ Err} = 2.50 \times 10^{-4}$ (Ratio 4.04), $\Gamma \text{ Err} = 2.91 \times 10^{-5}$ (Ratio 4.07)
  - $M=400 \implies \Delta \text{ Err} = 6.24 \times 10^{-5}$ (Ratio 4.01), $\Gamma \text{ Err} = 7.25 \times 10^{-6}$ (Ratio 4.02)
- **Short-Maturity Impact ($T=0.1\,\text{y}$ vs $T=1.0\,\text{y}$):**
  - At $T=1.0\,\text{y}$, diffusion has smoothed the kink ($\Gamma_{\text{peak}} = 0.01876$).
  - At $T=0.1\,\text{y}$, the kink is concentrated ($\Gamma_{\text{peak}} = 0.06269$, a $3.3\times$ increase).
  - Gamma error near strike rises from $7.25 \times 10^{-6}$ to $2.04 \times 10^{-4}$ on the same mesh.

---

## Slide 7: Payoff Kink, CN Oscillations & Rannacher Smoothing

### The Mechanism of Failure in Crank–Nicolson
- Terminal payoff has discontinuous derivative: $V''(S,0) = \delta(S-K)$.
- Crank–Nicolson amplification factor: $R_{\text{CN}}(z) = \frac{1 - z/2}{1 + z/2} \to -1$ as $z \to \infty$.
- **CN is A-stable but NOT L-stable**: high-frequency modes excited by the kink are preserved without damping, alternating sign at each step.
- **Consequence at Coarse Time Steps ($M=200, N=5$):**
  - Standard CN Gamma at $S=K$: **-0.139** (non-physical negative Gamma; true value is $+0.0188$).
  - Oscillation amplitude: **0.246**.
- **The Rannacher Fix (1984):**
  - Replace the first two time steps with Backward Euler ($R_{\text{BE}}(z) \to 0$ as $z \to \infty$, L-stable).
  - Strongly damps the high-frequency modes; subsequent steps resume Crank–Nicolson.

---

## Slide 8: Quantitative Comparison: Standard CN vs Rannacher

### Oscillation Suppression and Price Error Trade-Off ($M=200$)

| Time Steps $N$ | CN Gamma Oscillation | Rannacher Gamma Oscillation | Oscillation Suppression | CN Price $L_\infty$ Error | Rannacher Price $L_\infty$ Error |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | 0.24559 | 0.00140 | **175×** | $2.39 \times 10^{-1}$ | $7.89 \times 10^{-2}$ |
| 10 | 0.04032 | 0.00128 | **31×** | $3.39 \times 10^{-2}$ | $2.76 \times 10^{-2}$ |
| 20 | 0.00114 | 0.00109 | $1.05\times$ | **$9.45 \times 10^{-3}$** | **$1.44 \times 10^{-2}$** |

- **Key Practical Insight:**
  - At $N=5$, Rannacher eliminates oscillations completely and improves price error ($0.239 \to 0.079$).
  - At $N=20$, Rannacher price error is slightly larger than CN ($1.44 \times 10^{-2}$ vs $9.45 \times 10^{-3}$) due to the $\mathcal{O}(\Delta\tau)$ error from the initial implicit steps.
  - **Engineering Conclusion:** Rannacher smoothing is designed for sensitivity stability and eliminating non-physical negative Gammas, representing a minor trade-off in price accuracy for smooth Greeks.

---

## Slide 9: Summary & Recommendations

1. **Independent Verification:**
   - Both implicit and Crank–Nicolson schemes verified unconditionally stable and convergent.
   - Thomas algorithm achieves $\mathcal{O}(M)$ execution with zero stability degradation.
2. **Best Practices for Financial FDM:**
   - Always choose grid resolution such that $K$ coincides with an interior node ($K = j \Delta S$).
   - For temporal convergence analysis, always use a same-grid Richardson reference to avoid spatial floor artifacts.
   - Never run raw Crank–Nicolson on coarse grids or short maturities without Rannacher start-up damping.
3. **Automated Reproducibility:**
   - Complete project test suite: `python reproduce_all_results.py` (11 assertions passed in $<30\,\text{s}$).
