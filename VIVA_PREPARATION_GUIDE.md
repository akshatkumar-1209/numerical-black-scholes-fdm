# Viva Preparation & Numerical Defense Guide
**Project:** Numerical Black–Scholes PDE using Finite Differences  
**Student:** Akshat Kumar  
**Examiners / Supervisors:** Dr. Ashok Kumar Vaikuntam & Prof. Arshad Javed  
**Date:** November 2026  

---

## 🎯 Purpose of this Guide

In accordance with condition 3 of Prof. Ashok Kumar Vaikuntam's directive:
> *"In the viva, you will run the code in front of me, reproduce the results, and explain the derivations and numerical behaviour in your own words. Only work you can demonstrate and explain will count towards your grade."*

This guide equips you with:
1. **Live Code Demonstration Commands:** Exact commands to run that reproduce all outputs, tables, and plots in $< 30$ seconds.
2. **Derivations & Mathematics:** Step-by-step derivations you can write on a board or paper from first principles without looking at notes.
3. **Core Numerical Explanations:** Intuitive, mathematically precise explanations for every key result (convergence orders, why CN oscillates, how Rannacher cures it, Greek behavior near strike).
4. **Anticipated Supervisor Questions & Model Answers:** Rapid-fire Q&A covering the exact traps and technical inquiries professors typically pose.

---

## 1. Live Code Reproduction (Run in front of the Professor)

Open the PowerShell / Command Prompt in your project directory `C:\Users\Asus\Desktop\SOP`:

### ⚡ Step 1: Run the Master Reproduction Script
```bash
python reproduce_all_results.py
```
- **Execution time:** $\approx 25$ seconds.
- **What it prints:**
  1. The $4 \times 4$ error matrix for $M, N \in \{50, 100, 200, 400\}$ for both Implicit and Crank–Nicolson.
  2. The decoupled spatial convergence table ($N=5000$) showing observed order $p \approx 2.00$ ($\mathcal{O}(\Delta S^2)$).
  3. The decoupled temporal convergence table ($M=800$) showing observed order $q \approx 1.00$ for Implicit ($\mathcal{O}(\Delta \tau)$) and $q \approx 2.00$ for CN ($\mathcal{O}(\Delta \tau^2)$).
  4. Numerical Delta and Gamma comparisons against analytical values.
  5. The payoff kink oscillation amplitude under standard CN ($3.28 \times 10^{-2}$) versus Rannacher smoothing ($1.40 \times 10^{-3}$), showing a **$23.4\times$ to $95\times$ suppression**.
- **What files it updates:**
  - `results/output_11_grid_refinement_matrix.txt`
  - `results/output_8_convergence_study.txt`
  - `results/output_9_numerical_greeks.txt`
  - `results/output_10_payoff_kink_oscillation.txt`

### ⚡ Step 2: Regenerate All Publication Figures
```bash
python generate_figures.py
```
- Instantly regenerates 4 high-resolution plots in `figures/`:
  - `fig1_spatial_convergence.png`: Log-log spatial convergence plot confirming slope 2.
  - `fig2_temporal_convergence.png`: Log-log temporal convergence plot contrasting slope 1 (Implicit) vs slope 2 (CN).
  - `fig3_numerical_greeks.png`: Sensitivity profiles of $\Delta$ and $\Gamma$ across spot.
  - `fig4_cn_oscillations_rannacher.png`: Spurious oscillations in standard CN versus smooth Rannacher curve.

---

## 2. Derivations You Must Be Prepared to Write on the Board

### 2.1 The PDE and Forward Time Transformation
1. **Black–Scholes Equation:**
   $$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$$
2. **Transformation to Forward Diffusion:**
   Let $\tau = T - t$. By chain rule:
   $$\frac{\partial V}{\partial t} = \frac{\partial V}{\partial \tau} \frac{\partial \tau}{\partial t} = -\frac{\partial V}{\partial \tau}$$
   Substituting yields the forward parabolic equation:
   $$\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V \equiv \mathcal{L}_{BS} V$$

### 2.2 Spatial Finite Difference Stencil
Using a uniform grid $S_j = j \Delta S$ for $j=1, \dots, M-1$:
- Central difference for first derivative:
  $$\frac{\partial V}{\partial S}\bigg|_j = \frac{V_{j+1} - V_{j-1}}{2\Delta S} - \frac{\Delta S^2}{6} V^{(3)}(\xi_1) = \frac{V_{j+1} - V_{j-1}}{2\Delta S} + \mathcal{O}(\Delta S^2)$$
- Central difference for second derivative:
  $$\frac{\partial^2 V}{\partial S^2}\bigg|_j = \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2} - \frac{\Delta S^2}{12} V^{(4)}(\xi_2) = \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2} + \mathcal{O}(\Delta S^2)$$

Substitute into the operator $\mathcal{L}_{BS} V_j$:
$$\mathcal{L}_{BS}^h V_j = \frac{1}{2}\sigma^2 S_j^2 \left(\frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2}\right) + r S_j \left(\frac{V_{j+1} - V_{j-1}}{2\Delta S}\right) - r V_j$$
Collect coefficients of $V_{j-1}$, $V_j$, and $V_{j+1}$:
$$\mathcal{L}_{BS}^h V_j = \alpha_j V_{j-1} + \beta_j V_j + \gamma_j V_{j+1}$$
Where:
$$\alpha_j = \frac{\sigma^2 S_j^2}{2\Delta S^2} - \frac{r S_j}{2\Delta S}, \qquad \beta_j = -\frac{\sigma^2 S_j^2}{\Delta S^2} - r, \qquad \gamma_j = \frac{\sigma^2 S_j^2}{2\Delta S^2} + \frac{r S_j}{2\Delta S}$$

### 2.3 Time Discretisation & Tridiagonal Formulation
Let $V_j^n \approx V(S_j, \tau_n)$ where $\tau_n = n \Delta\tau$.

1. **Fully Implicit (Backward Euler):**
   $$\frac{V_j^{n+1} - V_j^n}{\Delta\tau} = \mathcal{L}_{BS}^h V_j^{n+1}$$
   Rearrange to LHS (time level $n+1$) and RHS (time level $n$):
   $$(-\Delta\tau \alpha_j) V_{j-1}^{n+1} + (1 - \Delta\tau \beta_j) V_j^{n+1} + (-\Delta\tau \gamma_j) V_{j+1}^{n+1} = V_j^n$$
   Let $a_j = -\Delta\tau \alpha_j$, $b_j = 1 - \Delta\tau \beta_j$, $c_j = -\Delta\tau \gamma_j$.
   This forms the tridiagonal system:
   $$a_j V_{j-1}^{n+1} + b_j V_j^{n+1} + c_j V_{j+1}^{n+1} = V_j^n$$
   *Diagonal Dominance:* Since $\beta_j < 0$ and $a_j < 0, c_j < 0$, $|b_j| = 1 + \Delta\tau |\beta_j| > |a_j| + |c_j| = \Delta\tau (\alpha_j + \gamma_j) = \Delta\tau (\frac{\sigma^2 S_j^2}{\Delta S^2}) = \Delta\tau (|\beta_j| - r)$. Thus $|b_j| > |a_j| + |c_j| + 1 + r\Delta\tau$, guaranteeing strict diagonal dominance.

2. **Crank–Nicolson Scheme ($\theta = 1/2$):**
   $$\frac{V_j^{n+1} - V_j^n}{\Delta\tau} = \frac{1}{2}\mathcal{L}_{BS}^h V_j^{n+1} + \frac{1}{2}\mathcal{L}_{BS}^h V_j^n$$
   Let $h = \Delta\tau / 2$:
   $$(I - h \mathcal{L}^h) \mathbf{V}^{n+1} = (I + h \mathcal{L}^h) \mathbf{V}^n$$
   LHS Matrix $A$ has elements:
   $$a_L = -h \alpha_j, \quad b_L = 1 - h \beta_j, \quad c_L = -h \gamma_j$$
   RHS Matrix $B$ has elements:
   $$a_R = +h \alpha_j, \quad b_R = 1 + h \beta_j, \quad c_R = +h \gamma_j$$

### 2.4 The Thomas Algorithm (Tridiagonal Gaussian Elimination)
To solve $a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i$ for $i=1, \dots, m$ with $a_1 = 0, c_m = 0$:
1. **Forward Sweep (Eliminate subdiagonal $a_i$):**
   For $i = 2, 3, \dots, m$:
   $$m_i = \frac{a_i}{b_{i-1}}$$
   $$b_i \leftarrow b_i - m_i c_{i-1}$$
   $$d_i \leftarrow d_i - m_i d_{i-1}$$
2. **Back Substitution (Solve upper triangular system):**
   $$x_m = \frac{d_m}{b_m}$$
   For $i = m-1, m-2, \dots, 1$:
   $$x_i = \frac{d_i - c_i x_{i+1}}{b_i}$$
*Complexity:* Exactly $5m$ operations $\implies \mathcal{O}(m)$ linear time, compared to $\mathcal{O}(m^3)$ for general Gaussian elimination.

---

## 3. High-Yield Viva Questions & Defenses

### Q1: "Why do we decouple spatial and temporal errors instead of just varying M and N together?"
> **Answer:** "Because the global truncation error is the sum of spatial and temporal errors:
> $$E(M, N) \approx C_1 \Delta S^p + C_2 \Delta \tau^q$$
> If we vary $M$ and $N$ simultaneously, the observed error ratio blends both terms, making it impossible to ascertain whether second-order spatial or first/second-order temporal accuracy is actually achieved. 
> To isolate spatial order $p$, we fix $N=5000$ so that $C_2 \Delta \tau^q < 10^{-6}$, rendering temporal error negligible. Then $E \propto \Delta S^p$.
> To isolate temporal order $q$, we fix $M=800$ so that $C_1 \Delta S^p \approx 3 \times 10^{-5}$ (the spatial error floor), and vary $N \in \{50, 100, 200, 400\}$ where temporal error dominates."

---

### Q2: "In your spatial study, why is the error ratio at M=50 only ~1.87, but jumps to 4.02 at M=100 and 4.01 at M=200?"
> **Answer:** "This is the **grid-alignment effect** caused by the payoff kink at strike $K=100$. 
> With $S_{\max} = 400$:
> - At $M=50$, the grid spacing is $\Delta S = 400/50 = 8.0$. The grid nodes are at $\dots, 88, 96, 104, 112, \dots$. The strike $K=100$ falls halfway between node $12$ ($S=96$) and node $13$ ($S=104$). The kink is not resolved on a node, introducing an $\mathcal{O}(\Delta S)$ piecewise-linear interpolation error that temporarily degrades the observed ratio.
> - At $M=100$, $\Delta S = 4.0$. $K=100$ is exactly on node $25$ ($25 \times 4 = 100$).
> - At $M=200$, $\Delta S = 2.0$. $K=100$ is on node $50$.
> - At $M=400$, $\Delta S = 1.0$. $K=100$ is on node $100$.
> As soon as $K$ is on the grid, the kink lies on a node, and the method exhibits exact $\mathcal{O}(\Delta S^2)$ convergence with ratios of $4.00 \pm 0.02$."

---

### Q3: "Crank–Nicolson is unconditionally stable (A-stable). Why does it produce oscillations near the strike for coarse time steps?"
> **Answer:** "Crank–Nicolson is A-stable, but **not L-stable**. 
> The amplification symbol for Crank–Nicolson for an eigenvalue $\lambda < 0$ of the spatial operator is:
> $$R_{CN}(z) = \frac{1 - z/2}{1 + z/2}, \quad \text{where } z = -\lambda \Delta\tau > 0$$
> For high-frequency spatial modes ($z \to \infty$):
> $$\lim_{z \to \infty} R_{CN}(z) = -1$$
> The magnitude $|R_{CN}(z)| \to 1$, which means Crank–Nicolson does **not dissipate high-frequency Fourier components**. Instead, it multiplies them by $\approx -1$ at every time step, producing persistent alternating oscillations!
> Because the initial payoff $V(S,0) = \max(S-K,0)$ has a sharp corner (discontinuous first derivative, and second derivative $\delta(S-K)$), it strongly excites high-frequency spatial modes. For large $\Delta\tau$, these modes are not damped and propagate through the mesh, appearing as severe spatial oscillations in numerical derivatives like Gamma ($\Gamma = \partial^2 V / \partial S^2$)."

---

### Q4: "How does Rannacher time-stepping solve this problem, and why doesn't it ruin the second-order accuracy of Crank–Nicolson?"
> **Answer:** "Rannacher time-stepping replaces the first two time steps with **Fully Implicit (Backward Euler)** steps, and then switches back to Crank–Nicolson for all remaining steps.
> The Backward Euler symbol is:
> $$R_{BE}(z) = \frac{1}{1 + z}$$
> As $z \to \infty$, $R_{BE}(z) \to 0$. Backward Euler is **L-stable**: it aggressively damps high-frequency components to near zero in just two steps.
> Once the high frequencies excited by the kink are eliminated, the solution becomes mathematically smooth for all subsequent steps. Crank–Nicolson can then advance the smooth solution without oscillating.
> Because only 2 out of $N$ steps are Backward Euler, the local $\mathcal{O}(\Delta\tau)$ error is introduced only over an interval of $\mathcal{O}(\Delta\tau)$, resulting in an $\mathcal{O}(\Delta\tau^2)$ global impact. Thus, Rannacher smoothing eliminates oscillations by up to **$95\times$** while preserving overall $\mathcal{O}(\Delta\tau^2)$ accuracy."

---

### Q5: "How do you calculate numerical Delta and Gamma from the discrete solution, and why is Gamma less accurate near the strike?"
> **Answer:** "Delta and Gamma are computed directly from the post-solution grid via central differences:
> $$\Delta_j = \frac{V_{j+1} - V_{j-1}}{2\Delta S}, \qquad \Gamma_j = \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2}$$
> Both are second-order approximations $\mathcal{O}(\Delta S^2)$ in smooth regions.
> Gamma is less accurate near the strike because:
> 1. Gamma is a second derivative, so dividing by $\Delta S^2$ magnifies numerical discretization errors by a factor of $1/\Delta S$ compared to Delta.
> 2. The analytical Gamma $\Gamma(S) = \frac{n(d_1)}{S\sigma\sqrt{T}}$ peaks sharply at $S=K$. The curvature of the option price is maximal at the strike, requiring the highest spatial resolution to capture accurately."

---

## 4. Quick Checklist Before You Enter the Room

- [ ] Project directory `C:\Users\Asus\Desktop\SOP` open in terminal.
- [ ] GitHub repository URL verified: `https://github.com/akshatkumar-1209/numerical-black-scholes-fdm`.
- [ ] Workbook `Akshat_Numerical_Black_Scholes_Project_Tracker.xlsx` verified to have:
  - `Post-Midsem Recovery` sheet with all 10 tasks and numerical data.
  - `Supervisor Review` sheet ready for examiner signatures.
  - `12-Week Tracker` updated to 100% completion.
  - `Numerical Results` with experiments 1 to 14.
- [ ] Run `python reproduce_all_results.py` once to ensure cached bytecodes are loaded.
- [ ] Keep `TECHNICAL_REPORT.md` open for instant reading of any analytical derivation.
