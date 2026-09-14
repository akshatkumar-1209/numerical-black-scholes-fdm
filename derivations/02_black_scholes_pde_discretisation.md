# Derivation 2: Black-Scholes PDE Discretisation and Tridiagonal System

**Author:** Akshat Kumar  
**Supervisor:** Dr. Ashok Kumar Vaikuntam  
**Reference:** Daniel J. Duffy, *Numerical Methods in Computational Finance* (Wiley, 2022), Chapters 6–7  

---

## 1. The Continuous Problem

The Black-Scholes PDE for a European option price $V(S, t)$ is given by:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0, \quad S > 0, \; t \in [0, T)$$

with terminal payoff at $t = T$:
$$V(S, T) = \Phi(S) = \begin{cases} \max(S - K, 0) & \text{Call} \\ \max(K - S, 0) & \text{Put} \end{cases}$$

### Transformation to Forward Time
Let $\tau = T - t$ be the time remaining until expiry. Then by the chain rule:
$$\frac{\partial V}{\partial t} = \frac{\partial V}{\partial \tau} \frac{d\tau}{dt} = -\frac{\partial V}{\partial \tau}$$

Substituting into the PDE yields a forward-in-time parabolic equation:

$$\frac{\partial V}{\partial \tau} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V, \quad \tau \in (0, T]$$

with initial condition at $\tau = 0$:
$$V(S, 0) = \Phi(S)$$

---

## 2. Domain Truncation & Boundary Conditions

To solve numerically, the semi-infinite domain $S \in [0, \infty)$ is truncated to a finite computational domain:
$$S \in [0, S_{\max}], \quad S_{\max} = 4K = 400$$

Boundary conditions:
- **At $S = 0$:**
  - Call: $V(0, \tau) = 0$
  - Put: $V(0, \tau) = K e^{-r\tau}$
- **At $S = S_{\max}$:**
  - Call: $V(S_{\max}, \tau) \approx S_{\max} - K e^{-r\tau}$
  - Put: $V(S_{\max}, \tau) \approx 0$

---

## 3. Discretisation Grid

Define a uniform grid:
- Spatial steps: $M$ intervals $\implies \Delta S = \frac{S_{\max}}{M}$, nodes $S_j = j \Delta S$ for $j = 0, 1, \dots, M$.
- Temporal steps: $N$ intervals $\implies \Delta \tau = \frac{T}{N}$, levels $\tau_n = n \Delta \tau$ for $n = 0, 1, \dots, N$.
- Let $V_j^n \approx V(S_j, \tau_n)$.

---

## 4. Finite Difference Spatial Operator

At internal spatial nodes $j = 1, 2, \dots, M-1$, define the spatial operator:

$$\mathcal{L}_{BS} V_j = \frac{1}{2}\sigma^2 S_j^2 \left( \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2} \right) + r S_j \left( \frac{V_{j+1} - V_{j-1}}{2 \Delta S} \right) - r V_j$$

Grouping by nodes $V_{j-1}, V_j, V_{j+1}$:

$$\mathcal{L}_{BS} V_j = \alpha_j V_{j-1} + \beta_j V_j + \gamma_j V_{j+1}$$

where:

$$\alpha_j = \frac{\sigma^2 S_j^2}{2 \Delta S^2} - \frac{r S_j}{2 \Delta S}$$

$$\beta_j = -\frac{\sigma^2 S_j^2}{\Delta S^2} - r$$

$$\gamma_j = \frac{\sigma^2 S_j^2}{2 \Delta S^2} + \frac{r S_j}{2 \Delta S}$$

Notice:
$$\alpha_j + \beta_j + \gamma_j = -r$$

---

## 5. Temporal Discretisation & Tridiagonal System

### Fully Implicit Scheme (Backward Euler in $\tau$):
$$\frac{V_j^{n+1} - V_j^n}{\Delta \tau} = \mathcal{L}_{BS} V_j^{n+1}$$

Rearranging to bring all unknown terms at time level $n+1$ to the left:

$$-\Delta \tau \alpha_j V_{j-1}^{n+1} + (1 - \Delta \tau \beta_j) V_j^{n+1} - \Delta \tau \gamma_j V_{j+1}^{n+1} = V_j^n$$

Setting:
- $a_j = -\Delta \tau \alpha_j = -\Delta \tau \left( \frac{\sigma^2 S_j^2}{2 \Delta S^2} - \frac{r S_j}{2 \Delta S} \right)$
- $b_j = 1 - \Delta \tau \beta_j = 1 + \Delta \tau \left( \frac{\sigma^2 S_j^2}{\Delta S^2} + r \right)$
- $c_j = -\Delta \tau \gamma_j = -\Delta \tau \left( \frac{\sigma^2 S_j^2}{2 \Delta S^2} + \frac{r S_j}{2 \Delta S} \right)$

This forms a linear system at each time step:

$$a_j V_{j-1}^{n+1} + b_j V_j^{n+1} + c_j V_{j+1}^{n+1} = V_j^n, \quad j = 1, \dots, M-1$$

In matrix form:

$$\begin{pmatrix}
b_1 & c_1 & 0 & \dots & 0 \\
a_2 & b_2 & c_2 & \dots & 0 \\
0 & a_3 & b_3 & c_3 & 0 \\
\vdots & & \ddots & \ddots & \vdots \\
0 & \dots & 0 & a_{M-1} & b_{M-1}
\end{pmatrix}
\begin{pmatrix}
V_1^{n+1} \\ V_2^{n+1} \\ V_3^{n+1} \\ \vdots \\ V_{M-1}^{n+1}
\end{pmatrix}
=
\begin{pmatrix}
V_1^n - a_1 V_0^{n+1} \\
V_2^n \\
V_3^n \\
\vdots \\
V_{M-1}^n - c_{M-1} V_M^{n+1}
\end{pmatrix}$$

This is precisely a **tridiagonal linear system**, which is solved in $\mathcal{O}(M)$ time using the Thomas algorithm.
