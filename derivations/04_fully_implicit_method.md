# Derivation 4: Fully Implicit Finite-Difference Method

**Author:** Akshat Kumar  
**Supervisor:** Dr. Ashok Kumar Vaikuntam  
**Reference:** Duffy (2022), Chapter 7  

---

## 1. Starting Point: Discrete Spatial Operator

The Black-Scholes spatial operator at interior node $j$ (from Derivation 2):

$$\mathcal{L}_{BS} V_j = \alpha_j V_{j-1} + \beta_j V_j + \gamma_j V_{j+1}$$

$$\alpha_j = \frac{\sigma^2 S_j^2}{2\Delta S^2} - \frac{r S_j}{2\Delta S}, \quad
  \beta_j  = -\frac{\sigma^2 S_j^2}{\Delta S^2} - r, \quad
  \gamma_j = \frac{\sigma^2 S_j^2}{2\Delta S^2} + \frac{r S_j}{2\Delta S}$$

---

## 2. Temporal Discretisation: Backward Euler

Evaluate the spatial operator at the **new** level $n+1$:

$$\frac{V_j^{n+1} - V_j^n}{\Delta\tau} = \mathcal{L}_{BS} V_j^{n+1}$$

Rearranging:

$$a_j V_{j-1}^{n+1} + b_j V_j^{n+1} + c_j V_{j+1}^{n+1} = V_j^n$$

where:

$$a_j = -\Delta\tau\,\alpha_j, \quad b_j = 1 - \Delta\tau\,\beta_j, \quad c_j = -\Delta\tau\,\gamma_j$$

---

## 3. Tridiagonal System and Boundary Conditions

At each time step $n \to n+1$, boundary values supply $V_0^{n+1}, V_M^{n+1}$:

| Option | $V_0^{n+1}$ | $V_M^{n+1}$ |
|:---:|:---:|:---:|
| Call | $0$ | $S_{\max} - K e^{-r\tau_{n+1}}$ |
| Put  | $K e^{-r\tau_{n+1}}$ | $0$ |

Boundary corrections to RHS: $\text{rhs}[0] \mathrel{-}= a_1 V_0^{n+1}$; $\text{rhs}[-1] \mathrel{-}= c_{M-1} V_M^{n+1}$.

---

## 4. Stability

The diagonal dominance condition: $b_j - (|a_j| + |c_j|) = 1 + r\Delta\tau > 0$ always.  
Therefore the matrix is an M-matrix and the scheme is **unconditionally stable**.

---

## 5. Accuracy

- Spatial: $\mathcal{O}(\Delta S^2)$ (central differences)
- Temporal: $\mathcal{O}(\Delta\tau)$ (backward Euler, first-order)
