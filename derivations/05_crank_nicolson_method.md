# Derivation 5: Crank-Nicolson Finite-Difference Method

**Author:** Akshat Kumar  
**Supervisor:** Dr. Ashok Kumar Vaikuntam  
**Reference:** Duffy (2022), Chapter 7  

---

## 1. The Theta-Scheme Family

$$\frac{V_j^{n+1} - V_j^n}{\Delta\tau} = \theta\,\mathcal{L}_{BS} V_j^{n+1} + (1-\theta)\,\mathcal{L}_{BS} V_j^n$$

| $\theta$ | Scheme | Time accuracy | Stability |
|:---:|:---|:---:|:---:|
| $0$ | Explicit (Forward Euler) | $\mathcal{O}(\Delta\tau)$ | Conditional |
| $1$ | Fully Implicit | $\mathcal{O}(\Delta\tau)$ | Unconditional |
| $\mathbf{1/2}$ | **Crank-Nicolson** | $\mathcal{O}(\Delta\tau^2)$ | Unconditional |

---

## 2. CN Derivation ($\theta = 1/2$)

Setting $\theta = 1/2$ and expanding $\mathcal{L}_{BS}$:

**LHS matrix $A$ (implicit side):**

$$a_j^A = -\frac{\Delta\tau}{2}\alpha_j, \quad
  b_j^A = 1 - \frac{\Delta\tau}{2}\beta_j, \quad
  c_j^A = -\frac{\Delta\tau}{2}\gamma_j$$

**RHS matrix $B$ (explicit side):**

$$a_j^B = \frac{\Delta\tau}{2}\alpha_j, \quad
  b_j^B = 1 + \frac{\Delta\tau}{2}\beta_j, \quad
  c_j^B = \frac{\Delta\tau}{2}\gamma_j$$

System per step: $A\mathbf{V}^{n+1} = B\mathbf{V}^n + \mathbf{b}^{\text{boundary}}$

---

## 3. Why CN Achieves $\mathcal{O}(\Delta\tau^2)$

CN averages the spatial operator at $\tau_n$ and $\tau_{n+1}$, which is equivalent to a **central difference in time** evaluated at the midpoint $\tau_{n+1/2}$.  By Taylor expansion, this midpoint evaluation has truncation error $\mathcal{O}(\Delta\tau^2)$, gaining a full order over the backward Euler method.

---

## 4. Boundary Corrections

Both the new ($n+1$) and old ($n$) boundary values enter the RHS:

$$\text{rhs}[0] \mathrel{+}= a_0^B V_0^n - a_0^A V_0^{n+1}$$
$$\text{rhs}[-1] \mathrel{+}= c_{M-1}^B V_M^n - c_{M-1}^A V_M^{n+1}$$

---

## 5. Accuracy and Stability

- Spatial: $\mathcal{O}(\Delta S^2)$  
- Temporal: $\mathcal{O}(\Delta\tau^2)$ (**second-order**)  
- Overall: $\mathcal{O}(\Delta S^2 + \Delta\tau^2)$  
- **Unconditionally stable** (LHS matrix A is M-matrix)

**Practical benefit:** at equal cost (same $M, N$), CN is far more accurate than the fully implicit scheme. Verified in `programs/crank_nicolson_solver.py`.
