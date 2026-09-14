# Derivation 3: The Thomas Algorithm (Tridiagonal Matrix Algorithm - TDMA)

**Author:** Akshat Kumar  
**Supervisor:** Dr. Ashok Kumar Vaikuntam  
**Reference:** Daniel J. Duffy, *Numerical Methods in Computational Finance* (Wiley, 2022)  

---

## 1. Mathematical Formulation

A tridiagonal linear system $A x = d$ of dimension $n \times n$ has the explicit form:

$$b_1 x_1 + c_1 x_2 = d_1$$
$$a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i, \quad i = 2, 3, \dots, n-1$$
$$a_n x_{n-1} + b_n x_n = d_n$$

where:
- $a = (0, a_2, \dots, a_n)$ is the sub-diagonal,
- $b = (b_1, b_2, \dots, b_n)$ is the main diagonal,
- $c = (c_1, c_2, \dots, c_{n-1}, 0)$ is the super-diagonal,
- $d = (d_1, d_2, \dots, d_n)$ is the right-hand side vector.

---

## 2. Derivation of Forward Elimination

We wish to transform $A$ into an upper bidiagonal matrix with $1$'s on the main diagonal (or modified diagonal $b'_i$).

For the first equation:
$$x_1 + \frac{c_1}{b_1} x_2 = \frac{d_1}{b_1} \implies c'_1 = \frac{c_1}{b_1}, \quad d'_1 = \frac{d_1}{b_1}$$

For equation $i$ ($2 \le i \le n$), eliminate the sub-diagonal entry $a_i$ using row $i-1$:
$$b'_i = b_i - a_i c'_{i-1}$$
$$d'_i = d_i - a_i d'_{i-1}$$
$$c'_i = \frac{c_i}{b'_i}$$
$$d'_i \leftarrow \frac{d'_i}{b'_i}$$

Equivalently, without normalizing the diagonal to $1$, the standard algorithm computes:

$$m_i = \frac{a_i}{b'_{i-1}}$$
$$b'_i = b_i - m_i c_{i-1}$$
$$d'_i = d_i - m_i d'_{i-1}$$

for $i = 2, 3, \dots, n$, with initial values $b'_1 = b_1$ and $d'_1 = d_1$.

---

## 3. Derivation of Backward Substitution

Once the matrix is upper bidiagonal, the last unknown $x_n$ is obtained immediately:

$$x_n = \frac{d'_n}{b'_n}$$

All preceding unknowns are computed in reverse order for $i = n-1, n-2, \dots, 1$:

$$x_i = \frac{d'_i - c_i x_{i+1}}{b'_i}$$

---

## 4. Computational Complexity

| Method | Operations (Additions/Multiplications) | Memory Complexity |
|:---|:---:|:---:|
| Standard Gaussian Elimination (Dense) | $\frac{2}{3} n^3 + \mathcal{O}(n^2)$ | $\mathcal{O}(n^2)$ |
| LU Decomposition (Dense) | $\frac{2}{3} n^3 + 2n^2$ | $\mathcal{O}(n^2)$ |
| **Thomas Algorithm (TDMA)** | **$5n$ operations ($\mathcal{O}(n)$)** | **$\mathcal{O}(n)$** |

For $n = 1000$ grid points, standard Gaussian elimination requires $\approx 6.7 \times 10^8$ operations, whereas the Thomas algorithm requires only $\approx 5000$ operations—a speedup of over **$130,000\times$**.

---

## 5. Numerical Stability Condition

The Thomas algorithm is numerically stable without pivoting if the tridiagonal matrix is **strictly diagonally dominant**:

$$|b_i| > |a_i| + |c_i|, \quad \forall i = 1, \dots, n$$

In our Black-Scholes implicit scheme:
- $b_i = 1 + \Delta \tau \left( \frac{\sigma^2 S_i^2}{\Delta S^2} + r \right)$
- $|a_i| + |c_i| = \Delta \tau \frac{\sigma^2 S_i^2}{\Delta S^2}$ (when grid satisfies no-oscillation condition $\Delta S \le \frac{\sigma^2 S_i}{r}$)

Thus:
$$b_i - (|a_i| + |c_i|) = 1 + r \Delta \tau > 1 > 0$$

The coefficient matrix is strictly diagonally dominant for all choices of $\Delta S$ and $\Delta \tau > 0$, guaranteeing unconditional numerical stability of the Thomas solver in our Black-Scholes FDM implementation.
