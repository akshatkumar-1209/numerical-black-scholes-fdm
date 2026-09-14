# Derivation 1: Finite-Difference Approximations from Taylor Expansions

**Author:** Akshat Kumar  
**Supervisor:** Dr. Ashok Kumar Vaikuntam  
**Reference:** Daniel J. Duffy, *Numerical Methods in Computational Finance* (Wiley, 2022), Chapters 3–4  

---

## 1. Objective

To derive the standard forward, backward, and central difference formulas for first and second derivatives from Taylor-series expansions, and to determine their leading truncation error terms.

---

## 2. Taylor Series Expansions

Assume $f: \mathbb{R} \to \mathbb{R}$ is sufficiently smooth ($f \in C^4$). For a small grid spacing $h > 0$, Taylor's theorem states:

$$f(x + h) = f(x) + h f'(x) + \frac{h^2}{2!} f''(x) + \frac{h^3}{3!} f^{(3)}(x) + \frac{h^4}{4!} f^{(4)}(\xi_1), \quad \xi_1 \in (x, x+h)$$

$$f(x - h) = f(x) - h f'(x) + \frac{h^2}{2!} f''(x) - \frac{h^3}{3!} f^{(3)}(x) + \frac{h^4}{4!} f^{(4)}(\xi_2), \quad \xi_2 \in (x-h, x)$$

---

## 3. First Derivative Approximations

### 3.1 Forward Difference (First Order)

Rearranging the expansion of $f(x + h)$:

$$h f'(x) = f(x + h) - f(x) - \frac{h^2}{2} f''(x) - \mathcal{O}(h^3)$$

Dividing by $h$:

$$f'(x) = \frac{f(x + h) - f(x)}{h} - \frac{h}{2} f''(\xi_1)$$

- **Finite difference formula:** $D_+ f(x) = \frac{f(x+h) - f(x)}{h}$
- **Truncation error:** $\mathcal{O}(h)$ — first-order accurate. Halving $h$ reduces error by a factor of 2.

### 3.2 Backward Difference (First Order)

Rearranging the expansion of $f(x - h)$:

$$h f'(x) = f(x) - f(x - h) + \frac{h^2}{2} f''(x) - \mathcal{O}(h^3)$$

Dividing by $h$:

$$f'(x) = \frac{f(x) - f(x - h)}{h} + \frac{h}{2} f''(\xi_2)$$

- **Finite difference formula:** $D_- f(x) = \frac{f(x) - f(x-h)}{h}$
- **Truncation error:** $\mathcal{O}(h)$ — first-order accurate.

### 3.3 Central Difference (Second Order)

Subtracting the expansion of $f(x - h)$ from $f(x + h)$:

$$f(x + h) - f(x - h) = 2h f'(x) + 2\frac{h^3}{3!} f^{(3)}(x) + \mathcal{O}(h^5)$$

Dividing by $2h$:

$$f'(x) = \frac{f(x + h) - f(x - h)}{2h} - \frac{h^2}{6} f^{(3)}(\xi)$$

- **Finite difference formula:** $D_0 f(x) = \frac{f(x+h) - f(x-h)}{2h}$
- **Truncation error:** $\mathcal{O}(h^2)$ — second-order accurate. Because symmetric even terms cancel, halving $h$ quarters the error.

---

## 4. Second Derivative Approximation

Adding the expansions of $f(x + h)$ and $f(x - h)$:

$$f(x + h) + f(x - h) = 2f(x) + h^2 f''(x) + 2\frac{h^4}{4!} f^{(4)}(\xi) + \mathcal{O}(h^6)$$

Rearranging for $f''(x)$:

$$h^2 f''(x) = f(x + h) - 2f(x) + f(x - h) - \frac{h^4}{12} f^{(4)}(\xi)$$

Dividing by $h^2$:

$$f''(x) = \frac{f(x + h) - 2f(x) + f(x - h)}{h^2} - \frac{h^2}{12} f^{(4)}(\xi)$$

- **Finite difference formula:** $D_+ D_- f(x) = \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$
- **Truncation error:** $\mathcal{O}(h^2)$ — second-order accurate. Halving $h$ quarters the error.

---

## 5. Application to the Black-Scholes Operator

In the Black-Scholes PDE, the asset price derivatives at grid node $S_j = j \Delta S$ are approximated by:

1. **Delta ($\frac{\partial V}{\partial S}$):**
   $$\left(\frac{\partial V}{\partial S}\right)_j \approx \frac{V_{j+1} - V_{j-1}}{2 \Delta S}, \quad \text{Error } = \mathcal{O}(\Delta S^2)$$

2. **Gamma ($\frac{\partial^2 V}{\partial S^2}$):**
   $$\left(\frac{\partial^2 V}{\partial S^2}\right)_j \approx \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2}, \quad \text{Error } = \mathcal{O}(\Delta S^2)$$

Both central stencils involve only nearest neighbours $(j-1, j, j+1)$, forming a 3-point stencil which produces a tridiagonal linear system.

---

## 6. Numerical Verification

Implemented and verified in `programs/finite_difference_derivatives.py` against analytical test function $f(x) = \sin(x) e^{-x/5}$ at $x = 1.0$:
- Forward/Backward error ratio as $h \to h/2 \approx 2.00$ ($\\mathcal{O}(h)$ verified).
- Central first and second derivative error ratio as $h \to h/2 \approx 4.00$ ($\\mathcal{O}(h^2)$ verified).
