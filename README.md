# Numerical Solution of the Black-Scholes Equation

Project on pricing European options numerically using finite-difference methods.
Supervisor: Dr. Ashok Kumar Vaikuntam

---

## Background

The Black-Scholes PDE for an option price V(S, t) is:

```
dV/dt + (1/2) sigma^2 S^2 d^2V/dS^2 + r S dV/dS - r V = 0
```

With the substitution tau = T - t, this becomes a forward-in-time diffusion equation.
The spatial domain is discretised on a uniform grid S_j = j * dS (j = 0, 1, ..., M)
and the PDE is reduced to a tridiagonal linear system at each time step.

---

## Programs

All three programs ask for inputs at runtime. There are no fixed values in the code.

**`programs/black_scholes.py`**
Computes the exact analytical solution using the Black-Scholes formula.
Outputs call price, put price, and all five Greeks for any S, K, T, r, sigma.

**`programs/implicit_fdm.py`**
Solves the Black-Scholes PDE numerically using backward Euler time stepping.
The scheme is first-order accurate in time and second-order in space.
Unconditionally stable for any time step size.

**`programs/crank_nicolson.py`**
Solves the Black-Scholes PDE using the Crank-Nicolson scheme.
Second-order accurate in both time and space, and more accurate per step than the implicit scheme.
Also unconditionally stable.

Supporting files: `thomas_solver.py` (tridiagonal solver used by both FDM programs),
`generate_grid.py` (grid setup), `finite_difference_derivatives.py` (FD verification).

---

## How to run

```
python programs/black_scholes.py
python programs/implicit_fdm.py
python programs/crank_nicolson.py
```

Each program loops until you press Ctrl+C. You can test as many parameter sets as you like.
For the FDM solvers, a starting point is Smax = 4*K, M = 100, N = 500.

---

## Derivations

Step-by-step derivations are in the `derivations/` folder:

- `01_finite_difference_approximations.md` — Taylor expansion, central differences, truncation error
- `02_black_scholes_pde_discretisation.md` — spatial stencil, coefficient formulas
- `03_thomas_algorithm.md` — derivation and implementation of the Thomas algorithm
- `04_fully_implicit_method.md` — backward Euler discretisation, stability proof
- `05_crank_nicolson_method.md` — Crank-Nicolson derivation, accuracy analysis

---

## Accuracy

For K = 100, T = 1, r = 0.05, sigma = 0.20, Smax = 400:

| Method | Grid | ATM error | Time order | Space order |
|--------|------|-----------|------------|-------------|
| Implicit FDM | M=100, N=1000 | 4.1e-02 | O(dt) | O(dS^2) |
| Crank-Nicolson | M=100, N=500 | 4.0e-02 | O(dt^2) | O(dS^2) |

The error reduces as M and N increase. Halving dS roughly quarters the error (O(dS^2)).
For Crank-Nicolson, halving dt also roughly quarters the error (O(dt^2)).

---

## Requirements

Python 3.x with NumPy. No other packages needed.

```
pip install numpy
```
