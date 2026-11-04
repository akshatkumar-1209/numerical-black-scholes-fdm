"""
fdm_core.py -- shared numerical core for the Numerical Black-Scholes project
============================================================================
One implementation of everything the study scripts need, so that every
experiment uses exactly the same solver:

  * analytical Black-Scholes price / Delta / Gamma (benchmark)
  * Thomas algorithm for tridiagonal systems (own implementation)
  * Fully implicit (backward Euler), Crank-Nicolson and Rannacher
    (implicit start-up + CN) time stepping on a uniform S-grid
  * error norms and observed-order helpers

PDE in time-to-maturity tau = T - t :
    V_tau = 0.5 sigma^2 S^2 V_SS + r S V_S - r V
Boundary conditions (Dirichlet):
    call: V(0,tau) = 0,            V(Smax,tau) = Smax - K exp(-r tau)
    put : V(0,tau) = K exp(-r tau), V(Smax,tau) = 0
"""
import numpy as np
from math import log, sqrt, exp, erf, pi

# ----------------------------------------------------------------- analytical
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def norm_pdf(x):
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)

def _d1(S, K, T, r, sigma):
    return (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

def bs_price(S, K, T, r, sigma, opt="call"):
    if S <= 0.0:
        return 0.0 if opt == "call" else K * exp(-r * T)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * sqrt(T)
    call = S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    return call if opt == "call" else call - S + K * exp(-r * T)

def bs_delta(S, K, T, r, sigma, opt="call"):
    if S <= 0.0:
        return 0.0 if opt == "call" else -1.0
    d = norm_cdf(_d1(S, K, T, r, sigma))
    return d if opt == "call" else d - 1.0

def bs_gamma(S, K, T, r, sigma):
    if S <= 0.0:
        return 0.0
    return norm_pdf(_d1(S, K, T, r, sigma)) / (S * sigma * sqrt(T))

# -------------------------------------------------------------------- Thomas
class Tridiagonal:
    """Tridiagonal matrix with sub-diagonal a (a[0] unused), diagonal b,
    super-diagonal c (c[-1] unused).  The LU factors are computed once
    (forward elimination); solve() does the forward/back substitution for a
    right-hand side.  The Black-Scholes matrices do not change from one time
    step to the next, so one factorisation serves all steps."""
    def __init__(self, a, b, c):
        n = len(b)
        a = list(map(float, a)); c = list(map(float, c))
        bp = list(map(float, b))
        m = [0.0] * n
        for i in range(1, n):
            m[i] = a[i] / bp[i - 1]
            bp[i] -= m[i] * c[i - 1]
        self.n, self.m, self.bp, self.c = n, m, bp, c

    def solve(self, d):
        n, m, bp, c = self.n, self.m, self.bp, self.c
        d = list(map(float, d))
        for i in range(1, n):
            d[i] -= m[i] * d[i - 1]
        x = [0.0] * n
        x[-1] = d[-1] / bp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / bp[i]
        return np.array(x)

def thomas(a, b, c, d):
    """One-shot Thomas solve (a = lower, b = main, c = upper diagonal)."""
    return Tridiagonal(a, b, c).solve(d)

# ----------------------------------------------------------------- PDE solver
def _bc(opt, K, Smax, r, tau):
    if opt == "call":
        return 0.0, Smax - K * exp(-r * tau)
    return K * exp(-r * tau), 0.0

def solve_fdm(K, T, r, sigma, Smax, M, N, opt="call", scheme="cn", n_start=2):
    """March the Black-Scholes PDE from tau=0 to tau=T.

    scheme = 'implicit'  : backward Euler,            O(dS^2 + dtau)
             'cn'        : Crank-Nicolson (theta=1/2), O(dS^2 + dtau^2)
             'rannacher' : n_start backward-Euler steps (size dtau), then CN
    Returns (S, V) with V the price at tau = T on S_j = j*Smax/M.
    """
    dS, dt = Smax / M, T / N
    S = np.linspace(0.0, Smax, M + 1)
    V = np.maximum(S - K, 0.0) if opt == "call" else np.maximum(K - S, 0.0)
    Sj = S[1:M]
    al = 0.5 * sigma ** 2 * Sj ** 2 / dS ** 2 - 0.5 * r * Sj / dS
    be = -sigma ** 2 * Sj ** 2 / dS ** 2 - r
    ga = 0.5 * sigma ** 2 * Sj ** 2 / dS ** 2 + 0.5 * r * Sj / dS
    # (I - dt L) V^{n+1} = V^n              (backward Euler)
    imp = Tridiagonal(-dt * al, 1.0 - dt * be, -dt * ga)
    # (I - dt/2 L) V^{n+1} = (I + dt/2 L) V^n (Crank-Nicolson)
    h = 0.5 * dt
    cnL = Tridiagonal(-h * al, 1.0 - h * be, -h * ga)
    aR, bR, cR = h * al, 1.0 + h * be, h * ga
    for step in range(N):
        tn, to = (step + 1) * dt, step * dt
        V0n, VMn = _bc(opt, K, Smax, r, tn)
        use_imp = scheme == "implicit" or (scheme == "rannacher" and step < n_start)
        vi = V[1:M]
        if use_imp:
            rhs = vi.copy()
            rhs[0] += dt * al[0] * V0n
            rhs[-1] += dt * ga[-1] * VMn
            new = imp.solve(rhs)
        else:
            V0o, VMo = _bc(opt, K, Smax, r, to)
            rhs = bR * vi
            rhs[1:] += aR[1:] * vi[:-1]
            rhs[:-1] += cR[:-1] * vi[1:]
            rhs[0] += aR[0] * V0o + h * al[0] * V0n
            rhs[-1] += cR[-1] * VMo + h * ga[-1] * VMn
            new = cnL.solve(rhs)
        V[0], V[1:M], V[M] = V0n, new, VMn
    return S, V

def richardson_time(K, T, r, sigma, Smax, M, N, opt="call"):
    """Implicit-Euler solution with the O(dtau) term removed: 2*V(N) - V(N/2).
    Used only to measure the *spatial* error of the implicit scheme."""
    S, V1 = solve_fdm(K, T, r, sigma, Smax, M, N, opt, "implicit")
    _, V2 = solve_fdm(K, T, r, sigma, Smax, M, N // 2, opt, "implicit")
    return S, 2.0 * V1 - V2

# ------------------------------------------------------------ errors / orders
def exact_vector(S, K, T, r, sigma, opt="call"):
    return np.array([bs_price(s, K, T, r, sigma, opt) for s in S])

def error_norms(S, V, ref):
    """L-inf and discrete L2 norm over interior nodes (boundary nodes are
    prescribed, so they carry no information)."""
    e = np.abs(V - ref)[1:-1]
    dS = S[1] - S[0]
    return float(e.max()), float(np.sqrt(np.sum(e ** 2) * dS))

def node_index(S, x, tol=1e-9):
    """Index of the grid node equal to x, or None if x is not a node."""
    j = int(round((x - S[0]) / (S[1] - S[0])))
    return j if abs(S[j] - x) < tol else None

def order(e_coarse, e_fine, ratio=2.0):
    """Observed order p = log(E_h / E_{h/ratio}) / log(ratio)."""
    if e_coarse is None or e_fine is None or e_fine <= 0 or e_coarse <= 0:
        return float("nan")
    return float(np.log(e_coarse / e_fine) / np.log(ratio))

def fd_greeks(S, V):
    """Central-difference Delta and Gamma at interior nodes of a price vector."""
    dS = S[1] - S[0]
    delta = np.full_like(V, np.nan); gamma = np.full_like(V, np.nan)
    delta[1:-1] = (V[2:] - V[:-2]) / (2.0 * dS)
    gamma[1:-1] = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / dS ** 2
    return delta, gamma
