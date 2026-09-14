# generate_grid.py
# ============================================================
# Generates the computational grids for the Black-Scholes FDM.
#
# Two grids are needed:
#   S-grid : M+1 equally-spaced points from 0 to S_max
#   tau-grid: N+1 equally-spaced time steps from 0 to T
#             (tau = T - t is time-to-maturity, runs forward)
#
# Reference: Duffy (2022), spatial discretisation chapter
# Author  : Akshat
# Project : Numerical Solution of the Black-Scholes Equation
# ============================================================

import numpy as np


def generate_s_grid(S_max, M):
    """
    Uniform asset-price grid.

    Parameters
    ----------
    S_max : float  -- maximum asset price (truncated domain boundary)
    M     : int    -- number of spatial intervals
                      (M+1 grid points: S[0]=0, ..., S[M]=S_max)

    Returns
    -------
    S  : ndarray, shape (M+1,)  -- asset price nodes
    dS : float                  -- spatial step size
    """
    dS = S_max / M
    S  = np.linspace(0.0, S_max, M + 1)
    return S, dS


def generate_tau_grid(T, N):
    """
    Uniform time-to-maturity grid.

    Parameters
    ----------
    T : float  -- maturity (total time)
    N : int    -- number of time steps
                  (N+1 points: tau[0]=0 (expiry), ..., tau[N]=T)

    Returns
    -------
    tau  : ndarray, shape (N+1,)  -- time-to-maturity nodes
    dtau : float                  -- time step size
    """
    dtau = T / N
    tau  = np.linspace(0.0, T, N + 1)
    return tau, dtau


def european_call_payoff(S, K):
    """
    Terminal payoff for European call option at tau=0 (t=T):
        V(S, tau=0) = max(S - K, 0)
    """
    return np.maximum(S - K, 0.0)


def european_put_payoff(S, K):
    """
    Terminal payoff for European put option at tau=0 (t=T):
        V(S, tau=0) = max(K - S, 0)
    """
    return np.maximum(K - S, 0.0)


if __name__ == "__main__":
    # Standard project parameters
    K     = 100.0
    T     = 1.0
    S_max = 400.0
    M     = 10     # small for display
    N     = 5

    S,   dS   = generate_s_grid(S_max, M)
    tau, dtau = generate_tau_grid(T, N)

    print("=" * 55)
    print("  Grid Generation Verification")
    print("  S_max=%g, M=%d, T=%g, N=%d" % (S_max, M, T, N))
    print("=" * 55)

    print("\nS-grid (M+1 = %d points, dS = %.4f):" % (M+1, dS))
    print("  " + str(np.round(S, 4)))
    print("  S[0]  = %.4f  (lower boundary)" % S[0])
    print("  S[-1] = %.4f  (upper boundary)" % S[-1])

    print("\ntau-grid (N+1 = %d points, dtau = %.4f):" % (N+1, dtau))
    print("  " + str(np.round(tau, 4)))

    call_payoff = european_call_payoff(S, K)
    put_payoff  = european_put_payoff(S, K)

    print("\nCall payoff at tau=0 (max(S-K,0)):")
    for i in range(len(S)):
        print("  S=%.1f -> payoff=%.4f" % (S[i], call_payoff[i]))

    print("\nPut payoff at tau=0 (max(K-S,0)):")
    for i in range(len(S)):
        print("  S=%.1f -> payoff=%.4f" % (S[i], put_payoff[i]))

    print()
    print("Boundary condition checks:")
    print("  Call at S=0   : %.4f (should be 0)" % call_payoff[0])
    print("  Call at S=Smax: %.4f (should be ~S_max-K=%.1f)" % (call_payoff[-1], S_max-K))
    print("  Put  at S=0   : %.4f (should be K=%.1f)" % (put_payoff[0], K))
    print("  Put  at S=Smax: %.4f (should be ~0)" % put_payoff[-1])
