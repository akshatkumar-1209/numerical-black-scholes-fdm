# black_scholes_analytic.py
# ============================================================
# Analytical European call and put option prices using the
# Black-Scholes formula, together with analytical Greeks.
# Reference: Daniel J. Duffy, Numerical Methods in Computational
#            Finance (Wiley, 2022)
# Author  : Akshat
# Project : Numerical Solution of the Black-Scholes Equation
# ============================================================

import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """
    Analytical Black-Scholes price for a European CALL option.
    S     : current asset price
    K     : strike price
    T     : time to maturity (years)
    r     : risk-free rate (continuous compounding)
    sigma : volatility (annualised)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """
    Analytical Black-Scholes price for a European PUT option.
    Uses put-call parity: P = C - S + K * exp(-r*T)
    """
    call = black_scholes_call(S, K, T, r, sigma)
    return call - S + K * np.exp(-r * T)


def analytical_delta(S, K, T, r, sigma, option_type='call'):
    """Analytical Delta dV/dS. Call: N(d1). Put: N(d1)-1."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def analytical_gamma(S, K, T, r, sigma):
    """Analytical Gamma d2V/dS2 (same for call and put)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


if __name__ == "__main__":
    K     = 100.0
    T     = 1.0
    r     = 0.05
    sigma = 0.20

    print("=" * 65)
    print("  Black-Scholes Analytical Benchmark")
    print("  K=%g,  T=%g yr,  r=%g,  sigma=%g" % (K, T, r, sigma))
    print("=" * 65)

    spot_prices = [60, 70, 80, 90, 95, 100, 105, 110, 120, 130, 140]
    print("\n%6s  %12s  %12s  %12s  %12s" % ("S","Call Price","Put Price","Delta(C)","Gamma"))
    print("-" * 62)

    for S in spot_prices:
        c = black_scholes_call(S, K, T, r, sigma)
        p = black_scholes_put(S, K, T, r, sigma)
        d = analytical_delta(S, K, T, r, sigma, "call")
        g = analytical_gamma(S, K, T, r, sigma)
        print("%6.1f  %12.6f  %12.6f  %12.6f  %12.6f" % (S, c, p, d, g))

    print()
    S = 100.0
    c = black_scholes_call(S, K, T, r, sigma)
    p = black_scholes_put(S, K, T, r, sigma)
    error = abs(c - p - S + K * np.exp(-r * T))
    print("Put-Call Parity Check (S=100):")
    print("  C = %.6f" % c)
    print("  P = %.6f" % p)
    print("  |C - P - S + K*exp(-rT)| = %.2e" % error)
    if error < 1e-10:
        print("  PASSED")
    else:
        print("  FAILED - check implementation")

    print()
    print("Key values for Excel Numerical Results sheet:")
    print("  ATM Call price : %.6f" % black_scholes_call(100.0, K, T, r, sigma))
    print("  ATM Put  price : %.6f" % black_scholes_put(100.0, K, T, r, sigma))
    print("  ATM Delta(call): %.6f" % analytical_delta(100.0, K, T, r, sigma))
    print("  ATM Gamma      : %.6f" % analytical_gamma(100.0, K, T, r, sigma))
