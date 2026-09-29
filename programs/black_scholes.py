from math import log, sqrt, exp, erf, pi

def N(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def n(x):
    return exp(-0.5 * x * x) / sqrt(2 * pi)

def black_scholes(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = S * N(d1) - K * exp(-r * T) * N(d2)
    put  = call - S + K * exp(-r * T)
    return {
        'call':    call,
        'put':     put,
        'd1':      d1,
        'd2':      d2,
        'delta_c': N(d1),
        'delta_p': N(d1) - 1,
        'gamma':   n(d1) / (S * sigma * sqrt(T)),
        'vega':    S * n(d1) * sqrt(T) / 100,
        'theta_c': (-S*n(d1)*sigma/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2))  / 365,
        'theta_p': (-S*n(d1)*sigma/(2*sqrt(T)) + r*K*exp(-r*T)*N(-d2)) / 365,
        'rho_c':    K * T * exp(-r * T) * N(d2)  / 100,
        'rho_p':   -K * T * exp(-r * T) * N(-d2) / 100,
    }

print("Black-Scholes Calculator  (press Ctrl+C to exit)\n")

while True:
    try:
        S     = float(input("Spot price S     : "))
        K     = float(input("Strike K         : "))
        T     = float(input("Maturity T (yr)  : "))
        r     = float(input("Risk-free rate r : "))
        sigma = float(input("Volatility sigma : "))
    except (ValueError, KeyboardInterrupt, EOFError):
        break

    v = black_scholes(S, K, T, r, sigma)

    print(f"\n  Call   : {v['call']:.6f}     Put    : {v['put']:.6f}")
    print(f"  d1     : {v['d1']:.6f}     d2     : {v['d2']:.6f}")
    print(f"  Delta  : {v['delta_c']:.6f} (call)   {v['delta_p']:.6f} (put)")
    print(f"  Gamma  : {v['gamma']:.6f}")
    print(f"  Vega   : {v['vega']:.6f}  (per 1% move in sigma)")
    print(f"  Theta  : {v['theta_c']:.6f} (call)   {v['theta_p']:.6f} (put)  (per day)")
    print(f"  Rho    : {v['rho_c']:.6f} (call)   {v['rho_p']:.6f} (put)  (per 1% move in r)\n")
