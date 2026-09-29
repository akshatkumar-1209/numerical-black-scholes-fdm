import numpy as np
from math import erf, sqrt, exp, log

def N(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def bs_price(S, K, T, r, sigma, opt):
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    c  = S * N(d1) - K * exp(-r * T) * N(d2)
    return c if opt == 'call' else c - S + K * exp(-r * T)

def thomas(a, b, c, d):
    b, d = b.copy(), d.copy()
    x = np.zeros(len(b))
    for i in range(1, len(b)):
        m = a[i] / b[i-1]
        b[i] -= m * c[i-1]
        d[i] -= m * d[i-1]
    x[-1] = d[-1] / b[-1]
    for i in range(len(b) - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    return x

def crank_nicolson(K, T, r, sigma, Smax, M, N, opt):
    dS, dt = Smax / M, T / N
    S = np.linspace(0, Smax, M + 1)
    V = np.maximum(S - K, 0) if opt == 'call' else np.maximum(K - S, 0)
    Sj = np.arange(1, M) * dS
    al = sigma**2 * Sj**2 / (2 * dS**2) - r * Sj / (2 * dS)
    be = -sigma**2 * Sj**2 / dS**2 - r
    ga = sigma**2 * Sj**2 / (2 * dS**2) + r * Sj / (2 * dS)
    h  = dt / 2
    aL, bL, cL = -h * al, 1 - h * be, -h * ga
    aR, bR, cR =  h * al, 1 + h * be,  h * ga
    for step in range(N):
        tn, to = (step + 1) * dt, step * dt
        V0n = 0 if opt == 'call' else K * exp(-r * tn)
        VMn = (Smax - K * exp(-r * tn)) if opt == 'call' else 0
        V0o = 0 if opt == 'call' else K * exp(-r * to)
        VMo = (Smax - K * exp(-r * to)) if opt == 'call' else 0
        vi  = V[1:M]
        rhs = bR * vi
        rhs[1:]  += aR[1:]  * vi[:-1]
        rhs[:-1] += cR[:-1] * vi[1:]
        rhs[0]   += aR[0]   * V0o  - aL[0]   * V0n
        rhs[-1]  += cR[-1]  * VMo  - cL[-1]  * VMn
        V[0] = V0n
        V[1:M] = thomas(aL, bL, cL, rhs)
        V[M] = VMn
    return S, V

print("Crank-Nicolson FDM Solver  (press Ctrl+C to exit)")
print("Tip: try Smax = 4*K, M = 100, N = 500\n")

while True:
    try:
        K     = float(input("Strike K           : "))
        T     = float(input("Maturity T (years) : "))
        r     = float(input("Risk-free rate r   : "))
        sigma = float(input("Volatility sigma   : "))
        Smax  = float(input("S_max              : "))
        M     =   int(input("Grid points M      : "))
        N_    =   int(input("Time steps N       : "))
        opt   =       input("Option (call/put)  : ").strip().lower()
        S_in  = float(input("Spot price S       : "))
    except (ValueError, KeyboardInterrupt, EOFError):
        break

    S_grid, V = crank_nicolson(K, T, r, sigma, Smax, M, N_, opt)
    j     = min(M, max(0, round(S_in / (Smax / M))))
    S_act = S_grid[j]
    v_fdm = V[j]
    v_ex  = bs_price(S_act, K, T, r, sigma, opt)

    print(f"\n  Grid node S  : {S_act:.4f}")
    print(f"  CN price     : {v_fdm:.6f}")
    print(f"  Exact price  : {v_ex:.6f}")
    print(f"  Error        : {abs(v_fdm - v_ex):.4e}")
    print(f"  dS = {Smax/M:.3f},  dt = {T/N_:.5f}\n")

    if input("  Print full table? (y/n): ").strip().lower() == 'y':
        step = max(1, M // 10)
        print(f"\n  {'S':>8}  {'CN FDM':>10}  {'Exact':>10}  {'Error':>10}")
        for jj in range(0, M + 1, step):
            s  = S_grid[jj]
            vf = V[jj]
            ve = bs_price(s, K, T, r, sigma, opt) if s > 0 else (0 if opt == 'call' else K * exp(-r * T))
            print(f"  {s:>8.2f}  {vf:>10.5f}  {ve:>10.5f}  {abs(vf - ve):>10.4e}")
    print()
