import numpy as np

def f(x):
    return np.sin(x) * np.exp(-x / 5.0)

def f_prime_exact(x):
    return np.cos(x) * np.exp(-x / 5.0) - (1.0/5.0) * np.sin(x) * np.exp(-x / 5.0)

def f_double_prime_exact(x):
    t1 = -np.sin(x) * np.exp(-x / 5.0)
    t2 = -2.0 * (1.0/5.0) * np.cos(x) * np.exp(-x / 5.0)
    t3 = (1.0/25.0) * np.sin(x) * np.exp(-x / 5.0)
    return t1 + t2 + t3

def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def backward_diff(f, x, h):
    return (f(x) - f(x - h)) / h

def central_diff_first(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)

def central_diff_second(f, x, h):
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / h**2

if __name__ == "__main__":
    x0 = 1.0
    exact1 = f_prime_exact(x0)
    exact2 = f_double_prime_exact(x0)
    print("=" * 72)
    print("  Finite-Difference Approximation Verification")
    print("  f(x) = sin(x)*exp(-x/5)   at   x =", x0)
    print("  Exact f'(x0)  =", round(exact1, 10))
    print("  Exact f''(x0) =", round(exact2, 10))
    print("=" * 72)
    print()
    print("--- First Derivative Convergence Study ---")
    print(f"{'h':>10}  {'Forward err':>14}  {'Backward err':>14}  {'Central err':>14}  {'Ratio(Cen)':>12}")
    print("-" * 70)
    prev_cen = None
    for h in [0.1, 0.05, 0.025, 0.01, 0.005, 0.001]:
        fwd = abs(forward_diff(f, x0, h) - exact1)
        bwd = abs(backward_diff(f, x0, h) - exact1)
        cen = abs(central_diff_first(f, x0, h) - exact1)
        ratio = (prev_cen / cen) if prev_cen else float("nan")
        print(f"{h:>10.4f}  {fwd:>14.3e}  {bwd:>14.3e}  {cen:>14.3e}  {ratio:>12.2f}")
        prev_cen = cen
    print()
    print("Forward/backward error halves -> O(h). Central error quarters -> O(h^2).")
    print()
    print("--- Second Derivative Convergence Study (V_SS analogue) ---")
    print(f"{'h':>10}  {'Central err':>14}  {'Ratio':>12}")
    print("-" * 40)
    prev_err = None
    for h in [0.1, 0.05, 0.025, 0.01, 0.005]:
        err = abs(central_diff_second(f, x0, h) - exact2)
        ratio = (prev_err / err) if prev_err else float("nan")
        print(f"{h:>10.4f}  {err:>14.3e}  {ratio:>12.2f}")
        prev_err = err
    print()
    print("Second derivative error quarters -> O(h^2). Confirmed.")
    print("NOTE: central_diff_second is the V_SS = [V(S+dS)-2V(S)+V(S-dS)]/dS^2 formula.")
