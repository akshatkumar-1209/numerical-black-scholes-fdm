"""
payoff_kink_cn_oscillation.py  --  Week 10
==========================================
Investigation of:
1. The non-smooth payoff (kink at S=K) and its effect on convergence
2. Spurious oscillations in the Crank-Nicolson method for small dt
3. The Rannacher time-stepping fix: replace the first two CN steps with
   Fully Implicit (backward-Euler) steps to damp oscillations

Theory:
  The terminal payoff V(S, 0) = max(S-K, 0) is only C^0 at S=K.
  Its second derivative is a Dirac delta: V''(S,0) = delta(S-K).
  The Crank-Nicolson scheme, being a rational approximation to exp(-L*dt),
  passes this non-smoothness forward and can produce spurious oscillations
  in V and its Greeks -- especially visible in Gamma near S=K.

  The Rannacher fix: use two implicit (BE) steps first, which damps the
  high-frequency components before CN takes over.

Parameters: K=100, T=1, r=0.05, sigma=0.20, Smax=400

Author: Akshat Kumar
"""

import numpy as np
from math import log, sqrt, exp, erf, pi

# ─────────────────────────────────────────────────────────────────────────────
# Analytical functions
# ─────────────────────────────────────────────────────────────────────────────

def _N(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _n(x):
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)

def bs_call(S, K, T, r, sigma):
    if S <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * _N(d1) - K * exp(-r * T) * _N(d2)

def bs_gamma_val(S, K, T, r, sigma):
    if S <= 0 or T <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return _n(d1) / (S * sigma * sqrt(T))

# ─────────────────────────────────────────────────────────────────────────────
# Thomas Algorithm
# ─────────────────────────────────────────────────────────────────────────────

def thomas(a, b, c, d):
    n = len(b); b = b.copy(); d = d.copy(); x = np.zeros(n)
    for i in range(1, n):
        m = a[i]/b[i-1]; b[i] -= m*c[i-1]; d[i] -= m*d[i-1]
    x[-1] = d[-1]/b[-1]
    for i in range(n-2,-1,-1): x[i] = (d[i]-c[i]*x[i+1])/b[i]
    return x

# ─────────────────────────────────────────────────────────────────────────────
# Solvers
# ─────────────────────────────────────────────────────────────────────────────

def _implicit_step(V, M, a, b, c, V0, VM):
    """One backward-Euler step (modifies V in place, returns interior)."""
    rhs = V[1:M].copy()
    rhs[0]  -= a[0]  * V0
    rhs[-1] -= c[-1] * VM
    return thomas(a, b, c, rhs)

def _cn_step(V, M, aL, bL, cL, aR, bR, cR, V0n, VMn, V0o, VMo):
    """One Crank-Nicolson step."""
    vi = V[1:M]
    rhs = bR * vi
    rhs[1:]  += aR[1:]  * vi[:-1]
    rhs[:-1] += cR[:-1] * vi[1:]
    rhs[0]   += aR[0]  * V0o - aL[0]  * V0n
    rhs[-1]  += cR[-1] * VMo - cL[-1] * VMn
    return thomas(aL, bL, cL, rhs)

def solve_cn_standard(K, T, r, sigma, Smax, M, N, opt='call'):
    """Standard CN solver."""
    dS=Smax/M; dt=T/N
    S=np.linspace(0,Smax,M+1)
    V=np.maximum(S-K,0.0) if opt=='call' else np.maximum(K-S,0.0)
    Sj=np.arange(1,M)*dS
    al=0.5*sigma**2*Sj**2/dS**2-0.5*r*Sj/dS
    be=-sigma**2*Sj**2/dS**2-r
    ga=0.5*sigma**2*Sj**2/dS**2+0.5*r*Sj/dS
    h=0.5*dt
    aL=-h*al; bL=1.0-h*be; cL=-h*ga
    aR=h*al;  bR=1.0+h*be; cR=h*ga
    a_im=-dt*al; b_im=1.0-dt*be; c_im=-dt*ga
    for step in range(N):
        tn=(step+1)*dt; to=step*dt
        V0n=0.0 if opt=='call' else K*exp(-r*tn)
        VMn=(Smax-K*exp(-r*tn)) if opt=='call' else 0.0
        V0o=0.0 if opt=='call' else K*exp(-r*to)
        VMo=(Smax-K*exp(-r*to)) if opt=='call' else 0.0
        V[0]=V0n; V[1:M]=_cn_step(V,M,aL,bL,cL,aR,bR,cR,V0n,VMn,V0o,VMo); V[M]=VMn
    return S, V

def solve_rannacher(K, T, r, sigma, Smax, M, N, opt='call', n_half_steps=2):
    """
    Conventional Rannacher time-stepping (Rannacher 1984, Giles & Carter 2006):
      Replaces the first n_half_steps // 2 Crank-Nicolson intervals
      with an even number of Backward Euler sub-steps of size dt/2.
      Conventional startup: n_half_steps = 2 replaces the first CN interval
      [0, dt] with two BE half-steps; CN then advances from t=dt through T.
      Remaining steps use Crank-Nicolson with full step dt.
    """
    dS = Smax / M
    dt = T / N
    S = np.linspace(0, Smax, M + 1)
    V = np.maximum(S - K, 0.0) if opt == 'call' else np.maximum(K - S, 0.0)
    Sj = np.arange(1, M) * dS

    al = 0.5 * sigma**2 * Sj**2 / dS**2 - 0.5 * r * Sj / dS
    be = -sigma**2 * Sj**2 / dS**2 - r
    ga = 0.5 * sigma**2 * Sj**2 / dS**2 + 0.5 * r * Sj / dS

    # CN step coefficients (dt)
    h = 0.5 * dt
    aL = -h * al; bL = 1.0 - h * be; cL = -h * ga
    aR =  h * al; bR = 1.0 + h * be; cR =  h * ga

    # Backward Euler half-step coefficients (dt / 2)
    dt_half = 0.5 * dt
    a_half = -dt_half * al; b_half = 1.0 - dt_half * be; c_half = -dt_half * ga

    # Phase 1: Backward Euler half-steps of size dt/2
    if n_half_steps < 0 or n_half_steps > 2 * N or n_half_steps % 2:
        raise ValueError("n_half_steps must be even and between 0 and 2*N")
    num_half = n_half_steps
    for s in range(num_half):
        tau = (s + 1) * dt_half
        V0 = 0.0 if opt == 'call' else K * exp(-r * tau)
        VM = (Smax - K * exp(-r * tau)) if opt == 'call' else 0.0
        V[0] = V0
        V[1:M] = _implicit_step(V, M, a_half, b_half, c_half, V0, VM)
        V[M] = VM

    # Phase 2: Crank-Nicolson steps of size dt
    start_step = num_half // 2
    for step in range(start_step, N):
        to = step * dt
        tn = (step + 1) * dt
        V0o = 0.0 if opt == 'call' else K * exp(-r * to)
        VMo = (Smax - K * exp(-r * to)) if opt == 'call' else 0.0
        V0n = 0.0 if opt == 'call' else K * exp(-r * tn)
        VMn = (Smax - K * exp(-r * tn)) if opt == 'call' else 0.0
        V[0] = V0n
        V[1:M] = _cn_step(V, M, aL, bL, cL, aR, bR, cR, V0n, VMn, V0o, VMo)
        V[M] = VMn

    return S, V

def solve_two_step_implicit_startup(K, T, r, sigma, Smax, M, N, opt='call', n_impl=2):
    """
    Two-full-step implicit startup variant:
      First n_impl steps use Backward Euler with full dt.
      Remaining steps use Crank-Nicolson with full dt.
    """
    dS = Smax / M
    dt = T / N
    S = np.linspace(0, Smax, M + 1)
    V = np.maximum(S - K, 0.0) if opt == 'call' else np.maximum(K - S, 0.0)
    Sj = np.arange(1, M) * dS

    al = 0.5 * sigma**2 * Sj**2 / dS**2 - 0.5 * r * Sj / dS
    be = -sigma**2 * Sj**2 / dS**2 - r
    ga = 0.5 * sigma**2 * Sj**2 / dS**2 + 0.5 * r * Sj / dS

    h = 0.5 * dt
    aL = -h * al; bL = 1.0 - h * be; cL = -h * ga
    aR =  h * al; bR = 1.0 + h * be; cR =  h * ga
    a_im = -dt * al; b_im = 1.0 - dt * be; c_im = -dt * ga

    for step in range(N):
        tn = (step + 1) * dt; to = step * dt
        V0n = 0.0 if opt == 'call' else K * exp(-r * tn)
        VMn = (Smax - K * exp(-r * tn)) if opt == 'call' else 0.0
        V0o = 0.0 if opt == 'call' else K * exp(-r * to)
        VMo = (Smax - K * exp(-r * to)) if opt == 'call' else 0.0

        if step < n_impl:
            V[0] = V0n; V[1:M] = _implicit_step(V, M, a_im, b_im, c_im, V0n, VMn); V[M] = VMn
        else:
            V[0] = V0n; V[1:M] = _cn_step(V, M, aL, bL, cL, aR, bR, cR, V0n, VMn, V0o, VMo); V[M] = VMn

    return S, V

def compute_gamma_grid(S, V, M):
    dS=S[1]-S[0]; gamma=np.zeros(M+1)
    for j in range(1,M): gamma[j]=(V[j+1]-2*V[j]+V[j-1])/(dS*dS)
    gamma[0]=(V[2]-2*V[1]+V[0])/(dS*dS); gamma[-1]=(V[-1]-2*V[-2]+V[-3])/(dS*dS)
    return gamma

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    K=100.0; T=1.0; r=0.05; sigma=0.20; Smax=400.0
    opt='call'

    print("=" * 72)
    print("  NUMERICAL BLACK-SCHOLES FDM -- WEEK 10")
    print("  Payoff Kink, CN Oscillations & Rannacher Smoothing")
    print("=" * 72)
    print(f"  Parameters: K={K}, T={T}, r={r}, sigma={sigma}, Smax={Smax}")
    print()

    # ─── PART 1: Payoff kink and effect on convergence ───────────────────────
    print("=" * 72)
    print("  PART 1 -- NON-SMOOTH PAYOFF: L-inf ERROR WITH AND WITHOUT KINK")
    print("=" * 72)
    print("""
  The payoff V(S,T=0) = max(S-K,0) is continuous but not differentiable
  at S=K (the 'kink').  The exact solution at maturity satisfies:

       V'(K+,0) = 1,   V'(K-,0) = 0   (discontinuous derivative)
       V''(S,0) = delta(S-K)               (distributional second derivative)

  This non-smoothness degrades the formal O(dS^2) spatial order to
  approximately O(dS^1) unless the strike K coincides with a grid node.
  When K lies exactly on a grid node, the scheme recovers O(dS^2).

  Below we compare convergence for two choices:
    (a) Unaligned grid: dS = Smax/M, K may not be a grid node
    (b) Aligned grid:   dS chosen so K is always a grid node
  """)

    M_vals=[25,50,100,200,400]
    N_large=5000

    for label, aligned in [("Unaligned grid (K may not coincide with node)", False),
                            ("Aligned grid   (K always coincides with a node)", True)]:
        print(f"  [{label}]")
        print(f"  {'M':>6}  {'dS':>8}  {'K on grid?':>12}  {'L-inf Err':>12}  {'Ratio':>8}")
        print("  " + "-" * 54)
        prev_e=None
        for M in M_vals:
            if aligned:
                # Ensure K on grid: choose M so dS divides K
                dS_try = Smax/M
                n_K = round(K/dS_try)
                dS  = K / n_K if n_K>0 else dS_try
                M_use = round(Smax/dS)
            else:
                M_use=M; dS=Smax/M
            S_g=np.linspace(0,Smax,M_use+1)
            _,V_g=solve_cn_standard(K,T,r,sigma,Smax,M_use,N_large,opt)
            errs=[]
            for j in range(1,M_use):
                if S_g[j]>0:
                    errs.append(abs(V_g[j]-bs_call(S_g[j],K,T,r,sigma)))
            e=max(errs) if errs else 0.0
            # Check if K on grid
            nodes=np.linspace(0,Smax,M_use+1)
            on_grid= any(abs(nodes-K)<1e-8)
            ratio=prev_e/e if prev_e and e>0 else float('nan')
            print(f"  {M_use:>6}  {Smax/M_use:>8.3f}  {'YES' if on_grid else 'NO':>12}  {e:>12.5e}  {ratio:>8.2f}")
            prev_e=e
        print()

    # ─── PART 2: CN oscillations at coarse time steps ────────────────────────
    print("=" * 72)
    print("  PART 2 -- CRANK-NICOLSON OSCILLATIONS FROM THE PAYOFF KINK")
    print("=" * 72)
    print("""
  For very coarse time grids (large dt), the Crank-Nicolson method can
  produce spurious oscillations near the payoff kink.  These arise because
  CN has no numerical dissipation: it preserves all Fourier modes without
  damping.  The initial discontinuity in the payoff derivative excites high-
  frequency modes that CN cannot damp.

  We demonstrate this by computing the maximum oscillation in Gamma
  (the second derivative of the option price w.r.t. S) near S=K.
  """)

    M_osc=200
    N_vals=[5, 10, 20, 50, 100, 500]

    print(f"  CN Gamma oscillation study (M={M_osc}, K={K})")
    print(f"  Gamma near S=K: analytical ATM value = {bs_gamma_val(K,K,T,r,sigma):.6f}")
    print()
    print(f"  {'N':>6}  {'dt':>9}  {'Gamma at K-dS':>14}  {'Gamma at K':>12}  {'Gamma at K+dS':>14}  {'Max|Osc|':>10}")
    print("  " + "-" * 70)

    dS_osc=Smax/M_osc
    for N_o in N_vals:
        _,V_cn=solve_cn_standard(K,T,r,sigma,Smax,M_osc,N_o,opt)
        S_cn=np.linspace(0,Smax,M_osc+1)
        gamma_cn=compute_gamma_grid(S_cn,V_cn,M_osc)
        j_K=round(K/dS_osc)
        g_km=gamma_cn[j_K-1]; g_k=gamma_cn[j_K]; g_kp=gamma_cn[j_K+1]
        osc=max(abs(g_km-g_k), abs(g_kp-g_k))
        print(f"  {N_o:>6}  {T/N_o:>9.4f}  {g_km:>14.6f}  {g_k:>12.6f}  {g_kp:>14.6f}  {osc:>10.4e}")
    print()

    # ─── PART 3: Rannacher smoothing comparison ───────────────────────────────
    print("=" * 72)
    print("  PART 3 -- CONVENTIONAL RANNACHER TIME-STEPPING (HALF-STEP BACKWARD EULER)")
    print("=" * 72)
    print("""
  Conventional Rannacher startup (Rannacher 1984, Giles & Carter 2006):
    The first Crank-Nicolson interval [0, dt] is replaced by two Backward Euler
    half-steps of size dt/2; subsequent intervals use full-step CN.
    Because Backward Euler is L-stable (R(z) -> 0 as z -> infty), high-frequency
    modes from the Dirac-delta second derivative of the payoff kink are strongly
    damped.  Subsequent steps resume Crank-Nicolson with step dt, preserving the
    global second-order O(dt^2) convergence.

  Below we compare standard CN, the conventional one-interval startup
  (two half-steps), a four-half-step extended startup, and two full BE steps.
  """)

    M_ran = 200
    dS_ran = Smax / M_ran
    j_K = round(K / dS_ran)

    print(f"  Primary Study: Standard CN vs Conventional Rannacher (2 half-steps over first interval)")
    print(f"  Grid: M={M_ran} (dS={dS_ran:.2f}), K={K:.1f}")
    print(f"\n  {'N':>6}  {'CN L-inf':>12}  {'Ran(2h) L-inf':>14}  {'CN Gamma Osc':>13}  {'Ran Gamma Osc':>14}  {'Damping':>9}")
    print("  " + "-" * 74)

    for N_r in [5, 10, 20, 50, 100, 200]:
        S_c, V_cn = solve_cn_standard(K, T, r, sigma, Smax, M_ran, N_r, opt)
        S_r, V_ran = solve_rannacher(K, T, r, sigma, Smax, M_ran, N_r, opt, n_half_steps=2)

        # L-inf
        errs_cn = [abs(V_cn[j] - bs_call(S_c[j], K, T, r, sigma)) for j in range(1, M_ran) if S_c[j] > 0]
        errs_ran = [abs(V_ran[j] - bs_call(S_r[j], K, T, r, sigma)) for j in range(1, M_ran) if S_r[j] > 0]
        e_cn = max(errs_cn); e_ran = max(errs_ran)

        # Gamma oscillation near K
        g_cn = compute_gamma_grid(S_c, V_cn, M_ran)
        g_ran = compute_gamma_grid(S_r, V_ran, M_ran)
        osc_cn = max(abs(g_cn[j_K-1] - g_cn[j_K]), abs(g_cn[j_K+1] - g_cn[j_K]))
        osc_ran = max(abs(g_ran[j_K-1] - g_ran[j_K]), abs(g_ran[j_K+1] - g_ran[j_K]))
        damp = osc_cn / osc_ran if osc_ran > 0 else float('nan')

        print(f"  {N_r:>6}  {e_cn:>12.5e}  {e_ran:>14.5e}  {osc_cn:>13.4e}  {osc_ran:>14.4e}  {damp:>8.1f}x")

    print("\n  Method Variant Comparison at N=5 (Coarsest grid, severe kink shock):")
    print("  " + "-" * 74)
    S_5, V_cn_5 = solve_cn_standard(K, T, r, sigma, Smax, M_ran, 5, opt)
    _, V_ran_5  = solve_rannacher(K, T, r, sigma, Smax, M_ran, 5, opt, n_half_steps=4)
    _, V_2h_5   = solve_rannacher(K, T, r, sigma, Smax, M_ran, 5, opt, n_half_steps=2)
    _, V_2f_5   = solve_two_step_implicit_startup(K, T, r, sigma, Smax, M_ran, 5, opt, n_impl=2)

    g_cn_5  = compute_gamma_grid(S_5, V_cn_5, M_ran)
    g_ran_5 = compute_gamma_grid(S_5, V_ran_5, M_ran)
    g_2h_5  = compute_gamma_grid(S_5, V_2h_5, M_ran)
    g_2f_5  = compute_gamma_grid(S_5, V_2f_5, M_ran)

    osc_cn_5  = max(abs(g_cn_5[j_K-1] - g_cn_5[j_K]), abs(g_cn_5[j_K+1] - g_cn_5[j_K]))
    osc_ran_5 = max(abs(g_ran_5[j_K-1] - g_ran_5[j_K]), abs(g_ran_5[j_K+1] - g_ran_5[j_K]))
    osc_2h_5  = max(abs(g_2h_5[j_K-1] - g_2h_5[j_K]), abs(g_2h_5[j_K+1] - g_2h_5[j_K]))
    osc_2f_5  = max(abs(g_2f_5[j_K-1] - g_2f_5[j_K]), abs(g_2f_5[j_K+1] - g_2f_5[j_K]))

    e_cn_5  = max([abs(V_cn_5[j] - bs_call(S_5[j], K, T, r, sigma)) for j in range(1, M_ran) if S_5[j] > 0])
    e_ran_5 = max([abs(V_ran_5[j] - bs_call(S_5[j], K, T, r, sigma)) for j in range(1, M_ran) if S_5[j] > 0])
    e_2h_5  = max([abs(V_2h_5[j] - bs_call(S_5[j], K, T, r, sigma)) for j in range(1, M_ran) if S_5[j] > 0])
    e_2f_5  = max([abs(V_2f_5[j] - bs_call(S_5[j], K, T, r, sigma)) for j in range(1, M_ran) if S_5[j] > 0])

    print(f"  Standard Crank-Nicolson        : Gamma Osc = {osc_cn_5:.5e}, L-inf Err = {e_cn_5:.5e}")
    print(f"  Rannacher (2 half-steps dt/2)  : Gamma Osc = {osc_2h_5:.5e}, L-inf Err = {e_2h_5:.5e}  (Damping: {osc_cn_5/osc_2h_5:.1f}x)")
    print(f"  Rannacher (4 half-steps dt/2)  : Gamma Osc = {osc_ran_5:.5e}, L-inf Err = {e_ran_5:.5e}  (Damping: {osc_cn_5/osc_ran_5:.1f}x)")
    print(f"  Implicit Startup (2 full steps): Gamma Osc = {osc_2f_5:.5e}, L-inf Err = {e_2f_5:.5e}  (Damping: {osc_cn_5/osc_2f_5:.1f}x)")

    # ─── PART 4: Summary ──────────────────────────────────────────────────────
    print()
    # ─── PART 4: Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  PART 4 -- SUMMARY AND METHODOLOGICAL CONCLUSIONS")
    print("=" * 72)
    print("""
  1. Non-smooth payoff (kink at S=K):
     The M=50 to M=100 refinement is alignment-sensitive (observed ratio 1.38,
     order about 0.47). Do not infer the asymptotic order from that interval.

  2. Crank-Nicolson oscillations:
     The CN scheme has no numerical dissipation (|R_CN(infty)| = 1).
     The discontinuous payoff derivative excites high-frequency spatial
     modes that CN propagates without damping.  At coarse time grids
     (N=5), this causes severe non-physical negative Gamma values
     (e.g. Gamma = -0.139 at K=100) and an oscillation amplitude of 0.246.

  3. Conventional Rannacher smoothing (half-step startup):
     Replacing the first CN interval by two Backward Euler half-steps damps
     high-frequency components. At N=5 the reproduction script measures an
     oscillation reduction from 0.24559 to 0.0022874 (107.364x) using its
     explicitly defined neighbor-difference metric. This is not a universal
     suppression factor; later CN steps begin at t=dt.

  4. Price vs Greek trade-off:
     At N=5 the reproduction script reports maximum price errors 0.239219
     for CN and 0.024480 for corrected Rannacher. This coarse-grid result
     does not establish a general price-accuracy advantage.
  """)

    print("=" * 72)
    print("  WEEK 10 STUDY COMPLETE")
    print("=" * 72)

