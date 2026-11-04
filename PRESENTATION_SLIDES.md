# Presentation: Numerical Black-Scholes PDE by Finite Differences

**Akshat Kumar | Supervisor: Dr. Ashok Kumar Vaikuntam**

## 1. Motivation and objective
Compare finite-difference pricing for European calls and puts with the analytical Black-Scholes solution; assess convergence and finite-difference Greeks; investigate the payoff kink and corrected Rannacher startup.

## 2. PDE and conditions
With time to maturity tau=T-t, V_tau=(sigma^2*S^2/2)*V_SS+r*S*V_S-r*V. Initial data are call max(S-K,0) and put max(K-S,0). Boundaries: call V(0,tau)=0, V(Smax,tau)=Smax-K*exp(-r*tau); put V(0,tau)=K*exp(-r*tau), V(Smax,tau)=0.

## 3. Spatial operator and solvers
Centered differences give L_h V_j=alpha_j V_(j-1)+beta_j V_j+gamma_j V_(j+1), with alpha=sigma^2*S_j^2/(2*dS^2)-r*S_j/(2*dS), beta=-sigma^2*S_j^2/dS^2-r, gamma=sigma^2*S_j^2/(2*dS^2)+r*S_j/(2*dS). Backward Euler uses (I-dt L_h)V^(n+1)=V^n; CN uses (I-dt L_h/2)V^(n+1)=(I+dt L_h/2)V^n. Thomas is an O(M) tridiagonal solver; algebraic pivot and conditioning properties do not establish time-scheme stability. BE is A-stable and L-stable; CN is A-stable but not L-stable.

## 4. Verification benchmark
K=100, T=1, r=0.05, sigma=0.20, Smax=400. At M=200, BE N=1000 call/put maximum interior errors are 1.100888e-2 and 1.088998e-2; max parity residual is 1.191934e-4. CN N=500 errors are 1.000748e-2 each; parity residual is 3.971806e-9. These measured residuals are finite.

## 5. Controlled convergence
Spatial, N=5000: for M=100, 200, 400, BE errors `4.01874e-2,1.02085e-2,2.69818e-3` (ratios 3.94,3.78); CN `4.00021e-2,1.00084e-2,2.49862e-3` (ratios 4.00,4.01). M=50 to 100 is irregular (ratio 1.38) because K is unaligned at M=50. Temporal, M=200 against same-grid Nref=12800: BE ratios approach 2 and CN ratios approach 4. Reference changes at Nref=25600 are 4.197888e-5 (BE) and 1.160432e-9 (CN). The 4x4 matrix is a separate combined refinement experiment and does not establish either order independently.

## 6. Greeks
`Delta_call=N(d1)`, `Delta_put=N(d1)-1`; `Gamma=n(d1)/(S sigma sqrt(T))`. Central grid differences are second-order stencils in smooth regions. At M=400,N=2000,T=1,S=K: CN call Delta 0.63676829 vs 0.63683065 (error 6.2359e-5), put Delta -0.36323171 vs -0.36316935, Gamma 0.01876927 vs 0.01876202 (error 7.2484e-6). ATM Gamma differs from its global maximum. `S*=K exp[-(r+3*sigma^2/2)T]`; S* approximately 89.58 for T=1, 98.91 for T=0.1. ATM Gamma is 0.01876202 and 0.06269314 respectively. M=50 is strike-unaligned; observed Gamma refinement is not second order in every interval.

## 7. Kink and conventional Rannacher startup
Replace first interval [0,dt] by two BE half-steps dt/2, enforcing boundaries at both intermediate times, then take N-1 full CN intervals. At M=200,N=5, adjacent-Gamma oscillation metric is 0.245586 for CN and 0.00228742 for startup (107.4x); max price errors are 0.239219 and 0.0244796. At N=20 price error is 0.00944552 (CN) versus 0.0106741 (startup); at N>=50 the oscillation metrics converge to the same spatial-curvature scale. No universal suppression or price advantage is claimed.

## 8. Limitations and conclusion
Uniform truncated spatial domain, European vanilla payoff and constant parameters. Results support expected orders only over specified controlled ranges. Extend with domain-error analysis, nonuniform grids, broader payoffs and independent conditioning diagnostics. See `TECHNICAL_REPORT.md` and generated files in `results/` for full methods, tables, and references.
