# Numerical Solution of the Black-Scholes PDE by Finite Differences

**Author:** Akshat Kumar  
**Supervisor:** Dr. Ashok Kumar Vaikuntam

## Abstract
This project implements fully implicit (Backward Euler) and Crank-Nicolson finite-difference solvers for European options, solves the tridiagonal systems with the Thomas algorithm, and compares prices and Greeks with closed-form Black-Scholes values. Controlled mesh studies, a separate combined refinement matrix, and a payoff-kink experiment are reproduced by `reproduce_all_results.py`. Results below are taken from the regenerated files in `results/` and are limited to the stated meshes and error metrics.

## 1. Introduction and motivation
The Black-Scholes PDE provides a tractable benchmark for numerical methods used in option pricing. Finite differences are transparent and adaptable, but their reported accuracy depends on spatial and temporal errors, boundaries, payoff regularity, and mesh alignment. The purpose here is implementation and numerical verification, not a claim of a new pricing method.

## 2. Mathematical formulation
Let `tau=T-t` denote time to maturity and write `V(S,tau)` for the option value. The transformed equation is
`V_tau = (sigma^2 S^2/2)V_SS + r S V_S - r V`, for `0<S<Smax`, `0<tau<=T`.
At `tau=0`, call payoff is `max(S-K,0)` and put payoff is `max(K-S,0)`. For a call, boundaries are `V(0,tau)=0`, `V(Smax,tau)=Smax-K exp(-r tau)`; for a put, `V(0,tau)=K exp(-r tau)`, `V(Smax,tau)=0`. A finite Smax truncates the domain and contributes boundary error.

Benchmark parameters are `K=100, T=1, r=0.05, sigma=0.20, Smax=400`. The analytical formulas use `d1=[ln(S/K)+(r+sigma^2/2)T]/(sigma sqrt(T))`, `d2=d1-sigma sqrt(T)`, `C=S N(d1)-K exp(-rT)N(d2)`, and `P=K exp(-rT)N(-d2)-S N(-d1)`.

## 3. Spatial and time discretisation
Use `S_j=j dS`, `dS=Smax/M`, and interior indices `j=1,...,M-1`. Define
`alpha_j=sigma^2 S_j^2/(2dS^2)-rS_j/(2dS)`, `beta_j=-sigma^2 S_j^2/dS^2-r`, and `gamma_j=sigma^2 S_j^2/(2dS^2)+rS_j/(2dS)`. Thus the centered semi-discrete operator is `L_h V_j=alpha_j V_{j-1}+beta_j V_j+gamma_j V_{j+1}`. Coefficients depend on j; drift can make alpha negative on some grids.

### 3.1 Fully Implicit (Backward Euler)
For step `dt`, `(I-dt L_h)V^{n+1}=V^n`. The matrix has sub-, main- and super-diagonals `-dt alpha`, `1-dt beta`, `-dt gamma`. Known boundary contributions are subtracted from the adjacent RHS entries at the new time. The method is first order in time and second order in space for smooth solutions; payoff regularity, boundaries, and mesh alignment affect observed errors.

### 3.2 Crank-Nicolson
`(I-dt L_h/2)V^{n+1}=(I+dt L_h/2)V^n`. The left matrix is tridiagonal; the old-time boundary contributions enter the right operator and the new-time contributions are moved to the RHS. The method is formally second order in time and second order in space for sufficiently regular solutions. CN is A-stable for the scalar diffusion test equation but not L-stable; payoff nonsmoothness can produce early high-frequency oscillations.

### 3.3 Thomas algorithm
For subdiagonal a, diagonal b, superdiagonal c and RHS d, forward elimination uses `m_i=a_i/b'_(i-1)`, `b'_i=b_i-m_i c_(i-1)`, `d'_i=d_i-m_i d'_(i-1)`. Back substitution is `x_n=d'_n/b'_n`, `x_i=(d'_i-c_i x_(i+1))/b'_i`. Indexing is over the interior system; boundary terms are added to the RHS before solving. The implementation is O(M) in work and storage. It requires nonzero pivots; diagonal dominance is a sufficient condition in common cases, not a blanket assertion for every grid. Solver stability/conditioning is distinct from time-scheme stability. The matrix is inspected through its coefficients; no general inference that the Thomas algorithm implies unconditional stability is made.

## 4. Analytical benchmark and parity
For M=200, fully implicit N=1000 has maximum interior call/put errors `1.100888e-2` and `1.088998e-2`; its measured maximum parity residual is `1.191934e-4`. CN M=200,N=500 has call/put errors `1.000748e-2` each and maximum parity residual `3.971806e-9`. The finite implicit residual is not machine precision. At ATM, analytical call and put prices are 10.45058357 and 5.573526 at the stated parameters. Reported errors use the experiment's maximum over interior nodes, whereas parity residual includes the full grid.

## 5. Spatial convergence
The controlled study fixes N=5000 and varies M. The signed observed-order convention is `p=log(E_coarse/E_fine)/log(h_coarse/h_fine)`, with `h=dS`. For M=100, 200, 400, implicit errors are `4.01874e-2,1.02085e-2,2.69818e-3` (ratios 3.94,3.78; p=1.98,1.92); CN errors are `4.00021e-2,1.00084e-2,2.49862e-3` (ratios 4.00,4.01; p approximately 2.00). M=50 to 100 is anomalous: K is between nodes at M=50 and aligned at M=100, and the observed error ratio is only 1.38. The whole sequence does not establish a uniform second-order rate for each interval.

The separate 4x4 error matrix (`results/output_11_grid_refinement_matrix.txt`) varies M and N simultaneously. It is descriptive and does not independently establish spatial or temporal order.

## 6. Temporal convergence
Fix M=200 and compare each N solution with a time-refined reference on the same spatial grid, `Nref=12800`. Refining the reference to 25600 changes the implicit solution by `4.197888e-5` and CN by `1.160432e-9` in the reported interior L-infinity norm. This supports the chosen reference for the tested tolerances, but does not make it exact. For N=50,100,200,400,800, implicit errors are approximately `2.13813e-2,1.06557e-2,5.28764e-3,2.60228e-3,1.25927e-3`, with ratios near 2 (orders 1.00 to 1.05). CN errors are `1.00969e-4,2.52436e-5,6.30988e-6,1.57632e-6,3.92923e-7`, with ratios near 4 (orders approximately 2). See `results/output_8_convergence_study.txt` for precision and the separately printed observed orders.

## 7. Numerical Delta and Gamma
The analytical Greeks are `Delta_call=N(d1)`, `Delta_put=N(d1)-1`, and `Gamma=n(d1)/(S sigma sqrt(T))`. The code uses central differences `(V[j+1]-V[j-1])/(2dS)` and `(V[j+1]-2V[j]+V[j-1])/dS^2`. At CN M=400,N=2000,T=1, S=K=100: call Delta 0.63676829 vs 0.63683065 (error `6.2359e-5`), put Delta -0.36323171 vs -0.36316935 (same error), and Gamma 0.01876927 vs 0.01876202 (error `7.2484e-6`). These are strike values, not global maxima.

Differentiating log Gamma shows its maximum occurs at `S*=K exp[-(r+3sigma^2/2)T]`, about 89.58 for T=1 and 98.91 for T=0.1. ATM Gamma is 0.01876202 at T=1 and 0.06269314 at T=0.1; at T=1, Gamma near S=90 is 0.02181975. Mesh study evidence is alignment sensitive: the Gamma error ratio M=50 to 100 is about 0.71, then about 4.07 and 4.02. Do not assert second order on every interval. Full call/put, off-strike and short-maturity output is in `results/output_9_numerical_greeks.txt`.

## 8. Payoff kink, CN oscillations and corrected Rannacher startup
The first nominal CN interval `[0,dt]` is replaced by two Backward Euler half-steps of `dt/2`; boundary data are evaluated at `dt/2` and `dt`. The remaining N-1 intervals use full CN. Total elapsed time is exactly T. This is the conventional one-interval Rannacher startup implemented in the experiment; the older two-full-step variant is separately labelled.

At M=200,N=5, the recorded adjacent-Gamma oscillation measure is 0.245586 for CN and 0.00228742 for Rannacher, suppression 107.4x. Maximum price errors are 0.239219 and 0.0244796. At N=20, the Rannacher price error (0.0106741) is slightly above CN (0.00944552); by N>=50 the oscillation measures are essentially equal. The suppression depends on N and the defined metric; it is not a universal factor. Full N table and variants are in `results/output_10_payoff_kink_oscillation.txt`.

## 9. Limitations and future work
The domain is truncated at finite Smax and the spatial grid is uniform. The study covers European vanilla options under constant parameters. It does not establish performance for discontinuous digitals, barriers, dividends, stochastic volatility, or calibrated market data. Future work could quantify boundary/domain error, use nonuniform grids, add independent solver residual and pivot diagnostics, compare with finite elements, and test broader payoff regularity and parameters.

## 10. Conclusion
The regenerated controlled experiments support the expected spatial and temporal behavior on the stated aligned/refined ranges, while revealing alignment-sensitive coarse meshes. Benchmark prices and parity defects are measured rather than described qualitatively. The corrected half-step Rannacher startup suppresses the chosen coarse-grid Gamma metric substantially, with mesh-dependent impact on price error. All claims are bounded by the tested configurations and are reproducible with the repository script.

## References

1. Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.
2. Rannacher, R. (1984). Finite element solution of diffusion problems with irregular data. *Numerische Mathematik*, 43, 309-327.
3. Duffy, D. J. (2022). *Numerical Methods in Computational Finance*. Wiley.
4. Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance*, 2nd ed. Wiley.

## Reproducibility
Run `python reproduce_all_results.py` from the repository root. It regenerates the combined refinement output and checks benchmark errors, parity, reference convergence, controlled spatial/temporal refinement, Greeks and Rannacher startup. Experiment-specific output files identify their generating programs. The verified run completed with `ALL ASSERTIONS PASSED`; assertions are verification bounds, not proofs of general convergence.
