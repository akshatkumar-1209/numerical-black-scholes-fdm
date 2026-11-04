# Derivation 7: Numerical Greeks?Delta and Gamma

## Analytical benchmarks
For spot S>0 and time to maturity T>0, let `d1=[ln(S/K)+(r+sigma^2/2)T]/(sigma sqrt(T))` and `d2=d1-sigma sqrt(T)`. Then `Delta_call=N(d1)`, `Delta_put=N(d1)-1`, and for either option `Gamma=n(d1)/(S sigma sqrt(T))`.

To locate the Gamma maximum, differentiate its logarithm with respect to `x=ln(S/K)`. Since `d1=(x+(r+sigma^2/2)T)/(sigma sqrt(T))`,
`d log Gamma/dx = -d1/(sigma sqrt(T))-1`. The stationary point satisfies `d1=-sigma sqrt(T)`, hence
`S*=K exp[-(r+3 sigma^2/2)T]`.
This is generally below K for positive r and sigma. For the benchmark parameters, S*?89.58 and Gamma(S*)?0.02183 at T=1; at T=0.1, S*?98.91. ATM Gamma at K is 0.01876202 (T=1) and 0.06269314 (T=0.1); these are distinct quantities.

## Grid formulas
On `S_j=j dS`, central differences are
`Delta_j=(V_{j+1}-V_{j-1})/(2 dS)` and
`Gamma_j=(V_{j+1}-2V_j+V_{j-1})/(dS)^2`.
Taylor expansion gives second-order truncation error where the solution is smooth. The payoff is nonsmooth at expiry, and finite-grid error also depends on strike alignment, boundary truncation, time error and the evaluated node; formal stencil order alone does not guarantee an observed rate for every pair of meshes.

## Verified benchmark results
For CN with M=400, N=2000 and T=1, at the aligned strike node S=K=100, call Delta is 0.63676829 (analytical 0.63683065; absolute error 6.2359e-5), and call Gamma is 0.01876927 (analytical 0.01876202; error 7.2484e-6). Put Delta is -0.36323171 (analytical -0.36316935; same error); call and put Gamma coincide to the printed precision. At S=90, the exact Gamma is 0.02181975, illustrating that the Gamma peak is away from K. At T=0.1, ATM Gamma is 0.06269314; do not conflate this with the global maximum.

For the M=50,100,200,400 mesh study, Delta error ratios are approximately 4.66, 4.04, 4.01. Gamma ratios are approximately 0.71, 4.07, 4.02: the first interval is irregular (M=50 has dS=8 and K lies between nodes). The data do not establish second order on every interval. See `../results/output_9_numerical_greeks.txt` for full call/put tables and maturity results.

## Interpretation
Greeks differentiate the discrete price profile and amplify local price errors, especially Gamma. Results must identify the maturity, solver, mesh, strike alignment and sampling location. ATM Gamma is the value at S=K; maximum Gamma is evaluated at S* above.
