# Derivation 6: Stability and Convergence Analysis

## 1. Error and observed order
For errors `E_c,E_f` at coarse/fine spacings `h_c,h_f`, observed order is
`p=log(E_c/E_f)/log(h_c/h_f)`. When `h_f=h_c/2`, a factor-four error reduction gives p?2; factor two gives p?1. A negative order indicates that the error increased under refinement and should not be hidden by an absolute value.

## 2. Spatial controlled experiment
`programs/convergence_study.py` fixes N=5000 and varies M. The generated output reports the actual errors, ratios, and signed orders. For M=100?200?400, CN ratios are 4.00 and 4.01 (orders ?2); fully implicit ratios are 3.94 and 3.78 (orders ?1.98 and 1.92). M=50?100 ratio is about 1.38 because the strike is not on the M=50 grid, while it is aligned at M=100. This interval is not evidence of a stable asymptotic order.

## 3. Temporal controlled experiment
Fix M=200 and compare against a high-time-resolution reference on that same spatial grid, Nref=12800. The regenerated check against 2Nref gives interior maximum changes 4.197888e-5 for fully implicit and 1.160432e-9 for CN. For tested N values, implicit temporal ratios approach 2 and CN ratios approach 4. The Nref solution is a numerical reference; its finite refinement difference is reported rather than treated as zero error.

## 4. Combined refinement matrix
The 4?4 matrix varies M and N together. Its entries illustrate mixed spatial and temporal effects but cannot independently establish either order. Use the controlled studies above for order claims.

## 5. Stability and linear systems
Backward Euler and CN have A-stable scalar test-equation stability functions; backward Euler is L-stable, while CN is not. These statements do not alone prove stability of every variable-coefficient finite-domain discretisation. The tridiagonal left matrix must be assessed through its actual coefficients, pivots, and boundary adjustments. Strict diagonal dominance can imply nonsingularity and supports no-pivot elimination in applicable cases, but it is not interchangeable with time-discretisation stability. Thomas solver conditioning and roundoff are separate numerical issues.

## 6. Reproducible results
Run `python reproduce_all_results.py` for assertions on analytical benchmark errors, parity residual, same-grid reference convergence, controlled refinement, Greeks and Rannacher. Detailed tables are in `../results/output_8_convergence_study.txt` and `../results/output_11_grid_refinement_matrix.txt`.
