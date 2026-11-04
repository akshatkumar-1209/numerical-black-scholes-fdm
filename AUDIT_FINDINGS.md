# Repository audit findings (2026-11-04)

## Scope and baseline
The canonical GitHub checkout was inspected before edits and preserved in a separate local clone on branch `audit/black-scholes-corrections`; the original flattened workspace was checkpointed separately. The tracked project includes Python FDM solvers/experiments, eight derivations, generated result text, figures, README, technical report and presentation outline.

## Findings and corrections

1. **Convergence reporting.** Observed order is defined as `log(E_coarse/E_fine)/log(h_coarse/h_fine)`. Controlled spatial and same-grid temporal studies are distinct from the combined 4?4 matrix. M=50?100 spatial ratios are alignment-sensitive and do not establish an order. The retained temporal reference M=200, Nref=12800 was checked against Nref=25600.
2. **Rannacher startup.** The former documented ?Rannacher? implementation used two full BE steps. The corrected conventional startup replaces [0,dt] with two BE steps of dt/2, applies boundary values at each half-time, then takes N?1 full CN intervals. At M=200,N=5 the measured oscillation drops from 0.245586 to 0.00228742 (107.4?); max price error changes from 0.239219 to 0.0244796.
3. **Greeks.** Central Delta/Gamma stencils are correct. Analytical Gamma's maximum is not generally at K; its stationary maximum is at `S*=K exp[-(r+3 sigma^2/2)T]`. ATM and peak Gamma, maturity, call/put Delta, and strike/off-strike errors are identified distinctly. M=50 is unaligned; observed error ratios do not support a blanket second-order claim.
4. **Linear algebra/stability.** Thomas elimination/back substitution is reviewed separately from time-discretisation stability. Claims equating strict diagonal dominance or use of Thomas with unconditional stability are removed; coefficient sign conditions depend on grid/parameters.
5. **Benchmark/parity.** Measured benchmark errors and maximum parity residuals are reported for each solver/configuration. Fully implicit parity defect is finite, not machine precision.
6. **Documentation and reproducibility.** Stale experiment claims, broken filenames, and combined-matrix order claims are being reconciled with regenerated outputs; the end-to-end script includes reference convergence and finite checks.

## Remaining scope note
Local XLSX trackers and PDF planning/reference documents in the original workspace are not tracked in the canonical repository and are not edited as project source deliverables.
