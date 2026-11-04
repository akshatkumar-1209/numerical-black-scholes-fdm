# Derivation 8: Payoff kink, Crank?Nicolson oscillations and Rannacher startup

## Payoff and damping
The call terminal data `max(S-K,0)` is continuous but not C1 at K. Its second derivative is distributional at expiry; diffusion smooths it for positive time, but high-frequency components can affect early discrete steps. Crank?Nicolson is A-stable but not L-stable: its amplification tends to -1 for strongly damped spatial modes, so these modes need not be attenuated. Backward Euler is L-stable and damps them.

## Conventional startup and time accounting
For N nominal time intervals of length `dt=T/N`, conventional one-interval Rannacher startup replaces the first interval [0,dt] by two Backward Euler half-steps of length `dt/2`. Boundary conditions are imposed at `dt/2` and `dt`. Then N-1 full CN steps advance from `dt` to T. Total elapsed time is `2(dt/2)+(N-1)dt=T`; no interval is omitted or duplicated. Each call/put boundary is evaluated at the corresponding new time level.

## Regenerated comparison (M=200, K=100, T=1)
Oscillation is the largest adjacent difference among the three Gamma samples at K?dS, K, K+dS, as implemented in `programs/payoff_kink_cn_oscillation.py`.

| N | CN price L? | Rannacher price L? | CN oscillation | Rannacher oscillation | suppression |
|--:|--:|--:|--:|--:|--:|
|5|2.39219e-1|2.44796e-2|2.4559e-1|2.2874e-3|107.4x|
|10|3.39422e-2|1.28997e-2|4.0316e-2|1.5217e-3|26.5?|
|20|9.44552e-3|1.06741e-2|1.1405e-3|1.0817e-3|1.1?|
|50|9.91465e-3|1.01147e-2|1.0841e-3|1.0842e-3|?1?|
|100|9.98497e-3|1.00350e-2|1.0842e-3|1.0842e-3|?1?|
|200|1.00026e-2|1.00151e-2|1.0842e-3|1.0842e-3|?1?|

At N=5, the startup suppresses the stated metric by 107.4x; this is not a universal rate or complete elimination. At larger N the metric is dominated by the smooth Gamma curvature over one spatial cell and the methods nearly coincide. Price error is not uniformly improved: at N=20 the Rannacher value is slightly larger.

The experiment also reports a four-half-step variant and the historical two-full-step startup separately. Neither replaces the conventional result above. Full values are in `../results/output_10_payoff_kink_oscillation.txt`.
