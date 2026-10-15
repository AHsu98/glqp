# Verified linear solves

`glqp.linear.factor_and_solve` owns numerical factorization, bounded refinement,
and residual verification. The requested matrix already includes the outer
solver's proximal perturbation. Refinement targets this requested matrix; it
removes numerical factorization error without removing intended proximal damping.

The QDLDL adapter exposes three operations: `matches(matrix)`, `update(matrix)`,
and `solve(rhs)`. Its constructor performs the initial factorization. Sparse
pattern changes rebuild the factors. Another backend can implement these same
operations without duplicating refinement and reporting. Backend selection and
additional solver implementations are deferred.

The numerical shift `tau_reg` is separate from `SolverSettings.prox_reg`.
Only QDLDL numerical failures or nonfinite initial backend solutions trigger a
retry with a tenfold numerical shift. A failed update invalidates the cached
factorization. Zero initial shift disables escalation. Timeouts, input errors,
and unexpected Python errors propagate rather than being retried as numerical
failures; GLQP only labels `LinearSolveError` as a linear-solve termination.

## Refinement and accuracy

The initial direction and corrections retain the residual-minimizing scalar
projection. Every trial correction recomputes `rhs - G @ trial_solution` after
floating-point addition. A correction that fails to improve the relative L2
residual, and does not meet the requested accuracy, is discarded and refinement
stops. Nonfinite or unusable corrections also leave the best finite iterate
intact. The configured refinement limit includes every attempted correction;
zero means no corrections.

One accuracy rule applies throughout: the absolute infinity-norm residual must
be at most `target_atol`, **or** the relative L2 residual must be at most
`target_rtol` (default 1e-6). This is the previous final acceptance rule. The
previous extra first correction even after that rule was satisfied is removed.
These are linear-system tolerances, not a claim of nonlinear or QP optimality.

`LinearSolveResult` provides `solution`, `factorization`, and `info`. For existing
callers, four-value unpacking and indexing still return the solution, attempted
refinement count, cached factorization, and freshly evaluated relative error.
`info.status` is explicitly `accurate` or `inaccurate`; `stop_reason` distinguishes
tolerance, refinement limit, stagnation, unusable/nonfinite corrections, and a
zero right-hand side. It also records the actual shift, factorization attempts,
absolute error, requested tolerances, and equation-block errors.

A finite inaccurate direction remains eligible for GLQP's existing line search.
This preserves inexact-step behavior while making its accuracy visible. There
is no automatic increase in proximal regularization in this change, and neither
the original-problem stopping tests nor independent benchmark validation are
relaxed. An unusable initial direction raises a numerical error.

## Diagnostics

The iteration history retains `refine` and `lin_rel_res`, and adds `lin_status`,
`lin_stop`, `lin_abs_res`, `lin_atol`, `lin_rtol`, `lin_tau`, and `lin_attempts`.
Block fields report absolute infinity-norm error and error divided by
`max(1, ||rhs_block||_inf)`. Blocks are `stationarity`, `scaled_inequality`, and
`equality` in the actual assembled system. The middle block is symmetrically
scaled; its error is not the original primal feasibility residual.

`SolverResults.initialization_linear_solve` records the equality initialization
solve, if one was needed. `last_linear_solve` records the last returned direction,
including one rejected by the outer line search. Per-iteration fields describe
the linear solve before boundary-direction clipping; clipping and slack repair
are separate operations. Original KKT residuals remain in the existing history.

Tests exercise fresh residual reporting on an ill-conditioned matrix, rounding
that worsens a correction, bounded unproductive refinement, inaccurate status,
sparse-factor reuse, tiny and zero right-hand sides, and exception propagation.
Performance is measured separately against a committed proximal-on reference.
No presolve, alternate backend, proximal adaptation, or nonlinear model changes
are included.
