# Opt-in predictor-corrector experiment

Use `SolverSettings(step_strategy="predictor_corrector")` to select the
experimental direction and barrier policy. The accepted default is `"newton"`.
The full derivation and GLM/proximal caveats are in the
[design note](../research/predictor-corrector-design-20261016.md).

For real inequalities, each iteration:

1. Assemble the current regularized Newton matrix once, using the current
   objective Hessian, slacks, multipliers, and outer regularization.
2. Solve the affine system with complementarity target zero. Estimate mean
   complementarity at its common positivity-limited affine step and set
   `sigma = clip(mu_affine / mean(s*y), 0, 1)**3`.
3. Select `mu = max(sigma*mean(s*y), min_mu, barrier_kkt_floor*KKT)` and solve
   the combined system using complementarity residual
   `s*y + ds_affine*dy_affine - mu`. Its other RHS blocks are unchanged.
4. Apply the existing direction editing and fraction-to-boundary rule, then
   line-search the combined direction using this same target. Evaluate the
   actual GLM objective and derivatives where required, not a quadratic proxy.

The combined solve already includes the affine contribution; it is not added
to the affine direction again. Both RHSs use bounded float64 right-GMRES and
fresh residuals against the requested matrix. A synchronous
`VerifiedLinearSystem` context reuses the numerical factor without updating
it for the second RHS. The factorization-only shift remains excluded from
the residual target. Numerical failure on factor reuse is explicit; no
silent refactor changes the matrix between these solves.

Absent/dummy or empty inequalities use the original single-direction path.
The old initial-target policy has no role in a real-inequality PC step:
without explicit `mu0`, initial diagnostics mark a minimum-target placeholder
as `awaiting_predictor`. An explicit `mu0` is honored as the **first combined
target**; later targets come from predictors. Explicit PC `mu0` must be finite
and nonnegative. `barrier_update_guard` and `barrier_reduction_limit` belong
to the original post-step policy and do not apply to PC. The target stays
fixed after an accepted step until the next predictor selects a new one.

`barrier_kkt_floor` remains independently configurable. The declared first
experiment compares the accepted method with PC at coefficients 0.01 and
zero. It keeps initialization, outer regularization, direction editing,
slack repair, line-search rules, and stopping tolerances otherwise fixed.

History records both refinement counts, affine step, sigma, selected target,
both linear statuses, and factorization attempts. With
`iteration_diagnostics=True`, the `predictor_corrector` record contains the
pre-direction target decision and affine Newton defects. The post-step
`barrier` record says the target is held. Combined-direction original Newton
defects include the intentional corrector term; the requested-system defect
subtracts both that term and the regularization effect. This prevents a
corrector contribution from being mislabeled as linear-solve error.

This is a regularized Newton experiment, not a newly implemented proximal-
point method. The existing shifts affect complementarity as well as primal
and dual equations. A local fixed-center interpretation is possible, but
the current line search does not evaluate that proximal subproblem's merit
and the solver does not maintain a separate proximal-center state. Both
solves freeze all those shifts. For nonlinear GLMs, complementarity-only
correction omits the nonlinear stationarity Taylor correction; retaining the
actual objective line search and independently validating GLM solutions are
essential checks, not a general convergence guarantee.
