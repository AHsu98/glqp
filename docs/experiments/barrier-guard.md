# Barrier guard and iteration observations

`SolverSettings.barrier_update_guard="previous"` uses the barrier target from
the start of the accepted iteration in the reduction floor. With the default
`barrier_reduction_limit=0.1`, an update can reduce that target by at most tenfold:

```text
mean_product = dot(s, y) / number_of_inequalities
xi = min(s * y) / mean_product
proposal = min(1.02 - xi, 1)^3 * mean_product
next_mu = max(proposal, 0.1 * previous_mu, min_mu, 0.01 * pre_repair_KKT)
```

This floor does not prevent increases. The other floors and the centrality
proposal are unchanged. Selection still precedes slack repair; diagnostics
retain both states. Explicit `mu0` is honored, and problems without real
inequalities retain their minimum-target behavior after the first step.

`barrier_update_guard="legacy"` reproduces the former rule, whose reduction
floor referred to the new proposal and was redundant for positive proposals.
The setting exists for controlled comparisons and replay. Changing the guard
can change iteration counts; it is not an implementation-only optimization.

## Observation API

```python
settings = SolverSettings(iteration_diagnostics=True)
x, result = problem.solve(settings=settings, iteration_callback=observe)
```

Both observations are opt-in. `result.diagnostics` is otherwise `None`.
When enabled, it contains `schema_version`, an `initial` point, and
`iterations`, including failed attempts where available. Existing history
columns retain their meaning; additional scalar columns start with `diag_`.

Each accepted iteration records:

- Previous target, raw proposal, mean complementarity, centrality, every
  candidate floor, and all exact ties at the selected maximum.
- Step bound, accepted fraction, trial/rejection counts, and the branch that
  accepted the step: Armijo, residual decrease, or finite Newton fallback.
- Direction norms and clipping counts; original Newton equation defects before
  and after clipping, the edit-induced defects, and the expected contribution
  of requested regularization. These differ from the verified linear residual.
- Slack-repair counts and changes to slack, primal residual, and complementarity,
  with pre/post-repair points. The pre-repair point has the preceding target;
  the post-repair point has the newly selected target.
- Primal/equality/stationarity residuals, stored and true-slack complementarity,
  positivity/centrality, and the maximum multiplier-to-slack ratio.

`observe(snapshot)` runs once at initialization (`iteration=0`, `stage="initial"`)
and after each accepted, repaired iterate (`stage="accepted"`). The frozen
`IterationSnapshot` exposes `x`, inequality multipliers `y`, stored slack `s`,
equality multipliers `nu`, objective, target, KKT norm, relative true-slack
complementarity, and native convergence flags. Arrays are borrowed read-only
views valid during the callback; copy them to retain observations. Do not mutate
their bases, the model, or settings. Return values are ignored and exceptions
propagate. Artificial inequality entries remain present when the model has no
real inequalities, so external validators must slice to the original dimensions.

Diagnostics perform extra matrix-vector products and allocate observations.
Callbacks and diagnostics count toward the solver time budget. Use them for
untimed algorithm analysis, and disable both for performance comparisons.
The benchmark repository's `tools/trace_barrier_updates.py` additionally applies
its independent original-QP certificate at every observed iterate and reports
the first passing iterate separately from native termination. It records a
larger diagnostic time budget explicitly and compares the final arrays/counts
against the corresponding plain run.

Predictor–corrector, initialization, residual-floor coefficient, clipping,
slack repair, line-search acceptance, and regularization are unchanged in this
experiment. They should be tested separately to keep effects attributable.
