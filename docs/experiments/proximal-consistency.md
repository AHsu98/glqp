# Centered proximal equations and predictor–corrector

This is an experimental option. Keep `proximal_policy="legacy"` and accepted
Newton defaults unchanged. Final numerical source `1f5ac45` passes all 39
GLMs with the centered method, but only 7/8 selected difficult QPs versus
existing PC's 8/8. QPCBOEI2 improves sharply; QSHELL slows and LISWET1 fails.

```python
settings = SolverSettings(
    step_strategy="predictor_corrector",
    proximal_policy="centered",
    proximal_reg_update="progress",
    proximal_corrector_limit=.5,
    prox_reg=2e-8,
    prox_dual_reg=None,
)
```

The physical residuals include `rho*(x-x_center)` in stationarity and
`-delta*(multiplier-multiplier_center)` in feasibility. Complementarity
remains `s*y-target`, without a slack proximal perturbation. Persistent
centers and coefficients remain fixed within the PC pair and line search.
The current GLM Hessian remains fixed within the pair; each line-search
trial evaluates the actual nonlinear gradient. Factorization-only `tau_reg`
is separate and does not enter these residuals.

`proximal_policy="no_slack"` supplies a block-removal control with the old
residuals/globalization. `proximal_reg_update="residual"` supplies a centered
control without adaptive coefficient continuation. The independent dual cap
is available in both nonlegacy policies; `None` inherits the primal cap.

The centered line search uses a uniformly scaled squared residual norm,
strict Armijo, and halving backtracking. A bound on the affine cross term
limits it to `.5` times the current residual norm by default. `None` disables
that bound. Actual insufficient descent after tiny-component clipping can
request a same-target centering RHS using the existing factorization.
Diagnostics account for every solve, including discarded correctors.
Slack repair must also respect the fixed-center merit.

Centers advance on a 5% decrease in the matching original residual, or
when the fixed-target proximal residual reaches `.1*max(target,tol)`.
The `progress` policy independently reduces a coefficient cap by ten after
inner completion, or after five missed progress gates when its center
offset is at least 10% of the original block residual (with a tolerance
floor). Changes apply only to the next frozen model. These are experimental
heuristics; no global convergence theorem is claimed.

Final parallel-8 measurements use native tolerance `1e-7`, original-data
certificate `1e-6`, 60 seconds, and the established QP benchmark stagnation
window of 5000. Adaptive centered PC uses 173 rather than 3241 iterations
on QPCBOEI2, 320 rather than 534 on YAO, and 596 rather than 71 on QSHELL.
LISWET1 fails after 482 accepted iterations. Conditional on the seven shared
successes, geometric wall time is 11.7% lower; the lost case prevents default
promotion. The 39 GLMs require 275 total iterations versus 280 for existing
PC. Removing the barrier floor separately gives 241; the earlier existing-PC
floor-off control already used 245. GLM timings remain excluded after host
activity flags and one whole-wave replay.

LISWET1 exposes an accuracy/continuation interaction: after dual-cap
reductions, ten-step GMRES refinement no longer produces a descent direction,
even with a centering rescue. The original complementarity certificate
correctly fails despite a small stored scalar KKT norm. Future work should
couple continuation to requested/original-coordinate direction accuracy,
rather than weaken globalization or stopping tests.

The full [benchmark checkpoint](../../../glqp-bench/benchmarks/progression/proximal-consistency-20261017/README.md)
contains clean QP profiles, all observations, rejected probes, and raw archive
hashes. The [design review](../research/proximal-consistency-design-20261017.md)
gives the residual/Jacobian derivation and literature context. The sibling
benchmark link assumes the standard workspace layout.
