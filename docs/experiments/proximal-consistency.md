# Centered proximal equations with Newton and predictor–corrector

This is an experimental option. Keep `proximal_policy="legacy"` and accepted
Newton defaults unchanged. Numerical source `1f5ac45` supports centered Newton
and centered predictor–corrector (PC). Both pass all 39 GLMs and 7/8 selected
difficult QPs. The later full centered-Newton run validates 127/138 MM problems
versus accepted Newton's 129/138; it is slower on the clean timing subset.
See the [experiment ledger](README.md) for the development sequence and decisions.

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

Set `step_strategy="newton"` to test the same centered formulation and
coefficient continuation without PC. That path requests one algorithmic RHS
per iteration and does not use PC's cross-term bound or centering rescue.

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

## Predictor–corrector screen

The parallel-8 PC measurements use native tolerance `1e-7`, original-data
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

The [PC benchmark checkpoint](../../../glqp-bench/benchmarks/progression/proximal-consistency-20261017/README.md)
contains clean QP profiles, all observations, rejected probes, and raw archive
hashes. The [design review](../research/proximal-consistency-design-20261017.md)
gives the residual/Jacobian derivation and literature context. The sibling
benchmark link assumes the standard workspace layout.

## Newton screen and full MM check

The [matched Newton screen](../../../glqp-bench/benchmarks/progression/proximal-newton-20261017/README.md)
holds PC off while comparing accepted legacy Newton, centered residual-based
regularization, and centered adaptive regularization. Adaptive centered Newton
reduces QPCBOEI2 from 3322 to 111 iterations and YAO from 2555 to 328. The
centered residual-update controls require 2282 and 3703 respectively, supporting
coefficient continuation as a source of these gains independently of PC.
QSHELL worsens from 397 to 725, and LISWET1 fails after 488 accepted iterations.
All 39 GLMs validate: total iterations are 421 for accepted Newton, 420 for
centered Newton, and 275 for centered PC. These GLM comparisons explicitly use
`max_stagnation=1000`; their times remain excluded after isolation failures.

The subsequent [full Newton checkpoint](../../../glqp-bench/benchmarks/progression/proximal-newton-full-20261017/README.md)
compares all 138 MM problems at the same tolerances and budgets. Accepted Newton
validates 129 and adaptive centered Newton 127. BOYD1 and QPILOTNO are gains;
LISWET1/6/7/10 are losses. On 125 shared successes, the geometric iteration ratio
is .949863 and the median ratio is 1. The audited timing subset contains 122
problems and 113 common successes: geometric wall time is 19.6% higher with
centered Newton, and solver time is 20.7% higher. Sixteen problems remain
excluded from timing after host activity flags; these are not full-suite timing
estimates. Native status and independent acceptance remain separate, including
three accepted centered solutions that terminate with line-search failure.

The exact LISWET1 trace establishes an inaccurate, already non-descent Newton
direction after a dual-cap reduction. The other three LISWET losses fail the
original complementarity certificate, but their ordinary records do not
establish the same mechanism. Their failing states need separate diagnostics.
The source still permits finite inaccurate linear directions to reach the
outer search, and coefficient continuation does not yet account for that risk.

There is also an unresolved centered-Newton edge case: for `.5*x^2` subject to
`x<=1`, the start `x=-1, s=2, y=1, mu0=2` exactly solves the current barrier
subproblem. The strict residual search rejects the zero slope before advancing
the target, although the original problem is unsolved. Accepted legacy Newton
advances its target and solves in seven steps. Changing only `mu0` to 1.9 lets
centered Newton solve in seven steps too. This model-advancement defect is
distinct from the LISWET1 linear-accuracy failure and remains documented for a
separate algorithm experiment.
