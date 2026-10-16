# Poisson first-step line-search budget

The three real bike-count examples fail the explicit `glm-v1` profile at the
first line search with a 50-trial cap. An instrumented run of `4aa1781` shows that
all three first obtain a finite merit at trial 16, but first satisfy Armijo at
trial 55. Merely avoiding overflow does not make the earlier finite steps
acceptable.

The starting predictor is zero, corresponding to a Poisson rate of one. The
mean response is about 4,504.35. With weights summing to `1 / mean(response)` and
intercept ridge `0.0001`, even the scalar intercept-only Newton calculation gives
a step near 3,104.83. The actual first intercept direction is about 3,104.63,
including the other coefficients and the solver's regularization. The geometric
backtracking factor is 0.9, so the search needs many trials to reach an appropriate
step from this initial direction.

For the unconstrained example, trial 50 has step `0.0057264169`, maximum predictor
18.14, and merit about 11,839.88, above its Armijo bound of -0.08895. Trial 55 has
step `0.0033813919`, maximum predictor 10.71, and merit -2.43878, below its bound
of -0.05243. The bounded and affine examples also first pass at trial 55. With
the explicit cap of 100, the free example finishes in eight outer iterations
and the other two in thirteen; all pass original-data GLM KKT validation.

The benchmark's explicit `glm-v2` profile therefore raises this cap to 100 while
retaining zero initialization and the same acceptance tests. The legacy
`glm-v1` profile and global solver default remain at 50. A larger finite cap
does not guarantee convergence for arbitrarily scaled counts. A log-mean
initialization or a different line-search contraction would be a separate
algorithm experiment.

`tools/diagnose_poisson_linesearch.py` records the first direction, each trial's
predictor range, merit and Armijo bound, explicit settings, source hashes,
coordinates, histories, warnings, and independent validation. The functional
diagnostic makes no timing claims. Selected trial records and all six outcomes
are retained in
[`experiments/poisson-linesearch.json`](experiments/poisson-linesearch.json);
the full local traces remain under `../.conda/poisson-linesearch-round2`.
