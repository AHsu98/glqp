# Work reused within one iteration

The iteration retains the quadratic product `Qx`, objective value, and objective
gradient evaluated at the accepted iterate. The next line search uses these
values directly. Its barrier term is evaluated again because the complementarity
target or slack repair may have changed. Each accepted step still recomputes
`Q @ x`, its gradient, and all original KKT residuals; no recursively updated
quadratic product or residual replaces those fresh evaluations.

The quadratic line-search coefficients `x @ Qx`, `x @ Qdx`, and `dx @ Qdx` are
evaluated once per direction. Trial primal, slack, dual, and predictor vectors
are shared by the feasibility check, merit check, residual fallback, and accepted
step. The merit formula, backtracking factor, acceptance tests, proximal damping,
and verified float64 GMRES refinement are unchanged. The reported final objective
is evaluated independently from the returned original variables.

The operation-count regression in `tests/test_iteration_work.py` requires one
fresh gradient per accepted iterate, and `2 + 2*steps` quadratic matrix products
for a solve whose attempts all accept: initialization, direction and fresh
accepted-iterate products, then final objective evaluation. A second solve
recomputes all values, including after problem matrices have been replaced.

An initial nine-case diagnostic compared six MM QPs and constrained Gaussian,
logistic, and Poisson examples with the accepted `3f27e71` implementation. Every
non-time history field and returned primal vector was bitwise identical. On this
i7-14700, three alternating, warm, CPU-0 observations reduced QPCBOEI2's median
solve time from 3.920 to 3.691 seconds; these exploratory timings guide profiling
and do not replace the separately recorded cold, parallel-8 campaigns.

The second round reuses sparse transpose views. `SparseTransposeCache` checks
matrix identity, shape, and the three compressed-storage arrays; numerical
writes into existing arrays remain visible, while replacement storage creates
a new view. Initialization clears the cache. This removes repeated sparse
wrapper construction from objective gradients, Hessians, and KKT residuals.
Nonlinear descent trials rejected by Armijo skip derivative and residual work
when the residual acceptance rule is disabled and the configured Newton fallback
does not yet apply. Whenever either rule could accept, the original fresh trial
residual is evaluated.

Against the first optimized checkpoint with the backend abstraction (`4233610`),
the same nine-case diagnostic again preserved every existing non-time history
field and primal vector bitwise. QPCBOEI2's median solve time fell a further
23.9%; its many backtracks amplify the transpose-wrapper cost. The small QPs
improved about 6–7%, while the factorization-heavy CONT-101 remained within 1%.
Keep these warm diagnostic results distinct from cold campaign measurements.

The portable `tools/profile_iteration_work.py` accepts reference and candidate
source snapshots, a benchmark checkout, and an output directory. It records
paired timings, independent QP checks, full iteration histories, and cProfile
statistics. Compact evidence is retained in
[`experiments/iteration-work-round1.json`](experiments/iteration-work-round1.json)
and [`experiments/iteration-work-round2.json`](experiments/iteration-work-round2.json).

The third round lets a loss declare `hessian_is_constant = True`. Gaussian
curvature is evaluated once at initialization and retained locally for that
solve. Later solves and public `get_H` calls read current weights and matrices;
tests cover both paths after weight, design, and quadratic changes. The generic
nonlinear path still evaluates curvature at each accepted iterate. Loss methods
also read predictor arrays without copying them and avoid an extra concatenation
when no derivative padding is needed; returned derivatives still own their data.

Five alternating warm observations of the constrained Gaussian example reduced
median solve time from about 24 to 7.5 ms, with bitwise-identical histories and
solutions. Logistic and Poisson histories also matched exactly, and their
timings changed by less than 3%. The Gaussian example has 1,000 observations and
40 predictors; its repeated sparse Hessian products dominate the previous time.
This is a workload-specific result, not a claimed speedup for arbitrary GLMs.
The [round-three diagnostic](experiments/iteration-work-round3.json) retains
the observations and source hashes. Use `--cases` and `--profile-case` to keep
future profiling panels small.
