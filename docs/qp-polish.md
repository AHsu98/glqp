# Experimental QP active-set polishing

`glqp.qp_polish` is an optional, standalone experiment. It does not change the
accepted interior-point solver, its settings, or its stopping criteria. The
problem is

```
minimize    0.5 x' P x + q' x + objective_constant
subject to  A x = b, G x <= h.
```

`P` must be the full symmetric positive-semidefinite matrix, not one triangle.
Convexity is assumed; the module does not perform a dense eigensolve. Equality
multipliers are `y`; inequality multipliers are `z >= 0`. These names differ
from the interior-point solver's result names (`nu` and `y`, respectively).
Use the original problem and original recovered multipliers after presolving.

```python
from glqp.qp_polish import PolishSettings, polish_qp

result = polish_qp(P, q, A, b, G, h, x, y, z,
    PolishSettings(strategy='add_drop', objective_constant=constant))
# The returned point is an owned copy of the input unless a candidate passed
# the original-data certificate and strictly improved its maximum residual.
x, y, z = result.x, result.y, result.z
```

Empty constraint matrices and their RHS/dual vectors may be `None`. Input
vectors must be one-dimensional, finite, and real. Inputs are never mutated;
canonical sparse matrices may be borrowed during the synchronous call.

## Active-set strategies

Identification always computes `h - G @ x`, rather than using stored IPM
slacks. Let `d_i = max(abs(h_i), max_j(abs(G_ij)))`, with a replacement of one
for an entirely zero row, and `c = max(max(abs(P)), max(abs(q)))`, with one
for an entirely zero objective. Scaled slacks and multipliers are
`s_i = (h_i - G_i @ x)/d_i` and `v_i = z_i*d_i/c`. This makes identification
invariant to nonzero positive objective and constraint rescaling.

| Strategy | Initial guess | Subsequent working-set changes |
| --- | --- | --- |
| `one_shot_slack` | `s <= active_tol` | None |
| `one_shot_ratio` | `s < 0` or `v > dual_slack_ratio*max(s, 0)` | None |
| `add_drop` | Ratio rule | Drop the most negative scaled active multiplier, otherwise add the most violated scaled inactive constraint |
| `pdas` | Ratio rule | Recompute the entire ratio rule simultaneously |

Add/drop uses `tolerance` as the negative-dual/violation threshold in these
scaled working-set coordinates. All strategies use the same final certificate
in original coordinates. The ratio rule is inspired by dual-versus-slack
active-set identification; it is not an implementation of OSQP's complete
polishing routine. OSQP documents its guess-and-extra-solve polishing approach
in its [solver documentation](https://osqp.org/docs/solver/index.html#polishing).

The two iterative strategies are bounded working-set heuristics. They do not
implement feasible blocking steps, a rank-independent basis, or a globally
convergent classical active-set QP method. A repeated active set ends the run.
By default an inaccurate working-set solve ends the run unless the complete
original certificate independently passes. The opt-in
`allow_inaccurate_updates=True` permits finite candidates from an inaccurate
solve to propose bounded active-set repairs. A clearly violated omitted row or
a large negative active multiplier can still identify a useful change. Such
updates are explicitly marked in the trace; inaccurate candidates receive no
relaxation of the final acceptance certificate. Wrong-sign duals are never
clipped.

## Sparse solves and degeneracy

Each pass forms a sparse full KKT matrix using all equalities and the selected
inequalities. Equality rows use the same row scaling as inequalities. The
objective is divided by `c`. Multipliers are mapped back to original units.
No dense nullspace, Gram matrix, or dense rank-revealing factorization is built.

The default `solve_mode='correction'` solves
`K * delta = rhs - K * anchor`, anchored at the input primal point and the
input equality/selected inequality multipliers. Inactive multipliers are zero.
The candidate is `anchor + delta`. This preserves useful dual-nullspace
components when active rows are dependent: an absolute regularized solve can
select negative multipliers even when a nonnegative representation exists.
The anchor is the same original input point on every pass. The explicit
`solve_mode='absolute'` option allows this choice to be measured separately.

The existing checked QDLDL backend and float64 right-preconditioned GMRES solve
the KKT system. A numerical diagonal shift has positive primal and negative
constraint entries. Refinement targets the **unshifted** system; the shift
does not introduce a physical proximal term. The same explicit `backend` and
`backend_options` interfaces support SuperLU, FERAL, and PARDISO experiments.
Numerical failures fall back gracefully. Malformed inputs, unavailable optional
backends, and programming/build errors propagate to the caller.

Dependent constraints and semidefinite objectives can be handled if the
requested KKT equations remain consistent and refinement verifies them.
Inconsistent guesses, an unbounded reduced QP, or an unsuitable recovered dual
representation can fail. Such a failure does not establish that the underlying
QP is unsolvable or that a guessed constraint is mathematically inactive.

## Acceptance, diagnostics, and limits

The certificate follows the benchmark's `qp-kkt-v1` equations using the full
original problem, including `objective_constant`:

* Primal: maximum equality residual and positive inequality violation, each
  divided by `max(1, abs(original RHS))` componentwise.
* Stationarity: infinity norm of `P*x + q + A.T*y + G.T*z`, divided componentwise
  by `1 + abs(P*x) + abs(q) + abs(A.T*y) + abs(G.T*z)`.
* Complementarity: `sum(abs(z*(h-G*x))) / max(1, abs(primal objective),
  abs(dual objective))`.
* Dual sign: maximum `max(-z, 0)/max(1, abs(z))`.

Every component must be finite and at most `tolerance`. In addition, the
maximum component must strictly decrease relative to the input. Already
certified inputs therefore remain certified, and an uncertified intermediate
improvement is never substituted for the input. `result.before` and `.after`
describe the actual input and returned output. A rejected attempt returns exact
copies of the input point, with `.accepted=False`.

`result.candidate` is `None` or an owned-array dictionary `{'x', 'y', 'z'}`
containing the last attempted point, including rejected attempts, for external
audit. `.candidate_active_set` identifies that point's active indices; it can
differ from `.active_set` if a later working-set solve failed. `.history` records
each active-set hash/count, added/dropped indices,
verified linear residuals, and original-data candidate certificate. Trace
records contain JSON-compatible scalars and lists. Nonfinite certificate
values are `None`; `.finite=False` prevents their acceptance. Result statuses
distinguish `accepted`, `unchanged`, `rejected`, `linear_failure`, `cycle`, and
`time_limit`; `.stop_reason` gives the specific reason.

Defaults are `active_tol=1e-6`, `dual_slack_ratio=1`, `tolerance=1e-8`,
`max_passes=12`, `linear_atol=1e-11`, `linear_rtol=1e-10`,
`max_refinement_steps=20`, `tau_reg=1e-9`, `max_solve_attempts=8`, and
`max_time=10` seconds. One-shot strategies always make at most one KKT solve.
The time limit is soft: it is checked between passes and cannot interrupt a
native factorization. A benchmark harness should enforce its own outer timeout.
`.elapsed_seconds` includes preprocessing, matrix assembly, and certificates.
`.factorizations` and `.refinements` sum work reported by successful solve
calls, including finite inaccurate returns. A thrown numerical failure does
not expose its retry count; `.failed_solve_calls` records these calls separately
and their trace has `lin_attempts=None`, rather than claiming zero native work.
