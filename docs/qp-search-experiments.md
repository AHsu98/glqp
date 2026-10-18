# QP scalar proposals and residual screening

These are opt-in experiments from 2026-10-18. Accepted defaults remain
`qp_linesearch='backtracking'` and `qp_residual_screen='off'`. The accepted
GLM scalar search is separate. See the [measured checkpoint](../../glqp-bench/benchmarks/progression/qp-scalar-search-20261018/README.md)
for frozen sources, matched tolerances, timings, and diagnostic evidence.

## Scalar proposals

For a fixed actual direction and barrier target, the QP merit is

\[
\phi(t)=\tfrac12(x+t\Delta x)^TQ(x+t\Delta x)
       -b^T(x+t\Delta x)-\mu\sum_i\log(s_i+t\Delta s_i).
\]

The solver already has `Qx` and `Qdx`. `QPScalarProposer` uses their quadratic
coefficients and the current trial slack to propose safeguarded Newton or
interpolation contractions, with halving if the model is unusable. It performs
no sparse products. Physical proximal shifts belong to the direction equations;
they are not added to this objective. The target, direction, and coefficients
remain fixed within each search.

`qp_linesearch='scalar'` changes proposals only after the existing finite,
interior, merit, original-residual, and applicable fallback checks reject a
trial. Full steps therefore retain their restoration opportunity, including
directions with a negative merit slope. Non-descent/nonfinite slopes use the
legacy contraction. The centered-proximal search is separate.

The acceptance predicates are unchanged, but trajectories need not be: a
larger contraction can cross the configured Newton fallback threshold earlier.
Diagnostics record each proposal and threshold crossing. This experiment must
not be described as numerically identical to backtracking.

## Residual rejection screens

The original residual acceptance requires a finite max norm `K_trial` with
either `K_trial < (1-c1*t)*K` or `K_trial < tol`. Fallback has its own predicate.
Every screen is bypassed when fallback is eligible, and none can accept a point.

`qp_residual_screen='complementarity'` computes the exact trial product
`trial_y*trial_s`. Its max norm is a lower bound on `K_trial`. If it meets or
exceeds both acceptance thresholds, the remaining constraint products cannot
rescue the trial and are skipped. Nonfinite products require fresh evaluation.
This shortcut uses the same floating-point operations as the original block.

`qp_residual_screen='bounded'` first tries that exact shortcut, then lazily
caches the affine blocks after the first trial has failed:

\[
\begin{aligned}
r_d(t)&=r_d+t(Q\Delta x+C^T\Delta y+E^T\Delta\nu),\\
r_p(t)&=r_p+t(C\Delta x+\Delta s),\\
r_e(t)&=r_e+tE\Delta x.
\end{aligned}
\]

It uses the clipped, inexact direction actually taken, without assuming an
exact Newton equation. Four directional constraint products are needed once per
activated ray. Matrix row-sum bounds are cached lazily within one solve.

Fresh and affine residuals can differ through cancellation and rounding. The
cache therefore bounds base residual error, direction products, trial-vector
formation, fresh residual assembly, and cached arithmetic. Its envelope matches
the legacy trial gradient `Qx+t*Qdx`, not a fresh `Q@(x+t*dx)`. The bound is
computed at the first screened step and reused at smaller steps; a larger step
rebuilds it. Each residual block uses its own error bound, so an uncertain
stationarity block cannot mask a decisive primal or equality residual.
Nonfinite or inconclusive bounds request fresh products.

The arithmetic model assumes float64 with gradual underflow. The envelope is
conservative under the documented sparse-dot model; it is not a general
interval-arithmetic certificate. Every possible acceptance and every fallback
still uses the original fresh check. Accepted points retain the original fresh
`Qx`/residual update, and products from rejected points are never reused there.

## Evidence and decision

The preselected 12-case Maros–Mészáros panel includes the long QPCBOEI2, YAO,
and LISWET1 runs alongside ordinary and factorization-heavy cases. It uses
native `1e-7`, original-data certificate `1e-6`, 60 seconds per solve, the
unchanged benchmark-v1 profile, and paced parallel-8 execution.

Scalar proposals retain 12/12 certificates and 13,140 total iterations. Their
clean geometric wall ratio is 1.0034; the three long trajectories are unchanged.
QPCBOEI2 has only 10 eligible descent searches in 3,322 iterations and makes no
scalar proposal. This is not evidence for a new iteration-reduction policy.

Both residual screens retain all 12 final solution array sets and iteration
counts exactly. The first bounded implementation uses the largest block error
against the largest residual norm. Its clean wall ratio is 1.0456, with
QPCBOEI2 rising from 2.10 to 3.39 seconds. The diagnostic shows 17,023 affine
checks and 2,838 cache initializations but zero affine rejections, adding
11,032 constraint matvecs net of the 80 exact complementarity skips.

A capture identifies stationarity error masking useful primal bounds. The
final revision uses block-specific lower bounds and native scalar arithmetic
for the envelope. Clean final wall ratios are 1.0088 for complementarity and
1.0239 for bounded screening; QPCBOEI2 still rises from 2.09 to 2.73 seconds.
One short-case repetition does not establish small timing differences. The
residual cache remains slower despite the tighter useful rejection bounds.
The final QPCBOEI2 trace preserves its complete non-time history and solution
bytes, rejects 9,436 trials through affine bounds, and removes 26,712 sparse
product calls net of cache setup. Saved sparse calls alone therefore do not
establish a complete-solve speedup on this problem.

No option is promoted from this screen, and this is not a full 138-case coverage
claim. The tests protect restoration, finite/progress fallback, cancellation,
actual clipped directions, empty blocks, and diagnostics parity. The final
implementation test run passes 783 tests with one existing expected warning.
