# Cached scalar search for GLMs

Accepted implementation: `af8e5bb` (2026-10-18), following the first measured
implementation `591ab3f`. `SolverSettings.glm_linesearch='scalar'` is the default.
Use `'backtracking'` to reproduce the previous search. The
[measurement checkpoint](../../glqp-bench/benchmarks/progression/scalar-search-20261018/README.md)
records the frozen sources, complete cohorts, independent certificates, and
excluded timing attempts.

## Fixed ray and cached work

For one direction, hold the barrier target and all problem parameters fixed:

\[
\phi(t)=f(z+t\delta z)
 +\tfrac12(x^TQx+2t x^TQ\Delta x+t^2\Delta x^TQ\Delta x)
 -b^Tx-tb^T\Delta x-\mu\sum_i\log(s_i+t\Delta s_i),
\quad \delta z=A\Delta x.
\]

The solver supplies `A@dx`, the quadratic coefficients, and linear dot products
once. [`linesearch.py`](../glqp/linesearch.py) receives arrays and scalars, no
matrices. Each trial uses vector arithmetic and scalar reductions; it performs
no sparse matvec, Hessian assembly, factorization, or linear solve. Derivatives
along this ray are

\[
\begin{aligned}
\phi'(t)&=f'(z+t\delta z)^T\delta z+x^TQ\Delta x
 +t\Delta x^TQ\Delta x-b^T\Delta x
 -\mu\sum_i\frac{\Delta s_i}{s_i+t\Delta s_i},\\
\phi''(t)&=\sum_j f''_j(z+t\delta z)\delta z_j^2
 +\Delta x^TQ\Delta x
 +\mu\sum_i\left(\frac{\Delta s_i}{s_i+t\Delta s_i}\right)^2.
\end{aligned}
\]

Nonnegative weighted squares use `sqrt(weight)*direction` before squaring,
avoiding overflow from squaring a huge direction before multiplying a tiny
curvature. Built-in likelihood kernels reuse exponential rates, stable logistic
tails, or Gaussian errors between value and derivative requests. Exact zero
Poisson weights and trailing unused predictor coordinates keep their existing
semantics. Accepted outer-point gradient/Hessian arithmetic is refreshed by the
solver as before; the ray cache does not change that ownership contract.

The [weighted-range correction](weighted-loss-range.md) at `5573cae` also
applies to cached Gaussian and Poisson kernels. It rescues exceptional weighted
values and derivatives while preserving ordinary arithmetic. The trial cache
still owns the reused rate or error; rescue does not introduce matrix products.

## Proposals and safeguards

1. Try the fraction-to-boundary full step using only its value. Most searches
   finish here, with no ray derivative work.
2. For a rejected Poisson step, evaluate base curvature once. Set
   `L=max(0, max(active A@dx))` and propose
   `log1p((-phi_prime_0)*L/phi_second_0)/L`, evaluated in the log domain,
   when `L>0`, base curvature is finite and positive, and the model has
   nonnegative finite weights and quadratic curvature. Otherwise contract.
   This can jump directly over a huge overflow region. The exponential model
   motivates a proposal; including barrier curvature does **not** give a global
   exponential bound on the barrier.
3. Otherwise use a Newton proposal at a finite rejected point. Safeguard it to
   `[.1*t, .5*t]`; if outside, try a quadratic interpolation from the base value
   and slope. Use halving when the model is unavailable, nonfinite, or has
   unsuitable curvature. The single Poisson proposal may jump below `.1*t`.
4. Accept only a finite, strictly interior point satisfying the original strict
   Armijo inequality with its configured additive epsilon. Also check finite
   free equality multipliers. A rejected upper step always shrinks; exhausting
   the existing trial budget still reports `failed_line_search`.

This is a safeguarded sufficient-decrease search, not exact line minimization.
Absolute merit arithmetic can still suffer cancellation near convergence.
No search tolerance, barrier rule, linear-solve tolerance, or objective clipping
is introduced.

The dispatch requires a finite negative merit slope and no enabled Newton
fallback. QP restoration, nonlinear non-descent restoration, explicitly enabled
Newton fallback, and the centered-proximal squared-residual merit retain their
existing acceptance rules. The objective searches now share trial scheduling
and safeguarded proposals; the [implementation map](accepted-algorithm.md#search-implementation)
describes the policy and ray adapters that preserve their numerical behavior.
`proximal_policy='no_slack'` has the same objective-plus-barrier
merit and is eligible. Physical proximal regularization changes the direction;
it is not added as an invented penalty on this ray.

## Custom losses and diagnostics

Subclasses, custom losses, and instance callback overrides use public callbacks.
If `d1f` or `d2f` raises `ArithmeticError` or `ValueError` at a finite rejected
point, the proposal is unavailable and the search contracts. Return conversion,
shape checks, `TypeError`, and `RuntimeError` remain visible. This handles, for
example, a finite `-sqrt(z)` value at zero with undefined derivatives.

Changing methods on the built-in classes themselves is unsupported; use a
subclass to change their mathematics. Observation-only class wrappers must
preserve values. Their public callback counts omit fused ray work, so use
`scalar_counters` for complete search-value/derivative counts.

`iteration_diagnostics=True` records trials, rejection reasons, proposal types,
scalar value/derivative counts, and accepted steps. Base evaluations count as
values but not positive-step trials. The untimed diagnostic tool
[`diagnose_scalar_search.py`](../tools/diagnose_scalar_search.py) separately counts
sparse products and verifies instrumented/plain solution and history parity.

## Results and remaining limitation

The three severe Poisson cases accept their initial step in two trials each;
backtracking exhausts 100. Bike box uses 14 total trials instead of 67 with the
same 13 iterations and the same sparse matvec counts. Small logistic/Gaussian
controls use 15/17 trials instead of 27/47.

At unchanged native `1e-7` and independent certificate `1e-6`, both revisions
pass 76/81. Scalar search reaches native convergence on all 81, but five weak-
ridge cases still fail the independent gap bound. In the matched severe12
native-`1e-9` control, scalar passes 12/12 versus backtracking's 9/12. Both keep
the 100-trial cap. Tightening native tolerance also tightens the requested
linear absolute tolerance, so this is not solely an outer-stopping experiment.

The final clean parallel-8 sweep observes wall/solve ratios .978862/.978364
and iteration ratio .977102 on 76 common successes. One repetition of short
cases does not establish a broad speedup. The main demonstrated gain is robust,
much shorter difficult searches. No automatic tolerance change is included.
