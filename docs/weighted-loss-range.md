# Weighted Gaussian and Poisson numerical range

Accepted arithmetic: `5573cae`, 2026-10-19. Weighted loss values can be finite
even when an unweighted intermediate overflows. For example, Poisson
`z=710, y=0, w=1e-100` has finite weighted value, gradient, and curvature near
`2.233994766e208`; Gaussian `z=1e200, y=0, w=1e-300` has value `5e99`.
The previous square/exponential-first arithmetic returned infinity.

## Arithmetic contract

[`obj.py`](../glqp/obj.py) and the cached kernels in
[`linesearch.py`](../glqp/linesearch.py) share a small exceptional-range helper,
[`weighted_loss.py`](../glqp/weighted_loss.py). Ordinary evaluations retain the
previous float64 operation order. Nonfinite or prematurely subnormal
intermediates trigger a rescue that represents products as binary mantissas
and integer exponents, applies weights before final rounding, and aligns
exponents for subtraction. Gaussian residual differences are also scaled when
their direct subtraction overflows.

Exceptional Poisson exponentials use `exp(z/4)` and scaled multiplication to
form the fourth power without intermediate overflow. Ordinary rates reuse the
previous rounded exponential. For a nonfinite mixed-sign aggregate, an exact
rational sum of the already-rounded binary mantissas preserves cancellation.
The solver uses no higher-precision transcendental evaluation. Exactly zero
weights omit observations; active nonfinite predictors remain invalid, and
truly unrepresentable results remain nonfinite. No values, predictors, weights,
or curvatures are clipped. Custom callbacks retain their existing dispatch.

This improves range, not the conditioning of cancellation. Near equality of
enormous `exp(z)` and response values, componentwise rounding error can still
be large relative to their small difference. Tests use componentwise error
bounds in that regime; the implementation does not promise a uniformly small
relative error for an ill-conditioned residual.

## Independent verification

The benchmark validator is versioned `glm-kkt-v3`. Its exceptional path uses
100-digit Decimal evaluation of the defining formulas from exact binary64
inputs, sharing neither solver helpers nor range reduction. Ordinary validator
arithmetic and every certificate inequality and tolerance are unchanged.
Historical `glm-kkt-v2` results and frozen replay sources remain available.

The new tests cover representable overflow/underflow rescue, response-product
overflow, zero weights, aggregate cancellation, true overflow, derivative
padding, and unchanged ordinary evaluation. Independent equality-constrained
witnesses verify original-data certificates for extreme Gaussian and Poisson
values. See the [integrated checkpoint](../../glqp-bench/benchmarks/progression/glm-fixes-20261019/README.md)
for source parity, complete test counts, and the separately scoped timing check.

This correction does not resolve the five weak-ridge gap-certificate failures
in the standard 81-case suite: their ordinary arithmetic stays unchanged. The
[stopping study](../../glqp-bench/benchmarks/progression/glm-stopping-20261019/README.md)
investigates those failures separately.
