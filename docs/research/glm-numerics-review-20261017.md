# GLM numerical review: scalar tails, Gram Hessians, and verification

Follow-up, 2026-10-19: the [weighted-range fix](../weighted-loss-range.md)
addresses premature positive-weight Gaussian/Poisson overflow and underflow.
The [operator audit](glm-operator-audit-20261019.md) adds opt-in diagnostics;
predictor drift does not explain the five certificate failures. The
[stopping study](../../../glqp-bench/benchmarks/progression/glm-stopping-20261019/README.md)
recovers all five experimentally with nine extra total iterations, and isolates
a separate barrier-floor issue at strict tolerances. The review below retains
its original source and evidence scope.

Review date: 2026-10-17. Baseline source is `6617a0c`, whose accepted numerical
path remains legacy Newton, QDLDL, and float64 right-GMRES. This review used
local source, installed library source, primary numerical-analysis literature,
and tiny untimed CPU-5/one-thread probes. It makes no performance claim.

**The most concrete correctness problem was scalar logistic arithmetic, including
a shared blind spot in independent validation.** A bounded scalar fix and tests
were authorized during the review. The Hessian, KKT equations, regularization,
globalization, and defaults were not changed. The completed comparison identifies
the scalar-fix revision as `850a6f5`; its final 81-case outcome and independent
accuracy checks are recorded below.

Forming `A.T @ D @ A` is mathematically the exact GLM Newton Hessian contribution.
It is a reasonable fast default for tall, moderately sized, well-scaled designs
with adequate ridge. However, rounding can destroy weak curvature before the
linear solver sees it; fresh GMRES residuals against the assembled matrix cannot
detect that loss. Measure this separately from factorization accuracy and from
the conditioning of the actual optimization problem before switching systems.

## Current computation and contracts

For `F(x)=f(Ax)+.5*x.T*Q*x-b.T*x`, the source computes

```text
z = A*x
g = A.T*f'(z) + Q*x - b
D = f''(z)
H = Q + A.T*diag(D)*A
```

The scalar losses are in [`obj.py`](../../glqp/obj.py), Hessian and outer
iteration in [`solver.py`](../../glqp/solver.py), and sparse storage and linear
verification in [`kkt.py`](../../glqp/kkt.py) and
[`linear.py`](../../glqp/linear.py). `C*x<=c` and `E*x=e` remain explicit
constraints; they are not penalty terms added to the GLM likelihood.

The Hessian implementation forms `A.multiply(D[:,None])`, converts that result
to CSC, and calls `dot_product_mkl(A.T, weighted_A)`. Installed versions were
NumPy 2.2.6, SciPy 1.15.3, sparse-dot-mkl 0.9.8, and Intel oneMKL 2026.1 build
20260612. Inspection of installed `sparse_dot_mkl/_sparse_sparse.py` confirms
that this is general sparse-sparse `mkl_sparse_spmm`, with input handles and
exported result storage on each call. It is not sparse QR or a matrix-free
Hessian. Upstream exposes a separate Gram routine and sparse QR wrapper, but
neither is used here. Its Gram routine returns a triangle and is not a direct
drop-in replacement for the solver's full-matrix residual contract.
([Upstream multiplication implementation](https://github.com/flatironinstitute/sparse_dot/blob/release/sparse_dot_mkl/_sparse_sparse.py),
[public wrappers](https://github.com/flatironinstitute/sparse_dot/blob/release/sparse_dot_mkl/sparse_dot.py).)

Gaussian `D` is constant. Its Hessian is already cached **within each solve**;
public `get_H` and a subsequent solve recompute current weights/design. Logistic
and Poisson curvature changes with `z`, so they rebuild the contribution after
accepted steps. An optimization proposal must retain this distinction rather
than charging Gaussian one Gram formation per iteration. In-place input mutation
during a solve is outside the callback contract.

The requested Newton system already has outer regularization; its legacy
primal block is `H+rho*I`. A distinct `tau` shifts the factorization only.
Right-GMRES corrects against the requested assembled `G`, with fresh float64
residuals, up to ten steps, and an absolute-or-relative accuracy test. It can
return a finite direction explicitly labeled inaccurate. QDLDL uses the upper
triangle, while verification uses the full assembled matrix.

## Confirmed scalar defects and the bounded correction

### Logistic tails: a false-optimal construction

The baseline formulas were `logaddexp(0,z)-y*z`, `expit(z)-y`, and
`expit(z)*(1-expit(z))`, multiplied by observation weights. At `z=40,y=1`,
all three evaluate to zero although their correct magnitudes are approximately
`4.248354255291589e-18`. The issue is cancellation/sigmoid saturation, well
before mathematical underflow. Large weights make the discarded values
material. SciPy's own `log_expit` documentation explains this class of precision
loss and provides stable tail examples; using a stable sigmoid alone does not
make subtraction from one stable.
([SciPy 1.15.3 log-expit documentation](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.special.log_expit.html).)

This exact one-variable probe was reproduced against `6617a0c`:

\[
F(x)=10^{20}\log(1+e^{-x})+\tfrac12x^2-40x,
\qquad x_0=40.
\]

| Quantity at x=40 | Baseline computation | Stable scalar reference |
|---|---:|---:|
| Likelihood | 0 | 424.8354255291589 |
| Likelihood gradient | 0 | -424.8354255291589 |
| Likelihood curvature | 0 | 424.8354255291589 |
| Total stationarity | 0 | -424.8354255291589 |
| Total objective | -800 | -375.1645744708412 |

The solver returned native `optimal` after zero steps. More seriously, the
benchmark GLM validator also accepted: it used a stable signed-label objective
but repeated `expit(z)-y` in its independent gradient, making its stationarity
and ridge-gap bound zero. An independent implementation can share an arithmetic
defect. This artificial extreme-weight example is not evidence that a previously
measured canonical case failed its mathematical certificate; that requires
separate revalidation.

The authorized scalar correction evaluates `t=exp(-abs(z))`. The softplus tail
is `log1p(t)`; the linear term is `(1-y)*z` for nonnegative `z` and `-y*z`
otherwise. The gradient uses the small probability `t/(1+t)` on the matching
side, and curvature is `w*t/(1+t)^2`. This preserves the old algebra for finite
nonbinary labels too. For positive finite weights and subnormal `t`, a rare
`exp(log(w)-abs(z))` path preserves a representable weighted tail; it does not
impose an artificial curvature floor. Ordinary finite arithmetic remains
direct. The benchmark derivative fix is separately implemented without importing
the solver's helper.

[`test_glm_tail_stability.py`](../../tests/test_glm_tail_stability.py) contains
40 bounded tests: independent 520-digit Decimal references through `|z|=1000`,
large-weight tails, sign symmetry, nonbinary-label algebra, padding/ownership,
and the false-optimal construction checked against an independent scalar root.
Together with the existing objective tests, 68 tests passed on CPU 5. These are
correctness tests; numerical trajectory parity and suite performance are separate.

### Poisson and zero weights

Baseline `w*exp(z)` evaluates the exponential before weighting. At `z=710`,
`w=0` gives NaNs even though the observation contributes exactly zero. The
authorized fix skips exactly zero-weight observations in all three Poisson
methods; mutable weights are rechecked each call, and the ordinary all-nonzero
path retains its arithmetic. Tests include mixed/all-zero weights, extreme
ignored predictors, derivative padding, and changing weights.

A distinct positive-weight range problem remains: with `y=0,w=1e-100,z=710`,
the three old/current values are infinite although the correct weighted rate is
about `2.233994766161686e208`. Evaluating `exp(z+log(w))` can rescue such cases,
but a complete change must handle zero weights, weighted response subtraction,
underflow, and objective cancellation consistently. It was deliberately not
mixed into the bounded fix. Gaussian `w*(z-y)^2` has an analogous potential
intermediate-range issue; this review did not demonstrate a canonical failure.

## What the Gram matrix can lose

Let `B=diag(sqrt(D))*A`, with nonnegative curvature. In exact arithmetic,
`H=Q+B.T*B`. With `Q=0` and full column rank, the 2-norm condition number of
`B.T*B` is `cond(B)^2`; with ridge and constraints, the relevant conditioning
also depends on those terms and on feasible directions. Thus a large condition
number of raw `A` alone does not diagnose the final KKT solve.

The separate algorithmic risk is forming a rounded Gram matrix. Weak curvature
can be swallowed by much larger entries, or `Q`/`rho` can be lost when added to
them. A more accurate solve of that rounded matrix cannot reconstruct missing
curvature. Higham and Mary's least-squares review distinguishes normal-equation
formation from augmented-system refinement and describes using Gram factors as
preconditioners while evaluating residuals from the original factors. This is
directly relevant to the residual contract proposed below; it is not evidence
that a particular replacement is faster on our GLMs.
([Higham and Mary, section 10](https://tmary.perso.lip6.fr/doc/surveyMixed.pdf).)

Two actual `GLQP.get_H` probes used `D=I` and `v=[1,-1]`:

| A and Q | Assembled H*v | `Q*v + A.T*(D*(A*v))` |
|---|---|---|
| `A=[[1,1],[1e-8,0]], Q=0` | `[0,0]` | `[1.0000000000000001e-16,0]` |
| `A=[[1e8,1e8],[1,0]], Q=1e-7*I` | `[0,0]` | `[1.0000001,-1e-7]` |

Both designs have condition number about `2e8`. The first assembled Hessian is
exactly all ones; the second exactly all `1e16`. The second case shows why an
absolute ridge of `1e-7` is not a universal safeguard against scaled Gram loss.
Conversely, normalized columns and ridge at that scale may adequately protect
many compact cases; the probes do not predict a new suite's failure rate.

General weighted multiplication need not produce bitwise symmetric entries.
For a deterministic 32-by-8 Gaussian probe (`default_rng(20261017)`, standard
normal entries, weights `10**uniform(-8,8)`), `max(abs(H-H.T))` was
`2.9802322387695312e-8` versus `max(abs(H))=253829092.35042652`, a relative
`1.1741097961517897e-16`. This small skew is expected rounding, not a demonstrated
failure. It nevertheless means the factor's symmetric interpretation and full
verification operator can differ. Track skew before considering triangle
symmetrization or a `sqrt(D)*A` Gram kernel; either changes arithmetic and needs
an isolated comparison.

Sparsity is another independent concern. A row touching many columns can make
those columns a clique in the Gram graph. A sparse tall design can therefore
produce a nearly dense n-by-n Hessian, followed by additional KKT factor fill.
Record `nnz(A)`, `nnz(H)`, KKT/factor nnz, and formation/factorization time rather
than equating sparse input with a cheap Newton solve.

## Verification and globalization blind spots

There are four distinct levels of evidence:

1. **Assembled-system accuracy:** current GMRES evaluates `rhs-G*d`; it removes
   factorization-shift error but does not check how `D` or `H` was formed.
2. **Original derivative action:** independently evaluate
   `Q*dx + A.T*(D*(A*dx))` and the unscaled constraint/complementarity Newton
   equations. This can expose Gram formation and scaling effects. It still
   needs trustworthy scalar derivatives and finite-range arithmetic.
3. **Accepted-point nonlinear KKT:** the current gradient is evaluated from
   scalar derivatives, not as `H*x`, which is a useful independent route. But
   `z` is updated as `z+t*A*dx` across accepted iterations; only `Q*x` is freshly
   recomputed there. Repeated updates can drift from `A*x`. Final reported
   objective and benchmark validation recompute `A*x`; native KKT stopping uses
   the carried predictor. This is a source-level risk, not a measured canonical
   failure. Measure `||z-A*x||` at selected/failing iterates before changing reuse.
4. **External certificate:** the benchmark re-evaluates original constraints,
   likelihood gradient, complementarity, signs, and a dual lower bound. Its
   old logistic gradient had the demonstrated shared blind spot above.

The accepted GLM search uses the actual nonlinear objective at each trial and
keeps its barrier target fixed. Nonlinear descent directions need Armijo unless
an explicit fallback is enabled; standard GLM profiles disable that fallback.
The objective has no explicit feasibility penalty. Constraints are handled by
the primal-dual direction, positivity cap, original KKT logic, and final checks.
Centered residual-merit search is a separate experiment; it evaluates actual
trial gradients and cannot cure a wrong scalar gradient formula.

For logistic separation, true curvature can become very small even after the
arithmetic correction. For Poisson, true curvature spans a large dynamic range
or can overflow. Treat loss of curvature due to the model separately from
premature numerical zeroing. Flooring `D` changes Newton's model; increasing
objective ridge changes the optimization problem. Neither should silently
replace a stable scalar evaluation.

## The weak-ridge validator is conservative, not inherently wrong

The benchmark has diagonal positive ridge `R`. With nonnegative inequality
dual `lambda`, define `r=grad(F)+E.T*nu+C.T*lambda`. Strong convexity in metric
`R` gives the global Lagrangian lower bound

\[
L_*\ \ge\ L(x)-\tfrac12r^TR^{-1}r.
\]

Consequently the implemented certificate uses

\[
\mathrm{gap}_{\mathrm{bound}}=
\tfrac12r^TR^{-1}r-\nu^T(Ex-e)+\lambda^T(c-Cx),
\]

with negative returned inequality duals clipped only for that bound; sign and
stationarity checks still examine the original duals. This is a valid bound
when the scalar gradient and convexity assumptions are correct. At an infeasible
point, `F(x)` is not itself a feasible primal upper bound, which is why the
separate feasibility checks remain necessary.

Small ridge makes the bound pessimistic if likelihood curvature is much larger.
For an unconstrained point with 96 residual coordinates of size `1e-7`, ridge
`1e-7`, and objective scale at most one, the ridge-only term is `4.8e-6`; all
coordinate residuals can look small while the `1e-6` gap gate fails. Conversely,
near logistic saturation, the weak ridge may actually be the only useful global
curvature lower bound. A local Hessian inverse is not automatically a global
certificate and must not replace the gate merely to improve coverage.

Classify native-optimal/certificate-gap failures separately. On Gaussian cases,
an independent QR/SVD reference or a stronger *proved global* curvature bound
can distinguish conservatism from real suboptimality. For logistic/Poisson,
use an independently verified conic/dual reference or rigorous domain bounds.
Retain existing acceptance thresholds and original problem definitions.

## Completed 81-case comparison and reference checks

The final `ab81` comparison used baseline `6617a0c` and scalar-fix `850a6f5`,
the unchanged `glm-v2` profile, native tolerance `1e-7`, independent
`glm-kkt-v2` tolerance `1e-6`, and `parallel-8`. The catalog contains the prior
39 cases, 30 compact stress cases, and 12 separately declared severe cases;
existing case definitions were preserved. Both revisions validate **76/81**,
with the same five failures and no changed iteration counts among the 76 common
passes. The final isolation monitor reports clean. The earlier 69-case timing
attempt was flagged and is excluded from timing conclusions; the bounded
references below make no timing claims.
([Campaign report](../../../glqp-bench/runs/glm-stress-20261017/ab81/reports/report.md),
[isolation record](../../../glqp-bench/runs/glm-stress-20261017/ab81-isolation-01/summary.json).)

All three severe Poisson-range constraint variants terminate at
`initial_failed_line_search`. The two remaining failures are
`glm_stress_hard_gaussian_conditioned_free_v1` and
`glm_stress_hard_logistic_separation_free_v1`: both report native `optimal`,
pass original-model stationarity, and fail only the ridge-based relative-gap
certificate. Both have ridge `1e-12` and zero linear objective term. Native
`optimal` is insufficient for this existing benchmark certificate; all five
campaign failures remain failures.

The two gap-only cases were investigated separately on CPU 5 with one native
thread. Exact saved inputs and solutions were loaded from the campaign, and
the frozen scalar-fix source was used for tolerance controls. Gaussian weighted
ridge least squares was solved with pivoted QR of
`[sqrt(w)*X; diag(sqrt(ridge))]`, without forming its Gram matrix. Its augmented
design condition number is approximately `1.93e6`. The logistic reference starts
from zero, uses independent stable scalar formulas, and factors the same
augmented weighted design by QR for each Newton direction, with an Armijo search.
It reaches a gradient tolerance of `1e-14` in 39 steps. These references pass the
unchanged independent validator. Additional long-double evaluations agree with
the double-precision objective comparisons; these are numerical checks, not
interval-arithmetic proofs.

| Severe free case | Saved objective | Saved objective excess over QR reference | Saved ridge-gap bound | QR-reference ridge-gap bound |
|---|---:|---:|---:|---:|
| Gaussian conditioned | 5.10083277e-5 | 1.33716e-8 | 6.95986e-5 | 1.60e-15 |
| Logistic separation | 8.25809309e-7 | 5.49007e-7 | 1.77451e-3 | 1.97e-20 |

For Gaussian, the independently computed quadratic error energy
`.5*||sqrt(w)*X*(x-x_ref)||^2 + .5*||sqrt(ridge)*(x-x_ref)||^2`
matches the objective difference; the reference-gradient cross term is below
`1e-18`. For logistic, there is also a simple global bound: every likelihood and
ridge term is nonnegative and `q=0`, so `F*>=0` and the saved point's objective
error is at most `8.25810e-7`. Its actual error relative to the much tighter
reference is approximately `5.49e-7`. Thus both saved points have objective
errors below `1e-6` on the benchmark's `max(1,abs(F))` scale, although neither
satisfies its chosen residual/ridge certificate. This diagnoses conservatism of
that certificate for these points, without reclassifying their benchmark status.
Coefficient accuracy is a different question: maximum coordinate differences
from the references are approximately `49.6` and `69.4`, respectively, reflecting
the very weak curvature.

Two explicitly separate controls changed **only native `tol`**, keeping every
other recorded `glm-v2` setting unchanged. At `tol=1e-9`, Gaussian passes the
existing certificate in 2 steps instead of 1; logistic passes in 27 instead of
23. At `tol=1e-11`, they pass in 3 and 44 steps. The controls retain their raw
solutions and are not replacements for the original campaign records. They show
that the current Gram/Newton path can meet this stricter certificate on these
two cases with tighter stopping. These failures alone do not establish a need
to replace Gram formation. A useful next isolated experiment is to align native
termination with the desired original-model objective certificate, while
preserving valid global bounds and measuring its extra iterations.
([Severe reference methods and results](../../../glqp-bench/runs/glm-stress-20261017/references/severe/README.md),
[raw numerical evidence](../../../glqp-bench/runs/glm-stress-20261017/references/severe/summary.json).)

A separate, preselected six-case Clarabel/CVXPY panel of the 30 compact cases
validated all six saved GLQP solutions but only four Clarabel references. The
four Gaussian/logistic reference objectives agree within `1.75e-8` on the same
scale. Poisson range/free returned `optimal_inaccurate`, and Poisson
zeros/monotone returned `optimal` but failed independent original-model checks.
Both references fail stationarity as well as the gap gate, so these are not
gap-only certificate failures. The zeros/monotone objective still agrees within
`2.14e-9`; objective agreement alone is insufficient evidence of a valid reference.
All six attempts and both reference failures are retained, without retries or
tolerance changes.
([Compact reference panel](../../../glqp-bench/runs/glm-stress-20261017/references/compact30/README.md).)

## Alternatives worth isolating

The completed [stress checkpoint](../../../glqp-bench/benchmarks/progression/glm-stress-20261017/README.md)
also contains selected original-operator/QR audits and exact first-search
traces. Severe Poisson first accepts on trials 109 (box/affine) and 124 (free),
beyond the profile's 100-trial cap. Cap 200 permits native convergence but leaves
gap-certificate failures; cap 200 plus native tolerance `1e-9` certifies all
three. Tightening native tolerance also tightens the linear absolute target,
so it is not an isolated outer-stopping experiment. These controls do not
reclassify the original 76/81 result or change defaults.

For severe correlated Gaussian, the sampled production primal direction
differs from requested-system QR by 0.34%. A dense solve of the same assembled
matrix agrees with QR to `4.5e-9` absolute, versus production's `6.2e-4` error.
This particular discrepancy is dominated by permitted solve error under
conditioning. Both accurate requested directions still fail the same gap
certificate, with remaining gradient approximately `-rho*x`. The result
supports investigating stopping and regularization together; it does not
establish a failure caused by Gram assembly.

| Approach | What it addresses | Cost and limits |
|---|---|---|
| Keep formed H, audit factored Hessian actions | Detect formation loss without changing trajectories | A few additional A/A.T products only in untimed diagnostics; does not fix a bad direction |
| Form `B=sqrt(D)*A`, use symmetric Gram kernel | Explicit symmetry and possibly better intermediate range/storage | Still forms a Gram matrix and still loses weak curvature; sqrt weighting changes rounding; output triangle must be expanded consistently |
| Use formed G as preconditioner, apply true requested operator through A/D | Correct toward unassembled derivative equations while retaining fast factors | Needs explicit operator/residual abstraction, more products, and convergence/accuracy tests; current CSC-only API does not already do this |
| Augment with observation variables | Avoid explicit observation Gram formation and its fill | Adds m unknowns; ordering and factor fill can dominate for tall data; no automatic condition-number guarantee |
| QR/SVD reference or QR-based solve | Preserve design-level information and expose rank/conditioning | Excellent small diagnostic reference; constrained Newton integration and arbitrary Q require more work; forming R.T*R afterward loses the benefit |
| Matrix-free Krylov with structured preconditioner | Avoid storing H on wide/fill-heavy designs | More iterations and matvecs; preconditioning/termination are central; symmetric MINRES needs a suitable symmetric positive-definite preconditioner |

An exact augmented version of the current scaled KKT matrix replaces
`H=Q+B.T*B` by an observation block. With `W=diag(sqrt(y+rho))` and
`T=diag(s+rho*y+rho^2)` on the accepted legacy path, use

\[
\begin{bmatrix}
Q+\rho I&C^TW&E^T&B^T\\
WC&-T&0&0\\
E&0&-\rho I&0\\
B&0&0&-I
\end{bmatrix}
\begin{bmatrix}\Delta x\\u\\\Delta\nu\\v\end{bmatrix}
=\begin{bmatrix}-r_d\\W^{-1}r_c-Wr_p\\-r_e\\0\end{bmatrix}.
\]

Eliminating `v=B*dx` recovers the existing system exactly. This construction
does not divide by `D`, so exact zero curvature does not require a reciprocal
or a fabricated positive floor. The numerical factor shift remains separate
from these requested equations. Its larger dimension and ordering need actual
measurements; this derivation is a candidate, not an implemented default.

QR/least-squares methods do not justify applying an unconstrained least-squares
solver blindly to the current indefinite KKT matrix. Gould, Hribar, and Nocedal
study normal-equation and augmented approaches for constrained quadratic
subproblems and the role of projections/preconditioning.
([Primary paper](https://www.numerical.rl.ac.uk/media/people/nick-gould/GoulHribNoce01_sisc.pdf).)
LSQR uses products with a design and its transpose, avoiding explicit normal
equations; its original paper documents improved numerical behavior relative
to several algebraically related schemes. LSQR or LSMR is useful as a
Gaussian/reference building block, not a complete constrained GLM interior-point
replacement by itself.
([Paige and Saunders](https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf),
[Fong and Saunders](https://arxiv.org/abs/1006.0758).)

## Priority and discriminating experiments

1. **Preserve the scalar correction and independent validation evidence.** The
   bounded fix has independent tail tests and the completed 81-case comparison;
   coverage is unchanged. Keep the old source and raw failures visible. No broad
   parameter tuning is needed to justify the independently reproduced tail fix.
2. **Extend the completed selected derivative/operator audits.** Record predictor
   range, weights, D range/zeros, ridge and rho scales, Hessian skew, fresh versus
   carried predictor, and assembled versus factorized Hessian action on the
   actual Newton direction. Separate scaled linear residual, original block
   defect, clipping change, and original certificate components.
3. **Extend compact controlled conditioning probes.** Vary column correlation and
   scale independently of ridge; compare the same requested direction with dense
   QR/SVD or higher-precision scalar/matrix references. Include zero-curvature
   rows and near-dependent equality constraints. Keep Gaussian's one-Hessian
   cache intact. A spectrum construction with no certificate discrepancy is
   useful evidence that Gram formation is adequate on that regime.
4. **Only then prototype operator-based refinement or augmentation.** Compare
   ordinary well-scaled controls and the diagnosed hard cases, holding outer
   policies fixed. Report cold complete-solve time, memory/fill, RHS/refinement
   counts, and common coverage, not only factor timing. Prefer the smaller
   operator-verification change if it corrects the diagnosed error efficiently.
5. **Treat positive-weight exponential range as a separate scalar experiment.**
   Add weighted-rate references before changing Poisson arithmetic. Avoid fixing
   overflow by clipping predictors, altering weights, or raising ridge.

No evidence here supports replacing the fast Gram/QDLDL default universally.
The confirmed tail defect has a localized remedy; formation loss, true weak
curvature, certificate conservatism, and slow globalization need distinct tests.
