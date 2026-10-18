# Accepted GLQP algorithm

This is a source-level reference for the accepted **legacy Newton method with
float64 right-preconditioned GMRES refinement**. The base implementation is
`1f5ac45`, also present at documentation revision `72a3642` (2026-10-17), with
the accepted GLM arithmetic and search updates listed below. Other experimental
options remain opt-in. This document records implemented behavior, including heuristics;
it does not claim a convergence theorem for their combination.
The structural cleanup at `f179f1d` preserves this numerical behavior; selected
exact-parity checks and regression tests are recorded in the
[cleanup checkpoint](../../glqp-bench/benchmarks/progression/cleanup-20261017/README.md).
The [experiment ledger](experiments/README.md) records the evidence behind these
choices and the alternatives that remain experimental.

The GLM scalar-arithmetic correction at `850a6f5` is also accepted: logistic
loss, gradient, and curvature retain representable weighted tails, and exactly
zero-weight Poisson observations are omitted before exponentiation. Newton
equations, Hessian assembly, regularization, and outer policies are unchanged.
The [GLM numerical review](research/glm-numerics-review-20261017.md) and
[81-case checkpoint](../../glqp-bench/benchmarks/progression/glm-stress-20261017/README.md)
record the reproducer, independent validation correction, and remaining limits.

The cached GLM scalar search at `af8e5bb` (2026-10-18) is now accepted and the
default for eligible nonlinear descent directions. It preserves the merit and
Armijo acceptance, using cached ray derivatives only to choose trial steps.
QP and centered-proximal searches keep their previous behavior. See the
[scalar-search reference](glm-scalar-search.md) for equations, safeguards,
custom-loss behavior, and the matched Poisson certificate results.

The subsequent [QP search experiments](qp-search-experiments.md) remain opt-in:
`qp_linesearch='backtracking'` and `qp_residual_screen='off'` are still the
defaults. The 12-case screen did not establish a speed or iteration improvement
from scalar proposals or residual rejection caching.

The authoritative entry points are [`SolverSettings` and `GLQP`](../glqp/solver.py).
Benchmark settings are a separate layer in the sibling repository's
[`adapters.py`](../../glqp-bench/src/glqp_bench/adapters.py). Those sibling links
assume the standard `glqp`/`glqp-bench` workspace layout.

## Problem, derivatives, and residuals

GLQP solves

\[
\min_x F(x)=f(Ax)+\tfrac12x^TQx-b^Tx,
\qquad Cx\le c,\quad Ex=e.
\]

The intended setting is convex: symmetric positive semidefinite `Q` and a
separable, twice differentiable convex loss `f`. The loss interface supplies
`f(z)`, `d1f(z)`, and the diagonal `d2f(z)`. The implemented losses in
[`obj.py`](../glqp/obj.py) are weighted Gaussian least squares, logistic negative
log-likelihood, and Poisson negative log-likelihood, omitting predictor-independent
likelihood constants. Bounds can be represented as rows of `C`.

For `z=Ax`,

\[
g=\nabla F(x)=A^Tf'(z)+Qx-b,\qquad
H=Q+A^T\operatorname{diag}(f''(z))A.
\]

If `A` is absent, `f` must also be absent and the solver takes its QP path. If
`Q` is omitted, the public parser supplies `1e-8*I`: this is part of the problem
objective, distinct from both regularizations below. Supply an explicit zero
matrix when that ridge is unwanted. The parser checks dimensions and the sign
of `Q`'s diagonal; it does not establish symmetry or positive semidefiniteness.

With positive stored slack `s`, positive inequality multiplier `y`, and free
equality multiplier `nu`, define

\[
r_d=g+C^Ty+E^T\nu,\quad r_p=Cx+s-c,\quad
r_e=Ex-e,\quad r_c^0=s\odot y.
\]

The original scalar KKT measure is
`K = max(||r_d||inf, ||r_p||inf, ||r_e||inf, ||s*y||inf)`.
The Newton complementarity residual at held barrier target `mu` is
`r_c = s*y - mu`. The code's `rx`, `rp`, and `req` correspond to `r_d`, `r_p`,
and `r_e`; its variable `rc` alternates between unperturbed and target-subtracted
complementarity at documented points in `solve`.

## Initialization and the first target

`initialize` starts with `x=0`, unless a finite vector `x0` is supplied. With
equalities, it approximately projects that vector onto `Ex=e` by solving

\[
\begin{bmatrix}I&E^T\\E&0\end{bmatrix}
\begin{bmatrix}x\\\nu\end{bmatrix}
=\begin{bmatrix}x_0\\e\end{bmatrix}.
\]

The projection multiplier becomes the initial `nu`; there is no public `nu0`
argument. Its requested system uses linear absolute tolerance `1e-12`, relative
tolerance `1e-6`, and the same refinement budget as the main solve. Its numerical
factor shift acts only on the negative equality block. For RHS max norm at most
`1e-15`, the code skips the solve and uses that tiny RHS as the solution vector.

For actual inequalities, absent warm starts give `s=max(c-Cx, .01)` componentwise
and `y=1`. Explicit `s0` and `y0` must be finite, correctly shaped, and strictly
positive; they are copied but need not satisfy primal feasibility or centrality.
When inequalities are omitted, the parser installs the dummy row `0*x <= 1`,
with `s=1`, `y=min_mu`. An explicitly empty inequality matrix is also treated as
having no real inequalities.

The accepted `initial_mu_policy='fixed'` selects `mu=100` for real inequalities
and `min_mu=1e-12` otherwise. An explicit `solve(mu0=...)` takes precedence.
Complementarity-based initialization is an opt-in experiment. There is no
precentering loop in the current solve path despite the retained
`max_precenter` setting.

## One Newton direction and the two regularizations

At each outer iteration, the legacy method sets

\[
\rho=\min(\texttt{prox_reg},10^{-3}K),\qquad
\texttt{prox_reg}=2\cdot10^{-8}\text{ by default}.
\]

It requests the four linearized equations

\[
\begin{aligned}
(H+\rho I)\Delta x+C^T\Delta y+E^T\Delta\nu&=-r_d,\\
C\Delta x+\Delta s-\rho\Delta y&=-r_p,\\
(Y+\rho I)\Delta s+S\Delta y&=-r_c,\\
E\Delta x-\rho\Delta\nu&=-r_e,
\end{aligned}
\]

where `Y=diag(y)` and `S=diag(s)`. Thus the accepted legacy `prox_reg` affects
the requested direction, including the slack derivative of complementarity.
The residuals remain the original residuals above. There are **no persistent
proximal centers** or centered physical residuals in this accepted path.

[`KKTWorkspace`](../glqp/kkt.py) eliminates `Delta s` and symmetrically scales
the inequality multiplier. With `W=diag(sqrt(y+rho))` and `Delta y=W*u`, it forms

\[
G=\begin{bmatrix}
H+\rho I&C^TW&E^T\\
WC&-\operatorname{diag}(s+\rho y+\rho^2)&0\\
E&0&-\rho I
\end{bmatrix},\quad
G\begin{bmatrix}\Delta x\\u\\\Delta\nu\end{bmatrix}
=\begin{bmatrix}-r_d\\W^{-1}r_c-Wr_p\\-r_e\end{bmatrix}.
\]

It recovers `Delta s=-r_p-C*Delta x+rho*Delta y`. One Newton iteration requests
one algorithmic RHS; iterative refinement can apply the factors several times.

The separate **factorization-only** parameter is `tau_reg=2e-8`. The backend
factors `G + tau*diag(I,-I,-I)`. Verification and correction always target `G`,
including the outer `rho` but excluding `tau`. Changing `rho` changes the
requested Newton equations; changing `tau` changes their preconditioner.

## QDLDL and verified float64 GMRES

[`backends.py`](../glqp/backends.py) supplies the default QDLDL factorization,
with pattern matching, numeric updates, and upper-triangle storage reuse.
[`linear.py`](../glqp/linear.py) owns the common verification policy:

1. Factor the shifted matrix and obtain an initial inverse action `v` on the
   RHS `h`. Rescale it by `(h.T*(G*v))/||G*v||2^2`, evaluated with numerical
   scaling to avoid unnecessary overflow.
2. Recompute `h-G*v` in float64. A direction is labeled accurate if
   `||h-G*v||inf <= .05*tol` **or** `||h-G*v||2/||h||2 <= 1e-6` in the main solve.
   These tests concern the assembled, scaled KKT system, not every recovered
   unscaled Newton block independently.
3. Run at most one right-preconditioned GMRES correction cycle, with at most
   `min(max_iterative_refinement, len(h))` steps. The default budget is ten.
   Arnoldi uses two-pass modified Gram-Schmidt; the small least-squares solve
   uses SVD (`gelsd`). Stored preconditioned basis vectors reconstruct candidates.
   Each candidate receives a freshly computed residual against `G`.
4. Retain the best verified candidate. Even an initially accurate solution is
   offered a first polishing correction when its nonzero absolute error is at
   least `.1*atol`, or its relative error exceeds `rtol`. Once a verified
   accurate best candidate exists after a correction, the cycle stops; the
   polish condition does not request a second full cycle or ten mandatory steps.

Factor applications, residuals, and Krylov arithmetic are float64. There is no
extended-precision residual path. An exactly zero RHS returns a zero direction
without factorization. Main and initialization solves allow up to 12 attempts
for backend numerical failures, multiplying `tau` by ten between attempts;
zero initial `tau` disables escalation. Merely exhausting refinement accuracy
does not trigger a new factorization. Finite inaccurate directions are labeled
as such and passed to the outer acceptance logic. Input/programming errors are
not silently converted into numerical retries.

## Clipping, positivity, and line search

After the linear solve, the accepted path modifies only tiny slack/multiplier
components: for `s_i <= 1e-14`, replace `Delta s_i` by
`max(-s_i, Delta s_i)`; apply the corresponding rule to `y_i <= 1e-14`.
This is direction clipping, not a projection onto the KKT equations, and it can
change the recovered direction's equation residuals.

The common step starts at

\[
t_{\max}=\min\left(1,
 .99\min_{\Delta s_i<0}\frac{-s_i}{\Delta s_i},
 .99\min_{\Delta y_i<0}\frac{-y_i}{\Delta y_i}\right),
\]

with empty minima interpreted as infinity. The `.99` is `boundary_frac`.
The held-target merit and actual clipped-direction slope are

\[
M_\mu(x,s)=F(x)-\mu\sum_i\log s_i,\qquad
D M_\mu=g^T\Delta x-\mu\sum_i\Delta s_i/s_i.
\]

The implementation evaluates the barrier part of this slope as
`dot(Delta s, -mu/s)`. That division can overflow at subnormal slack values;
there is no explicit finite-slope guard before the legacy Armijo comparison.

Each trial uses the actual loss `f(Ax+t*A*Delta x)` and a cached exact quadratic
polynomial along the direction. Reject nonfinite primal, slack, or inequality
multiplier trials, nonpositive slacks/inequality multipliers, and nonfinite merit.
The target stays fixed
through the search. Equality multipliers are free in sign and do not receive
an explicit finiteness check before legacy Armijo; scalar search also checks
their finiteness. The legacy acceptance routes are tried in this order:

1. **Armijo:** `M_trial < M_current + .005*t*D M + 1e-8`, with coefficients
   controlled by `armijo_param` and `armijo_additive_eps`.
2. **Original KKT decrease:** available for a QP, or when `D M >= 0`; require
   finite trial `K_trial < (1-.005*t)*K`, or `K_trial < tol`. Trial GLM
   derivatives and constraints are evaluated when this test is needed. A
   nonlinear GLM descent direction does not use this residual-only shortcut.
3. **Configured Newton fallback:** with accepted `newton_fallback_policy='finite'`,
   accept a finite trial KKT norm once `t < let_newton_cook*tmax`. This fallback
   requires neither merit nor residual decrease. It is inactive when
   `let_newton_cook=0`, as in public general-objective settings and GLM profiles.

For finite negative GLM merit slopes with Newton fallback inactive, the default
`glm_linesearch='scalar'` accepts by Armijo and chooses rejected-trial steps with
the [cached scalar controller](glm-scalar-search.md). Full steps need no scalar
derivatives. Its safeguarded Newton/interpolation proposals contract by at least
half; the Poisson log-growth proposal can jump farther. Other paths, or explicit
`glm_linesearch='backtracking'`, multiply rejected steps by **.9**. The retained
`gamma=.5` setting is unused in both controllers. Exhausting the
profile's trial budget yields `failed_line_search`. The actual objective,
gradient, and original KKT blocks are refreshed at an accepted point; constant
QP/Gaussian Hessians are reused, while nonlinear GLM Hessians are refreshed.

## Next barrier target and post-step slack repair

At the accepted point, **before slack repair**, compute

\[
\bar\mu=\frac{s^Ty}{k},\quad
\xi=\frac{\min_i s_i y_i}{\bar\mu},\quad
\mu_{\mathrm{raw}}=\min(1.02-\xi,1)^3\bar\mu.
\]

The accepted legacy target update in [`barrier.py`](../glqp/barrier.py) is

\[
\mu_{\mathrm{next}}=
\max\left(\mu_{\mathrm{raw}},\ .1\mu_{\mathrm{raw}},
10^{-12},\ .01 K_{\mathrm{pre\ repair}}\right).
\]

The `.1` guard multiplies the **new raw proposal**, not the previous target.
For a finite nonnegative proposal it is redundant with the proposal itself:
it does not enforce a tenfold maximum reduction between iterations. This
behavior remains accepted after the preceding-target alternative regressed
some cases. `barrier_update_guard='previous'` changes that term to
`barrier_reduction_limit*mu_previous`; it is opt-in. The `.01*K` floor and
`min_mu` remain independently configurable. With no real inequalities, the
next target is simply `min_mu`.

Separately, the legacy repair triggers when the **initial positivity cap**
`tmax < 1e-4` and the accepted point has `min(s*y)/mu_bar < .1`.
For all components with `(s_i*y_i)/mu_bar <= .1`, it sets
`s_i=.1*mu_bar/y_i`, leaving `x`, `y`, and `nu` unchanged. It then recomputes
primal feasibility, stored complementarity, and `K`. There is no merit or
feasibility-growth guard in this accepted repair. It can worsen feasibility,
and it does not reselect the already chosen next target. The next Newton RHS
uses repaired `s*y-mu_next`.

This repair is different from tiny-component direction clipping: the former
changes an accepted stored slack; the latter changes the proposed direction
before positivity and search.

## Stopping and diagnostics

At the next loop entry, native `optimal` requires both

\[
K\le\texttt{tol},\qquad
\frac{\sum_i|y_i(c_i-C_i x)|}{\max(1,|F(x)|)}\le\texttt{tol}.
\]

The second test uses **original constraint slack**, not stored `s`, and sums
absolute products. It catches many small complementarity products and small
slack inconsistencies amplified by large multipliers. It is zero for dummy
inequalities and evaluated only when `K <= 100*tol`. Satisfying both tests at
`100*tol` sets the near-optimal flag, not `optimal`.

Stagnation stops when, after enough iterations, `K` is at least `.99` times
the logged value `max_stagnation` rows back. Iteration/time limits, line-search
failure, and numerical linear-solve failure have separate termination tags;
near-optimal variants can prefix failure/limit tags. Convergence is checked at
loop entry, so a final accepted step at the iteration limit may not receive
another convergence check. The solver timer starts after initialization and is
checked at loop entry; it is not a hard deadline inside factorization or search.
The benchmark's externally measured wall budget is a separate control.
The `feasible` result flag records having observed
stored primal/equality residuals below `1e-8`; use returned residuals and the
original-data validator for a final certificate.

`iteration_diagnostics=True` records barrier candidates and active floors,
clipping, requested/original direction defects, line-search decisions, repair,
and original/stored complementarity. `last_linear_solve` and history expose
linear status, stopping reason, shift, attempts, refinements, and residuals.
For Newton, an accepted history row's `mu` is the **next** target, while its
line-search diagnostics refer to the held target used for that step.
Synchronous callbacks observe initialization and accepted points after repair;
their arrays are borrowed read-only views that must be copied to retain.
Diagnostics and callbacks consume time and stay off in comparative timings.

## Public defaults versus benchmark profiles

Passing `settings=SolverSettings()` selects the dataclass defaults. Omitting
`settings` entirely makes a QP use `let_newton_cook=.9`; a GLM still uses zero.
The benchmark adapter always constructs explicit settings, then applies any
declared variant overrides. Campaign tolerance and time budget are separate
from the public defaults.

| Setting | Public `SolverSettings()` | QP `benchmark-v1` | GLM `glm-v2` |
|---|---:|---:|---:|
| `tol` | `1e-8` | campaign value, normally `1e-7` | campaign value, normally `1e-7` |
| `max_time` | 600 s | campaign value, normally 60 s | campaign value, normally 60 s |
| `max_iter` | 200 | 10000 | 1000 |
| `max_stagnation` | 20 | 5000 | 100 |
| `max_linesearch_steps` | 50 | 40 | 100 |
| `glm_linesearch` | `scalar` | inherited, QP path bypasses | inherited (`scalar` in current source) |
| `let_newton_cook` | 0 | .5 | 0 |
| `max_iterative_refinement` | 10 | 10 | 10 |

Unchanged baseline choices are `step_strategy='newton'`,
`proximal_policy='legacy'`, `prox_reg=2e-8`, `tau_reg=2e-8`,
`linear_solver_backend='qdldl'`, `barrier_update_guard='legacy'`,
`barrier_reduction_limit=.1`, `barrier_kkt_floor=.01`,
`initial_mu_policy='fixed'`, `slack_repair='legacy'`, and
`newton_fallback_policy='finite'`. `iteration_diagnostics` defaults to false.
`defaults-v1` explicitly sets the QP fallback to `.9` while taking the requested
campaign tolerance/time budget. Historical `glm-v1` differs from `glm-v2` by its
50-trial line-search cap. The 100-trial GLM profile addresses the documented
[Poisson first-step budget](poisson-linesearch.md), without changing acceptance.
Individual campaigns may declare further overrides: the recent proximal GLM
comparisons use `max_stagnation=1000`, rather than the profile's 100. Their saved
settings, not the profile name alone, define the measured configuration.

The benchmark's independent original-data acceptance normally uses `1e-6`.
QP acceptance gates scaled feasibility, stationarity, aggregate complementarity,
and dual-sign residuals; the GLM validator additionally gates a positive-ridge
optimality-gap bound. These are not identical to native `tol`. See the sibling
[`QP validator`](../../glqp-bench/src/glqp_bench/qp.py) and
[`GLM validator`](../../glqp-bench/src/glqp_bench/glm.py).
Native success, independent acceptance, and time-budget success should remain
separate when reporting results.

Routine execution is `parallel-8`: eight independent problems with one
numerical thread each. This does not change the Newton algorithm or give a
single problem eight threads. Serial and one-problem/eight-thread modes are
codified separately in the benchmark
[`execution policy`](../../glqp-bench/docs/execution.md). Source freezes,
isolation monitoring, and stopping other work during timings remain required.

## Experimental switches and implementation map

| Area | Source and role | Accepted boundary |
|---|---|---|
| Problem and outer iteration | [`solver.py`](../glqp/solver.py), [`obj.py`](../glqp/obj.py), [`util.py`](../glqp/util.py) | Legacy Newton and loss/constraint interface described above |
| Barrier target | [`barrier.py`](../glqp/barrier.py) | Previous-target guard, complementarity initialization, and floor ablations are opt-in |
| Repair/fallback controls | [`outer_policy.py`](../glqp/outer_policy.py) | Guarded/off repair and progress/off fallback are opt-in |
| Symmetric Newton matrix | [`kkt.py`](../glqp/kkt.py) | Separate block shifts support experiments; legacy uses one scalar rho in all blocks |
| Linear verification | [`linear.py`](../glqp/linear.py) | Float64 GMRES applies to every backend; multiple-RHS reuse supports experimental PC |
| Sparse factors | [`backends.py`](../glqp/backends.py) | SuperLU, PARDISO, and POUNCE's FERAL engine are explicit alternatives; QDLDL remains default |
| Predictor–corrector | [`predictor_corrector.py`](../glqp/predictor_corrector.py) | `step_strategy='predictor_corrector'` is opt-in; affine/combined RHSs share a frozen matrix; no-real-inequality cases use Newton |
| Centered proximal equations | [`proximal.py`](../glqp/proximal.py) | `proximal_policy='centered'` adds persistent centers and fixed-residual merit; `no_slack` is only a block-removal control |
| Standalone C++ presolve | [`presolve.py`](../glqp/presolve.py), [`_native/presolve.cpp`](../glqp/_native/presolve.cpp) | Explicit preprocessing/recovery wrapper; disabled by default; QP reductions do not transform GLM losses |
| Observation | [`diagnostics.py`](../glqp/diagnostics.py) | Optional telemetry; decision-neutral apart from elapsed time |

Centered proximal coefficient continuation (`proximal_reg_update='progress'`)
and PC can be enabled independently where supported. Centered Newton is
therefore not synonymous with PC. Its fixed-center residual merit, halving
backtracking, repair guard, and center/parameter updates are different from the
accepted legacy search. PC-specific bounded correctors and same-target rescue
RHSs do not run in centered Newton. Neither experiment has been promoted;
see the [proximal design](research/proximal-consistency-design-20261017.md) and
[experiment report](experiments/proximal-consistency.md).

The reusable KKT/shift buffers, cached transpose views, accepted-trial product
reuse, QP fast path, and constant-Hessian reuse are implementation optimizations
already present in the accepted path. They avoid unnecessary allocations and
products while retaining its equations and acceptance arithmetic.

## Practical limitations

The accepted method uses heuristic regularization, clipping, slack repair,
and fallback rules. Merit decrease alone does not enforce constraint progress;
the finite fallback and unguarded repair can accept increases. Absolute native
KKT tolerances and a fixed initial target are sensitive to problem scaling.
Poisson trials may overflow and require substantial backtracking. A finite
inaccurate scaled linear direction can pass the outer search; requested-system
accuracy alone also does not bound every recovered unscaled equation tightly.
The retained [legacy slope-overflow capture](../../glqp-bench/benchmarks/progression/predictor-corrector-20261016/slope-capture/README.md)
also exposes nonfinite slopes reaching acceptance; an arithmetic correction
needs a separate numerical experiment.
There is no homogeneous embedding or general infeasibility/unboundedness
certificate in this solve path. Validate returned vectors against original
data rather than treating every termination tag or stored scalar as a proof.

The synthetic exact-barrier-root failure documented in the centered experiments
is specific to their strict residual merit and outer-model advancement. The
accepted legacy Newton path accepts that zero step and advances its target.
It should not be conflated with observed LISWET1 coefficient/linear-accuracy
failures in the experimental centered methods.
