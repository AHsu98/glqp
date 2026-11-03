# Proposed numerical improvements, 2026-10-16

The first algorithm experiment should be a predictor–corrector barrier policy
with one factorization per outer iteration. Before that experiment, expose the
decisions currently hidden in the iteration history and separate factorization
from a verified solve with a new right-hand side. Next, study centrality repair
and the three distinct regularization effects independently. These are proposals,
not established improvements or changes to the accepted solver defaults.

The evidence points to outer-iteration behavior as a substantial opportunity:
QPCBOEI2 takes 3,322 iterations despite every requested-system solve reporting
accurate, and three LISWET failures have small feasibility/stationarity errors
but large complementarity errors. It does **not** establish that a particular
barrier rule, clipping operation, or regularizer caused those outcomes.

This review pins GLQP at `d9bf350c2893307f862329d06db6eda08d4e38c6`; the
numerical files match the `4154fa9` experiment source. Benchmark instructions
were read at `a87781d9bb9172c163b6fd5508e8e0d993f96daf`. No solver, data,
adapter, or default was changed. The only new numerical work was a five-variable
dense equation check on CPU 12, one thread; no performance measurements were run.
[Evidence and file hashes](algorithm-improvement-evidence-20261016.json) and
[external source pins](algorithm-improvement-sources-20261016.json) accompany
this memo.

The comparator packages measured by the current benchmark harness are
**PIQP 0.6.4 and QOCO 0.1.3**. This source review instead inspected **PIQP
v0.6.3 and QOCO v0.3.2**. Those versions are not numerically interchangeable:
do not attribute the reviewed tuning rules or implementation details to the
measured packages without checking their exact release sources. Comparator
results belong to their recorded package versions and experiment manifests.

| Priority | Proposal | Type | First decision to resolve |
| --- | --- | --- | --- |
| 0 | Decision telemetry; factor once, verify multiple RHS solves | Implementation prerequisite | Which mechanism limits the difficult trajectories, and what does an extra solve cost? |
| 1 | Affine predictor plus combined corrector; controlled single-RHS barrier alternative | Algorithm | Can the barrier fall faster without losing centrality or original-data accuracy? |
| 2 | Replace unconditional slack repair with a guarded repair or centrality correction | Algorithm | Can short steps be repaired without creating larger feasibility/stationarity errors? |
| 3 | Separate primal, dual, and complementarity regularization; later a true proximal method | Algorithm | Is regularization error larger than the Newton progress being requested? |
| 4 | Explicit reversible scaling and scale-aware initialization | Equivalent model, changed trajectory | Are unit choices driving the barrier floor, tiny-component tests, or regularization? |
| 5 | Better GLM starting point and safeguarded line search; review QP fallback | Algorithm | Can rejected trials and unproductive accepted steps be reduced? |
| 6 | Bound-aware elimination and selected structured linear solves | Linear algebra/reformulation | Can a smaller system reduce factor cost without worsening conditioning? |

## What the current implementation actually solves

Write the objective as
\(F(x)=f(Ax)+\tfrac12x^TQx-b^Tx\), with \(Cx+s=c\), \(Ex=e\),
\(s>0\), and inequality multipliers \(y>0\). Define

\[
r_d=\nabla F(x)+C^Ty+E^T\nu,\quad r_p=Cx+s-c,\quad
r_e=Ex-e,\quad r_c=s\circ y-\mu\mathbf1.
\]

The outer `KKT_res` uses **unperturbed** complementarity \(s\circ y\).
Convergence additionally checks aggregate complementarity using actual
constraint slack \(c-Cx\), not just stored \(s\). That extra check prevents a
small slack-consistency error combined with a large multiplier from masquerading
as an accurate original solution. Keep it in every experiment.
See [solver.py:214](../../glqp/solver.py#L214),
[solver.py:232](../../glqp/solver.py#L232), and
[solver.py:376](../../glqp/solver.py#L376).

**Barrier target.** With real inequalities, initialization uses \(\mu=100\),
\(s_i=\max(c_i-C_ix,0.01)\), and \(y_i=1\) unless supplied. Equality projection
precedes this initialization. After an accepted step the update is

\[
\bar\mu={s^Ty\over k},\qquad
\xi={\min_i s_i y_i\over\bar\mu},\qquad
\mu^+=\max\{\min(1.02-\xi,1)^3\bar\mu,\ \mu_{\min},\ 0.01R\},
\]

where \(R=\|(r_d,r_p,s\circ y,r_e)\|_\infty\). For finite positive products,
the additional code term `0.1*mu` is redundant: it is formed from the **new**
proposal, not the preceding iteration's target. This is a minimum-centrality
rule with a residual floor; it does not use an affine predictor. At perfect
centrality its multiplier is about \(8\cdot10^{-6}\), while for \(\xi\le0.02\)
it is one. The target need not decrease. The residual floor mixes constraint,
stationarity, and complementarity magnitudes in their current units.
See [solver.py:132](../../glqp/solver.py#L132),
[solver.py:335](../../glqp/solver.py#L335), and
[solver.py:549](../../glqp/solver.py#L549).

**Requested regularized Newton equations.** The outer rule is
\(\rho=\min(\texttt{prox_reg},10^{-3}R)\), with default `prox_reg=2e-8`.
Let \(H=\nabla^2F(x)\), \(S=\operatorname{diag}(s)\), and
\(Y=\operatorname{diag}(y)\). Expanding the matrix assembled in
[kkt.py:79](../../glqp/kkt.py#L79) and the recovery in
[solver.py:283](../../glqp/solver.py#L283) gives

\[
\begin{aligned}
(H+\rho I)\Delta x+C^T\Delta y+E^T\Delta\nu&=-r_d,\\
C\Delta x+\Delta s-\rho\Delta y&=-r_p,\\
E\Delta x-\rho\Delta\nu&=-r_e,\\
(Y+\rho I)\Delta s+S\Delta y&=-r_c.
\end{aligned}
\]

The last equation matters: `w=sqrt(y+rho)` and the diagonal
`-(s+rho*y+rho**2)` regularize complementarity too. Even an exact solution of
this requested system has defects in the original Newton equations of

\[
(-\rho\Delta x,\quad +\rho\Delta y,\quad
+\rho\Delta\nu,\quad -\rho\Delta s).
\]

Their sizes depend on the direction, not just on how small \(\rho\) looks.
Here “original Newton” means the unregularized equations at the same fixed
barrier target; the final original-problem KKT certificate has zero barrier.
The accompanying dense checks verify these identities to \(5.0\cdot10^{-16}\).
With \(\rho=10^{-2}\), they exhibit original defects as large as 0.0134 while
the requested-system residual is \(2.2\cdot10^{-16}\). This is an algebra
demonstration, not evidence that that large \(\rho\) occurs in the benchmarks.

**Numerical factor shift.** A separate \(\tau\) shifts the matrix given to the
factor backend. `factor_and_solve` then uses bounded float64 right-GMRES and
fresh residuals against the requested matrix \(G\), which already contains
\(\rho\). Acceptance is absolute infinity error **or** relative L2 error,
with the existing first correction/polishing behavior. Accuracy in \(G\) does
not measure the four original Newton defects above. The factor shift, outer
regularization, and final original-problem certificate must have separate names
and diagnostics. See [linear.py:225](../../glqp/linear.py#L225) and
[linear-solves.md](../linear-solves.md). Clarabel likewise restores its true KKT
diagonal after factor regularization for refinement; that separation is a useful
design precedent, not a reason to change GLQP's accepted GMRES contract.
[Pinned Clarabel implementation](https://github.com/oxfordcontrol/Clarabel.rs/blob/25540f559592068d0c8a80e46ded1b21760212a1/src/solver/core/kktsolvers/direct/quasidef/directldlkktsolver.rs#L217).

**Two later modifications bypass that linear verification.** Before computing
the fraction-to-boundary step, the code replaces \(\Delta s_i\) by
\(\max(-s_i,\Delta s_i)\) whenever \(s_i\le10^{-14}\), and similarly modifies
\(\Delta y_i\). This caps motion toward the boundary; it is not a literal
active-set freeze. It changes the verified direction without reporting the new
Newton defect. After line-search acceptance, if the *preceding direction's*
`tmax < 1e-4` and \(\xi<0.1\), the code repairs low products by assigning
\(s_i\leftarrow0.1\bar\mu/y_i\). It refreshes slack-dependent residuals, as
required, but does not repeat the acceptance test. The chosen next \(\mu\)
remains the one computed before repair.
See [solver.py:415](../../glqp/solver.py#L415),
[solver.py:560](../../glqp/solver.py#L560), and
[the existing repair regression](../../tests/test_slack_adjustment.py).

**Globalization.** The merit function is \(F(x)-\mu\sum_i\log s_i\), with no
constraint penalty. Armijo is one acceptance route. For QPs, or non-descent
merit directions, sufficient decrease in the unperturbed KKT norm is another.
With `let_newton_cook>0`, a sufficiently short trial can also be accepted merely
because its KKT residual is finite. Positivity and finite merit are still
required. The benchmark QP profile uses `let_newton_cook=0.5`; therefore the
fallback can first apply after seven 0.9 backtracks. The GLM profiles use zero
fallback, and nonlinear merit-descent directions must satisfy Armijo to avoid a
known cycling mode. See [solver.py:429](../../glqp/solver.py#L429),
[solver.py:490](../../glqp/solver.py#L490), and
[search regression tests](../../tests/test_search_and_results.py).

## Evidence that should determine the experiments

| Retained observation | What it supports | What it does not establish |
| --- | --- | --- |
| Current QPCBOEI2 history: 3,322 iterations, all 3,322 linear statuses accurate; 3,214 logged targets equal `0.01*KKT`; 2,798 steps equal `0.9**7` | Trace the barrier floor, acceptance branch, regularization defects, and repair together | Step length alone does not identify the acceptance branch; accurate \(G\) solves do not establish accurate original Newton equations |
| At QPCBOEI2 iteration 1,001, constraint error is `1.047e-3`, stationarity `2.55e-9`, complementarity `1.052e-5`, target `1.047e-5` | Progress is then limited by primal constraint consistency, while the barrier tracks it | Removing the floor alone will fix it |
| LISWET8: 5,026 iterations, scaled complementarity `3.902e-3`; LISWET9/12: 10,000 iterations, `0.3790`/`0.3599` | These are useful centrality/complementarity stress cases; their scaled feasibility errors are below `4.5e-7` and stationarity below `6.4e-10` | They are simply inaccurate factor solves or would pass a legitimate relaxed stopping rule |
| QPILOTNO: 10,000 iterations, original scaled feasibility `3.331e-6`, complementarity `3.379e-6` | Test original feasibility recovery and late-stage regularization too | Its near-optimal solver tag is an accepted certificate; validation tolerance is `1e-6` |
| BOYD1/2, CONT-300, CVXQP1_L, CVXQP3_L time out in the full panel | Preserve them as failures and distinguish analysis/factor costs from outer iteration costs | The timeout records establish an iteration bottleneck; those records have no iteration counts |
| QFFFFF80 singleton-equality presolve changes 68 iterations to 1,751; retaining removed constant inequality rows still takes 1,785 | A valuable equivalent-formulation robustness test for scaling, initialization, and regularization | Merely restoring the barrier row count cures it; the existing ablation disproves that |
| Bike Poisson starts at zero, first intercept direction about 3,105; first finite trial 16, first Armijo trial 55 | The original 50-trial cap is insufficient; startup and trial selection are clear targets | Response clipping, changing weights, or a different objective is needed |

The first five rows come from the current numerical files and the retained
full campaign; hashes, exact options, residuals, and selected QPC history rows
are in [the evidence JSON](algorithm-improvement-evidence-20261016.json).
QPC's detailed history is a CPU-11 diagnostic, not a standard performance
measurement. The full QP campaign accepted 169/178 candidates. The older
QFFFFF80 ablation pins `5b0002e` and explicitly separates its CPU-8 diagnostic
timings; see [its report](../experiments/presolve/qfffff80-regression.md).
The [Poisson diagnostic](../poisson-linesearch.md) pins the earlier `4aa1781`
source. Its cap-100 variants pass original-data validation in 8 free / 13 box /
13 affine outer iterations. `glm-v2` already records that cap explicitly while
`glm-v1` and the solver default remain at 50.

## Priority 0: make outer-policy experiments interpretable

Add an optional iteration diagnostic with \(\bar\mu\), \(\xi\), raw barrier
proposal, floor activity, \(\rho\), `tmax`, accepted step, acceptance reason,
line-search trial count, numbers of direction edits and repairs, and their
induced residual changes. Record maximum `y/s`, direction block norms, and the
original Newton defects before and after direction editing. Existing residual
blocks must remain fresh; cached products must never stand in for validation at
a different point. Diagnostic mode should not alter arithmetic, stopping, or
the accepted iterate. Verify identical histories and complete solution vectors
with it disabled and enabled on small controls and QPCBOEI2.

Introduce an explicit factor lifetime: assemble/factor a fixed \(G\) once,
then call `solve_verified(rhs)` for each RHS. A second call to today's
`factor_and_solve` performs a numeric factor update even if the caller supplies
the existing factor object. Reusing only the object is not sufficient. The
new interface should retain each RHS's fresh float64 residual, polishing rule,
GMRES bound, factor-shift identity, and accuracy report. Invalidate it on any
matrix, backend, option, or shift change; preserve ownership of borrowed
workspace storage. Predictor and corrector RHSs are sequentially dependent,
so batching them into an assumed simultaneous solve is inappropriate.

This is an implementation experiment: require bitwise old-path equivalence
and test RHS preservation and factor-update counts. Capture factor time,
backsolve/refinement time, and full-step time separately. The meaningful cost
comparison is one factor plus two verified solves versus one factor plus one
verified solve, including any changed outer iteration count.

## Priority 1: a predictor–corrector barrier policy

For real inequalities, solve an affine direction with complementarity RHS
\(-s\circ y\). Compute positivity-limited affine step lengths and

\[
\mu_{\mathrm{aff}}={(s+\alpha_p\Delta s_{\mathrm{aff}})^T
 (y+\alpha_d\Delta y_{\mathrm{aff}})\over k},\qquad
\sigma=\operatorname{clip}(\mu_{\mathrm{aff}}/\bar\mu,0,1)^3.
\]

Then solve the **combined** direction, using the same matrix/factor and
complementarity RHS
\[
-s\circ y-\Delta s_{\mathrm{aff}}\circ\Delta y_{\mathrm{aff}}
  +\sigma\bar\mu\mathbf1.
\]
The other RHS blocks remain those for the combined Newton step. This is not an
additional direction to add to an already combined solve. Initially use a
common affine positivity step (fraction one) and retain the current common
accepted step and line search. Separate primal/dual accepted lengths, alternative
affine damping, and corrector damping are later independent choices. Skip the
extra solve for the dummy-inequality path.

This family is supported by the
[PIQP paper, equations 6–9](https://arxiv.org/pdf/2304.00290v2) and the
[QOCO paper, equations 20–21](https://arxiv.org/html/2503.12658v3).
Pinned implementations differ: [PIQP v0.6.3](https://github.com/PREDICT-EPFL/piqp/blob/8279ed73a42ddc55e3a8be561dfab7be9b3146b8/include/piqp/solver.tpp#L716)
uses damped separate affine lengths; [QOCO v0.3.2](https://github.com/qoco-org/qoco/blob/598de4e1aeaa9c522b3bc5bb1a56f3d7287dc14f/src/cone.c#L354)
uses a common affine length. [Clarabel v0.11.1](https://github.com/oxfordcontrol/Clarabel.rs/blob/25540f559592068d0c8a80e46ded1b21760212a1/src/solver/core/solver.rs#L543)
uses \((1-\alpha_{\mathrm{aff}})^3\) in its conic embedding, not the same
formula. Do not combine their constants as if they defined one algorithm.

First compare legacy policy, predictor–corrector retaining the legacy
`0.01*R` target floor, and predictor–corrector without that residual floor.
The floor-retaining arm uses
\(\mu_{\mathrm{target}}=\max(\sigma\bar\mu,\mu_{\min},0.01R)\)
in the combined RHS; the other uses
\(\max(\sigma\bar\mu,\mu_{\min})\). Keep \(\rho\), repair, initialization,
and globalization otherwise fixed. This distinguishes the extra correction
from floor removal. On QPs this is a well-motivated candidate, not a convergence
claim for the resulting GLQP hybrid. On nonlinear GLMs the existing merit test
still applies and may reject the combined direction.

A cheaper contrasting arm is a single-RHS monotone barrier method: hold
\(\mu\) until the current barrier subproblem residual is sufficiently small,
then reduce it by a linear/superlinear schedule. The
[pinned POUNCE monotone implementation](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-algorithm/src/mu/monotone.rs#L59)
uses a gate and a schedule based on `min(0.2*mu, mu**1.5)`, with additional
floor/scaling logic. An experimental GLQP arm must define its residual scales
and barrier units explicitly; directly raising a dimensional, unscaled \(\mu\)
to a power is not scale invariant. For example, fix reference scales at
initialization, form a dimensionless subproblem error \(E_\mu\) and
\(\tilde\mu=\mu/\mu_{\mathrm{ref}}\), gate at
\(E_\mu\le10\tilde\mu\), and apply the schedule to \(\tilde\mu\).
This is a proposed controlled comparator, not a port of the full Ipopt method.

Start with QPCBOEI2, LISWET8/9/12, QPILOTNO, original/reduced QFFFFF80, and
DUAL1/QSCSD1 controls. A 200-iteration diagnostic prefix can reject obviously
unproductive arms; it cannot certify a failed problem or prove final speed.
Run complete 60-second observations only for the promising arms. Advance an
arm if it gains an original-data pass or reduces factor count by at least 25%
on two difficult cases **and** lowers measured total cost, with no control
accuracy losses. Those are research gates, not predicted outcomes.

## Priority 2: repair centrality without hiding its cost

The current slack-only repair has an exact side effect:
\(\delta r_p=\delta s\). A multiplier-only repair instead gives
\(\delta r_d=C^T\delta y\). Neither is free. In the retained algebra example,
\(s=(10^{-6},1)\), \(y=(10^{-9},1)\), \(C=I\), and initially consistent
slacks yield \(\bar\mu\approx0.5\). If the tiny-step trigger is assumed,
raising the first product to \(0.1\bar\mu\) by changing only its slack creates
about \(5\cdot10^7\) primal residual; changing only its multiplier creates
about \(5\cdot10^4\) stationarity residual. This illustrates the mechanism,
not a naturally observed solver trajectory. Division by a sufficiently tiny
positive multiplier can also overflow; proposed repair code must validate the
whole candidate before committing it.

One comparison arm is a wide multiplier projection,
\[
y_i^+=\operatorname{clip}\left(y_i,
 {\mu\over\kappa s_i},{\kappa\mu\over s_i}\right).
\]
This preserves primal slack consistency but requires a new stationarity and
complementarity evaluation. Wächter–Biegler's algorithm uses such a projection;
the [author-hosted paper, equation 16](https://cepac.cheme.cmu.edu/pasi2011/library/biegler/ipopt.pdf)
and [POUNCE's pinned implementation](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-algorithm/src/ipopt_alg.rs#L4770)
are specific precedents. Its wide `kappa_sigma=1e10` bound is not equivalent to
forcing every product above `0.1*mu_bar`. A tight centrality band is a materially
more aggressive algorithm, not an innocuous clipping parameter adjustment.

Prefer testing a coordinated direction correction once predictor–corrector is
available. For a proposed enlarged step, project its complementarity products
onto a band around \(\bar\mu\), then solve a correction with zero stationarity,
primal-feasibility, and equality RHS and a complementarity RHS equal to the
projection discrepancy. Add that correction only if the computed boundary step
improves; subject the resulting direction to the same line search. This keeps
the regularized linear equations coordinated and uses the existing factor.
It does not make the nonlinear product projection exact.

[POUNCE's orthant corrector](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-convex/src/correctors.rs)
uses a `[0.1*mu,10*mu]` band, an enlarged trial step, and a required step gain;
its direct driver gates corrections on short steps. Those thresholds are
implementation-specific, and ungated corrections can impede the fast final
decrease of complementarity. Start GLQP with **at most one** extra correction,
then compare its cost with the factor savings. The original
[Gondzio paper](https://doi.org/10.1007/BF00249643) motivates multiple corrections
when additional solves are cheap relative to factorization; its performance
results are not GLQP estimates.

Before replacing either tiny-component direction editing or post-step repair,
record the induced block errors and test a candidate-only acceptance gate:
finite/interior variables, refreshed original residuals, and an explicit bound
on deterioration relative to the unmodified accepted candidate. Retain the
unmodified candidate if repair fails the gate. Test legacy repair, wide dual
projection, and one coordinated correction as separate arms, with repair
disabled as a diagnostic control. Do not silently fall back to a different arm.

Use LISWET8, QSHELL, QPCBOEI2, tiny-slack/tiny-dual toy cases, and positive
inequality row rescalings by \(10^{-6}\) and \(10^6\). Require finite positive
iterates, exact reported residuals, original-coordinate certificates, and no
new small-step cycles. A repair should earn its place by reducing total failed
trials/iterations or recovering a pass, not merely by improving a centrality
ratio immediately after editing the point.

## Priority 3: make regularization controllable and interpretable

First expose separate experimental parameters \(\rho_x,\delta_y,\delta_e\)
and \(\epsilon_y\). They respectively perturb the primal Hessian, inequality
equation, equality equation, and the multiplier in complementarity. The
current method is the special case where all four equal \(\rho\). Its same
scaled assembly generalizes to

\[
W=\sqrt{Y+\epsilon_yI},\qquad
G=\begin{bmatrix}
H+\rho_xI & C^TW & E^T\\
WC & -\operatorname{diag}(s+\delta_y(y+\epsilon_y)) & 0\\
E & 0 & -\delta_eI
\end{bmatrix}.
\]

Test \(\epsilon_y=0\) while holding the other terms at the legacy values,
then independently test lower primal/dual terms. With \(\epsilon_y=0\), the
original complementarity linearization is restored, but very small \(y\)
can make the scaled RHS worse. Removing a regularizer may expose rank
deficiency or poor factor conditioning. Expected benefit and risk must both
be measured; “smaller regularization is more accurate” is insufficient.

The practical diagnostic is the regularization-induced block defect relative
to the reduction being requested. A future adaptive arm could permit one
refactor with a smaller outer regularizer when that defect exceeds, for
example, 10% of a scaled block residual budget. Increase the factor-only shift
only through its explicit numerical policy, and re-verify against the requested
matrix. The 10% threshold and retry budget are proposed hypotheses, not defaults
borrowed from theory. First log the ratio: if it is negligible on the difficult
cases, deprioritize this arm.

A true proximal method is a larger, separate branch. For fixed centers
\((\bar x,\bar y,\bar\nu)\), its subproblem residuals would include

\[
r_d+\rho_x(x-\bar x),\quad
r_p-\delta_y(y-\bar y),\quad
r_e-\delta_e(\nu-\bar\nu),\quad s\circ y-\mu\mathbf1.
\]

It also needs rules for updating centers and regularizers and still tests
the original KKT conditions. PIQP updates centers according to primal/dual
progress and changes penalties with complementarity progress;
[its current implementation](https://github.com/PREDICT-EPFL/piqp/blob/8279ed73a42ddc55e3a8be561dfab7be9b3146b8/include/piqp/solver.tpp#L786)
does not use exactly the same heuristic constants as the 2023 paper.
The [IP-PMM analysis](https://doi.org/10.1007/s10589-020-00240-9) concerns a
specified proximal algorithm with its own assumptions and neighborhood rules.
Those guarantees do not automatically apply to GLQP's current Jacobian shifts
or to an incomplete mixture of the two methods.

Use rank-deficient/repeated equality toys, original/reduced QFFFFF80,
QPCBOEI2, and QPILOTNO. Check dense original-equation identities before sparse
runs. Compare initialization in both formulations, deterministic row/column
permutations, and recovered original residuals. A goal is to eliminate the
large formulation-dependent iteration increase without sacrificing reductions
that help other problems. Do not encode an input-name exception.

## Priority 4: scaling and initialization as explicit experiments

The barrier floor, `1e-14` direction-edit threshold, initial slack floor 0.01,
initial target 100, and \(\rho\) all depend on units. Equilibrating only the
factor matrix does not make these outer decisions invariant. Conversely,
reformulating the model changes the floating-point trajectory even when its
mathematical optimum is unchanged. Treat that as a visible option with complete
recovery and original-data validation.

For positive diagonal variable scaling \(D\) and row scalings \(R_c,R_e\),
use \(x=D\hat x\), \(\hat C=R_cCD\), \(\hat c=R_cc\),
\(\hat E=R_eED\), and \(\hat e=R_ee\). Transform the objective as
\(\hat Q=DQD\), \(\hat A=AD\), \(\hat b=Db\). Recover
\(s=R_c^{-1}\hat s\), \(y=R_c\hat y\), and
\(\nu=R_e\hat\nu\). Then each slack–multiplier product is unchanged by
row scaling. If the objective is additionally multiplied by \(\kappa>0\),
dual recovery divides by \(\kappa\), and the internal barrier has the same
cost scaling. Never use scaled stopping alone to claim an original solution.
Check any barrier floor against the actual aggregate complementarity test:
near a central, slack-consistent point, its contribution is approximately
\(k\mu/\max(1,|F|)\). A fixed floor that exceeds the required accuracy in
original units can prevent certification regardless of the Newton accuracy.

Ruiz-style bounded equilibration is a concrete candidate already implemented
in [PIQP](https://github.com/PREDICT-EPFL/piqp/blob/8279ed73a42ddc55e3a8be561dfab7be9b3146b8/include/piqp/sparse/preconditioner.tpp)
and [QOCO](https://github.com/qoco-org/qoco/blob/598de4e1aeaa9c522b3bc5bb1a56f3d7287dc14f/src/equilibration.c).
Record the scaling vectors, limits, zero-row handling, setup cost, and mapped
warm starts. Test variable/row scaling separately from objective scaling.
An inexpensive separate arm sets the initial barrier from the initialized
\(s^Ty/k\) instead of 100; it must preserve explicitly supplied starts and
the no-inequality path. Neither candidate should be coupled to a looser final
certificate.

## Priority 5: GLM startup and globalization

The bike issue is already understood well enough to avoid a blind global cap
increase: `glm-v2` explicitly supplies 100 trials. The next experiment can
reduce the wasted trials while preserving the objective. Where the model
explicitly identifies an unconstrained intercept, solve the one-dimensional
intercept-only regularized problem before the full Newton iteration. For
Poisson, its derivative is

\[
\sum_i w_i(\exp(a)-t_i)+q_{00}a-b_0.
\]

Use a bracketed scalar solve including the intercept ridge; `log(mean(t))`
alone is not the exact regularized minimizer. The current benchmark's box and
affine constraints leave the intercept unrestricted, so that construction is
feasible when all other coefficients are zero. Do not infer an intercept from
an arbitrary user's matrix, and retain zero initialization as the controlled
comparison. Gaussian/logistic variants, all-zero Poisson responses, single-class
logistic labels, and explicit warm starts need their own checks.

A separate line-search arm can choose its first trial from the already computed
predictor direction: \(t_0=\min(t_{\max},L/\|A\Delta x\|_\infty)\), followed
by the unchanged finite/interior and Armijo tests. Pre-register a small set of
\(L\) values (for example 1, 5, 10) on a development panel and select on held-out
GLMs; this cap is a heuristic, not a claimed self-concordant step bound.
Safeguarded interpolation after rejection is another alternative. Do not clip
\(Ax\), exponentials, responses, or weights to make the tests pass; that changes
the function being solved or its derivatives.

For the QP fallback, first inspect the new acceptance labels. If many accepted
fallback steps do not reduce any relevant residual, compare a residual-decrease
gate or an explicit merit function
\(F-\mu\sum\log s+\eta\|(r_p,r_e)\|_1\), with a documented penalty update.
Adding that penalty requires checking descent under the regularized direction;
it is not a one-line guarantee. A full filter method is a later, larger project:
the [Wächter–Biegler method](https://doi.org/10.1007/s10107-004-0559-y) includes
restoration and other safeguards beyond accepting either objective or residual
decrease. Preserve the existing nonlinear cycling and inconsistent-slack QP
regressions when exploring this area.

First require that all three bike variants pass from their recorded starts and
that trial counts fall substantially. Then use `glm-quick`, a large sparse
affine case from each loss, and weighted Adult. No new acceptance losses, no
objective/certificate discrepancy beyond the current independent validation,
and an end-to-end gain after startup cost are necessary to promote an option.

## Priority 6: reduce factor work where structure justifies it

Native solver/assembly improvements are already measured; another allocation
cleanup should be justified by a current profile. A more substantial option is
to treat coefficient bounds separately from general rows. Eliminating bound
directions contributes a diagonal term to the primal Hessian and can shrink
the KKT system without general constraint fill. Eliminating arbitrary rows can
instead make a sparse problem dense or square its conditioning difficulty.
Measure fill, factor memory, solve/refinement cost, and a fresh **unreduced**
Newton residual before judging this successful.

POUNCE offers explicit block and small-Schur solvers, but they require a useful
partition; the existing [backend study](../linear-backends.md) did not measure
those paths. Start with box-constrained GLMs and one structured control QP,
then compare against the current monolithic backend at identical thread budgets.
The existing wider screen does not support a universal replacement of QDLDL,
and eight-thread PARDISO was slower on the three measured hard controls.
Do not combine a backend switch, a new outer algorithm, and a thread-count change
in the same causal comparison.

## Experiment and promotion rules

Follow the benchmark's [research workflow](../../../glqp-bench/docs/research-workflow.md):
immutable committed solver/harness exports, exact per-variant options, original
data hashes, saved primal/dual arrays, independently revalidated certificates,
and retained failures. Use `parallel-8`, one numerical thread and one observation
per variant/problem for screening. Multithreaded experiments stay on the three
hard cases. During any timed campaign, **all agents stop tests, builds, numerical
probes, and other compute**, including work on efficiency cores. Reading and
editing research notes can continue.

Implementation-only changes must preserve accepted histories and complete
solutions bit for bit on the established controls. Algorithm changes are
expected to alter histories; they must preserve the original mathematical
problem and satisfy the existing original-data validator. Compare objectives
and dual certificates, not identical dual coordinates when solutions are
nonunique. Do not relax the accepted GMRES criterion or final certificate as
part of a barrier/repair experiment.

Promote in stages: algebra/edge-case checks, a bounded diagnostic panel,
`quick` plus `glm-quick`, then one full QP and `glm-extended` comparison for a
candidate worth keeping. Preserve the 50-trial Poisson failures in historical
comparisons; new GLM studies use the explicit `glm-v2` profile. Record both
runtime and factor/solve/trial counts. No previously accepted case may be lost
without an explicit decision; do not trade away accuracy for an attractive
geometric mean. Treat over-20% slowdowns on observations above 50 ms as flags
for three paired repeats, and require a repeated aggregate gain before promoting
a default. These screening thresholds are proposed project criteria, not a
statistical confidence interval from one observation.

The immediate completion milestone is an instrumented, behavior-preserving
baseline and one factor-reusing predictor–corrector experiment with the two
floor treatments. Repair and regularization experiments then use the diagnostic
evidence rather than a broad simultaneous parameter sweep. True proximal
centers, a complete filter/restoration method, infeasibility embeddings, and
active-set polishing remain separate research projects until the earlier
experiments identify a need.

## Pinned sources and reproduction

| Source | Version inspected | Relevant scope |
| --- | --- | --- |
| PIQP | v0.6.3, `8279ed73a42ddc55e3a8be561dfab7be9b3146b8` | Predictor/corrector, independent proximal penalties/centers, equilibration |
| Clarabel.rs | v0.11.1, `25540f559592068d0c8a80e46ded1b21760212a1` | Factor versus refinement target; conic predictor/corrector |
| QOCO | v0.3.2, `598de4e1aeaa9c522b3bc5bb1a56f3d7287dc14f` | Common-step affine centering, factor reuse, equilibration |
| POUNCE | `c70e06dc82504118836046188d5f4e8f019e4a53` | Distinct NLP and convex drivers; multiplier projection, monotone barrier, orthant corrections |
| PIQP paper | arXiv `2304.00290v2`, 2023-09-15 | Proximal residual formulation and practical algorithm; not every current tuning constant |
| Clarabel paper | arXiv `2405.12762v1`, 2024-05-21 | Conic homogeneous embedding and numerical regularization |
| QOCO paper | arXiv `2503.12658v3` | Predictor/corrector equations and conic solver design |
| Wächter–Biegler | Author-hosted 2004 preprint; published 2006, DOI `10.1007/s10107-004-0559-y` | Filter globalization and multiplier projection |
| Pougkakiotis–Gondzio | 2021, DOI `10.1007/s10589-020-00240-9` | A specified interior-point proximal method and its analysis |
| Gondzio | 1996, DOI `10.1007/BF00249643` | Multiple centrality corrections with factor reuse |

The [source manifest](algorithm-improvement-sources-20261016.json) records
immutable code URLs, archive hashes, and versioned paper URLs. These sources
describe different solvers; no cross-solver timing or theoretical guarantee is
claimed for GLQP. No external implementation was copied into numerical code.
The PIQP/QOCO rows above are research source pins, not the versions of the
measured comparator packages identified at the start of this memo.

Reproduce the small local algebra check outside a timing campaign:

```sh
source ../activate-glqp.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 12 python docs/research/check_algorithm_proposals.py
```

Add `--include-local-history` to verify the retained QPC hashes and extract the
full campaign's failure records when those ignored workspace artifacts are
available. The checked-in compact JSON preserves the extracted evidence even
when the raw diagnostic directory is not present in a new checkout.
