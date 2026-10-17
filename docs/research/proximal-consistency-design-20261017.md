# A consistent proximal model for predictor–corrector

Design review, 2026-10-17, with the first implementation's diagnostic failure
analysis appended below. This note separates a physical proximal subproblem
from a diagonal perturbation of a Newton matrix. It proposes an opt-in
implementation and controlled experiments; it does not establish a broad
performance or convergence improvement. The accepted Newton path should
remain unchanged.

## Why the distinction matters here

The previous predictor–corrector checkpoint found that disabling all outer
regularization reduced iterations dramatically on QPCBOEI2, LISWET1, and
YAO, but lost QSHELL. Regularized PC passed all 22 selected QPs and all 39
GLMs. In a QPCBOEI2 trace, the original primal Newton defect `rho*dy` was
almost the entire current KKT residual despite accurate requested-system
solves. This is evidence for separating regularization blocks and examining
their residual meaning, not evidence that every block should be removed.

The old equations have both `-rho*dy` in primal feasibility and
`rho*ds` in complementarity. Their RHS contains no persistent proximal
centers. The old line search evaluates the unregularized objective minus
the log barrier. Simply deleting `rho*ds` supplies a useful control, but
does not implement a complete proximal method.

See the retained
[`predictor-corrector-20261016` benchmark checkpoint](../../../glqp-bench/benchmarks/progression/predictor-corrector-20261016/README.md)
and the previous [equation review](predictor-corrector-design-20261016.md).
The checkpoint link is workspace-relative rather than a stand-alone
repository link.

## Physical subproblem and residual signs

Use the problem convention already implemented by GLQP:

\[
\min_x F(x),\qquad Cx+s=c,\quad Ex=e,\quad s>0,
\]

with original residuals

\[
r_d=\nabla F(x)+C^Ty+E^T\nu,\quad
r_p=Cx+s-c,\quad r_e=Ex-e.
\]

Introduce persistent centers \((\bar x,\bar y,\bar\nu)\), a primal
parameter \(\rho\ge0\), and a dual parameter \(\delta\ge0\). For a
fixed barrier target \(\mu\), define the saddle function

\[
\begin{aligned}
\mathcal L_{\mu,\rho,\delta}
={}&F(x)-\mu\sum_i\log s_i
+y^Tr_p+\nu^Tr_e
+\frac\rho2\|x-\bar x\|_2^2\\
&-\frac\delta2\|y-\bar y\|_2^2
-\frac\delta2\|\nu-\bar\nu\|_2^2.
\end{aligned}
\tag{1}
\]

Stationarity gives the residual map

\[
R_{\mu,\rho,\delta}(x,s,y,\nu)=
\begin{bmatrix}
r_d+\rho(x-\bar x)\\
r_p-\delta(y-\bar y)\\
r_e-\delta(\nu-\bar\nu)\\
s\circ y-\mu\mathbf1
\end{bmatrix}.
\tag{2}
\]

The last block multiplies slack stationarity by positive slacks. There is
**no slack center and no slack proximal term** in this model. The same
regularization coefficients occur in the residual and its Jacobian:

\[
\begin{aligned}
(H+\rho I)d_x+C^Td_y+E^Td_\nu&=-R_d,\\
Cd_x+d_s-\delta d_y&=-R_p,\\
Ed_x-\delta d_\nu&=-R_e,\\
Yd_s+Sd_y&=-\widehat R_c.
\end{aligned}
\tag{3}
\]

Here \(H=\nabla^2F(x)\). Separate inequality/equality dual coefficients
are also possible; then use the matching coefficient in each residual,
Jacobian, and eliminated penalty. They should be separate declared
experiments rather than implicit differences between code paths.

For \(\delta>0\), maximizing (1) over unrestricted auxiliary multipliers
yields the primal objective

\[
\begin{aligned}
P_{\mu,\rho,\delta}(x,s)
={}&F(x)-\mu\sum_i\log s_i+\frac\rho2\|x-\bar x\|^2\\
&+\bar y^Tr_p+\bar\nu^Tr_e
+\frac1{2\delta}\left(\|r_p\|^2+\|r_e\|^2\right).
\end{aligned}
\tag{4}
\]

This displays the physical proximal/augmented-Lagrangian interpretation.
At an arbitrary infeasible primal-dual point, however, the explicit `y`
need not equal `ybar + rp/delta`. Therefore a direction from (3) is not
automatically a descent direction for (4). With \(\delta=2\cdot10^{-8}\),
the penalty also magnifies constraint terms by \(5\cdot10^7\). Adding only
`rho/2*||x-xbar||²` to the old merit omits these dual-proximal terms and is
not a consistent implementation of (4).

The structure of (2) agrees with the first-order conditions in
[PIQP, equations 6–7](https://arxiv.org/pdf/2304.00290v2). Its practical
method combines PC with independently updated primal and dual estimates.
The authors explicitly distinguish that practical heuristic from a complete
convergence proof. The original
[IP-PMM paper, sections 5.1.3–5.1.4](https://www.pure.ed.ac.uk/ws/portalfiles/portal/174251471/ProxIPM_22oct20.pdf)
likewise separates primal, dual, and barrier parameters and discusses
progress-dependent center updates. We use those papers to motivate an
experiment, not to transfer their theoretical guarantees to GLQP.

## Predictor–corrector with fixed centers

At the iteration's starting point, compute the original residuals once and
add the center offsets in (2). Freeze \(\bar x,\bar y,\bar\nu,\rho,\delta\)
and \(H\) across both verified solves and the subsequent line search.
The affine solve uses

\[
J a=-[R_d,R_p,R_e,s\circ y]^T.
\tag{5}
\]

Use the existing affine positivity estimate and centering policy to choose
one target \(\mu_t\). Form \(q=a_s\circ a_y\), and solve the full combined
direction with

\[
Jd=-[R_d,R_p,R_e,s\circ y-\mu_t\mathbf1+q]^T.
\tag{6}
\]

No additional proximal cross term occurs: fixed-center quadratic terms are
linear in the residual map. Do not add the affine direction again. Keep
the same requested matrix and factor for (5) and (6), verify each RHS, and
hold \(\mu_t\) fixed during globalization. The cross term q belongs to the
direction construction; **it is absent from the actual trial residual** (2).

For GLMs, (2) evaluates the true trial gradient. The two solves use the
current Hessian; the next Hessian is evaluated only after acceptance. This
remains a complementarity-only PC method: it omits the second-order
stationarity remainder involving the loss's third derivative. A line search
must be able to reject the resulting direction.

The symmetric scaled KKT matrix with `w=sqrt(y)` is

\[
G=\begin{bmatrix}
H+\rho I&C^TW&E^T\\
WC&-\operatorname{diag}(s+\delta y)&0\\
E&0&-\delta I
\end{bmatrix},\qquad
b=\begin{bmatrix}-R_d\\W^{-1}\widehat R_c-WR_p\\-R_e\end{bmatrix}.
\tag{7}
\]

Recover \(d_y=W\widehat d_y\), and
\(d_s=-R_p-Cd_x+\delta d_y\). This is a small extension of the existing
workspace abstraction. Numerical factorization shifts remain separate:
`tau_reg` perturbs a preconditioner/factor, while verification still targets
the unshifted requested matrix (7).

## Recommended first globalization: the same proximal residual

A practical first implementation can globalize the genuine residual map
(2) without forming the large penalties in (4). At the current point and
chosen target set

\[
\Psi(w)=\tfrac12\|\Omega R(w)\|_2^2,\qquad w=(x,s,y,\nu),
\tag{8}
\]

where \(\Omega\) denotes fixed positive residual weights, unrelated to the KKT
scaling in (7). A single
common normalization `max(1, norm_inf(R_current))` avoids squaring very large
numbers without changing the relative weights of residual components. Freeze
this normalization throughout the search. More elaborate row/block weights
are a separate tuning dimension, not needed for the first experiment.

The exact directional derivative is

\[
\Psi'(w;d)=(\Omega R)^T(\Omega Jd).
\tag{9}
\]

An exact centering Newton direction has \(Jd=-R\), hence derivative
\(-\|\Omega R\|^2\). The combined PC direction instead has

\[
\Psi'(w;d)=-\|\Omega R\|^2-(\Omega R_c)^T(\Omega q),
\tag{10}
\]

before clipping and up to the verified linear error. Thus PC is not
automatically a descent direction for this merit. Compute the actual
derivative when directions have been edited: using (10) after clipping
silently assesses a different direction. Both actual gradient and objective
must remain finite at a trial; objective finiteness is a domain check even
when acceptance is based on the residual.

The line search can require finite, interior trials and ordinary Armijo
decrease of (8). It must not accept a point merely because the step became
small. Evaluate all residual blocks with the same centers, parameters, and
target. Existing bare-objective/residual/fallback acceptance branches are
controls for the old strategy, not alternate tests for this new merit.

A simpler first alternative is direct fixed-target infinity-norm contraction,

\[
\|R(w+td)\|_\infty\le(1-ct)\|R(w)\|_\infty.
\tag{11}
\]

This avoids squared norms and merit-slope arithmetic. For a smooth residual
and an exact centering Newton direction, sufficiently short steps contract
the norm. It has no analogous guarantee for arbitrary clipped PC directions.
Choose either (8) or (11) explicitly; do not mix original and proximal
residuals or change their scales between rejected trials.

If the PC direction has nonfinite/nonnegative slope or exhausts backtracking, a coherent
rescue is a **same-target centering Newton solve**: remove q from (6), retain
the same matrix/factor/centers/target, and search from the unchanged current
point. Count this third RHS and all attempted searches. This is a useful
second iteration of the implementation if the minimal two-RHS version
fails. Existing clipping can still spoil the rescue's exact descent
identity; record that effect or disable such edits explicitly in this mode.
Do not conceal failure with unconditional finite-step acceptance.

Residual globalization is an alternative numerical merit for the stationary
equations of (1). It does not claim monotone decrease of the physical primal
objective (4), nor a proof of global convergence for adaptive targets and
inexact outer solves.

## Centers, parameter lifetime, and stopping

Initialize centers at the initialized actual iterate. Consequently the first
proximal residuals equal the original residuals, but later center offsets
need not vanish. Unconditionally resetting every center at every iteration
would remove these offsets again, reducing the design to a regularized
Newton interpretation rather than a persistent proximal method.

A minimal practical center policy is:

1. At the beginning of the iteration retain the original stationarity norm
   `d_before` and original feasibility norm
   `p_before=max(norm_inf(rp), norm_inf(re))`.
2. Complete the accepted step and any explicitly authorized repair. Compute
   the corresponding original norms at the final actual point.
3. Update `xbar=x` only when `d_after <= .95*d_before`.
4. Update `ybar=y` and `nubar=nu` together only when
   `p_after <= .95*p_before`.
5. Otherwise retain each affected center. Make updates after search, never
   between predictor/corrector or during a rejected trial.

The 5% threshold is an experiment setting motivated by existing practical
methods. It is not uniquely correct. Updating a center when both norms are
zero is harmless. The use of original residuals prevents a self-fulfilling
center update based on merely reducing the proximal perturbation.

Improvement gates alone can stall. For example, an initial original block
norm can be zero, and an accurately solved displaced proximal subproblem
can then have a nonzero original block norm. It can never meet a relative
improvement threshold against zero. Add an explicit inner-completion rule:
at the final accepted point, using the **old frozen** centers, parameters,
and held target, refresh all centers if

\[
\|R_{\mu_t,\rho,\delta}(w_{\mathrm{accepted}})\|_\infty
\le 0.1\max(\mu_t,\mathrm{tol}).
\tag{13}
\]

This is an inexact proximal-subproblem accuracy criterion, separate from
original-problem stopping. Use the unscaled residual for (13). No additional
relative contraction is required: if the starting point is already near
the fixed subproblem root while the original problem is not solved, that
is precisely when the centers must move. Log `inner_accuracy` separately
from the two original-residual progress gates. This practical tolerance is
an experiment choice, not a quoted PIQP update or a convergence guarantee.

For the first comparison retain the previous numerical scale as a control:
start `rho=delta=2e-8`. A declared boundary-only rule can cap both at
`.001*original_KKT`, as the old implementation does; use nonincreasing
parameters if the aim is a continuation experiment. Do not silently change
parameters during factor retries: that changes the physical subproblem,
unlike increasing `tau_reg`. A progress-dependent independent rho/delta
reduction is a later ablation. A fixed positive floor also needs an explicit
policy for making center offsets vanish to the required original tolerance.

Slack repair is especially important. A repaired slack is a new point,
possibly outside the accepted merit decrease. The principled choices are to
disable repair in this new mode, or treat it as a separately checked
restoration step using the same frozen residual merit. Keeping legacy repair
must be labelled as such and measured; it forfeits a claim that every stored
point passed the proximal line search. Center updates and next-iteration
residuals must use the final repaired point, never a discarded candidate.

The selected first implementation can retain repair proposals but accept
them only when the exact, finite residual merit with the same frozen
centers/parameters/target is no greater than the already accepted trial
merit. Slack-only repair changes only primal and complementarity residual
blocks; reuse the actual gradient/objective at the unchanged x. This guard
preserves the accepted merit bound. Mark it `proximal_merit_guarded`, and
perform center updates only after accepting or rejecting the proposal.

Termination remains based on the original KKT conditions and original-data
true-slack certificate. A tiny value of (2) is insufficient: a displaced
proximal subproblem can be solved accurately while the original problem
remains infeasible or nonstationary. Equality-only and unconstrained cases
have meaningful versions of the same physical model, without PC's extra
complementarity solve; define their opt-in behavior explicitly.

## Diagnostics and tests that distinguish the models

Record original and proximal residual block norms, center displacements and
their weighted residual contributions, separate rho/delta coefficients,
center update decisions, chosen target, actual merit before/after/slope,
rescue count, and matrix/factor/RHS counts. Keep original and proximal norms
separate in naming and plots.

Before clipping, the original Newton defects under (6) are

\[
\begin{bmatrix}
-\rho\big((x-\bar x)+d_x\big)\\
\delta\big((y-\bar y)+d_y\big)\\
\delta\big((\nu-\bar\nu)+d_\nu\big)\\
-q
\end{bmatrix},
\tag{12}
\]

plus actual linear-solve error. Existing diagnostics subtract only the
direction shifts, so adding persistent centers requires an extension.
Split the center-offset effect from the Jacobian regularization effect;
otherwise a correct proximal solve looks like an inaccurate linear solve.
The expected complementarity regularization contribution is zero.

Useful independent checks include:

- A dense four-block solve with nonzero center displacements, unequal
  rho/delta, and explicit complementarity correction; compare recovered
  directions and every block residual with (3).
- Finite-difference derivatives of the exact residual merit on small
  Gaussian/logistic/Poisson examples with nonzero proximal center offsets.
- Frozen center/parameter/target assertions across both RHSs and rejected
  trial points, including a rejected nonlinear trial.
- Separate primal and dual center-update cases; no update on insufficient
  improvement; residuals are recomputed under changed centers next iteration.
- A solved proximal subproblem whose original KKT conditions fail, to
  verify that it cannot trigger native success.
- Factor reuse with two RHSs and, if implemented, a three-RHS rescue.
- Byte-identical old-default behavior when the new option is disabled.

## Declared experiment sequence

First separate the old matrix blocks while preserving its old RHS and merit:
legacy PC, standard-complementarity PC, and a control changing only the dual
shift. These diagnose which block causes the observed defects. They must be
called regularized-Newton controls, not physical proximal solvers.

Then compare the physical residual/center/globalization option against the
same standard-complementarity matrix control, initially keeping the same
rho/delta scale and PC target floor. This shows whether consistency helps
beyond simply deleting the slack shift. Only then change the center policy,
parameter continuation, or target floor one at a time where evidence points.

Use QPCBOEI2, LISWET1, YAO, CONT-101, and QSHELL as the first diagnostic panel;
QSHELL guards against promoting the known zero-rho failure. Broaden promising
candidates to the previous 22 QPs and all 39 GLMs. Include nonlinear Poisson
cases with difficult startup backtracking. Preserve common tolerances and
original-data validation. Use parallel-8 for controlled campaigns, stop all
other agent work during timing, retain failed variants and whole-wave audit
evidence, and report numerical acceptance separately from auditable timings.

## First screen: negative slope alone is insufficient

The first centered implementation, solver revision `e013f5d`, passed only
2 of the 8 screening QPs; six attempts ended in failed line search. The
QPCBOEI2 diagnostic is retained under
`glqp-bench/runs/proximal-consistency-20261017/diagnostics/qpc-centered-v1`.
It reproduces the ordinary attempt's solution arrays and iteration count.
The following observations describe that failed attempt, not an improvement.

QPCBOEI2 accepted four steps and then exhausted all 40 trials on step five.
Both requested linear solves were accurate at every iteration, no direction
clipping occurred, and no proximal center had yet moved. Center-offset
contributions stayed below approximately `4.8e-7`. Thus persistent-center
algebra is not the immediate cause of this early failure.

The first PC cross product had infinity norm `1.40e8`. It generated a dual
direction of infinity norm `4.71e9` and a positivity-limited step of only
`3.10e-10`. Its residual-merit slope was strongly negative, about `-87.2`,
so a sign-only descent safeguard did not detect this poor direction. A
large negative derivative does not imply useful progress when the allowable
step is extremely small.

On step five, the applied cross product had infinity norm `1.907e6`, while
the current residual infinity norm was `9.884e4`. The scaled merit was
`.5254` and its slope was `-.04303`. An exact centering Newton direction
would have slope `-2*merit`, approximately `-1.0508`: the PC direction
retained only 4.1% of that decrease. Its slope was still negative, so the
sign-only rescue again did not trigger. With initial step `.01656` and
backtracking factor `.9`, the smallest of 40 tested steps was approximately
`2.72e-4`; this budget need not reach the local descent region.

These observations motivate two distinct safeguards: bound the correction
before constructing the combined direction, and demand sufficient actual
descent after clipping. A faster backtracking contraction expands the
searched step range without weakening the acceptance rule.

### A bounded complementarity correction

At the selected target, let R denote the exact fixed-center residual (2),
and let q denote the raw affine cross product. For a declared
\(0<\eta<1\), use

\[
\beta=\begin{cases}
1,&\|q\|_2=0,\\
\min\left(1,\eta\,\frac{\|R\|_2}{\|q\|_2}\right),&\|q\|_2>0,
\end{cases}\qquad
q_{\rm applied}=\beta q.
\tag{14}
\]

The combined RHS uses `sy-target+q_applied`. Its matrix, target, centers,
and physical proximal equations remain unchanged. This is a declared
damped-PC policy, distinct from the original full affine cross correction.
Record raw and applied norms and beta; the direction diagnostics must
subtract the **applied** correction when assessing linear accuracy.

Write the merit's uniform residual scale as a scalar a. For an exact,
unclipped combined direction,

\[
\begin{aligned}
\Psi'(d)
&=-\|R/a\|_2^2-\langle R_c/a,\beta q/a\rangle\\
&\le-\|R/a\|_2^2+\|R_c/a\|_2\,\|\beta q/a\|_2\\
&\le-(1-\eta)\|R/a\|_2^2.
\end{aligned}
\tag{15}
\]

Thus the bound controls the correction as an inexact-Newton residual and
guarantees a definite fraction of the centering direction's residual-merit
descent in the exact, unclipped model. Uniform scaling cancels from the
ratio in (14), allowing stable norm calculations on scaled vectors. Do not
form squared norms of unscaled extreme residuals.

The proposed first bound is `eta=.5`, exposed as
`proximal_corrector_limit=.5`; `None` disables it for an explicit control.
This can restrict a direction even when its unbounded correction happens
to make the slope more negative, addressing the first-iteration tiny-step
failure above. It is not sufficient by itself to guarantee a large
positivity-limited step or global convergence.

### Actual sufficient descent and backtracking

After applying the solver's direction edits, require the actual slope to
be finite and satisfy

\[
\Psi'(d)\le-\gamma\|R/a\|_2^2,
\qquad \gamma=0.1\quad\text{for the first experiment}.
\tag{16}
\]

If it fails, use the same-target centering rescue and assess its actual
clipped slope too. This accounts for clipping and requested-system errors
that are absent from (15). Merely substituting the predicted slope from
(15) would not test the direction eventually searched.

For the centered strategy, try backtracking factor `.5` in place of `.9`.
At most 40 trials then span a factor `2^-39`, rather than `.9^39`, while
retaining the identical exact-residual Armijo condition. This is a separate
globalization change; report it together with the bounded-correction and
sufficient-descent controls rather than attributing all gains to a new
proximal model. Keep the existing barrier floor fixed initially: it
dominated all five QPCBOEI2 predictors in this trace and is another distinct
tuning dimension.

The next measurements must still establish acceptance on the screening
panel before broadening. The derivation of (15) is local and does not claim
that the experiment will recover the six failed cases.

## Progress-based physical parameter continuation

Numerical revision `1f5ac45` adds the opt-in
`proximal_reg_update="progress"` for the centered policy. The default
`"residual"` retains the earlier coefficient rule. The progress option
maintains separate nonincreasing caps for rho and delta. Before freezing an
iteration, each actual coefficient is the minimum of its persistent cap,
its user cap, and `.001*original_KKT`. The accepted affine/combined/search
model is never changed retroactively.

After the accepted point and any guarded repair, consider the two blocks
independently:

| Coefficient | Original residual norm | Physical center-offset norm |
| --- | --- | --- |
| rho | `norm_inf(rd)` | `norm_inf(rho*(x-xbar))` |
| delta | `max(norm_inf(rp), norm_inf(re))` | `max(norm_inf(delta*(y-ybar)), norm_inf(delta*(nu-nubar)))` |

A block's consecutive-miss counter resets when its original residual
satisfies the existing 5% improvement gate; otherwise it increments.
Schedule that block's next cap to be
`min(previous_cap, .1*actual_coefficient)` when either:

1. The current fixed subproblem meets the inner-accuracy criterion (13), or
2. The block has missed its progress gate for at least five accepted steps
   **and** its center-offset norm is at least
   `.1*max(original_block_residual, tol)`.

Reset the corresponding miss counter on this trigger. A changed parameter
also advances the recorded model generation. The usual center updates remain
independent: original-residual progress can refresh one center group, and
inner convergence refreshes all centers. Merely reducing a cap because of
the offset gate does not force a center refresh. The next iteration
recomputes residuals and matrix with the new coefficient.

The offset gate matters. Five small accepted steps alone do not establish
that a proximal term causes the slow progress. On early QSHELL iterates,
original feasibility is around `5.4e4` while the dual center-offset term is
only around `1e-3`; automatically deleting delta in response to those steps
would not follow the observed residual balance. Conversely, near QPCBOEI2's
late fixed subproblem, original feasibility `8.48e-4` is almost entirely
`delta*(y-ybar)`, while the fixed-target proximal residual is only about
`6.9e-7`. Reducing delta strengthens the corresponding augmented penalty
and is directly relevant to the remaining original error.

This policy is a declared continuation heuristic motivated by residual
accounting. The factors `.1`, five misses, and the offset threshold are
experimental choices, not a theorem or a reproduced PIQP parameter policy.
No global convergence guarantee is claimed for the combination with
changing barrier targets, clipped directions, and inexact PC subproblems.
Each resulting numerical outcome must still pass the original-data
certificate. Numerical source `1f5ac45` is the experiment identity; later
documentation edits do not change its equations.

### Stronger primal regularization was not a useful QSHELL fix

The initial short development probe stopped QSHELL after 43 iterations
because its stagnation window was only 20. With the longer declared window,
the bounded-correction centered control already reaches native optimal and
passes the independent certificate in **596 iterations**. Thus the short
probe did not establish that this configuration cannot solve QSHELL.

Two additional untimed probes kept the dual cap at `2e-8`, correction bound
at `.5`, and stagnation window at 1000 while increasing the primal cap:

| Primal cap | QSHELL outcome within 1000 iterations |
| --- | --- |
| `2e-8` | Optimal and accepted, 596 iterations |
| `.01` | Optimal and accepted, 781 iterations |
| `1` | Iteration limit, independent certificate failed |

Larger primal curvature was therefore a rejected experiment on this case.
The separate `physical_scale_probe.py` and raw
`strong-primal001-qshell.json` / `strong-primal1-qshell.json` are retained in
`glqp-bench/runs/proximal-consistency-20261017/probes`. These probes ran on
CPU 5, one at a time, and are correctness/iteration evidence only; their
elapsed times must not enter performance profiles. The controlled follow-up
and three exact diagnostic reproductions are recorded below.

## Final QP screen and reproduced failure mechanism

The clean `screen-v4/screen` campaign uses the declared eight-QP development
panel. Existing regularized PC passes **8/8**; centered PC with residual-only
coefficient caps passes **6/8**; the progress-based coefficient policy passes
**7/8**. Removing the barrier target floor from that progress variant also
passes **7/8**. Both progress variants lose LISWET1: the retained-floor
variant accepts 482 steps before failing the next direction/search, while
the floor-free variant accepts 239. This is a useful opt-in physical model,
not an accepted replacement for the existing solver defaults or a broad
coverage result.

Three untimed diagnostics use the final campaign's frozen `glqp_current`
source/settings on CPU 1. Every replay preserves the plain run's solution
arrays and accepted-iteration counts exactly:

| Problem | Accepted iterations | First common certificate | Native outcome |
| --- | ---: | ---: | --- |
| QPCBOEI2 | 173 | 173 | Optimal |
| QSHELL | 596 | 592 | Optimal |
| LISWET1 | 482 | Never | Failed line search on attempt 483 |

The raw traces are under
`glqp-bench/runs/proximal-consistency-20261017/diagnostics/`, in
`qpcboei2-adaptive-v4`, `qshell-adaptive-v4`, and `liswet1-adaptive-v4`.
Their diagnostic elapsed times are excluded from timing comparisons.

QPCBOEI2 uses 173 factors and 346 verified RHS solves, all labelled accurate,
without a centering rescue. It accepts every step through the declared
proximal Armijo test. Its progress policy reduces the primal cap twice and
the dual cap three times through the stalled-offset rule, then reduces both
once through inner accuracy. The last used coefficients are approximately
`rho=2e-11` and `delta=2e-12`. Original feasibility ends near `6.09e-9`, and
true-slack aggregate complementarity passes the independent certificate.

QSHELL uses 596 factors and 1192 accurate RHS solves, also without a rescue.
The stalled-offset rule never reduces its parameters; both reductions come
from inner accuracy near the end. All 596 accepted steps satisfy proximal
Armijo. It first passes the common certificate at iteration 592 and reaches
native optimal at 596. This confirms that its early small-step phase needs
to be distinguished from QPCBOEI2's proximal-offset-limited late progress.

### LISWET1: loss of original-coordinate linear accuracy

LISWET1's final failure does **not** exhaust backtracking. At attempt 483,
the affine, combined, and centering-rescue solves each hit the ten-step
GMRES refinement limit. The rescue's scaled requested-system residual is
`5.70e-8`, above its `5e-9` absolute target, with relative residual about
`.0157`. The solve is correctly labelled `inaccurate`.

Recovered original-coordinate complementarity has a requested-equation
defect of `9.78e-7`, compared with a current proximal residual infinity norm
of `8.13e-8`. The rescue has zero corrector term, and no direction clipping
occurs on this final attempt, so neither deliberate PC correction nor
clipping explains that defect. Its actual residual-merit slope is positive,
`9.85e-13`. The line search correctly rejects this direction before any
trial evaluation; accepting it merely because its components are finite
would violate the declared globalization rule.

Across the replay there are 483 affine/combined attempts, 14 inaccurate
affine solves, 14 inaccurate final combined/rescue solves, and three
centering rescues, for 969 total verified RHS solves. Direction clipping
occurs on 442 earlier attempts. Nine guarded repairs alter 14,794 slack
entries, with maximum primal-residual change only about `8.5e-12`; none
increases the accepted fixed-model merit.

The parameter/accuracy chronology identifies a concrete interaction. The
only stalled-offset reductions of delta occur after accepted steps 22 and
470. The very next affine attempts, 23 and 471, are inaccurate; attempts
471 through 483 all remain inaccurate. The final physical coefficients are
`rho=8.22e-11` and `delta=1.05e-12`, while factor-only `tau_reg` remains
`2e-8`. This association is not a proof of causality, but it motivates
checking verified linear accuracy when continuing the physical parameters
and examining factor-shift/refinement behavior before further merit tuning.
QPCBOEI2 successfully uses comparably small coefficients, so a universal
coefficient floor is not established by this one failure.

The final accepted LISWET1 point illustrates why original-data stopping
must remain authoritative. Its ordinary KKT infinity norm is `8.22e-8`,
below the native scalar tolerance, and stored-slack complementarity sums
to only `1.26e-4`. However, true-slack aggregate complementarity is `.4073`,
or approximately `1.63e-4` after objective scaling. The independent checker
therefore rejects it, and the existing aggregate native stopping test also
withholds success. No failure was converted to success by loosening that
test or by reporting only the regularized residual.
