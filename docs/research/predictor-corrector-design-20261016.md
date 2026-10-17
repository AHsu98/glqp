# Predictor-corrector design for the current GLQP equations

Design review, 2026-10-16. This note derives an opt-in experiment; it does not
claim improved convergence or performance. The accepted Newton strategy stays
the default. No benchmark was run for this review.

## Recommendation

Add `step_strategy="predictor_corrector"` alongside `"newton"`. Reuse one
fixed requested matrix for an affine solve and a combined solve, each with
its own verified linear residual. Choose the predictor-corrector target before
the combined solve and use that same target throughout its line search.
Keep the existing common accepted step, true GLM evaluations, clipping,
repair, outer regularizer, and stopping tests for the first controlled trial.
Skip the extra solve for absent/dummy inequalities. Do not add a GLM
third-derivative correction or a new proximal algorithm in this first trial.

There are two distinct cautions. A combined direction need not be a descent
direction for the existing GLM merit function. Also, current `prox_reg`
perturbs complementarity as well as the other equations; calling it a
standard quadratic proximal objective would be inaccurate. Both issues can
be handled honestly without rejecting the experiment or silently changing
the model being solved.

## Residuals and the actual requested equations

Write the original problem as

\[
\min_x F(x)=f(Ax)+\tfrac12x^TQx-b^Tx,
\quad Cx+s=c,\quad Ex=e,\quad s\ge0.
\]

At the current interior point define

\[
\begin{aligned}
r_d &= \nabla F(x)+C^Ty+E^T\nu,\\
r_p &= Cx+s-c,\qquad r_e=Ex-e,\\
r_c(\mu)&=s\circ y-\mu\mathbf1,\\
H &= Q+A^T\operatorname{diag}(f''(Ax))A,\quad
S=\operatorname{diag}(s),\quad Y=\operatorname{diag}(y).
\end{aligned}
\]

The code in `glqp/solver.py` and `glqp/kkt.py` requests

\[
\begin{aligned}
(H+\rho I)d_x+C^Td_y+E^Td_\nu&=-r_d,\\
Cd_x+d_s-\rho d_y&=-r_p,\\
Ed_x-\rho d_\nu&=-r_e,\\
(Y+\rho I)d_s+Sd_y&=-r_c(\mu).
\end{aligned}\tag{1}
\]

Here \(\rho=\min(\texttt{prox_reg},10^{-3}R)\), where \(R\) is the current
unperturbed KKT infinity norm. The factorization shift `tau_reg` is separate:
linear refinement verifies against the requested matrix that includes rho,
not the additionally shifted factorization matrix.

For compactness call the matrix of (1) \(J_\rho\), with block order
\((x,s,y,\nu)\). A supplied effective complementarity residual \(\widehat r_c\)
is sufficient to represent either direction below. The existing symmetric
elimination uses

\[
W=\operatorname{diag}(\sqrt{y+\rho}),\quad
G=\begin{bmatrix}
H+\rho I&C^TW&E^T\\
WC&-\operatorname{diag}(s+\rho y+\rho^2)&0\\
E&0&-\rho I
\end{bmatrix},
\]

\[
G\begin{bmatrix}d_x\\W^{-1}d_y\\d_\nu\end{bmatrix}
=\begin{bmatrix}-r_d\\W^{-1}\widehat r_c-Wr_p\\-r_e\end{bmatrix},
\qquad d_s=-r_p-Cd_x+\rho d_y.\tag{2}
\]

Thus the affine/combined pair changes an RHS, not a matrix. Freeze the
iterate, H, rho, scaling and matrix storage until both verified solves finish.
Do not update a center or Hessian after the affine solve. Borrowed RHS storage
must not overwrite data still needed to verify an earlier solve.

## Affine direction, centering parameter, and combined direction

Solve the affine direction \(a\) with target zero:

\[
J_\rho a=-\begin{bmatrix}r_d\\r_p\\r_e\\s\circ y\end{bmatrix}.
\tag{3}
\]

Use the raw verified affine direction, before the existing tiny-component
direction edits. Compute one common affine fraction

\[
\alpha_a=\min\left(1,
\min_{a_{s,i}<0}\frac{-s_i}{a_{s,i}},
\min_{a_{y,i}<0}\frac{-y_i}{a_{y,i}}\right).
\]

Empty minima are infinity. Fraction one here estimates the affine boundary;
it is not an accepted iterate and no GLM objective needs evaluation there.
The final combined step retains the existing fraction-to-boundary safety
factor. Separate primal/dual affine fractions are a different choice, not
required for this first experiment.

Set

\[
\bar\mu=\frac{s^Ty}{k},\qquad
\mu_a=\frac{(s+\alpha_a a_s)^T(y+\alpha_a a_y)}{k},\qquad
\sigma=\operatorname{clip}(\mu_a/\bar\mu,0,1)^3.
\tag{4}
\]

Require finite positive input slacks/duals and finite quantities in (3)-(4).
A boundary estimate can acquire tiny negative components through rounding;
record any numerical treatment explicitly and do not mask a genuinely invalid
direction or overflow by silently setting sigma to zero.

Normally choose

\[
\mu_t=\max(\sigma\bar\mu,\mu_{\min},\kappa R),\qquad
\kappa=\texttt{barrier_kkt_floor}.\tag{5}
\]

The residual floor uses R at the current point, before the step. Since the
floor may dominate, the final target need not lie below mean complementarity.
Record sigma, its proposed target, each floor and the selected target
separately. Compare \(\kappa=0.01\) and zero as declared independent arms;
do not infer the benefit of predictor-corrector from a simultaneous floor
removal.

With \(q=a_s\circ a_y\), solve the **combined** direction

\[
J_\rho d=-\begin{bmatrix}
r_d\\r_p\\r_e\\s\circ y-\mu_t\mathbf1+q
\end{bmatrix}.\tag{6}
\]

In the existing assembly API this means
`rc_effective = s*y - mu_target + ds_aff*dy_aff`. The other residual inputs
remain unchanged. Equation (6) already contains the affine contribution;
do not add the affine direction again. Equivalently, a corrector-only solve
has zero first three RHS blocks and last RHS \(\mu_t\mathbf1-q\), and its
solution can be added to \(a\). Implement one convention consistently.

The cross product uses full affine directions, not their positivity-limited
versions: do not insert \(\alpha_a^2\) into q merely because (4) uses a
limited affine estimate. Damping the correction is an optional later policy.
There is no additional rho cross term, since all rho contributions to (1)
are linear for fixed rho and centers.

The predictor/combined construction is motivated by the practical equations
in the [PIQP paper, equations 6-11](https://arxiv.org/pdf/2304.00290v2).
That paper uses separate damped affine fractions and explicit proximal-center
updates; our common affine fraction and retained GLQP globalization are
declared choices, not a claim to reproduce PIQP or inherit its behavior.

## Target lifetime and API semantics

The target in (6), the log-barrier merit, its directional derivative, and
any perturbed-residual acceptance test must be the same \(\mu_t\). Select it
before calculating those values and hold it fixed during backtracking.
After acceptance retain that target for the point/history. Recompute the next
target from the next affine predictor. The old post-step centrality proposal
and `barrier_update_guard`/`barrier_reduction_limit` do not apply in this mode.
Applying both target updates would introduce an unintended second policy.

Honor an explicit user `mu0` as the **first combined target**, then use (5)
on later iterations. Report this first-step override distinctly from sigma
selection. Without explicit `mu0`, fixed/complementarity initialization of
the old target is Newton-only and has no algorithmic role in this strategy;
mark it inactive in diagnostics and documentation. The implementation uses
`min_mu` as an initialized-point observation placeholder, labels its source
`awaiting_predictor`, and does not evaluate the inactive initialization
policy. That placeholder never determines the first direction. An explicit
`mu0` instead remains authoritative for the first combined target. The first combined override and its
input validation should be tested explicitly.

The dummy/empty-inequality path retains its existing solve and target rules;
there is no meaningful mean complementarity or affine correction to compute.

## GLMs and the existing line search

For a GLM keep H evaluated at the current point during both solves. The
combined direction is a candidate, not a second accepted iterate. Evaluate
the actual nonlinear \(f(A(x+t d_x))\), and any needed actual gradient, in
the current line search. Recompute H after accepting a new point as before.
Do not substitute a quadratic GLM model for the actual trial objective.

The current merit and its derivative are

\[
\Phi_{\mu_t}(x,s)=F(x)-\mu_t\sum_i\log s_i,\qquad
\Phi'_{\mu_t}(d)=\nabla F(x)^Td_x-\mu_t\sum_i d_{s,i}/s_i.
\]

The correction can turn a descent direction into a non-descent direction.
Even a centering Newton direction is not automatically descent for this
particular merit when feasibility is inconsistent and equations are
regularized. Keep the existing acceptance branches for the controlled trial,
and record the actual combined slope, trials, accepted branch and step.
Do not give the affine predictor a free accepted step or bypass the GLM
objective test because it is called predictor-corrector. Retaining the
current merit test does not constitute a new global-convergence theorem.

If GLM regressions identify the correction as the problem, a coherent later
safeguard is a same-target **centering Newton rescue**: remove q from (6),
perform another verified solve using the same factor, and line-search that
direction from the original point. Its extra solve and trials must be counted.
This is preferable to switching the target halfway through a failed line
search. It is not part of the first minimal experiment, and it still does
not guarantee descent under the current infeasible merit.

The established nonlinear filter approach holds the barrier parameter fixed
within a search and combines acceptance with additional mechanisms such as
restoration. Its nonlinear-constraint second-order correction addresses a
different residual from our complementarity correction; GLQP's constraints
are affine. See [Wachter and Biegler, sections 2.2-2.4](https://cepac.cheme.cmu.edu/pasi2011/library/biegler/ipopt.pdf).
Do not borrow a convergence claim by copying only its line-search label.

### Optional nonlinear stationarity correction

The only nonlinear residual in a QP's unregularized KKT map is
complementarity. For a nonlinear GLM, stationarity also has a second-order
Taylor term. If third derivatives exist, its leading term along a is

\[
q_d=\tfrac12\nabla^3F(x)[a_x,a_x]
=\tfrac12 A^T\left(f'''(Ax)\circ(Aa_x)^2\right).
\tag{7}
\]

A full second-order residual-correction variant would use
\(-r_d-q_d\) in (6), along with the same complementarity correction q.
For Gaussian loss this additional term is zero. Logistic and Poisson losses
generally have a nonzero term; the current loss API exposes first/second
derivatives, not third derivatives.

Omitting (7) makes (6) a complementarity-corrected Newton heuristic for a
nonlinear GLM; it does not make its computed direction invalid. It needs
empirical GLM validation under the actual line search. Evaluating the exact
gradient remainder at a full affine point is not a free substitute: that
point may overflow a Poisson evaluation, and dividing a damped remainder by
a tiny squared step can amplify cancellation. Keep this extra correction a
separate measured experiment.

## What current regularization means physically

Introduce fixed centers \((\bar x,\bar s,\bar y,\bar\nu)\) and consider

\[
\begin{aligned}
\widetilde r_d&=r_d+\rho(x-\bar x),\\
\widetilde r_p&=r_p-\rho(y-\bar y),\\
\widetilde r_e&=r_e-\rho(\nu-\bar\nu),\\
\widetilde r_c&=s\circ y-\mu\mathbf1+\rho(s-\bar s).
\end{aligned}\tag{8}
\]

If all centers equal the iteration's starting point, the residuals there
equal the original residuals, and their Jacobian is exactly (1). This is an
algebraic fixed-center interpretation of the current requested direction.
Hold those centers and rho fixed across predictor, combined solve and search
when using that interpretation. Resetting centers after the affine predictor
would change the problem whose residual is being corrected.

The first three residuals resemble a standard primal-dual quadratic proximal
formulation. The fourth does **not**: ordinary PIQP-style proximal
regularization leaves \(s\circ y-\mu\mathbf1\) unchanged. In particular,
adding a simple quadratic slack penalty does not produce this fourth block.

There is nevertheless a precise nonstandard slack-penalty interpretation.
Define, for positive s and fixed positive \(\bar s\),

\[
D(s;\bar s)=\sum_i
\left(s_i-\bar s_i-\bar s_i\log(s_i/\bar s_i)\right).
\]

This is nonnegative, convex in s, and has zero gradient at the center. The
saddle function

\[
\begin{aligned}
\mathcal L={}&F(x)+y^T(Cx+s-c)+\nu^T(Ex-e)
-\mu\sum_i\log s_i\\
&+\tfrac\rho2\|x-\bar x\|^2
-\tfrac\rho2\|y-\bar y\|^2
-\tfrac\rho2\|\nu-\bar\nu\|^2+\rho D(s;\bar s)
\end{aligned}
\]

has slack stationarity
\(y-\mu/s+\rho(1-\bar s/s)=0\). Multiplication by S gives precisely the
fourth equation in (8). This derivation supplies an algebraic Bregman-type
slack penalty; it is not a claim that the existing solver implements or
globalizes that proximal subproblem.

The current line search evaluates only F minus the log barrier. It does not
evaluate the quadratic/Bregman terms or the fixed-center residuals (8) at
trial points, and there is no persistent center-update policy. A genuine
proximal strategy would require those decisions explicitly. For example,
eliminating the duals from (8) yields a primal augmented-Lagrangian objective
containing the constraint penalties as well as these primal/slack terms;
that objective is different from the current merit. Keep this larger change
separate from the first predictor-corrector experiment.

The [IP-PMM paper](https://www.pure.ed.ac.uk/ws/portalfiles/portal/174251471/ProxIPM_22oct20.pdf)
analyzes a specified proximal method and explicitly distinguishes its
practical predictor-corrector implementation from that analysis. Its
guarantees cannot be transferred to the current GLQP hybrid just from the
presence of diagonal shifts.

## Verification and diagnostics

Distinguish three objects:

1. The requested linear residual for (6), including q and rho. Verify both
   solves separately against their requested RHS and recover all four blocks.
2. The original Newton defect at the chosen target, without q or rho. Before
   clipping, its expected block contributions are
   \((-\rho d_x,\rho d_y,\rho d_\nu,-q-\rho d_s)\), plus actual solve error.
   If (7) is later used, stationarity also contains \(-q_d\). Keep those
   intended correction contributions separate from regularization and
   clipping; otherwise telemetry falsely reports the corrector as solve error.
3. The actual nonlinear trial residual at fixed target \(\mu_t\). It contains
   neither an artificial q term nor a subtraction of rho effects. This is
   the residual used by any existing original-equation progress test.

Log the affine fraction/mean estimate/sigma, selected target and override,
cross-product magnitude, both linear statuses, factor/update and solve
counts, original block defects, combined clipping changes, and line-search
acceptance. Preserve observation-on/off solution parity. Factor reuse must
actually avoid a second numeric update; reusing only a Python factor handle
through today's refactor-and-solve entry point does not establish that.

Check algebra on small QP and GLM systems, equality/dummy paths, explicit
first-target override, and a fixed-rho affine-plus-corrector identity before
performance work. Use the existing GLM nonlinear-search regression tests.
Then measure complete cost: one factor plus two verified solves, any added
search work, and changes in outer iteration count. A lower iteration count
alone does not prove a speedup.
