# Experimental predictor–corrector

The default remains `step_strategy="newton"`. Select
`SolverSettings(step_strategy="predictor_corrector")` for the experimental
two-RHS strategy. It applies only to real inequalities; equality-only and
unconstrained problems retain their previous single-RHS path.

The numerical implementation was frozen at `2c33342`. Detailed derivation,
GLM considerations, physical-proximal interpretation, and primary references
are in [the design note](../research/predictor-corrector-design-20261016.md).
The sibling benchmark repository contains the complete checkpoint at
`benchmarks/progression/predictor-corrector-20261016/README.md`.

## Target selection and linear solves

An affine solve targets zero complementarity. Its common positivity-limited
step predicts `mu_aff`; the centering factor is
`clip(mu_aff/mean(s*y), 0, 1)**3`. The combined solve uses the effective
complementarity residual `s*y + ds_aff*dy_aff - target`, where the target is
the maximum of the predicted target, `min_mu`, and `barrier_kkt_floor*KKT`.
The combined direction is a full direction, not a correction to add twice.

Both RHSs use the same current Hessian, slacks, multipliers, and outer rho.
`VerifiedLinearSystem` reuses the numerical factorization, while independently
verifying both solutions against the requested matrix with float64 right-GMRES.
The existing `factor_and_solve` path remains unchanged for default Newton.
The factorization-only shift is excluded from the requested residual target.
Failure on reuse is explicit; no silent refactor changes the pair's matrix.

An explicit `solve(mu0=...)` governs the first combined target; subsequent
targets come from the predictor. The old initial-target policy and post-step
barrier update do not govern PC. With no explicit `mu0`, initialized-point
diagnostics label their `min_mu` placeholder `awaiting_predictor`.
Explicit PC `mu0` must be finite and nonnegative. `barrier_update_guard` and
`barrier_reduction_limit` govern only the old post-step policy and are inactive
for PC. The chosen target stays fixed until the next predictor.

The target is chosen before constructing the combined direction and held
through its merit and progress tests. Existing tiny-component edits apply
only to the combined direction. Current nonlinear objective evaluations,
nonfinite-trial rejection, and line-search policies remain in force. No GLM
third-derivative stationarity correction is assumed.

History includes both refinement counts and linear statuses, affine step,
sigma, target, and factorization attempts. With `iteration_diagnostics=True`,
the `predictor_corrector` record exposes the pre-direction target decision
and affine defects; the post-step `barrier` record marks the held target.
Combined-direction original Newton defects include the intentional corrector
term, while requested-system defects subtract both it and the rho effect.
This keeps an intentional correction distinct from linear-solve error.

## Experiment results

All observations use native tolerance 1e-7 and an independent original-data
certificate at 1e-6. Routine timing uses the accepted parallel-8 configuration,
one native thread each, with all other agent jobs paused.

- Regularized PC passes all 22 selected QPs: geometric iteration ratio .4863,
  wall ratio .7430, and solve ratio .7390 against accepted Newton. It improves
  20 iteration counts, ties HS51, and worsens LISWET1. The 14 cases outside
  the original development panel show approximately the same wall speedup.
- PC passes all 39 GLMs. On the 26 constrained cases, iteration ratios are
  .6145 with the default KKT floor and .5180 with `barrier_kkt_floor=0`.
  The 13 free controls retain their iteration counts. All GLM timing figures
  are excluded after host-activity flags; no GLM speed claim is established.
- Setting `prox_reg=0` changes none of those GLM iteration counts. It reduces
  PC iterations on QPCBOEI2 from 3241 to 56, LISWET1 from 7969 to 70, and YAO
  from 534 to 66 with the floor enabled, but loses QSHELL and worsens CONT-101
  from 22 to 96. The zero-rho options pass only 21/22 QPs.
- Six GLM diagnostic solves reproduce plain solutions and counts. All 25 PC
  steps have negative slopes and Armijo acceptance, with no Hessian changes
  inside either solve pair. Poisson's long initial line search persists.

These one-repeat screens overlap and are not full Maros–Mészáros coverage.
Keep PC opt-in, retain default regularization, and preserve the counterexamples.

## Regularization and next experiments

`prox_reg=0` removes requested-equation regularization. The independent
factorization-only `tau_reg=2e-8` remains active, and residual verification
still targets the requested system. This is not an unstable raw-factor solve.

Current rho modifies stationarity, primal, equality, and complementarity
blocks; the last contains `(Y+rho I) ds + S dy`. There is no explicit
proximal-center state and the merit function does not contain a matching
proximal objective. Thus these are regularized Newton directions, not an
implemented physical proximal subproblem with consistent globalization.

A genuine proximal subproblem can use PC, provided its centers, residual
definition, and regularization stay fixed during both solves and line search.
Center/rho updates belong at a separate outer policy boundary. Next isolate
the current regularization blocks, for example retaining primal/dual shifts
while restoring `Y ds + S dy`, and audit `rho*dy` relative to the original
primal residual. Disabling every block globally is not supported by the data.

A separate captured pre-existing LISWET1 issue makes the merit slope nonfinite
when `-mu/s` overflows at subnormal slacks, despite a representable final dot
product. It affects both Newton and PC. No fix was mixed into these runs;
the benchmark checkpoint retains a state/reproducer and an overflow-resistant
evaluation analysis. This is another isolated follow-up, not proof that its
arithmetic fix alone improves convergence.

## Verification

424 solver tests pass, including independent dense four-block comparisons,
nonzero-rho correction algebra, target timing/override tests, nonlinear
Gaussian/logistic/Poisson cases, overflow rejection, factor reuse and fresh
residual checks, and default/equality-only parity. Eight default-path solves
against frozen `48554fb` preserve solution arrays, non-time history, and
termination exactly. The numerical checkpoint is followed only by documentation.
