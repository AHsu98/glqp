# Independent outer-policy experiments

These controls isolate the experiments following the previous-target guard
study. The accepted defaults remain fixed initialization, legacy guard and
slack repair, finite-only Newton fallback, and KKT floor coefficient 0.01.
Predictor–corrector, clipping, regularization and stopping tolerances are unchanged.

| Setting | Values | Accepted default |
| --- | --- | --- |
| `initial_mu_policy` | `fixed`, `complementarity` | `fixed` |
| `initial_mu_scale` | finite positive multiplier | `1.0` |
| `barrier_update_guard` | `legacy`, `previous` | `legacy` |
| `barrier_kkt_floor` | finite nonnegative coefficient | `0.01` |
| `slack_repair` | `legacy`, `off`, `guarded` | `legacy` |
| `repair_growth_limit` | finite factor at least 1 | `1.1` |
| `newton_fallback_policy` | `finite`, `progress`, `off` | `finite` |

`complementarity` sets the initial target to
`max(min_mu, initial_mu_scale * dot(s,y) / k)` using the initialized or
warm-started vectors. `fixed` retains 100 for real inequalities. Explicit
`solve(mu0=...)` overrides the policy. Absent/empty inequalities retain their
minimum-target behavior. The scale does not affect the fixed policy.

The barrier floor is `barrier_kkt_floor * pre_repair_KKT`; zero disables this
floor while preserving the centrality proposal, minimum target, and selected
previous-target/legacy guard. Vary it independently of initialization.

`guarded` retains the existing repair trigger and proposed slack values. It
constructs separate candidate storage and requires positive finite values and
both of these bounds before replacing the accepted point:

```text
candidate_primal_inf <= max(repair_growth_limit * accepted_primal_inf, tol)
candidate_KKT_inf    <= max(repair_growth_limit * accepted_KKT_inf, tol)
```

Here KKT uses unperturbed `s*y`. The absolute allowance is explicitly the
unchanged native tolerance. Rejection retains the pre-repair accepted point
and selected target. `off` disables repair entirely. Since this repair raises
small products only to 0.1 times their mean, it cannot increase the maximum
product; the primal bound is the operative restriction, while the KKT bound
is a defensive check. Repair changes auxiliary slack, so its immediate effect
on the original-data certificate based on x/duals differs from its effect on
the stored-slack residual.

`progress` restricts only the existing Newton fallback. It compares

```text
phi_mu = max(norm(rd, inf), norm(rp, inf), norm(re, inf), norm(s*y - mu, inf))
```

at the current and trial points, using the same current target. The fallback
requires `trial_phi < (1 - armijo_param * step) * current_phi`, or
`trial_phi < tol`. Existing Armijo and unperturbed-residual acceptance remain
independent routes; this is not a globally monotone residual method. `off`
disables the fallback. The 0.9 backtracking factor and `let_newton_cook` step
threshold remain unchanged, keeping contraction and acceptance separate.

These controls validate both at construction and at solve entry. Diagnostics
retain policy metadata, initial-target selection, proposed/applied/rejected
repairs and candidate bounds, and each fallback progress check. Enable
`iteration_diagnostics` only in untimed observation runs. Its extra matrix
products, storage and callback validation count toward the solver time budget.

The benchmark repository's `configs/outer-policy-v1.json` declares the initial
five groups; `tools/outer_policy_study.py` freezes each group and runs prepared
groups sequentially under the host guard. Every group uses the same eight QPs,
parallel-8 with one native thread per worker, rotating variant order, native
tolerance 1e-7, independent tolerance 1e-6, and a two-second untimed launch
delay. A guard flag retains the attempt for review and never approves its times.

## Completed experiments, 2026-10-16

Source `f685076` was tested in five initial eight-QP screens, two
initialization/floor interaction screens, and a two-QP backtracking check.
The mean-initialization/legacy-guard candidate also passed all 39 GLMs and
all 22 quick QPs/applications. Accepted defaults remain unchanged.

- Mean initialization eliminates the LISWET2/6 regressions associated with
  fixed 100 plus the previous-target guard (758/2735 → 18/20 iterations).
  With the legacy guard it worsens LISWET1 6604 → 8812 and YAO 2555 → 3380.
  Geometric iteration counts fall 6.7% on GLMs and 7.5% on the quick suite;
  those campaigns' timings are excluded after host-activity flags.
- Guarded repair prevents QPCBOEI2's iteration-119 KKT increase
  0.0101868 → 0.0387648, but changes its final count only 3323 → 3322.
  A later finite fallback increases KKT independently of repair.
- Progress-gated and disabled fallback both lose LISWET1 and QPCBOEI2.
  Extending progress fallback from 40 to 100 line-search trials rescues
  neither. Failed-direction telemetry points to outer regularization and
  original-coordinate Newton defects: scaled linear accuracy alone does
  not establish sufficient original-equation progress.
- Reducing/removing the floor and combining a smaller floor with two
  initialization candidates yields mixed or worse results. QPCBOEI2 stays
  near 3300 iterations; floor activation alone did not identify the cause.

The benchmark repository's
`benchmarks/progression/outer-policy-20261016/README.md` contains results,
focused timing audits, exclusions, failed-step diagnostics, and raw-artifact
hashes. Four diagnostic trajectories match plain solutions/counts bitwise.
All 367 solver tests pass, including 67 new policy tests. Predictor–corrector,
clipping, regularization, and native stopping criteria remain unchanged.
The next priority is controlled direction-accuracy/regularization experiments.
