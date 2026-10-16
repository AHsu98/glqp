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

All settings validate both at construction and at solve entry. Diagnostics
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
