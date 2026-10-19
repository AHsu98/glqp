# GLM predictor and original-operator audit, 2026-10-19

This is an untimed numerical diagnostic, not a performance comparison. It uses
the accepted runtime at `1c10c00`, with only the observational integration and
new helper copied into a frozen source tree. The concurrent weighted-loss
arithmetic experiments are excluded. Source hashes before and after both runs
match. Raw results and frozen source are in
[`glqp-bench/runs/glm-operators-20261019`](../../../glqp-bench/runs/glm-operators-20261019).

## Diagnostic contract

`iteration_diagnostics=True` now adds two observations for GLMs:

- Initial and accepted-point `glm_predictor` compares the carried predictor
  `z` with fresh `A@x`, and measures the resulting scalar-gradient and original
  stationarity changes. It includes the last accepted point, even when the next
  loop terminates before computing another direction.
- Per-direction `glm_hessian` compares formed `H@dx` with
  `Q@dx + A.T@(f''(z)*(A@dx))`, using the same carried predictor and actual primal
  direction. It records both action norms, their difference, Rayleigh products,
  and curvature range. These are two float64 evaluations, not an exact reference.

These checks exclude physical proximal terms, numerical factor shifts, and
constraint blocks. Existing direction diagnostics separately report requested
regularization defects and slack/multiplier clipping. That clipping does not
alter `dx`. The helpers do no work for QPs or when diagnostics are disabled, and
their results do not affect stopping, directions, or acceptance.

## Severe problems

All twelve severe cases were solved twice at each native tolerance, with
diagnostics off and on. All **24 pairs** have identical `x,y,s,nu`, termination,
and common non-time history. Independent `glm-kkt-v2` validation remains at
`1e-6`: native tolerance `1e-7` certifies **7/12**, and `1e-9` certifies **12/12**.
Changing native tolerance also changes the requested linear-solve accuracy;
these controls do not isolate outer stopping from inner solve accuracy.

| Five gap-only cases at native `1e-7` | Iterations at `1e-7` → `1e-9` | Final stationarity change from fresh predictor at `1e-7` |
|---|---:|---:|
| Conditioned Gaussian, free | 1 → 2 | 0 |
| Nearly separated logistic, free | 23 → 27 | 4.14e-21 |
| Poisson range, affine | 14 → 16 | 6.83e-16 |
| Poisson range, box | 15 → 18 | 8.00e-16 |
| Poisson range, free | 9 → 11 | 1.90e-15 |

Predictor drift is therefore not an explanation for these five failures. Across
their native-`1e-7` trajectories, the largest Hessian action discrepancy is
`7.08e-13` absolute and `5.92e-10` relative to the original action. In the tighter
conditioned Gaussian run, the relative difference reaches `9.82e-7` on a small
late direction, while the absolute difference is `3.57e-15`; the final point
passes the certificate. These direction-specific observations do not bound
error in other weak modes.

The three successful, severely scaled Gaussian cases amplify tiny predictor
drift into stationarity changes up to `6.18e-11` along their trajectories. That
is below the tested tolerances, but could matter if tolerances are tightened
further. The present evidence does not justify an unconditional fresh predictor
product each accepted iteration.

## Constructed Gram-loss controls

Two original-design controls reproduce complete erasure of a weak direction:

| Design `A`, ridge `Q` | Direction | Formed action infinity norm | Original-design action infinity norm |
|---|---|---:|---:|
| `[[1,1],[1e-8,0]]`, zero | `[1,-1]` | 0 | 1.0000000000000001e-16 |
| `[[1e8,1e8],[1,0]]`, `1e-7 I` | `[1,-1]` | 0 | 1.0000001 |

The risk is real, but these controls do not establish that Gram formation causes
the suite's remaining gap failures. There is still no evidence here for replacing
the formed Hessian default. The new diagnostic supplies a cheap-to-request
measurement when a future problem is suspected to suffer this failure mode.

## Explicit Newton fallback control

A one-variable logistic problem starts at `x=5`, with labels `[0,1]`, weights
`[.9,.1]`, design `[[1],[1]]`, and ridge `1e-4`. Its independent scalar optimum
is `-2.1947882989601757`. All controls use the same GLM settings except the
fallback policy, with `let_newton_cook=.5`, `max_iter=100`, and
`max_stagnation=20` to keep the known failure bounded.

| Policy | Result | Iterations | Final original gradient | Fallback acceptances |
|---|---|---:|---:|---:|
| `off` | optimal | 9 | 1.09e-12 | 0 |
| `finite` | stagnated, `x=999.0437` | 22 | 0.999904 | 12 |
| `progress` | optimal | 10 | 2.64e-11 | 1 |

The finite fallback accepts insufficiently contracted nonlinear steps. The
progress policy repairs this particular control, but it is not a proof of safe
globalization for all GLMs. Standard GLM profiles keep the fallback inactive;
this audit does not change any acceptance policy or default.

Reproduction from a frozen source:

```bash
python tools/audit_glm_operators.py --source SOURCE --output NEW_OUTPUT --cpu 6
python tools/audit_glm_operators.py --source SOURCE --output NEW_TIGHT_OUTPUT --cpu 6 --tol 1e-9
```

The tool saves both solutions, complete diagnostics, independent certificates,
common-history parity, source hashes, and the bounded fallback controls.
