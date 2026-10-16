# Standalone presolver: scalar-quadratic iteration

The scalar pass provides useful local gains, with unchanged 129/138 coverage.
Across those 129 common successes, full scalar-enabled presolve reduced geometric
wall time by 4.5% versus presolve off. Its incremental gain over existing local
presolve is 1.0% overall and 8.1% on the nine problems it actually changes. These
full-suite numbers use one observation per problem; the focused repeated results
below establish the larger wins and expose regressions. Presolve stays opt-in.

## Implementation and numerical scope

The dependency-free C++17 library accepts full symmetric float64 QPs in
canonical CSC form. A C ABI and `glqp-presolve` text CLI are usable without
Python. The Python adapter packs buffers, calls the library, copies its
results, and exposes primal/equality-dual/inequality-dual reconstruction.
`PresolveOptions` independently switches singleton equalities, fixed bounds,
redundant bounds, local arithmetic fallback, and scalar-quadratic minima.
All new options remain disabled by default.

Commit `a6f9e36` adds `separable_quadratics=True`: for an originally diagonal
positive Hessian column with only singleton inequalities remaining, minimize
the scalar quadratic over its bounds. Outward quotient intervals certify
whether the unconstrained minimum is interior or clipped to one bound.
Ambiguous classification keeps the variable. The first version deliberately
excludes Hessian coupling to previously fixed variables, which would require
intervals for a shifted objective coefficient.

The bound multiplier is reconstructed with the original row scale. Reversing
the elimination tape propagates its contribution to earlier fixed variables
before their multipliers are recovered. Original primal feasibility,
stationarity, complementarity, dual signs, and objective are the final checks.
The library does not certify convexity and assumes a convex input QP.

Commit `9d59be9` removes unnecessary tape entries for interior scalar minima.
Their primal values already live in the fixed-value array, and they require
no multiplier. The existing empty-tape path then skips recovery CSR construction
and stationarity products when all eliminations are interior minima.

Native option fields are experimental ABI: rebuild external C/C++ clients
with the library, including when a field occupies former structure padding.
Python's source/header/compiler hash selects a matching native build.

## Correctness and structural effect

All 261 Python tests pass. Native AddressSanitizer, UndefinedBehaviorSanitizer,
and leak detection pass 39,097 assertions. The suite covers exact and scaled
bounds, reversed equality/bound cascades, empty and zero-variable models,
malformed inputs, overflow, guarded contradictions, ambiguous activity, and
randomized original-coordinate KKT checks. Two independent source reviews
found no numerical blocker.

With the scalar option disabled, every transformed matrix, vector, map,
objective offset, and recovered array is bitwise identical to `d9bf350` on
all 138 bundled QPs. With it enabled, the storage optimization remains
bitwise identical to `a6f9e36` across the same inputs.

With local fallback and scalar minima enabled, 49 of 138 QPs reduce, with no
whole-problem fallbacks. The scalar pass removes 6,546 additional variables
across nine problems. Total removals become 11,414 variables, 4,518 equality
rows, and 17,955 inequality rows. These are structural counts, not speedups.

| Problem | Additional scalar variables | Variables after local / after scalar | Inequalities after local / after scalar |
| --- | ---: | ---: | ---: |
| MOSARQP1 | 1,743 | 2,500 / 757 | 3,200 / 1,457 |
| MOSARQP2 | 267 | 900 / 633 | 1,500 / 1,233 |
| PRIMAL1 | 125 | 325 / 200 | 86 / 86 |
| PRIMAL2 | 254 | 649 / 395 | 97 / 97 |
| PRIMAL3 | 72 | 745 / 673 | 112 / 112 |
| PRIMAL4 | 242 | 1,489 / 1,247 | 76 / 76 |
| Q25FV47 | 1 | 1,546 / 1,545 | 1,848 / 1,847 |
| STCQP1 | 2,560 | 3,830 / 1,270 | 7,660 / 2,540 |
| STCQP2 | 1,282 | 2,045 / 763 | 4,090 / 1,526 |

## Measurement protocol

The complete-solve campaigns use one frozen source revision for three GLQP
variants: presolve off; presolve with local fallback; and the same options
plus scalar minima. The numerical solver and QDLDL/float64 GMRES settings
are identical. Full sweeps use all 138 MM QPs, the standard `parallel-8`
profile, one numerical thread per worker, and a 60-second wall budget.
Our numerical jobs are serialized during each timed campaign. Compiler work,
tests, profiling, source editing, and other agent numerical jobs are paused.
Host activity is sampled separately; complete waves affected by the declared
CPU-contention threshold are excluded and replayed. The selected data use the
first clean complete wave, never the fastest or most successful attempt.
The fresh series uses a uniform two-second pause before cold worker launches,
outside measured solver time, to reduce bursts of host-service activity.

Wall time includes packing, native presolve, model construction, solve, and
original-coordinate recovery. Native compilation occurs beforehand and its
source/compiler/flags/binary hashes are recorded. Original-data validation
and saved-solution compression occur after the solve timer. Status counts
are reported separately; timing ratios use common problems accepted on every
compared repetition, preventing failures from entering speedup averages.

The three-repeat focused cohort is MOSARQP1, MOSARQP2, PRIMAL1, PRIMAL2,
PRIMAL4, STCQP1, STCQP2, and QFFFFF80. It covers bounded and interior minima,
small and large reductions, and the known equality-elimination regression.
PRIMAL3 and Q25FV47 have the smallest new structural opportunities and are
excluded from the repeat cohort unless the full sweep exposes a concern.

## Component measurements

The storage optimization was measured alone on CPU 10 with 101 alternating
observations per version and problem. PRIMAL1 recovery fell from 37.1 to
31.8 microseconds and PRIMAL4 from 51.9 to 34.5 microseconds. A synthetic
100,000-variable independent diagonal QP fell from 616 to 80.1 microseconds.
MOSARQP1, STCQP2, QSHELL, and an unchanged tridiagonal control stayed within
4% of their reference times. Every recovered array was bitwise identical.
These component results do not measure the benefit of enabling presolve.

## Complete-solve outcomes

All variants solve the same 129 of 138 QPs within 60 seconds and pass the
original-data KKT checker at 1e-6, with native GLQP tolerance 1e-7. Each has five
timeouts and four inaccurate results. There are no newly accepted or lost cases.

| Variant | Common successes | Geometric wall ratio to presolve off |
| --- | ---: | ---: |
| Presolve off | 129 | 1.0000 |
| Presolve with local fallback | 129 | 0.9640 |
| Local fallback plus scalar minima | 129 | 0.9548 |

The scalar/local ratio is 0.9904 over all 129 common successes, 0.9193 on the
nine scalar-reduced problems, and 0.9960 on the other 120. Small differences
near 1% in this one-observation sweep should not be treated as precise effects.

Focused confirmation uses three cold observations per variant and problem,
rotating solver order on the same CPU. Every repetition passes original KKT
validation. The table gives median complete-solve wall time in milliseconds.

| Problem | Off | Local | Local + scalar | Observation |
| --- | ---: | ---: | ---: | --- |
| MOSARQP1 | 24.63 | 25.45 | 20.49 | Scalar pass reduces local time 19.5% |
| STCQP1 | 103.09 | 42.63 | 28.09 | 3.67x faster than off; 34.1% less time than local |
| STCQP2 | 177.47 | 25.48 | 21.85 | 8.12x faster than off; 14.3% less time than local |
| PRIMAL4 | 40.24 | 40.29 | 44.70 | Scalar pass is 11.0% slower than local |
| QFFFFF80 | 84.97 | 2465.03 | 2447.50 | Existing equality-elimination regression remains |

MOSARQP2 and PRIMAL1–2 show no convincing material improvement in these repeats.
PRIMAL4's regression is mostly in the solve: median solve/recovery time rises
39.09 to 43.11 ms; presolve itself rises 0.34 to 0.74 ms. An untimed diagnostic
finds the same 26 outer iterations, 26 factorization attempts, and 17 refinement
steps. Factor nonzeros decrease from 20,372 to 20,130. Thus neither extra outer
iterations nor increased factor fill explains the regression; its remaining
cause is not isolated. This observation prevents a blanket promotion of the
new pass even though it substantially benefits the STC and MOSARQP1 cases.

## Limitations and configuration

The earlier QFFFFF80 equality-elimination regression remains: the previous
repeated diagnosis changed 68 iterations to 1,751 after singleton-equality
elimination. Restoring removed constant inequality rows did not cure it.
The new scalar pass finds no eligible variable there and does not address
that convergence effect. `singleton_equalities=False` is an explicit general
opt-out; there is no problem-name heuristic.

The standalone CLI has the new pass explicitly wired:

```sh
cmake -S glqp/_native -B build/presolve -DCMAKE_BUILD_TYPE=Release
cmake --build build/presolve
build/presolve/glqp-presolve --local-fallback --separable-quadratics input.gqp > reduced.gqp
```

Running `glqp-presolve input.gqp` leaves both `local_fallback` and
`separable_quadratics` disabled. To keep singleton equalities while testing
the other passes, add `--no-singleton-equalities`. This is the general
configuration for the QFFFFF80 regression control; it is not selected
automatically by the problem name. The corresponding Python options are:

```python
PresolveOptions(local_fallback=True, separable_quadratics=True,
                singleton_equalities=False)
```

The separate three-repeat QFFFFF80 control confirms the escape hatch on the
final source: off takes 84.10 ms and 68 iterations, scalar-enabled presolve takes
2496.11 ms and 1751 iterations, and retaining singleton equality rows takes
84.02 ms and 68 iterations. Every recovered certificate passes original KKT
validation. This is a general switch, not a problem-name heuristic; disabling
equality elimination globally may sacrifice useful reductions on other models.

Keep the default unchanged. Enable the scalar pass explicitly when its
structural opportunity is relevant, and check complete-solve behavior. The
benchmark companion report records all failures and both improvements and
regressions, rather than recommending presolve solely from reduced dimensions.

## Evidence

- [API, build, and standalone format](../../presolve.md)
- [All138 enabled structural identities](iteration-5-structural.json)
- [Default-option compatibility](iteration-5-default-compatibility.json)
- [Storage-iteration compatibility](iteration-6-compatibility.json)
- [Final-source native sanitizer record](iteration-6-sanitizer.json)
- [Exclusive recovery diagnostic](iteration-6-recovery.json)
- [Earlier equality-elimination regression diagnosis](qfffff80-regression.md)
- [Audited complete-solve sweep](iteration-7-full.json)
- [Audited three-repeat panel](iteration-7-repeats.json)
- [Audited equality control](iteration-7-equality-control.json)
- [Untimed PRIMAL4 work counts](iteration-7-primal4-work.json)

`summarize_solve_panel.py --run CAMPAIGN --report APPROVED_REPORT --output SUMMARY.json`
generates source-linked paired statistics from completed campaigns.
It follows the approved whole-wave selection, requires original-coordinate
artifact audits, and excludes accurate results exceeding the wall budget
from accepted timings. Run it after the timing lock is released.
