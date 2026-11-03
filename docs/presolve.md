# Native QP presolve

`glqp.presolve` is an opt-in structural presolver for
`min 0.5*x.T@Q@x + q.T@x + objective_constant`, with `A@x=b` and `G@x<=h`.
The C++17 library has no numerical-library dependency. Python only packs
canonical CSC buffers and copies outputs; sparse entry loops and primal/dual
reconstruction run in C++.

The current passes eliminate fixed variables from singleton equalities or
exactly matching opposite bounds, propagate those substitutions, remove
certified empty rows, and retain strongest singleton inequalities. Singleton
equality elimination, fixed-bound elimination, and redundant-bound removal
have separate switches; empty-row removal is part of propagation.
An additional opt-in pass minimizes independent positive scalar quadratics
over their remaining singleton bounds. No approximate equality or small
coefficient is used as an elimination rule. General aggregation, scaling,
coupled quadratic substitution, and GLM predictor substitution are outside
this version's scope.

```python
from glqp.presolve import PresolveOptions, prepare_native, presolve_qp

build = prepare_native()  # Call before starting a benchmark timer.
reduced = presolve_qp(Q, q, A, b, G, h,
    options=PresolveOptions(singleton_equalities=True,
                           fixed_bounds=True, redundant_bounds=True))
# Solve reduced.Q, reduced.q, reduced.A, reduced.b, reduced.G, reduced.h.
x, equality_dual, inequality_dual = reduced.recover(x_reduced, y_reduced, z_reduced)
# The original objective is the reduced objective + reduced.objective_offset.
reduced.close()
```

For the solver's convention, use `q=-problem.b`, `A=problem.E`, `b=problem.e`,
`G=problem.C`, and `h=problem.c`. A reduction to zero variables needs no
numerical solve: call `recover([])` and use the objective offset. Removed
inequalities receive zero duals, except an exactly fixed bound pair or an
active scalar-minimum bound, whose nonnegative multiplier is reconstructed. Equality pivots are
undone in reverse order, updating the stationarity residual along their
original sparse rows. This is necessary when singleton equalities emerge
after earlier fixed variables are removed.

Always validate feasibility, stationarity, complementarity, and dual signs
against original problem data. Convexity is assumed, as in the solver; the
presolver does not certify positive semidefiniteness. The full symmetric
Hessian is required, not just one triangle. All data must be finite.

## Conservative arithmetic

Substitution tracks outward-rounded intervals with `fma` product errors and
error-free sum remainders. Exact floating-point operations keep point
intervals. Ambiguous empty rows return `status="inconclusive"` and the original
unchanged QP. Overflow also falls back. A certified contradiction raises
`PresolveInfeasible`.

`PresolveOptions.infeasibility_guard` defaults to `1e-12`, relative to the
relevant right-hand side/bound magnitude (with minimum scale 1). It only
turns tiny contradictions into an inconclusive fallback. It never converts
nearby bounds into a fixed variable and never declares a contradictory row
redundant. Set it to zero for strict interval-certified contradiction tests.
This matters for rounded benchmark data: QPCBOEI1 contains bounds differing
by about one unit in the last place, and STCQP1 exposes a tiny contradictory
equality after substitution.

The experimental `PresolveOptions(local_fallback=True)` limits these fallbacks
to the affected variables where possible. Before accepting a fixation, the
presolver checks every neighboring row that would become constant. If one
would be ambiguous, it retains the variable and all pivot rows. It likewise
retains singleton bounds whose contradiction lies within the guard. Other
variables can still be eliminated, with the usual reverse dual recovery.
This preflight happens before any pivot is removed or right-hand side is
updated, including when an opposing bound pair supplies the fixation.

The result reports `deferred_variables` and `retained_ambiguous_rows`
separately from actual reduction counts. The latter counts retained rows
observed blocking an elimination or exposing a guarded bound conflict; it
is not an exhaustive classification of all dependent rows. Deferred
variables remain deferred for the entire call. If independent reductions
succeed, the status is `reduced`; if none succeed, it is `inconclusive`.
An initially empty row with a guarded contradiction, or arithmetic overflow,
still causes the original whole-problem fallback. The option is false by default, and
does not relax any constraint or increase the infeasibility guard.

## Independent scalar quadratics

`PresolveOptions(separable_quadratics=True)` additionally minimizes
`0.5*d*x*x + q*x` when `d>0`, the original Hessian column has no off-diagonal
entries, and every remaining constraint on the variable is a singleton
inequality. The optimum is `-q/d`, clipped to the strongest lower and upper
bounds. This removes a variable at its optimum; it does not assert that the
original feasible set fixed that variable. The remaining QP and objective
offset preserve the optimum.

The pass uses outward quotient intervals to distinguish an interior minimum
from a lower-bound or upper-bound minimum. Overlapping intervals retain the
variable. A preflight checks every other bound before removing any row or
changing reconstruction state. Unrepresentable scalar quotients likewise
retain that variable. `separable_variables` counts accepted minima, and
`separable_ambiguous_variables` counts otherwise eligible variables retained
because the arithmetic or an earlier deferred decision prevented certification.
The option is false by default and independent of `local_fallback`.

Interior minima need no additional multiplier. For a clipped minimum the
reverse tape reconstructs `z = -(d*x+q)/a` on its active bound `a*x<=h`,
including the original row scale. The sign is nonnegative by the certified
activity decision. Original bound rows may also contain variables fixed by
earlier equality or bound pivots; reversing the tape propagates their dual
contribution before reconstructing those earlier multipliers. Columns coupled
to an earlier fixed variable are intentionally excluded: supporting them
would require certified intervals for the shifted linear objective.

## Build and reproducibility

`prepare_native()` compiles lazily using `$CXX` or `c++`. The compiler must
support C++17 and Linux shared libraries. Source/header digests, compiler
identity, flags, and platform form the cache key. Builds use a process lock
and an atomic library replacement; eight cold benchmark workers can share
the result. The default cache is `/tmp/glqp-native-<uid>`; set
`GLQP_NATIVE_CACHE` to select a persistent location. Source trees are never
modified by compilation.

Build metadata includes the source SHA-256 and compiled library SHA-256.
The wrapper verifies the binary checksum before loading it. Both values
belong in experiment provenance. Native builds use `-O3 -ffp-contract=off`,
without fast-math or host-specific instruction flags. Build time is excluded
from a prepared solve timer. Presolve packing, native work, copying reduced
outputs, and primal/dual reconstruction belong in end-to-end solve timing.

The independent C API is in [`presolve.h`](../glqp/_native/presolve.h).
It copies input data, owns all reconstruction state, and has no global mutable
solver state. Free its opaque result with `gqp_free`. Output pointers are
valid until that call. `gqp_recover` handles primal and dual recovery.

Adding experimental option fields changes the native `gqp_options` layout.
Rebuild C and C++ clients together with this library; previously compiled
clients must not be mixed with the new layout, even when the structure's size
is unchanged because a new field occupies former padding. The Python build cache
includes both the header and source in its key, so this change selects a new
matching native build automatically.

To build without Python:

```sh
cmake -S glqp/_native -B build/presolve -DCMAKE_BUILD_TYPE=Release
cmake --build build/presolve
build/presolve/glqp-presolve original.gqp > reduced.gqp
build/presolve/glqp-presolve --local-fallback --separable-quadratics original.gqp > reduced.gqp
```

The CLI accepts a `GQP1` text stream. Its first line is `GQP1`, then
`n p m objective_constant`. Next come the `q`, `b`, and `h` vectors, in that
order. Then each of `Q`, `A`, and `G` is represented by a nonzero count
followed by `column row value` triples with zero-based indices. The CLI
sorts entries; duplicates and stored zeros are rejected by the library.
The output uses the same format with the new objective constant. Diagnostic
text goes to stderr. Exit codes 2, 3, 4, and 5 mean infeasible, inconclusive,
invalid input, and internal error respectively. Use the library API when
dual reconstruction is required.

The CLI also exposes `--no-singleton-equalities`, `--no-fixed-bounds`,
`--no-redundant-bounds`, and `--infeasibility-guard NUMBER`. An omitted filename
or `-` reads standard input. `--help` lists all switches. The default CLI
options match `PresolveOptions()`.

## Experiments

[`experiments/presolve/measure.py`](experiments/presolve/measure.py) measures
presolve alone over the bundled QPs and checks transformed objective and
retained-row identities at random points. It records all samples, input
hashes, source/build identity, affinity, dimensions, and reduction counts.
Its small repeated timings are a diagnostic of native overhead, not evidence
of end-to-end solver improvement. Compare complete solves on the normal
`parallel-8` execution policy and validate original-coordinate solutions.

Presolving remains optional until coverage and complete solve timings justify
promoting it. Changed systems may alter convergence as well as factorization
cost; a smaller matrix alone is not a performance win.

The first structural scan reduced 40 of 138 bundled QPs and fell back on
three cases with arithmetic ambiguity or tiny contradictions. It removed
4,368 variables, 4,198 equalities, and 7,274 inequalities. In QSHELL, total
nonzeros fell from 74,873 to 8,726; in STCQP2, variables fell from 4,097 to
2,045 and all 2,052 equality rows disappeared. These are structural results,
pending complete solve comparisons.

The second implementation iteration keeps every reduced matrix, right-hand
side, map, objective offset, and recovered solution **bitwise identical** on
all 138 problems. It changes data movement and bookkeeping:

- An ordered CSC cursor checks Hessian symmetry in linear time.
- Exact XOR bookkeeping identifies the last column and coefficient in a row.
- Only eliminated pivot rows acquire a CSR view for reverse dual recovery.
- Unchanged problems avoid constructing a second set of sparse matrices or
  scanning the objective, and release native identity-recovery state early.

[`iteration-2-interleaved.json`](experiments/presolve/iteration-2-interleaved.json)
records five alternating old/new observations per problem, on CPU 8 with one
numerical thread. Against first implementation commit `094e27e`, the geometric
mean native-time ratio was 0.737 and total presolve-time ratio was 0.635
(about 26% and 36% lower). This is a short single-core overhead diagnostic,
not a measurement of solve speedup or the standard parallel-8 campaign.
The full old/new native-source and binary hashes are recorded in the file.

The focused tests include analytic zero-variable optima, randomized convex
QP solutions, mixed equality/bound cascades with original KKT checks, scaled
and redundant bounds, nonfinite data, underflow, and uncertain contradictions.

The third iteration skips all stationarity matrix-vector products during
recovery when no variable was fixed: scattering the retained primal/dual
coordinates and zeroing removed multipliers is already sufficient. The
[alternating recovery diagnostic](experiments/presolve/iteration-3-recovery.json)
compares against `27b20a9` over 101 observations per case. Recovery was about
6% cheaper on QPCBOEI2 and 11% cheaper on QISRAEL; QSHELL, which requires
fixed-variable recovery, was unchanged. A synthetic 100,000-variable
tridiagonal QP with one redundant bound showed a 67% reduction in recovery
time. All recovered arrays were bitwise identical. These are microsecond-scale
component timings and do not establish complete-solve gains.

The standalone C ABI also has an AddressSanitizer/UndefinedBehaviorSanitizer
check, including leak detection:

```sh
taskset -c 8 python tools/check_presolve_native.py \
  --output docs/experiments/presolve/sanitizer-validation.json
```

The [recorded run](experiments/presolve/sanitizer-validation.json) passed
38,911 assertions, including 100 seeded mixed equality/bound cascades,
original-coordinate KKT and objective checks, zero-variable optima, empty
dimensions and rows, null buffers for zero-length vectors, reconstruction
after input buffers are destroyed, malformed CSC indices and values, and
overflow/inconclusive paths. GCC 11.5 compiled the instrumented C++ directly;
no Python wrapper or numerical library participates in the tested path.
ASan, UBSan, and LeakSanitizer reported no findings. The file records compiler
flags, source and binary hashes, CPU affinity, and complete output.

The first sandboxed attempt reached all successful assertions, but its
LeakSanitizer exit check failed because the sandbox uses `ptrace`. That
[attempt is retained](experiments/presolve/sanitizer-sandbox-attempt.json).
The successful run used the same checks outside that wrapper, keeping leak
detection enabled.

The fourth iteration starts with a [survey of all 138 bundled QPs](experiments/presolve/iteration-4-opportunities.json).
Exact general row duplicates or rows implied by an identical equality
would remove only 32 additional rows across nine problems. That limited
opportunity did not justify adding a global row-hashing pass. The larger
opportunity was keeping certified work when a small part of a QP is
ambiguous.

With local fallback enabled, the [structural diagnostic](experiments/presolve/iteration-4-local-final.json)
reduces 43 problems and has no whole-problem fallbacks on this suite. It
removes 4,868 variables, 4,518 equalities, and 8,260 inequalities. Relative
to the existing default, only the three previous fallback cases change:

| Problem | Additional fixed variables | Removed equalities | Removed inequalities | Deferred variables | Retained ambiguous rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| QPCBOEI1 | 1 | 1 | 15 | 1 | 2 |
| QPILOTNO | 232 | 52 | 437 | 1 | 1 |
| STCQP1 | 267 | 267 | 534 | 85 | 85 |

All 138 transformations pass original-objective and retained-row identity
checks at three random points per problem. These structural counts establish
additional usable reductions, not faster or more reliable complete solves.
The [initial prototype diagnostic](experiments/presolve/iteration-4-local.json)
is retained separately from the final source-hashed diagnostic.

The [disabled-option comparison](experiments/presolve/iteration-4-default-compatibility.json)
against `5b0002e` checks bitwise identity of every reduced matrix, vector,
map, objective offset, and recovered array on all 138 QPs. Three alternating
observations per problem give native-time and wrapper-total ratios near 1;
default preprocessing overhead is unchanged at this measurement precision.
The focused Python suite has 54 passing checks, including 12 randomized
row-order and dyadic-scaling cases with ambiguous chains and original KKT
validation. The [expanded sanitizer run](experiments/presolve/iteration-4-sanitizer.json)
passes 38,994 assertions with ASan, UBSan, and leak detection enabled.

The broader complete-solve campaign also found a material regression:
QFFFFF80 takes 68 iterations without presolve and 1,751 with it. Three
alternating frozen CPU-8 repetitions reproduce the slowdown. The
[pass-ablation diagnosis](experiments/presolve/qfffff80-regression.md) isolates
the singleton-equality path; `PresolveOptions(singleton_equalities=False)`
restores the unchanged model and 68 iterations. Keeping redundant bounds
alone does not help. A separate structural ablation restores all 37 constant
inequalities while keeping the reduced variable and equality dimensions; it still
takes 1,785 iterations, so keeping those barrier terms is not a remedy.
All recovered solutions pass original-data validation.
This evidence reinforces keeping presolve optional and its passes explicit;
there is no input-specific heuristic or default promotion.

The fifth iteration adds the separately enabled scalar-quadratic pass. The
[all-problem structural validation](experiments/presolve/iteration-5-structural.json)
uses `local_fallback=True, separable_quadratics=True` and validates objective
and retained-row identities at three random reduced points per input. It
accepts 6,546 additional scalar minima across nine of the 138 QPs, with no
ambiguous scalar decisions or whole-problem fallbacks in this collection.
Totals become 11,414 eliminated variables, 4,518 removed equalities, and
17,955 removed inequalities; 49 problems have a reduction.

| Problem | Scalar variables removed | Variables before/after scalar pass | Inequalities before/after scalar pass |
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

The [disabled-option comparison](experiments/presolve/iteration-5-default-compatibility.json)
against `d9bf350` preserves every reduced matrix, vector, map, objective
offset, and recovered primal/dual array bitwise on all 138 inputs, with three
checks per input. Both diagnostics omit timing observations because they ran
outside an exclusive measurement window. Structural reductions do not by
themselves establish faster solves, and this pass does not alter QFFFFF80.
The [native sanitizer validation](experiments/presolve/iteration-5-sanitizer.json)
passes 39,097 assertions, including free and clipped scalar minima, original
scaled-bound multipliers, reverse equality-pivot propagation, and uncertain
activity retention. Python tests add 12 randomized original-KKT cases and
exercise every standalone CLI option.

The sixth iteration stores only pivots that need a dual multiplier on the
recovery tape. Interior scalar minima already have their primal values in
the fixed-value array. Omitting their row-free entries avoids constructing
recovery CSR and computing stationarity matrix-vector products when every
elimination is an interior minimum, as on PRIMAL1 through PRIMAL4. Mixed
equality, fixed-bound, and active scalar-bound pivots keep their existing
reverse ordering.

The [enabled-option compatibility check](experiments/presolve/iteration-6-compatibility.json)
compares against `a6f9e36` with both optional passes enabled. Every reduced
output and recovered primal/dual array remains bitwise identical on all 138
inputs over three checks. The [final-source sanitizer run](experiments/presolve/iteration-6-sanitizer.json)
also passes 39,097 assertions; the complete Python suite passes 261 tests.
These correctness checks omit timing observations. A subsequent
[alternating recovery diagnostic](experiments/presolve/iteration-6-recovery.json)
ran alone on the machine, pinned to CPU 10, with 101 observations per version
and problem. PRIMAL1 recovery fell from 37.1 to 31.8 microseconds (14% less)
and PRIMAL4 from 51.9 to 34.5 microseconds (34% less). A synthetic 100,000-variable
independent diagonal QP fell from 616 to 80.1 microseconds (87% less).
MOSARQP1, STCQP2, QSHELL, and an unchanged tridiagonal control stayed within
4% of their reference recovery times. Every recovered array remained bitwise
identical. These component timings measure the storage optimization against
`a6f9e36`; complete-solve comparisons are needed to assess the scalar pass.
