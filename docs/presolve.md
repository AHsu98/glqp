# Native QP presolve

`glqp.presolve` is an opt-in structural presolver for
`min 0.5*x.T@Q@x + q.T@x + objective_constant`, with `A@x=b` and `G@x<=h`.
The C++17 library has no numerical-library dependency. Python only packs
canonical CSC buffers and copies outputs; sparse entry loops and primal/dual
reconstruction run in C++.

The current passes eliminate fixed variables from singleton equalities or
exactly matching opposite bounds, propagate those substitutions, remove
certified empty rows, and retain strongest singleton inequalities. Each pass
can be switched independently. No approximate equality or small coefficient
is used as an elimination rule. General aggregation, scaling, Hessian-based
fixing, and GLM predictor substitution are outside this version's scope.

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
inequalities receive zero duals, except an exactly fixed bound pair, whose
appropriate nonnegative multiplier is reconstructed. Equality pivots are
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

To build without Python:

```sh
cmake -S glqp/_native -B build/presolve -DCMAKE_BUILD_TYPE=Release
cmake --build build/presolve
build/presolve/glqp-presolve original.gqp > reduced.gqp
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
