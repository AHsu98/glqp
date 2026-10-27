# Sparse linear backends

QDLDL remains the default. Choose another factorization independently of GLQP's
float64 residual verification and bounded right-GMRES refinement:

```python
from glqp import SolverSettings

settings = SolverSettings(linear_solver_backend="pardiso")
settings = SolverSettings(linear_solver_backend="feral",
                          linear_solver_options={"ordering": "auto"})
x, result = problem.solve(settings=settings)
```

The backend receives the shifted factorization matrix `G + tau*R`. Projection,
refinement, stopping criteria, and all reported linear residuals still target
the requested `G`, which includes the outer proximal regularization. No backend
fallback, extra refinement, or automatic backend selection is hidden in this
interface. Different factorizations can still change rounding and QP iteration
trajectories; compare independently validated solutions, not just factor speed.

## Implementations

| Name | Factorization | Reused on a fixed pattern | Numerical threading |
| --- | --- | --- | --- |
| `qdldl` | Symmetric quasi-definite LDLᵀ | Upper-triangle value map, symbolic ordering, factor storage | Serial |
| `superlu` | General pivoted sparse LU | SciPy exposes no separate symbolic reuse | Serial factorization |
| `feral` | Rust multifrontal symmetric indefinite LDLᵀ | Native matrix, triangular value map, symbolic analysis | Rayon when explicitly budgeted |
| `pardiso` | oneMKL symmetric indefinite LDLᵀ | Triangular value map, index buffers, symbolic analysis | oneMKL/OpenMP budget |

FERAL requires the optional `feral-solver==0.19.0` distribution. The unrelated
PyPI package named `feral` is not this solver. PARDISO uses the already required
`sparse-dot-mkl==0.9.8` ABI bindings and an available oneMKL runtime; it introduces
no second MKL instance or new integer ABI. SuperLU comes with SciPy.

PARDISO caches typed pointers to its validated matrix buffers and scalar
arguments. Its instance-local binding calls the same loaded MKL symbol; the
shared `sparse-dot-mkl` binding keeps its ndarray validation. Numeric updates
refill owned storage, and each inverse application returns a new vector.
Regression tests compare results bit for bit with the original checked binding
after updates, including strided read-only right-hand sides and garbage
collection. A factorization instance, like the native context, is not reentrant.

`SparseFactorization` in `glqp/backends.py` defines `matches(matrix)`,
`update(matrix)`, `solve(rhs)`, `close()`, and `diagnostics()`. Construction
performs the initial analysis and numeric factorization. Matrices are canonical,
full symmetric float64 CSC arrays; updates require the same storage pattern.
The factor cache identity includes the backend, canonical options, and complete
sparse pattern. A changed identity closes the old factor and rebuilds it.
`solve` preserves its RHS and returns a new vector. Native resources are closed
on replacement and numerical failure; PARDISO also has destructor cleanup.

QDLDL's adapter uses its explicit `upper=True` input interface and a cached
upper-CSC value map. Numeric updates refill one existing buffer, avoiding a
SciPy triangular extraction and sparse object construction per Newton step.
Regression tests compare permutation, factor arrays, and inverse applications
bit for bit against QDLDL's original full-matrix interface across updates.

Only numerical factorization failures are eligible for the existing shift retry.
Memory, malformed input, unknown options, and unexpected backend errors propagate.
FERAL can report `SUCCESS` for a partial factor with zero pivots, so the adapter
also checks its zero-inertia count. No numerical inertia threshold is imposed.
QDLDL now uses a private checked binding of the same 0.1.9.post1 kernels.
Its numeric update checks the native return code and invalidates a failed
factor. Subsequent solves reject that invalid state. The existing retry then
discards it and increases only the factorization shift. Finite inaccurate
GMRES results retain their previous behavior. See the [checked build contract](checked-qdldl.md).

## Explicit options

`linear_solver_options` is a dictionary or `None`. Unknown keys and invalid
values fail before factoring, even with a zero right-hand side.

| Backend | Options and defaults |
| --- | --- |
| QDLDL | No additional options |
| SuperLU | `ordering="COLAMD"`; also `MMD_AT_PLUS_A`, `MMD_ATA`, `NATURAL` |
| FERAL | `ordering="amd"`, `scaling="auto"`; ordering also `amf`, `metis`, `scotch`, `kahip`, `auto`, `auto_race`; scaling also `none`, `infnorm`, `mc64` |
| PARDISO | `ordering="metis"` or `"amd"`, `scaling=True`, `matching=True`, `pivot_exponent=8` (integer 1–20) |

PARDISO's symmetric scaling requires matching. Its pivot exponent controls a
factorization perturbation, not GLQP's requested KKT system. FERAL calls
`Solver.solve`, never `solve_refined`. SuperLU refinement is disabled. PARDISO
sets `iparm[7]=0` **and** `iparm[20]=3`: the latter retains 1×1/2×2 Bunch–Kaufman
pivoting while disabling the otherwise mandatory correction on perturbed
pivots. Every PARDISO solve checks that its reported internal refinement count
is zero. See [Intel's parameter reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/pardiso-iparm-parameter.html).

The benchmark runner configures `BENCH_THREADS`, `RAYON_NUM_THREADS`, MKL/OpenMP
limits, and process affinity before import. FERAL uses serial factorization for
a one-thread budget; a parallel budget requires the matching Rayon environment
setting before its first use. Thread counts are process-level experiment setup;
do not change a previously initialized Rayon pool's environment mid-process.
PARDISO follows the MKL thread limit. More cores do not make QDLDL parallel.

## What was learned from POUNCE

Research pinned [POUNCE c70e06dc](https://github.com/jkitchin/pounce/tree/c70e06dc82504118836046188d5f4e8f019e4a53)
and its [FERAL dependency version 0.19.0](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/Cargo.toml#L277), whose release source is
[FERAL 6fbea336](https://github.com/jkitchin/feral/tree/6fbea336ee3a7428d908355d482fe9637d59f884).
The implemented choices behind POUNCE's `linear_solver` option are FERAL and
optional HSL MA57; its inherited option registry also lists names that the
[implementation rejects](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-algorithm/src/application.rs#L2690).
The experiments here measure that FERAL engine inside GLQP's numerical contract.
They establish factorization and complete-GLQP behavior, with the scope below.

| POUNCE design and pinned implementation | What this GLQP study exercises |
| --- | --- |
| [Sparse solver interface](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-linsol/src/sparse_sym_iface.rs) separates structure initialization, numeric values, factor/solve, and factor-quality escalation. [TSymLinearSolver](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-linsol/src/t_sym_solver.rs) marshals formats, applies optional symmetric scaling, handles retry statuses, and supports multiple RHS columns. | Pattern/option cache identity, numeric updates, and repeated single-vector solves. Multi-RHS batching and POUNCE's retry/quality policy are outside these measurements. |
| [FERAL adapter](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-feral/src/lib.rs#L1218) caches triplet-to-CSC scatter positions and numeric storage, preserves structural zeros, and reuses symbolic analysis. Its [backsolve](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-feral/src/lib.rs#L1441) uses Rust `solve_into` APIs and reusable output scratch. | Cached triangular value maps, fixed-pattern FERAL numeric refactors, and inverse applications through its Python wheel. Timings include this Python boundary and its returned-vector allocation. |
| [Actual FeralConfig defaults](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-feral/src/lib.rs#L691) select automatic ordering, ordering preprocessing, and scaling. | AMD and automatic ordering with automatic scaling were measured. Thread budgets were imposed explicitly by the benchmark profiles. |
| FERAL's wrapper defaults to internal refinement with a [10-step cap](https://github.com/jkitchin/feral/blob/6fbea336ee3a7428d908355d482fe9637d59f884/src/numeric/solve.rs#L1954). The [limited-memory IPM option path](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-algorithm/src/application.rs#L8195) disables it unless overridden; the [outer primal-dual solver](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-algorithm/src/kkt/pd_full_space_solver.rs) refines its unreduced KKT system and can request better factors. | Plain FERAL `solve` followed by GLQP's existing bounded right-GMRES and fresh requested-system residual. No POUNCE refinement defaults or factor-quality escalation were imported. |
| [Factor acceptance](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-feral/src/lib.rs#L1305) distinguishes zero pivots, small pivots, and unreliable or incorrect inertia, which steer POUNCE's regularization. | Native status and exact zero-pivot rejection, followed by GLQP residual verification and its existing shift retry. POUNCE's pivot floors and expected-inertia rules are outside this comparison. |
| [Small-Schur solver](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-feral/src/schur.rs) and [arrowhead block solver](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-feral/src/block.rs) factor caller-specified blocks and a dense Schur complement; the latter runs independent blocks in parallel. | Monolithic sparse factorization only. Structured block methods remain a separate experiment requiring a partition and full residual/performance validation. |
| [MA57 adapter](https://github.com/jkitchin/pounce/blob/c70e06dc82504118836046188d5f4e8f019e4a53/crates/pounce-hsl/src/ma57.rs) supports symbolic reuse, pivot-quality changes, workspace-growth retries, and an explicit batching option. | MA57 was not installed or measured. PARDISO and SuperLU are independent GLQP comparison backends, not POUNCE implementations. |

The useful design lessons are explicit factor lifecycles, reusable storage,
observable options, and residual checks owned by the outer solver. POUNCE's
refinement and quality controls also show why a fast raw factorization alone
cannot settle the complete-solver choice. These results do not measure POUNCE
end to end or rank its untested structured and MA57 paths.

POUNCE is EPL-2.0; FERAL is MIT with separate third-party notices for incorporated
components. This integration uses the unmodified FERAL wheel as an optional
dependency and copies no POUNCE or FERAL implementation. MA57 needs separately
licensed HSL code and was not installed for these experiments.

## Measuring changes

Use the benchmark's versioned `parallel-8` execution policy for routine paired
QP campaigns. Apply a backend setting to the candidate explicitly and retain
the settings in its manifest. Use the standalone
`glqp-bench/benchmarks/linear-systems/backend-20261016/probe.py` for fixed matrices:
it captures selected Newton systems from a frozen QDLDL solver and separately
times constructor, repeated numeric update, inverse application, and the full
verified solve. The three repetitions are a targeted microbenchmark, not a new
full-suite repeat policy. Record failures and memory alongside speed.

The initial eight-system screen found a large PARDISO numeric-update improvement
on CVXQP2_L, while QDLDL remained faster on thin LISWET and UBH1 systems. FERAL
and pivoted alternatives factored the captured QSCFXM1 matrix that rejected a
fresh QDLDL construction. SuperLU exceeded a 90-second diagnostic budget on
BOYD2, motivating a shorter per-worker cap with continued testing of other
backends. These findings justify targeted QP confirmation; they do not establish
a universal replacement for QDLDL. The benchmark report retains exact source
hashes, options, raw repetitions, residual checks, and subsequent iterations.

A subsequent CPU-10 diagnostic isolated BOYD2's PARDISO timeout in phase 11
(analysis). Explicit `{"scaling": False, "matching": False}` with METIS completed
analysis in 0.391 seconds and numeric factorization in 0.052 seconds, then passed
the requested-system check with one existing GMRES correction. Matching enabled
with scaling disabled still exceeded the 8-second process budget; AMD without
matching also timed out during analysis. The captured matrix has median degree
3 but maximum degree 93,263. These observations identify a useful explicit
configuration for this structure, without establishing a universal ordering or
matching rule. Scaling and matching remain enabled by default.

The same diagnostic found repeated ctypes conversions mattered on QPCBOEI2.
Caching native arguments reduced numeric-update time from about 158 to 134
microseconds and inverse application from 59 to 40 microseconds, with unchanged
factor fill and residuals. AMD reduced its factor entries from 4,625 to 4,265,
but QDLDL still took only 46 and 9 microseconds for those two operations.
These efficiency-core microbenchmarks establish implementation effects, not
paired end-to-end QP performance on the standard performance-core panel. See
`glqp-bench/benchmarks/linear-systems/backend-20261016/adapter-round4.md` for all
observations, timeouts, exact options, and frozen-source hashes.
