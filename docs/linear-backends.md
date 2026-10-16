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
| `qdldl` | Symmetric quasi-definite LDLᵀ | Symbolic ordering and factor storage | Serial |
| `superlu` | General pivoted sparse LU | SciPy exposes no separate symbolic reuse | Serial factorization |
| `feral` | Rust multifrontal symmetric indefinite LDLᵀ | Native matrix, triangular value map, symbolic analysis | Rayon when explicitly budgeted |
| `pardiso` | oneMKL symmetric indefinite LDLᵀ | Triangular value map, index buffers, symbolic analysis | oneMKL/OpenMP budget |

FERAL requires the optional `feral-solver==0.19.0` distribution. The unrelated
PyPI package named `feral` is not this solver. PARDISO uses the already required
`sparse-dot-mkl==0.9.8` ABI bindings and an available oneMKL runtime; it introduces
no second MKL instance or new integer ABI. SuperLU comes with SciPy.

`SparseFactorization` in `glqp/backends.py` defines `matches(matrix)`,
`update(matrix)`, `solve(rhs)`, `close()`, and `diagnostics()`. Construction
performs the initial analysis and numeric factorization. Matrices are canonical,
full symmetric float64 CSC arrays; updates require the same storage pattern.
The factor cache identity includes the backend, canonical options, and complete
sparse pattern. A changed identity closes the old factor and rebuilds it.
`solve` preserves its RHS and returns a new vector. Native resources are closed
on replacement and numerical failure; PARDISO also has destructor cleanup.

Only numerical factorization failures are eligible for the existing shift retry.
Memory, malformed input, unknown options, and unexpected backend errors propagate.
FERAL can report `SUCCESS` for a partial factor with zero pivots, so the adapter
also checks its zero-inertia count. No numerical inertia threshold is imposed.
QDLDL's known unchecked numeric-update return code remains a reason to inspect
the freshly verified residual; this integration does not rewrite its native ABI.

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
and its FERAL dependency version 0.19.0, whose release source is
[FERAL 6fbea336](https://github.com/jkitchin/feral/tree/6fbea336ee3a7428d908355d482fe9637d59f884).
POUNCE separates its symmetric solver interface from concrete FERAL and optional
HSL MA57 implementations. Its FERAL adapter caches triplet-to-CSC placement and
numeric buffers across iterates, reuses symbolic analysis, and exposes separate
ordering, scaling, pivot-quality, and refinement controls. Its comments explain
the cost of nested backend and outer refinement. These design lessons motivated
the small lifecycle above; GLQP does not import POUNCE's algorithm, inertia-based
regularization policy, or solver code.

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
