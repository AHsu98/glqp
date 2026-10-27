# Checked QDLDL updates

GLQP's `qdldl` backend uses a private binding of QDLDL Python 0.1.9.post1.
The installed public `qdldl` package is left unchanged, so frozen earlier
solver revisions can still use their original dependency during comparisons.

The upstream constructor checks the return from `QDLDL_factor`, but its update
ignored that return. A failed update could leave partially refreshed factors
which still produced finite, incorrect inverse applications. The saved QSCFXM1
95-to-96 transition reproduces this failure. The private update checks that
same integer return, marks its factors invalid before updating, and rejects
subsequent solves if the update failed. Constructor failure also frees partially
allocated work arrays. The C QDLDL/AMD kernels and numeric operation order are
unchanged; retained licenses and original source hashes accompany the vendor
sources in `glqp/_native/checked_qdldl/upstream.json`.

The Python adapter maps the native numerical exception to its existing
`FactorizationError`. `factor_and_solve` discards that factor and retries with
`10*tau`; a zero initial shift and the existing attempt limit remain final.
The physical T30 recipe, tolerances, GMRES budget and nonlinear policies do not
change. Finite inaccurate refinement is still returned as such. Nonfinite
solution/correction handling stays in `glqp.linear`. Additional right-hand sides
in `VerifiedLinearSystem` retain their existing failure policy: invalidate and
raise without replacing a factor inside the fixed-factor context.

The hot adapter update gathers values into its owned upper CSC buffer and uses
the private `_update_values` entry point. Its pattern has already passed
`matches`; the native entry point checks vector shape/storage length and the
numeric status without an extra sparse-pattern scan or factor export. Direct
`Solver.update(matrix)` calls validate the complete supplied pattern. Factor
exports remain optional diagnostics. Malformed matrices/RHS shapes raise input
errors; compiler, import and cache failures occur outside the numerical retry.

## Preparing a build

The source builder currently supports 64-bit Linux with C/C++ compilers, CMake,
Python development headers, and pinned `pybind11==2.13.6` headers. Install the
build extra in a managed environment and prepare before measurement:

```sh
python -m pip install '.[native-build]'
python -c 'from glqp.checked_qdldl import prepare_checked_qdldl; print(prepare_checked_qdldl())'
```

Preparation is also lazy on the first QDLDL factor construction. No network or
package installation happens in preparation. There is no silent fallback to the
unchecked dependency. `CC`/`CXX` may name compiler executables. The build is
single-process and serialized with a file lock. The default cache is
`/tmp/glqp-checked-qdldl-<uid>`; `GLQP_CHECKED_QDLDL_CACHE` can relocate it.

The cache identity records vendor/patch/loader source hashes, compiler versions,
explicit floating-point flags, Python ABI, pinned pybind11/header hashes, and
CMake version. Kernels use float64 and 64-bit indices; there is no fast math or
CPU-specific tuning. The private module is `_glqp_qdldl_checked`; module-local
pybind registration and hidden symbols keep it separate from the public module.

`prepare_checked_qdldl()` returns `module_name`, `library`, `library_sha256`,
`manifest`, `manifest_sha256`, `build_key`, and the complete build `identity`.
The manifest itself contains no required absolute binary path. To freeze a
build, copy its library and manifest, then set both paths before the first
import/preparation in each cold process:

```sh
export GLQP_CHECKED_QDLDL_LIBRARY=/absolute/frozen/_glqp_qdldl_checked.so
export GLQP_CHECKED_QDLDL_MANIFEST=/absolute/frozen/build.json
```

The loader verifies the source hash, Python ABI, manifest identity and binary
checksum before importing. Frozen loading requires neither build headers nor a
compiler. The library and manifest must be supplied together. A cached module
is reused within one process; change builds only in a new process. Benchmark
preflight must prepare and freeze the candidate before timing, and record the
unchanged public dependency separately for the baseline.

The native unit suite checks healthy factor/inverse bytes against the public
package, failed constructor/update propagation, invalid-state rejection and
recovery, malformed inputs, exact retry limits and no factor export during
updates. Broader trajectory and coverage effects belong to the separately
frozen benchmark qualification; this implementation does not claim those
outcomes in advance.
