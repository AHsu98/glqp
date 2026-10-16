"""Sparse factorization adapters; residual verification lives in ``linear``.

Adapters factor the matrix they receive and expose an unrefined inverse action.
They never modify GLQP's requested system, tolerances, or outer regularization.
Numerical failures have a separate exception from malformed inputs and missing
optional libraries, so only genuine factorization failures can trigger a retry.
"""
import ctypes
import os

import numpy as np
import qdldl
from scipy.sparse import csc_array
from scipy.sparse.linalg import splu


class LinearSolveError(RuntimeError):
    """A numerical failure prevented a finite linear direction."""


class FactorizationError(LinearSolveError):
    """A recoverable backend failure; a larger numerical shift may help."""


class SparseFactorization:
    """Common lifecycle for a canonical, full symmetric float64 CSC matrix.

    ``matches`` tests the complete storage pattern. ``update`` requires that
    pattern and replaces numeric values; ``solve`` leaves its RHS untouched.
    ``close`` releases native resources and is idempotent. Diagnostics are
    deliberately outside the solve path because some libraries copy factors.
    """
    name = None

    def __init__(self, matrix):
        self.shape = matrix.shape
        self.indptr = matrix.indptr.copy()
        self.indices = matrix.indices.copy()

    def matches(self, matrix):
        return (self.shape == matrix.shape
                and np.array_equal(self.indptr, matrix.indptr)
                and np.array_equal(self.indices, matrix.indices))

    def close(self):
        pass

    def diagnostics(self):
        return {'backend': self.name}

    def _lower_pattern(self):
        # Lower CSC is also upper CSR for a symmetric matrix. Cache this gather
        # map once; later iterations neither sort nor rebuild sparse matrices.
        columns = np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))
        positions = np.flatnonzero(self.indices >= columns)
        return (positions, np.searchsorted(positions, self.indptr),
                self.indices[positions])


class QDLDLFactorization(SparseFactorization):
    name = 'qdldl'

    def __init__(self, matrix):
        super().__init__(matrix)
        try:
            self.factor = qdldl.Solver(matrix)
        except RuntimeError as exc:
            raise FactorizationError(str(exc)) from exc

    def update(self, matrix):
        try:
            self.factor.update(matrix)
        except RuntimeError as exc:
            raise FactorizationError(str(exc)) from exc

    def solve(self, rhs):
        try:
            return self.factor.solve(rhs)
        except RuntimeError as exc:
            raise FactorizationError(str(exc)) from exc

    def diagnostics(self):
        lower, diagonal, permutation = self.factor.factors()
        return dict(backend=self.name, factor_nnz=int(lower.nnz + len(diagonal)),
                    factor_array_bytes=int(lower.data.nbytes + lower.indices.nbytes
                        + lower.indptr.nbytes + diagonal.nbytes + permutation.nbytes))


class SuperLUFactorization(SparseFactorization):
    """Pivoted LU control. SciPy does not expose symbolic-only reuse."""
    name = 'superlu'

    def __init__(self, matrix):
        super().__init__(matrix)
        self.update(matrix)

    def update(self, matrix):
        try:
            self.factor = splu(csc_array(matrix), permc_spec='COLAMD',
                               options={'Equil': False, 'IterRefine': 'NOREFINE'})
        except RuntimeError as exc:
            raise FactorizationError(str(exc)) from exc

    def solve(self, rhs):
        try:
            return self.factor.solve(rhs)
        except RuntimeError as exc:
            raise FactorizationError(str(exc)) from exc

    def diagnostics(self):
        lower, upper = self.factor.L, self.factor.U
        size = sum(a.nbytes for factor in (lower, upper)
                   for a in (factor.data, factor.indices, factor.indptr))
        return dict(backend=self.name, factor_nnz=int(lower.nnz + upper.nnz),
                    factor_array_bytes=int(size), symbolic_reuse=False)


class FeralFactorization(SparseFactorization):
    """POUNCE's Rust LDLᵀ engine through the optional feral-solver package."""
    name = 'feral'

    def __init__(self, matrix):
        super().__init__(matrix)
        try:
            import feral
        except ImportError as exc:
            raise ImportError("Backend 'feral' requires feral-solver==0.19.0") from exc
        self._feral = feral
        threads = int(os.environ.get('BENCH_THREADS', '1'))
        if threads > 1 and os.environ.get('RAYON_NUM_THREADS') != str(threads):
            raise ValueError('Set RAYON_NUM_THREADS=BENCH_THREADS before using parallel FERAL')
        self.factor = feral.Solver(parallel=threads > 1, ordering='amd',
                                  partial_singular_warning=False)
        self._positions, indptr, indices = self._lower_pattern()
        self.matrix = feral.CscMatrix(matrix.shape[0], indptr.astype(np.int64),
            indices.astype(np.int64), np.asarray(matrix.data[self._positions], dtype=float))
        self._factor()

    def _factor(self):
        try:
            status, inertia = self.factor.factor(self.matrix)
        except (self._feral.SingularError, self._feral.NumericFailure,
                self._feral.DelayBudgetExceeded) as exc:
            raise FactorizationError(str(exc)) from exc
        # FERAL may report SUCCESS for a partial factor containing exact zero
        # pivots; inertia still reports them. Do not use that state as inverse.
        if status != self._feral.FactorStatus.SUCCESS or (inertia and inertia.n_zero):
            raise FactorizationError(f'FERAL factorization status {status}; inertia {inertia}')

    def update(self, matrix):
        self.matrix.set_values(np.asarray(matrix.data[self._positions], dtype=float))
        self._factor()

    def solve(self, rhs):
        # Deliberately use solve, never solve_refined: GLQP owns refinement.
        try:
            return self.factor.solve(np.asarray(rhs, dtype=float))
        except (self._feral.SingularError, self._feral.NumericFailure) as exc:
            raise FactorizationError(str(exc)) from exc

    def diagnostics(self):
        return dict(backend=self.name, version=self._feral.__version__,
                    factor_nnz=self.factor.factor_nnz,
                    symbolic_calls=self.factor.symbolic_call_count,
                    ordering=self.factor.ordering, scaling=self.factor.scaling,
                    parallel=self.factor.parallel, inner_refinement=False)


class PardisoFactorization(SparseFactorization):
    """oneMKL symmetric-indefinite LDLᵀ with symbolic reuse and no inner IR.

    sparse-dot-mkl's pinned ABI bindings select MKL_INT consistently with its
    sparse BLAS. Calling that binding directly avoids its convenience wrapper's
    repeated index casts and permutation allocation on every back-substitution.
    """
    name = 'pardiso'

    def __init__(self, matrix):
        super().__init__(matrix)
        from sparse_dot_mkl._mkl_interface._cfunctions import MKL
        from sparse_dot_mkl.solvers import pardisoinit
        self._mkl = MKL
        self._closed = True
        self._positions, indptr, indices = self._lower_pattern()
        if max(matrix.shape[0], len(indices)) > np.iinfo(MKL.MKL_INT_NUMPY).max:
            raise OverflowError('PARDISO matrix exceeds the active MKL integer interface')
        self._indptr = np.ascontiguousarray(indptr, dtype=MKL.MKL_INT_NUMPY)
        self._indices = np.ascontiguousarray(indices, dtype=MKL.MKL_INT_NUMPY)
        self._values = np.ascontiguousarray(matrix.data[self._positions], dtype=float)
        self._perm = np.zeros(matrix.shape[0], dtype=MKL.MKL_INT_NUMPY)
        self._dummy = np.zeros(matrix.shape[0])
        self._pt, self.iparm = pardisoinit(-2, single_precision=False, zero_indexing=True)
        self.iparm[7] = 0
        # 3 retains Bunch-Kaufman pivoting but disables mandatory refinement
        # on perturbed pivots. Merely setting iparm[7]=0 is insufficient.
        self.iparm[20] = 3
        self.iparm[10] = 1
        self.iparm[12] = 1
        self._closed = False
        try:
            self._call(11)
            self._call(22)
        except BaseException:
            self.close()
            raise

    def _call(self, phase, rhs=None):
        if self._closed:
            raise RuntimeError('PARDISO factorization has been closed')
        mkl = self._mkl
        integer = mkl.MKL_INT
        arg = lambda value: ctypes.byref(integer(value))
        rhs = self._dummy if rhs is None else np.ascontiguousarray(rhs, dtype=float)
        solution = self._dummy if phase != 33 else np.empty_like(rhs)
        error = integer(0)
        mkl._pardiso(self._pt, arg(1), arg(1), arg(-2), arg(phase), arg(self.shape[0]),
            self._values, self._indptr, self._indices, self._perm, arg(1), self.iparm,
            arg(0), rhs, solution, ctypes.byref(error))
        if error.value in (-4, -7):
            raise FactorizationError(f'PARDISO phase {phase}: numerical error {error.value}')
        if error.value in (-2, -9):
            raise MemoryError(f'PARDISO phase {phase}: insufficient memory ({error.value})')
        if error.value:
            raise RuntimeError(f'PARDISO phase {phase}: error {error.value}')
        if phase == 33 and self.iparm[6] != 0:
            raise RuntimeError('PARDISO unexpectedly performed internal iterative refinement')
        return solution

    def update(self, matrix):
        self._values[:] = matrix.data[self._positions]
        self._call(22)

    def solve(self, rhs):
        return self._call(33, rhs)

    def close(self):
        if not getattr(self, '_closed', True):
            try:
                self._call(-1)
            finally:
                self._closed = True

    def __del__(self):
        # Cleanup cannot raise during interpreter shutdown or mask a failed
        # constructor. Normal explicit close() still reports backend errors.
        try:
            self.close()
        except Exception:
            pass

    def diagnostics(self):
        return dict(backend=self.name, factor_nnz=int(self.iparm[17]),
                    factor_memory_kib=int(self.iparm[16]),
                    analysis_peak_kib=int(self.iparm[14]),
                    perturbed_pivots=int(self.iparm[13]), inner_refinements=int(self.iparm[6]),
                    positive_eigenvalues=int(self.iparm[21]),
                    negative_eigenvalues=int(self.iparm[22]),
                    ordering='metis', scaling=True, matching=True)


BACKENDS = {backend.name: backend for backend in (
    QDLDLFactorization, SuperLUFactorization, FeralFactorization, PardisoFactorization)}


def backend_type(name):
    """Resolve an explicit backend name; never silently fall back."""
    try:
        return BACKENDS[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Unknown linear backend {name!r}; choose {tuple(BACKENDS)}') from exc
