"""Sparse factorization adapters; residual verification lives in ``linear``.

Adapters factor the matrix they receive and expose an unrefined inverse action.
They never modify GLQP's requested system, tolerances, or outer regularization.
Numerical failures have a separate exception from malformed inputs and missing
optional libraries, so only genuine factorization failures can trigger a retry.
"""
import ctypes
import os

import numpy as np
from . import checked_qdldl as qdldl
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

    def __init__(self, matrix, options=None):
        if (matrix.format != 'csc' or matrix.shape[0] != matrix.shape[1]
                or not matrix.shape[0] or matrix.dtype != np.dtype('float64')):
            raise ValueError('A nonempty square float64 CSC matrix is required')
        matrix.check_format(full_check=True)
        if not matrix.has_canonical_format or not np.isfinite(matrix.data).all():
            raise ValueError('Factorization matrix must be canonical and finite')
        self.shape = matrix.shape
        self.indptr = matrix.indptr.copy()
        self.indices = matrix.indices.copy()
        self.options = backend_options(self.name, options)

    def matches(self, matrix):
        return (self.shape == matrix.shape
                and np.array_equal(self.indptr, matrix.indptr)
                and np.array_equal(self.indices, matrix.indices))

    def close(self):
        pass

    def diagnostics(self):
        return {'backend': self.name, 'options': self.options.copy()}

    def _rhs(self, rhs):
        rhs = np.asarray(rhs, dtype=float)
        if rhs.ndim != 1 or rhs.size != self.shape[0]:
            raise ValueError('Backend RHS must be a vector matching the matrix dimension')
        return rhs

    def _triangle_pattern(self, *, upper=False):
        # QDLDL uses upper CSC; FERAL uses lower CSC, which is also upper CSR
        # for PARDISO. Later iterations only gather values into this pattern.
        columns = np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))
        positions = np.flatnonzero(self.indices <= columns if upper else self.indices >= columns)
        return (positions, np.searchsorted(positions, self.indptr),
                self.indices[positions])


class QDLDLFactorization(SparseFactorization):
    name = 'qdldl'

    def __init__(self, matrix, options=None):
        super().__init__(matrix, options)
        self._positions, indptr, indices = self._triangle_pattern(upper=True)
        self._upper = csc_array((matrix.data[self._positions], indices, indptr), shape=matrix.shape)
        # Build/import failures must not be mistaken for numerical failures.
        qdldl.prepare_checked_qdldl()
        try:
            self.factor = qdldl.Solver(self._upper, upper=True)
        except RuntimeError as exc:
            raise FactorizationError(str(exc)) from exc

    def update(self, matrix):
        np.take(matrix.data, self._positions, out=self._upper.data)
        try:
            self.factor._update_values(self._upper.data)
        except RuntimeError as exc:
            raise FactorizationError(str(exc)) from exc

    def solve(self, rhs):
        try:
            return self.factor.solve(self._rhs(rhs))
        except RuntimeError as exc:
            raise FactorizationError(str(exc)) from exc

    def diagnostics(self):
        lower, diagonal, permutation = self.factor.factors()
        return dict(backend=self.name, options=self.options.copy(), factor_nnz=int(lower.nnz + len(diagonal)),
                    factor_array_bytes=int(lower.data.nbytes + lower.indices.nbytes
                        + lower.indptr.nbytes + diagonal.nbytes + permutation.nbytes))


class SuperLUFactorization(SparseFactorization):
    """Pivoted LU control. SciPy does not expose symbolic-only reuse."""
    name = 'superlu'

    def __init__(self, matrix, options=None):
        super().__init__(matrix, options)
        self.update(matrix)

    def update(self, matrix):
        try:
            self.factor = splu(csc_array(matrix), permc_spec=self.options['ordering'],
                               options={'Equil': False, 'IterRefine': 'NOREFINE'})
        except RuntimeError as exc:
            raise FactorizationError(str(exc)) from exc

    def solve(self, rhs):
        try:
            return self.factor.solve(self._rhs(rhs))
        except RuntimeError as exc:
            raise FactorizationError(str(exc)) from exc

    def diagnostics(self):
        lower, upper = self.factor.L, self.factor.U
        size = sum(a.nbytes for factor in (lower, upper)
                   for a in (factor.data, factor.indices, factor.indptr))
        return dict(backend=self.name, options=self.options.copy(), factor_nnz=int(lower.nnz + upper.nnz),
                    factor_array_bytes=int(size), symbolic_reuse=False)


class FeralFactorization(SparseFactorization):
    """POUNCE's Rust LDLᵀ engine through the optional feral-solver package."""
    name = 'feral'

    def __init__(self, matrix, options=None):
        super().__init__(matrix, options)
        try:
            import feral
        except ImportError as exc:
            raise ImportError("Backend 'feral' requires feral-solver==0.19.0") from exc
        self._feral = feral
        threads = int(os.environ.get('BENCH_THREADS', '1'))
        if threads > 1 and os.environ.get('RAYON_NUM_THREADS') != str(threads):
            raise ValueError('Set RAYON_NUM_THREADS=BENCH_THREADS before using parallel FERAL')
        self.factor = feral.Solver(parallel=threads > 1, ordering=self.options['ordering'],
                                  scaling=self.options['scaling'],
                                  partial_singular_warning=False)
        self._positions, indptr, indices = self._triangle_pattern()
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
            return self.factor.solve(self._rhs(rhs))
        except (self._feral.SingularError, self._feral.NumericFailure) as exc:
            raise FactorizationError(str(exc)) from exc

    def diagnostics(self):
        return dict(backend=self.name, options=self.options.copy(), version=self._feral.__version__,
                    factor_nnz=self.factor.factor_nnz,
                    symbolic_calls=self.factor.symbolic_call_count,
                    ordering=self.factor.ordering, scaling=self.factor.scaling,
                    parallel=self.factor.parallel, inner_refinement=False)


class PardisoFactorization(SparseFactorization):
    """oneMKL symmetric-indefinite LDLᵀ with symbolic reuse and no inner IR.

    sparse-dot-mkl's pinned ABI binding selects MKL_INT consistently with its
    sparse BLAS. An instance-local pointer binding to that same function keeps
    validated, owned matrix buffers and scalar arguments across calls. The
    shared sparse-dot-mkl binding and its array checks are never modified.
    """
    name = 'pardiso'

    def __init__(self, matrix, options=None):
        super().__init__(matrix, options)
        from sparse_dot_mkl._mkl_interface._cfunctions import MKL
        from sparse_dot_mkl.solvers import pardisoinit
        self._mkl = MKL
        self._closed = True
        self._positions, indptr, indices = self._triangle_pattern()
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
        self.iparm[1] = {'amd': 0, 'metis': 2}[self.options['ordering']]
        self.iparm[9] = self.options['pivot_exponent']
        self.iparm[10] = int(self.options['scaling'])
        self.iparm[12] = int(self.options['matching'])
        self._bind_native_call()
        self._closed = False
        try:
            self._call(11)
            self._call(22)
        except BaseException:
            self.close()
            raise

    def _bind_native_call(self):
        """Bind already validated storage once, using the active MKL ABI.

        Every cached pointer owns its NumPy array or ctypes scalar. Matrix
        buffers are refilled, never resized. RHS/output pointers are temporary
        and their arrays stay alive throughout the synchronous native call.
        As with the native factorization, an instance is not reentrant.
        """
        integer = self._mkl.MKL_INT
        integer_pointer = ctypes.POINTER(integer)
        self._double_pointer = ctypes.POINTER(ctypes.c_double)
        if self._pt.itemsize != ctypes.sizeof(ctypes.c_void_p):
            raise RuntimeError('PARDISO handle storage differs from the native pointer width')
        # Cast the loaded symbol; do not load another MKL library or change
        # the ndarray argument types on the shared sparse-dot-mkl function.
        self._native_symbol = self._mkl._pardiso
        prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p,
            integer_pointer, integer_pointer, integer_pointer, integer_pointer,
            integer_pointer, self._double_pointer, integer_pointer, integer_pointer,
            integer_pointer, integer_pointer, integer_pointer, integer_pointer,
            self._double_pointer, self._double_pointer, integer_pointer)
        self._native_call = prototype(ctypes.cast(self._native_symbol, ctypes.c_void_p).value)
        arg = lambda value: ctypes.byref(integer(value))
        self._phase_arguments = {phase: arg(phase) for phase in (-1, 11, 22, 33)}
        self._error = integer(0)
        self._native_prefix = (self._pt.ctypes.data_as(ctypes.c_void_p), arg(1), arg(1), arg(-2))
        self._native_matrix = (arg(self.shape[0]),
            self._values.ctypes.data_as(self._double_pointer),
            self._indptr.ctypes.data_as(integer_pointer),
            self._indices.ctypes.data_as(integer_pointer),
            self._perm.ctypes.data_as(integer_pointer), arg(1),
            self.iparm.ctypes.data_as(integer_pointer), arg(0))
        self._dummy_pointer = self._dummy.ctypes.data_as(self._double_pointer)
        self._error_pointer = ctypes.byref(self._error)

    def _call(self, phase, rhs=None):
        if self._closed:
            raise RuntimeError('PARDISO factorization has been closed')
        if phase == 33:
            rhs = np.ascontiguousarray(rhs, dtype=float)
            solution = np.empty_like(rhs)
            rhs_pointer = rhs.ctypes.data_as(self._double_pointer)
            solution_pointer = solution.ctypes.data_as(self._double_pointer)
        else:
            solution = self._dummy
            rhs_pointer = solution_pointer = self._dummy_pointer
        self._error.value = 0
        self._native_call(*self._native_prefix, self._phase_arguments[phase],
            *self._native_matrix, rhs_pointer, solution_pointer, self._error_pointer)
        error = self._error.value
        if error in (-4, -7):
            raise FactorizationError(f'PARDISO phase {phase}: numerical error {error}')
        if error in (-2, -9):
            raise MemoryError(f'PARDISO phase {phase}: insufficient memory ({error})')
        if error:
            raise RuntimeError(f'PARDISO phase {phase}: error {error}')
        if phase == 33 and self.iparm[6] != 0:
            raise RuntimeError('PARDISO unexpectedly performed internal iterative refinement')
        return solution

    def update(self, matrix):
        np.take(matrix.data, self._positions, out=self._values)
        self._call(22)

    def solve(self, rhs):
        return self._call(33, self._rhs(rhs))

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
        return dict(backend=self.name, options=self.options.copy(), factor_nnz=int(self.iparm[17]),
                    factor_memory_kib=int(self.iparm[16]),
                    analysis_peak_kib=int(self.iparm[14]),
                    perturbed_pivots=int(self.iparm[13]), inner_refinements=int(self.iparm[6]),
                    positive_eigenvalues=int(self.iparm[21]),
                    negative_eigenvalues=int(self.iparm[22]),
                    ordering=self.options['ordering'], scaling=self.options['scaling'],
                    matching=self.options['matching'])


BACKENDS = {backend.name: backend for backend in (
    QDLDLFactorization, SuperLUFactorization, FeralFactorization, PardisoFactorization)}

_OPTION_DEFAULTS = {
    'qdldl': {},
    'superlu': {'ordering': 'COLAMD'},
    'feral': {'ordering': 'amd', 'scaling': 'auto'},
    'pardiso': {'ordering': 'metis', 'scaling': True, 'matching': True, 'pivot_exponent': 8},
}
_ORDERINGS = {
    'superlu': ('COLAMD', 'MMD_AT_PLUS_A', 'MMD_ATA', 'NATURAL'),
    'feral': ('amd', 'amf', 'metis', 'scotch', 'kahip', 'auto', 'auto_race'),
    'pardiso': ('amd', 'metis'),
}


def backend_type(name):
    """Resolve an explicit backend name; never silently fall back."""
    try:
        return BACKENDS[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Unknown linear backend {name!r}; choose {tuple(BACKENDS)}') from exc


def backend_options(name, supplied=None):
    """Validate explicit settings and return the canonical cache identity.

    These options select factorization implementations only. Internal iterative
    refinement, automatic fallback, and outer regularization are not options.
    """
    backend_type(name)
    defaults = _OPTION_DEFAULTS[name]
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ValueError('Linear backend options must be a mapping or None')
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f'Unsupported {name} options: {sorted(map(str, unknown))}')
    options = {**defaults, **supplied}
    if name in _ORDERINGS and options['ordering'] not in _ORDERINGS[name]:
        raise ValueError(f'Invalid {name} ordering {options["ordering"]!r}')
    if name == 'feral' and options['scaling'] not in ('auto', 'none', 'infnorm', 'mc64'):
        raise ValueError(f'Invalid FERAL scaling {options["scaling"]!r}')
    if name == 'pardiso':
        if any(type(options[key]) is not bool for key in ('scaling', 'matching')):
            raise ValueError('PARDISO scaling and matching must be booleans')
        if options['scaling'] and not options['matching']:
            raise ValueError('PARDISO symmetric scaling requires matching')
        if type(options['pivot_exponent']) is not int or not 1 <= options['pivot_exponent'] <= 20:
            raise ValueError('PARDISO pivot_exponent must be an integer from 1 to 20')
    return options
