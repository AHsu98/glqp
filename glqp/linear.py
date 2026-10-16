"""Sparse backend solves with verified residuals against the requested system.

``G`` includes the outer algorithm's proximal regularization. ``tau_reg`` is
only a factorization shift: refinement always targets G, not the shifted matrix.
The backend owns sparse-pattern reuse; refinement and accuracy reporting are
independent of QDLDL so another backend can implement the same three operations.
"""
from dataclasses import dataclass
import numpy as np
import qdldl
from scipy.linalg import lstsq, norm
from scipy.sparse import csc_array
from .backends import (
    LinearSolveError, FactorizationError as _FactorizationError,
    QDLDLFactorization as _QDLDLFactorization, backend_type,
    backend_options as _backend_options,
)


class ShiftedSystemWorkspace:
    """Reuse canonical storage for ``G + tau * shift`` without changing G.

    Coefficients are always read afresh. Pattern comparisons include both
    matrices, and exact zero entries are removed just as in sparse addition.
    Noncanonical or non-float64 inputs use SciPy's original addition path.
    Returned storage is borrowed until the next call to ``assemble``.
    """

    def __init__(self):
        self.matrix = None

    def _matches(self, G, shift):
        return (self.matrix is not None and self.matrix.shape == G.shape
            and np.array_equal(G.indptr, self._g_indptr)
            and np.array_equal(G.indices, self._g_indices)
            and np.array_equal(shift.indptr, self._shift_indptr)
            and np.array_equal(shift.indices, self._shift_indices))

    def _build(self, G, shift):
        # Construct the structural union independently of coefficient zeros
        # and cancellations. Numeric assembly below handles those separately.
        left, right = G.copy(), shift.copy()
        left.data.fill(1.)
        right.data.fill(1.)
        self.matrix = csc_array(left + right)
        self._g_indptr, self._g_indices = G.indptr.copy(), G.indices.copy()
        self._shift_indptr = shift.indptr.copy()
        self._shift_indices = shift.indices.copy()

        def keys(matrix):
            columns = np.repeat(np.arange(matrix.shape[1], dtype=np.int64), np.diff(matrix.indptr))
            return columns*matrix.shape[0] + matrix.indices

        union = keys(self.matrix)
        self._g_slots = np.searchsorted(union, keys(G))
        self._shift_slots = np.searchsorted(union, keys(shift))

    def assemble(self, G, shift, tau):
        if (G.dtype != np.dtype('float64') or shift.dtype != np.dtype('float64')
                or not G.has_canonical_format or not shift.has_canonical_format):
            return csc_array(G + tau*shift)
        if not self._matches(G, shift):
            self._build(G, shift)
        data = self.matrix.data
        data.fill(0.)
        data[self._g_slots] = G.data
        data[self._shift_slots] += tau*shift.data
        if np.any(data == 0):
            # Keep the union available for later zero crossings, but give the
            # factor backend exactly the pattern ordinary addition produces.
            result = self.matrix.copy()
            result.eliminate_zeros()
            return result
        return self.matrix


def _maxnorm(vector):
    return float(np.max(np.abs(vector), initial=0.))


def _relative_linear_error(residual, rhs):
    # A common scale avoids squaring huge or subnormal unscaled values.
    scale = max(_maxnorm(residual), _maxnorm(rhs))
    if scale == 0:
        return 0.
    denominator = np.linalg.norm(rhs / scale)
    return float(np.linalg.norm(residual / scale) / denominator) if denominator else np.inf


def _projection_scale(rhs, image):
    scale = _maxnorm(image)
    if scale == 0 or not np.isfinite(scale):
        raise LinearSolveError("Linear direction has no finite nonzero image")
    scaled = image / scale
    value = np.dot(rhs / scale, scaled) / np.dot(scaled, scaled)
    if not np.isfinite(value):
        raise LinearSolveError("Nonfinite linear refinement scale")
    return float(value)


def _gmres_refine(G, rhs, x, inverse, max_steps, atol, rtol):
    """One bounded right-GMRES correction solve with fresh float64 residuals.

    The factor inverse and Arnoldi work stay in float64. Store B(v_j) so the
    correction uses the same rounded directions as the Arnoldi products.
    Residual evaluation uses the same arithmetic as the original IR solver.
    """
    def residual(candidate):
        return rhs - G @ candidate

    best_x = x
    best_residual = residual(x)
    if not np.all(np.isfinite(best_residual)):
        raise _FactorizationError("Nonfinite initial linear residual")
    best_relative = _relative_linear_error(best_residual, rhs)
    absolute = _maxnorm(best_residual)
    best_accurate = absolute <= atol or best_relative <= rtol
    polish = absolute > 0 and (absolute >= .1 * atol or best_relative > rtol)
    if best_accurate and not polish:
        return best_x, best_residual, 0, 'tolerance'

    limit = min(max_steps, len(rhs))
    if limit == 0:
        return best_x, best_residual, 0, 'refinement_limit'
    correction_rhs = np.asarray(best_residual, dtype=float)
    beta = norm(correction_rhs)
    if beta == 0:
        return best_x, best_residual, 0, 'zero_computed_residual'
    V = np.zeros((len(rhs), limit + 1), order='F')
    Z = np.zeros((len(rhs), limit), order='F')
    H = np.zeros((limit + 1, limit))
    V[:, 0] = correction_rhs / beta
    small_rhs = np.zeros(limit + 1)
    small_rhs[0] = beta
    reason, refinements = 'refinement_limit', 0
    for j in range(limit):
        Z[:, j] = inverse(V[:, j])
        refinements += 1
        if not np.all(np.isfinite(Z[:, j])):
            reason = 'nonfinite_correction'
            break
        w = G @ Z[:, j]
        if not np.all(np.isfinite(w)):
            reason = 'nonfinite_correction'
            break
        before = norm(w)
        # Two-pass modified Gram-Schmidt; do not form normal equations.
        for _ in range(2):
            for i in range(j + 1):
                h = np.dot(V[:, i], w)
                H[i, j] += h
                w -= h * V[:, i]
        after = norm(w)
        H[j + 1, j] = after
        try:
            coefficients = lstsq(H[:j+2, :j+1], small_rhs[:j+2],
                                 lapack_driver='gelsd')[0]
        except np.linalg.LinAlgError:
            reason = 'small_solve_failure'
            break
        trial = x + Z[:, :j+1] @ coefficients
        trial_residual = residual(trial)
        if not np.all(np.isfinite(trial)) or not np.all(np.isfinite(trial_residual)):
            reason = 'nonfinite_correction'
            break
        relative = _relative_linear_error(trial_residual, rhs)
        accurate = _maxnorm(trial_residual) <= atol or relative <= rtol
        improved = relative < best_relative
        if improved or (accurate and not best_accurate):
            best_x, best_residual, best_relative = trial, trial_residual, relative
            best_accurate = accurate
        if best_accurate:
            reason = 'tolerance' if improved else 'stagnated'
            break
        # Subspace exhaustion is distinct from a verified accurate direction.
        if after == 0 or after <= np.finfo(float).eps * before:
            reason = 'arnoldi_breakdown'
            break
        V[:, j+1] = w / after
    return best_x, best_residual, refinements, reason


@dataclass(frozen=True)
class LinearSolveInfo:
    status: str
    stop_reason: str
    absolute_error: float
    relative_error: float
    target_atol: float
    target_rtol: float
    tau_reg: float
    attempts: int
    refinements: int
    block_errors: dict
    backend: str = 'qdldl'

    def history_fields(self):
        fields = dict(lin_status=self.status, lin_stop=self.stop_reason,
                      lin_abs_res=self.absolute_error, lin_rel_res=self.relative_error,
                      lin_atol=self.target_atol, lin_rtol=self.target_rtol,
                      lin_tau=self.tau_reg, lin_attempts=self.attempts,
                      lin_backend=self.backend)
        for name, errors in self.block_errors.items():
            fields[f'lin_{name}_abs'] = errors['absolute']
            fields[f'lin_{name}_rhs_scaled'] = errors['rhs_scaled']
        return fields


@dataclass(frozen=True)
class LinearSolveResult:
    solution: np.ndarray
    factorization: object
    info: LinearSolveInfo

    # Preserve the four-value unpacking/indexing used by existing callers and
    # diagnostic probes. New callers can use the explicit result and info fields.
    def __iter__(self):
        return iter((self.solution, self.info.refinements,
                     self.factorization, self.info.relative_error))

    def __getitem__(self, index):
        return tuple(self)[index]


def factor_and_solve(
    G, rhs, reg_shift, init_tau_reg, solver, target_atol=1e-10,
    max_solve_attempts=20, max_refinement_steps=10, *, target_rtol=1e-6,
    residual_blocks=None, backend='qdldl', backend_options=None, workspace=None,
):
    """Return the best verified finite direction and an explicit accuracy status.

    Accuracy preserves the existing final criterion: absolute infinity-norm
    error <= atol OR relative L2 error <= rtol. A finite inaccurate result is
    explicitly labeled; the outer algorithm decides whether to use that inexact
    step. Block diagnostics use max(1, ||rhs_block||_inf) as their scale and do
    not silently change the acceptance criterion.

    Only backend numerical failures retry with a larger factorization shift.
    Timeouts and programming/input errors propagate. A zero initial shift
    disables shift escalation. One right-GMRES correction cycle uses at most
    max_refinement_steps factor solves (default ten), retaining the best verified
    direction. Residuals are freshly evaluated against the requested G in
    float64, as are factor applications and Krylov arithmetic.
    With a nonzero residual, the first correction also polishes an otherwise
    accurate direction when abs_error >= 0.1 * atol or rel_error > rtol.
    An optional ShiftedSystemWorkspace reuses factorization-input storage;
    refinement still reads G and evaluates every residual afresh.
    """
    # Solver-owned CSC arrays already have the required representation.
    # Keep normalization for other formats, matrix classes, and subclasses.
    if type(G) is not csc_array:
        G = csc_array(G)
    if type(reg_shift) is not csc_array:
        reg_shift = csc_array(reg_shift)
    rhs = np.asarray(rhs, dtype=float)
    if rhs.ndim != 1 or G.shape != (rhs.size, rhs.size) or reg_shift.shape != G.shape:
        raise ValueError("Linear matrix, shift, and right-hand side shapes must agree")
    if not np.all(np.isfinite(rhs)) or not np.all(np.isfinite(G.data)):
        raise ValueError("Linear system must be finite")
    if not np.all(np.isfinite(reg_shift.data)):
        raise ValueError("Factorization shift must be finite")
    if (not np.isfinite(init_tau_reg) or init_tau_reg < 0
            or not np.isfinite(target_atol) or target_atol < 0
            or not np.isfinite(target_rtol) or target_rtol < 0
            or max_solve_attempts < 1 or max_refinement_steps < 0):
        raise ValueError("Invalid linear-solve limits or tolerances")
    blocks = residual_blocks or {}
    # Preserve the historical instrumentation hook while resolving every
    # alternate backend explicitly. A wrong name never silently selects QDLDL.
    factory = _QDLDLFactorization if backend == 'qdldl' else backend_type(backend)
    options = _backend_options(backend, backend_options)

    def close(factor):
        release = getattr(factor, 'close', None)
        if release is not None:
            release()

    def result(x, residual, factorization, tau, attempts, refinements, reason):
        absolute = _maxnorm(residual)
        relative = _relative_linear_error(residual, rhs)
        accurate = absolute <= target_atol or relative <= target_rtol
        if accurate and reason == 'refinement_limit':
            reason = 'tolerance'
        errors = {name: {'absolute': _maxnorm(residual[part]),
                         'rhs_scaled': _maxnorm(residual[part]) / max(1., _maxnorm(rhs[part]))}
                  for name, part in blocks.items()}
        info = LinearSolveInfo('accurate' if accurate else 'inaccurate', reason,
                               absolute, relative, target_atol, target_rtol,
                               tau, attempts, refinements, errors, backend)
        return LinearSolveResult(x, factorization, info)

    if not np.any(rhs):
        return result(np.zeros_like(rhs), np.zeros_like(rhs), solver, 0., 0, 0, 'zero_rhs')

    tau = init_tau_reg
    for attempt in range(1, max_solve_attempts + 1):
        try:
            shifted = (csc_array(G + tau * reg_shift) if workspace is None
                       else workspace.assemble(G, reg_shift, tau))
            shifted.sum_duplicates()
            shifted.sort_indices()
            if not np.all(np.isfinite(shifted.data)):
                raise LinearSolveError("Nonfinite shifted factorization matrix")
            if (not isinstance(solver, factory) or getattr(solver, 'options', {}) != options
                    or not solver.matches(shifted)):
                close(solver)
                solver = factory(shifted, options=options) if options else factory(shifted)
            else:
                solver.update(shifted)
            x = solver.solve(rhs)
            if not np.all(np.isfinite(x)):
                raise _FactorizationError("Backend returned a nonfinite linear solution")
            x = x * _projection_scale(rhs, G @ x)
            if not np.all(np.isfinite(x)):
                raise _FactorizationError("Nonfinite projected linear solution or residual")
            x, residual, refinements, reason = _gmres_refine(
                G, rhs, x, solver.solve, max_refinement_steps, target_atol, target_rtol)
            return result(x, residual, solver, tau, attempt, refinements, reason)
        except _FactorizationError:
            close(solver)
            if attempt == max_solve_attempts or tau == 0:
                raise
            # A failed update may leave the cached factors unusable.
            solver = None
            tau *= 10.
    raise AssertionError("Unreachable linear-solve retry state")
