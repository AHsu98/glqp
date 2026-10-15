"""QDLDL-backed solves with verified residuals against the requested system.

``G`` includes the outer algorithm's proximal regularization. ``tau_reg`` is
only a factorization shift: refinement always targets G, not the shifted matrix.
The backend owns sparse-pattern reuse; refinement and accuracy reporting are
independent of QDLDL so another backend can implement the same three operations.
"""
from dataclasses import dataclass
import numpy as np
import qdldl
from scipy.sparse import csc_array


class LinearSolveError(RuntimeError):
    """A numerical failure prevented a finite linear direction."""


class _FactorizationError(LinearSolveError):
    """A recoverable backend failure; a larger numerical shift may help."""


class _QDLDLFactorization:
    def __init__(self, matrix):
        self.shape = matrix.shape
        self.indptr = matrix.indptr.copy()
        self.indices = matrix.indices.copy()
        try:
            self.factor = qdldl.Solver(matrix)
        except RuntimeError as exc:
            raise _FactorizationError(str(exc)) from exc

    def matches(self, matrix):
        return (self.shape == matrix.shape
                and np.array_equal(self.indptr, matrix.indptr)
                and np.array_equal(self.indices, matrix.indices))

    def update(self, matrix):
        try:
            self.factor.update(matrix)
        except RuntimeError as exc:
            raise _FactorizationError(str(exc)) from exc

    def solve(self, rhs):
        try:
            return self.factor.solve(rhs)
        except RuntimeError as exc:
            raise _FactorizationError(str(exc)) from exc


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

    def history_fields(self):
        fields = dict(lin_status=self.status, lin_stop=self.stop_reason,
                      lin_abs_res=self.absolute_error, lin_rel_res=self.relative_error,
                      lin_atol=self.target_atol, lin_rtol=self.target_rtol,
                      lin_tau=self.tau_reg, lin_attempts=self.attempts)
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
    max_solve_attempts=20, max_refinement_steps=5, *, target_rtol=1e-6,
    residual_blocks=None,
):
    """Return the best verified finite direction and an explicit accuracy status.

    Accuracy preserves the existing final criterion: absolute infinity-norm
    error <= atol OR relative L2 error <= rtol. A finite inaccurate result is
    explicitly labeled; the outer algorithm decides whether to use that inexact
    step. Block diagnostics use max(1, ||rhs_block||_inf) as their scale and do
    not silently change the acceptance criterion.

    Only backend numerical failures retry with a larger factorization shift.
    Timeouts and programming/input errors propagate. A zero initial shift
    disables shift escalation. At most max_refinement_steps corrections are
    tried per factorization; an unhelpful correction never replaces the best one.
    With a nonzero residual, the first correction also polishes an otherwise
    accurate direction when abs_error >= 0.1 * atol or rel_error > rtol.
    """
    G = csc_array(G)
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
                               tau, attempts, refinements, errors)
        return LinearSolveResult(x, factorization, info)

    if not np.any(rhs):
        return result(np.zeros_like(rhs), np.zeros_like(rhs), solver, 0., 0, 0, 'zero_rhs')

    tau = init_tau_reg
    for attempt in range(1, max_solve_attempts + 1):
        try:
            shifted = csc_array(G + tau * reg_shift)
            shifted.sum_duplicates()
            shifted.sort_indices()
            if not np.all(np.isfinite(shifted.data)):
                raise LinearSolveError("Nonfinite shifted factorization matrix")
            if not isinstance(solver, _QDLDLFactorization) or not solver.matches(shifted):
                solver = _QDLDLFactorization(shifted)
            else:
                solver.update(shifted)
            x = solver.solve(rhs)
            if not np.all(np.isfinite(x)):
                raise _FactorizationError("Backend returned a nonfinite linear solution")
            x = x * _projection_scale(rhs, G @ x)
            residual = rhs - G @ x
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(residual)):
                raise _FactorizationError("Nonfinite projected linear solution or residual")
            reason, refinements = 'refinement_limit', 0
            for _ in range(max_refinement_steps):
                relative = _relative_linear_error(residual, rhs)
                absolute = _maxnorm(residual)
                already_accurate = absolute <= target_atol or relative <= target_rtol
                # Restore the original first-polish trigger within the same
                # bounded loop; subsequent corrections use the final criterion.
                polish = refinements == 0 and absolute > 0 and (
                    absolute >= 0.1 * target_atol or relative > target_rtol)
                if already_accurate and not polish:
                    reason = 'tolerance'
                    break
                correction = solver.solve(residual)
                refinements += 1
                if not np.all(np.isfinite(correction)):
                    reason = 'nonfinite_correction'
                    break
                image = G @ correction
                try:
                    alpha = _projection_scale(residual, image)
                except LinearSolveError:
                    reason = 'unusable_correction'
                    break
                trial = x + alpha * correction
                # Always measure the residual of the stored trial, including
                # rounding in the addition. Recursive residuals can drift.
                trial_residual = rhs - G @ trial
                if not np.all(np.isfinite(trial)) or not np.all(np.isfinite(trial_residual)):
                    reason = 'nonfinite_correction'
                    break
                trial_relative = _relative_linear_error(trial_residual, rhs)
                accurate = _maxnorm(trial_residual) <= target_atol or trial_relative <= target_rtol
                if trial_relative >= relative and (already_accurate or not accurate):
                    reason = 'stagnated'
                    break
                x, residual = trial, trial_residual
                if accurate:
                    reason = 'tolerance'
                    break
            return result(x, residual, solver, tau, attempt, refinements, reason)
        except _FactorizationError:
            if attempt == max_solve_attempts or tau == 0:
                raise
            # A failed update may leave the cached factors unusable.
            solver = None
            tau *= 10.
    raise AssertionError("Unreachable linear-solve retry state")
