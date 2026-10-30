"""Experimental, bounded active-set polishing for a convex quadratic program.

This module is independent of the interior-point solver. It guesses active
inequalities, solves an equality-constrained QP, and accepts the candidate only
after checking the complete original problem. See ``docs/qp-polish.md``.
"""
from dataclasses import dataclass, field
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy.sparse import bmat, csc_array, diags, vstack

from .backends import LinearSolveError, backend_options as validate_backend_options
from .linear import factor_and_solve


@dataclass(frozen=True)
class PolishSettings:
    strategy: str = 'one_shot_slack'
    active_tol: float = 1e-6
    dual_slack_ratio: float = 1.0
    max_passes: int = 12
    tolerance: float = 1e-8
    linear_atol: float = 1e-11
    linear_rtol: float = 1e-10
    max_refinement_steps: int = 20
    tau_reg: float = 1e-9
    max_solve_attempts: int = 8
    max_time: float = 10.0
    backend: str = 'qdldl'
    backend_options: dict = field(default_factory=dict)
    objective_constant: float = 0.0
    solve_mode: str = 'correction'

    def __post_init__(self):
        if self.strategy not in ('one_shot_slack', 'one_shot_ratio', 'add_drop', 'pdas'):
            raise ValueError('Unknown active-set polishing strategy')
        if self.solve_mode not in ('correction', 'absolute'):
            raise ValueError('solve_mode must be correction or absolute')
        for name in ('active_tol', 'linear_atol', 'linear_rtol', 'tau_reg'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('tolerance', 'dual_slack_ratio', 'max_time'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name, minimum in (('max_passes', 1), ('max_refinement_steps', 0),
                              ('max_solve_attempts', 1)):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if not np.isfinite(self.objective_constant):
            raise ValueError('objective_constant must be finite')
        object.__setattr__(self, 'backend_options',
                           validate_backend_options(self.backend, self.backend_options))


@dataclass
class PolishResult:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    accepted: bool
    status: str
    stop_reason: str
    before: dict
    after: dict
    history: list
    active_set: np.ndarray
    factorizations: int
    refinements: int
    elapsed_seconds: float
    candidate: dict | None = None
    failed_solve_calls: int = 0
    candidate_active_set: np.ndarray | None = None


def _maxabs(vector):
    return float(np.max(np.abs(vector), initial=0.0))


def _vector(value, size, name):
    if value is None and size == 0:
        return np.empty(0)
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real')
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f'{name} must be a finite vector with shape ({size},)')
    return vector


def _matrix(value, columns, name):
    if value is None:
        return csc_array((0, columns), dtype=np.float64)
    if np.iscomplexobj(value):
        raise ValueError(f'{name} must be real')
    matrix = csc_array(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != columns or not np.isfinite(matrix.data).all():
        raise ValueError(f'{name} must be finite and have {columns} columns')
    if not matrix.has_canonical_format:
        matrix = matrix.copy()
        matrix.sum_duplicates()
        matrix.sort_indices()
    return matrix


def _row_scales(matrix, rhs):
    scales = np.abs(rhs).copy()
    np.maximum.at(scales, matrix.indices, np.abs(matrix.data))
    scales[scales == 0] = 1.0
    return scales


def _scaled_rows(matrix, scales):
    result = matrix.copy()
    result.data /= scales[result.indices]
    return result


def _certificate(P, q, A, b, G, h, x, y, z, settings):
    """Same original-data gates as the benchmark's qp-kkt-v1 certificate."""
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        px, ax, gx = P @ x, A @ x, G @ x
        aty, gtz = A.T @ y, G.T @ z
        quadratic = float(0.5 * x @ px)
        objective = quadratic + float(q @ x) + settings.objective_constant
        dual_objective = -quadratic - float(b @ y) - float(h @ z) + settings.objective_constant
        primal = max(_maxabs((ax - b) / np.maximum(1.0, np.abs(b))),
                     _maxabs(np.maximum(gx - h, 0.0) / np.maximum(1.0, np.abs(h))))
        stationarity_scale = 1.0 + np.abs(px) + np.abs(q) + np.abs(aty) + np.abs(gtz)
        stationarity = _maxabs((px + q + aty + gtz) / stationarity_scale)
        complementarity_l1 = float(np.sum(np.abs(z * (h - gx))))
        complementarity = complementarity_l1 / max(1.0, abs(objective), abs(dual_objective))
        dual_sign = _maxabs(np.maximum(-z, 0.0) / np.maximum(1.0, np.abs(z)))
    finite = all(np.isfinite(value) for value in
                 (objective, dual_objective, primal, stationarity, complementarity_l1,
                  complementarity, dual_sign)) and all(np.isfinite(v).all() for v in
                 (x, y, z, px, ax, gx, aty, gtz, stationarity_scale))
    values = dict(objective=objective, dual_objective=dual_objective,
                  primal_residual_scaled=primal, stationarity_residual_scaled=stationarity,
                  complementarity_residual_scaled=complementarity,
                  dual_sign_violation_scaled=dual_sign)
    # None keeps diagnostic traces strict-JSON serializable on numerical failure.
    values = {key: float(value) if np.isfinite(value) else None for key, value in values.items()}
    score = max(primal, stationarity, complementarity, dual_sign) if finite else None
    return dict(values, finite=bool(finite), max_residual=score,
                accepted=bool(finite and score <= settings.tolerance))


def _ratio_set(slack, dual, ratio):
    return (slack < 0) | (dual > ratio * np.maximum(slack, 0.0))


def polish_qp(P, q, A, b, G, h, x, y, z, settings=None):
    """Polish ``min .5*x'P*x + q'x; A*x=b, G*x<=h`` with canonical duals.

    ``P`` must be full symmetric and positive semidefinite (convexity is assumed,
    not checked by a dense eigensolve). Empty constraints/duals may be ``None``.
    Every returned array is owned. Rejected attempts return exact copies of the
    input point; only fully certified, strictly improved candidates are accepted.
    The time budget is checked between sparse solves, not inside native calls.
    """
    start = perf_counter()
    settings = PolishSettings() if settings is None else settings
    if not isinstance(settings, PolishSettings):
        raise TypeError('settings must be PolishSettings or None')
    q_array = np.asarray(q)
    if q_array.ndim != 1 or not q_array.size:
        raise ValueError('q must be a nonempty vector')
    n = q_array.size
    P, A, G = _matrix(P, n, 'P'), _matrix(A, n, 'A'), _matrix(G, n, 'G')
    if P.shape != (n, n) or (P - P.T).nnz:
        raise ValueError('P must be square and full symmetric')
    p, m = A.shape[0], G.shape[0]
    q, b, h = _vector(q, n, 'q'), _vector(b, p, 'b'), _vector(h, m, 'h')
    x, y, z = _vector(x, n, 'x'), _vector(y, p, 'y'), _vector(z, m, 'z')
    before = _certificate(P, q, A, b, G, h, x, y, z, settings)
    output = [x.copy(), y.copy(), z.copy()]
    after, history, active_indices = before.copy(), [], np.empty(0, dtype=np.int64)
    accepted, status, stop_reason = False, 'unchanged', 'already_exact'
    factorizations = refinements = failed_solve_calls = 0
    last_candidate = candidate_active_set = None
    if before['max_residual'] == 0:
        return PolishResult(*output, accepted, status, stop_reason, before, after, history,
                            active_indices, factorizations, refinements, perf_counter() - start)

    # Row and objective scaling make active-set guesses invariant to nonzero
    # global objective/constraint rescaling. No dense matrix is constructed.
    a_scale, g_scale = _row_scales(A, b), _row_scales(G, h)
    objective_scale = max(_maxabs(P.data), _maxabs(q)) or 1.0
    scaled_P = P.copy()
    scaled_P.data /= objective_scale
    scaled_A, scaled_G = _scaled_rows(A, a_scale), _scaled_rows(G, g_scale)
    scaled_b, scaled_h = b / a_scale, h / g_scale
    slack, dual = (h - G @ x) / g_scale, z * (g_scale / objective_scale)
    if settings.strategy == 'one_shot_slack':
        active = slack <= settings.active_tol
    else:
        active = _ratio_set(slack, dual, settings.dual_slack_ratio)
    passes = 1 if settings.strategy.startswith('one_shot') else settings.max_passes
    visited, factor = set(), None
    status, stop_reason = 'rejected', 'pass_limit'
    try:
        for iteration in range(passes):
            if perf_counter() - start >= settings.max_time:
                status, stop_reason = 'time_limit', 'time_limit'
                break
            key = np.packbits(active).tobytes()
            if key in visited:
                status, stop_reason = 'cycle', 'repeated_active_set'
                break
            visited.add(key)
            active_indices = np.flatnonzero(active)
            B = csc_array(vstack((scaled_A, scaled_G[active_indices, :]), format='csc'))
            kkt = csc_array(bmat([[scaled_P, B.T], [B, None]], format='csc'))
            rhs = np.concatenate((-q / objective_scale, scaled_b, scaled_h[active_indices]))
            anchor = (np.concatenate((x, y * (a_scale / objective_scale),
                                      z[active_indices] * (g_scale[active_indices] / objective_scale)))
                      if settings.solve_mode == 'correction' else np.zeros_like(rhs))
            correction_rhs = rhs - kkt @ anchor
            shift = csc_array(diags(np.concatenate((np.ones(n), -np.ones(B.shape[0])))))
            trace = dict(pass_index=iteration, active_count=int(active.sum()), added=[], dropped=[],
                         active_set_sha256=sha256(key).hexdigest())
            history.append(trace)
            if not np.isfinite(correction_rhs).all() or not np.isfinite(anchor).all():
                trace.update(linear_failure='Nonfinite scaled working-set RHS', lin_attempts=0)
                status, stop_reason = 'linear_failure', 'nonfinite_working_set_rhs'
                break
            try:
                linear = factor_and_solve(
                    kkt, correction_rhs, shift, settings.tau_reg, factor,
                    target_atol=settings.linear_atol, target_rtol=settings.linear_rtol,
                    max_solve_attempts=settings.max_solve_attempts,
                    max_refinement_steps=settings.max_refinement_steps,
                    residual_blocks={'stationarity': slice(0, n), 'constraints': slice(n, None)},
                    backend=settings.backend, backend_options=settings.backend_options)
            except LinearSolveError as exc:
                failed_solve_calls += 1
                trace.update(linear_failure=str(exc), lin_attempts=None)
                status, stop_reason = 'linear_failure', 'linear_failure'
                break
            factor = linear.factorization
            factorizations += linear.info.attempts
            refinements += linear.info.refinements
            trace.update(linear.info.history_fields())
            solution = anchor + linear.solution
            candidate_x = solution[:n].copy()
            candidate_y = solution[n:n+p] * (objective_scale / a_scale)
            candidate_z = np.zeros(m)
            candidate_z[active_indices] = solution[n+p:] * (objective_scale / g_scale[active_indices])
            candidate = _certificate(P, q, A, b, G, h, candidate_x, candidate_y, candidate_z, settings)
            last_candidate = dict(x=candidate_x, y=candidate_y, z=candidate_z)
            candidate_active_set = active_indices.copy()
            trace['certificate'] = candidate
            baseline_score = before['max_residual'] if before['finite'] else np.inf
            if candidate['accepted'] and candidate['max_residual'] < baseline_score:
                output = [candidate_x, candidate_y, candidate_z]
                after, accepted, status, stop_reason = candidate, True, 'accepted', 'certified_improvement'
                break
            if candidate['accepted']:
                status, stop_reason = 'unchanged', 'no_certificate_improvement'
                break
            if not candidate['finite']:
                status, stop_reason = 'rejected', 'nonfinite_candidate'
                break
            if settings.strategy.startswith('one_shot'):
                stop_reason = 'certificate_rejected'
                break
            if linear.info.status != 'accurate':
                stop_reason = 'inaccurate_working_set_solve'
                break
            slack = (h - G @ candidate_x) / g_scale
            dual = candidate_z * (g_scale / objective_scale)
            updated = active.copy()
            if settings.strategy == 'pdas':
                updated = _ratio_set(slack, dual, settings.dual_slack_ratio)
            else:
                # Drop a wrong-sign active multiplier first. Otherwise add the
                # most violated inactive constraint. These are bounded heuristic
                # EQP working-set changes, not a feasible primal active-set method.
                negative = np.where(active, dual, np.inf)
                violated = np.where(~active, slack, np.inf)
                if negative.size and negative.min() < -settings.tolerance:
                    updated[int(np.argmin(negative))] = False
                elif violated.size and violated.min() < -settings.tolerance:
                    updated[int(np.argmin(violated))] = True
            trace['added'] = np.flatnonzero(updated & ~active).tolist()
            trace['dropped'] = np.flatnonzero(active & ~updated).tolist()
            if np.array_equal(updated, active):
                stop_reason = 'no_working_set_change'
                break
            active = updated
    finally:
        if factor is not None:
            factor.close()
    return PolishResult(*output, accepted, status, stop_reason, before, after, history,
                        active_indices.copy(), factorizations, refinements, perf_counter() - start,
                        last_candidate, failed_solve_calls, candidate_active_set)
