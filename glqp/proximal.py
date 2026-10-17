"""Fixed-center primal/dual proximal equations and their globalization.

The default solver never enters this module's numerical path. The centered
experiment uses standard complementarity, exact nonlinear trial gradients,
and one fixed residual map during each predictor/corrector and line search.
Factorization-only stabilization remains the linear solver's responsibility.
"""
from dataclasses import dataclass
import numbers

import numpy as np

from .diagnostics import start_line_search_diagnostics, reject_trial


# Fixed choices in the centered experiments, separate from SolverSettings.
_REGULARIZATION_STALL_LIMIT = 5
_REGULARIZATION_REDUCTION = .1
_REGULARIZATION_OFFSET_FRACTION = .1
_BACKTRACKING_FACTOR = .5


def inf_norm(value):
    return float(np.max(np.abs(value), initial=0.))


def validate_proximal_settings(settings):
    if (not isinstance(settings.proximal_policy, str)
            or settings.proximal_policy not in ('legacy', 'no_slack', 'centered')):
        raise ValueError("proximal_policy must be 'legacy', 'no_slack', or 'centered'")
    for name in ('prox_reg', 'prox_dual_reg'):
        value = getattr(settings, name)
        if value is None and name == 'prox_dual_reg':
            continue
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative')
    for name in ('proximal_center_reduction', 'proximal_inner_tolerance'):
        value = getattr(settings, name)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or not 0 < value < 1):
            raise ValueError(f'{name} must be strictly between zero and one')
    limit = settings.proximal_corrector_limit
    if limit is not None and (isinstance(limit, (bool, np.bool_))
            or not isinstance(limit, numbers.Real) or not np.isfinite(limit) or not 0 < limit < 1):
        raise ValueError('proximal_corrector_limit must be None or strictly between zero and one')
    if settings.proximal_policy == 'legacy' and settings.prox_dual_reg is not None:
        raise ValueError('Independent dual regularization requires a nonlegacy proximal policy')
    if (not isinstance(settings.proximal_reg_update, str)
            or settings.proximal_reg_update not in ('residual', 'progress')):
        raise ValueError("proximal_reg_update must be 'residual' or 'progress'")
    if settings.proximal_reg_update == 'progress' and settings.proximal_policy != 'centered':
        raise ValueError('Progress regularization updates require the centered proximal policy')


def _center(value):
    result = value.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class FixedProximalModel:
    """One residual map shared by a Newton/PC direction and its line search.

    Centers are immutable arrays owned by ``ProximalState``. Updating that
    state replaces its arrays, so an already frozen model remains unchanged.
    ``rho`` shifts stationarity; ``delta`` shifts both feasibility blocks.
    Complementarity remains ``s*y - target`` and has no proximal slack term.
    """

    x_center: np.ndarray
    y_center: np.ndarray
    nu_center: np.ndarray
    rho: float
    delta: float
    generation: int

    def residuals(self, x, y, nu, rx, rp, req):
        return (rx + self.rho*(x-self.x_center),
                rp - self.delta*(y-self.y_center),
                req - self.delta*(nu-self.nu_center))

    def offsets(self, x, y, nu):
        return dict(stationarity=self.rho*(x-self.x_center),
                    primal=-self.delta*(y-self.y_center),
                    equality=-self.delta*(nu-self.nu_center))


class ProximalState:
    """Accepted-point state: centers, regularization caps, and progress gates."""

    def __init__(self, x, y, nu, rx, rp, req, *, rho_cap=np.inf, delta_cap=np.inf):
        self.x, self.y, self.nu = map(_center, (x, y, nu))
        self.stationarity = inf_norm(rx)
        self.feasibility = max(inf_norm(rp), inf_norm(req))
        self.generation = 0
        self.rho_cap, self.delta_cap = float(rho_cap), float(delta_cap)
        self.primal_progress_misses = self.dual_progress_misses = 0

    def select_regularization(self, rho, delta):
        """Apply persistent nonincreasing caps before freezing a new model."""
        self.rho_cap = min(self.rho_cap, float(rho))
        self.delta_cap = min(self.delta_cap, float(delta))
        return self.rho_cap, self.delta_cap

    def freeze(self, rho, delta):
        return FixedProximalModel(
            self.x, self.y, self.nu, float(rho), float(delta), self.generation)

    def update(self, fixed, x, y, nu, rx, rp, rc, req, target, settings):
        """Advance state only after finalizing the accepted point and repair.

        Residuals are from the original problem and ``rc`` is ``s*y``. Inner
        accuracy is measured with the old frozen model and the search target,
        before replacing any centers or selecting the next coefficients.
        """
        rd, rpi, re = fixed.residuals(x, y, nu, rx, rp, req)
        proximal_norm = max(inf_norm(rd), inf_norm(rpi), inf_norm(rc-target), inf_norm(re))
        inner_limit = settings.proximal_inner_tolerance*max(target, settings.tol)
        inner_converged = proximal_norm <= inner_limit
        stationarity = inf_norm(rx)
        feasibility = max(inf_norm(rp), inf_norm(req))
        primal_progress = (stationarity < self.stationarity and
                           stationarity <= settings.proximal_center_reduction*self.stationarity)
        dual_progress = (feasibility < self.feasibility and
                         feasibility <= settings.proximal_center_reduction*self.feasibility)
        move_x = inner_converged or primal_progress
        move_dual = inner_converged or dual_progress
        offsets = {key: inf_norm(value) for key, value in fixed.offsets(x, y, nu).items()}
        regularization_update = None
        if settings.proximal_reg_update == 'progress':
            regularization_update = self._update_regularization(
                fixed, offsets, stationarity, feasibility, primal_progress,
                dual_progress, inner_converged, settings.tol)
        if inner_converged:
            refresh_reason = 'inner_accuracy'
        elif move_x or move_dual:
            refresh_reason = 'original_block_progress'
        else:
            refresh_reason = 'none'
        metrics = dict(
            generation=fixed.generation,
            rho=fixed.rho,
            delta=fixed.delta,
            center_stationarity_before=self.stationarity,
            center_feasibility_before=self.feasibility,
            original_stationarity_after=stationarity,
            original_feasibility_after=feasibility,
            fixed_target_proximal_residual_inf=proximal_norm,
            inner_limit=inner_limit,
            inner_converged=bool(inner_converged),
            primal_progress=bool(primal_progress),
            dual_progress=bool(dual_progress),
            primal_center_updated=bool(move_x),
            dual_centers_updated=bool(move_dual),
            refresh_reason=refresh_reason,
            center_offset_inf=offsets,
        )
        if regularization_update is not None:
            metrics['regularization_update'] = regularization_update
        if move_x:
            self.x = _center(x)
        if move_dual:
            self.y, self.nu = _center(y), _center(nu)
        parameters_changed = (regularization_update is not None
                              and any(row['reduced'] for row in regularization_update.values()))
        if move_x or move_dual or parameters_changed:
            self.generation += 1
        self.stationarity, self.feasibility = stationarity, feasibility
        metrics['next_generation'] = self.generation
        return metrics

    def _update_regularization(self, fixed, offsets, stationarity, feasibility,
                               primal_progress, dual_progress, inner_converged, tol):
        """Schedule independent reductions after an accepted, finalized point.

        Five missing progress gates alone are not evidence that regularization
        is causing a stall. Require its center offset to account for at least
        one tenth of the corresponding original residual, or complete the
        current proximal subproblem. The next model uses the new coefficients;
        the frozen model used by the accepted PC pair/search stays untouched.
        """
        rows = {}
        blocks = (
            ('primal', 'rho_cap', 'primal_progress_misses', fixed.rho,
             stationarity, offsets['stationarity'], primal_progress),
            ('dual', 'delta_cap', 'dual_progress_misses', fixed.delta,
             feasibility, max(offsets['primal'], offsets['equality']), dual_progress),
        )
        for name, cap_name, counter_name, coefficient, residual, offset, progress in blocks:
            old_cap = getattr(self, cap_name)
            old_misses = getattr(self, counter_name)
            misses = 0 if progress else old_misses+1
            dominance_limit = _REGULARIZATION_OFFSET_FRACTION*max(residual, tol)
            offset_dominates = offset >= dominance_limit
            stalled = misses >= _REGULARIZATION_STALL_LIMIT
            trigger = inner_converged or (stalled and offset_dominates)
            new_cap = min(old_cap, _REGULARIZATION_REDUCTION*coefficient) if trigger else old_cap
            if inner_converged:
                reason = 'inner_accuracy'
            elif trigger:
                reason = 'stalled_proximal_offset'
            elif progress:
                reason = 'progress'
            elif stalled:
                reason = 'offset_below_gate'
            else:
                reason = 'waiting_for_progress'
            rows[name] = dict(
                coefficient_before=float(coefficient),
                cap_before=old_cap,
                cap_after=new_cap,
                progress=bool(progress),
                misses_before=old_misses,
                misses_observed=misses,
                misses_after=0 if trigger else misses,
                stall_limit=_REGULARIZATION_STALL_LIMIT,
                stalled=bool(stalled),
                center_offset_inf=offset,
                original_residual_inf=residual,
                dominance_limit=dominance_limit,
                offset_dominates=bool(offset_dominates),
                inner_converged=bool(inner_converged),
                reduction_factor=_REGULARIZATION_REDUCTION,
                triggered=bool(trigger),
                reduced=bool(new_cap < old_cap),
                reason=reason,
            )
            setattr(self, cap_name, new_cap)
            setattr(self, counter_name, 0 if trigger else misses)
        return rows


def residual_vector(rx, rp, rc, req):
    """Keep one block order for the merit and its directional derivative."""
    return np.concatenate((rx, rp, rc, req))


def residual_slope(model, H, dx, ds, dy, dnu, y, s, rx, rp, rc, req, rho, delta):
    """Derivative for the supplied direction and fixed starting residual.

    Callers pass the direction after any clipping. This uses actual Jacobian
    products rather than assuming the linear solve supplied an exact Newton
    direction; a result marked inaccurate can still reach this calculation.
    """
    residual = residual_vector(rx, rp, rc, req)
    scale = max(1., inf_norm(residual))
    jacobian_direction = residual_vector(
        H@dx + rho*dx + model.C.T@dy + model.E.T@dnu,
        model.C@dx + ds - delta*dy, y*ds + s*dy, model.E@dx - delta*dnu)
    with np.errstate(over='ignore', invalid='ignore'):
        return float(np.dot(residual/scale, jacobian_direction/scale))


def merit_value(blocks, scale):
    with np.errstate(over='ignore', invalid='ignore'):
        residual = residual_vector(*blocks)/scale
        return .5*float(np.dot(residual, residual))


@dataclass
class ProximalTrial:
    step: float
    x: np.ndarray
    s: np.ndarray
    y: np.ndarray
    nu: np.ndarray
    z: np.ndarray
    products: tuple
    glm_gradient: object


def search_proximal_step(model, fixed, *, x, y, s, nu, z, Qx, primal, H,
                        residuals, direction, target, tmax, slope_hint=None):
    """Armijo on the exact fixed-center KKT residual, with no free fallback.

    Uniform residual scaling is frozen for this search. It prevents overflow
    in a squared norm without changing relative block weighting or acceptance.
    The exact GLM gradient and finite objective are checked at every trial.
    """
    settings = model.settings
    rx, rp, rc, req = residuals
    rd, rpi, re = fixed.residuals(x, y, nu, rx, rp, req)
    current = (rd, rpi, rc-target, re)
    scale = max(1., *(inf_norm(v) for v in current))
    merit0 = merit_value(current, scale)
    dx, ds, dy, dnu = direction
    slope = (residual_slope(model, H, dx, ds, dy, dnu, y, s,
                           rd, rpi, rc-target, re, fixed.rho, fixed.delta)
             if slope_hint is None else slope_hint)
    diag = start_line_search_diagnostics(tmax, slope, fallback_policy='off',
                                        perturbed_kkt_before=max(inf_norm(v) for v in current))
    diag.update(merit_kind='fixed_center_squared_kkt', residual_scale=scale,
                merit_before=merit0, merit_after=None, target=float(target),
                center_generation=fixed.generation, effective_armijo_additive_eps=0.)
    # This policy requires descent before attempting the Armijo search.
    if not np.isfinite(slope) or slope >= 0:
        reject_trial(diag, 'non_descent_proximal_direction')
        return None, diag
    dz = z if model.dummy_A else model.A@dx
    Qdx = model.Q@dx
    AT = model._transpose_views.get('A', model.A)
    xQx, xQdx, dxQdx = np.dot(x, Qx), np.dot(x, Qdx), np.dot(dx, Qdx)
    t = tmax
    for _ in range(settings.max_linesearch_steps):
        diag['trial_count'] += 1
        if not np.isfinite(t) or t <= 0:
            reject_trial(diag, 'invalid_step')
            break
        tx, ts, ty, tn = x+t*dx, s+t*ds, y+t*dy, nu+t*dnu
        if (not all(np.all(np.isfinite(v)) for v in (tx, ts, ty, tn))
                or np.any(ts <= 0) or np.any(ty <= 0)):
            reject_trial(diag, 'nonfinite_or_nonpositive_trial')
            t *= _BACKTRACKING_FACTOR
            continue
        tz = z if model.dummy_A else z+t*dz
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            objective = model.f(tz)+.5*(xQx+2*t*xQdx+t*t*dxQdx)-np.dot(tx,model.b)
            glm_gradient = 0. if model.dummy_A else AT@model.f.d1f(tz)
            gradient = glm_gradient+Qx+t*Qdx
            products = model._constraint_products(tx, ty, tn)
            bx, bp, bc, be = model._kkt_residuals(gradient, ty, ts, products)
            brd, brp, bre = fixed.residuals(tx, ty, tn, bx, bp, be)
            value = merit_value((brd, brp, bc-target, bre), scale)
        diag['residual_check_count'] += 1
        if (np.isfinite(objective) and np.isfinite(value)
                and value <= merit0+settings.armijo_param*t*slope):
            diag.update(acceptance_reason='proximal_armijo', accepted_step=float(t),
                        accepted_fraction=float(t/tmax), merit_after=value)
            return ProximalTrial(t, tx, ts, ty, tn, tz, products, glm_gradient), diag
        diag['armijo_failure_count'] += 1
        reject_trial(diag, 'nonfinite_proximal_trial' if not np.isfinite(value) or not np.isfinite(objective)
                     else 'proximal_armijo_failed')
        t *= _BACKTRACKING_FACTOR
    return None, diag


def guard_proximal_repair(fixed, x, y, nu, rx, rp, rc, req, proposed_rp, proposed_rc,
                          target, search_diag):
    """A slack-only repair may not increase the accepted fixed-center merit."""
    rd, rpi, re = fixed.residuals(x,y,nu,rx,rp,req)
    _, candidate_rp, _ = fixed.residuals(x,y,nu,rx,proposed_rp,req)
    scale = search_diag['residual_scale']
    before = merit_value((rd,rpi,rc-target,re), scale)
    after = merit_value((rd,candidate_rp,proposed_rc-target,re), scale)
    return bool(np.isfinite(after) and after <= before), dict(
        proximal_merit_before=before, proximal_merit_candidate=after,
        proximal_merit_guard=True)
