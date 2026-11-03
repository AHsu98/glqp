"""Objective-plus-barrier search with explicit restoration acceptance.

The shared controller schedules trials and proposals. This policy retains the
legacy Armijo, original-KKT restoration, and configured Newton fallback tests.
Unlike the strict GLM policy in linesearch.py, it can accept a merit increase.
Cached ray coefficients use the original Q and the actual clipped direction.
"""
from dataclasses import dataclass

import numpy as np

from .diagnostics import reject_trial, record_fallback_progress
from .qp_linesearch import QPScalarProposer
from .qp_ray import QPResidualNorms, QPResidualRay
from .search_control import TrialDecision, search_trials
from .util import maxnorm


@dataclass
class ObjectiveTrial:
    step: float
    merit: float
    predictor: np.ndarray | None
    slack: np.ndarray
    x: np.ndarray
    y: np.ndarray
    interior: bool
    products: object = None
    glm_gradient: object = None


class CachedObjectiveRay:
    """Legacy merit evaluator with the same interface as CachedGLMRay.

    ``point`` is (x, s, y, nu, z), ``direction`` is (dx, ds, dy, dnu), and
    ``quadratic`` is (Qx, Qdx, xQx, xQdx, dxQdx). All arrays are borrowed.
    Keep dot(trial_x, b), the public loss callback, and zero-mu barrier
    arithmetic here: regrouping them into the fused GLM evaluator can change
    borderline acceptance decisions. No matrix product is needed for a value.
    """

    def __init__(self, model, *, point, direction, dz, quadratic, mu, transpose):
        self.model = model
        self.transpose = transpose
        self.point, self.direction = point, direction
        self.dz, self.quadratic, self.mu = dz, quadratic, mu

    def evaluate(self, step):
        x, s, y, nu, z = self.point
        dx, ds, dy, dnu = self.direction
        Qx, Qdx, xQx, xQdx, dxQdx = self.quadratic
        trial_x = x + step*dx
        trial_s, trial_y = s + step*ds, y + step*dy
        interior = bool(np.all(np.isfinite(trial_x))
                        and np.all(np.isfinite(trial_s))
                        and np.all(np.isfinite(trial_y))
                        and not np.any(trial_s <= 0) and not np.any(trial_y <= 0))
        trial_z, merit = None, np.inf
        if interior:
            trial_z = z if self.model.dummy_A else z + step*self.dz
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                primal = (self.model.f(trial_z)
                          + .5*(xQx + 2*step*xQdx + step*step*dxQdx)
                          - np.dot(trial_x, self.model.b))
                barrier = -self.mu*np.sum(np.log(trial_s))
                merit = primal + barrier
        return ObjectiveTrial(step, merit, trial_z, trial_s, trial_x, trial_y, interior)


def search_restoration_ray(ray, settings, *, tmax, merit0, slope0, residuals,
                           kkt_res, perturbed_kkt_before, residual_norms=None,
                           diagnostics=None, proposer_factory=QPScalarProposer):
    """Return the accepted trial and the reusable per-solve screening norms.

    Residual screens may only reject. Every residual/fallback acceptance uses
    fresh products, stored on that trial alone for accepted-iterate reuse.
    The strict GLM Armijo policy uses the same controller through
    :func:`linesearch.search_scalar_ray`; centered proximal merit stays separate.
    """
    model = ray.model
    x, s, y, nu, z = ray.point
    dx, ds, dy, dnu = ray.direction
    Qx, Qdx, xQx, xQdx, dxQdx = ray.quadratic
    rx, rp, req = residuals
    mu = ray.mu
    qp_scalar = (model.QP_mode and settings.qp_linesearch == 'scalar'
                 and np.isfinite(slope0) and slope0 < 0)
    residual_acceptance = model.QP_mode or slope0 >= 0
    qp_proposer = residual_ray = None
    if diagnostics is not None:
        diagnostics['algorithm'] = 'qp_scalar' if qp_scalar else 'backtracking'
        if model.QP_mode:
            diagnostics.update(qp_scalar_requested=settings.qp_linesearch == 'scalar',
                qp_scalar_eligible=bool(qp_scalar), qp_scalar_proposals=[],
                qp_scalar_counters=None, fallback_eligible_count=0,
                residual_screen=settings.qp_residual_screen,
                residual_screen_checks=0, complementarity_rejections=0,
                affine_screen_checks=0, affine_rejections=0,
                affine_cache_initializations=0)

    def reject(point, reason, *, stop=False):
        if diagnostics is not None:
            reject_trial(diagnostics, reason)
        return TrialDecision(point, reason, stop=stop)

    def accept(point, reason):
        if diagnostics is not None:
            diagnostics['acceptance_reason'] = reason
        return TrialDecision(point, reason, accepted=True)

    def screen_rejects(point, index):
        nonlocal residual_norms, residual_ray
        step, trial_y, trial_s = point.step, point.y, point.slack
        with np.errstate(over='ignore', invalid='ignore'):
            complementarity_norm = maxnorm(trial_y*trial_s)
        if diagnostics is not None:
            diagnostics['residual_screen_checks'] += 1
        if (np.isfinite(complementarity_norm)
                and complementarity_norm >= (1-settings.armijo_param*step)*kkt_res
                and complementarity_norm >= settings.tol):
            if diagnostics is not None:
                diagnostics['complementarity_rejections'] += 1
            return True
        # Pay for direction products only after the first trial. Uncertain
        # bounds fall through to fresh products, and fallback bypasses screens.
        if settings.qp_residual_screen == 'bounded' and index > 0:
            if residual_ray is None:
                if residual_norms is None:
                    residual_norms = QPResidualNorms(model.C, model.E)
                residual_ray = QPResidualRay(model, residual_norms,
                    rx=rx, rp=rp, req=req, x=x, y=y, s=s, nu=nu,
                    dx=dx, dy=dy, ds=ds, dnu=dnu, Qx=Qx, Qdx=Qdx)
                if diagnostics is not None:
                    diagnostics['affine_cache_initializations'] += 1
            report = residual_ray.screen(step, trial_y, trial_s,
                residual_limit=(1-settings.armijo_param*step)*kkt_res,
                tolerance=settings.tol)
            if diagnostics is not None:
                diagnostics['affine_screen_checks'] += 1
            if report.can_reject:
                if diagnostics is not None:
                    diagnostics['affine_rejections'] += 1
                return True
        return False

    def attempt(step, index):
        # Legacy diagnostics count an invalid step; the strict GLM policy
        # checks it before counting. The controller leaves this to each policy.
        if diagnostics is not None:
            diagnostics['trial_count'] += 1
        if not np.isfinite(step) or step <= 0:
            return reject(None, 'invalid_step', stop=True)
        point = ray.evaluate(step)
        if not point.interior:
            return reject(point, 'nonfinite_or_nonpositive_trial')
        if not np.isfinite(point.merit):
            return reject(point, 'nonfinite_merit')
        # Preserve the legacy comparison even for a nonfinite slope/bound.
        if point.merit < merit0 + settings.armijo_param*step*slope0 + settings.armijo_additive_eps:
            return accept(point, 'armijo')
        if diagnostics is not None:
            diagnostics['armijo_failure_count'] += 1
        newton_fallback = (settings.newton_fallback_policy != 'off'
                           and step < settings.let_newton_cook*tmax)
        if diagnostics is not None and model.QP_mode:
            diagnostics['fallback_eligible_count'] += int(newton_fallback)
        if not (residual_acceptance or newton_fallback):
            return reject(point, 'acceptance_tests_failed')
        if (model.QP_mode and settings.qp_residual_screen != 'off'
                and not newton_fallback and screen_rejects(point, index)):
            return reject(point, 'acceptance_tests_failed')

        if diagnostics is not None:
            diagnostics['residual_check_count'] += 1
        point.glm_gradient = 0. if model.dummy_A else ray.transpose@model.f.d1f(point.predictor)
        ls_gradf = point.glm_gradient + Qx + step*Qdx
        point.products = model._constraint_products(point.x, point.y, nu+step*dnu)
        ls_blocks = model._kkt_residuals(ls_gradf, point.y, point.slack, point.products)
        ls_kkt_res = maxnorm(np.hstack(ls_blocks))
        if residual_acceptance and np.isfinite(ls_kkt_res) and (
                ls_kkt_res < (1-settings.armijo_param*step)*kkt_res
                or ls_kkt_res < settings.tol):
            return accept(point, 'residual')
        if np.isfinite(ls_kkt_res) and newton_fallback:
            if settings.newton_fallback_policy == 'progress':
                ls_rx, ls_rp, ls_rc, ls_req = ls_blocks
                trial_perturbed_kkt = maxnorm(np.hstack([ls_rx, ls_rp, ls_rc-mu, ls_req]))
                contraction_limit = (1-settings.armijo_param*step)*perturbed_kkt_before
                progress = bool(np.isfinite(trial_perturbed_kkt) and (
                    trial_perturbed_kkt < contraction_limit or trial_perturbed_kkt < settings.tol))
                if diagnostics is not None:
                    record_fallback_progress(diagnostics, step, perturbed_kkt_before,
                                             trial_perturbed_kkt, contraction_limit, progress)
                if not progress:
                    return reject(point, 'fallback_no_progress')
            return accept(point, 'newton_fallback')
        return reject(point, 'acceptance_tests_failed')

    def propose(step, decision):
        nonlocal qp_proposer
        if not qp_scalar:
            return .9*step
        # The controller calls this only after all gates reject and another
        # trial remains. Accepted full steps do no scalar-model work.
        if qp_proposer is None:
            qp_proposer = proposer_factory(s, ds, mu, xQdx, dxQdx, np.dot(dx, model.b),
                                           merit0, slope0)
        proposed, kind = qp_proposer.propose(step, decision.point.slack, decision.point.merit)
        if diagnostics is not None:
            threshold = settings.let_newton_cook*tmax
            diagnostics['qp_scalar_counters'] = qp_proposer.counters
            diagnostics['qp_scalar_proposals'].append(dict(
                from_step=float(step), next_step=float(proposed), kind=kind,
                crosses_fallback=bool(settings.newton_fallback_policy != 'off'
                    and step >= threshold and proposed < threshold)))
        return proposed

    result = search_trials(tmax, settings.max_linesearch_steps, attempt, propose)
    return result.point, residual_norms
