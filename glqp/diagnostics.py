"""Opt-in observations of the outer algorithm; no solver decisions live here.

Newton block defects refer to the original, unregularized equations at the
same current barrier target. They are different from the residual of the
regularized, scaled matrix used by the linear solver. All expensive products
in this module run only for explicit diagnostics or callback observations.
"""
from dataclasses import dataclass

import numpy as np


def _inf(value):
    return float(np.max(np.abs(value))) if len(value) else 0.


def _block_norms(blocks):
    return {name: _inf(value) for name, value in blocks.items()}


@dataclass(frozen=True)
class IterationSnapshot:
    """Synchronous callback state; arrays are read-only borrowed views.

    The callback must not mutate the arrays (including through their bases),
    the model, or its settings. Views are valid only during the callback;
    callers retaining a state must copy the arrays. The return value is ignored.
    Exceptions propagate. Callback work counts toward the solver's time limit.
    ``y`` is the inequality multiplier and ``nu`` the equality multiplier;
    artificial inequality entries remain present for an unconstrained model.
    """
    iteration: int
    stage: str
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    nu: np.ndarray
    objective: float
    barrier_target: float
    kkt_residual: float
    relative_complementarity: float
    native_converged: bool
    native_near_converged: bool


def observe_iteration(callback, model, iteration, stage, x, y, s, nu,
                      objective, barrier_target, kkt_residual, point=None):
    """Publish a state without feeding any callback value into the algorithm."""
    def readonly_view(value):
        result = value.view()
        result.flags.writeable = False
        return result

    relative = (model._relative_complementarity(x, y, objective) if point is None
                else point['true_slack_complementarity_scaled'])
    tol = model.settings.tol
    callback(IterationSnapshot(
        iteration=iteration, stage=stage,
        x=readonly_view(x), y=readonly_view(y), s=readonly_view(s), nu=readonly_view(nu),
        objective=float(objective), barrier_target=float(barrier_target),
        kkt_residual=float(kkt_residual), relative_complementarity=float(relative),
        native_converged=bool(kkt_residual <= tol and relative <= tol),
        native_near_converged=bool(kkt_residual <= 100*tol and relative <= 100*tol)))


def point_diagnostics(model, x, y, s, objective, barrier_target, kkt_residual,
                      rx, rp, rc, req):
    """Scalar state, including complementarity against the true original slack.

    ``rc`` here is unperturbed y*s. The dummy row is excluded from original
    inequality diagnostics, exactly as in the native aggregate stopping test.
    """
    products = y*s
    mu_bar = np.dot(s, y)/model.k if model.k else None
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        xi = np.min(products)/mu_bar if model.k else None
        ratio = np.max(y/s) if len(y) else None
        if model.dummy_ineq:
            original_min, mismatch, original_comp = None, 0., 0.
        else:
            original_slack = model.c - model.C@x
            original_min = float(np.min(original_slack))
            mismatch = _inf(s - original_slack)
            original_comp = float(np.sum(np.abs(y*original_slack)))
        relative = (original_comp / max(1., abs(objective))
                    if np.isfinite(objective) else np.inf)
    tol = model.settings.tol
    return dict(objective=float(objective), barrier_target=float(barrier_target),
                kkt_residual=float(kkt_residual),
                stationarity_residual_inf=_inf(rx), primal_residual_inf=_inf(rp),
                equality_residual_inf=_inf(req), stored_complementarity_inf=_inf(rc),
                stored_slack_complementarity_l1=float(np.sum(np.abs(products))),
                true_slack_complementarity_l1=original_comp,
                true_slack_complementarity_scaled=float(relative),
                original_slack_min=original_min, stored_slack_inconsistency_inf=mismatch,
                mu_bar=None if mu_bar is None else float(mu_bar),
                xi=None if xi is None else float(xi),
                min_s=float(np.min(s)) if len(s) else None,
                min_y=float(np.min(y)) if len(y) else None,
                max_y_over_s=None if ratio is None else float(ratio),
                native_converged=bool(kkt_residual <= tol and relative <= tol),
                native_near_converged=bool(kkt_residual <= 100*tol and relative <= 100*tol))


def original_newton_blocks(model, H, x_direction, s_direction, y_direction,
                           nu_direction, y, s, rx, rp, rc, req):
    """Unregularized Newton LHS minus RHS, at fixed current target mu.

    ``rc`` is the perturbed residual y*s-mu. Even an exact requested-system
    solution has defects (-rho*dx, rho*dy, rho*dnu, -rho*ds).
    """
    return dict(
        stationarity=H@x_direction + model.C.T@y_direction + model.E.T@nu_direction + rx,
        primal=model.C@x_direction + s_direction + rp,
        equality=model.E@x_direction + req,
        complementarity=y*s_direction + s*y_direction + rc)


def direction_diagnostics(model, H, dx, ds, dy, dnu, old_ds, old_dy,
                          y, s, rx, rp, rc, req, rho, threshold, *,
                          complementarity_correction=None, dual_reg=None,
                          complementarity_reg=None, center_offsets=None):
    """Measure before/after editing and the defect induced by the edits alone."""
    before = original_newton_blocks(model, H, dx, old_ds, old_dy, dnu,
                                     y, s, rx, rp, rc, req)
    after = original_newton_blocks(model, H, dx, ds, dy, dnu,
                                    y, s, rx, rp, rc, req)
    delta = rho if dual_reg is None else dual_reg
    slack_shift = rho if complementarity_reg is None else complementarity_reg
    regularization = dict(stationarity=-rho*dx, primal=delta*old_dy,
                          equality=delta*dnu, complementarity=-slack_shift*old_ds)
    requested = {key: before[key]-regularization[key] for key in before}
    if center_offsets is not None:
        for key, value in center_offsets.items():
            requested[key] += value
    # The combined corrector intentionally changes the original Newton RHS.
    # Keep that change visible in original defects, but exclude it from the
    # residual of the system we actually requested from the linear solver.
    if complementarity_correction is not None:
        requested['complementarity'] += complementarity_correction
    delta_ds, delta_dy = ds-old_ds, dy-old_dy
    induced = dict(stationarity=model.C.T@delta_dy, primal=delta_ds,
                   equality=np.zeros_like(req), complementarity=y*delta_ds+s*delta_dy)
    return dict(
        before_norm_inf=_block_norms(dict(dx=dx, ds=old_ds, dy=old_dy, dnu=dnu)),
        after_norm_inf=_block_norms(dict(dx=dx, ds=ds, dy=dy, dnu=dnu)),
        threshold=float(threshold), small_s_count=int(np.count_nonzero(s <= threshold)),
        small_y_count=int(np.count_nonzero(y <= threshold)),
        edited_ds_count=int(np.count_nonzero(delta_ds)),
        edited_dy_count=int(np.count_nonzero(delta_dy)),
        delta_ds_inf=_inf(delta_ds), delta_dy_inf=_inf(delta_dy),
        original_newton_before_inf=_block_norms(before),
        original_newton_after_inf=_block_norms(after),
        regularization_defect_inf=_block_norms(regularization),
        **({} if center_offsets is None else dict(
            center_residual_offset_inf=_block_norms(center_offsets))),
        requested_regularized_before_inf=_block_norms(requested),
        **({} if complementarity_correction is None else dict(
            corrector_complementarity_defect_inf=_inf(complementarity_correction))),
        edit_induced_defect_inf=_block_norms(induced),
        observed_defect_change_inf=_block_norms(
            {key: after[key]-before[key] for key in before}))


def start_iteration_diagnostics(iteration, rho):
    return dict(iteration=iteration, status='started', rho=float(rho),
                barrier=None, direction=None, line_search=None, repair=None,
                iterate=None, linear_solve=None)


def policy_diagnostics(settings):
    names = ('initial_mu_policy', 'initial_mu_scale', 'barrier_update_guard',
             'barrier_reduction_limit', 'barrier_kkt_floor', 'slack_repair',
             'repair_growth_limit', 'newton_fallback_policy', 'step_strategy',
             'proximal_policy', 'prox_dual_reg', 'proximal_center_reduction',
             'proximal_inner_tolerance', 'proximal_corrector_limit', 'proximal_reg_update',
             'glm_linesearch')
    return {name: getattr(settings, name) for name in names}


def start_line_search_diagnostics(tmax, merit_slope, *, fallback_policy='finite',
                                  perturbed_kkt_before=None):
    return dict(acceptance_reason=None, trial_count=0, rejection_count=0,
                rejection_reasons={}, armijo_failure_count=0, residual_check_count=0,
                tmax=float(tmax), accepted_step=None, accepted_fraction=None,
                merit_slope=float(merit_slope), fallback_policy=fallback_policy,
                fallback_perturbed_kkt_before=(None if perturbed_kkt_before is None
                                               else float(perturbed_kkt_before)),
                fallback_progress_check_count=0, fallback_progress_rejection_count=0,
                fallback_progress_checks=[], last_trial_perturbed_kkt=None)


def record_fallback_progress(diagnostics, step, old_norm, trial_norm, limit, accepted):
    diagnostics['fallback_progress_check_count'] += 1
    diagnostics['fallback_progress_rejection_count'] += int(not accepted)
    diagnostics['last_trial_perturbed_kkt'] = float(trial_norm)
    diagnostics['fallback_progress_checks'].append(dict(
        step=float(step), previous_perturbed_kkt=float(old_norm),
        trial_perturbed_kkt=float(trial_norm), contraction_limit=float(limit),
        accepted=bool(accepted)))


def record_scalar_search(diagnostics, result):
    """Map the scalar controller's observations to common search diagnostics."""
    counts = result.counters
    diagnostics.update(
        algorithm='scalar', scalar_status=result.status,
        scalar_counters=counts, scalar_history=result.history,
        trial_count=counts['trial_count'],
        armijo_failure_count=counts['armijo_failure_count'],
        acceptance_reason='armijo' if result.trial is not None else None)
    for reason, key in (
            ('nonfinite_or_nonpositive_trial', 'invalid_trial_count'),
            ('nonfinite_merit', 'nonfinite_count'),
            ('acceptance_tests_failed', 'armijo_failure_count')):
        if counts[key]:
            diagnostics['rejection_reasons'][reason] = counts[key]
    diagnostics['rejection_count'] = sum(diagnostics['rejection_reasons'].values())


def start_repair_diagnostics(settings):
    return dict(policy=settings.slack_repair, proposed=False, proposed_count=0,
                applied=False, rejected=False, decision='not_triggered',
                rejection_reason=None, rejection_reasons=(),
                candidate_finite_positive=None, candidate_primal_inf=None,
                candidate_kkt_inf=None, primal_limit=None, kkt_limit=None)


def reject_trial(diagnostics, reason):
    diagnostics['rejection_count'] += 1
    reasons = diagnostics['rejection_reasons']
    reasons[reason] = reasons.get(reason, 0) + 1


def slack_repair_diagnostics(triggered, old_s, s, old_rp, rp, old_rc, rc, pre, post):
    delta = s-old_s
    return dict(triggered=bool(triggered), count=int(np.count_nonzero(delta)),
                slack_delta_inf=_inf(delta), slack_delta_l1=float(np.sum(np.abs(delta))),
                primal_residual_delta_inf=_inf(rp-old_rp),
                complementarity_delta_inf=_inf(rc-old_rc), pre=pre, post=post)


def history_diagnostics(iteration):
    """Compact scalar columns; full nested observations stay on the result."""
    barrier, direction = iteration['barrier'], iteration['direction']
    search, repair, point = iteration['line_search'], iteration['repair'], iteration['iterate']
    return dict(diag_mu_before=barrier['previous_target'], diag_mu_raw=barrier['raw_proposal'],
                diag_mu_bar=barrier['mu_bar'], diag_xi=barrier['xi'],
                diag_mu_guard_floor=barrier['guard_floor'], diag_mu_min_floor=barrier['min_floor'],
                diag_mu_kkt_floor=barrier['kkt_floor'], diag_mu_active_reasons=barrier['active_reasons'],
                diag_acceptance=search['acceptance_reason'], diag_trials=search['trial_count'],
                diag_rejections=search['rejection_count'], diag_tmax=search['tmax'],
                diag_accepted_fraction=search['accepted_fraction'], diag_rho=iteration['rho'],
                diag_direction_edits=direction['edited_ds_count']+direction['edited_dy_count'],
                diag_repair_count=repair['count'], diag_kkt_pre_repair=barrier['pre_repair_kkt'],
                diag_repair_proposed=repair['proposed'], diag_repair_applied=repair['applied'],
                diag_repair_rejected=repair['rejected'],
                diag_fallback_progress_checks=search['fallback_progress_check_count'],
                diag_true_slack_comp_scaled=point['true_slack_complementarity_scaled'])


def proximal_history_fields(update, search):
    """Compact centered-policy history from already computed decisions."""
    return dict(
        prox_generation=update['generation'],
        prox_primal_center_updated=update['primal_center_updated'],
        prox_dual_centers_updated=update['dual_centers_updated'],
        prox_inner_converged=update['inner_converged'],
        prox_fixed_target_residual=update['fixed_target_proximal_residual_inf'],
        prox_merit_before=search['merit_before'],
        prox_merit_after=search['merit_after'],
        prox_line_search_trials=search['trial_count'])
