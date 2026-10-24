"""Experimental complementarity predictor/corrector at a frozen Newton matrix.

The affine point predicts centrality; it is never installed as an iterate.
The affine and combined RHSs, and optional centering rescue, use the same
current GLM Hessian and outer regularization. The returned direction is a
full step, not a correction to add again. Globalization remains the caller's
responsibility.
"""
from dataclasses import dataclass
import numbers

import numpy as np

from .diagnostics import direction_diagnostics
from .kkt import KKTWorkspace
from .linear import LinearSolveError, VerifiedLinearSystem
from .outer_policy import DIRECTION_CLIP_THRESHOLD, clip_tiny_directions
from .util import get_step_size


_SUFFICIENT_DESCENT_FRACTION = .1


def validate_step_strategy(settings):
    if (not isinstance(settings.step_strategy, str)
            or settings.step_strategy not in ('newton', 'predictor_corrector')):
        raise ValueError("step_strategy must be 'newton' or 'predictor_corrector'")


def _bounded_correction(correction, residual, limit):
    """Bound the corrector as an inexact-Newton residual perturbation.

    If ``||q|| <= eta ||R||`` and ``J d = -R-q``, then
    ``R.T J d <= -(1-eta) ||R||**2``. Separate max-norm scaling avoids
    squaring large values merely to choose the correction weight.
    """
    residual_max = float(np.max(np.abs(residual), initial=0.))
    correction_max = float(np.max(np.abs(correction), initial=0.))
    residual_unit_norm = (float(np.linalg.norm(residual/residual_max))
                         if residual_max else 0.)
    correction_unit_norm = (float(np.linalg.norm(correction/correction_max))
                           if correction_max else 0.)
    weight = 1.
    if limit is not None and correction_max:
        if not limit or not residual_max:
            weight = 0.
        else:
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                weight = float(min(1., limit*(residual_max/correction_max)
                                   *(residual_unit_norm/correction_unit_norm)))
    applied = correction*weight
    scale = max(1., residual_max)
    normalized = residual/scale
    norm_squared = float(np.dot(normalized, normalized))
    with np.errstate(over='ignore', invalid='ignore'):
        raw_norm = float(np.float64(correction_max)*correction_unit_norm)
        applied_norm = float(np.float64(correction_max*weight)*correction_unit_norm)
        residual_norm = float(np.float64(residual_max)*residual_unit_norm)
    decision = dict(
        limit=limit,
        weight=weight,
        raw_norm=raw_norm,
        applied_norm=applied_norm,
        residual_norm=residual_norm,
        residual_scale=scale,
        scaled_residual_norm_squared=norm_squared,
        sufficient_descent_limit=-_SUFFICIENT_DESCENT_FRACTION*norm_squared,
    )
    return applied, decision


@dataclass
class PredictorCorrectorResult:
    """Returned direction plus accounting for every attempted RHS solve.

    ``combined_info`` describes the returned direction, including a centering
    rescue when selected. ``extra_refinements`` retains work spent on the
    discarded combined direction. The full diagnostic record is ``decision``.
    """

    dx: np.ndarray
    ds: np.ndarray
    dy: np.ndarray
    dnu: np.ndarray
    factorization: object
    affine_info: object
    combined_info: object
    target: float
    correction: np.ndarray
    decision: dict
    extra_refinements: int = 0

    @property
    def refinements(self):
        return (self.affine_info.refinements + self.combined_info.refinements
                + self.extra_refinements)

    def history_fields(self):
        """Stable compact history columns; detailed rescue data stays in decision."""
        return dict(
            pc_affine_step=self.decision['affine_step'],
            pc_sigma=self.decision['sigma'],
            pc_target=self.target,
            pc_affine_refinements=self.affine_info.refinements,
            pc_combined_refinements=self.combined_info.refinements,
            pc_affine_status=self.affine_info.status,
            pc_combined_status=self.combined_info.status,
            pc_factorization_attempts=self.decision['factorization_attempts'],
        )


def compute_predictor_corrector(model, x, y, s, H, rx, rp, req, kkt_res, *,
                                prox_reg, solver=None, target_override=None,
                                dual_reg=None, complementarity_reg=None,
                                residual_safeguard=False, center_offsets=None):
    """Reuse a factor for the affine, combined, and optional centering RHSs.

    Each RHS gets fresh requested-system verification. Its reported accuracy
    status is retained; verification does not imply every result is accurate.

    ``target_override`` honors explicit solve(mu0=...) on the first iteration.
    Otherwise the predictor chooses a target before the combined solve and
    line search. The previous-target guard and post-step centrality update do
    not apply to this strategy. No GLM third derivative is assumed.

    Explicit block shifts follow ``KKTWorkspace.assemble``. With standard
    complementarity, ``residual_safeguard`` bounds the corrector perturbation
    and replaces an insufficient-descent combined direction with a same-target
    centering direction, using a third RHS and the existing factorization.
    The safeguard tests the direction after the caller's tiny-component
    clipping without applying those edits here.
    """
    if model.dummy_ineq or not len(s):
        raise ValueError('Predictor-corrector requires real inequalities')
    if target_override is not None and (
            isinstance(target_override, (bool, np.bool_))
            or not isinstance(target_override, numbers.Real)
            or not np.isfinite(target_override) or target_override < 0):
        raise ValueError('Predictor-corrector mu0 must be finite and nonnegative')
    settings = model.settings
    rho_y = prox_reg if dual_reg is None else dual_reg
    rho_s = prox_reg if complementarity_reg is None else complementarity_reg
    if residual_safeguard and rho_s != 0:
        raise ValueError('Residual safeguard requires standard complementarity')
    products = s*y
    mean = float(np.dot(s, y)/len(s))
    if not np.isfinite(mean) or mean <= 0:
        raise LinearSolveError('Nonfinite or nonpositive predictor complementarity')
    if model._kkt_workspace is None:
        model._kkt_workspace = KKTWorkspace(H, model.C, model.E)
    workspace = model._kkt_workspace
    block_options = {}
    if dual_reg is not None:
        block_options['dual_reg'] = dual_reg
    if complementarity_reg is not None:
        block_options['complementarity_reg'] = complementarity_reg
    G, rhs, w = workspace.assemble(
        H, y, s, rx, rp, products, req, prox_reg, **block_options)

    def recover(solution):
        dx = solution[:model.n]
        dy = w*solution[model.n:model.n+model.k]
        ds = -rp - model.C@dx + rho_y*dy
        return dx, ds, dy, solution[model.n+model.k:]

    def clipped_residual_slope(direction, rc):
        # Only the centered safeguard needs this residual-merit calculation.
        from .proximal import residual_slope

        dx, ds, dy, dnu = direction
        clipped_ds, clipped_dy = ds.copy(), dy.copy()
        clip_tiny_directions(s, y, clipped_ds, clipped_dy)
        return residual_slope(model, H, dx, clipped_ds, clipped_dy, dnu,
                              y, s, rx, rp, rc, req, prox_reg, rho_y)

    extra_refinements = 0
    extra_attempts = 0
    rhs_solves = 2
    safeguard_decision = {}

    with VerifiedLinearSystem(
        G, model.reg_shift, settings.tau_reg, solver,
        target_atol=settings.linear_atol_factor*settings.tol,
        target_rtol=settings.linear_rtol, max_solve_attempts=12,
        max_refinement_steps=settings.max_iterative_refinement,
        residual_blocks={'stationarity': slice(0, model.n),
                         'scaled_inequality': slice(model.n, model.n+model.k),
                         'equality': slice(model.n+model.k, None)},
        backend=settings.linear_solver_backend,
        backend_options=settings.linear_solver_options,
        workspace=model._shift_workspace,
    ) as system:
        affine = system.solve_verified(rhs)
        ax, a_s, a_y, a_nu = recover(affine.solution)
        # The raw verified affine direction predicts attainable centrality.
        # Existing tiny-component edits apply only to the eventual combined step.
        alpha = float(get_step_size(s, a_s, y, a_y, frac=1.))
        with np.errstate(over='ignore', invalid='ignore'):
            affine_mean = float(np.dot(s+alpha*a_s, y+alpha*a_y)/len(s))
            sigma = float(np.clip(affine_mean/mean, 0., 1.)**3)
            correction = a_s*a_y
            raw_target = sigma*mean
            kkt_floor = settings.barrier_kkt_floor*kkt_res
            target = (float(target_override) if target_override is not None else
                      float(max(raw_target, settings.min_mu, kkt_floor)))
        if (not np.isfinite(alpha) or not 0 <= alpha <= 1
                or not np.isfinite(affine_mean) or not np.isfinite(target)
                or not np.all(np.isfinite(correction))):
            raise LinearSolveError('Nonfinite predictor-corrector target or correction')
        if residual_safeguard:
            centered_rc = products - target
            correction, bound = _bounded_correction(
                correction, np.concatenate((rx, rp, centered_rc, req)),
                settings.proximal_corrector_limit)
        # Positive residual convention: KKT RHS contains minus this residual.
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            combined_rc = products + correction - target
            rhs = workspace.assemble_rhs(rx, rp, combined_rc, req)
        if not np.all(np.isfinite(combined_rc)) or not np.all(np.isfinite(rhs)):
            raise LinearSolveError('Nonfinite predictor-corrector right-hand side')
        combined = system.solve_verified(rhs)
        dx, ds, dy, dnu = recover(combined.solution)
        if residual_safeguard:
            slope = clipped_residual_slope((dx, ds, dy, dnu), centered_rc)
            rescue = dict(applied=False, corrector_residual_slope=float(slope))
            safeguard_decision = dict(
                direction_choice='predictor_corrector',
                rescue=rescue,
                residual_merit_slope=float(slope),
                corrector_bound=bound,
            )
            if (not np.isfinite(slope) or slope >= 0
                    or slope > bound['sufficient_descent_limit']):
                # The corrector cross term is not part of the nonlinear
                # residual map. A verified Newton step for that same map and
                # target provides a coherent alternative without refactoring.
                discarded = combined
                rhs = workspace.assemble_rhs(rx, rp, centered_rc, req)
                if not np.all(np.isfinite(rhs)):
                    raise LinearSolveError('Nonfinite centering right-hand side')
                combined = system.solve_verified(rhs)
                dx, ds, dy, dnu = recover(combined.solution)
                extra_refinements = discarded.info.refinements
                extra_attempts = discarded.info.attempts
                rhs_solves += 1
                centering_slope = clipped_residual_slope((dx, ds, dy, dnu), centered_rc)
                if not np.isfinite(slope):
                    rescue_reason = 'nonfinite_slope'
                elif slope >= 0:
                    rescue_reason = 'non_descent'
                else:
                    rescue_reason = 'insufficient_descent'
                rescue.update(
                    applied=True,
                    reason=rescue_reason,
                    centering_residual_slope=float(centering_slope),
                    discarded_corrector_linear_solve=discarded.info.history_fields(),
                    discarded_corrector_refinements=discarded.info.refinements,
                    discarded_correction_inf=float(np.max(np.abs(correction), initial=0.)))
                safeguard_decision['direction_choice'] = 'centering_rescue'
                safeguard_decision['residual_merit_slope'] = float(centering_slope)
                correction = np.zeros_like(correction)
        factorization = system.factorization

    candidates = dict(raw_proposal=raw_target, min_floor=settings.min_mu,
                      kkt_floor=kkt_floor)
    decision = dict(
        mu_bar=mean,
        affine_mu=affine_mean,
        affine_step=alpha,
        sigma=sigma,
        **candidates,
        selection_kkt=float(kkt_res),
        target=target,
        target_source='explicit_initial' if target_override is not None else 'predictor',
        active_reasons=(('explicit_initial',) if target_override is not None else
                        tuple(key for key,value in candidates.items() if value == target)),
        rho=float(prox_reg),
        correction_inf=float(np.max(np.abs(correction), initial=0.)),
        affine_linear_solve=affine.info.history_fields(),
        combined_linear_solve=combined.info.history_fields(),
        affine_refinements=affine.info.refinements,
        combined_refinements=combined.info.refinements,
        factorization_attempts=affine.info.attempts+combined.info.attempts+extra_attempts,
        verified_rhs_solves=rhs_solves,
        selection_timing='before_direction',
        matrix_policy='fixed_H_s_y_rho',
        correction_policy='complementarity_only',
    )
    decision.update(block_options)
    decision.update(safeguard_decision)
    if settings.iteration_diagnostics:
        diagnostic_rx, diagnostic_rp, diagnostic_req = rx, rp, req
        diagnostic_options = block_options.copy()
        if center_offsets is not None:
            diagnostic_rx = rx - center_offsets['stationarity']
            diagnostic_rp = rp - center_offsets['primal']
            diagnostic_req = req - center_offsets['equality']
            diagnostic_options['center_offsets'] = center_offsets
        decision['affine_direction'] = direction_diagnostics(
            model, H, ax, a_s, a_y, a_nu, a_s, a_y,
            y, s, diagnostic_rx, diagnostic_rp, products, diagnostic_req,
            prox_reg, DIRECTION_CLIP_THRESHOLD, **diagnostic_options)
    return PredictorCorrectorResult(dx, ds, dy, dnu, factorization,
        affine.info, combined.info, target, correction, decision, extra_refinements)


def describe_held_target(target, mu_bar, xi, pre_repair_kkt, settings):
    """Post-step record: predictor choice is held until the next predictor.

    The actual pre-direction candidates and floor live in the iteration's
    ``predictor_corrector`` decision. No post-step floor is applied here.
    """
    return dict(previous_target=float(target), raw_proposal=float(target),
        target=float(target), mu_bar=float(mu_bar), xi=float(xi),
        guard_reference='predictor_corrector_held', guard_floor=None,
        min_floor=float(settings.min_mu), kkt_floor=None,
        kkt_floor_coefficient=float(settings.barrier_kkt_floor),
        pre_repair_kkt=float(pre_repair_kkt),
        active_reasons=('held_predictor_corrector_target',), selection_timing='held_after_step')
