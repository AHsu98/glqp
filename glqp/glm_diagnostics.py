"""Opt-in checks against the original GLM design, outside solver decisions.

The caller supplies the carried predictor and the *objective* Hessian, before
physical proximal terms and factorization shifts. These float64 comparisons
diagnose discrepancies; neither action is an exact-arithmetic reference.
No result from this module should participate in termination or line search.
"""

import numpy as np


def _inf(value):
    return float(np.max(np.abs(value), initial=0.))


def glm_predictor_diagnostics(model, x, z):
    """Compare carried ``z`` with a fresh ``A @ x`` at this same iterate.

    Scalar derivatives are evaluated at both predictors. The stationarity
    difference uses the original design; quadratic and constraint terms cancel.
    This function borrows its arguments without modifying them.
    """
    fresh_z = model.A @ x
    carried_gradient = model.f.d1f(z)
    fresh_gradient = model.f.d1f(fresh_z)
    gradient_difference = carried_gradient - fresh_gradient
    drift = _inf(z - fresh_z)
    return dict(
        carried_minus_fresh_inf=drift,
        relative_drift_inf=drift / max(1., _inf(fresh_z)),
        scalar_gradient_drift_inf=_inf(gradient_difference),
        stationarity_drift_inf=_inf(model.A.T @ gradient_difference),
        carried_minus_fresh_loss=float(model.f(z) - model.f(fresh_z)),
        fresh_predictor_norm_inf=_inf(fresh_z))


def glm_hessian_diagnostics(model, H, z, dx):
    """Compare ``H @ dx`` with ``Q @ dx + A.T @ (f''(z) * (A @ dx))``.

    Curvature is evaluated at the carried predictor used to assemble ``H``;
    predictor drift is a separate observation. Both actions exclude physical
    proximal regularization, numerical factor shifts, and constraint blocks.
    Clipping slack/multiplier directions does not change ``dx`` or this action.
    The existing direction diagnostics measure those edits and requested-system
    residuals separately. A small relative discrepancy can still destroy a weak
    mode, so the absolute actions and Rayleigh products are also retained.
    """
    curvature = np.asarray(model.f.d2f(z))
    assembled = H @ dx
    original = model.Q @ dx + model.A.T @ (curvature * (model.A @ dx))
    difference = _inf(assembled - original)
    positive = curvature[(curvature > 0.) & np.isfinite(curvature)]
    return dict(
        operator='original_objective_hessian',
        curvature_predictor='carried',
        physical_proximal_terms_included=False,
        numerical_factor_shift_included=False,
        direction_norm_inf=_inf(dx),
        assembled_action_inf=_inf(assembled),
        original_operator_action_inf=_inf(original),
        assembled_minus_original_inf=difference,
        relative_to_original_inf=difference / max(_inf(original), np.finfo(float).tiny),
        assembled_rayleigh=float(dx @ assembled),
        original_operator_rayleigh=float(dx @ original),
        curvature_zero_count=int(np.count_nonzero(curvature == 0.)),
        curvature_negative_count=int(np.count_nonzero(curvature < 0.)),
        curvature_nonfinite_count=int(np.count_nonzero(~np.isfinite(curvature))),
        curvature_minimum_positive=float(np.min(positive)) if len(positive) else None,
        curvature_max_abs=_inf(curvature))
