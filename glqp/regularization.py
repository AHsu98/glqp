"""Iteration decay, frozen before each outer direction is assembled.

Physical coefficients belong to the requested Newton equations. The separate
tau is only the initial factorization shift; numerical retries may increase it.
Neither schedule changes within a line search or a predictor/corrector step.
"""
from dataclasses import dataclass
import math
import numbers

import numpy as np


def validate_decay_settings(settings):
    for name in ('prox_decay_iterations', 'tau_decay_iterations'):
        value = getattr(settings, name)
        if value is not None and (
                isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be None or finite and strictly positive')


def iteration_decay(index, timescale):
    """Return T/(T+i), using normalization if the denominator overflows.

    ``index`` is the number of completed accepted outer iterations, starting
    at zero. A float timescale is supported; no clock is stored in settings.
    A sufficiently small positive factor may underflow to zero in float64.
    """
    if timescale is None or index == 0:
        return 1.
    scale = float(timescale)
    denominator = scale + index
    if math.isfinite(denominator):
        return scale/denominator
    # Normalize only when the literal denominator overflows. The ordinary
    # path follows the declared formula's arithmetic exactly.
    if index <= scale:
        return 1. / (1. + index/scale)
    ratio = scale/index
    return ratio / (1. + ratio)


@dataclass(frozen=True)
class DecayedRegularization:
    index: int
    prox_factor: float
    tau_factor: float
    rho_before_decay: float
    delta_before_decay: float
    rho: float
    delta: float
    tau_configured: float
    tau: float

    def history_fields(self, rho_requested, delta_requested):
        """Preserve both the schedule and any subsequent centered continuation."""
        return dict(reg_decay_index=self.index, reg_prox_factor=self.prox_factor,
            reg_tau_factor=self.tau_factor,
            reg_rho_before_decay=float(self.rho_before_decay),
            reg_delta_before_decay=float(self.delta_before_decay),
            reg_rho_after_decay=float(self.rho), reg_delta_after_decay=float(self.delta),
            reg_rho_requested=float(rho_requested), reg_delta_requested=float(delta_requested),
            reg_tau_configured=float(self.tau_configured), reg_tau_requested=float(self.tau))


def decay_regularization(settings, index, rho, delta):
    """Scale already residual-limited coefficients before progress continuation."""
    prox_factor = iteration_decay(index, settings.prox_decay_iterations)
    tau_factor = iteration_decay(index, settings.tau_decay_iterations)
    return DecayedRegularization(index, prox_factor, tau_factor, rho, delta,
        rho if settings.prox_decay_iterations is None else rho*prox_factor,
        delta if settings.prox_decay_iterations is None else delta*prox_factor,
        settings.tau_reg,
        settings.tau_reg if settings.tau_decay_iterations is None else settings.tau_reg*tau_factor)
