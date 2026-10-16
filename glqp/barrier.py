"""Barrier target selection, kept separate from centrality and slack repair."""
import numbers

import numpy as np


def validate_barrier_settings(settings):
    """Validate again at solve entry because experiment settings are mutable."""
    if settings.barrier_update_guard not in ('previous', 'legacy'):
        raise ValueError("barrier_update_guard must be 'previous' or 'legacy'")
    limit = settings.barrier_reduction_limit
    if (isinstance(limit, (bool, np.bool_)) or not isinstance(limit, numbers.Real)
            or not np.isfinite(limit) or not 0 <= limit <= 1):
        raise ValueError('barrier_reduction_limit must be finite and between 0 and 1')
    if not isinstance(settings.iteration_diagnostics, bool):
        raise ValueError('iteration_diagnostics must be a bool')


def select_barrier_target(previous_target, raw_proposal, pre_repair_kkt, settings):
    """Limit reduction relative to the previous target, or reproduce legacy.

    The legacy guard uses the proposal itself. Keep the old max operations
    and their evaluation order so the default legacy limit reproduces the
    accepted implementation's floating-point arithmetic exactly.
    """
    reference = (previous_target if settings.barrier_update_guard == 'previous'
                 else raw_proposal)
    lower_bound = np.max([reference * settings.barrier_reduction_limit,
                          settings.min_mu, 0.01 * pre_repair_kkt])
    return np.maximum(raw_proposal, lower_bound)


def describe_barrier_update(previous_target, raw_proposal, target, mu_bar, xi,
                            pre_repair_kkt, settings, *, dummy=False):
    """Describe each candidate, preserving every exact tie at the maximum."""
    reference = (previous_target if settings.barrier_update_guard == 'previous'
                 else raw_proposal)
    candidates = dict(raw_proposal=raw_proposal,
                      guard_floor=reference * settings.barrier_reduction_limit,
                      min_floor=settings.min_mu, kkt_floor=0.01 * pre_repair_kkt)
    active = (('dummy',) if dummy else
              tuple(name for name, value in candidates.items() if value == target))
    return dict(previous_target=float(previous_target),
                raw_proposal=float(raw_proposal), target=float(target),
                mu_bar=None if mu_bar is None else float(mu_bar),
                xi=None if xi is None else float(xi),
                guard_reference='dummy' if dummy else settings.barrier_update_guard,
                guard_floor=None if dummy else float(candidates['guard_floor']),
                min_floor=float(settings.min_mu),
                kkt_floor=None if dummy else float(candidates['kkt_floor']),
                pre_repair_kkt=float(pre_repair_kkt), active_reasons=active)
