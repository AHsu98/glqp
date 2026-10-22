"""Barrier target selection, kept separate from centrality and slack repair."""
from dataclasses import replace
import numbers

import numpy as np


def validate_barrier_settings(settings):
    """Validate again at solve entry because experiment settings are mutable."""
    if (not isinstance(settings.min_mu_policy, str)
            or settings.min_mu_policy not in ('fixed', 'constraint_count')):
        raise ValueError("min_mu_policy must be 'fixed' or 'constraint_count'")
    if settings.barrier_update_guard not in ('previous', 'legacy'):
        raise ValueError("barrier_update_guard must be 'previous' or 'legacy'")
    limit = settings.barrier_reduction_limit
    if (isinstance(limit, (bool, np.bool_)) or not isinstance(limit, numbers.Real)
            or not np.isfinite(limit) or not 0 <= limit <= 1):
        raise ValueError('barrier_reduction_limit must be finite and between 0 and 1')
    if not isinstance(settings.iteration_diagnostics, bool):
        raise ValueError('iteration_diagnostics must be a bool')
    if (not isinstance(settings.initial_mu_policy, str)
            or settings.initial_mu_policy not in ('fixed', 'complementarity')):
        raise ValueError("initial_mu_policy must be 'fixed' or 'complementarity'")
    for name, positive in (('initial_mu_scale', True), ('barrier_kkt_floor', False),
                           ('min_mu', False), ('tol', True)):
        value = getattr(settings, name)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not np.isfinite(value) or (value <= 0 if positive else value < 0)):
            bound = 'positive' if positive else 'nonnegative'
            raise ValueError(f'{name} must be finite and {bound}')


def resolve_barrier_floor(settings, real_inequalities):
    """Copy settings once so every barrier path uses the same effective floor.

    An aggregate complementarity test can plateau near ``k * min_mu``. The
    constraint-count policy reserves a tenth of its absolute stopping budget
    for that floor. Count genuine input inequalities, excluding any internal
    dummy row. Keep the configured value in the returned provenance; never
    modify the caller's reusable settings or cap an already-small floor upward.
    """
    cap = (.1 * settings.tol / real_inequalities if real_inequalities else None)
    effective = settings.min_mu
    if settings.min_mu_policy == 'constraint_count' and cap is not None:
        effective = min(effective, cap)
    provenance = dict(policy=settings.min_mu_policy,
                      configured_min_mu=float(settings.min_mu),
                      effective_min_mu=float(effective),
                      real_inequalities=int(real_inequalities),
                      count_cap=None if cap is None else float(cap))
    return replace(settings, min_mu=effective), provenance


def initial_barrier_target(s, y, dummy_inequalities, mu0, settings):
    """Honor an explicit target before considering an initialization policy."""
    if mu0 is not None:
        return mu0
    if dummy_inequalities or not len(s):
        return settings.min_mu
    if settings.initial_mu_policy == 'fixed':
        return 100.
    with np.errstate(over='ignore', invalid='ignore'):
        target = np.maximum(settings.min_mu, settings.initial_mu_scale*np.dot(s, y)/len(s))
    if not np.isfinite(target):
        raise ValueError("initial_mu_policy='complementarity' produced a nonfinite target")
    return target


def describe_initial_target(s, y, dummy_inequalities, mu0, target, settings):
    source = ('explicit' if mu0 is not None else
              'dummy' if dummy_inequalities or not len(s) else settings.initial_mu_policy)
    mean = float(np.dot(s, y)/len(s)) if len(s) else None
    scaled = (float(settings.initial_mu_scale*np.dot(s, y)/len(s)) if len(s) else None)
    return dict(policy=settings.initial_mu_policy, source=source,
                scale=float(settings.initial_mu_scale), mu_bar=mean,
                scaled_complementarity=scaled, min_mu=float(settings.min_mu), target=float(target))


def select_barrier_target(previous_target, raw_proposal, pre_repair_kkt, settings):
    """Limit reduction relative to the previous target, or reproduce legacy.

    The legacy guard uses the proposal itself. Keep the old max operations
    and their evaluation order so the default legacy limit reproduces the
    accepted implementation's floating-point arithmetic exactly.
    """
    reference = (previous_target if settings.barrier_update_guard == 'previous'
                 else raw_proposal)
    lower_bound = np.max([reference * settings.barrier_reduction_limit,
                          settings.min_mu, settings.barrier_kkt_floor * pre_repair_kkt])
    return np.maximum(raw_proposal, lower_bound)


def describe_barrier_update(previous_target, raw_proposal, target, mu_bar, xi,
                            pre_repair_kkt, settings, *, dummy=False):
    """Describe each candidate, preserving every exact tie at the maximum."""
    reference = (previous_target if settings.barrier_update_guard == 'previous'
                 else raw_proposal)
    candidates = dict(raw_proposal=raw_proposal,
                      guard_floor=reference * settings.barrier_reduction_limit,
                      min_floor=settings.min_mu, kkt_floor=settings.barrier_kkt_floor * pre_repair_kkt)
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
                kkt_floor_coefficient=float(settings.barrier_kkt_floor),
                pre_repair_kkt=float(pre_repair_kkt), active_reasons=active)
