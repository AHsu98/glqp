"""Distinct acceptance contracts survive the shared trial controller."""
from types import SimpleNamespace

import numpy as np
import pytest

from glqp import SolverSettings
from glqp.diagnostics import start_line_search_diagnostics
from glqp.linesearch import CachedGLMRay, search_scalar_ray
from glqp.objective_search import CachedObjectiveRay, search_restoration_ray
from glqp.obj import GaussianNLL


def objective_ray(dx=-1., *, nu=0., dnu=0.):
    empty = np.empty(0)
    model = SimpleNamespace(QP_mode=True, dummy_A=True, b=np.zeros(1),
                            f=lambda predictor: 0.)
    model._constraint_products = lambda x, y, nu: object()
    model._kkt_residuals = lambda gradient, y, s, products: (
        gradient, empty, empty, empty)
    return CachedObjectiveRay(model, point=(np.ones(1), empty, empty,
        np.array([nu]), empty), direction=(np.array([dx]), empty, empty,
        np.array([dnu])), dz=empty, quadratic=(np.ones(1), np.array([dx]),
        1., dx, dx*dx), mu=0., transpose=None)


def restoration(ray, *, tmax=1., merit0=.5, slope0=-1., **options):
    settings = SolverSettings(max_linesearch_steps=1, qp_linesearch='scalar',
                              armijo_additive_eps=0., **options)
    diagnostics = start_line_search_diagnostics(tmax, slope0)
    empty = np.empty(0)

    def unexpected(*args):
        pytest.fail('One trial leaves no budget for constructing a proposal')

    result, _ = search_restoration_ray(ray, settings, tmax=tmax, merit0=merit0,
        slope0=slope0, residuals=(np.ones(1), empty, empty), kkt_res=1.,
        perturbed_kkt_before=None, diagnostics=diagnostics,
        proposer_factory=unexpected)
    return result, diagnostics


def strict_search(*, tmax=1., merit0=.5, slope0=-1., additive_eps=0.,
                  trial_is_valid=None):
    empty = np.empty(0)
    ray = CachedGLMRay(GaussianNLL([0.], [1.]), np.ones(1), -np.ones(1),
                       empty, empty, 0., 0., 0., 0., 0., 0.)
    return search_scalar_ray(ray, tmax=tmax, merit0=merit0, slope0=slope0,
        armijo_param=.005, additive_eps=additive_eps, max_trials=1,
        trial_is_valid=trial_is_valid)


@pytest.mark.parametrize('step', [0., -1., np.inf, np.nan])
def test_invalid_step_retains_each_policys_counting_convention(step):
    point, legacy = restoration(objective_ray(), tmax=step)
    strict = strict_search(tmax=step)
    assert point is None and legacy['trial_count'] == 1
    assert legacy['rejection_reasons'] == {'invalid_step': 1}
    assert strict.trial is None and strict.status == 'invalid_step'
    assert strict.counters['trial_count'] == strict.counters['value_evaluations'] == 0


def test_exhausted_restoration_search_never_builds_scalar_model():
    point, diagnostics = restoration(objective_ray(dx=-4.), slope0=-4.)
    assert point is None
    assert diagnostics['trial_count'] == diagnostics['residual_check_count'] == 1
    assert diagnostics['qp_scalar_counters'] is None
    assert diagnostics['qp_scalar_proposals'] == []


def test_strict_policy_rejects_infinite_armijo_bound_legacy_comparison_retains_it():
    # Both operands are finite, but adding the tolerance overflows the bound.
    # The historical policies deliberately differ at this numerical boundary.
    with np.errstate(over='ignore'):
        strict = strict_search(merit0=1e308, additive_eps=1e308)
        ray = objective_ray()
        settings = SolverSettings(max_linesearch_steps=1, armijo_additive_eps=1e308)
        empty = np.empty(0)
        point, _ = search_restoration_ray(ray, settings, tmax=1., merit0=1e308,
            slope0=-1., residuals=(np.ones(1), empty, empty), kkt_res=1.,
            perturbed_kkt_before=None)
    assert strict.trial is None and strict.counters['armijo_failure_count'] == 1
    assert point is not None and point.step == 1. and point.merit == 0.


def test_equality_dual_guard_is_strict_policy_validation_not_legacy_merit_validation():
    # A full step has a perfectly finite objective and overflows only nu.
    nu = dnu = np.array([1e308])

    def valid(step):
        with np.errstate(over='ignore'):
            return bool(np.all(np.isfinite(nu+step*dnu)))

    strict = strict_search(trial_is_valid=valid)
    point, diagnostics = restoration(objective_ray(nu=nu[0], dnu=dnu[0]))
    assert strict.trial is None and strict.counters['invalid_trial_count'] == 1
    assert strict.counters['value_evaluations'] == 0
    assert point is not None and diagnostics['acceptance_reason'] == 'armijo'
