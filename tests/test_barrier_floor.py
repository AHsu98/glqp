"""Resolve the absolute barrier floor without changing unrelated solver paths."""
from dataclasses import asdict
import warnings

import numpy as np
from numpy.testing import assert_array_equal
import pytest
from scipy.sparse import csc_array, eye_array

from glqp import GLQP, SolverSettings
from glqp.obj import LogisticNLL


def solve(problem, settings, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return problem.solve(settings=settings, verbose=False, **kwargs)[1]


def small_problem(glm=False):
    kwargs = dict(Q=eye_array(2), b=np.array([1., -.5]),
                  C=csc_array([[1., 0.], [0., -1.], [1., 1.]]),
                  c=np.array([.2, .4, 1.]))
    if glm:
        kwargs.update(A=eye_array(2), f=LogisticNLL([0., 1.], [1., 1.]))
    return GLQP(**kwargs)


def assert_same_trajectory(first, second):
    assert first.termination_tag == second.termination_tag
    for name in ('x', 'y', 's', 'nu'):
        assert_array_equal(getattr(first, name), getattr(second, name))
    columns = [name for name in first.history if name != 'time']
    assert_array_equal(first.history[columns], second.history[columns])


@pytest.mark.parametrize('name,value', [
    ('min_mu_policy', 'typo'), ('min_mu_policy', None), ('min_mu_policy', True),
    ('min_mu', -1.), ('min_mu', np.nan), ('min_mu', np.inf), ('min_mu', True),
    ('tol', 0.), ('tol', -1.), ('tol', np.nan), ('tol', np.inf), ('tol', True),
])
@pytest.mark.parametrize('mutated', [False, True])
def test_invalid_floor_inputs_cannot_be_hidden_by_the_cap(name, value, mutated):
    with pytest.raises(ValueError, match=name):
        if mutated:
            settings = SolverSettings()
            setattr(settings, name, value)
        else:
            settings = SolverSettings(**{name: value})
        small_problem().initialize(settings=settings)


@pytest.mark.parametrize('policy', ['constraint_count', 'fixed'])
def test_reusing_caller_settings_resolves_each_problem_independently(policy):
    settings = SolverSettings(tol=1e-9, min_mu=1e-6, min_mu_policy=policy)
    original = asdict(settings)
    for count in (2, 8, 1):
        model = GLQP(Q=eye_array(1), C=csc_array((count, 1)), c=np.ones(count))
        *_, effective = model.initialize(settings=settings)
        expected = .1 * settings.tol / count if policy == 'constraint_count' else settings.min_mu
        assert effective is model.settings and effective is not settings
        assert effective.min_mu == expected
        assert asdict(settings) == original
        assert model.barrier_floor == dict(
            policy=policy, configured_min_mu=1e-6, effective_min_mu=expected,
            real_inequalities=count, count_cap=.1 * settings.tol / count)


@pytest.mark.parametrize('empty', [False, True])
def test_dummy_or_empty_inequalities_do_not_cap_the_configured_floor(empty):
    kwargs = dict(C=csc_array((0, 1)), c=np.zeros(0)) if empty else {}
    model = GLQP(Q=eye_array(1), b=np.ones(1), **kwargs)
    settings = SolverSettings(tol=1e-9, min_mu=1e-5, iteration_diagnostics=True)
    result = solve(model, settings)
    assert result.settings.min_mu == settings.min_mu == 1e-5
    assert result.barrier_floor['real_inequalities'] == 0
    assert result.barrier_floor['count_cap'] is None
    assert result.diagnostics['barrier_floor'] == result.barrier_floor
    assert result.diagnostics['initial_target']['min_mu'] == 1e-5
    assert_array_equal(result.history['mu'], np.full(len(result.history), 1e-5))


@pytest.mark.parametrize('glm', [False, True])
@pytest.mark.parametrize('minimum', [1e-12, 1e-20])
def test_unchanged_floor_preserves_all_existing_numerical_outputs(glm, minimum):
    options = dict(tol=1e-7, min_mu=minimum, iteration_diagnostics=True)
    capped = solve(small_problem(glm), SolverSettings(**options))
    fixed = solve(small_problem(glm), SolverSettings(**options, min_mu_policy='fixed'))
    assert_same_trajectory(capped, fixed)
    assert capped.settings.min_mu == fixed.settings.min_mu == minimum


@pytest.mark.parametrize('step,proximal', [
    ('newton', 'legacy'), ('predictor_corrector', 'legacy'),
    ('newton', 'centered'), ('predictor_corrector', 'centered'),
])
@pytest.mark.parametrize('glm', [False, True])
def test_resolved_floor_matches_an_explicit_fixed_floor_across_direction_paths(step, proximal, glm):
    options = dict(tol=1e-9, max_iter=100, iteration_diagnostics=True,
                   step_strategy=step, proximal_policy=proximal,
                   initial_mu_policy='complementarity')
    cap = .1 * options['tol'] / 3
    settings = SolverSettings(**options, min_mu=1e-4)
    capped = solve(small_problem(glm), settings)
    fixed = solve(small_problem(glm), SolverSettings(**options, min_mu=cap, min_mu_policy='fixed'))
    assert_same_trajectory(capped, fixed)
    assert settings.min_mu == 1e-4
    assert capped.settings.min_mu == cap
    assert capped.barrier_floor['configured_min_mu'] == 1e-4
    assert capped.barrier_floor['effective_min_mu'] == cap
    assert capped.diagnostics['barrier_floor'] == capped.barrier_floor
    assert capped.diagnostics['initial_target']['min_mu'] == cap
    assert capped.diagnostics['policies']['min_mu_policy'] == 'constraint_count'
    assert all(row['barrier']['min_floor'] == cap for row in capped.diagnostics['iterations'])


@pytest.mark.parametrize('step', ['newton', 'predictor_corrector'])
def test_initial_observation_and_direction_share_the_resolved_floor(step):
    model = GLQP(Q=eye_array(1), b=np.ones(1), C=csc_array((1, 1)), c=np.array([1e-8]))
    settings = SolverSettings(tol=1e-9, min_mu=1e-5, initial_mu_policy='complementarity',
                              step_strategy=step, max_iter=1, iteration_diagnostics=True)
    result = solve(model, settings, s0=np.array([1e-8]), y0=np.array([1e-8]))
    expected = .1 * settings.tol
    assert result.diagnostics['initial']['barrier_target'] == expected
    assert result.diagnostics['initial_target']['min_mu'] == expected
    assert result.diagnostics['iterations'][0]['barrier']['min_floor'] == expected


def test_many_inequalities_cannot_hold_aggregate_stopping_above_tolerance():
    # Genuine but redundant inequalities 0*x <= 1 isolate k*mu without a
    # difficult factorization. The optimum is x=.3, with objective below one.
    k, tolerance = 2048, 1e-9
    def problem():
        return GLQP(Q=eye_array(1), b=np.array([.3]), C=csc_array((k, 1)), c=np.ones(k))
    options = dict(tol=tolerance, max_stagnation=8, max_iter=100)
    fixed = solve(problem(), SolverSettings(**options, min_mu_policy='fixed'))
    capped = solve(problem(), SolverSettings(**options))
    assert fixed.termination_tag == 'near_opt_stagnated'
    assert np.sum(np.abs(fixed.y)) > tolerance
    assert np.sum(np.abs(fixed.y)) == pytest.approx(k * 1e-12, rel=1e-10)
    assert capped.termination_tag == 'optimal'
    assert np.sum(np.abs(capped.y)) <= tolerance
    assert abs(capped.x[0] - .3) <= tolerance
    assert np.all(capped.y >= 0)
    assert capped.settings.min_mu == .1 * tolerance / k
    assert capped.barrier_floor['configured_min_mu'] == 1e-12
