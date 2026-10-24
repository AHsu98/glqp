"""Iteration schedules belong to an outer direction, not a trial or RHS."""
from dataclasses import asdict
import warnings

import numpy as np
import pytest
from scipy.sparse import csc_array

from glqp import GLQP, SolverSettings
from glqp.obj import LogisticNLL
from glqp.regularization import iteration_decay, decay_regularization
import glqp.linear as linear
import glqp.solver as driver
from glqp.proximal import ProximalState


def problem(glm=False):
    extra = (dict(A=csc_array([[1., 0.], [0., 1.], [1., -1.]]),
                  f=LogisticNLL(np.array([0., 1., 0.]), np.ones(3))) if glm else {})
    return GLQP(Q=csc_array(np.eye(2)), b=np.array([1., 2.]),
        C=csc_array(np.eye(2)), c=np.array([10., 10.]),
        E=csc_array([[1., 1.]]), e=np.array([2.]), **extra)


def solve(model, settings):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return model.solve(settings=settings, verbose=False)[1]


@pytest.mark.parametrize('name', ['prox_decay_iterations', 'tau_decay_iterations'])
@pytest.mark.parametrize('value', [-1., 0., np.nan, np.inf, -np.inf, True,
                                  np.bool_(False), '100', 1j])
@pytest.mark.parametrize('mutated', [False, True])
def test_invalid_timescales_rejected_before_initialization(name, value, mutated):
    with pytest.raises(ValueError, match=name):
        if mutated:
            settings = SolverSettings()
            setattr(settings, name, value)
            problem().initialize(settings=settings)
        else:
            SolverSettings(**{name: value})


@pytest.mark.parametrize('value', [None, 1, .25, np.float32(.1), np.float64(100.),
                                  np.nextafter(0., 1.), np.finfo(float).max])
def test_positive_float_timescales_and_disabled_value_supported(value):
    settings = SolverSettings(prox_decay_iterations=value, tau_decay_iterations=value)
    assert settings.prox_decay_iterations == settings.tau_decay_iterations == value


@pytest.mark.parametrize('timescale', [.1, 1., 10., 100., 1e300, np.nextafter(0., 1.)])
def test_decay_has_exact_unaged_start_and_stable_float_clock(timescale):
    assert iteration_decay(0, timescale) == 1.
    values = [iteration_decay(index, timescale) for index in (0, 1, 10, 100, 10000)]
    assert all(0 <= value <= 1 for value in values)
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert iteration_decay(1, timescale) == pytest.approx(timescale/(timescale+1.), rel=3e-16, abs=0.)
    if timescale == np.nextafter(0., 1.):
        assert iteration_decay(1, timescale) == timescale
    assert iteration_decay(10**308, 1e308) == pytest.approx(.5)
    assert iteration_decay(100, None) == 1.


def test_user_formula_corresponds_to_a_hundred_iteration_timescale():
    for index in (0, 1, 10, 100, 1000, 10000):
        assert iteration_decay(index, 100.) == pytest.approx(10./(10.+index/10.), rel=3e-16)


@pytest.mark.parametrize('strategy', ['newton', 'predictor_corrector'])
@pytest.mark.parametrize('glm', [False, True])
def test_disabled_schedules_never_enter_helper_or_add_history(monkeypatch, strategy, glm):
    def forbidden(*args, **kwargs):
        pytest.fail('Disabled schedules must bypass the numerical helper')

    monkeypatch.setattr(driver, 'decay_regularization', forbidden)
    implicit = solve(problem(glm), SolverSettings(step_strategy=strategy, max_iter=4,
                                                  iteration_diagnostics=True))
    explicit = solve(problem(glm), SolverSettings(step_strategy=strategy, max_iter=4,
        iteration_diagnostics=True, prox_decay_iterations=None, tau_decay_iterations=None))
    for name in ('x', 'y', 's', 'nu'):
        assert getattr(implicit, name).tobytes() == getattr(explicit, name).tobytes()
    assert implicit.history.drop(columns=['time']).equals(explicit.history.drop(columns=['time']))
    assert not any(name.startswith('reg_') for name in implicit.history.columns)
    assert not any('regularization' in row for row in implicit.diagnostics['iterations'])


@pytest.mark.parametrize('strategy', ['newton', 'predictor_corrector'])
@pytest.mark.parametrize('policy', ['legacy', 'no_slack', 'centered'])
@pytest.mark.parametrize('residual_factor', [1e-5, 1.])
def test_physical_decay_scales_both_minimum_branches_and_blocks(strategy, policy, residual_factor):
    settings = SolverSettings(max_iter=4, max_stagnation=100, tol=1e-14,
        step_strategy=strategy, proximal_policy=policy,
        prox_reg=.2, prox_dual_reg=None if policy == 'legacy' else .3,
        prox_residual_factor=residual_factor, prox_decay_iterations=.25,
        tau_decay_iterations=2.5, iteration_diagnostics=True)
    result = solve(problem(), settings)
    assert len(result.history) >= 2
    previous = result.diagnostics['initial']['kkt_residual']
    for index, row in enumerate(result.diagnostics['iterations']):
        record = row['regularization']
        primal, dual = min(.2, previous*residual_factor), min(.2 if policy == 'legacy' else .3, previous*residual_factor)
        assert record['reg_decay_index'] == index
        assert record['reg_kkt_before'] == previous
        assert record['reg_rho_before_decay'] == primal
        assert record['reg_delta_before_decay'] == dual
        assert record['reg_prox_factor'] == iteration_decay(index, .25)
        assert record['reg_tau_factor'] == iteration_decay(index, 2.5)
        assert record['reg_rho_requested'] == primal*record['reg_prox_factor'] == row['rho']
        assert record['reg_delta_requested'] == dual*record['reg_prox_factor']
        assert record['reg_tau_requested'] == settings.tau_reg*record['reg_tau_factor']
        assert record['configured_rho_cap'] == .2
        assert record['configured_delta_cap'] == (.2 if policy == 'legacy' else .3)
        if row['status'] == 'accepted':
            history = result.history.iloc[index]
            assert all(history[key] == value for key, value in record.items() if key.startswith('reg_'))
            previous = row['iterate']['kkt_residual']


def test_centered_progress_selection_happens_after_the_frozen_schedule(monkeypatch):
    incoming = []

    def select(state, rho, delta):
        incoming.append((rho, delta))
        return rho/4., delta/2.

    monkeypatch.setattr(ProximalState, 'select_regularization', select)
    result = solve(problem(), SolverSettings(max_iter=3, proximal_policy='centered',
        proximal_reg_update='progress', prox_reg=.2, prox_dual_reg=.3,
        prox_residual_factor=1., prox_decay_iterations=1., iteration_diagnostics=True))
    assert len(incoming) >= 2
    for selected, row in zip(incoming, result.diagnostics['iterations']):
        trace = row['regularization']
        assert selected == (trace['reg_rho_after_decay'], trace['reg_delta_after_decay'])
        assert trace['reg_rho_requested'] == selected[0]/4.
        assert trace['reg_delta_requested'] == selected[1]/2.
        assert row['proximal']['rho'] == trace['reg_rho_requested']
        assert row['proximal']['delta'] == trace['reg_delta_requested']


@pytest.mark.parametrize('strategy', ['newton', 'predictor_corrector'])
def test_tau_changes_only_main_solve_and_pc_reuses_its_frozen_shift(monkeypatch, strategy):
    calls, pc_steps = [], []
    original = linear.factor_and_solve

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        recorded = dict(kwargs)
        if 'init_tau_reg' not in recorded:
            recorded['init_tau_reg'] = args[3]
        calls.append((recorded, result.info))
        return result

    monkeypatch.setattr(driver, 'factor_and_solve', observed)
    monkeypatch.setattr(linear, 'factor_and_solve', observed)
    model = problem()
    pc = model.solve_predictor_corrector

    def observed_pc(*args, **kwargs):
        result = pc(*args, **kwargs)
        pc_steps.append((kwargs, result))
        return result

    monkeypatch.setattr(model, 'solve_predictor_corrector', observed_pc)
    settings = SolverSettings(max_iter=3, step_strategy=strategy, tau_decay_iterations=.5)
    result = solve(model, settings)
    initial = [(args, info) for args, info in calls if 'primal' in args['residual_blocks']]
    main = [(args, info) for args, info in calls if 'scaled_inequality' in args['residual_blocks']]
    assert len(initial) == 1 and len(main) == len(result.history) == 3
    assert initial[0][0]['init_tau_reg'] == settings.tau_reg
    assert initial[0][1].target_atol == 1e-12 and initial[0][1].target_rtol == 1e-6
    for index, (args, info) in enumerate(main):
        tau = settings.tau_reg*iteration_decay(index, .5)
        assert args['init_tau_reg'] == tau == info.tau_reg
        assert info.target_atol == .05*settings.tol and info.target_rtol == 1e-6
        assert args['max_refinement_steps'] == 10
        assert result.history.iloc[index]['reg_prox_factor'] == 1.
        if strategy == 'predictor_corrector':
            kwargs, pc_result = pc_steps[index]
            assert kwargs['tau_reg'] == pc_result.affine_info.tau_reg == pc_result.combined_info.tau_reg == tau


@pytest.mark.parametrize('strategy', ['newton', 'predictor_corrector'])
@pytest.mark.parametrize('override', [None, 0., 3e-9])
def test_direct_direction_calls_are_unaged_without_explicit_tau_override(strategy, override):
    model = problem()
    settings = SolverSettings(tau_decay_iterations=.1, prox_decay_iterations=.1)
    x, y, s, nu, _ = model.initialize(settings=settings)
    H = model.get_H(np.zeros(model.m))
    rx, rp, rc, req = model.KKT_res(x, model.Q@x, y, s, nu)
    for _ in range(2):
        if strategy == 'newton':
            model.solve_KKT(x, y, s, H, rx, rp, rc-.1, req, .1, prox_reg=2e-8, tau_reg=override)
            info = model._linear_info
        else:
            result = model.solve_predictor_corrector(x, y, s, H, rx, rp, req, 1.,
                                                     prox_reg=2e-8, tau_reg=override)
            info = result.affine_info
            assert info.tau_reg == result.combined_info.tau_reg
        assert info.tau_reg == (settings.tau_reg if override is None else override)


@pytest.mark.parametrize('prox,tau', [(1., None), (None, 1.), (.5, 2.)])
def test_schedule_clock_resets_each_solve_without_mutating_settings(prox, tau):
    config = SolverSettings(max_iter=4, prox_decay_iterations=prox, tau_decay_iterations=tau)
    before = asdict(config)
    model = problem()
    first, second = solve(model, config), solve(model, config)
    assert asdict(config) == before
    assert asdict(first.settings) == asdict(second.settings) == before
    assert first.history['reg_decay_index'].tolist() == list(range(len(first.history)))
    assert first.history.drop(columns=['time']).equals(second.history.drop(columns=['time']))
    assert first.x.tobytes() == second.x.tobytes()


def test_zero_regularization_stays_zero_under_independent_schedules():
    settings = SolverSettings(prox_decay_iterations=1., tau_decay_iterations=.5, tau_reg=0.)
    decayed = decay_regularization(settings, 1000, 0., 0.)
    assert decayed.rho == decayed.delta == decayed.tau == 0.
