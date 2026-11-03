"""Experimental coefficient and inner-accuracy settings keep their scope."""
import numpy as np
import pytest
from scipy.sparse import csc_array

from glqp import GLQP, SolverSettings
import glqp.linear as linear
import glqp.solver as solver


def problem():
    # A nonzero equality RHS exercises the separate initialization solve.
    return GLQP(Q=csc_array(np.eye(2)), b=np.array([1., 2.]),
                C=csc_array(np.eye(2)), c=np.array([10., 10.]),
                E=csc_array([[1., 1.]]), e=np.array([2.]))


@pytest.mark.parametrize('name', ['prox_residual_factor', 'linear_atol_factor', 'linear_rtol'])
@pytest.mark.parametrize('value', [-1., np.nan, np.inf, -np.inf, True, np.bool_(False), None, '0.1', 1j])
@pytest.mark.parametrize('mutated', [False, True])
def test_invalid_coefficients_are_rejected_before_initialization(name, value, mutated):
    with pytest.raises(ValueError, match=name):
        if mutated:
            config = SolverSettings()
            setattr(config, name, value)
            problem().initialize(settings=config)
        else:
            SolverSettings(**{name: value})


@pytest.mark.parametrize('value', [0., 0, np.float64(.1), np.float32(.1)])
def test_nonnegative_real_coefficients_are_supported(value):
    config = SolverSettings(prox_residual_factor=value, linear_atol_factor=value,
                            linear_rtol=value)
    assert config.prox_residual_factor == config.linear_atol_factor == config.linear_rtol == value


@pytest.mark.parametrize('strategy', ['newton', 'predictor_corrector'])
def test_main_accuracy_controls_do_not_change_equality_projection(monkeypatch, strategy):
    original, calls = linear.factor_and_solve, []

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((kwargs, result.info))
        return result

    # Newton calls the imported entry point; PC's verified-system wrapper
    # calls the linear module entry point and then reuses its factors.
    monkeypatch.setattr(solver, 'factor_and_solve', observed)
    monkeypatch.setattr(linear, 'factor_and_solve', observed)
    config = SolverSettings(step_strategy=strategy, tol=1e-6, max_iter=2,
                            linear_atol_factor=.003, linear_rtol=2e-9)
    with pytest.warns(UserWarning, match='Maximum Iterations'):
        _, result = problem().solve(settings=config, verbose=False)
    assert len(result.history) == 2
    initial = [(args, info) for args, info in calls if 'primal' in args['residual_blocks']]
    main = [(args, info) for args, info in calls if 'scaled_inequality' in args['residual_blocks']]
    assert len(initial) == 1 and len(main) == 2
    assert initial[0][1].target_atol == 1e-12
    assert initial[0][1].target_rtol == 1e-6
    for args, info in main:
        assert info.target_atol == .003*config.tol
        assert info.target_rtol == 2e-9
        assert info.tau_reg == config.tau_reg == 2e-8
        assert args['max_refinement_steps'] == config.max_iterative_refinement == 10


@pytest.mark.parametrize('policy', ['legacy', 'no_slack', 'centered'])
@pytest.mark.parametrize('factor', [0., 1e-5, 1.])
def test_residual_factor_controls_both_requested_physical_coefficients(monkeypatch, policy, factor):
    model = problem()
    original, coefficients = model.solve_KKT, []

    def observed(*args, **kwargs):
        coefficients.append((kwargs['prox_reg'], kwargs.get('dual_reg', kwargs['prox_reg']),
                             kwargs.get('complementarity_reg', kwargs['prox_reg'])))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, 'solve_KKT', observed)
    config = SolverSettings(proximal_policy=policy, prox_reg=.2,
        prox_dual_reg=None if policy == 'legacy' else .3,
        prox_residual_factor=factor, max_iter=1, iteration_diagnostics=True)
    with pytest.warns(UserWarning, match='Maximum Iterations'):
        _, result = model.solve(settings=config, verbose=False)
    initial = result.diagnostics['initial']['kkt_residual']
    primal = min(.2, initial*factor)
    dual = min(.2 if policy == 'legacy' else .3, initial*factor)
    assert coefficients == [(primal, dual, primal if policy == 'legacy' else 0.)]
    assert result.diagnostics['iterations'][0]['rho'] == primal
    assert result.diagnostics['policies']['prox_residual_factor'] == factor
    assert result.diagnostics['policies']['linear_atol_factor'] == .005
    assert result.diagnostics['policies']['linear_rtol'] == 1e-6
