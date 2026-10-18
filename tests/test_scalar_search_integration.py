"""Solver-level contracts for the default cached scalar GLM descent search."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.optimize import brentq
from scipy.sparse import csc_array

from glqp import GLQP, SolverSettings
from glqp.obj import GaussianNLL, LogisticNLL, PoissonNLL


class CountingMatrix(csc_array):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.products = 0

    def __matmul__(self, other):
        self.products += 1
        return super().__matmul__(other)


def scalar_poisson(**options):
    problem = GLQP(f=PoissonNLL([1e6], [1e-6]), A=csc_array([[1.]]),
                   Q=csc_array([[1e-12]]))
    settings = SolverSettings(max_iter=100,
                              max_linesearch_steps=30, tol=1e-9, **options)
    return problem, settings


def test_large_count_poisson_search_reaches_independent_optimum():
    problem, settings = scalar_poisson(iteration_diagnostics=True)
    x, result = problem.solve(settings=settings, verbose=False)
    reference = brentq(lambda v: 1e-6*np.exp(v)-1+1e-12*v, 10., 20.)
    assert result.termination_tag == 'optimal'
    assert_allclose(x, [reference], atol=1e-8, rtol=0)
    first = result.diagnostics['iterations'][0]['line_search']
    assert first['algorithm'] == 'scalar'
    assert first['trial_count'] <= 10
    assert first['accepted_fraction'] < 1e-3
    assert first['acceptance_reason'] == 'armijo'
    assert first['scalar_counters']['poisson_model_steps'] >= 1


def test_scalar_search_diagnostics_do_not_change_the_trajectory():
    outputs = []
    for enabled in (False, True):
        problem, settings = scalar_poisson(iteration_diagnostics=enabled)
        outputs.append(problem.solve(settings=settings, verbose=False))
    (plain_x, plain), (observed_x, observed) = outputs
    assert_array_equal(plain_x, observed_x)
    for name in ('s', 'y', 'nu'):
        assert_array_equal(getattr(plain, name), getattr(observed, name))
    assert len(plain.history) == len(observed.history)
    assert plain.termination_tag == observed.termination_tag


def test_rejected_scalar_trials_use_cached_sparse_products(monkeypatch):
    problem = GLQP(f=GaussianNLL([0.], [1.]), A=csc_array([[1.]]),
                   Q=csc_array([[1.]]))
    for name in ('A', 'Q', 'C', 'E'):
        setattr(problem, name, CountingMatrix(getattr(problem, name)))
    monkeypatch.setattr(problem, 'solve_KKT', lambda *args, **kwargs:
        (np.array([-40.]), np.zeros(problem.k), np.zeros(problem.k),
         np.zeros(problem.p), None, 0, 0.))
    with pytest.warns(UserWarning, match='Maximum Iterations'):
        x, result = problem.solve(x0=np.ones(1), verbose=False,
            settings=SolverSettings(glm_linesearch='scalar', max_iter=1,
                                    armijo_additive_eps=0.))
    assert len(result.history) == 1
    assert 0 < result.history.iloc[0]['step'] < .05
    assert abs(x[0]) < 1
    # Initial and final objective products, plus one direction and one fresh
    # accepted Qx. Rejected scalar trials cannot increase these counts.
    assert problem.Q.products == 4
    assert problem.A.products == 3  # initial predictor, direction, final predictor


@pytest.mark.parametrize('search', ['backtracking', 'scalar'])
def test_scalar_option_preserves_qp_restoration(monkeypatch, search):
    problem = GLQP(Q=csc_array([[1.]]), b=np.zeros(1),
                   C=csc_array([[1.]]), c=np.array([10.]))
    monkeypatch.setattr(problem, 'solve_KKT', lambda *args, **kwargs:
        (np.array([-4.]), np.array([8.]), np.array([-.99]),
         np.zeros(0), None, 0, 0.))
    with pytest.warns(UserWarning, match='Maximum Iterations'):
        x, result = problem.solve(x0=np.ones(1), s0=np.ones(1), y0=np.ones(1),
            mu0=.001, verbose=False, settings=SolverSettings(glm_linesearch=search,
            max_iter=1, let_newton_cook=0., armijo_additive_eps=0., iteration_diagnostics=True))
    assert result.history.iloc[0]['step'] == 1.
    assert_array_equal(x, [-3.])
    assert result.diagnostics['iterations'][0]['line_search']['algorithm'] == 'backtracking'


def test_scalar_option_preserves_explicit_newton_fallback_route():
    problem = GLQP(f=LogisticNLL([0., 1.], [.9, .1]),
                   A=csc_array([[1.], [1.]]), Q=csc_array([[1e-4]]))
    outputs = []
    for search in ('backtracking', 'scalar'):
        with pytest.warns(UserWarning, match='Maximum Iterations'):
            x, result = problem.solve(x0=np.array([5.]), verbose=False,
                settings=SolverSettings(glm_linesearch=search, let_newton_cook=.9,
                                        max_iter=1, iteration_diagnostics=True))
        outputs.append((x, result))
    assert_array_equal(outputs[0][0], outputs[1][0])
    assert len(outputs[0][1].history) == len(outputs[1][1].history)
    assert all(it['line_search']['algorithm'] == 'backtracking'
               for it in outputs[1][1].diagnostics['iterations'])


@pytest.mark.parametrize('value', ['unknown', '', None, 1, True])
def test_invalid_scalar_search_option_rejected(value):
    with pytest.raises(ValueError, match='glm_linesearch'):
        SolverSettings(glm_linesearch=value)


def test_mutated_search_option_validated_before_solving():
    settings = SolverSettings()
    settings.glm_linesearch = 'unknown'
    with pytest.raises(ValueError, match='glm_linesearch'):
        GLQP(Q=csc_array([[1.]])).solve(settings=settings, verbose=False)


def test_nonlinear_non_descent_can_restore_kkt_with_scalar_option(monkeypatch):
    problem = GLQP(f=GaussianNLL([0.], [1.]), A=csc_array([[1.]]),
                   Q=csc_array([[0.]]), C=csc_array([[1.]]), c=np.array([1.]))
    monkeypatch.setattr(problem, 'solve_KKT', lambda *args, **kwargs:
        (np.array([-.5]), np.array([-.5]), np.zeros(1), np.zeros(0), None, 0, 0.))
    with pytest.warns(UserWarning, match='Maximum Iterations'):
        x, result = problem.solve(x0=np.array([2.]), s0=np.ones(1), y0=np.ones(1),
            mu0=100., verbose=False, settings=SolverSettings(glm_linesearch='scalar',
            max_iter=1, armijo_additive_eps=0., iteration_diagnostics=True))
    search = result.diagnostics['iterations'][0]['line_search']
    assert search['merit_slope'] > 0
    assert search['algorithm'] == 'backtracking'
    assert search['acceptance_reason'] == 'residual'
    assert_array_equal(x, [1.5])


def test_centered_proximal_merit_keeps_its_search():
    outputs = []
    for search in ('backtracking', 'scalar'):
        problem = GLQP(f=LogisticNLL([0., 1.], [.9, .1]),
                       A=csc_array([[1.], [1.]]), Q=csc_array([[.01]]))
        outputs.append(problem.solve(x0=np.array([1.]), verbose=False,
            settings=SolverSettings(glm_linesearch=search, proximal_policy='centered',
                                    iteration_diagnostics=True)))
    assert_array_equal(outputs[0][0], outputs[1][0])
    assert outputs[0][1].termination_tag == outputs[1][1].termination_tag == 'optimal'
    assert len(outputs[0][1].history) == len(outputs[1][1].history)
    assert all('scalar_counters' not in it['line_search']
               for it in outputs[1][1].diagnostics['iterations'])


def test_disabled_fallback_and_no_slack_policy_allow_scalar_search():
    problem, settings = scalar_poisson(let_newton_cook=.9,
        newton_fallback_policy='off', proximal_policy='no_slack', iteration_diagnostics=True)
    _, result = problem.solve(settings=settings, verbose=False)
    assert result.termination_tag == 'optimal'
    assert result.diagnostics['iterations'][0]['line_search']['algorithm'] == 'scalar'
