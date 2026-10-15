"""Regression contracts for line search, initialization, and returned results."""
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import brentq
from scipy.sparse import csc_array
from scipy.special import expit

from glqp import GLQP, SolverSettings
from glqp.obj import LogisticNLL


def set_direction(monkeypatch, problem, direction):
    monkeypatch.setattr(problem, "solve_KKT", lambda *args, **kwargs:
                        (np.array(direction), np.zeros(problem.k), np.zeros(problem.k),
                         np.zeros(problem.p), None, 0, 0.))


def test_backtracking_can_reject_several_trials_before_accepting(monkeypatch):
    p = GLQP(Q=csc_array([[1.]]), b=np.zeros(1))
    set_direction(monkeypatch, p, [-4.])
    settings = SolverSettings(max_iter=1, max_linesearch_steps=10,
                              let_newton_cook=0., armijo_additive_eps=0.)
    with pytest.warns(UserWarning):
        x, result = p.solve(x0=np.ones(1), settings=settings, verbose=False)
    step = result.history.iloc[0]["step"]
    assert 0 < step < .5
    assert_allclose(x, [1. - 4.*step])
    assert .5 * x @ x < .5
    assert "failed_line_search" not in result.termination_tag


@pytest.mark.parametrize("max_steps", [0, 4])
def test_exhausted_search_leaves_the_iterate_unchanged(monkeypatch, max_steps):
    p = GLQP(Q=csc_array([[1.]]), b=np.zeros(1))
    set_direction(monkeypatch, p, [1.])  # An ascent direction: every trial fails.
    settings = SolverSettings(max_iter=3, max_linesearch_steps=max_steps,
                              let_newton_cook=0., armijo_additive_eps=0.)
    with pytest.warns(UserWarning, match="Failed Line Search"):
        x, result = p.solve(x0=np.ones(1), settings=settings, verbose=False)
    assert result.termination_tag == "failed_line_search"
    assert_allclose(x, [1.], atol=0)
    assert result.history.empty
    assert result.primal == .5


def test_fallback_does_not_accept_nonfinite_merit_or_evaluate_invalid_gradient(monkeypatch):
    class LogBarrier:
        def __call__(self, z):
            return -np.log(z[0]) if z[0] > 0 else np.inf
        def d1f(self, z):
            assert np.all(z > 0), "Derivative evaluated outside the domain"
            return -1 / z
        def d2f(self, z):
            return 1 / z**2

    p = GLQP(f=LogBarrier(), A=csc_array([[1.]]), Q=csc_array([[1.]]))
    set_direction(monkeypatch, p, [-4.])
    settings = SolverSettings(max_iter=1, max_linesearch_steps=20, let_newton_cook=.9)
    with pytest.warns(UserWarning):
        x, result = p.solve(x0=np.array([2.]), settings=settings, verbose=False)
    assert x[0] > 0
    assert np.isfinite(result.primal)


def test_logistic_search_converges_to_independent_scalar_optimum():
    p = GLQP(f=LogisticNLL([0, 1], [.9, .1]), A=csc_array([[1.], [1.]]),
             Q=csc_array([[1e-4]]))
    x, result = p.solve(x0=np.array([5.]), verbose=False)
    reference = brentq(lambda z: expit(z) - .1 + 1e-4 * z, -3, 0)
    assert result.termination_tag == "optimal"
    assert_allclose(x, [reference], atol=1e-6)
    objectives = np.r_[p.f(np.array([5., 5.])) + .5e-4*25,
                       result.history["primal"].to_numpy()]
    assert np.all(np.diff(objectives) <= 1e-8)


def test_non_descent_restoration_can_still_accept_a_kkt_improvement(monkeypatch):
    p = GLQP(Q=csc_array([[1.]]), b=np.zeros(1), C=csc_array([[1.]]), c=np.ones(1))
    monkeypatch.setattr(p, "solve_KKT", lambda *args, **kwargs:
                        (np.array([-.5]), np.array([-.5]), np.zeros(1),
                         np.zeros(0), None, 0, 0.))
    settings = SolverSettings(max_iter=1, let_newton_cook=0., armijo_additive_eps=0.)
    with pytest.warns(UserWarning):
        x, result = p.solve(x0=np.array([2.]), s0=np.ones(1), settings=settings, verbose=False)
    assert_allclose(x, [1.5])
    assert result.history.iloc[0]["step"] == 1.
    assert result.history.iloc[0]["KKT_res"] < 3.


def test_quadratic_restoration_keeps_residual_acceptance_for_descent_direction(monkeypatch):
    # A supplied slack can be positive but inconsistent with Cx + s = c.
    # Preserve the QP residual test while restoring that consistency, even
    # when a merit-descent direction fails Armijo at its full trial step.
    p = GLQP(Q=csc_array([[1.]]), b=np.zeros(1), C=csc_array([[1.]]), c=np.array([10.]))
    monkeypatch.setattr(p, "solve_KKT", lambda *args, **kwargs:
                        (np.array([-4.]), np.array([8.]), np.array([-.99]),
                         np.zeros(0), None, 0, 0.))
    settings = SolverSettings(max_iter=1, let_newton_cook=0., armijo_additive_eps=0.)
    with pytest.warns(UserWarning):
        x, result = p.solve(x0=np.ones(1), s0=np.ones(1), y0=np.ones(1),
                            mu0=.001, settings=settings, verbose=False)
    assert result.history.iloc[0]["step"] == 1.
    assert_allclose(x, [-3.])
    assert result.history.iloc[0]["KKT_res"] < 8.
    assert .5*x@x - .001*np.log(result.s).sum() > .5


@pytest.mark.parametrize("name,value", [
    ("s0", [-1.]), ("s0", [0.]), ("s0", [np.nan]), ("s0", [np.inf]),
    ("y0", [-1.]), ("y0", [0.]), ("y0", [np.nan]), ("y0", [np.inf]),
    ("x0", [np.nan]), ("x0", [np.inf]),
    ("s0", [[1.]]), ("y0", []), ("x0", [1., 2.]),
])
def test_invalid_warm_start_is_rejected_explicitly(name, value):
    p = GLQP(Q=csc_array([[1.]]), b=np.ones(1), C=csc_array([[1.]]), c=np.zeros(1))
    with pytest.raises(ValueError, match=name):
        p.solve(**{name: value}, verbose=False)


def test_valid_positive_warm_starts_are_copied_without_changing_inputs():
    p = GLQP(Q=csc_array([[1.]]), b=np.ones(1), C=csc_array([[1.]]), c=np.zeros(1))
    values = dict(x0=np.array([-1.]), s0=np.array([1.]), y0=np.array([1e-12]))
    x, y, s, _, _ = p.initialize(**values)
    for result, name in ((x, "x0"), (s, "s0"), (y, "y0")):
        assert_allclose(result, values[name], atol=0)
        assert not np.shares_memory(result, values[name])


@pytest.mark.parametrize("equality", [False, True])
@pytest.mark.parametrize("explicit_warm_start", [False, True])
def test_empty_inequalities_solve_with_or_without_equalities(equality, explicit_warm_start):
    kwargs = dict(Q=csc_array(np.eye(2)), b=np.array([1., -2.]),
                  C=csc_array((0, 2)), c=np.array([]))
    if equality:
        kwargs.update(E=csc_array([[1., 1.]]), e=np.array([0.]))
    start = dict(s0=np.array([]), y0=np.array([])) if explicit_warm_start else {}
    p = GLQP(**kwargs)
    x, result = p.solve(**start, verbose=False)
    expected = [1.5, -1.5] if equality else [1., -2.]
    assert result.termination_tag == "optimal"
    assert_allclose(x, expected, atol=1e-7)
    assert result.s.shape == result.y.shape == (0,)
    assert result.feasible


@pytest.mark.parametrize("max_iter", [1, 200])
def test_returned_and_logged_objectives_match_the_returned_iterate(max_iter):
    p = GLQP(Q=csc_array([[1.]]), b=np.array([.1]))
    x, result = p.solve(settings=SolverSettings(max_iter=max_iter), verbose=False)
    expected = .5 * x @ x - .1 * x[0]
    assert result.primal == pytest.approx(expected, abs=1e-14)
    assert result.history.iloc[-1]["primal"] == pytest.approx(expected, abs=1e-14)
