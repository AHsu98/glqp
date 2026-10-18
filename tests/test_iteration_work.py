"""Guard expensive work counts while retaining fresh accepted-iterate values."""

import numpy as np
import pytest
from scipy.sparse import csc_array

from glqp import GLQP, SolverSettings
from glqp.obj import GaussianNLL
from glqp.util import SparseTransposeCache


class CountingMatrix(csc_array):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.products = 0
        self.transposes = 0

    def __matmul__(self, other):
        self.products += 1
        return super().__matmul__(other)

    def transpose(self, *args, **kwargs):
        self.transposes += 1
        return super().transpose(*args, **kwargs)


class CountingGaussian(GaussianNLL):
    def __init__(self, y, w):
        super().__init__(y, w)
        self.values = 0
        self.gradients = 0

    def f(self, z):
        self.values += 1
        return super().f(z)

    def d1f(self, z):
        self.gradients += 1
        return super().d1f(z)


def test_accepted_iterates_supply_next_merit_without_duplicate_matvecs(monkeypatch):
    objective = CountingGaussian(np.zeros(2), np.ones(2))
    problem = GLQP(f=objective, A=csc_array(np.eye(2)), Q=csc_array(np.diag([2., 3.])))
    problem.Q = CountingMatrix(problem.Q)
    for name in ("A", "C", "E"):
        setattr(problem, name, CountingMatrix(getattr(problem, name)))
    observed_gradients = []

    def direction(x, y, s, H, rx, rp, rc, req, mu, **kwargs):
        # Deterministic descent with an unchanged interior barrier ensures one
        # accepted line trial per step, independently of backend behavior.
        observed_gradients.append(rx.copy())
        return (-.25*x, np.zeros(problem.k), np.zeros(problem.k),
                np.zeros(problem.p), None, 0, 0.)

    monkeypatch.setattr(problem, "solve_KKT", direction)
    with pytest.warns(UserWarning, match="Maximum Iterations"):
        x, result = problem.solve(x0=np.array([1., 2.]), verbose=False,
                                  settings=SolverSettings(max_iter=3))
    steps = len(result.history)
    assert steps == 3
    # Initial/current Qx, then Qdx and a fresh Qx per accepted step, and the
    # independent final original-objective evaluation. No merit-only Qx.
    assert problem.Q.products == 2 + 2*steps
    assert objective.gradients == 1 + steps
    assert objective.values == 2 + 2*steps
    assert [getattr(problem, name).transposes for name in ("A", "C", "E")] == [1, 1, 1]
    for index, gradient in enumerate(observed_gradients):
        np.testing.assert_allclose(gradient, np.array([3., 8.])*.75**index,
                                   rtol=0, atol=0)
    np.testing.assert_allclose(x, np.array([1., 2.])*.75**steps, rtol=0, atol=0)


def test_repeated_solve_recomputes_objective_after_problem_values_change():
    problem = GLQP(Q=csc_array(np.diag([2., 4.])), b=np.array([2., 4.]))
    first, first_result = problem.solve(verbose=False)
    assert first_result.termination_tag == "optimal"
    np.testing.assert_allclose(first, [1., 1.], atol=1e-7, rtol=0)
    problem.Q = csc_array(np.diag([4., 8.]))
    problem.b = np.array([1., 6.])
    second, second_result = problem.solve(verbose=False)
    assert second_result.termination_tag == "optimal"
    np.testing.assert_allclose(second, [.25, .75], atol=1e-7, rtol=0)
    assert second_result.primal == pytest.approx(-2.375, abs=1e-12)


def test_rejected_nonlinear_descent_trials_do_not_evaluate_unused_residuals(monkeypatch):
    objective = CountingGaussian(np.zeros(1), np.ones(1))
    problem = GLQP(f=objective, A=csc_array([[1.]]), Q=csc_array([[1.]]))

    def direction(*args, **kwargs):
        return (np.array([-4.]), np.zeros(problem.k), np.zeros(problem.k),
                np.zeros(problem.p), None, 0, 0.)

    monkeypatch.setattr(problem, "solve_KKT", direction)
    with pytest.warns(UserWarning, match="Maximum Iterations"):
        x, result = problem.solve(x0=np.ones(1), verbose=False,
            settings=SolverSettings(max_iter=1, max_linesearch_steps=20,
                                    let_newton_cook=0., armijo_additive_eps=0.,
                                    glm_linesearch='backtracking'))
    # Several finite merit rejections precede one accepted step. Their KKT
    # norms cannot authorize a nonlinear descent step with fallback disabled.
    assert objective.values > 4
    assert objective.gradients == 2
    assert 0 < result.history.iloc[0]['step'] < .5
    np.testing.assert_allclose(x, 1 - 4*result.history.iloc[0]['step'])


def test_sparse_transpose_views_follow_values_and_refresh_replaced_storage():
    matrix = csc_array([[1., 3.], [2., 0.]])
    cache = SparseTransposeCache()
    original = cache.get("A", matrix)
    assert cache.get("A", matrix) is original
    matrix.data[0] = 7.
    assert cache.get("A", matrix) is original
    np.testing.assert_array_equal(original.toarray(), matrix.toarray().T)
    for attribute in ("data", "indices", "indptr"):
        previous = cache.get("A", matrix)
        setattr(matrix, attribute, getattr(matrix, attribute).copy())
        current = cache.get("A", matrix)
        assert current is not previous
        np.testing.assert_array_equal(current.toarray(), matrix.toarray().T)
    replacement = csc_array([[3., 0.], [0., 4.]])
    current = cache.get("A", replacement)
    assert current is not original
    np.testing.assert_array_equal(current.toarray(), replacement.toarray().T)
    cache.clear()
    assert cache.get("A", replacement) is not current


def test_qp_does_not_multiply_dummy_design_or_evaluate_dummy_derivatives(monkeypatch):
    problem = GLQP(Q=csc_array(np.diag([2., 4.])), b=np.array([2., 4.]))
    problem.A = CountingMatrix(problem.A)

    def unused_derivative(z):
        pytest.fail("A QP has no likelihood derivative")

    monkeypatch.setattr(problem.f, "d1f", unused_derivative)
    x, result = problem.solve(verbose=False)
    assert result.termination_tag == "optimal"
    assert problem.A.products == 0
    np.testing.assert_allclose(x, [1., 1.], atol=1e-7, rtol=0)


@pytest.mark.parametrize("with_glm", [False, True])
def test_residual_accepted_trial_reuses_only_exact_point_products(monkeypatch, with_glm):
    objective = CountingGaussian(np.zeros(1), np.ones(1))
    kwargs = dict(f=objective, A=csc_array([[1.]])) if with_glm else {}
    problem = GLQP(Q=csc_array([[1.]]), C=csc_array([[1.]]), c=np.ones(1), **kwargs)
    for name in ("Q", "C", "E"):
        setattr(problem, name, CountingMatrix(getattr(problem, name)))
    monkeypatch.setattr(problem, "solve_KKT", lambda *args, **kwargs:
        (np.array([-.5]), np.array([-.5]), np.zeros(1), np.zeros(0), None, 0, 0.))
    # The barrier increases, but the non-descent restoration reduces KKT.
    with pytest.warns(UserWarning, match="Maximum Iterations"):
        x, result = problem.solve(x0=np.array([2.]), s0=np.ones(1), verbose=False,
            settings=SolverSettings(max_iter=1, let_newton_cook=0., armijo_additive_eps=0.))
    np.testing.assert_array_equal(x, [1.5])
    assert problem.C.products == problem.E.products == 2  # Initial and trial.
    assert problem.Q.products == 4  # Initial, direction, fresh accepted, final.
    if with_glm:
        assert objective.gradients == 2  # Initial and exact accepted trial.
        assert objective.values == 4  # Accepted objective is still fresh.
    expected = (2 if with_glm else 1)*x + result.y
    assert result.dual_res == abs(expected[0])
    assert result.cons_viol == 1.


def test_rejected_trial_products_do_not_survive_later_armijo_acceptance(monkeypatch):
    problem = GLQP(Q=csc_array([[1.]]), C=csc_array([[1.]]), c=np.array([100.]))
    points = []
    products = problem._constraint_products

    def observe(x, y, nu):
        points.append(x.copy())
        return products(x, y, nu)

    monkeypatch.setattr(problem, "_constraint_products", observe)
    monkeypatch.setattr(problem, "solve_KKT", lambda *args, **kwargs:
        (np.array([-4.]), np.zeros(1), np.zeros(1), np.zeros(0), None, 0, 0.))
    with pytest.warns(UserWarning, match="Maximum Iterations"):
        x, result = problem.solve(x0=np.ones(1), s0=np.array([99.]), y0=np.array([.001]),
            verbose=False, settings=SolverSettings(max_iter=1, max_linesearch_steps=20,
                let_newton_cook=0., armijo_additive_eps=0.))
    assert len(points) > 3
    assert not np.array_equal(points[-2], x)
    np.testing.assert_array_equal(points[-1], x)
    assert result.cons_viol == abs((problem.C@x + result.s-problem.c)[0])
