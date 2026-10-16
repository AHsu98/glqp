"""Guard expensive work counts while retaining fresh accepted-iterate values."""

import numpy as np
import pytest
from scipy.sparse import csc_array

from glqp import GLQP, SolverSettings
from glqp.obj import GaussianNLL


class CountingMatrix(csc_array):
    def __matmul__(self, other):
        self.products += 1
        return super().__matmul__(other)


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
    problem.Q.products = 0
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
