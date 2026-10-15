"""Fixed quadratic curvature and changing GLM curvature share one interface."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csc_array

from glqp import GLQP
from glqp.obj import PoissonNLL
import glqp.solver as solver_module


def test_absent_glm_returns_quadratic_without_sparse_product(monkeypatch):
    problem = GLQP(Q=csc_array([[2., .3], [.3, 4.]]))

    def unexpected_product(*args, **kwargs):
        raise AssertionError("An absent GLM must not evaluate sparse curvature")

    monkeypatch.setattr(solver_module, "dot_product_mkl", unexpected_product)
    assert problem.get_H(np.zeros(1)) is problem.Q
    # Replacing Q between solves must not leave a stale cached Hessian.
    problem.Q = csc_array([[3., 0.], [0., 5.]])
    assert problem.get_H(np.zeros(1)) is problem.Q


def test_glm_curvature_refreshes_and_preserves_quadratic_base():
    A = np.array([[1., 2.], [2., -1.], [.5, .3]])
    Q = np.array([[.7, .2], [.2, 1.1]])
    weights = np.array([.8, 0., 1.3])
    problem = GLQP(A=csc_array(A), Q=csc_array(Q),
                   f=PoissonNLL([0., 2., 1.], weights))
    original = problem.Q.data.copy()
    for x in (np.zeros(2), np.array([.3, -.8]), np.zeros(2)):
        z = A @ x
        expected = Q + A.T @ np.diag(weights * np.exp(z)) @ A
        assert_allclose(problem.get_H(z).toarray(), expected, rtol=2e-15, atol=2e-15)
        assert_array_equal(problem.Q.data, original)


def test_solving_does_not_add_regularization_to_quadratic_base():
    problem = GLQP(Q=csc_array([[2., .3], [.3, 4.]]), b=np.array([1., 2.]))
    original = problem.Q.toarray().copy()
    for _ in range(2):
        x, result = problem.solve(verbose=False)
        assert result.termination_tag == "optimal"
        assert_allclose(x, np.linalg.solve(original, problem.b), atol=1e-7)
        assert_array_equal(problem.Q.toarray(), original)
