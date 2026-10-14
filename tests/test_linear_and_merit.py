"""Small numerical regressions; no external benchmark data is required."""
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import csc_array, eye_array

from glqp import GLQP, SolverSettings
from glqp.util import factor_and_solve
import glqp.solver as solver_module


def linear_solve(matrix, rhs, cached=None):
    return factor_and_solve(
        csc_array(matrix), np.asarray(rhs, dtype=float),
        reg_shift=csc_array(eye_array(len(rhs))), init_tau_reg=0.,
        solver=cached, target_atol=1e-11, max_solve_attempts=1,
        max_refinement_steps=1,
    )


def test_zero_linear_rhs_returns_exact_finite_solution():
    with np.errstate(divide="raise", invalid="raise"):
        x, refinements, cached, relative_error = linear_solve([[2., 0.], [0., -3.]], [0., 0.])
    assert_allclose(x, [0., 0.], atol=0)
    assert refinements == 0
    assert relative_error == 0
    # A skipped factorization must not prevent a subsequent nonzero solve.
    x, _, _, _ = linear_solve([[2., 0.], [0., -3.]], [4., 3.], cached)
    assert_allclose(x, [2., -1.], atol=1e-12)


@pytest.mark.parametrize("first,second", [
    ([[2., 0.], [0., 3.]], [[2., .5], [.5, 3.]]),
    ([[2., .5], [.5, 3.]], [[2., 0.], [0., 3.]]),
    ([[4., 1., 0.], [1., 4., 0.], [0., 0., 4.]],
     [[4., 0., 1.], [0., 4., 0.], [1., 0., 4.]]),
])
def test_changed_sparsity_matches_independent_dense_solution(first, second):
    rhs = np.arange(1., len(first) + 1)
    _, _, cached, _ = linear_solve(first, rhs)
    x, _, cached, _ = linear_solve(second, rhs, cached)
    assert_allclose(x, np.linalg.solve(second, rhs), atol=1e-11, rtol=1e-11)
    assert_allclose(np.asarray(second) @ x, rhs, atol=1e-11)
    # Numerical updates with the same pattern should still reuse the factorization.
    third = 2 * np.asarray(second)
    x, _, reused, _ = linear_solve(third, rhs, cached)
    assert reused is cached
    assert_allclose(x, np.linalg.solve(third, rhs), atol=1e-11, rtol=1e-11)


def test_zero_rhs_does_not_mark_an_old_factorization_as_current():
    first, second = [[2., 0.], [0., 3.]], [[2., .5], [.5, 3.]]
    _, _, cached, _ = linear_solve(first, [1., 2.])
    _, _, cached, _ = linear_solve(second, [0., 0.], cached)
    x, _, _, _ = linear_solve(second, [1., 2.], cached)
    assert_allclose(x, np.linalg.solve(second, [1., 2.]), atol=1e-11)


def test_tiny_nonzero_rhs_is_not_rounded_to_a_zero_solve():
    x, refinements, _, relative_error = linear_solve([[2.]], [2e-200])
    assert_allclose(x, [1e-200], rtol=1e-12, atol=0)
    assert refinements == 0
    assert relative_error == 0


def test_zero_kkt_rhs_does_not_break_a_valid_quadratic():
    p = GLQP(Q=csc_array([[1.]]), b=np.array([1.]),
             C=csc_array([[1.]]), c=np.array([100.]))
    x, result = p.solve(verbose=False)
    assert result.termination_tag == "optimal"
    assert_allclose(x, [1.], atol=1e-7)
    assert np.isfinite(result.dual_res)


def test_fractional_quadratic_trial_uses_its_actual_objective(monkeypatch):
    # At t=1/4 this direction reduces the objective from .5 to .15625,
    # although the gradient's max norm grows from 1 to 2.5. Consequently
    # Armijo must accept it without relying on the alternative KKT test.
    p = GLQP(Q=csc_array(np.diag([1., 100.])), b=np.zeros(2))
    direction = np.array([-2., .1])
    monkeypatch.setattr(p, "solve_KKT", lambda *args, **kwargs:
                        (direction, np.zeros(1), np.zeros(1), np.zeros(0), None, 0, 0.))
    monkeypatch.setattr(solver_module, "get_step_size", lambda *args, **kwargs: .25)
    settings = SolverSettings(max_iter=1, let_newton_cook=0., armijo_additive_eps=0.)
    with pytest.warns(UserWarning):
        x, result = p.solve(x0=np.array([1., 0.]), settings=settings, verbose=False)
    assert_allclose(x, [.5, .025], atol=1e-14)
    assert result.history.iloc[0]["step"] == .25
