"""Numerical stopping contracts for total complementarity at the returned x."""
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import eye_array

from glqp import GLQP, SolverSettings


def test_many_small_complementarity_products_do_not_imply_convergence():
    # Independent bound-constrained quadratics have optimum x=0, y=1.
    # Each initial product is below tol, but their sum exceeds tol twentyfold.
    # A small initial barrier target isolates termination from warm-start centering.
    n, tol = 40, 1e-7
    slack = np.full(n, .5*tol)
    p = GLQP(Q=eye_array(n), b=np.ones(n), C=eye_array(n), c=np.zeros(n))
    x, result = p.solve(x0=-slack, s0=slack, y0=1+slack, mu0=1e-12,
                        settings=SolverSettings(tol=tol), verbose=False)
    assert result.termination_tag == 'optimal'
    assert len(result.history) > 0
    assert np.sum(np.abs(result.y * (-x))) <= tol
    assert_allclose(x, np.zeros(n), atol=tol)


@pytest.mark.parametrize('signed_start', [[-5e-8], [-5e-8, 5e-8]])
def test_large_duals_require_complementarity_from_actual_constraint_slack(signed_start):
    # Stored slack has tiny products, but its small consistency error is
    # amplified by the dual. With two rows, signed true products also cancel.
    x0 = np.array(signed_start)
    n, tol = len(x0), 1e-7
    dual = np.full(n, 1e4)
    p = GLQP(Q=eye_array(n), b=x0+dual, C=eye_array(n), c=np.zeros(n))
    x, result = p.solve(x0=x0, s0=np.full(n, 1e-12), y0=dual, mu0=1e-12,
                        settings=SolverSettings(tol=tol), verbose=False)
    assert result.termination_tag == 'optimal'
    assert len(result.history) > 0
    assert np.sum(np.abs(result.y * (-x))) <= tol
    assert_allclose(x, np.zeros(n), atol=tol/1e4)


def test_aggregate_complementarity_is_relative_to_objective_scale():
    # Optimum is x=1 with objective -60. The initial objective error is
    # already within the requested relative tolerance, despite a larger
    # absolute sum of complementarity products.
    n, tol = 40, 1e-7
    slack = np.full(n, .5*tol)
    p = GLQP(Q=eye_array(n), b=np.full(n, 2.), C=eye_array(n), c=np.ones(n))
    x, result = p.solve(x0=1-slack, s0=slack, y0=1+slack,
                        settings=SolverSettings(tol=tol), verbose=False)
    assert result.termination_tag == 'optimal'
    assert result.history.empty
    assert abs(result.primal + 60.) / 60. <= tol
    assert np.sum(np.abs(result.y*(1-x))) / abs(result.primal) <= tol
