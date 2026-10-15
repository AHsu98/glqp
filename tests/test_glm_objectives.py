"""Likelihood derivatives and constrained solves checked independently."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.optimize import brentq
from scipy.sparse import csc_array

from glqp import GLQP
from glqp.obj import GaussianNLL, LogisticNLL, PoissonNLL


OBJECTIVES = [
    pytest.param(GaussianNLL, [1.2, -0.7, 2.0], id="gaussian"),
    pytest.param(LogisticNLL, [0, 1, 0], id="logistic"),
    pytest.param(PoissonNLL, [0, 2, 4], id="poisson"),
]


@pytest.mark.parametrize("objective_type,response", OBJECTIVES)
@pytest.mark.parametrize("observations", [1, 3])
@pytest.mark.parametrize("padding", [0, 2])
def test_likelihood_derivatives_match_finite_differences(
    objective_type, response, observations, padding
):
    # Unequal weights and a zero weight exercise scaling and inactive data.
    weights = np.array([0.7, 0.0, 2.3])[:observations]
    objective = objective_type(response[:observations], weights)
    z = np.r_[np.array([-0.4, 0.2, 1.1])[:observations], [0.8] * padding]
    original_z = z.copy()
    step = 1e-5
    gradient = objective.d1f(z)
    hessian = objective.d2f(z)
    assert gradient.shape == hessian.shape == z.shape

    gradient_fd = np.empty_like(z)
    hessian_fd = np.empty((len(z), len(z)))
    for index in range(len(z)):
        delta = np.zeros_like(z)
        delta[index] = step
        gradient_fd[index] = (objective(z + delta) - objective(z - delta)) / (2 * step)
        hessian_fd[:, index] = (objective.d1f(z + delta) - objective.d1f(z - delta)) / (2 * step)

    assert_allclose(gradient, gradient_fd, rtol=2e-8, atol=2e-9)
    assert_allclose(np.diag(hessian), hessian_fd, rtol=2e-8, atol=2e-9)
    assert np.all(hessian >= 0)
    assert_array_equal(gradient[observations:], np.zeros(padding))
    assert_array_equal(hessian[observations:], np.zeros(padding))
    assert_array_equal(z, original_z)


@pytest.mark.parametrize("objective_type,response", OBJECTIVES)
def test_padding_does_not_enter_likelihood_or_derivatives(objective_type, response):
    objective = objective_type(response, [0.7, 0.0, 2.3])
    z = np.array([-0.4, 0.2, 1.1])
    # Evaluating exp on the ignored coordinates would overflow for Poisson.
    padded = np.r_[z, 1000.0, -1000.0]
    with np.errstate(over="raise", invalid="raise"):
        assert objective(padded) == objective(z)
        assert_array_equal(objective.d1f(padded), np.r_[objective.d1f(z), 0.0, 0.0])
        assert_array_equal(objective.d2f(padded), np.r_[objective.d2f(z), 0.0, 0.0])


def test_poisson_uses_standard_weighted_negative_log_likelihood_scale():
    # Rates 2 and 4 give 0.5*2 + 2*(4 - 2*log(4)); no outer factor of 1/2.
    objective = PoissonNLL([0, 2], [0.5, 2.0])
    z = [np.log(2.0), np.log(4.0), 1000.0]
    expected = 9.0 - 4.0 * np.log(4.0)
    assert objective.f(z) == pytest.approx(expected, abs=1e-14)
    assert objective(z) == objective.f(z)


def assert_original_kkt(x, result, gradient, C, c, E, e):
    """Check original constraints, multipliers, and stationarity without flags."""
    assert result.termination_tag == "optimal"
    assert np.all(np.isfinite(x))
    assert np.max(C @ x - c) <= 2e-8
    assert_allclose(E @ x, e, atol=2e-8, rtol=0)
    assert np.all(result.y >= 0)
    assert_allclose(gradient + C.T @ result.y + E.T @ result.nu, 0, atol=2e-7, rtol=0)
    assert_allclose(result.y * (c - C @ x), 0, atol=2e-7, rtol=0)


@pytest.mark.parametrize("padded", [False, True])
def test_constrained_gaussian_matches_independent_quadratic_solution(padded):
    design = np.array([[1.0, 2.0], [2.0, -1.0]])
    response, weights = np.array([2.0, -1.0]), np.array([0.7, 1.8])
    Q, b = np.diag([0.2, 0.4]), np.array([0.1, -0.2])
    C, c = np.array([[-1.0, 0.0]]), np.array([-0.1])
    E, e = np.array([[1.0, 1.0]]), np.array([0.6])

    # Gaussian likelihood is an exact quadratic. Solve its dense KKT system
    # with the inequality active, and verify that its multiplier is positive.
    H = design.T @ np.diag(weights) @ design + Q
    linear = design.T @ (weights * response) + b
    constraints = np.vstack([C, E])
    kkt = np.block([[H, constraints.T], [constraints, np.zeros((2, 2))]])
    reference = np.linalg.solve(kkt, np.r_[linear, c, e])
    assert reference[2] > 0

    A = np.vstack([design, [17.0, -23.0]]) if padded else design
    problem = GLQP(f=GaussianNLL(response, weights), A=csc_array(A),
                   Q=csc_array(Q), b=b, C=csc_array(C), c=c, E=csc_array(E), e=e)
    x, result = problem.solve(verbose=False)
    assert_allclose(x, reference[:2], atol=2e-7, rtol=0)
    residual = design @ x - response
    gradient = design.T @ (weights * residual) + Q @ x - b
    assert_original_kkt(x, result, gradient, C, c, E, e)
    expected_objective = 0.5 * np.dot(weights, residual**2) + 0.5 * x @ Q @ x - b @ x
    assert result.primal == pytest.approx(expected_objective, abs=1e-10)


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("active_bound", [False, True])
def test_constrained_poisson_matches_independent_scalar_solution(padded, active_bound):
    design = np.array([[1.0, -0.2], [0.4, 0.8], [1.5, 0.3]])
    response, weights = np.array([0.0, 3.0, 2.0]), np.array([0.7, 1.3, 0.2])
    Q, b = np.diag([0.2, 0.4]), np.array([0.1, -0.2])
    E, e = np.array([[1.0, 1.0]]), np.array([0.7])

    def independent_gradient(x):
        return design.T @ (weights * (np.exp(design @ x) - response)) + Q @ x - b

    # Eliminate the equality x[1] = 0.7 - x[0]. Strict convexity makes
    # the scalar stationary point unique; an upper bound clips this point.
    def derivative(t):
        gradient = independent_gradient(np.array([t, e[0] - t]))
        return gradient[0] - gradient[1]

    unconstrained = brentq(derivative, -3.0, 3.0, xtol=1e-14)
    upper = unconstrained + (-0.25 if active_bound else 0.25)
    reference_t = min(unconstrained, upper)
    reference = np.array([reference_t, e[0] - reference_t])
    C, c = np.array([[1.0, 0.0]]), np.array([upper])
    A = np.vstack([design, [13.0, -7.0]]) if padded else design
    problem = GLQP(f=PoissonNLL(response, weights), A=csc_array(A),
                   Q=csc_array(Q), b=b, C=csc_array(C), c=c, E=csc_array(E), e=e)
    x, result = problem.solve(x0=np.array([3.0, -2.3]), verbose=False)
    assert_allclose(x, reference, atol=2e-7, rtol=0)
    assert_original_kkt(x, result, independent_gradient(x), C, c, E, e)
    z = design @ x
    expected_objective = np.dot(weights, np.exp(z) - response * z) + 0.5 * x @ Q @ x - b @ x
    assert result.primal == pytest.approx(expected_objective, abs=1e-10)
