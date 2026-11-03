"""Independent scalar references for weighted likelihood tails."""
from decimal import Decimal, localcontext

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.optimize import brentq
from scipy.sparse import csc_array

from glqp import GLQP, SolverSettings
from glqp.obj import LogisticNLL, PoissonNLL


def decimal_logistic(z, response, weight):
    """Evaluate the original formulas in sufficient precision, without SciPy."""
    with localcontext() as context:
        context.prec = 520
        z, response, weight = map(Decimal.from_float, (float(z), float(response), float(weight)))
        exponential = z.exp()
        probability = exponential / (1+exponential)
        return tuple(map(float, (
            weight*((1+exponential).ln()-response*z),
            weight*(probability-response),
            weight*probability*(1-probability),
        )))


@pytest.mark.parametrize("magnitude,weight", [
    (0., .7), (1.5, 2.3), (40., 1.), (40., 1e20),
    (400., 1e150), (740., 1e300), (1000., 1e300),
])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_weighted_logistic_tail_matches_decimal(magnitude, weight, sign):
    z = sign*magnitude
    response = float(sign > 0)
    loss = LogisticNLL([response], [weight])
    expected = decimal_logistic(z, response, weight)
    actual = (loss([z]), loss.d1f([z])[0], loss.d2f([z])[0])
    # Log-scaled rescue combines logarithms of very large/small operands.
    assert_allclose(actual, expected, rtol=3e-13, atol=0.)
    assert actual[2] > 0


@pytest.mark.parametrize("response", [0., .2, .7, 1.])
@pytest.mark.parametrize("z", [-40., -1.5, 0., 1.5, 40.])
def test_logistic_preserves_label_algebra(response, z):
    loss = LogisticNLL([response], [.7])
    expected = decimal_logistic(z, response, .7)
    assert_allclose((loss([z]), loss.d1f([z])[0], loss.d2f([z])[0]),
                    expected, rtol=2e-14, atol=0.)


def test_logistic_sign_symmetry_padding_and_ownership():
    z = np.array([-40., 400., -740., 1000., 1e5])
    response = np.array([0., 1., 1., 0.])
    weights = np.array([1e20, 1e150, 1e100, 1e100])
    loss, flipped = LogisticNLL(response, weights), LogisticNLL(1-response, weights)
    z.flags.writeable = False
    assert loss(z) == flipped(-z)
    assert_array_equal(loss.d1f(z), -flipped.d1f(-z))
    assert_array_equal(loss.d2f(z), flipped.d2f(-z))
    for derivative in (loss.d1f, loss.d2f):
        value = derivative(z)
        expected = value.copy()
        assert value[-1] == 0
        value[:] = 17.
        assert_array_equal(derivative(z), expected)
    assert_array_equal(loss.w, weights)


def test_weighted_tail_does_not_false_terminate_at_warm_start():
    # The former expit(40)-1 gradient was zero and returned x=40 as optimal.
    weight = 1e20
    problem = GLQP(f=LogisticNLL([1.], [weight]), A=csc_array([[1.]]),
                   Q=csc_array([[1.]]), b=np.array([40.]))
    x, result = problem.solve(x0=np.array([40.]), verbose=False,
                              settings=SolverSettings(tol=1e-8))

    def derivative(t):
        return t-40.-weight/(1+np.exp(t))

    reference = brentq(derivative, 40., 50., xtol=1e-14)
    assert result.termination_tag == 'optimal'
    assert len(result.history) > 0
    assert_allclose(x, [reference], rtol=0., atol=2e-9)
    assert abs(derivative(x[0])) < 1e-8
    likelihood = decimal_logistic(x[0], 1., weight)[0]
    assert result.primal == pytest.approx(likelihood+.5*x[0]**2-40*x[0], abs=1e-10)


@pytest.mark.parametrize("response", [0., 3., 1e200])
def test_poisson_zero_weight_observations_are_not_evaluated(response):
    loss = PoissonNLL([response, 2., response], [0., .5, 0.])
    z = np.array([1000., np.log(4.), -1e200, 1000.])
    with np.errstate(over='raise', invalid='raise'):
        assert loss(z) == pytest.approx(.5*(4.-2.*np.log(4.)))
        assert_array_equal(loss.d1f(z), [0., 1., 0., 0.])
        assert_array_equal(loss.d2f(z), [0., 2., 0., 0.])


def test_poisson_zero_weights_can_change_between_calls():
    loss = PoissonNLL([0.], [0.])
    with np.errstate(over='raise', invalid='raise'):
        assert loss([1000.]) == 0
        assert_array_equal(loss.d1f([1000.]), [0.])
        assert_array_equal(loss.d2f([1000.]), [0.])
    loss.w[0] = 2.
    assert loss([0.]) == 2.
    assert_array_equal(loss.d1f([0.]), [2.])
    assert_array_equal(loss.d2f([0.]), [2.])
