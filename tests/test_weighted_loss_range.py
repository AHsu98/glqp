"""Weighted loss range rescue checked against independent Decimal formulas."""
from decimal import Decimal, localcontext

import numpy as np
import pytest

from glqp.linesearch import CachedGLMRay
from glqp.obj import GaussianNLL, PoissonNLL


def reference(kind, z, y, w):
    # Exact binary64 inputs, rather than their shortened printed decimals.
    with localcontext() as context:
        context.prec = 100
        value, gradient, curvature = Decimal(0), [], []
        for zi, yi, wi in zip(z, y, w):
            eta, response, weight = map(lambda x: Decimal.from_float(float(x)), (zi, yi, wi))
            if not weight:
                v = g = h = Decimal(0)
            elif kind is GaussianNLL:
                v = weight*(eta-response)**2/2
                g, h = weight*(eta-response), weight
            else:
                rate = eta.exp()
                v, g, h = weight*(rate-response*eta), weight*(rate-response), weight*rate
            value += v
            gradient.append(float(g))
            curvature.append(float(h))
        return float(value), np.array(gradient), np.array(curvature)


RANGE_CASES = [
    pytest.param(PoissonNLL, [710.], [0.], [1e-100], id='poisson-rate-overflow'),
    pytest.param(PoissonNLL, [1400.], [0.], [1e-320], id='poisson-extreme-rate-overflow'),
    pytest.param(PoissonNLL, [-1000.], [0.], [1e300], id='poisson-rate-underflow'),
    pytest.param(PoissonNLL, [-1400.], [0.], [1e308], id='poisson-extreme-rate-underflow'),
    pytest.param(PoissonNLL, [-720.], [0.], [1e300], id='poisson-subnormal-rate'),
    pytest.param(PoissonNLL, [710.], [1e308], [1e-300], id='poisson-response-overflow'),
    pytest.param(PoissonNLL, [710.], [1e308], [-1e-300], id='poisson-signed-weight'),
    pytest.param(PoissonNLL, [-1000.3], [5e-324], [1e300], id='poisson-tiny-response'),
    pytest.param(GaussianNLL, [1e200], [0.], [1e-300], id='gaussian-square-overflow'),
    pytest.param(GaussianNLL, [1e-200], [0.], [1e300], id='gaussian-square-underflow'),
    pytest.param(GaussianNLL, [1e-160], [0.], [1e300], id='gaussian-subnormal-square'),
    pytest.param(GaussianNLL, [1e308], [-1e308], [1e-310], id='gaussian-residual-overflow'),
    pytest.param(GaussianNLL, [1.2e154], [0.], [2.], id='gaussian-half-after-overflow'),
    pytest.param(GaussianNLL, [1e200], [0.], [-1e-300], id='gaussian-signed-weight'),
    pytest.param(GaussianNLL, [1e200, 2.], [0., 1.], [0., 3.], id='gaussian-zero-weight'),
    pytest.param(GaussianNLL, [1e308, 2.], [-1e308, 1.], [0., 3.], id='gaussian-zero-residual-overflow'),
]


@pytest.mark.parametrize('kind,z,y,w', RANGE_CASES)
def test_public_losses_match_decimal_over_wide_weighted_range(kind, z, y, w):
    loss, z = kind(y, w), np.array(z)
    expected = reference(kind, z, y, w)
    with np.errstate(all='raise'):
        actual = loss(z), loss.d1f(z), loss.d2f(z)
    for actual_part, expected_part in zip(actual, expected):
        np.testing.assert_allclose(actual_part, expected_part, rtol=2e-14, atol=0)


@pytest.mark.parametrize('kind,z,y,w', RANGE_CASES)
def test_cached_ray_shares_public_rescue_and_preserves_derivative_padding(kind, z, y, w):
    loss, z = kind(y, w), np.array(z)
    # Unit directions test first/second derivatives without adding an unrelated
    # direction-product overflow. Ignore huge trailing padding as usual.
    direction = np.r_[np.ones(len(z)), 1e200]
    ray = CachedGLMRay(loss, np.r_[z, 1e200], direction, np.empty(0), np.empty(0),
                       0., 0., 0., 0., 0., 0.)
    trial = ray.evaluate(0.)
    expected = reference(kind, z, y, w)
    assert trial.merit == loss(z)
    slope, curvature = ray.derivatives(trial)
    np.testing.assert_allclose([slope, curvature],
                              [sum(expected[1]), sum(expected[2])], rtol=2e-14, atol=0)
    np.testing.assert_array_equal(loss.d1f(np.r_[z, 1e200]), np.r_[loss.d1f(z), 0.])
    np.testing.assert_array_equal(loss.d2f(np.r_[z, 1e200]), np.r_[loss.d2f(z), 0.])


@pytest.mark.parametrize('kind', [GaussianNLL, PoissonNLL])
def test_ordinary_values_and_derivatives_keep_exact_legacy_arithmetic(kind):
    rng = np.random.default_rng(20261019)
    for _ in range(40):
        z, y, w = rng.normal(size=97), rng.uniform(0, 5, 97), rng.uniform(.01, 3, 97)
        if kind is GaussianNLL:
            expected = .5*np.dot(w, (y-z)**2), w*(z-y), w
        else:
            rate = np.exp(z)
            expected = np.dot(w, rate-y*z), w*(rate-y), w*rate
        loss = kind(y, w)
        for actual, baseline in zip((loss(z), loss.d1f(z), loss.d2f(z)), expected):
            np.testing.assert_array_equal(actual, baseline)


def test_signed_gaussian_sum_preserves_small_term_after_overflowed_terms_cancel():
    # Normalizing by the largest exponent before summation would lose 0.5.
    loss = GaussianNLL([0., 0., 0.], [1., -1., 1.])
    assert loss(np.array([1e200, 1e200, 1.])) == .5


def test_signed_poisson_sum_preserves_small_term_after_overflowed_terms_cancel():
    loss = PoissonNLL([0., 0., 0.], [1., -1., 1.])
    assert loss(np.array([710., 710., 0.])) == 1.


@pytest.mark.parametrize('quantity', ['objective', 'gradient'])
def test_poisson_range_rescue_cancellation_has_componentwise_float64_accuracy(quantity):
    # Near cancellation, relative error in the difference can be large even
    # for an accurately rounded exponential. Check forward error against the
    # magnitudes of the terms, not an ill-conditioned relative gradient bound.
    z = np.nextafter(np.log(np.finfo(float).max), np.inf)
    y = np.finfo(float).max if quantity == 'gradient' else np.finfo(float).max/z
    w = 1e-100
    loss = PoissonNLL([y], [w])
    expected = reference(PoissonNLL, [z], [y], [w])
    actual = loss(np.array([z])) if quantity == 'objective' else loss.d1f(np.array([z]))[0]
    correct = expected[0] if quantity == 'objective' else expected[1][0]
    term_scale = w*y*(z if quantity == 'objective' else 1.)
    assert np.isfinite(actual)
    assert abs(actual-correct) <= 4*np.finfo(float).eps*term_scale


@pytest.mark.parametrize('z,w', [(710., 1e-100), (-1000., 1e300)])
def test_exceptional_single_observation_preserves_scalar_weight_broadcasting(z, w):
    loss = PoissonNLL([0.], w)
    expected = reference(PoissonNLL, [z], [0.], [w])
    np.testing.assert_allclose(loss(np.array([z])), expected[0], rtol=2e-14)
    np.testing.assert_allclose(loss.d1f(np.array([z])), expected[1], rtol=2e-14)
    np.testing.assert_allclose(loss.d2f(np.array([z])), expected[2], rtol=2e-14)


@pytest.mark.parametrize('z,w,value', [(1e308, 5e-324, np.inf), (-1e308, 1e308, 0.)])
def test_poisson_predictor_beyond_any_rescuable_weighted_rate(z, w, value):
    loss = PoissonNLL([0.], [w])
    assert loss(np.array([z])) == value
    np.testing.assert_array_equal(loss.d1f(np.array([z])), [value])
    np.testing.assert_array_equal(loss.d2f(np.array([z])), [value])


@pytest.mark.parametrize('kind', [GaussianNLL, PoissonNLL])
def test_true_weighted_overflow_remains_nonfinite(kind):
    loss = kind([0.], [1.])
    z = np.array([1e200 if kind is GaussianNLL else 1000.])
    assert np.isposinf(loss(z))


def test_active_nonfinite_gaussian_predictor_is_not_hidden_by_rescue():
    # Keep the existing public/cached-ray finiteness policy for active inputs.
    loss = GaussianNLL([0.], [1.])
    assert not np.isfinite(loss(np.array([np.inf])))


def test_poisson_zero_weights_and_mutation_are_respected_by_range_rescue():
    loss = PoissonNLL([0., 1e308], [0., 1e-300])
    z = np.array([np.inf, 710.])
    assert loss(z) == PoissonNLL([1e308], [1e-300])(z[1:])
    loss.w[:] = 0.
    assert loss(z) == 0.
    np.testing.assert_array_equal(loss.d1f(z), [0., 0.])
    np.testing.assert_array_equal(loss.d2f(z), [0., 0.])
