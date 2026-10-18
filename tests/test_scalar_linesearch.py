"""Independent scalar derivatives and bounded globalization safeguards."""
from decimal import Decimal, localcontext

import numpy as np
import pytest

from glqp.linesearch import CachedGLMRay, search_scalar_ray
from glqp.obj import GaussianNLL, LogisticNLL, PoissonNLL


def make_ray(loss, z, dz, **overrides):
    options = dict(s=np.empty(0), ds=np.empty(0), mu=0., xQx=0.,
                   xQdx=0., dxQdx=0., bx=0., bdx=0.)
    options.update(overrides)
    return CachedGLMRay(loss, np.asarray(z), np.asarray(dz), **options)


def search(ray, *, merit0=None, slope0=None, **overrides):
    if merit0 is None or slope0 is None:
        base = ray.evaluate(0.)
        merit0 = base.merit if merit0 is None else merit0
        slope0 = ray.derivatives(base)[0] if slope0 is None else slope0
    controls = dict(tmax=1., merit0=merit0, slope0=slope0, armijo_param=.005,
                    additive_eps=1e-8, max_trials=100, record_history=True)
    controls.update(overrides)
    return search_scalar_ray(ray, **controls)


def D(value):
    return Decimal(str(value))


def decimal_ray(loss, z, dz, step, *, s=(), ds=(), mu=0., xQx=0.,
                xQdx=0., dxQdx=0., bx=0., bdx=0.):
    """80-digit scalar formulas independent of production implementation."""
    with localcontext() as context:
        context.prec = 80
        t = D(step)
        value = slope = second = D(0)
        for zi, di, yi, wi in zip(z, dz, loss.y, np.broadcast_to(loss.w, loss.y.shape)):
            eta, direction, y, weight = D(zi)+t*D(di), D(di), D(yi), D(wi)
            if isinstance(loss, GaussianNLL):
                v, g, h = (eta-y)**2/2, eta-y, D(1)
            elif isinstance(loss, LogisticNLL):
                label = 2*y-1
                tail = (-abs(eta)).exp()
                v = (1+tail).ln()+((1-y)*eta if eta >= 0 else -y*eta)
                probability = (-label*eta).exp()
                g = -label*probability/(1+probability)
                h = tail/(1+tail)**2
            else:
                rate = eta.exp()
                v, g, h = rate-y*eta, rate-y, rate
            value += weight*v
            slope += weight*g*direction
            second += weight*h*direction*direction
        value += (D(xQx)+2*t*D(xQdx)+t*t*D(dxQdx))/2-D(bx)-t*D(bdx)
        slope += D(xQdx)+t*D(dxQdx)-D(bdx)
        second += D(dxQdx)
        for si, di in zip(s, ds):
            slack = D(si)+t*D(di)
            value -= D(mu)*slack.ln()
            slope -= D(mu)*D(di)/slack
            second += D(mu)*(D(di)/slack)**2
        return np.array([float(value), float(slope), float(second)])


@pytest.mark.parametrize('loss', [GaussianNLL([.5, -.7], [2., .3]),
                                  LogisticNLL([1., 0.], [2., .3]),
                                  PoissonNLL([5., 2.], [2., .3])])
@pytest.mark.parametrize('step', [0., .37, .98])
def test_mixed_ray_matches_independent_decimal_derivatives(loss, step):
    z, dz = np.array([.3, -.4]), np.array([.8, -1.1])
    kw = dict(s=np.array([.7, 2.1]), ds=np.array([-.2, .3]), mu=.4,
              xQx=1.3, xQdx=-.5, dxQdx=.7, bx=.2, bdx=-.1)
    ray = make_ray(loss, z, dz, **kw)
    point = ray.evaluate(step)
    actual = np.array([point.merit, *ray.derivatives(point)])
    expected = decimal_ray(loss, z, dz, step, **kw)
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=1e-15)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_logistic_weighted_tail_ray_preserves_sign_symmetry(sign):
    z = sign*np.array([40., 700.])
    dz = sign*np.array([.1, .3])
    loss = LogisticNLL(np.full(2, (1+sign)/2), [1e20, 1e300])
    ray = make_ray(loss, z, dz)
    point = ray.evaluate(.25)
    actual = np.array([point.merit, *ray.derivatives(point)])
    # At z=700 the Decimal softplus needs more than 300 digits to retain its
    # tiny unweighted tail before multiplying by 1e300.
    expected_first = decimal_ray(LogisticNLL(loss.y[:1], loss.w[:1]), z[:1], dz[:1], .25)
    eta = abs(float(z[1]+.25*dz[1]))
    weighted_tail = np.exp(np.log(1e300)-eta)
    expected = expected_first+np.array([weighted_tail, -.3*weighted_tail, .09*weighted_tail])
    np.testing.assert_allclose(actual, expected, rtol=6e-14, atol=1e-15)
    assert actual[0] > 0 and actual[1] < 0 and actual[2] > 0


def test_poisson_curvature_does_not_square_huge_direction_first():
    loss = PoissonNLL([0.], [1.])
    ray = make_ray(loss, [-700.], [1e200])
    point = ray.evaluate(0.)
    slope, curvature = ray.derivatives(point)
    with localcontext() as context:
        context.prec = 80
        rate = D(-700).exp()
        expected = float(rate*D('1e200')), float(rate*D('1e200')**2)
    assert np.isfinite(curvature)
    np.testing.assert_allclose([slope, curvature], expected, rtol=3e-15)


@pytest.mark.parametrize('step', [0., .99])
def test_barrier_derivatives_use_slack_ratios_before_scaling(step):
    loss = GaussianNLL([0.], [0.])
    kw = dict(s=np.array([1e-300]), ds=np.array([-.99e-300]), mu=2.)
    ray = make_ray(loss, [0.], [0.], **kw)
    point = ray.evaluate(step)
    expected = decimal_ray(loss, [0.], [0.], step, **kw)
    np.testing.assert_allclose([point.merit, *ray.derivatives(point)], expected,
                               rtol=3e-14, atol=1e-12)


def test_zero_weight_poisson_never_evaluates_unused_exponential():
    loss = PoissonNLL([0., 3.], [0., 2.])
    ray = make_ray(loss, [10000., 1.], [1e300, .2])
    point = ray.evaluate(.5)
    expected = decimal_ray(PoissonNLL([3.], [2.]), [1.], [.2], .5)
    np.testing.assert_allclose([point.merit, *ray.derivatives(point)], expected,
                               rtol=2e-15, atol=2e-16)


class CountingQuadratic:
    def __init__(self):
        self.calls = dict(value=0, gradient=0, curvature=0)

    def __call__(self, z):
        self.calls['value'] += 1
        return .5*np.dot(z, z)

    def d1f(self, z):
        self.calls['gradient'] += 1
        return z.copy()

    def d2f(self, z):
        self.calls['curvature'] += 1
        return np.ones_like(z)


def test_full_step_fast_path_does_not_request_derivatives():
    loss = CountingQuadratic()
    ray = make_ray(loss, [1.], [-1.])
    result = search(ray, merit0=.5, slope0=-1.)
    assert result.trial.step == 1 and result.status == 'armijo'
    assert loss.calls == dict(value=1, gradient=0, curvature=0)
    assert result.counters['trial_count'] == result.counters['value_evaluations'] == 1
    assert result.counters['derivative_evaluations'] == 0


def test_newton_proposal_finds_quadratic_ray_minimum():
    ray = make_ray(GaussianNLL([0.], [1.]), [1.], [-4.])
    result = search(ray)
    assert result.trial.step == .25
    assert result.counters['newton_steps'] == 1
    assert result.counters['trial_count'] == 2


@pytest.mark.parametrize('count', [1e4, 1e6, 1e10])
def test_poisson_log_growth_proposal_skips_many_nonfinite_trials(count):
    # The z=0 Newton direction of exp(z)-count*z is count-1. Full t=1
    # overflows, whereas the exact ray minimizer is log(count)/(count-1).
    direction = count-1
    ray = make_ray(PoissonNLL([count], [1.]), [0.], [direction])
    result = search(ray, merit0=1., slope0=-direction*direction)
    assert result.status == 'armijo'
    assert result.counters['trial_count'] <= 3
    assert result.counters['poisson_model_steps'] == 1
    assert result.counters['nonfinite_count'] == 1
    np.testing.assert_allclose(result.trial.step, np.log(count)/direction, rtol=3e-15)


def test_poisson_growth_proposal_includes_barrier_and_is_still_checked():
    count = 1e6
    ray = make_ray(PoissonNLL([count], [1.]), [0.], [count-1],
                   s=np.array([1.]), ds=np.array([-.99]), mu=.1)
    base = ray.evaluate(0.)
    slope = ray.derivatives(base)[0]
    result = search(ray, merit0=base.merit, slope0=slope)
    assert result.status == 'armijo' and result.trial.step > 0
    assert np.all(result.trial.slack > 0)
    assert result.trial.merit < base.merit+.005*result.trial.step*slope+1e-8
    assert result.counters['trial_count'] <= 4


def test_domain_failures_never_trigger_custom_derivatives():
    class Restricted(CountingQuadratic):
        def __call__(self, z):
            self.calls['value'] += 1
            return np.inf if np.any(z <= 0) else float((z[0]+.5)**2)

        def d1f(self, z):
            raise AssertionError('No derivative needed at rejected nonfinite points')

        def d2f(self, z):
            raise AssertionError('No derivative needed at rejected nonfinite points')

    loss = Restricted()
    result = search(make_ray(loss, [1.], [-2.]), merit0=2.25, slope0=-6.)
    assert result.trial.step == .25
    assert result.counters['nonfinite_count'] == 2
    assert result.counters['derivative_evaluations'] == 0


def test_nonfinite_derivatives_fall_back_to_contraction():
    class BadDerivatives:
        def __call__(self, z):
            return float(np.exp(z[0])-100*z[0])

        def d1f(self, z):
            return np.full_like(z, np.nan)

        def d2f(self, z):
            return np.full_like(z, np.nan)

    result = search(make_ray(BadDerivatives(), [0.], [100.]), merit0=1., slope0=-9900.)
    assert result.status == 'armijo'
    assert result.counters['newton_steps'] == result.counters['interpolation_steps'] == 0
    assert result.counters['contraction_steps'] == result.counters['trial_count']-1


@pytest.mark.parametrize('weights,quadratic', [([-1.], 10.), ([1.], -1.)])
def test_known_negative_curvature_uses_safe_contraction(weights, quadratic):
    dz = [1.] if weights[0] < 0 else [-3.]
    ray = make_ray(GaussianNLL([0.], weights), [1.], dz, dxQdx=quadratic)
    result = search(ray)
    assert result.status == 'armijo'
    assert result.counters['trial_count'] > 1
    assert result.counters['derivative_evaluations'] == 0
    assert result.counters['contraction_steps'] == result.counters['trial_count']-1


def test_caller_domain_predicate_precedes_loss_evaluation():
    ray = make_ray(GaussianNLL([0.], [1.]), [1.], [-1.],
                   s=np.array([1.]), ds=np.array([-3.]))
    result = search(ray, merit0=.5, slope0=-1., trial_is_valid=lambda t: t <= .3)
    assert result.trial.step == .25
    assert result.counters['invalid_trial_count'] == 2
    assert result.counters['value_evaluations'] == 1


def test_no_budget_or_invalid_direction_returns_explicit_status_without_callbacks():
    for kwargs, status in [(dict(max_trials=0), 'trial_limit'),
                           (dict(slope0=0.), 'not_descent'),
                           (dict(slope0=np.nan), 'not_descent'),
                           (dict(tmax=0.), 'invalid_step'),
                           (dict(merit0=np.inf), 'invalid_base_merit')]:
        loss = CountingQuadratic()
        ray = make_ray(loss, [1.], [-1.])
        options = dict(merit0=.5, slope0=-1.)
        options.update(kwargs)
        result = search(ray, **options)
        assert result.status == status and result.trial is None
        assert result.counters['value_evaluations'] == result.counters['trial_count'] == 0
        assert loss.calls == dict(value=0, gradient=0, curvature=0)


def test_trial_limit_preserves_every_failure_and_optional_history():
    class Infinite(CountingQuadratic):
        def __call__(self, z):
            return np.inf

    for history in (True, False):
        result = search(make_ray(Infinite(), [1.], [-1.]), merit0=.5, slope0=-1.,
                        max_trials=3, record_history=history)
        assert result.status == 'trial_limit' and result.trial is None
        assert result.counters['trial_count'] == result.counters['nonfinite_count'] == 3
        if history:
            assert [row['step'] for row in result.history] == [1., .5, .25]
        else:
            assert result.history is None


def test_derivative_cache_and_input_immutability():
    z, dz, s, ds = np.array([1.]), np.array([-3.]), np.array([1.]), np.array([-.1])
    originals = [a.copy() for a in (z, dz, s, ds)]
    ray = make_ray(GaussianNLL([0.], [1.]), z, dz, s=s, ds=ds, mu=.1)
    point = ray.evaluate(.3)
    assert ray.derivatives(point) == ray.derivatives(point)
    assert ray.derivative_evaluations == 1
    ray.evaluate(.4)
    with pytest.raises(ValueError, match='latest'):
        ray.derivatives(point)
    for actual, original in zip((z, dz, s, ds), originals):
        np.testing.assert_array_equal(actual, original)


def test_instance_callback_override_is_not_bypassed():
    loss = GaussianNLL([0.], [1.])
    loss.f = lambda z: 17.
    point = make_ray(loss, [0.], [0.]).evaluate(0.)
    assert point.merit == 17.


@pytest.mark.parametrize('loss', [GaussianNLL([1., -.2], [2., .3]),
                                  LogisticNLL([1., 0.], [2., .3]),
                                  PoissonNLL([1., 3.], [2., .3])])
def test_fused_kernels_match_public_losses_and_rebuilt_rays_see_mutable_weights(loss):
    z, dz = np.array([.3, -.4, 700.]), np.array([.2, -.7, 20.])
    for weights in ([2., .3], [0., 5.]):
        loss.w[:] = weights
        ray = make_ray(loss, z, dz)
        point = ray.evaluate(.4)
        assert point.merit == loss(point.predictor)
        expected_slope = loss.d1f(point.predictor)@dz
        curvature = loss.d2f(point.predictor)
        expected_curvature = np.dot(curvature*dz, dz)
        np.testing.assert_allclose(ray.derivatives(point), [expected_slope, expected_curvature],
                                   rtol=3e-15, atol=1e-16)


def test_fused_logistic_weighted_underflow_rescue_matches_public_loss():
    loss = LogisticNLL([1., 0.], [1e300, 1e300])
    ray = make_ray(loss, [1000., -1000.], [.2, -.3])
    point = ray.evaluate(.5)
    assert point.merit > 0 and point.merit == loss(point.predictor)
    np.testing.assert_allclose(ray.derivatives(point),
        [loss.d1f(point.predictor)@ray.dz, np.dot(loss.d2f(point.predictor)*ray.dz, ray.dz)],
        rtol=3e-15)


def test_zero_barrier_target_does_not_form_overflowing_slack_ratio():
    ray = make_ray(GaussianNLL([0.], [1.]), [1.], [-1.],
                   s=np.array([1e-300]), ds=np.array([1e300]), mu=0.)
    point = ray.evaluate(0.)
    assert ray.derivatives(point) == (-1., 1.)


def test_predictor_finiteness_respects_builtin_unused_coordinates():
    ray = make_ray(PoissonNLL([0., 2.], [0., 1.]), [0., 0.], [np.inf, 1.])
    point = ray.evaluate(1.)
    assert np.isfinite(point.merit)
    assert np.isfinite(ray.derivatives(point)).all()
    padded = make_ray(GaussianNLL([0.], [1.]), [1., np.inf], [-1., 0.])
    assert np.isfinite(padded.evaluate(1.).merit)
    custom = make_ray(CountingQuadratic(), [1., np.inf], [-1., 0.])
    assert not np.isfinite(custom.evaluate(1.).merit)
    assert custom.loss.calls['value'] == 0


def test_full_step_does_not_scan_curvature_model_or_build_nonpoisson_mask():
    ray = make_ray(GaussianNLL([0.], [1.]), [1.], [-1.])
    assert ray._model_allowed is None and not ray._active_ready
    result = search(ray, merit0=.5, slope0=-1.)
    assert result.status == 'armijo'
    assert ray._model_allowed is None and not ray._active_ready


@pytest.mark.parametrize('loss', [GaussianNLL([1.], [2.]),
                                  LogisticNLL([1.], [2.]),
                                  PoissonNLL([1.], [2.])])
def test_padded_predictors_do_not_enter_likelihood_ray(loss):
    ray = make_ray(loss, [.3, 1e200], [.2, 1e200])
    point = ray.evaluate(.4)
    expected = decimal_ray(loss, [.3], [.2], .4)
    np.testing.assert_allclose([point.merit, *ray.derivatives(point)], expected, rtol=4e-15)


@pytest.mark.parametrize('kwargs', [dict(armijo_param=0.), dict(armijo_param=1.),
                                    dict(additive_eps=-1.), dict(max_trials=-1)])
def test_invalid_search_controls_raise(kwargs):
    with pytest.raises(ValueError, match='controls'):
        search(make_ray(GaussianNLL([0.], [1.]), [1.], [-1.]),
               merit0=.5, slope0=-1., **kwargs)
