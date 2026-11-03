"""Independent scalar QP/barrier proposals; no solver or benchmark runs."""
from decimal import Decimal, localcontext

import numpy as np
import pytest

from glqp.qp_linesearch import QPScalarProposer


def D(value):
    return Decimal(str(value))


def decimal_ray(step, s, ds, mu, xQdx, dxQdx, bdx):
    with localcontext() as context:
        context.prec = 100
        t, target = D(step), D(mu)
        value = t*(D(xQdx)-D(bdx))+t*t*D(dxQdx)/2
        slope = D(xQdx)-D(bdx)+t*D(dxQdx)
        curvature = D(dxQdx)
        for si, di in zip(s, ds):
            slack = D(si)+t*D(di)
            value -= target*slack.ln()
            slope -= target*D(di)/slack
            curvature += target*(D(di)/slack)**2
        return value, slope, curvature


def proposer(s=(), ds=(), mu=0., xQdx=-1., dxQdx=4., bdx=0.,
             merit0=0., slope0=-1.):
    return QPScalarProposer(s, ds, mu, xQdx, dxQdx, bdx, merit0, slope0)


@pytest.mark.parametrize('step', [.8, .9, 1.])
@pytest.mark.parametrize('mu', [0., .3, 2.])
def test_mixed_quadratic_barrier_newton_proposal_matches_decimal(step, mu):
    s, ds = np.array([2., .8]), np.array([.2, -.1])
    args = dict(s=s, ds=ds, mu=mu, xQdx=-.4, dxQdx=10., bdx=2.)
    merit0, slope0, _ = decimal_ray(0., **args)
    merit, slope, curvature = decimal_ray(step, **args)
    model = proposer(**args, merit0=float(merit0), slope0=float(slope0))
    assert model.counters['derivative_evaluations'] == 0
    candidate, kind = model.propose(step, s+step*ds, float(merit))
    expected = float(D(step)-slope/curvature)
    assert .1*step <= expected <= .5*step
    assert kind == 'newton'
    assert candidate == pytest.approx(expected, rel=3e-15)
    assert model.counters == dict(derivative_evaluations=1, newton_steps=1,
                                  interpolation_steps=0, contraction_steps=0)


@pytest.mark.parametrize('q', [10., 100., 1e6])
def test_exact_quadratic_interpolation_clips_small_newton_step(q):
    model = proposer(xQdx=-.5, dxQdx=q, slope0=-.5)
    candidate, kind = model.propose(1., [], -.5+.5*q)
    assert kind == 'interpolation' and candidate == .1


def test_barrier_only_zero_quadratic_curvature_still_has_valid_model():
    # phi(t)=-3*t-log(1-.75*t); phi'(0)=-2.25 and phi''(t)>0.
    s, ds = np.array([1.]), np.array([-.75])
    model = proposer(s=s, ds=ds, mu=1., xQdx=-3., dxQdx=0., slope0=-2.25)
    merit, slope, curvature = decimal_ray(1., s, ds, 1., -3., 0., 0.)
    candidate, kind = model.propose(1., s+ds, float(merit))
    # Newton at t=1 gives t=1, outside the safeguard; the value-based fit
    # is clipped to .5. A zero Q block does not disable barrier curvature.
    assert curvature > 0 and slope == 0
    assert kind == 'interpolation' and candidate == .5
    assert model.counters['derivative_evaluations'] == 1


def test_completely_linear_ray_has_zero_curvature_and_contracts():
    model = proposer(dxQdx=0.)
    assert model.propose(1., [], 2.) == (.5, 'contraction')
    assert model.counters['derivative_evaluations'] == 1


def test_tiny_slacks_and_large_barrier_ratios_match_decimal():
    s, ds = np.array([1e-300]), np.array([-.8e-300])
    args = dict(s=s, ds=ds, mu=.5, xQdx=-.5, dxQdx=6., bdx=0.)
    merit0, slope0, _ = decimal_ray(0., **args)
    merit, slope, curvature = decimal_ray(1., **args)
    model = proposer(**args, merit0=float(merit0), slope0=float(slope0))
    candidate, kind = model.propose(1., s+ds, float(merit))
    expected = float(D(1)-slope/curvature)
    assert kind == 'newton'
    assert candidate == pytest.approx(expected, rel=3e-15)


def test_barrier_is_weighted_before_squaring_huge_finite_ratio():
    # The exact trial ratio is about1e200: its square overflows, whereas
    # mu=1e-300 gives finite curvature about1e100. A tiny positive step
    # keeps trial_s=s+t*ds consistent with the supplied ray.
    step = 1e-201
    s, ds = np.array([1e-200]), np.array([1.])
    model = proposer(s=s, ds=ds, mu=1e-300, xQdx=-1e-101, dxQdx=0.,
                     merit0=0., slope0=-1.1e-100)
    _, kind = model.propose(step, s+step*ds, 1.)
    # A valid finite curvature reaches interpolation; an overflowing
    # ratio**2 implementation would instead take the contraction branch.
    assert kind == 'interpolation'
    assert model.counters['interpolation_steps'] == 1


def test_zero_mu_never_uses_an_overflowing_slack_ratio():
    step = 1e-320
    s, ds = np.array([1e-300]), np.array([1e300])
    model = proposer(s=s, ds=ds, mu=0., xQdx=-2e-321, dxQdx=1.,
                     slope0=-2e-321)
    candidate, kind = model.propose(step, s+step*ds, 0.)
    assert kind == 'newton'
    assert candidate == pytest.approx(2e-321, abs=np.nextafter(0., 1.))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_trial_merit_contracts_without_derivatives(bad):
    model = proposer()
    assert model.propose(1., [], bad) == (.5, 'contraction')
    assert model.counters['derivative_evaluations'] == 0


@pytest.mark.parametrize('slack', [[0.], [-1.], [np.inf], [np.nan]])
def test_invalid_trial_slack_contracts_without_derivatives(slack):
    model = proposer(s=[1.], ds=[-.5], mu=.1)
    assert model.propose(1., slack, 1.) == (.5, 'contraction')
    assert model.counters['derivative_evaluations'] == 0


@pytest.mark.parametrize('field,bad', [('merit0', np.inf), ('slope0', np.nan),
    ('slope0', 0.), ('slope0', 1.), ('dxQdx', -1.), ('dxQdx', np.inf),
    ('xQdx', np.nan), ('bdx', np.inf)])
def test_ineligible_model_falls_back_without_derivatives(field, bad):
    model = proposer(**{field: bad})
    assert model.propose(1., [], 1.) == (.5, 'contraction')
    assert model.counters['derivative_evaluations'] == 0


def test_nonfinite_directional_curvature_falls_back_after_evaluation():
    model = proposer(s=[1e-300], ds=[1e300], mu=1.)
    assert model.propose(1e-320, [1e-20], 1.) == (5e-321, 'contraction')
    assert model.counters['derivative_evaluations'] == 1


@pytest.mark.parametrize('step', [0., -1., np.nan, np.inf])
def test_invalid_step_returns_termination_signal(step):
    model = proposer()
    assert model.propose(step, [], 1.) == (0., 'contraction')
    assert model.counters['derivative_evaluations'] == 0


def test_interpolation_denominator_zero_contracts_without_division_error():
    model = proposer(xQdx=-3., dxQdx=1., slope0=-1.)
    # Newton lies beyond the upper bound; merit-merit0 equals step*slope0.
    assert model.propose(1., [], -1.) == (.5, 'contraction')


def test_underflowing_interpolation_uses_representable_half_step():
    smallest = np.nextafter(0., 1.)
    step = 2*smallest
    model = proposer(xQdx=-.02, dxQdx=1., slope0=-.02)
    # Force Newton outside its interval and interpolation's clipped fraction
    # to underflow; the final half step remains representable.
    candidate, kind = model.propose(step, [], step)
    assert candidate == smallest and kind == 'contraction'


def test_proposals_never_mutate_borrowed_arrays():
    s, ds = np.array([2., 1.]), np.array([-.1, .2])
    trial_s = s+ds
    originals = [value.copy() for value in (s, ds, trial_s)]
    model = proposer(s=s, ds=ds, mu=.1)
    model.propose(1., trial_s, 1.)
    for actual, expected in zip((s, ds, trial_s), originals):
        np.testing.assert_array_equal(actual, expected)


def test_malformed_vector_shapes_remain_visible():
    with pytest.raises(ValueError, match='matching'):
        proposer(s=[1.], ds=[1., 2.])
    with pytest.raises(ValueError, match='trial slack'):
        proposer(s=[1.], ds=[-.1]).propose(1., [], 1.)


@pytest.mark.parametrize('mu', [-1., np.nan, np.inf])
def test_invalid_barrier_target_rejected(mu):
    with pytest.raises(ValueError, match='barrier target'):
        proposer(mu=mu)
