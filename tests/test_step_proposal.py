"""Safeguarded proposal choices and their floating-point boundary policies."""
import numpy as np
import pytest

from glqp.step_proposal import safeguarded_step


@pytest.mark.parametrize('require_positive_step', [False, True])
def test_newton_precedes_value_interpolation(require_positive_step):
    # Newton selects .25; fitting the rejected value would instead select .1.
    assert safeguarded_step(1., 10., 0., -1., 3., 4.,
                            require_positive_step=require_positive_step) == (.25, 'newton')


@pytest.mark.parametrize('merit,expected', [(10., .1), (1., .25), (-.5, .5)])
@pytest.mark.parametrize('require_positive_step', [False, True])
def test_value_interpolation_and_safeguards(merit, expected, require_positive_step):
    # The derivative's Newton step is 1, outside the permitted interval.
    assert safeguarded_step(1., merit, 0., -1., 0., 1.,
                            require_positive_step=require_positive_step) == (
                                expected, 'interpolation')


@pytest.mark.parametrize('slope,curvature', [(np.nan, 1.), (np.inf, 1.),
    (0., 0.), (0., -1.), (0., np.nan), (0., np.inf)])
def test_unusable_derivatives_disable_interpolation_too(slope, curvature):
    assert safeguarded_step(1., 1., 0., -1., slope, curvature) == (.5, 'contraction')


@pytest.mark.parametrize('merit', [-1., -2., np.nan, np.inf])
@pytest.mark.parametrize('require_positive_step', [False, True])
def test_unusable_interpolation_falls_back_to_halving(merit, require_positive_step):
    assert safeguarded_step(1., merit, 0., -1., 0., 1.,
                            require_positive_step=require_positive_step) == (.5, 'contraction')


@pytest.mark.parametrize('step', [0., -1., np.nan, np.inf])
def test_invalid_step_returns_termination_signal(step):
    assert safeguarded_step(step, 1., 0., -1., 0., 1.) == (0., 'contraction')


def test_qp_policy_replaces_zero_interpolation_with_representable_half_step():
    smallest = np.nextafter(0., 1.)
    step = 2*smallest
    args = (step, step, 0., -.02, -.02, 1.)
    assert safeguarded_step(*args) == (0., 'interpolation')
    assert safeguarded_step(*args, require_positive_step=True) == (smallest, 'contraction')


def test_zero_newton_proposal_retains_each_policy_underflow_reason():
    smallest = np.nextafter(0., 1.)
    args = (smallest, 1., 0., -1., smallest, 1.)
    assert safeguarded_step(*args) == (0., 'newton')
    assert safeguarded_step(*args, require_positive_step=True) == (0., 'contraction')
