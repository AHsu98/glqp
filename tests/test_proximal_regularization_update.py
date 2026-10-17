"""Boundary-only progress continuation of independent proximal coefficients."""
from dataclasses import FrozenInstanceError

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from glqp import SolverSettings
from glqp.proximal import ProximalState


def settings(**kwargs):
    return SolverSettings(proximal_policy='centered', proximal_reg_update='progress', **kwargs)


def initial_state():
    return ProximalState(np.zeros(1), np.ones(1), np.zeros(1),
                         np.array([10.]), np.array([20.]), np.zeros(1),
                         rho_cap=.2, delta_cap=.3)


def update(state, fixed, *, x=1., y=2., stationarity=10., feasibility=20., options=None):
    return state.update(fixed, np.array([x]), np.array([y]), np.zeros(1),
                         np.array([stationarity]), np.array([feasibility]),
                         np.ones(1), np.zeros(1), .1, options or settings())


@pytest.mark.parametrize('block,x,y,expected', [
    ('primal', 10., 2., (.02, .3)),
    ('dual', 1., 11., (.2, .03)),
])
def test_five_misses_reduce_only_the_block_with_material_proximal_offset(block, x, y, expected):
    state = initial_state()
    fixed = state.freeze(*state.select_regularization(.2, .3))
    frozen_residual = fixed.residuals(np.array([x]), np.array([y]), np.zeros(1),
                                      np.array([10.]), np.array([20.]), np.zeros(1))
    for step in range(5):
        record = update(state, fixed, x=x, y=y)['regularization_update']
        assert record[block]['reduced'] is (step == 4)
        assert record[block]['misses_observed'] == step+1
        if step < 4:
            assert (state.rho_cap, state.delta_cap) == (.2, .3)
    assert_allclose((state.rho_cap, state.delta_cap), expected, rtol=1e-15)
    other = 'dual' if block == 'primal' else 'primal'
    assert not record[other]['reduced']
    assert record[block]['misses_after'] == 0
    assert record[other]['misses_after'] == 5
    assert record[block]['reason'] == 'stalled_proximal_offset'
    # A completed step changes only the next subproblem. The old pair's matrix,
    # residual map, and centers are still the same immutable frozen model.
    assert fixed.rho == .2 and fixed.delta == .3 and fixed.generation == 0
    assert state.generation == 1
    for old, current in zip(frozen_residual,
        fixed.residuals(np.array([x]), np.array([y]), np.zeros(1),
                         np.array([10.]), np.array([20.]), np.zeros(1))):
        assert_array_equal(old, current)
    following = state.freeze(*state.select_regularization(.2, .3))
    assert_allclose((following.rho, following.delta), expected, rtol=1e-15)
    assert_array_equal(following.x_center, fixed.x_center)
    assert_array_equal(following.y_center, fixed.y_center)
    with pytest.raises(FrozenInstanceError):
        fixed.delta = following.delta


def test_stalled_original_residual_without_material_offset_does_not_reduce_regularization():
    state = initial_state()
    fixed = state.freeze(.2, .3)
    for _ in range(7):
        rows = update(state, fixed)['regularization_update']
    assert (state.rho_cap, state.delta_cap) == (.2, .3)
    for row in rows.values():
        assert row['misses_after'] == 7
        assert row['stalled'] and not row['offset_dominates'] and not row['triggered']
        assert row['reason'] == 'offset_below_gate'


def test_original_progress_resets_each_counter_independently():
    state = initial_state()
    fixed = state.freeze(.2, .3)
    for _ in range(4):
        update(state, fixed)
    rows = update(state, fixed, stationarity=9.)['regularization_update']
    assert rows['primal']['progress'] and rows['primal']['misses_after'] == 0
    assert not rows['dual']['progress'] and rows['dual']['misses_after'] == 5
    assert state.primal_progress_misses == 0 and state.dual_progress_misses == 5
    assert (state.rho_cap, state.delta_cap) == (.2, .3)


def test_inner_accuracy_reduces_both_caps_without_stall_or_offset_dominance():
    state = initial_state()
    fixed = state.freeze(.2, .3)
    record = state.update(fixed, np.zeros(1), np.ones(1), np.zeros(1),
                           np.zeros(1), np.zeros(1), np.ones(1), np.zeros(1), 1., settings())
    rows = record['regularization_update']
    assert record['inner_converged']
    assert_allclose((state.rho_cap, state.delta_cap), (.02, .03), rtol=1e-15)
    for row in rows.values():
        assert not row['offset_dominates']
        assert not row['stalled']
        assert row['reason'] == 'inner_accuracy'
        assert row['misses_after'] == 0
    assert fixed.rho == .2 and fixed.delta == .3


def test_residual_caps_cannot_reincrease_a_progress_coefficient():
    state = initial_state()
    assert state.select_regularization(.01, .03) == (.01, .03)
    assert state.select_regularization(.2, .3) == (.01, .03)
    assert state.select_regularization(.005, .3) == (.005, .03)


def test_zero_coefficient_stays_zero_without_becoming_negative():
    state = initial_state()
    fixed = state.freeze(*state.select_regularization(0., .3))
    state.update(fixed, np.zeros(1), np.ones(1), np.zeros(1),
                  np.zeros(1), np.zeros(1), np.ones(1), np.zeros(1), 1., settings())
    assert state.rho_cap == 0.
    assert state.delta_cap == pytest.approx(.03)


def test_residual_policy_preserves_caps_and_counter_state():
    state = initial_state()
    fixed = state.freeze(.2, .3)
    options = SolverSettings(proximal_policy='centered')
    for _ in range(7):
        record = update(state, fixed, x=10., y=11., options=options)
        assert 'regularization_update' not in record
    assert (state.rho_cap, state.delta_cap) == (.2, .3)
    assert state.primal_progress_misses == state.dual_progress_misses == 0


@pytest.mark.parametrize('value', [None, True, 0, 'bad'])
def test_invalid_update_policy_is_rejected(value):
    with pytest.raises(ValueError, match='proximal_reg_update'):
        SolverSettings(proximal_policy='centered', proximal_reg_update=value)


@pytest.mark.parametrize('policy', ['legacy', 'no_slack'])
def test_progress_updates_require_a_persistent_centered_model(policy):
    with pytest.raises(ValueError, match='require the centered'):
        SolverSettings(proximal_policy=policy, proximal_reg_update='progress')
