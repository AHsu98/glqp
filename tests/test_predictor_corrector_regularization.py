"""Separate PC regularization blocks and same-matrix centering rescue."""
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import csc_array

from glqp import GLQP, SolverSettings
from glqp.predictor_corrector import (
    compute_predictor_corrector, VerifiedLinearSystem, _bounded_correction,
)


def four_blocks(h, c, e, s, y, rho, delta, slack_shift):
    n, k, p = len(h), len(s), len(e)
    return np.block([
        [h + rho*np.eye(n), np.zeros((n, k)), c.T, e.T],
        [c, np.eye(k), -delta*np.eye(k), np.zeros((k, p))],
        [np.zeros((k, n)), np.diag(y + slack_shift), np.diag(s), np.zeros((k, p))],
        [e, np.zeros((p, k)), np.zeros((p, k)), -delta*np.eye(p)],
    ])


@pytest.mark.parametrize("shifts", [(.13, .07, .23), (.2, .03, 0.),
                                   (0., .2, 0.), (.2, 0., 0.)])
@pytest.mark.parametrize("equalities", [False, True])
def test_both_rhs_directions_match_independent_unequal_four_blocks(shifts, equalities):
    h = np.array([[4., .3], [.3, 2.]])
    c = np.array([[1., -.5], [-.2, 1.], [-1., -.3]])
    e = np.array([[1., 1.]]) if equalities else np.zeros((0, 2))
    s, y = np.array([.4, 1.1, .7]), np.array([1.3, .6, 2.])
    rx, rp, req = np.array([.7, -.2]), np.array([.1, -.3, .2]), np.full(len(e), .05)
    model = GLQP(Q=csc_array(h), C=csc_array(c), c=np.ones(3),
                 E=csc_array(e), e=np.zeros(len(e)))
    model.initialize(settings=SolverSettings(step_strategy='predictor_corrector', tol=1e-10))
    rho, delta, slack_shift = shifts
    result = compute_predictor_corrector(model, np.zeros(2), y, s, csc_array(h),
        rx, rp, req, 2., prox_reg=rho, dual_reg=delta,
        complementarity_reg=slack_shift, target_override=.4)
    matrix = four_blocks(h, c, e, s, y, *shifts)
    affine = np.linalg.solve(matrix, -np.r_[rx, rp, s*y, req])
    correction = affine[2:5]*affine[5:8]
    expected_rhs = -np.r_[rx, rp, s*y + correction - .4, req]
    expected = np.linalg.solve(matrix, expected_rhs)
    actual = np.r_[result.dx, result.ds, result.dy, result.dnu]
    assert_allclose(result.correction, correction, atol=2e-10, rtol=2e-8)
    assert_allclose(actual, expected, atol=2e-10, rtol=2e-8)
    assert_allclose(matrix @ actual, expected_rhs, atol=2e-10, rtol=2e-8)
    assert result.decision['dual_reg'] == delta
    assert result.decision['complementarity_reg'] == slack_shift
    assert 'rescue' not in result.decision
    assert result.extra_refinements == 0


def scalar_case(*, h=.001, c=.1, s=1., y=1.):
    model = GLQP(Q=csc_array([[h]]), C=csc_array([[c]]), c=np.ones(1))
    model.initialize(settings=SolverSettings(step_strategy='predictor_corrector', tol=1e-10))
    # Rescue tests deliberately exercise the unbounded corrector ablation.
    model.settings.proximal_corrector_limit = None
    args = (model, np.zeros(1), np.array([y]), np.array([s]), model.Q,
            np.ones(1), np.zeros(1), np.zeros(0), 1.)
    options = dict(prox_reg=.002, dual_reg=.003, complementarity_reg=0., target_override=.1)
    return model, args, options


def test_true_non_descent_corrector_is_replaced_by_same_target_centering(monkeypatch):
    model, args, options = scalar_case()
    calls = []
    solve = VerifiedLinearSystem.solve_verified

    def record(system, rhs):
        result = solve(system, rhs)
        calls.append((system, system.factorization, rhs.copy(), result))
        return result

    monkeypatch.setattr(VerifiedLinearSystem, 'solve_verified', record)
    result = compute_predictor_corrector(*args, **options, residual_safeguard=True)
    assert len(calls) == result.decision['verified_rhs_solves'] == 3
    assert all(call[0] is calls[0][0] for call in calls)
    assert all(call[1] is calls[0][1] for call in calls)
    assert result.affine_info.attempts == 1
    assert [call[3].info.attempts for call in calls] == [1, 0, 0]
    assert result.decision['factorization_attempts'] == 1
    assert result.combined_info is calls[2][3].info
    assert result.target == .1
    assert_array_equal(result.correction, np.zeros(1))
    assert result.decision['direction_choice'] == 'centering_rescue'
    rescue = result.decision['rescue']
    assert rescue['applied'] and rescue['reason'] == 'non_descent'
    assert rescue['corrector_residual_slope'] > 0.
    assert rescue['centering_residual_slope'] < 0.
    assert result.decision['residual_merit_slope'] == rescue['centering_residual_slope']
    matrix = four_blocks(model.Q.toarray(), model.C.toarray(), np.zeros((0, 1)),
                         np.ones(1), np.ones(1), .002, .003, 0.)
    residual = np.array([1., 0., .9])
    actual = np.r_[result.dx, result.ds, result.dy, result.dnu]
    assert_allclose(matrix @ actual, -residual, atol=1e-9, rtol=1e-9)
    assert rescue['discarded_corrector_linear_solve'] == calls[1][3].info.history_fields()


def test_discarded_corrector_refinements_are_included_in_total(monkeypatch):
    _, args, options = scalar_case()
    solve = VerifiedLinearSystem.solve_verified
    refinements = iter((2, 3, 5))

    def controlled_counts(system, rhs):
        result = solve(system, rhs)
        return replace(result, info=replace(result.info, refinements=next(refinements)))

    monkeypatch.setattr(VerifiedLinearSystem, 'solve_verified', controlled_counts)
    result = compute_predictor_corrector(*args, **options, residual_safeguard=True)
    assert result.affine_info.refinements == 2
    assert result.combined_info.refinements == 5
    assert result.extra_refinements == 3
    assert result.refinements == 10
    assert result.decision['rescue']['discarded_corrector_refinements'] == 3


@pytest.mark.parametrize("bad_slope", [np.inf, np.nan, 0.])
def test_nonfinite_or_zero_residual_slope_triggers_same_factor_rescue(monkeypatch, bad_slope):
    from glqp import proximal

    _, args, options = scalar_case()
    slopes = iter((bad_slope, -2.))
    monkeypatch.setattr(proximal, 'residual_slope', lambda *args: next(slopes))
    result = compute_predictor_corrector(*args, **options, residual_safeguard=True)
    assert result.decision['verified_rhs_solves'] == 3
    assert result.decision['residual_merit_slope'] == -2.
    assert result.decision['rescue']['reason'] == (
        'non_descent' if bad_slope == 0. else 'nonfinite_slope')


def test_safeguard_checks_clipped_direction_without_editing_returned_arrays(monkeypatch):
    from glqp import proximal

    _, args, options = scalar_case(s=1e-16, y=1e-16)
    observed = []

    def slope(model, h, dx, ds, dy, dnu, y, s, *rest):
        observed.append((ds.copy(), dy.copy()))
        assert np.all(ds >= -s)
        assert np.all(dy >= -y)
        return -1.

    monkeypatch.setattr(proximal, 'residual_slope', slope)
    plain = compute_predictor_corrector(*args, **options)
    guarded = compute_predictor_corrector(*args, **options, residual_safeguard=True)
    assert len(observed) == 1
    assert guarded.decision['verified_rhs_solves'] == 2
    assert not guarded.decision['rescue']['applied']
    assert guarded.decision['direction_choice'] == 'predictor_corrector'
    for key in ('dx', 'ds', 'dy', 'dnu', 'correction'):
        assert_array_equal(getattr(plain, key), getattr(guarded, key))


def test_residual_safeguard_rejects_shifted_complementarity():
    _, args, options = scalar_case()
    options['complementarity_reg'] = .001
    with pytest.raises(ValueError, match='requires standard complementarity'):
        compute_predictor_corrector(*args, **options, residual_safeguard=True)


def test_affine_diagnostics_use_the_requested_independent_blocks():
    model, args, options = scalar_case()
    model.settings.iteration_diagnostics = True
    result = compute_predictor_corrector(*args, **options, residual_safeguard=True)
    diagnostic = result.decision['affine_direction']
    assert diagnostic['regularization_defect_inf']['complementarity'] == 0.
    assert diagnostic['regularization_defect_inf']['primal'] > 0.
    assert max(diagnostic['requested_regularized_before_inf'].values()) < 1e-9


def test_affine_diagnostics_separate_original_residuals_and_center_offsets():
    model, args, options = scalar_case()
    model.settings.iteration_diagnostics = True
    offsets = dict(stationarity=np.array([.2]), primal=np.array([-.3]),
                   equality=np.zeros(0))
    result = compute_predictor_corrector(*args, **options, center_offsets=offsets)
    diagnostic = result.decision['affine_direction']
    matrix = four_blocks(model.Q.toarray(), model.C.toarray(), np.zeros((0, 1)),
                         np.ones(1), np.ones(1), .002, .003, 0.)
    affine = np.linalg.solve(matrix, -np.array([1., 0., 1.]))
    expected_original = dict(stationarity=abs(-.002*affine[0] - .2),
                             primal=abs(.003*affine[2] + .3),
                             equality=0., complementarity=0.)
    for key, expected in expected_original.items():
        assert diagnostic['original_newton_before_inf'][key] == pytest.approx(expected, abs=1e-9)
    assert diagnostic['center_residual_offset_inf'] == dict(stationarity=.2, primal=.3, equality=0.)
    assert max(diagnostic['requested_regularized_before_inf'].values()) < 1e-9


def test_bounded_corrector_enforces_inexact_newton_residual_and_descent_bounds():
    model, args, options = scalar_case()
    unbounded = compute_predictor_corrector(*args, **options)
    model.settings.proximal_corrector_limit = .5
    bounded = compute_predictor_corrector(*args, **options, residual_safeguard=True)
    residual = np.array([1., 0., .9])
    matrix = four_blocks(model.Q.toarray(), model.C.toarray(), np.zeros((0, 1)),
                         np.ones(1), np.ones(1), .002, .003, 0.)
    direction = np.r_[bounded.dx, bounded.ds, bounded.dy]
    correction = np.r_[0., 0., bounded.correction]
    jacobian_direction = matrix @ direction
    assert np.linalg.norm(correction) <= .5*np.linalg.norm(residual)*(1+1e-14)
    assert_allclose(jacobian_direction, -residual-correction, atol=1e-9, rtol=1e-9)
    assert residual @ jacobian_direction <= -.5*np.dot(residual, residual)+1e-9
    assert np.linalg.norm(jacobian_direction) <= 1.5*np.linalg.norm(residual)+1e-9
    assert bounded.decision['verified_rhs_solves'] == 2
    assert not bounded.decision['rescue']['applied']
    bound = bounded.decision['corrector_bound']
    assert bound['raw_norm'] == pytest.approx(np.linalg.norm(unbounded.correction))
    assert bound['applied_norm'] == pytest.approx(np.linalg.norm(bounded.correction))
    assert bound['weight'] < 1.
    assert_allclose(bounded.correction, bound['weight']*unbounded.correction)
    assert bound['sufficient_descent_limit'] == pytest.approx(-.1*np.dot(residual, residual))


def test_negative_but_insufficient_slope_triggers_centering_rescue(monkeypatch):
    from glqp import proximal

    _, args, options = scalar_case()
    slopes = iter((-.01, -2.))
    monkeypatch.setattr(proximal, 'residual_slope', lambda *args: next(slopes))
    result = compute_predictor_corrector(*args, **options, residual_safeguard=True)
    assert result.decision['verified_rhs_solves'] == 3
    assert result.decision['rescue']['reason'] == 'insufficient_descent'
    assert result.decision['residual_merit_slope'] == -2.


@pytest.mark.parametrize("scale", [1e-200, 1., 1e200])
def test_corrector_bound_avoids_overflow_and_is_homogeneous(scale):
    correction, decision = _bounded_correction(
        scale*np.array([6., 8.]), scale*np.array([3., 4.]), .5)
    assert_allclose(correction/scale, np.array([1.5, 2.]), atol=1e-15, rtol=1e-15)
    assert decision['weight'] == pytest.approx(.25)
    assert np.isfinite(decision['raw_norm'])
    assert np.isfinite(decision['applied_norm'])


@pytest.mark.parametrize("limit", [None, 0., .5])
def test_zero_corrector_remains_zero_without_invalid_norm_ratios(limit):
    correction, decision = _bounded_correction(np.zeros(2), np.zeros(3), limit)
    assert_array_equal(correction, np.zeros(2))
    assert decision['weight'] == 1.
    assert decision['raw_norm'] == decision['applied_norm'] == 0.
