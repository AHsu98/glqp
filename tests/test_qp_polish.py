"""Behavioral tests of experimental polishing, independently of the IPM."""
import json

import numpy as np
import pytest
from scipy.sparse import csc_array

from glqp.backends import FactorizationError
from glqp.qp_polish import PolishSettings, polish_qp


def scalar(x=.9999999, z=1., **options):
    return polish_qp(csc_array([[1.]]), np.array([-2.]), None, None,
                     csc_array([[1.]]), np.array([1.]), np.array([x]), None,
                     np.array([z]), PolishSettings(**options))


@pytest.mark.parametrize('strategy', ['one_shot_slack', 'one_shot_ratio', 'add_drop', 'pdas'])
@pytest.mark.parametrize('backend', ['qdldl', 'superlu'])
def test_correct_active_set_certifies(strategy, backend):
    result = scalar(strategy=strategy, backend=backend)
    assert result.accepted and result.status == 'accepted'
    assert result.after['max_residual'] < result.before['max_residual']
    np.testing.assert_allclose(result.x, [1.], atol=1e-10)
    np.testing.assert_allclose(result.z, [1.], atol=1e-10)
    json.dumps(result.history, allow_nan=False)


def test_ratio_can_identify_active_constraint_farther_than_slack_cutoff():
    slack = scalar(x=.99, strategy='one_shot_slack')
    ratio = scalar(x=.99, strategy='one_shot_ratio')
    assert not slack.accepted and slack.stop_reason == 'certificate_rejected'
    np.testing.assert_array_equal(slack.x, [.99])
    np.testing.assert_allclose(slack.candidate['x'], [2.], atol=1e-10)
    assert ratio.accepted


@pytest.mark.parametrize('strategy', ['add_drop', 'pdas'])
def test_working_set_updates_rescue_wrong_and_missing_constraints(strategy):
    # The initial guess has only row 1 active. It needs dropping, and row 0
    # must be added: an unconstrained solve violates x[0] <= 1.
    result = polish_qp(csc_array(np.eye(2)), np.array([-2., 0.]), None, None,
                       csc_array(np.eye(2)), np.ones(2), np.array([0., 1.]),
                       None, np.array([0., 2.]), PolishSettings(strategy=strategy))
    assert result.accepted
    np.testing.assert_allclose(result.x, [1., 0.], atol=1e-10)
    np.testing.assert_allclose(result.z, [1., 0.], atol=1e-10)
    assert any(0 in row['added'] for row in result.history)
    assert any(1 in row['dropped'] for row in result.history)


@pytest.mark.parametrize('strategy', ['one_shot_slack', 'one_shot_ratio'])
def test_no_clipping_of_wrong_sign_multiplier(strategy):
    result = polish_qp(csc_array([[1.]]), np.zeros(1), None, None,
                       csc_array([[1.]]), np.ones(1), np.ones(1), None,
                       np.array([2.]), PolishSettings(strategy=strategy))
    assert not result.accepted
    assert result.history[0]['certificate']['dual_sign_violation_scaled'] > .9
    np.testing.assert_array_equal(result.z, [2.])


def test_dependent_equalities_and_semidefinite_objective():
    result = polish_qp(csc_array(np.diag([1., 0.])), np.zeros(2),
                       csc_array([[1., 1.], [2., 2.]]), np.array([1., 2.]),
                       None, None, np.array([.2, .8]), np.zeros(2), None)
    assert result.accepted
    np.testing.assert_allclose(result.x, [0., 1.], atol=1e-8)
    assert result.history[0]['lin_status'] == 'accurate'


def test_degenerate_opposing_active_rows_keep_dual_nullspace():
    # x=0 is fixed by opposing inequalities. Positive multipliers [2, 1]
    # balance q=-1. The absolute regularized solve selects [0.5, -0.5];
    # an anchored correction retains the useful positive nullspace component.
    kwargs = dict(P=csc_array([[1.]]), q=np.array([-1.]), A=None, b=None,
                  G=csc_array([[1.], [-1.]]), h=np.zeros(2),
                  x=np.array([1e-7]), y=None, z=np.array([2., 1.]))
    correction = polish_qp(**kwargs, settings=PolishSettings(solve_mode='correction'))
    absolute = polish_qp(**kwargs, settings=PolishSettings(solve_mode='absolute'))
    assert correction.accepted
    assert np.min(correction.z) > .9
    assert not absolute.accepted
    np.testing.assert_array_equal(absolute.z, kwargs['z'])


@pytest.mark.parametrize('row_scale', [1e-9, 1., 1e9])
@pytest.mark.parametrize('objective_scale', [1e-8, 1., 1e8])
def test_row_and_objective_scaling(row_scale, objective_scale):
    result = polish_qp(csc_array([[objective_scale]]), np.array([-2. * objective_scale]),
                       None, None, csc_array([[row_scale]]), np.array([row_scale]),
                       np.array([.99]), None, np.array([objective_scale / row_scale]),
                       PolishSettings(strategy='one_shot_ratio'))
    assert result.accepted
    np.testing.assert_allclose(result.x, [1.], atol=1e-10)
    np.testing.assert_allclose(result.z * row_scale / objective_scale, [1.], atol=1e-10)


def test_unconstrained_and_already_exact_endpoints():
    result = polish_qp(csc_array([[2.]]), np.array([-4.]), None, None, None,
                       None, np.array([1.]), None, None)
    assert result.accepted
    np.testing.assert_allclose(result.x, [2.], atol=1e-11)
    exact = polish_qp(csc_array([[2.]]), np.array([-4.]), None, None, None,
                      None, np.array([2.]), None, None)
    assert not exact.accepted and exact.status == 'unchanged'
    assert exact.stop_reason == 'already_exact' and exact.factorizations == 0


def test_inputs_immutable_and_results_owned():
    P, G = csc_array([[1.]]), csc_array([[1.]])
    q, h, x, z = np.array([-2.]), np.ones(1), np.array([.9999999]), np.ones(1)
    arrays = [P.data, P.indices, P.indptr, G.data, G.indices, G.indptr, q, h, x, z]
    snapshots = [a.copy() for a in arrays]
    for a in arrays:
        a.flags.writeable = False
    result = polish_qp(P, q, None, None, G, h, x, None, z)
    for a, snapshot in zip(arrays, snapshots):
        np.testing.assert_array_equal(a, snapshot)
        assert not a.flags.writeable
    assert result.accepted
    for a in (result.x, result.y, result.z):
        assert a.flags.owndata
        assert not any(np.shares_memory(a, original) for original in arrays)


def test_failed_factorization_returns_original_point(monkeypatch):
    def fail(*args, **kwargs):
        raise FactorizationError('synthetic singular factor')
    monkeypatch.setattr('glqp.qp_polish.factor_and_solve', fail)
    result = scalar()
    assert not result.accepted and result.status == 'linear_failure'
    assert result.failed_solve_calls == 1 and result.history[0]['lin_attempts'] is None
    np.testing.assert_array_equal(result.x, [.9999999])
    np.testing.assert_array_equal(result.z, [1.])
    assert result.after == result.before


def test_programming_and_missing_backend_errors_propagate(monkeypatch):
    def fail(*args, **kwargs):
        raise ImportError('missing optional backend')
    monkeypatch.setattr('glqp.qp_polish.factor_and_solve', fail)
    with pytest.raises(ImportError):
        scalar()
    with pytest.raises(ValueError, match='symmetric'):
        polish_qp(csc_array([[1., 2.], [0., 1.]]), np.zeros(2), None, None,
                  None, None, np.ones(2), None, None)
    with pytest.raises(ValueError):
        scalar(max_passes=0)
    with pytest.raises(ValueError):
        scalar(backend='not-a-backend')


def test_soft_time_and_pass_limits_preserve_original_point(monkeypatch):
    calls = iter([0., 11., 11.])
    monkeypatch.setattr('glqp.qp_polish.perf_counter', lambda: next(calls))
    result = scalar()
    assert result.status == 'time_limit' and result.factorizations == 0
    np.testing.assert_array_equal(result.x, [.9999999])


def test_one_pass_limit_rejects_uncertified_intermediate():
    result = scalar(x=0., z=0., strategy='add_drop', max_passes=1)
    assert not result.accepted and result.stop_reason == 'pass_limit'
    assert result.history[0]['added'] == [0]
    np.testing.assert_array_equal(result.x, [0.])


def test_objective_offset_matches_certificate_scale():
    base = scalar(x=.99, strategy='one_shot_ratio')
    offset = scalar(x=.99, strategy='one_shot_ratio', objective_constant=1e6)
    assert offset.before['objective'] == base.before['objective'] + 1e6
    assert offset.before['complementarity_residual_scaled'] < base.before['complementarity_residual_scaled']


def test_inconsistent_active_constraints_cannot_be_certified():
    result = polish_qp(csc_array([[1.]]), np.zeros(1), None, None,
                       csc_array([[1.], [-1.]]), np.array([0., -1.]),
                       np.array([.5]), None, np.ones(2))
    assert not result.accepted and not result.after['accepted']
    np.testing.assert_array_equal(result.x, [.5])


def test_unbounded_reduced_qp_is_not_fixed_by_numerical_shift():
    result = polish_qp(csc_array([[0.]]), np.array([-1.]), None, None,
                       csc_array([[1.]]), np.ones(1), np.zeros(1), None,
                       np.zeros(1), PolishSettings(strategy='one_shot_ratio'))
    assert result.status == 'linear_failure' and not result.accepted
    np.testing.assert_array_equal(result.x, [0.])


def test_certified_but_worse_candidate_preserves_certified_input(monkeypatch):
    from glqp.qp_polish import factor_and_solve

    def less_accurate(*args, **kwargs):
        result = factor_and_solve(*args, **kwargs)
        result.solution[0] += 1e-9
        return result

    monkeypatch.setattr('glqp.qp_polish.factor_and_solve', less_accurate)
    result = scalar(x=1. - 1e-10)
    assert result.before['accepted'] and result.history[-1]['certificate']['accepted']
    assert not result.accepted and result.status == 'unchanged'
    assert result.stop_reason == 'no_certificate_improvement'
    np.testing.assert_array_equal(result.x, [1. - 1e-10])


def test_repeated_working_set_stops_before_another_factorization(monkeypatch):
    # Supply cycling guesses to exercise the termination contract independently
    # of floating-point tie breaking in the particular ratio rule.
    guesses = iter([np.array([False, False]), np.array([False, True]),
                    np.array([False, False])])
    monkeypatch.setattr('glqp.qp_polish._ratio_set', lambda *args: next(guesses))
    result = polish_qp(csc_array([[1.]]), np.array([-3.]), None, None,
                       csc_array([[1.], [1.]]), np.array([1., 2.]),
                       np.zeros(1), None, np.zeros(2), PolishSettings(strategy='pdas'))
    assert result.status == 'cycle' and not result.accepted
    assert result.stop_reason == 'repeated_active_set' and len(result.history) == 2


def test_zero_rows_and_noncanonical_inputs_are_not_modified():
    P = csc_array((np.array([.5, .5]), np.array([0, 0]), np.array([0, 2])), shape=(1, 1))
    old_data, old_indices, old_indptr = P.data.copy(), P.indices.copy(), P.indptr.copy()
    result = polish_qp(P, np.array([-2.]), csc_array([[0.]]), np.zeros(1),
                       csc_array([[1.], [0.]]), np.array([1., 0.]),
                       np.array([.9999999]), np.zeros(1), np.array([1., 0.]))
    assert result.accepted
    for original, old in ((P.data, old_data), (P.indices, old_indices), (P.indptr, old_indptr)):
        np.testing.assert_array_equal(original, old)
