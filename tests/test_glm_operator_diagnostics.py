"""Original-design diagnostics must expose drift and lost weak curvature."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from glqp import GLQP, SolverSettings
from glqp.glm_diagnostics import glm_hessian_diagnostics, glm_predictor_diagnostics
from glqp.obj import GaussianNLL, LogisticNLL


@pytest.mark.parametrize('scale,ridge', [(1., 0.), (1e8, 1e-7)])
def test_actual_gram_assembly_can_erase_an_original_design_mode(scale, ridge):
    A = sparse.csc_array([[scale, scale], [scale * 1e-8, 0.]])
    Q = sparse.eye_array(2, format='csc') * ridge
    model = GLQP(A=A, Q=Q, f=GaussianNLL(np.zeros(2), np.ones(2)))
    z, dx = np.zeros(2), np.array([1., -1.])
    row = glm_hessian_diagnostics(model, model.get_H(z), z, dx)
    assert row['assembled_action_inf'] == 0.
    assert row['original_operator_action_inf'] == pytest.approx((scale * 1e-8)**2 + ridge)
    assert row['relative_to_original_inf'] == 1.
    assert row['assembled_rayleigh'] == 0.
    assert row['original_operator_rayleigh'] > 0.
    assert row['physical_proximal_terms_included'] is False
    assert row['numerical_factor_shift_included'] is False


def test_predictor_drift_measures_original_gradient_change_without_mutating_state():
    A = sparse.csc_array([[1., 2.], [3., -1.]])
    model = SimpleNamespace(A=A, f=GaussianNLL([1., 2.], [.5, 2.]))
    x = np.array([2., -1.])
    fresh_z = A @ x
    z = fresh_z + np.array([.25, -.5])
    before = x.copy(), z.copy()
    row = glm_predictor_diagnostics(model, x, z)
    difference = np.array([.125, -1.])
    assert row['carried_minus_fresh_inf'] == .5
    assert row['relative_drift_inf'] == .5 / 7.
    assert row['scalar_gradient_drift_inf'] == 1.
    assert row['stationarity_drift_inf'] == np.max(np.abs(A.toarray().T @ difference))
    assert row['carried_minus_fresh_loss'] == model.f(z) - model.f(fresh_z)
    np.testing.assert_array_equal(x, before[0])
    np.testing.assert_array_equal(z, before[1])


def test_hessian_check_uses_nonlinear_curvature_at_carried_predictor():
    A = sparse.csc_array([[1., -2.], [3., 1.]])
    Q = sparse.csc_array([[.3, .1], [.1, .7]])
    model = GLQP(A=A, Q=Q, f=LogisticNLL([0., 1.], [.2, .8]))
    z, dx = np.array([-3., 4.]), np.array([.5, -.7])
    row = glm_hessian_diagnostics(model, model.get_H(z), z, dx)
    tail = np.exp(-np.abs(z))
    D = np.array([.2, .8]) * tail / (1. + tail)**2
    reference = Q.toarray() @ dx + A.toarray().T @ (D * (A.toarray() @ dx))
    assert row['original_operator_action_inf'] == pytest.approx(np.max(np.abs(reference)))
    assert row['assembled_minus_original_inf'] < 1e-15
    assert row['curvature_minimum_positive'] == pytest.approx(np.min(D))
    assert row['curvature_zero_count'] == 0


@pytest.mark.parametrize('with_glm,enabled', [(True, False), (False, False), (False, True)])
def test_operator_helpers_do_no_work_unless_glm_diagnostics_enabled(monkeypatch, with_glm, enabled):
    import glqp.solver as solver_module

    def forbidden(*args, **kwargs):
        raise AssertionError('Original-design observations were not requested')

    monkeypatch.setattr(solver_module, 'glm_predictor_diagnostics', forbidden)
    monkeypatch.setattr(solver_module, 'glm_hessian_diagnostics', forbidden)
    kwargs = dict(A=sparse.csc_array([[1.], [1.]]), f=LogisticNLL([0., 1.], [.9, .1])) if with_glm else {}
    model = GLQP(Q=sparse.csc_array([[.1]]), **kwargs)
    _, result = model.solve(x0=np.array([1.]), verbose=False,
        settings=SolverSettings(iteration_diagnostics=enabled))
    assert result.termination_tag == 'optimal'


def test_glm_operator_observations_preserve_arrays_and_common_history():
    results = []
    for enabled in (False, True):
        model = GLQP(A=sparse.csc_array([[1.], [1.]]), Q=sparse.csc_array([[1e-4]]),
                     f=LogisticNLL([0., 1.], [.9, .1]))
        _, result = model.solve(x0=np.array([5.]), verbose=False,
            settings=SolverSettings(iteration_diagnostics=enabled))
        results.append(result)
    plain, observed = results
    assert plain.termination_tag == observed.termination_tag == 'optimal'
    for name in ('x', 'y', 's', 'nu'):
        np.testing.assert_array_equal(getattr(plain, name), getattr(observed, name))
    columns = [name for name in plain.history.columns if 'time' not in name.lower()]
    assert plain.history[columns].equals(observed.history[columns])
    assert plain.diagnostics is None
    assert observed.diagnostics['initial']['glm_predictor']['carried_minus_fresh_inf'] == 0.
    assert all('glm_hessian' in row for row in observed.diagnostics['iterations'])
    final = observed.diagnostics['iterations'][-1]['iterate']['glm_predictor']
    assert final['stationarity_drift_inf'] < 1e-14
