"""Independent numerical checks for the diagnostic's reported distinctions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse


spec = importlib.util.spec_from_file_location(
    "glm_numerics_probe", Path(__file__).resolve().parents[1] / "tools/diagnose_glm_numerics.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_actions_include_quadratic_term_and_preserve_extended_weak_mode():
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("Platform has no extended precision longdouble")
    A = sparse.csc_array([[1., 1.], [1., 1. + 1e-8]])
    Q = sparse.diags([1e-20, 1e-20], format="csc")
    D = np.ones(2)
    H = Q + A.T @ A
    vector = np.array([1., -1.])
    row = probe.action_comparison(A, Q, D, H, {"weak": vector}, extended=True)[0]
    assert row["extended_rayleigh"] > 0
    assert row["factored_minus_extended_inf"] < row["assembled_minus_extended_inf"]
    independent = 2e-20 + np.longdouble(A[1, 1]-1.)**2
    assert row["extended_rayleigh"] == pytest.approx(float(independent), rel=1e-3)


def test_curvature_control_exposes_positive_logistic_tail_lost_to_subtraction():
    row = probe.curvature_report("logistic", np.array([-40., 40.]), np.ones(2),
                                 np.array([np.exp(-40.), 0.]))
    assert row["logistic_tail_control"]["positive_control_but_zero_solver_count"] == 1
    assert row["logistic_tail_control"]["lost_curvature_sum"] == pytest.approx(np.exp(-40.))


def test_requested_and_shifted_residuals_are_distinct():
    G = sparse.csc_array([[2., 1.], [1., -3.]])
    shift = sparse.diags([1., -1.], format="csc")
    rhs, tau = np.array([4., -2.]), .125
    solution = np.linalg.solve(G.toarray(), rhs)
    row = probe.system_residuals(G, shift, rhs, solution, tau)
    assert row["requested_residual_inf"] < 1e-14
    assert row["shifted_residual_inf"] == pytest.approx(tau*np.max(np.abs(solution)))
    shifted_solution = np.linalg.solve((G+tau*shift).toarray(), rhs)
    shifted = probe.system_residuals(G, shift, rhs, shifted_solution, tau)
    assert shifted["shifted_residual_inf"] < 1e-14
    assert shifted["requested_residual_inf"] > .01


def test_qr_control_matches_independent_full_kkt_with_both_constraint_types():
    A = sparse.csc_array([[1., .2], [-.5, 2.], [2., -1.]])
    Q = sparse.diags([.3, .7], format="csc")
    D = np.array([.4, 1.2, .9])
    rho, delta, rho_s = .02, .03, .04
    C, E = np.array([[1., -2.], [-1., 0.]]), np.array([[1., 1.]])
    y, s = np.array([.7, 1.1]), np.array([2., .4])
    w = np.sqrt(y+rho_s)
    V = np.vstack([w[:, None]*C, E])
    p = np.concatenate([s+delta*(y+rho_s), [delta]])
    H = Q.toarray() + A.toarray().T @ np.diag(D) @ A.toarray()
    G = sparse.csc_array(np.block([[H+rho*np.eye(2), V.T], [V, -np.diag(p)]]))
    rhs = np.array([1., -3., .4, -.2, .7])
    solution, report = probe.qr_newton_control(A, Q, D, G, rhs, rho)
    np.testing.assert_allclose(solution, np.linalg.solve(G.toarray(), rhs), rtol=1e-13, atol=1e-13)
    assert report["requested_residual_inf"] < 1e-13
    assert report["extended_operator_residual_inf"] < 1e-13


def test_qr_control_declines_zero_equality_shift_instead_of_dividing_by_zero():
    A, Q = sparse.eye(2, format="csc"), sparse.eye(2, format="csc")
    G = sparse.csc_array([[2.1, 0., 1.], [0., 2.1, 1.], [1., 1., 0.]])
    solution, report = probe.qr_newton_control(A, Q, np.ones(2), G, np.ones(3), .1)
    assert solution is None
    assert "skipped" in report


def test_callback_predictor_copy_is_not_replaced_by_later_rejected_trial():
    capture = probe.StateCapture([0])
    derivative = capture.capture_gradient(lambda z: 2*z)
    accepted_z = np.array([.3, -.7])
    derivative(accepted_z)
    snapshot = SimpleNamespace(iteration=0, stage="initial", x=np.ones(2),
        y=np.ones(1), s=np.ones(1), nu=np.empty(0), objective=1., barrier_target=.1,
        kkt_residual=.2, relative_complementarity=.1, native_converged=False)
    capture.callback(snapshot)
    accepted_z[:] = 20.
    derivative(np.array([1e6, -1e6]))
    np.testing.assert_array_equal(capture.states[0]["carried_z"], [.3, -.7])
    np.testing.assert_array_equal(capture.states[0]["carried_scalar_gradient"], [.6, -1.4])
