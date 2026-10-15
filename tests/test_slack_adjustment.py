"""Residuals must describe the actual iterate after an interior slack repair."""
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import eye_array

from glqp import GLQP, SolverSettings


@pytest.mark.parametrize("max_iter", [1, 2])
def test_slack_repair_updates_reported_residuals_and_next_newton_rhs(monkeypatch, max_iter):
    # A tiny fraction-to-boundary step leaves one product far below the
    # average. Small duals make the slack repair's feasibility error dominate
    # the KKT norm, so a stale norm cannot hide behind complementarity.
    initial_slack = np.array([1e-8, 1.])
    dual = np.full(2, 1e-4)
    problem = GLQP(Q=eye_array(2), b=dual, C=eye_array(2), c=initial_slack)
    calls = []

    def direction(x, y, s, H, rx, rp, rc, req, mu, **kwargs):
        calls.append(tuple(np.array(v, copy=True) for v in (x, y, s, rx, rp, rc, req, mu)))
        ds = np.array([-1., 0.]) if len(calls) == 1 else np.zeros(2)
        return np.zeros(2), ds, np.zeros(2), np.zeros(0), None, 0, 0.

    monkeypatch.setattr(problem, "solve_KKT", direction)
    with pytest.warns(UserWarning, match="Maximum Iterations"):
        x, result = problem.solve(x0=np.zeros(2), s0=initial_slack, y0=dual,
                                  mu0=1e-12, verbose=False,
                                  settings=SolverSettings(max_iter=max_iter))

    first = result.history.iloc[0]
    assert 0 < first["step"] < 1e-4
    unadjusted = initial_slack + first["step"] * np.array([-1., 0.])
    assert result.s[0] > 1e6 * unadjusted[0]  # Exercise the repair itself.
    if max_iter == 2:
        next_x, y, s, rx, rp, rc, req, mu = calls[1]
        assert_allclose(rx, problem.Q @ next_x + problem.C.T @ y - problem.b, atol=1e-15)
        assert_allclose(rp, problem.C @ next_x + s - problem.c, atol=1e-15)
        assert_allclose(rc, y * s - mu, atol=1e-15)

    rp = problem.C @ x + result.s - problem.c
    rc = result.y * result.s
    expected_kkt = max(np.max(np.abs(rp)), np.max(np.abs(rc)))
    assert first["cons_viol"] == pytest.approx(np.max(np.abs(rp)))
    assert first["comp_res"] == pytest.approx(np.max(np.abs(rc)))
    assert first["KKT_res"] == pytest.approx(expected_kkt)
    assert result.cons_viol == pytest.approx(np.max(np.abs(rp)))
