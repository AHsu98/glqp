"""Predictor–corrector equations, globalization, and original certificates."""
import warnings
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.optimize import brentq
from scipy.sparse import csc_array, eye_array
from scipy.special import expit

from glqp import GLQP, SolverSettings
from glqp.obj import GaussianNLL, LogisticNLL, PoissonNLL


def solve_quietly(problem, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return problem.solve(verbose=False, **kwargs)


def dense_newton_matrix(H, C, E, s, y, rho):
    """Independent unscaled equations, in (dx, ds, dy, dnu) order."""
    n, k, p = len(H), len(s), len(E)
    return np.block([
        [H + rho*np.eye(n), np.zeros((n, k)), C.T, E.T],
        [C, np.eye(k), -rho*np.eye(k), np.zeros((k, p))],
        [np.zeros((k, n)), np.diag(y + rho), np.diag(s), np.zeros((k, p))],
        [E, np.zeros((p, k)), np.zeros((p, k)), -rho*np.eye(p)],
    ])


def boundary_step(s, ds, y, dy):
    limits = [1.]
    limits.extend((-s[ds < 0]/ds[ds < 0]).tolist())
    limits.extend((-y[dy < 0]/dy[dy < 0]).tolist())
    return min(limits)


@pytest.mark.parametrize("rho", [0., .2])
@pytest.mark.parametrize("override", [None, 0., .7])
def test_combined_direction_solves_original_equations_and_uses_affine_cross_term(rho, override):
    H = np.array([[4., .3], [.3, 2.]])
    C = np.array([[1., -.5], [-.2, 1.], [-1., -.3]])
    E = np.array([[1., 1.]])
    s, y = np.array([.4, 1.1, .7]), np.array([1.3, .6, 2.])
    rx, rp, req = np.array([.7, -.2]), np.array([.1, -.3, .2]), np.array([.05])
    model = GLQP(Q=csc_array(H), C=csc_array(C), c=np.ones(3),
                 E=csc_array(E), e=np.zeros(1))
    settings = SolverSettings(step_strategy="predictor_corrector", tol=1e-10,
                              barrier_kkt_floor=.03, min_mu=1e-8)
    model.initialize(x0=np.zeros(2), s0=s, y0=y, settings=settings)
    kkt_norm = max(np.max(np.abs(np.r_[rx, rp, req])), np.max(s*y))
    result = model.solve_predictor_corrector(np.zeros(2), y, s, csc_array(H),
        rx, rp, req, kkt_norm, prox_reg=rho, target_override=override)

    matrix = dense_newton_matrix(H, C, E, s, y, rho)
    affine = np.linalg.solve(matrix, -np.r_[rx, rp, s*y, req])
    ads, ady = affine[2:5], affine[5:8]
    alpha = boundary_step(s, ads, y, ady)
    mean = np.mean(s*y)
    affine_mean = np.mean((s + alpha*ads)*(y + alpha*ady))
    sigma = np.clip(affine_mean/mean, 0., 1.)**3
    target = max(sigma*mean, settings.min_mu, settings.barrier_kkt_floor*kkt_norm)
    if override is not None:
        target = override
    correction = ads*ady
    rhs = -np.r_[rx, rp, s*y + correction-target, req]
    combined = np.linalg.solve(matrix, rhs)
    observed = np.r_[result.dx, result.ds, result.dy, result.dnu]
    assert_allclose(result.correction, correction, rtol=2e-8, atol=2e-10)
    assert result.decision["affine_step"] == pytest.approx(alpha, rel=2e-8, abs=2e-10)
    assert result.decision["affine_mu"] == pytest.approx(affine_mean, rel=2e-8, abs=2e-10)
    assert result.decision["sigma"] == pytest.approx(sigma, rel=2e-8, abs=2e-10)
    assert result.target == pytest.approx(target, rel=2e-8, abs=2e-10)
    assert_allclose(matrix @ observed, rhs, rtol=2e-8, atol=2e-10)
    assert_allclose(observed, combined, rtol=2e-8, atol=2e-10)
    # A combined RHS already contains the affine residuals. Adding the affine
    # step again would double-count them and fails this independent system.
    assert np.linalg.norm(matrix @ (observed + affine)-rhs, np.inf) > .1
    assert result.affine_info.attempts == 1
    assert result.combined_info.attempts == 0


def test_each_predictor_requests_two_verified_rhs_solves_on_one_matrix(monkeypatch):
    from glqp.predictor_corrector import VerifiedLinearSystem

    calls = []
    original = VerifiedLinearSystem.solve_verified

    def record(system, rhs, *args, **kwargs):
        calls.append((system, rhs.copy()))
        return original(system, rhs, *args, **kwargs)

    monkeypatch.setattr(VerifiedLinearSystem, "solve_verified", record)
    model = GLQP(Q=eye_array(2), C=eye_array(2), c=np.ones(2))
    model.initialize(settings=SolverSettings(step_strategy="predictor_corrector"))
    result = model.solve_predictor_corrector(np.zeros(2), np.array([1., 2.]),
        np.array([.4, .7]), eye_array(2), np.array([.2, -.5]), np.array([.1, -.2]),
        np.zeros(0), 1.4, prox_reg=1e-4)
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0]
    assert not np.array_equal(calls[0][1], calls[1][1])
    assert result.affine_info.attempts == 1 and result.combined_info.attempts == 0


@pytest.mark.parametrize("minimum,kkt_floor,expected", [(2., 0., 2.), (1e-12, .4, 4.)])
def test_predictor_target_obeys_the_configured_minimum_and_current_kkt_floor(minimum, kkt_floor, expected):
    model = GLQP(Q=eye_array(1), C=eye_array(1), c=np.ones(1))
    model.initialize(settings=SolverSettings(step_strategy="predictor_corrector",
        min_mu=minimum, barrier_kkt_floor=kkt_floor))
    result = model.solve_predictor_corrector(np.zeros(1), np.ones(1), np.ones(1),
        eye_array(1), np.ones(1), np.zeros(1), np.zeros(0), 10., prox_reg=0.)
    assert result.target == expected


@pytest.mark.parametrize("value", ["bad", None, True, 42])
@pytest.mark.parametrize("mutated", [False, True])
def test_invalid_step_strategy_is_rejected_before_solving(value, mutated):
    with pytest.raises(ValueError, match="step_strategy"):
        if mutated:
            settings = SolverSettings()
            settings.step_strategy = value
        else:
            settings = SolverSettings(step_strategy=value)
        GLQP(Q=eye_array(1), b=np.ones(1)).solve(settings=settings, verbose=False)


def assert_same_trajectory(first, second):
    x0, r0 = first
    x1, r1 = second
    assert r0.termination_tag == r1.termination_tag
    assert_array_equal(x0, x1)
    for name in ("s", "y", "nu"):
        assert_array_equal(getattr(r0, name), getattr(r1, name))
    columns = [name for name in r0.history if name != "time" and not name.startswith("diag_")]
    assert_array_equal(r0.history[columns], r1.history[columns])


@pytest.mark.parametrize("empty", [False, True])
def test_no_inequality_path_is_exactly_the_existing_newton_path(monkeypatch, empty):
    inequality = dict(C=csc_array((0, 2)), c=np.zeros(0)) if empty else {}
    kwargs = dict(Q=csc_array([[2., .2], [.2, 3.]]), b=np.array([1., -2.]),
                  E=csc_array([[1., 1.]]), e=np.array([.3]), **inequality)
    newton = solve_quietly(GLQP(**kwargs), settings=SolverSettings())
    model = GLQP(**kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("No genuine inequalities: affine solve must be skipped")

    monkeypatch.setattr(model, "solve_predictor_corrector", forbidden)
    pc = solve_quietly(model, settings=SolverSettings(step_strategy="predictor_corrector"))
    assert_same_trajectory(newton, pc)


def test_newton_remains_the_default_strategy():
    kwargs = dict(Q=eye_array(2), b=np.array([1., -.5]),
                  C=csc_array([[1., 0.], [0., -1.]]), c=np.array([.2, .4]))
    outcomes = [solve_quietly(GLQP(**kwargs), settings=SolverSettings(**options))
                for options in ({}, dict(step_strategy="newton"))]
    assert_same_trajectory(*outcomes)


@pytest.mark.parametrize("mu0", [0., 17.])
def test_explicit_initial_target_only_overrides_the_first_combined_direction(monkeypatch, mu0):
    model = GLQP(Q=eye_array(2), b=np.array([1., -.5]),
                 C=eye_array(2), c=np.array([.2, .4]))
    real_solve = model.solve_predictor_corrector
    calls = []

    def observe(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        calls.append((kwargs.get("target_override"), result.target))
        return result

    monkeypatch.setattr(model, "solve_predictor_corrector", observe)
    _, result = solve_quietly(model, mu0=mu0, settings=SolverSettings(
        max_iter=2, step_strategy="predictor_corrector", let_newton_cook=.5))
    assert len(result.history) == len(calls) == 2
    assert calls[0] == (mu0, mu0)
    assert calls[1][0] is None
    assert calls[1][1] > 0.
    assert_array_equal(result.history["mu"].to_numpy(), [value[1] for value in calls])


def test_line_search_and_direction_diagnostics_use_the_selected_target(monkeypatch):
    model = GLQP(Q=csc_array([[2., .2], [.2, 1.]]), b=np.array([1., -.5]),
                 C=csc_array([[1., 0.], [0., -1.]]), c=np.array([.2, .4]))
    original = model.solve_predictor_corrector
    calls = []

    def record(x, y, s, H, rx, rp, req, kkt, **kwargs):
        result = original(x, y, s, H, rx, rp, req, kkt, **kwargs)
        calls.append(dict(x=x.copy(), s=s.copy(), y=y.copy(), rx=rx.copy(), rp=rp.copy(),
            req=req.copy(), dx=result.dx.copy(), ds=result.ds.copy(),
            dy=result.dy.copy(), rho=kwargs["prox_reg"], target=result.target,
            correction=result.correction.copy()))
        return result

    monkeypatch.setattr(model, "solve_predictor_corrector", record)
    _, result = solve_quietly(model, settings=SolverSettings(max_iter=3,
        step_strategy="predictor_corrector", newton_fallback_policy="progress",
        iteration_diagnostics=True, let_newton_cook=.5))
    assert len(calls) == len(result.history) == 3
    for call, row, logged_mu in zip(calls, result.diagnostics["iterations"], result.history["mu"]):
        target = call["target"]
        expected_slope = call["dx"] @ (model.Q @ call["x"]-model.b) - target*np.sum(call["ds"]/call["s"])
        expected_phi = np.max(np.abs(np.r_[call["rx"], call["rp"],
            call["s"]*call["y"]-target, call["req"]]))
        assert row["line_search"]["merit_slope"] == pytest.approx(expected_slope)
        assert row["line_search"]["fallback_perturbed_kkt_before"] == pytest.approx(expected_phi)
        assert row["barrier"]["target"] == row["iterate"]["barrier_target"] == logged_mu == target
        direction = row["direction"]
        assert direction["edited_ds_count"] == direction["edited_dy_count"] == 0
        expected_defect = np.max(np.abs(-call["rho"]*call["ds"]-call["correction"]))
        assert direction["original_newton_before_inf"]["complementarity"] == pytest.approx(expected_defect, abs=1e-8)
        assert direction["corrector_complementarity_defect_inf"] == np.max(np.abs(call["correction"]))
        assert direction["requested_regularized_before_inf"]["complementarity"] < 1e-8


def test_newton_initialization_policy_does_not_select_predictor_targets():
    kwargs = dict(Q=eye_array(2), b=np.array([1., -.5]),
                  C=csc_array([[1., 0.], [0., -1.]]), c=np.array([.2, .4]))
    outcomes = [solve_quietly(GLQP(**kwargs), settings=SolverSettings(
        step_strategy="predictor_corrector", initial_mu_policy=policy, initial_mu_scale=1e5))
        for policy in ("fixed", "complementarity")]
    assert_same_trajectory(*outcomes)


@pytest.mark.parametrize("diagnostics", [False, True])
def test_inactive_initial_scale_cannot_overflow_a_predictor_solve(diagnostics):
    kwargs = dict(Q=eye_array(1), b=np.array([2.]), C=eye_array(1), c=np.ones(1))
    options = dict(step_strategy="predictor_corrector", iteration_diagnostics=diagnostics)
    reference = solve_quietly(GLQP(**kwargs), s0=np.array([2.]), y0=np.array([2.]),
                             settings=SolverSettings(**options))
    huge = solve_quietly(GLQP(**kwargs), s0=np.array([2.]), y0=np.array([2.]),
        settings=SolverSettings(**options, initial_mu_policy="complementarity", initial_mu_scale=1e308))
    assert_same_trajectory(reference, huge)
    if diagnostics:
        initial = huge[1].diagnostics["initial_target"]
        assert initial["source"] == "awaiting_predictor"
        assert not initial["used_for_first_direction"]
        assert initial["scaled_complementarity"] is None


def test_finite_affine_components_with_overflowed_combined_rhs_fail_cleanly(monkeypatch):
    from glqp import predictor_corrector
    from glqp.linear import LinearSolveError

    # The second row limits alpha to 1e-308, so affine mean and both
    # cross-products are finite, but product[0]+correction[0] overflows.
    class ControlledAffineSystem:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def solve_verified(self, rhs):
            self.calls += 1
            assert self.calls == 1, "The nonfinite combined RHS must never reach the backend"
            return SimpleNamespace(solution=np.array([1e154, 1e77, 1e-308]))

    monkeypatch.setattr(predictor_corrector, "VerifiedLinearSystem", ControlledAffineSystem)
    model = GLQP(Q=eye_array(1), C=csc_array([[-1.], [1e154]]), c=np.ones(2))
    model.initialize(settings=SolverSettings(step_strategy="predictor_corrector"))
    with pytest.raises(LinearSolveError, match="right-hand side"):
        model.solve_predictor_corrector(np.zeros(1), np.array([1e154, 1.]),
            np.array([1e154, 1.]), eye_array(1), np.zeros(1), np.zeros(2),
            np.zeros(0), 1., prox_reg=0.)


@pytest.mark.parametrize("fallback", ["finite", "progress"])
def test_diagnostics_and_callbacks_do_not_change_predictor_corrector_trajectory(fallback):
    kwargs = dict(Q=eye_array(2), b=np.array([1., -.5]),
                  C=csc_array([[1., 0.], [0., -1.]]), c=np.array([.2, .4]))
    states = []
    options = dict(step_strategy="predictor_corrector", max_iter=100,
                   newton_fallback_policy=fallback, let_newton_cook=.5)
    plain = solve_quietly(GLQP(**kwargs), settings=SolverSettings(**options))
    observed = solve_quietly(GLQP(**kwargs), settings=SolverSettings(
        **options, iteration_diagnostics=True), iteration_callback=lambda state:
        states.append((state.iteration, state.x.copy(), state.barrier_target)))
    assert_same_trajectory(plain, observed)
    assert plain[1].diagnostics is None
    assert len(states) == len(observed[1].history) + 1
    assert_array_equal(states[-1][1], observed[0])
    assert_array_equal([state[2] for state in states[1:]], observed[1].history["mu"])


@pytest.mark.parametrize("objective_type,response,mean_function", [
    (GaussianNLL, np.array([.2, -.3, .6]), lambda z: z),
    (LogisticNLL, np.array([0., 1., 1.]), expit),
    (PoissonNLL, np.array([0., 3., 2.]), np.exp),
])
@pytest.mark.parametrize("active", [False, True])
def test_constrained_glm_original_certificate_matches_independent_scalar_solution(
        objective_type, response, mean_function, active):
    A = np.array([[1., -.2], [.4, .8], [1.5, .3]])
    weights, Q, b = np.array([.7, 1.3, .2]), np.diag([.2, .4]), np.array([.1, -.2])
    E, e = np.array([[1., 1.]]), np.array([.7])

    def gradient(x):
        return A.T @ (weights*(mean_function(A @ x)-response)) + Q @ x-b

    def scalar_derivative(t):
        g = gradient(np.array([t, .7-t]))
        return g[0]-g[1]

    stationary = brentq(scalar_derivative, -4., 4., xtol=1e-14)
    upper = stationary + (-.25 if active else .25)
    reference = np.array([min(stationary, upper), .7-min(stationary, upper)])
    C, c = np.array([[1., 0.]]), np.array([upper])
    model = GLQP(f=objective_type(response, weights), A=csc_array(A), Q=csc_array(Q), b=b,
                 C=csc_array(C), c=c, E=csc_array(E), e=e)
    x, result = solve_quietly(model, x0=np.array([3., -2.3]),
        settings=SolverSettings(step_strategy="predictor_corrector", max_iter=150))
    assert result.termination_tag == "optimal"
    assert_allclose(x, reference, rtol=0, atol=2e-7)
    assert_allclose(E @ x, e, rtol=0, atol=2e-8)
    assert np.max(C @ x-c) <= 2e-8
    assert np.all(result.y >= 0.)
    assert_allclose(gradient(x) + C.T @ result.y + E.T @ result.nu, 0., rtol=0, atol=2e-7)
    assert_allclose(result.y*(c-C @ x), 0., rtol=0, atol=2e-7)
    assert np.isfinite(result.history.drop(columns="time").select_dtypes(include="number")).all().all()


def test_nonfinite_glm_trials_are_rejected_before_evaluating_derivatives():
    class LogLoss:
        invalid_trials = 0

        def __call__(self, z):
            if z[0] <= 0:
                self.invalid_trials += 1
                return np.inf
            return -np.log(z[0])

        def d1f(self, z):
            assert z[0] > 0 and np.isfinite(z[0])
            return -1./z

        def d2f(self, z):
            assert z[0] > 0 and np.isfinite(z[0])
            return 1./z**2

    loss = LogLoss()
    # Its first Newton direction crosses the likelihood's domain boundary.
    # A zero inequality row is still genuine and exercises both PC RHSs.
    model = GLQP(f=loss, A=eye_array(1), b=np.array([-10.]),
                 C=csc_array((1, 1)), c=np.ones(1))
    x, result = solve_quietly(model, x0=np.ones(1), settings=SolverSettings(
        step_strategy="predictor_corrector", max_iter=1, max_linesearch_steps=100,
        let_newton_cook=.5, iteration_diagnostics=True))
    assert loss.invalid_trials > 0
    assert len(result.history) == 1
    assert np.all(np.isfinite(np.r_[x, result.s, result.y]))
    assert x[0] > 0. and np.all(result.s > 0.) and np.all(result.y > 0.)
    search = result.diagnostics["iterations"][0]["line_search"]
    assert search["rejection_reasons"]["nonfinite_merit"] > 0

