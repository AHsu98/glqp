"""Barrier-policy and observation contracts, independent of benchmark timings."""
import warnings
from dataclasses import FrozenInstanceError

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import csc_array, eye_array

from glqp import GLQP, SolverSettings
from glqp.obj import LogisticNLL


def zero_direction(problem, monkeypatch):
    targets = []

    def direction(x, y, s, H, rx, rp, rc, req, mu, **kwargs):
        targets.append(float(mu))
        return (np.zeros(problem.n), np.zeros(problem.k), np.zeros(problem.k),
                np.zeros(problem.p), None, 0, 0.)

    monkeypatch.setattr(problem, "solve_KKT", direction)
    return targets


@pytest.mark.parametrize("initial_mu", [100., 1000.])
def test_guard_uses_preceding_target_across_accepted_iterations(monkeypatch, initial_mu):
    # Central products propose 8e-6, while the raw residual floor is .01.
    # A stationary accepted direction isolates the old-target guard from
    # changes in residuals or centrality, across several actual iterations.
    expected = {
        "previous": initial_mu * np.array([.1, .01, .001]),
        "legacy": np.full(3, .01),
    }
    for policy in ("previous", "legacy"):
        p = GLQP(Q=eye_array(1), b=np.ones(1), C=eye_array(1), c=np.ones(1))
        targets = zero_direction(p, monkeypatch)
        with pytest.warns(UserWarning, match="Maximum Iterations"):
            _, result = p.solve(x0=np.zeros(1), s0=np.ones(1), y0=np.ones(1),
                mu0=initial_mu, verbose=False, settings=SolverSettings(
                    max_iter=3, barrier_update_guard=policy))
        assert_allclose(result.history["mu"], expected[policy], rtol=2e-16)
        assert_allclose(targets, np.r_[initial_mu, expected[policy][:-1]], rtol=2e-16)
        if policy == "previous":
            assert np.all(result.history["mu"].to_numpy() >= .1*np.asarray(targets))
        else:
            assert result.history.iloc[0]["mu"] < .1*targets[0]


def test_previous_target_guard_is_the_default():
    assert SolverSettings().barrier_update_guard == "previous"


@pytest.mark.parametrize("empty_constraints", [False, True])
@pytest.mark.parametrize("policy", ["previous", "legacy"])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_no_inequality_path_keeps_minimum_target(monkeypatch, empty_constraints, policy, diagnostics):
    kwargs = ({"C": csc_array((0, 1)), "c": np.zeros(0)}
              if empty_constraints else {})
    p = GLQP(Q=eye_array(1), b=np.ones(1), **kwargs)
    targets = zero_direction(p, monkeypatch)
    settings = SolverSettings(max_iter=2, min_mu=3e-12, barrier_update_guard=policy,
                              iteration_diagnostics=diagnostics)
    with pytest.warns(UserWarning, match="Maximum Iterations"):
        _, result = p.solve(x0=np.zeros(1), mu0=100., settings=settings, verbose=False)
    assert_array_equal(result.history["mu"].to_numpy(), np.full(2, settings.min_mu))
    assert targets == [100., settings.min_mu]


@pytest.mark.parametrize("name,value", [
    ("barrier_update_guard", "typo"), ("barrier_update_guard", None),
    ("barrier_update_guard", True), ("iteration_diagnostics", "yes"),
    ("iteration_diagnostics", 1), ("iteration_diagnostics", None),
    ("barrier_reduction_limit", -.1), ("barrier_reduction_limit", 1.1),
    ("barrier_reduction_limit", np.inf), ("barrier_reduction_limit", np.nan),
])
def test_invalid_observation_and_guard_settings_fail_explicitly(name, value):
    with pytest.raises(ValueError, match=name):
        settings = SolverSettings(**{name: value})
        GLQP(Q=eye_array(1), b=np.ones(1)).solve(settings=settings, verbose=False)


@pytest.mark.parametrize("name,value", [
    ("barrier_update_guard", "typo"), ("iteration_diagnostics", 1),
    ("barrier_reduction_limit", np.nan),
])
def test_mutated_settings_are_validated_before_solve(name, value):
    settings = SolverSettings()
    setattr(settings, name, value)
    with pytest.raises(ValueError, match=name):
        GLQP(Q=eye_array(1), b=np.ones(1)).solve(settings=settings, verbose=False)


@pytest.mark.parametrize("limit", [0., .25, 1.])
def test_guard_reduction_limit_controls_actual_next_target(monkeypatch, limit):
    p = GLQP(Q=eye_array(1), b=np.ones(1), C=eye_array(1), c=np.ones(1))
    zero_direction(p, monkeypatch)
    with pytest.warns(UserWarning, match="Maximum Iterations"):
        _, result = p.solve(x0=np.zeros(1), s0=np.ones(1), y0=np.ones(1), mu0=100.,
            settings=SolverSettings(max_iter=1, barrier_reduction_limit=limit), verbose=False)
    assert result.history.iloc[0]["mu"] == pytest.approx(max(.01, 100.*limit))


@pytest.mark.parametrize("policy", ["previous", "legacy"])
@pytest.mark.parametrize("with_glm", [False, True])
def test_diagnostics_do_not_change_solutions_or_existing_history(policy, with_glm):
    kwargs = dict(Q=csc_array(np.eye(2)), b=np.array([-1., .8]),
        C=csc_array([[1., 0.], [0., -1.]]), c=np.array([.4, .8]),
        E=csc_array([[1., 1.]]), e=np.array([.1]))
    if with_glm:
        kwargs.update(A=csc_array([[1., 1.], [1., -1.]]),
                      f=LogisticNLL([0., 1.], [1., 1.]))
    outcomes = []
    for enabled in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            outcomes.append(GLQP(**kwargs).solve(verbose=False, settings=SolverSettings(
                max_iter=150, barrier_update_guard=policy, iteration_diagnostics=enabled)))
    (plain_x, plain), (diagnostic_x, diagnostic) = outcomes
    assert plain.termination_tag == diagnostic.termination_tag
    assert_array_equal(plain_x, diagnostic_x)
    for name in ("y", "s", "nu"):
        assert_array_equal(getattr(plain, name), getattr(diagnostic, name))
    columns = [column for column in plain.history.columns if column != "time"]
    assert_array_equal(plain.history[columns], diagnostic.history[columns])
    assert plain.diagnostics is None
    assert diagnostic.diagnostics["schema_version"] == 1


def test_callback_observes_initial_and_accepted_readonly_iterates_without_changing_solve():
    kwargs = dict(Q=eye_array(2), b=np.array([1., -.5]),
                  C=csc_array([[1., 0.], [0., -1.]]), c=np.array([.2, .4]))
    settings = SolverSettings(tol=1e-8)
    reference_x, reference = GLQP(**kwargs).solve(settings=settings, verbose=False)
    p = GLQP(**kwargs)
    observations = []

    def observe(state):
        with pytest.raises(FrozenInstanceError):
            state.iteration = -1
        for name in ("x", "y", "s", "nu"):
            vector = getattr(state, name)
            assert not vector.flags.writeable
            if vector.size:
                with pytest.raises(ValueError, match="read-only"):
                    vector[0] = 123.
        assert np.all(state.y > 0.) and np.all(state.s > 0.)
        objective = .5*state.x @ (p.Q @ state.x) - p.b @ state.x
        residuals = (p.Q @ state.x - p.b + p.C.T @ state.y,
                     p.C @ state.x + state.s - p.c, state.s*state.y)
        kkt = max(np.max(np.abs(block), initial=0.) for block in residuals)
        complementarity = np.sum(np.abs(state.y*(p.c-p.C @ state.x)))/max(1., abs(objective))
        assert state.objective == pytest.approx(objective, abs=1e-15)
        assert state.kkt_residual == pytest.approx(kkt, abs=1e-15)
        assert state.relative_complementarity == pytest.approx(complementarity, abs=1e-15)
        assert state.native_converged == (kkt <= settings.tol and complementarity <= settings.tol)
        assert state.native_near_converged == (kkt <= 100*settings.tol and complementarity <= 100*settings.tol)
        observations.append((state.iteration, state.stage, state.x.copy(), state.y.copy(),
                             state.s.copy(), state.nu.copy(), state.barrier_target))

    x, result = p.solve(settings=settings, verbose=False, iteration_callback=observe)
    assert len(observations) == len(result.history) + 1
    assert observations[0][:2] == (0, "initial")
    assert [item[0] for item in observations] == list(range(len(observations)))
    assert all(item[1] == "accepted" for item in observations[1:])
    assert_array_equal(x, reference_x)
    for name in ("y", "s", "nu"):
        assert_array_equal(getattr(result, name), getattr(reference, name))
    assert_array_equal(result.history.drop(columns="time"), reference.history.drop(columns="time"))
    for item, row in zip(observations[1:], result.history.itertuples()):
        assert item[-1] == row.mu


def test_callback_runs_once_for_already_converged_initial_point():
    observed = []
    x, result = GLQP(Q=eye_array(1), b=np.zeros(1)).solve(
        verbose=False, iteration_callback=lambda state: observed.append(
            (state.iteration, state.stage, state.native_converged, state.x.copy())))
    assert result.termination_tag == "optimal"
    assert result.history.empty
    assert len(observed) == 1 and observed[0][:3] == (0, "initial", True)
    assert_array_equal(observed[0][3], x)


def test_callback_sees_repaired_state_and_current_residuals(monkeypatch):
    initial_slack = np.array([1e-8, 1.])
    initial_dual = np.full(2, 1e-4)
    p = GLQP(Q=eye_array(2), b=initial_dual, C=eye_array(2), c=initial_slack)
    monkeypatch.setattr(p, "solve_KKT", lambda *args, **kwargs:
        (np.zeros(2), np.array([-1., 0.]), np.zeros(2), np.zeros(0), None, 0, 0.))
    observations = []

    def observe(state):
        observations.append(dict(iteration=state.iteration, stage=state.stage,
            s=state.s.copy(), y=state.y.copy(), x=state.x.copy(),
            kkt=state.kkt_residual, mu=state.barrier_target,
            complementarity=state.relative_complementarity))

    with pytest.warns(UserWarning, match="Maximum Iterations"):
        x, result = p.solve(x0=np.zeros(2), s0=initial_slack, y0=initial_dual,
            mu0=1e-12, verbose=False, settings=SolverSettings(max_iter=1,
            iteration_diagnostics=True), iteration_callback=observe)
    assert [state["stage"] for state in observations] == ["initial", "accepted"]
    assert_array_equal(observations[0]["s"], initial_slack)
    accepted = observations[1]
    assert accepted["s"][0] > 1e6*(initial_slack[0]-result.history.iloc[0]["step"])
    assert_array_equal(accepted["x"], x)
    assert_array_equal(accepted["s"], result.s)
    assert_array_equal(accepted["y"], result.y)
    residual = max(np.max(np.abs(p.C @ x + result.s-p.c)),
                   np.max(np.abs(result.s*result.y)))
    assert accepted["kkt"] == pytest.approx(residual, abs=1e-15)
    assert accepted["kkt"] == result.history.iloc[0]["KKT_res"]
    assert accepted["mu"] == result.history.iloc[0]["mu"]
    # The callback certificate uses actual constraint slack, not repaired s.
    expected_comp = np.sum(np.abs(result.y*(p.c-p.C @ x)))
    assert accepted["complementarity"] == pytest.approx(expected_comp, abs=1e-15)
    repair = result.diagnostics["iterations"][0]["repair"]
    before_s = initial_slack - np.array([result.history.iloc[0]["step"], 0.])
    delta = result.s-before_s
    assert repair["triggered"] and repair["count"] == 1
    assert repair["slack_delta_inf"] == pytest.approx(np.max(np.abs(delta)))
    assert repair["slack_delta_l1"] == pytest.approx(np.sum(np.abs(delta)))
    assert repair["primal_residual_delta_inf"] == pytest.approx(np.max(np.abs(delta)))
    assert repair["complementarity_delta_inf"] == pytest.approx(np.max(np.abs(result.y*delta)))
    assert repair["pre"]["primal_residual_inf"] == pytest.approx(np.max(np.abs(before_s-initial_slack)))
    assert repair["post"]["primal_residual_inf"] == pytest.approx(np.max(np.abs(result.s-initial_slack)))


def test_guard_diagnostics_identify_binding_previous_target(monkeypatch):
    p = GLQP(Q=eye_array(1), b=np.ones(1), C=eye_array(1), c=np.ones(1))
    zero_direction(p, monkeypatch)
    with pytest.warns(UserWarning, match="Maximum Iterations"):
        _, result = p.solve(x0=np.zeros(1), s0=np.ones(1), y0=np.ones(1), mu0=100.,
            verbose=False, settings=SolverSettings(max_iter=1, iteration_diagnostics=True))
    barrier = result.diagnostics["iterations"][0]["barrier"]
    assert barrier["previous_target"] == 100.
    assert barrier["raw_proposal"] == pytest.approx(.02**3)
    assert barrier["target"] == 10.
    assert barrier["mu_bar"] == barrier["xi"] == 1.
    assert barrier["guard_reference"] == "previous"
    assert barrier["guard_floor"] == 10.
    assert barrier["kkt_floor"] == .01
    assert tuple(barrier["active_reasons"]) == ("guard_floor",)


def test_direction_diagnostics_match_finite_difference_original_newton_equations(monkeypatch):
    p = GLQP(Q=csc_array([[3., .2], [.2, 2.]]), b=np.array([.4, -.3]),
        C=csc_array([[1., 2.], [-1., .5]]), c=np.array([2., 1.]),
        E=csc_array([[1., -1.]]), e=np.zeros(1))
    solve = p.solve_KKT
    captured = {}

    def capture(x, y, s, H, rx, rp, rc, req, mu, **kwargs):
        answer = solve(x, y, s, H, rx, rp, rc, req, mu, **kwargs)
        captured.update(x=x.copy(), y=y.copy(), s=s.copy(), mu=mu,
            rx=rx.copy(), rp=rp.copy(), rc=rc.copy(), req=req.copy(),
            rho=kwargs["prox_reg"], directions=[value.copy() for value in answer[:4]])
        return answer

    monkeypatch.setattr(p, "solve_KKT", capture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, result = p.solve(x0=np.array([.2, .2]), s0=np.array([.8, 1.2]),
            y0=np.array([.4, .7]), mu0=.2, verbose=False, settings=SolverSettings(
                max_iter=1, iteration_diagnostics=True, prox_reg=.03))
    dx, ds, dy, dnu = captured["directions"]
    x, s, y, mu = [captured[name] for name in ("x", "s", "y", "mu")]
    # Equality multipliers do not affect this derivative; use zero as an
    # arbitrary evaluation point and the actual residual RHS separately.
    def equations(alpha):
        xx, ss, yy = x+alpha*dx, s+alpha*ds, y+alpha*dy
        return (p.Q @ xx-p.b+p.C.T @ yy+p.E.T @ (alpha*dnu),
                p.C @ xx+ss-p.c, p.E @ xx-p.e, yy*ss-mu)

    step = 1e-5
    plus, minus = equations(step), equations(-step)
    rhs = [captured[name] for name in ("rx", "rp", "req", "rc")]
    defects = [(high-low)/(2*step)+residual for high, low, residual in zip(plus, minus, rhs)]
    names = ("stationarity", "primal", "equality", "complementarity")
    info = result.diagnostics["iterations"][0]["direction"]
    rho = captured["rho"]
    assert rho > 0.
    expected_regularization = (-rho*dx, rho*dy, rho*dnu, -rho*ds)
    for name, defect, expected in zip(names, defects, expected_regularization):
        assert info["original_newton_before_inf"][name] == pytest.approx(
            np.max(np.abs(defect), initial=0.), abs=5e-10)
        assert info["regularization_defect_inf"][name] == pytest.approx(
            np.max(np.abs(expected), initial=0.), abs=1e-15)
        assert_allclose(defect, expected, atol=5e-10, rtol=1e-6)


def test_direction_clipping_reports_counts_and_induced_block_errors(monkeypatch):
    s, y = np.array([1e-16, 1.]), np.array([.7, 1e-16])
    x = np.ones(2)
    p = GLQP(Q=eye_array(2), b=y, C=eye_array(2), c=x+s)
    dx, ds, dy = np.array([-.2, -.1]), np.array([-2e-15, .3]), np.array([-.02, -4e-15])
    monkeypatch.setattr(p, "solve_KKT", lambda *args, **kwargs:
        (dx.copy(), ds.copy(), dy.copy(), np.zeros(0), None, 0, 0.))
    with pytest.warns(UserWarning, match="Maximum Iterations"):
        _, result = p.solve(x0=x, s0=s, y0=y, mu0=1e-20, verbose=False,
            settings=SolverSettings(max_iter=1, iteration_diagnostics=True))
    info = result.diagnostics["iterations"][0]["direction"]
    delta_s = np.array([-s[0]-ds[0], 0.])
    delta_y = np.array([0., -y[1]-dy[1]])
    assert info["edited_ds_count"] == info["edited_dy_count"] == 1
    assert info["delta_ds_inf"] == pytest.approx(np.max(np.abs(delta_s)), abs=1e-30)
    assert info["delta_dy_inf"] == pytest.approx(np.max(np.abs(delta_y)), abs=1e-30)
    expected = dict(stationarity=np.max(np.abs(p.C.T @ delta_y)),
                    primal=np.max(np.abs(delta_s)), equality=0.,
                    complementarity=np.max(np.abs(y*delta_s+s*delta_y)))
    for name, value in expected.items():
        assert info["edit_induced_defect_inf"][name] == pytest.approx(value, abs=1e-30)
    assert info["before_norm_inf"]["ds"] == info["after_norm_inf"]["ds"] == .3
    # A block's maximum direction norm can stay unchanged despite edits.
    assert info["observed_defect_change_inf"]["primal"] > 0.


@pytest.mark.parametrize("max_trials,reason", [(8, "newton_fallback"), (3, None)])
def test_search_diagnostics_count_rejections_and_do_not_invent_accepted_iterates(monkeypatch, max_trials, reason):
    p = GLQP(Q=eye_array(1), b=np.zeros(1))
    monkeypatch.setattr(p, "solve_KKT", lambda *args, **kwargs:
        (np.ones(1), np.zeros(p.k), np.zeros(p.k), np.zeros(p.p), None, 0, 0.))
    stages = []
    with pytest.warns(UserWarning):
        _, result = p.solve(x0=np.ones(1), verbose=False,
            settings=SolverSettings(max_iter=1, max_linesearch_steps=max_trials,
                let_newton_cook=.5, iteration_diagnostics=True, armijo_additive_eps=0.),
            iteration_callback=lambda state: stages.append(state.stage))
    attempts = result.diagnostics["iterations"]
    assert len(attempts) == 1
    search = attempts[0]["line_search"]
    assert search["acceptance_reason"] == reason
    assert search["trial_count"] == max_trials
    assert search["rejection_count"] == (max_trials-1 if reason else max_trials)
    assert stages == (["initial", "accepted"] if reason else ["initial"])
    if reason:
        assert search["accepted_fraction"] == pytest.approx(.9**7)
        assert search["accepted_step"] == result.history.iloc[0]["step"]
    else:
        assert result.history.empty
