"""Controlled outer-policy experiments preserve certificates and observations."""
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import csc_array, eye_array

from glqp import GLQP, SolverSettings


def install_direction(monkeypatch, problem, dx=None, ds=None, dy=None):
    captured = []
    values = (np.zeros(problem.n) if dx is None else np.asarray(dx, dtype=float),
              np.zeros(problem.k) if ds is None else np.asarray(ds, dtype=float),
              np.zeros(problem.k) if dy is None else np.asarray(dy, dtype=float),
              np.zeros(problem.p))

    def direction(x, y, s, H, rx, rp, rc, req, mu, **kwargs):
        captured.append(float(mu))
        return (*(value.copy() for value in values), None, 0, 0.)

    monkeypatch.setattr(problem, "solve_KKT", direction)
    return captured


def solve_one(problem, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return problem.solve(verbose=False, **kwargs)


@pytest.mark.parametrize("policy,scale,override,expected", [
    ("fixed", 1., None, 100.),
    ("complementarity", 1., None, 1.4),
    ("complementarity", .25, None, .35),
    ("complementarity", 1., 17., 17.),
    ("complementarity", .25, 0., 0.),
])
def test_initial_mu_uses_initialized_products_and_respects_explicit_override(
        monkeypatch, policy, scale, override, expected):
    p = GLQP(Q=eye_array(1), C=csc_array([[1.], [-1.]]), c=np.array([.2, .8]))
    targets = install_direction(monkeypatch, p)
    observed = []
    solve_one(p, x0=np.zeros(1), s0=np.array([.2, .8]), y0=np.array([2., 3.]),
        mu0=override, settings=SolverSettings(max_iter=1, initial_mu_policy=policy,
            initial_mu_scale=scale, iteration_diagnostics=True),
        iteration_callback=lambda state: observed.append(state.barrier_target))
    assert targets[0] == pytest.approx(expected)
    assert observed[0] == targets[0]


def test_complementarity_initialization_has_minimum_floor(monkeypatch):
    p = GLQP(Q=eye_array(1), b=np.ones(1), C=eye_array(1), c=np.ones(1))
    targets = install_direction(monkeypatch, p)
    solve_one(p, x0=np.zeros(1), s0=np.array([1e-6]), y0=np.array([1e-6]),
        settings=SolverSettings(max_iter=1, initial_mu_policy="complementarity", min_mu=1e-5))
    assert targets[0] == 1e-5


def test_overflowed_complementarity_initialization_fails_before_a_newton_solve(monkeypatch):
    p = GLQP(Q=eye_array(1), C=eye_array(1), c=np.ones(1))
    targets = install_direction(monkeypatch, p)
    with pytest.raises(ValueError, match="initial_mu_policy.*nonfinite"):
        p.solve(x0=np.zeros(1), s0=np.array([1e200]), y0=np.array([1e200]),
            settings=SolverSettings(initial_mu_policy="complementarity"), verbose=False)
    assert not targets


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("override", [None, 7.])
def test_initial_mu_policy_handles_no_inequalities(monkeypatch, empty, override):
    kwargs = dict(C=csc_array((0, 1)), c=np.zeros(0)) if empty else {}
    p = GLQP(Q=eye_array(1), b=np.ones(1), **kwargs)
    targets = install_direction(monkeypatch, p)
    _, result = solve_one(p, mu0=override, settings=SolverSettings(max_iter=1,
        initial_mu_policy="complementarity", initial_mu_scale=4.,
        min_mu=2e-12, iteration_diagnostics=True))
    assert targets[0] == (2e-12 if override is None else override)
    assert result.history.iloc[0]["mu"] == 2e-12


@pytest.mark.parametrize("coefficient", [0., .001, .01, .2])
def test_kkt_floor_changes_next_barrier_target_without_changing_raw_proposal(monkeypatch, coefficient):
    p = GLQP(Q=eye_array(1), b=np.ones(1), C=eye_array(1), c=np.ones(1))
    install_direction(monkeypatch, p)
    _, result = solve_one(p, x0=np.zeros(1), s0=np.ones(1), y0=np.ones(1),
        settings=SolverSettings(max_iter=1, barrier_update_guard="legacy",
            barrier_kkt_floor=coefficient, iteration_diagnostics=True))
    update = result.diagnostics["iterations"][0]["barrier"]
    assert update["raw_proposal"] == pytest.approx(.02**3)
    assert update["kkt_floor"] == coefficient
    assert update["target"] == pytest.approx(max(.02**3, coefficient))


INVALID_OPTIONS = [
    ("initial_mu_policy", "bad"), ("initial_mu_policy", None),
    ("initial_mu_scale", 0.), ("initial_mu_scale", -1.),
    ("initial_mu_scale", np.inf), ("initial_mu_scale", True),
    ("barrier_kkt_floor", -1.), ("barrier_kkt_floor", np.nan),
    ("barrier_kkt_floor", np.inf), ("barrier_kkt_floor", False),
    ("slack_repair", "bad"), ("slack_repair", None),
    ("repair_growth_limit", .99), ("repair_growth_limit", np.inf),
    ("repair_growth_limit", True), ("newton_fallback_policy", "bad"),
    ("newton_fallback_policy", None),
]


@pytest.mark.parametrize("name,value", INVALID_OPTIONS)
@pytest.mark.parametrize("mutate", [False, True])
def test_invalid_policy_settings_fail_on_construction_or_solve_entry(name, value, mutate):
    with pytest.raises(ValueError, match=name):
        if mutate:
            settings = SolverSettings()
            setattr(settings, name, value)
        else:
            settings = SolverSettings(**{name: value})
        GLQP(Q=eye_array(1), b=np.ones(1)).solve(settings=settings, verbose=False)


def repair_problem(monkeypatch, *, inconsistent=False, overflow=False):
    s = np.array([.01 if overflow else 1e-8, 1.])
    y = np.array([np.nextafter(0., 1.), 1.]) if overflow else np.full(2, 1e-4)
    c = s - (np.array([1., 0.]) if inconsistent else 0.)
    p = GLQP(Q=eye_array(2), b=y, C=eye_array(2), c=c)
    install_direction(monkeypatch, p, ds=[-1000. if overflow else -1., 0.])
    return p, s, y


def repair_solve(monkeypatch, policy, *, inconsistent=False, overflow=False, growth=1.1):
    p, s, y = repair_problem(monkeypatch, inconsistent=inconsistent, overflow=overflow)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, result = solve_one(p, x0=np.zeros(2), s0=s, y0=y, mu0=1e-12,
            settings=SolverSettings(max_iter=1, slack_repair=policy,
                repair_growth_limit=growth, iteration_diagnostics=True))
    return p, x, result


def test_rejected_guarded_repair_retains_the_unmodified_accepted_state(monkeypatch):
    p, off_x, off = repair_solve(monkeypatch, "off")
    _, guarded_x, guarded = repair_solve(monkeypatch, "guarded")
    _, legacy_x, legacy = repair_solve(monkeypatch, "legacy")
    assert_array_equal(guarded_x, off_x)
    for name in ("y", "s", "nu"):
        assert_array_equal(getattr(guarded, name), getattr(off, name))
    assert legacy.s[0] > 1e6*off.s[0]
    row = guarded.diagnostics["iterations"][0]
    repair = row["repair"]
    assert repair["triggered"] and repair["proposed"] and repair["rejected"]
    assert repair["proposed_count"] == 1
    assert repair["count"] == 0 and not repair["applied"]
    assert repair["slack_delta_inf"] == repair["primal_residual_delta_inf"] == 0.
    assert repair["candidate_primal_inf"] > repair["primal_limit"]
    assert guarded.cons_viol == pytest.approx(np.max(np.abs(p.C @ guarded_x+guarded.s-p.c)))
    disabled = off.diagnostics["iterations"][0]["repair"]
    assert not disabled["proposed"] and not disabled["applied"]
    assert disabled["rejection_reason"] == "disabled"


def test_guarded_repair_allows_bounded_candidate_and_growth_limit_is_effective(monkeypatch):
    _, _, legacy = repair_solve(monkeypatch, "legacy", inconsistent=True)
    _, _, allowed = repair_solve(monkeypatch, "guarded", inconsistent=True, growth=1.1)
    _, _, rejected = repair_solve(monkeypatch, "guarded", inconsistent=True, growth=1.)
    assert_array_equal(allowed.s, legacy.s)
    repair = allowed.diagnostics["iterations"][0]["repair"]
    assert repair["proposed"] and repair["applied"] and not repair["rejected"]
    assert repair["count"] == repair["proposed_count"] == 1
    assert repair["candidate_primal_inf"] <= repair["primal_limit"]
    assert repair["candidate_kkt_inf"] <= repair["kkt_limit"]
    assert rejected.s[0] < 1e-6*allowed.s[0]
    assert rejected.diagnostics["iterations"][0]["repair"]["rejected"]


def test_nonfinite_repair_candidate_cannot_corrupt_an_accepted_point(monkeypatch):
    _, off_x, off = repair_solve(monkeypatch, "off", overflow=True)
    _, guarded_x, guarded = repair_solve(monkeypatch, "guarded", overflow=True)
    assert_array_equal(guarded_x, off_x)
    assert_array_equal(guarded.s, off.s)
    assert_array_equal(guarded.y, off.y)
    assert np.isfinite(guarded.s).all() and np.all(guarded.s > 0.)
    repair = guarded.diagnostics["iterations"][0]["repair"]
    assert repair["proposed"] and repair["rejected"] and not repair["applied"]
    assert not repair["candidate_finite_positive"]
    assert repair["count"] == 0


@pytest.mark.parametrize("policy,expected", [("finite", True), ("progress", True), ("off", False)])
def test_fallback_progress_uses_current_perturbed_target_even_when_unperturbed_norm_grows(
        monkeypatch, policy, expected):
    # Objective and unperturbed KKT both worsen: x and y increase from one.
    # With mu=10, however, complementarity residual y*s-mu contracts from -9.
    p = GLQP(Q=eye_array(1), b=np.zeros(1), C=csc_array((1, 1)), c=np.ones(1))
    install_direction(monkeypatch, p, dx=[1.], ds=[0.], dy=[1.])
    _, result = solve_one(p, x0=np.ones(1), s0=np.ones(1), y0=np.ones(1), mu0=10.,
        settings=SolverSettings(max_iter=1, max_linesearch_steps=8, let_newton_cook=.5,
            newton_fallback_policy=policy, armijo_additive_eps=0., iteration_diagnostics=True))
    row = result.diagnostics["iterations"][0]
    assert (len(result.history) == 1) == expected
    if expected:
        step = result.history.iloc[0]["step"]
        assert step == pytest.approx(.9**7)
        assert result.history.iloc[0]["KKT_res"] > 1.
    if policy == "progress":
        checks = row["line_search"]["fallback_progress_checks"]
        assert len(checks) == 1 and checks[0]["accepted"]
        assert checks[0]["previous_perturbed_kkt"] == 9.
        assert checks[0]["trial_perturbed_kkt"] == pytest.approx(9.-.9**7)
        assert checks[0]["contraction_limit"] == pytest.approx((1.-.005*.9**7)*9.)


@pytest.mark.parametrize("policy,accepted", [("finite", True), ("progress", False), ("off", False)])
def test_progress_fallback_rejects_finite_trial_without_perturbed_contraction(monkeypatch, policy, accepted):
    p = GLQP(Q=eye_array(1), b=np.zeros(1), C=csc_array((1, 1)), c=np.ones(1))
    install_direction(monkeypatch, p, dx=[1.], ds=[0.], dy=[0.])
    _, result = solve_one(p, x0=np.ones(1), s0=np.ones(1), y0=np.ones(1), mu0=10.,
        settings=SolverSettings(max_iter=1, max_linesearch_steps=8, let_newton_cook=.5,
            newton_fallback_policy=policy, armijo_additive_eps=0., iteration_diagnostics=True))
    assert (len(result.history) == 1) == accepted
    if policy == "progress":
        search = result.diagnostics["iterations"][0]["line_search"]
        assert search["fallback_progress_rejection_count"] == 1
        check = search["fallback_progress_checks"][0]
        assert check["previous_perturbed_kkt"] == check["trial_perturbed_kkt"] == 9.
        assert not check["accepted"]


class QuadraticLoss:
    hessian_is_constant = True

    def __call__(self, z):
        return .5*np.dot(z, z)

    def d1f(self, z):
        return z

    def d2f(self, z):
        return np.ones_like(z)


@pytest.mark.parametrize("policy,accepted", [("finite", True), ("progress", False)])
def test_unperturbed_improvement_does_not_stand_in_for_progress_fallback(monkeypatch, policy, accepted):
    # Nonlinear-path descent disables the separate unperturbed-residual route.
    # For the first eight trials Armijo fails; at the fallback trial sy falls
    # from100, but residual at mu=100 rises as stationarity grows from2.
    p = GLQP(f=QuadraticLoss(), A=eye_array(1), Q=eye_array(1),
             C=csc_array((1, 1)), c=np.ones(1))
    install_direction(monkeypatch, p, dx=[-10.], ds=[0.], dy=[-1.])
    _, result = solve_one(p, x0=np.ones(1), s0=np.ones(1), y0=np.array([100.]), mu0=100.,
        settings=SolverSettings(max_iter=1, max_linesearch_steps=8, let_newton_cook=.5,
            newton_fallback_policy=policy, armijo_additive_eps=0., iteration_diagnostics=True))
    assert (len(result.history) == 1) == accepted
    search = result.diagnostics["iterations"][0]["line_search"]
    assert search["merit_slope"] < 0.
    if accepted:
        assert result.history.iloc[0]["KKT_res"] < 100.
    else:
        check = search["fallback_progress_checks"][0]
        assert check["previous_perturbed_kkt"] == 2.
        assert check["trial_perturbed_kkt"] == pytest.approx(abs(2.*(1.-10.*.9**7)))
        assert not check["accepted"]


@pytest.mark.parametrize("options", [
    dict(initial_mu_policy="complementarity", initial_mu_scale=.5),
    dict(barrier_update_guard="previous", barrier_kkt_floor=0.),
    dict(slack_repair="off"), dict(slack_repair="guarded", repair_growth_limit=1.05),
    dict(newton_fallback_policy="progress"), dict(newton_fallback_policy="off"),
])
def test_diagnostics_do_not_change_new_policy_trajectories(options):
    kwargs = dict(Q=eye_array(2), b=np.array([1., -.5]),
                  C=csc_array([[1., 0.], [0., -1.]]), c=np.array([.2, .4]))
    outcomes = [solve_one(GLQP(**kwargs), settings=SolverSettings(
        max_iter=120, iteration_diagnostics=enabled, **options)) for enabled in (False, True)]
    (x0, r0), (x1, r1) = outcomes
    assert r0.termination_tag == r1.termination_tag
    assert_array_equal(x0, x1)
    for name in ("y", "s", "nu"):
        assert_array_equal(getattr(r0, name), getattr(r1, name))
    columns = [name for name in r0.history if name != "time"]
    assert_array_equal(r0.history[columns], r1.history[columns])


def test_existing_defaults_match_explicit_legacy_policy_options():
    legacy = dict(initial_mu_policy="fixed", initial_mu_scale=1., barrier_kkt_floor=.01,
                  slack_repair="legacy", repair_growth_limit=1.1, newton_fallback_policy="finite")
    p = dict(Q=eye_array(2), b=np.array([1., -.5]),
             C=csc_array([[1., 0.], [0., -1.]]), c=np.array([.2, .4]))
    (x0, r0), (x1, r1) = [solve_one(GLQP(**p), settings=SolverSettings(**options))
                          for options in ({}, legacy)]
    assert_array_equal(x0, x1)
    for name in ("y", "s", "nu"):
        assert_array_equal(getattr(r0, name), getattr(r1, name))
    assert_array_equal(r0.history.drop(columns="time"), r1.history.drop(columns="time"))
