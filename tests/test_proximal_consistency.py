"""Independent residual, derivative, center-lifetime, and globalization checks."""
from dataclasses import FrozenInstanceError

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import csc_array
from scipy.special import expit

from glqp import GLQP, SolverSettings
from glqp.obj import GaussianNLL, LogisticNLL, PoissonNLL
from glqp.proximal import (
    FixedProximalModel, ProximalState, guard_proximal_repair, merit_value,
    residual_slope, search_proximal_step,
)
from glqp.util import get_step_size


def settings(**kwargs):
    return SolverSettings(proximal_policy='centered', **kwargs)


def test_fixed_center_residuals_include_offsets_with_correct_dual_signs():
    fixed = FixedProximalModel(np.array([1., -2.]), np.array([.4, .8]),
                              np.array([-.5]), .2, .3, 4)
    x, y, nu = np.array([2., -4.]), np.array([1.4, .3]), np.array([.5])
    rx, rp, req = np.array([.7, -.1]), np.array([.2, -.5]), np.array([.4])
    original = [v.copy() for v in (x, y, nu, rx, rp, req)]
    rd, rpi, re = fixed.residuals(x, y, nu, rx, rp, req)
    assert_allclose(rd, [.9, -.5])
    assert_allclose(rpi, [-.1, -.35])
    assert_allclose(re, [.1])
    for value, saved in zip((x, y, nu, rx, rp, req), original):
        assert_array_equal(value, saved)


@pytest.mark.parametrize('family', ['gaussian', 'logistic', 'poisson'])
def test_squared_merit_derivative_matches_independent_nonlinear_difference(family):
    a = np.array([[1., -.4], [.2, .9], [-.3, .7]])
    q = np.array([[.8, .1], [.1, .6]])
    c, e = np.array([[1., .2], [-.4, 1.]]), np.array([[.3, -.5]])
    weights, observations = np.array([.7, 1.3, .9]), np.array([0., 1., 1.])
    loss = dict(gaussian=GaussianNLL, logistic=LogisticNLL, poisson=PoissonNLL)[family]
    model = GLQP(f=loss(observations, weights), A=csc_array(a), Q=csc_array(q),
                 b=np.array([.2, -.3]), C=csc_array(c), c=np.array([.7, 1.1]),
                 E=csc_array(e), e=np.array([.2]))
    x, s, y, nu = np.array([.4, -.2]), np.array([.8, 1.2]), np.array([.5, .7]), np.array([.3])
    fixed = FixedProximalModel(np.array([-.1, .2]), np.array([.2, .8]),
                              np.array([-.2]), .17, .09, 2)
    dx, ds, dy, dnu = np.array([-.2, .6]), np.array([.3, -.4]), np.array([.1, .2]), np.array([-.1])
    target = .13

    def independent_residual(t):
        tx, ts, ty, tn = x+t*dx, s+t*ds, y+t*dy, nu+t*dnu
        z = a@tx
        loss_gradient = {'gaussian': lambda: weights*(z-observations),
                         'logistic': lambda: weights*(expit(z)-observations),
                         'poisson': lambda: weights*(np.exp(z)-observations)}[family]()
        return np.r_[a.T@loss_gradient + q@tx-model.b + c.T@ty+e.T@tn + .17*(tx-fixed.x_center),
                     c@tx+ts-model.c - .09*(ty-fixed.y_center),
                     ts*ty-target,
                     e@tx-model.e - .09*(tn-fixed.nu_center)]

    current = independent_residual(0.)
    scale = max(1., np.max(np.abs(current)))
    observed = residual_slope(model, model.get_H(a@x), dx, ds, dy, dnu, y, s,
                              current[:2], current[2:4], current[4:6], current[6:], .17, .09)
    step = 2e-6
    plus, minus = independent_residual(step)/scale, independent_residual(-step)/scale
    expected = (np.dot(plus, plus)-np.dot(minus, minus))/(4*step)
    assert_allclose(observed, expected, rtol=2e-8, atol=2e-10)


def test_residual_merit_uniform_scaling_prevents_squaring_overflow():
    blocks = (np.array([1e200, -2e200]), np.array([3e200]), np.zeros(0), np.zeros(0))
    assert merit_value(blocks, 3e200) == pytest.approx(7/9)


@pytest.mark.parametrize('stationarity,feasibility,move_x,move_dual',
                         [(8., 19.5, True, False), (9.8, 18., False, True),
                          (9.8, 19.5, False, False), (9., 18., True, True)])
def test_original_block_progress_updates_centers_independently(stationarity, feasibility, move_x, move_dual):
    state = ProximalState(np.zeros(1), np.ones(1), np.zeros(1),
                          np.array([10.]), np.array([20.]), np.zeros(1))
    fixed = state.freeze(.2, .3)
    x, y, nu = np.array([2.]), np.array([3.]), np.array([4.])
    diag = state.update(fixed, x, y, nu, np.array([stationarity]), np.array([feasibility]),
                        np.ones(1), np.zeros(1), .1, settings())
    assert diag['primal_center_updated'] is move_x
    assert diag['dual_centers_updated'] is move_dual
    assert not diag['inner_converged']
    assert_array_equal(state.x, x if move_x else np.zeros(1))
    assert_array_equal(state.y, y if move_dual else np.ones(1))
    assert_array_equal(state.nu, nu if move_dual else np.zeros(1))


def test_progress_reference_is_the_previous_point_not_the_last_center_update():
    state = ProximalState(np.zeros(1), np.ones(1), np.zeros(0),
                          np.array([10.]), np.array([20.]), np.zeros(0))
    for stationarity in (9.8, 9.4):
        diag = state.update(state.freeze(.2, .3), np.array([2.]), np.array([3.]), np.zeros(0),
                            np.array([stationarity]), np.array([20.]), np.ones(1),
                            np.zeros(0), .1, settings())
        assert not diag['primal_center_updated']
    assert diag['center_stationarity_before'] == 9.8


def test_inner_accuracy_advances_centers_when_original_reference_norms_are_zero():
    state = ProximalState(np.zeros(1), np.ones(1), np.zeros(1),
                          np.zeros(1), np.zeros(1), np.zeros(1))
    fixed = state.freeze(.2, .3)
    x, y, nu = np.array([2.]), np.array([3.]), np.array([4.])
    # This point solves the displaced proximal equations exactly, while all
    # original residual blocks are nonzero and cannot improve against zero.
    diag = state.update(fixed, x, y, nu, np.array([-.4]), np.array([.6]),
                        np.array([.05]), np.array([1.2]), .05, settings())
    assert diag['fixed_target_proximal_residual_inf'] < 1e-15
    assert diag['inner_converged']
    assert not diag['primal_progress'] and not diag['dual_progress']
    assert diag['refresh_reason'] == 'inner_accuracy'
    assert diag['primal_center_updated'] and diag['dual_centers_updated']
    assert_array_equal(state.x, x)
    assert_array_equal(state.y, y)
    assert_array_equal(state.nu, nu)


def test_freezing_centers_owns_copies_and_later_updates_preserve_old_model():
    x, y, nu = np.zeros(1), np.ones(1), np.zeros(1)
    state = ProximalState(x, y, nu, np.ones(1), np.ones(1), np.zeros(1))
    fixed = state.freeze(.2, .3)
    x[:] = 8.
    assert_array_equal(fixed.x_center, np.zeros(1))
    with pytest.raises(ValueError):
        fixed.y_center[:] = 7.
    with pytest.raises(FrozenInstanceError):
        fixed.rho = 9.
    state.update(fixed, np.array([2.]), np.array([3.]), np.array([4.]),
                  np.array([.5]), np.array([.5]), np.ones(1), np.zeros(1), .1, settings())
    assert state.generation == 1 and fixed.generation == 0
    assert_array_equal(fixed.x_center, np.zeros(1))
    assert_array_equal(fixed.y_center, np.ones(1))
    assert_array_equal(fixed.nu_center, np.zeros(1))


@pytest.mark.parametrize('candidate_rp,candidate_rc,accepted',
                         [(.2, 1., True), (1., 1., True), (.1, 3., False), (np.inf, 1., False)])
def test_slack_repair_must_preserve_the_accepted_fixed_center_merit(candidate_rp, candidate_rc, accepted):
    fixed = FixedProximalModel(np.zeros(1), np.ones(1), np.zeros(0), .2, .3, 0)
    ok, diag = guard_proximal_repair(fixed, np.zeros(1), np.ones(1), np.zeros(0),
                                    np.zeros(1), np.ones(1), np.ones(1), np.zeros(0),
                                    np.array([candidate_rp]), np.array([candidate_rc]),
                                    1., {'residual_scale': 2.})
    assert ok is accepted
    assert diag['proximal_merit_before'] == .125
    expected = (candidate_rp**2+(candidate_rc-1.)**2)/8
    assert diag['proximal_merit_candidate'] == pytest.approx(expected)


def test_poisson_search_uses_true_trial_gradient_and_fixed_hessian_and_centers():
    class ObservedPoisson(PoissonNLL):
        def __init__(self):
            super().__init__([30.], [1.])
            self.gradient_points, self.hessian_calls = [], 0

        def d1f(self, z):
            self.gradient_points.append(z.copy())
            return super().d1f(z)

        def d2f(self, z):
            self.hessian_calls += 1
            return super().d2f(z)

    loss = ObservedPoisson()
    model = GLQP(f=loss, A=csc_array([[1.]]), Q=csc_array([[.5]]),
                 C=csc_array([[-1.]]), c=np.array([3.]))
    model.settings = settings(max_linesearch_steps=100)
    x, s, y, nu = np.array([-2.]), np.array([10.]), np.array([.2]), np.zeros(0)
    z, qx, target = x.copy(), .5*x, .1
    gradient = np.exp(x)-30.+qx
    raw = (gradient-y, -x+s-3., s*y, np.zeros(0))
    fixed = FixedProximalModel(np.array([-1.]), np.array([.5]), np.zeros(0), .04, .03, 7)
    rd, rp, re = fixed.residuals(x, y, nu, raw[0], raw[1], raw[3])
    h = model.get_H(z)
    dense_jacobian = np.array([[float(h[0, 0])+.04, 0., -1.],
                              [-1., 1., -.03], [0., y[0], s[0]]])
    dense_direction = np.linalg.solve(dense_jacobian, -np.r_[rd, rp, raw[2]-target])
    dx, ds, dy = (dense_direction[i:i+1] for i in range(3))
    direction = dx, ds, dy, np.zeros(0)
    tmax = get_step_size(s, ds, y, dy, frac=.99)
    primal = loss(z)+.25*np.dot(x, x)
    before_hessian = loss.hessian_calls
    trial, diag = search_proximal_step(model, fixed, x=x, y=y, s=s, nu=nu, z=z,
        Qx=qx, primal=primal, H=h, residuals=raw, direction=direction,
        target=target, tmax=tmax)
    assert trial is not None and diag['trial_count'] > 1
    assert loss.hessian_calls == before_hessian
    assert len(loss.gradient_points) == diag['trial_count']
    assert diag['center_generation'] == 7 and diag['target'] == target
    assert_array_equal(fixed.x_center, [-1.])
    assert_array_equal(fixed.y_center, [.5])
    tx, ts, ty = trial.x, trial.s, trial.y
    true_rd = np.exp(tx)-30.+.5*tx-ty+.04*(tx+1.)
    true_rp = -tx+ts-3.-.03*(ty-.5)
    true_rc = ts*ty-target
    expected = .5*np.dot(np.r_[true_rd,true_rp,true_rc]/diag['residual_scale'],
                        np.r_[true_rd,true_rp,true_rc]/diag['residual_scale'])
    assert diag['merit_after'] == pytest.approx(expected, rel=1e-13)
    assert diag['merit_after'] <= diag['merit_before'] + model.settings.armijo_param*trial.step*diag['merit_slope']
    assert_allclose(trial.glm_gradient, np.exp(tx)-30.)
    linearized_gradient = np.exp(x)-30.+np.exp(x)*trial.step*dx
    assert np.linalg.norm(trial.glm_gradient-linearized_gradient) > .1


@pytest.mark.parametrize('field,value', [
    ('proximal_policy', 'typo'), ('proximal_policy', None),
    ('prox_reg', -1.), ('prox_reg', True), ('prox_reg', np.nan),
    ('prox_dual_reg', -1.), ('prox_dual_reg', True), ('prox_dual_reg', np.inf),
    ('proximal_center_reduction', 0.), ('proximal_center_reduction', 1.),
    ('proximal_inner_tolerance', 0.), ('proximal_inner_tolerance', np.nan),
    ('proximal_corrector_limit', 0.), ('proximal_corrector_limit', 1.),
    ('proximal_corrector_limit', np.nan), ('proximal_corrector_limit', True),
])
@pytest.mark.parametrize('mutated', [False, True])
def test_invalid_proximal_settings_are_rejected_at_construction_and_initialization(field, value, mutated):
    with pytest.raises(ValueError, match=field):
        if mutated:
            config = settings()
            setattr(config, field, value)
            GLQP(Q=csc_array([[1.]])).initialize(settings=config)
        else:
            SolverSettings(**{'proximal_policy':'centered', field:value})


def test_independent_dual_cap_requires_a_declared_nonlegacy_policy():
    with pytest.raises(ValueError, match='Independent dual regularization'):
        SolverSettings(prox_dual_reg=.1)
    for policy in ('no_slack', 'centered'):
        config = SolverSettings(proximal_policy=policy, prox_reg=0., prox_dual_reg=.1)
        assert config.prox_dual_reg == .1
