"""QP proposals must preserve the solver's independent acceptance routes."""
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from scipy.sparse import csc_array

from glqp import GLQP, SolverSettings
from glqp.obj import LogisticNLL
import glqp.solver as solver_module


def fixed_direction(monkeypatch, problem, dx, ds=None, dy=None):
    monkeypatch.setattr(problem, 'solve_KKT', lambda *args, **kwargs:
        (np.asarray(dx), np.zeros(problem.k) if ds is None else np.asarray(ds),
         np.zeros(problem.k) if dy is None else np.asarray(dy),
         np.zeros(problem.p), None, 0, 0.))


def run_one(problem, **options):
    with pytest.warns(UserWarning, match='Maximum Iterations'):
        return problem.solve(verbose=False, settings=SolverSettings(
            qp_linesearch='scalar', max_iter=1, armijo_additive_eps=0.,
            iteration_diagnostics=True, **options), x0=np.ones(problem.n))


def test_rejected_qp_trials_use_scalar_proposals_after_residual_rejection(monkeypatch):
    problem = GLQP(Q=csc_array([[1.]]))
    fixed_direction(monkeypatch, problem, [-40.])
    x, result = run_one(problem)
    search = result.diagnostics['iterations'][0]['line_search']
    assert search['algorithm'] == 'qp_scalar'
    assert search['trial_count'] < 10
    assert search['residual_check_count'] >= 1  # full trial gets restoration chance
    assert search['acceptance_reason'] == 'armijo'
    assert search['qp_scalar_counters']['derivative_evaluations'] >= 1
    assert abs(x[0]) < 1


@pytest.mark.parametrize('screen', ['off', 'complementarity', 'bounded'])
def test_full_step_restoration_never_constructs_scalar_model(monkeypatch, screen):
    problem = GLQP(Q=csc_array([[1.]]), C=csc_array([[1.]]), c=np.array([10.]))
    fixed_direction(monkeypatch, problem, [-4.], [8.], [-.99])
    def unexpected(*args, **kwargs):
        raise AssertionError('An accepted full restoration step needs no model')
    monkeypatch.setattr(solver_module, 'QPScalarProposer', unexpected)
    with pytest.warns(UserWarning, match='Maximum Iterations'):
        x, result = problem.solve(x0=np.ones(1), s0=np.ones(1), y0=np.ones(1),
            mu0=.001, verbose=False, settings=SolverSettings(qp_linesearch='scalar',
                qp_residual_screen=screen, max_iter=1, armijo_additive_eps=0.,
                iteration_diagnostics=True))
    search = result.diagnostics['iterations'][0]['line_search']
    assert search['merit_slope'] < 0
    assert search['accepted_step'] == 1.
    assert search['acceptance_reason'] == 'residual'
    assert search['qp_scalar_counters'] is None
    assert_array_equal(x, [-3.])


def test_non_descent_qp_keeps_legacy_restoration(monkeypatch):
    problem = GLQP(Q=csc_array([[1.]]), C=csc_array([[1.]]), c=np.ones(1))
    fixed_direction(monkeypatch, problem, [-.5], [-.5])
    with pytest.warns(UserWarning, match='Maximum Iterations'):
        x, result = problem.solve(x0=np.array([2.]), s0=np.ones(1), verbose=False,
            settings=SolverSettings(qp_linesearch='scalar', max_iter=1,
                                    iteration_diagnostics=True))
    search = result.diagnostics['iterations'][0]['line_search']
    assert search['merit_slope'] > 0
    assert search['algorithm'] == 'backtracking'
    assert search['acceptance_reason'] == 'residual'
    assert_array_equal(x, [1.5])


def test_scalar_proposal_retains_and_reports_finite_fallback(monkeypatch):
    problem = GLQP(Q=csc_array([[1.]]))
    fixed_direction(monkeypatch, problem, [-400.])
    x, result = run_one(problem, let_newton_cook=.5)
    search = result.diagnostics['iterations'][0]['line_search']
    assert search['acceptance_reason'] == 'newton_fallback'
    assert search['fallback_eligible_count'] == 1
    assert any(p['crosses_fallback'] for p in search['qp_scalar_proposals'])
    assert .5*x[0]**2 > .5  # Existing finite fallback can accept a merit increase.


def test_progress_fallback_still_rejects_a_worsening_scalar_trial(monkeypatch):
    problem = GLQP(Q=csc_array([[1.]]))
    fixed_direction(monkeypatch, problem, [-400.])
    x, result = run_one(problem, let_newton_cook=.5, newton_fallback_policy='progress')
    search = result.diagnostics['iterations'][0]['line_search']
    assert search['fallback_progress_rejection_count'] >= 1
    assert search['rejection_reasons']['fallback_no_progress'] >= 1
    assert abs(x[0]) < 1


def test_qp_scalar_option_does_not_change_glm_search():
    results = []
    for mode in ('backtracking', 'scalar'):
        problem = GLQP(f=LogisticNLL([0., 1.], [.9, .1]),
            A=csc_array([[1.], [1.]]), Q=csc_array([[1e-4]]))
        results.append(problem.solve(x0=np.array([5.]), verbose=False,
            settings=SolverSettings(qp_linesearch=mode, iteration_diagnostics=True))[1])
    a, b = results
    for name in ('x', 's', 'y', 'nu'):
        assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.termination_tag == b.termination_tag == 'optimal'
    assert len(a.history) == len(b.history)
    assert all(i['line_search']['algorithm'] == 'scalar' for i in b.diagnostics['iterations'])


def test_qp_scalar_option_does_not_override_centered_merit():
    results = []
    for mode in ('backtracking', 'scalar'):
        problem = GLQP(Q=csc_array([[1.]]), b=np.array([2.]),
                       C=csc_array([[1.]]), c=np.array([1.]))
        results.append(problem.solve(verbose=False,
            settings=SolverSettings(qp_linesearch=mode, proximal_policy='centered',
                iteration_diagnostics=True))[1])
    a, b = results
    for name in ('x', 's', 'y', 'nu'):
        assert_array_equal(getattr(a, name), getattr(b, name))
    assert a.termination_tag == b.termination_tag == 'optimal'
    assert len(a.history) == len(b.history)
    assert all('qp_scalar_proposals' not in i['line_search'] for i in b.diagnostics['iterations'])


@pytest.mark.parametrize('bad', [None, True, 1, '', 'unknown'])
def test_invalid_qp_search_setting_rejected(bad):
    with pytest.raises(ValueError, match='qp_linesearch'):
        SolverSettings(qp_linesearch=bad)


def test_mutated_qp_search_setting_rejected():
    settings = SolverSettings()
    settings.qp_linesearch = 'unknown'
    with pytest.raises(ValueError, match='qp_linesearch'):
        GLQP(Q=csc_array([[1.]])).solve(settings=settings, verbose=False)


def test_qp_scalar_diagnostics_preserve_solution_and_history():
    results = []
    for diagnostics in (False, True):
        problem = GLQP(Q=csc_array([[2., .5], [.5, 1.]]), b=np.array([2., 3.]),
                       C=csc_array([[1., 1.], [-1., 0.]]), c=np.array([1., 0.]))
        results.append(problem.solve(verbose=False,
            settings=SolverSettings(qp_linesearch='scalar',
                iteration_diagnostics=diagnostics))[1])
    plain, observed = results
    assert plain.termination_tag == observed.termination_tag == 'optimal'
    for name in ('x', 's', 'y', 'nu'):
        a, b = getattr(plain, name), getattr(observed, name)
        assert a.tobytes() == b.tobytes()
    columns = [c for c in plain.history.columns if 'time' not in c.lower()]
    assert plain.history[columns].equals(observed.history[columns])


def test_complementarity_screen_saves_products_with_exact_fallback_parity(monkeypatch):
    outputs, calls = [], []
    for screen in ('off', 'complementarity'):
        problem = GLQP(Q=csc_array([[1.]]), C=csc_array([[1.]]), c=np.array([10.]))
        fixed_direction(monkeypatch, problem, [-40.], [100.], [100.])
        product_calls = []
        original = problem._constraint_products
        def counted(*args):
            product_calls.append(1)
            return original(*args)
        monkeypatch.setattr(problem, '_constraint_products', counted)
        with pytest.warns(UserWarning, match='Maximum Iterations'):
            _, result = problem.solve(x0=np.ones(1), s0=np.ones(1), y0=np.ones(1),
                mu0=.001, verbose=False, settings=SolverSettings(max_iter=1,
                    qp_residual_screen=screen, let_newton_cook=.5,
                    armijo_additive_eps=0., iteration_diagnostics=True))
        outputs.append(result)
        calls.append(len(product_calls))
    plain, screened = outputs
    for name in ('x', 'y', 's', 'nu'):
        assert getattr(plain, name).tobytes() == getattr(screened, name).tobytes()
    old, new = [r.diagnostics['iterations'][0]['line_search'] for r in outputs]
    assert old['accepted_step'] == new['accepted_step']
    assert old['trial_count'] == new['trial_count'] == 8
    assert old['acceptance_reason'] == new['acceptance_reason'] == 'newton_fallback'
    assert new['complementarity_rejections'] == 7
    assert calls[0]-calls[1] == 7
    assert old['residual_check_count'] == 8 and new['residual_check_count'] == 1


@pytest.mark.parametrize('policy', ['finite', 'progress'])
def test_fallback_eligible_trials_always_get_fresh_residuals(monkeypatch, policy):
    problem = GLQP(Q=csc_array([[1.]]), C=csc_array([[1.]]), c=np.array([10.]))
    fixed_direction(monkeypatch, problem, [-40.], [100.], [100.])
    with pytest.warns(UserWarning):
        _, result = problem.solve(x0=np.ones(1), s0=np.ones(1), y0=np.ones(1),
            mu0=.001, verbose=False, settings=SolverSettings(max_iter=1,
                qp_residual_screen='complementarity', let_newton_cook=2.,
                newton_fallback_policy=policy, iteration_diagnostics=True))
    search = result.diagnostics['iterations'][0]['line_search']
    assert search['residual_screen_checks'] == 0
    assert search['complementarity_rejections'] == 0
    assert search['residual_check_count'] > 0


@pytest.mark.parametrize('bad', [None, True, 1, '', 'unknown'])
def test_invalid_qp_residual_screen_rejected(bad):
    with pytest.raises(ValueError, match='qp_residual_screen'):
        SolverSettings(qp_residual_screen=bad)


def test_mutated_qp_residual_screen_rejected():
    settings = SolverSettings()
    settings.qp_residual_screen = 'unknown'
    with pytest.raises(ValueError, match='qp_residual_screen'):
        GLQP(Q=csc_array([[1.]])).solve(settings=settings, verbose=False)


def test_affine_screen_preserves_fallback_and_avoids_rejected_fresh_checks(monkeypatch):
    results = []
    for screen in ('off', 'bounded'):
        problem = GLQP(Q=csc_array([[1.]]))
        fixed_direction(monkeypatch, problem, [-40.])
        with pytest.warns(UserWarning, match='Maximum Iterations'):
            _, result = problem.solve(x0=np.ones(1), verbose=False,
                settings=SolverSettings(max_iter=1, qp_residual_screen=screen,
                    let_newton_cook=.5, armijo_additive_eps=0., iteration_diagnostics=True))
        results.append(result)
    for name in ('x', 'y', 's', 'nu'):
        assert getattr(results[0], name).tobytes() == getattr(results[1], name).tobytes()
    old, new = [r.diagnostics['iterations'][0]['line_search'] for r in results]
    assert old['trial_count'] == new['trial_count'] == 8
    assert old['accepted_step'] == new['accepted_step']
    assert old['acceptance_reason'] == new['acceptance_reason'] == 'newton_fallback'
    assert new['complementarity_rejections'] == 0
    assert new['affine_cache_initializations'] == 1
    assert new['affine_rejections'] == 6
    assert old['residual_check_count'] == 8 and new['residual_check_count'] == 2
