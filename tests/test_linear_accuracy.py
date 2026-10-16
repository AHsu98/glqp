"""Independent residual checks and controlled backend failures for refinement."""
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import csc_array, eye_array

from glqp import GLQP, SolverSettings
import glqp.linear as linear


def solve(G, rhs, **kwargs):
    n = len(rhs)
    return linear.factor_and_solve(csc_array(G), np.asarray(rhs, dtype=float),
        csc_array(eye_array(n)), kwargs.pop('init_tau_reg', 1e-6), None,
        target_atol=1e-14, **kwargs)


def test_reported_error_is_the_residual_of_the_returned_ill_conditioned_solution():
    rng = np.random.default_rng(1)
    basis = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    G = csc_array(basis @ np.diag([1e-7, -1e-3, 1., -1e3, 1e7]) @ basis.T)
    rhs = rng.normal(size=5)
    result = solve(G, rhs, max_refinement_steps=10)
    residual = rhs - G @ result.solution
    fresh = np.linalg.norm(residual) / np.linalg.norm(rhs)
    assert_allclose(result.info.relative_error, fresh, rtol=1e-13, atol=0.)
    assert result.info.absolute_error == np.max(np.abs(residual))
    expected = np.max(np.abs(residual)) <= 1e-14 or fresh <= 1e-6
    assert (result.info.status == 'accurate') == expected


def scripted_backend(monkeypatch, answers):
    calls = []
    class Factor:
        def __init__(self, matrix):
            pass
        def solve(self, rhs):
            calls.append(rhs.copy())
            return np.array(answers[min(len(calls)-1, len(answers)-1)], dtype=float)
    monkeypatch.setattr(linear, '_QDLDLFactorization', Factor)
    return calls


@pytest.mark.parametrize('rtol', [1e-6, 2.])
def test_rounding_worsens_a_correction_and_the_best_iterate_is_preserved(monkeypatch, rtol):
    # Even an exact correction in real arithmetic can lose low bits when added
    # to a large iterate. A recursive residual misses this deterioration.
    eps = np.finfo(float).eps
    G = csc_array([[1., 1.], [1., 1.+eps]])
    rhs = np.array([1., -2.])
    initial = np.array([2.**53, -2.**53])
    residual = rhs-G@initial
    correction = np.linalg.solve(G.toarray(), residual)
    assert np.linalg.norm(rhs-G@(initial+correction)) > np.linalg.norm(residual)
    calls = scripted_backend(monkeypatch, [initial, correction])
    result = solve(G, rhs, max_refinement_steps=10, target_rtol=rtol)
    assert_allclose(result.solution, initial, atol=0., rtol=0.)
    assert len(calls) == 2
    assert result.info.refinements == 1
    assert result.info.status == ('accurate' if rtol == 2. else 'inaccurate')
    assert result.info.stop_reason == ('stagnated' if rtol == 2. else 'arnoldi_breakdown')
    assert_allclose(result.info.relative_error, np.linalg.norm(residual)/np.linalg.norm(rhs))


@pytest.mark.parametrize('rhs,budget,expected_refinements', [
    ([1., 1e-8], 5, 1),       # Relative criterion already passes.
    ([1., 5e-15], 5, 1),      # Both pass, but absolute error triggers polishing.
    ([1e-12, 5e-16], 5, 1),   # Absolute error is tiny, but relative error is large.
    ([1., 5e-16], 5, 0),      # Neither first-polish condition triggers.
    ([1., 0.], 5, 0),         # An exact direction needs no correction.
    ([1., 1e-8], 0, 0),       # Polishing must respect a zero refinement budget.
])
def test_first_polish_restores_accuracy_without_exceeding_the_budget(
        monkeypatch, rhs, budget, expected_refinements):
    calls = scripted_backend(monkeypatch, [[rhs[0], 0.], [0., rhs[1]]])
    result = solve(np.eye(2), rhs, max_refinement_steps=budget)
    assert len(calls) == 1 + expected_refinements
    assert result.info.refinements == expected_refinements
    assert result.info.status == 'accurate'
    expected = rhs if expected_refinements else [rhs[0], 0.]
    assert_allclose(result.solution, expected, rtol=0., atol=0.)
    assert result.info.absolute_error == max(abs(np.asarray(rhs)-result.solution))


@pytest.mark.parametrize('budget,expected_calls', [(0, 1), (5, 3)])
def test_inaccurate_result_is_explicit_and_unproductive_refinement_is_bounded(monkeypatch, budget, expected_calls):
    calls = scripted_backend(monkeypatch, [[1., 0.]])
    result = solve(np.eye(2), [1., 1.], max_refinement_steps=budget,
                   residual_blocks={'first': slice(0, 1), 'second': slice(1, 2)})
    assert len(calls) == expected_calls
    assert result.info.status == 'inaccurate'
    assert result.info.absolute_error == 1.
    assert result.info.block_errors['first']['absolute'] == 0.
    assert result.info.block_errors['second']['rhs_scaled'] == 1.
    # Existing unpacking and indexed diagnostic probes remain supported.
    x, count, factor, relative = result
    assert result[0] is x and result[3] == relative
    assert count == expected_calls-1


def test_nonfinite_correction_does_not_replace_a_finite_iterate(monkeypatch):
    scripted_backend(monkeypatch, [[1., 0.], [np.nan, np.inf]])
    result = solve(np.eye(2), [1., 1.])
    assert_allclose(result.solution, [1., 0.])
    assert result.info.status == 'inaccurate'
    assert result.info.stop_reason == 'nonfinite_correction'


def test_numerical_backend_failure_retries_and_reports_the_actual_shift(monkeypatch):
    factory = linear.qdldl.Solver
    shifts = []
    def fail_once(matrix):
        shifts.append(matrix.diagonal()[0]-2.)
        if len(shifts) == 1:
            raise RuntimeError('injected numerical factorization failure')
        return factory(matrix)
    monkeypatch.setattr(linear.qdldl, 'Solver', fail_once)
    result = solve(np.diag([2., -3.]), [4., 3.], init_tau_reg=2e-8)
    assert result.info.attempts == 2
    assert result.info.tau_reg == 2e-7
    assert_allclose(shifts, [2e-8, 2e-7], atol=5e-16, rtol=0.)
    assert_allclose(result.solution, [2., -1.], atol=1e-6)


@pytest.mark.parametrize('error', [TimeoutError, TypeError, ValueError])
def test_timeout_and_programming_errors_do_not_retry_as_factorization_failures(monkeypatch, error):
    calls = []
    def fail(matrix):
        calls.append(1)
        raise error('injected non-numerical failure')
    monkeypatch.setattr(linear.qdldl, 'Solver', fail)
    with pytest.raises(error, match='injected'):
        solve([[2.]], [1.], max_solve_attempts=12)
    assert len(calls) == 1


@pytest.mark.parametrize('error', [TimeoutError, ValueError])
def test_outer_solve_does_not_relabel_timeouts_or_programming_errors(monkeypatch, error):
    problem = GLQP(Q=csc_array([[1.]]), b=np.array([1.]))
    def fail(*args, **kwargs):
        raise error('injected Newton failure')
    monkeypatch.setattr(problem, 'solve_KKT', fail)
    with pytest.raises(error, match='injected Newton'):
        problem.solve(settings=SolverSettings(), verbose=False)


def test_numerical_failures_are_still_reported_as_linear_failures(monkeypatch):
    problem = GLQP(Q=csc_array([[1.]]), b=np.array([1.]))
    def fail(*args, **kwargs):
        raise linear.LinearSolveError('injected numerical failure')
    monkeypatch.setattr(problem, 'solve_KKT', fail)
    with pytest.warns(UserWarning, match='Failed Linear Solve'):
        _, result = problem.solve(settings=SolverSettings(), verbose=False)
    assert result.termination_tag == 'failed_linear_solve'
    assert isinstance(result.exception, linear.LinearSolveError)


def test_actual_equation_blocks_and_initialization_are_exposed_in_results():
    p = GLQP(Q=csc_array(np.eye(2)), b=np.array([1., 2.]),
             C=csc_array([[1., 0.], [0., 1.]]), c=np.array([10., 10.]),
             E=csc_array([[1., 1.]]), e=np.array([2.]))
    x, result = p.solve(verbose=False)
    assert result.termination_tag == 'optimal'
    assert_allclose(x, [.5, 1.5], atol=1e-6)
    assert result.initialization_linear_solve['lin_status'] == 'accurate'
    assert {'lin_status', 'lin_tau', 'lin_stationarity_abs', 'lin_scaled_inequality_abs',
            'lin_equality_abs'} <= set(result.history.columns)
    last = result.history.iloc[-1]
    assert last['lin_rel_res'] == result.last_linear_solve['lin_rel_res']
    assert last['lin_abs_res'] == max(last['lin_stationarity_abs'],
                                    last['lin_scaled_inequality_abs'], last['lin_equality_abs'])
