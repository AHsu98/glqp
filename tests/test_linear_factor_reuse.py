"""A predictor/corrector pair shares factors, but verifies each requested RHS."""
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import csc_array

import glqp.linear as linear


@pytest.fixture
def counting_factor(monkeypatch):
    original = linear._QDLDLFactorization

    class CountingFactor(original):
        constructions = 0
        updates = 0
        applications = 0
        releases = 0
        fail_solve = False

        def __init__(self, matrix, **kwargs):
            type(self).constructions += 1
            super().__init__(matrix, **kwargs)

        def update(self, matrix):
            type(self).updates += 1
            super().update(matrix)

        def solve(self, rhs):
            type(self).applications += 1
            if self.fail_solve:
                raise linear._FactorizationError('injected reused-factor failure')
            return super().solve(rhs)

        def close(self):
            type(self).releases += 1
            super().close()

    monkeypatch.setattr(linear, '_QDLDLFactorization', CountingFactor)
    return CountingFactor


def example():
    return (csc_array([[3., .25, .5], [.25, 2., -.1], [.5, -.1, -1.]]),
            csc_array(np.diag([1., 1., -1.])))


def test_two_rhs_share_one_numeric_factor_and_next_context_updates(counting_factor):
    G, shift = example()
    workspace = linear.ShiftedSystemWorkspace()
    rhs = np.array([1., 2., -3.])
    with linear.VerifiedLinearSystem(G, shift, 1e-5, workspace=workspace) as system:
        first = system.solve_verified(rhs)
        second = system.solve_verified(rhs[::-1])
        factor = system.factorization
        assert first.factorization is second.factorization is factor
        assert first.info.attempts == 1
        assert second.info.attempts == 0
        assert counting_factor.constructions == 1
        assert counting_factor.updates == 0
        assert counting_factor.applications == 2 + first.info.refinements + second.info.refinements
    assert system.factorization is factor
    assert counting_factor.releases == 0
    # The previous context ended, so a new outer iteration can change G.
    G.data *= 1.25
    with linear.VerifiedLinearSystem(G, shift, 1e-5, factor, workspace=workspace) as system:
        third = system.solve_verified(rhs)
        fourth = system.solve_verified(-rhs)
    assert counting_factor.constructions == 1
    assert counting_factor.updates == 1
    assert third.factorization is fourth.factorization is factor
    assert third.info.attempts == 1 and fourth.info.attempts == 0
    assert_allclose(G @ third.solution, rhs, atol=1e-10, rtol=1e-6)


@pytest.mark.parametrize('backend', ['qdldl', 'superlu'])
@pytest.mark.parametrize('tau', [0., 1e-8, .5])
def test_default_arithmetic_and_all_accuracy_fields_are_preserved(backend, tau):
    G, shift = example()
    options = dict(backend=backend, target_atol=1e-13, target_rtol=1e-10,
                   residual_blocks={'primal': slice(0, 2), 'dual': slice(2, None)})
    rhss = [np.array([1., -2., 3.]), np.array([2., 4., -1.])]
    ordinary = [linear.factor_and_solve(G, rhs, shift, tau, None, **options) for rhs in rhss]
    with linear.VerifiedLinearSystem(G, shift, tau, **options) as system:
        actual = [system.solve_verified(rhs) for rhs in rhss]
    for i, (direct, reused) in enumerate(zip(ordinary, actual)):
        assert_array_equal(reused.solution, direct.solution)
        assert reused.info == replace(direct.info, attempts=0 if i else direct.info.attempts)
        x, steps, factor, error = reused
        assert x is reused.solution and factor is system.factorization
        assert steps == reused.info.refinements and error == reused.info.relative_error


def test_reuse_verifies_requested_matrix_not_the_shifted_factor(counting_factor):
    G, shift = example()
    rhs = np.array([2., -3., .5])
    with linear.VerifiedLinearSystem(G, shift, 2., target_atol=1e-13,
                                    target_rtol=1e-12) as system:
        system.solve_verified(np.ones(3))
        result = system.solve_verified(rhs)
    original_residual = rhs - G @ result.solution
    shifted_residual = rhs - (G + 2. * shift) @ result.solution
    assert result.info.status == 'accurate'
    assert result.info.absolute_error == np.max(np.abs(original_residual))
    assert np.max(np.abs(original_residual)) < 1e-11
    assert np.max(np.abs(shifted_residual)) > 1.
    assert result.info.tau_reg == 2.
    assert counting_factor.constructions == 1 and counting_factor.updates == 0


def test_inaccurate_reuse_is_explicit_and_never_refactors(counting_factor):
    G = csc_array(np.diag([1., 3.]))
    shift = csc_array(np.eye(2))
    with linear.VerifiedLinearSystem(G, shift, 1., max_refinement_steps=0,
                                    target_atol=1e-14, target_rtol=1e-14) as system:
        first = system.solve_verified([1., 0.])
        second = system.solve_verified([1., 1.])
    assert first.info.status == 'accurate'
    assert second.info.status == 'inaccurate'
    assert second.info.stop_reason == 'refinement_limit'
    assert second.info.refinements == second.info.attempts == 0
    assert second.info.absolute_error == np.max(np.abs([1., 1.] - G @ second.solution))
    assert counting_factor.constructions == 1 and counting_factor.updates == 0
    assert counting_factor.applications == 2


@pytest.mark.parametrize('budget', [0, 1, 2])
def test_each_rhs_has_its_own_bounded_refinement_budget(counting_factor, budget):
    G, shift = example()
    with linear.VerifiedLinearSystem(G, shift, 1., max_refinement_steps=budget,
                                    target_atol=0., target_rtol=0.) as system:
        for rhs in (np.array([1., 2., 3.]), np.array([-2., 1., -3.])):
            before = counting_factor.applications
            result = system.solve_verified(rhs)
            assert result.info.refinements <= budget
            assert counting_factor.applications - before == 1 + result.info.refinements


def test_zero_rhs_does_not_mark_an_inherited_stale_factor_ready(counting_factor):
    G, shift = example()
    stale = linear.factor_and_solve(2. * G, np.ones(3), shift, .1, None).factorization
    previous_calls = counting_factor.applications
    with linear.VerifiedLinearSystem(G, shift, .1, stale) as system:
        zero = system.solve_verified(np.zeros(3))
        assert zero.info.attempts == zero.info.refinements == 0
        assert counting_factor.updates == 0 and counting_factor.applications == previous_calls
        result = system.solve_verified(np.array([1., -2., 3.]))
        assert counting_factor.updates == 1
        assert result.info.attempts == 1
        assert_allclose(G @ result.solution, [1., -2., 3.], atol=1e-10, rtol=1e-6)
        calls = counting_factor.applications
        zero = system.solve_verified(np.zeros(3))
        assert counting_factor.applications == calls
        assert zero.info.stop_reason == 'zero_rhs'
        assert zero.info.attempts == zero.info.refinements == 0
        assert_array_equal(zero.solution, np.zeros(3))


@pytest.mark.parametrize('rhs', [np.ones(2), np.ones((3, 1)), [1., np.nan, 2.]])
def test_invalid_new_rhs_does_not_touch_ready_factor(counting_factor, rhs):
    G, shift = example()
    with linear.VerifiedLinearSystem(G, shift, 1e-8) as system:
        first = system.solve_verified(np.ones(3))
        calls = counting_factor.applications
        with pytest.raises(ValueError):
            system.solve_verified(rhs)
        assert counting_factor.applications == calls
        second = system.solve_verified(np.ones(3))
        assert_array_equal(first.solution, second.solution)


@pytest.mark.parametrize('raised', [False, True])
def test_borrowed_storage_is_readonly_and_original_flags_return(raised):
    G, shift = example()
    shift.indices.flags.writeable = False
    arrays = [a for m in (G, shift) for a in (m.data, m.indices, m.indptr)]
    original = [a.flags.writeable for a in arrays]
    system = linear.VerifiedLinearSystem(G, shift, 1e-8)
    with pytest.raises(RuntimeError, match='active matrix context'):
        system.solve_verified(np.ones(3))
    try:
        with system:
            assert all(not a.flags.writeable for a in arrays)
            with pytest.raises(ValueError, match='read-only'):
                G.data[0] = 10.
            system.solve_verified(np.ones(3))
            if raised:
                raise LookupError('exit through an exception')
    except LookupError:
        assert raised
    assert [a.flags.writeable for a in arrays] == original
    with pytest.raises(RuntimeError, match='active matrix context'):
        system.solve_verified(np.ones(3))
    with pytest.raises(RuntimeError, match='entered only once'):
        with system:
            pass


def test_replaced_matrix_storage_is_rejected_before_another_factor_application(counting_factor):
    G, shift = example()
    old_data = G.data
    with linear.VerifiedLinearSystem(G, shift, .1) as system:
        system.solve_verified(np.ones(3))
        calls = counting_factor.applications
        G.data = old_data.copy()
        with pytest.raises(RuntimeError, match='storage changed'):
            system.solve_verified(np.ones(3))
        assert counting_factor.applications == calls
    assert old_data.flags.writeable


def test_reused_numerical_failure_propagates_without_refactor_and_discards_factor(counting_factor):
    G, shift = example()
    with linear.VerifiedLinearSystem(G, shift, 1e-8) as system:
        system.solve_verified(np.ones(3))
        system.factorization.fail_solve = True
        with pytest.raises(linear.LinearSolveError, match='injected reused-factor failure'):
            system.solve_verified(np.ones(3))
        assert system.factorization is None
        assert counting_factor.constructions == 1 and counting_factor.updates == 0
        assert counting_factor.releases == 1
        with pytest.raises(RuntimeError, match='active matrix context'):
            system.solve_verified(np.ones(3))
    assert G.data.flags.writeable and shift.data.flags.writeable


def test_first_rhs_retry_shift_is_retained_for_all_later_rhs(monkeypatch):
    native = linear.qdldl.Solver
    attempted = []

    def fail_once(matrix, **kwargs):
        attempted.append(matrix.diagonal().copy())
        if len(attempted) == 1:
            raise RuntimeError('injected first factorization failure')
        return native(matrix, **kwargs)

    monkeypatch.setattr(linear.qdldl, 'Solver', fail_once)
    G, shift = example()
    with linear.VerifiedLinearSystem(G, shift, 1e-6) as system:
        first = system.solve_verified(np.ones(3))
        second = system.solve_verified(np.array([1., -2., 3.]))
    assert len(attempted) == 2
    assert first.info.attempts == 2 and second.info.attempts == 0
    assert first.info.tau_reg == second.info.tau_reg == 1e-6 * 10.
    assert_array_equal(attempted[1], (G + first.info.tau_reg * shift).diagonal())
