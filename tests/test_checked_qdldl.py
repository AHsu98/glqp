"""Numerical failure propagation with an isolated, status-checked QDLDL build.

These tests exercise small native systems, not optimizer benchmark trajectories.
Healthy arithmetic is compared with the separately installed upstream module.
"""
import hashlib
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest
import qdldl as upstream
from scipy.sparse import csc_array, diags_array

import glqp.checked_qdldl as checked
import glqp.linear as linear
from glqp.backends import FactorizationError, QDLDLFactorization


def nonsingular():
    return csc_array([[1., 1.], [1., 2.]])


def singular_same_pattern():
    return csc_array([[1., 1.], [1., 1.]])


def assert_same_bytes(left, right):
    assert left.shape == right.shape
    assert left.dtype == right.dtype
    assert left.tobytes() == right.tobytes()


def test_private_build_keeps_public_dependency_and_factory_unchanged():
    factory, path = upstream.Solver, upstream.__file__
    with open(path, 'rb') as stream:
        before = hashlib.file_digest(stream, 'sha256').hexdigest()
    provenance = checked.prepare_checked_qdldl()
    assert provenance
    assert sys.modules['qdldl'] is upstream
    assert upstream.Solver is factory
    assert upstream.__file__ == path
    with open(path, 'rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == before
    assert type(checked.Solver(nonsingular())).__module__ != 'qdldl'


def test_successful_native_updates_preserve_upstream_factor_and_solve_bytes():
    matrix = csc_array([[4., .5, .25], [.5, 3., -.25], [.25, -.25, -2.]])
    candidate, reference = checked.Solver(matrix), upstream.Solver(matrix)
    rhs = np.array([1.25, -3.5, 2.])
    preserved = rhs.copy()
    for scale in (1., 1.1, 1e-7, 1e7, .5):
        updated = matrix * scale
        candidate.update(updated)
        reference.update(updated)
        left_l, left_d, left_p = candidate.factors()
        right_l, right_d, right_p = reference.factors()
        for left, right in ((left_l.data, right_l.data), (left_l.indices, right_l.indices),
                            (left_l.indptr, right_l.indptr), (left_d, right_d),
                            (left_p, right_p), (candidate.solve(rhs), reference.solve(rhs))):
            assert_same_bytes(left, right)
        assert_allclose(updated @ candidate.solve(rhs), rhs, atol=2e-14, rtol=2e-14)
        assert_same_bytes(rhs, preserved)


def test_constructor_and_update_both_reject_real_zero_pivot():
    with pytest.raises(RuntimeError):
        checked.Solver(singular_same_pattern())
    factor = checked.Solver(nonsingular())
    with pytest.raises(RuntimeError):
        factor.update(singular_same_pattern())
    # A failed update leaves partially computed arrays, never a usable factor.
    with pytest.raises(RuntimeError):
        factor.solve(np.ones(2))


def test_successful_update_can_recover_an_invalidated_native_factor():
    factor = checked.Solver(nonsingular())
    with pytest.raises(RuntimeError):
        factor.update(singular_same_pattern())
    restored = 2. * nonsingular()
    factor.update(restored)
    assert_allclose(restored @ factor.solve(np.array([1., -2.])), [1., -2.], atol=1e-14)


@pytest.mark.parametrize('matrix', [csc_array(np.ones((2, 3))), csc_array((0, 0))])
def test_malformed_constructor_input_is_not_a_numerical_failure(matrix):
    with pytest.raises(ValueError):
        checked.Solver(matrix)


@pytest.mark.parametrize('matrix', [csc_array(np.eye(3)), csc_array(np.eye(2))])
def test_bad_update_shape_or_storage_size_preserves_last_good_factor(matrix):
    # Numeric update's same-pattern contract remains the caller's obligation;
    # these malformed cases are caught by cheap dimension/storage-size checks.
    factor = checked.Solver(nonsingular())
    expected = factor.solve(np.ones(2))
    with pytest.raises(ValueError):
        factor.update(matrix)
    assert_same_bytes(factor.solve(np.ones(2)), expected)


@pytest.mark.parametrize('rhs', [np.ones(3), np.ones((2, 1))])
def test_bad_rhs_shape_is_not_a_numerical_factorization_failure(rhs):
    factor = checked.Solver(nonsingular())
    with pytest.raises(ValueError):
        factor.solve(rhs)


@pytest.fixture
def counted_factory(monkeypatch):
    class Counted(QDLDLFactorization):
        constructions = 0
        updates = 0

        def __init__(self, matrix, **kwargs):
            type(self).constructions += 1
            super().__init__(matrix, **kwargs)

        def update(self, matrix):
            type(self).updates += 1
            return super().update(matrix)

    monkeypatch.setattr(linear, '_QDLDLFactorization', Counted)
    return Counted


@pytest.mark.parametrize('inherit', [False, True])
def test_real_factor_failure_uses_existing_shift_retry(counted_factory, inherit):
    tau = .125  # Shift makes [[1,1],[1,1]] without changing its sparse pattern.
    matrix = csc_array([[1.-tau, 1.], [1., 1.-tau]])
    shift = csc_array(diags_array([1., 1.]))
    previous = counted_factory(nonsingular()) if inherit else None
    result = linear.factor_and_solve(matrix, np.array([1., -2.]), shift, tau, previous,
        target_atol=1e-11, target_rtol=1e-10, max_solve_attempts=3)
    assert result.info.attempts == 2
    assert result.info.tau_reg == 10*tau
    assert result.info.status == 'accurate'
    assert result.factorization is not previous
    assert counted_factory.updates == int(inherit)
    assert counted_factory.constructions == 2
    assert_allclose(matrix @ result.solution, [1., -2.], atol=1e-11, rtol=1e-10)
    residual = np.array([1., -2.])-matrix@result.solution
    assert result.info.absolute_error == np.max(np.abs(residual))
    result.factorization.close()


@pytest.mark.parametrize('tau,budget', [(0., 20), (.125, 1)])
def test_zero_shift_and_exhausted_budget_stop_after_failed_update(
        counted_factory, tau, budget):
    old = counted_factory(nonsingular())
    with pytest.raises(FactorizationError):
        linear.factor_and_solve(csc_array([[1.-tau, 1.], [1., 1.-tau]]), np.ones(2),
            csc_array(diags_array([1., 1.])), tau, old, max_solve_attempts=budget)
    assert counted_factory.updates == 1
    assert counted_factory.constructions == 1
    with pytest.raises(FactorizationError):
        old.solve(np.ones(2))


def test_ordinary_finite_inaccurate_direction_does_not_trigger_retry(counted_factory):
    result = linear.factor_and_solve(csc_array(diags_array([1., 3.])), np.ones(2),
        csc_array(diags_array([1., 1.])), 1., None, max_refinement_steps=0,
        target_atol=1e-14, target_rtol=1e-14, max_solve_attempts=10)
    assert result.info.status == 'inaccurate'
    assert result.info.stop_reason == 'refinement_limit'
    assert result.info.attempts == 1
    assert result.info.tau_reg == 1.
    assert counted_factory.constructions == 1 and counted_factory.updates == 0
    result.factorization.close()


def test_adapter_update_failure_check_does_not_export_factors(monkeypatch):
    native = checked.Solver

    class NoFactorExport:
        def __init__(self, *args, **kwargs):
            self.native = native(*args, **kwargs)

        def update(self, *args, **kwargs):
            return self.native.update(*args, **kwargs)

        def _update_values(self, *args, **kwargs):
            return self.native._update_values(*args, **kwargs)

        def solve(self, *args, **kwargs):
            return self.native.solve(*args, **kwargs)

        def factors(self):
            raise AssertionError('The update failure check must not export sparse factors')

    monkeypatch.setattr(checked, 'Solver', NoFactorExport)
    factor = QDLDLFactorization(nonsingular())
    factor.update(2*nonsingular())
    assert_allclose((2*nonsingular()) @ factor.solve(np.ones(2)), np.ones(2), atol=1e-14)
    with pytest.raises(FactorizationError):
        factor.update(singular_same_pattern())


def test_build_failure_is_not_retried_as_numeric_factorization_failure(monkeypatch):
    calls = []

    def unavailable():
        calls.append(1)
        raise RuntimeError('checked backend build unavailable')

    monkeypatch.setattr(checked, 'prepare_checked_qdldl', unavailable)
    with pytest.raises(RuntimeError, match='build unavailable') as raised:
        linear.factor_and_solve(nonsingular(), np.ones(2), csc_array(np.eye(2)),
            1e-8, None, max_solve_attempts=20)
    assert not isinstance(raised.value, FactorizationError)
    assert len(calls) == 1
