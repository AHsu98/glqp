"""Accuracy and work contracts for the bounded right-GMRES correction."""
import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import csc_array
from scipy.sparse.linalg import gmres

from glqp import SolverSettings
from glqp.linear import _gmres_refine


def test_right_preconditioning_matches_independent_scipy_correction():
    rng = np.random.default_rng(451)
    A = rng.normal(size=(12, 12)) + 5*np.eye(12)
    B = rng.normal(size=(12, 12))*.04 + np.eye(12)
    b, x0 = rng.normal(size=(2, 12))
    residual = b.astype(np.longdouble)-A.astype(np.longdouble)@x0.astype(np.longdouble)
    z, _ = gmres(A@B, residual.astype(float), restart=5, maxiter=1, atol=0., rtol=0.)
    calls = []
    def inverse(v):
        calls.append(v.copy())
        return B@v
    x, r, count, reason = _gmres_refine(csc_array(A), b, x0, inverse, 5, 0., 0.)
    assert_allclose(x, x0+B@z, rtol=1e-12, atol=1e-12)
    assert count == len(calls) == 5
    assert reason == 'refinement_limit'
    assert_allclose(r, b.astype(np.longdouble)-A.astype(np.longdouble)@x.astype(np.longdouble), rtol=0., atol=1e-17)


def test_gmres_can_progress_after_individually_unproductive_steps():
    # A cyclic permutation requires all four directions. Stopping at the first
    # unchanged residual would incorrectly declare stagnation.
    A = np.roll(np.eye(4), 1, axis=0)
    b = np.array([1., 0., 0., 0.])
    calls = []
    def inverse(v):
        calls.append(v.copy())
        return v.copy()
    x, r, count, reason = _gmres_refine(csc_array(A), b, np.zeros(4), inverse, 4, 1e-14, 1e-14)
    assert_allclose(A@x, b, atol=1e-14)
    assert count == len(calls) == 4
    assert reason == 'tolerance'


def test_stored_basis_avoids_remapping_a_rounded_inverse_action():
    b = np.array([1.234, 5.678])
    calls = []
    def inverse(v):
        calls.append(v.copy())
        return np.round(v, 1)
    x, r, count, reason = _gmres_refine(csc_array(np.eye(2)), b, np.zeros(2), inverse, 2, 1e-13, 1e-13)
    assert_allclose(x, b, rtol=0., atol=1e-13)
    assert count == len(calls) == 2  # No additional inverse call to reconstruct x.
    assert reason == 'tolerance'


def test_residual_check_accounts_for_product_cancellation():
    A = csc_array([[3.]])
    x0 = np.array([np.nextafter(1e16, np.inf)])
    b = np.array([3e16+4])
    x, r, count, _ = _gmres_refine(A, b, x0, lambda v: v.copy(), 0, 0., 0.)
    assert (b-A@x)[0] == -4.
    if np.finfo(np.longdouble).nmant > np.finfo(float).nmant:
        assert r[0] == -2.
    assert count == 0


def test_default_budget_is_ten_and_zero_budget_skips_inverse():
    assert SolverSettings().max_iterative_refinement == 10
    def inverse(v):
        raise AssertionError('Zero budget must not apply the inverse')
    x, r, count, reason = _gmres_refine(csc_array(np.eye(2)), np.ones(2), np.zeros(2), inverse, 0, 0., 0.)
    assert count == 0
    assert_allclose(x, np.zeros(2))
    assert_allclose(r, np.ones(2))
    assert reason == 'refinement_limit'
