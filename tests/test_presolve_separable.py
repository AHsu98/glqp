"""Certified scalar quadratic minima and their original-coordinate duals."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import sparse

from glqp.presolve import PresolveOptions, presolve_qp


SEPARABLE = PresolveOptions(separable_quadratics=True)


def test_unconstrained_scalar_quadratics_reduce_to_the_exact_optimum():
    Q = sparse.diags([2., 4., 8.], format="csc")
    q = np.array([1., -4., 6.])
    with presolve_qp(Q, q, options=SEPARABLE) as reduced:
        assert reduced.n == 0
        assert reduced.stats["separable_variables"] == 3
        x, y, z = reduced.recover([])
        assert_array_equal(x, [-.5, 1., -.75])
        assert y.size == z.size == 0
        assert_allclose(Q @ x + q, 0., atol=0.)
        assert_allclose(reduced.objective_offset, .5 * x @ (Q @ x) + q @ x, atol=0.)


@pytest.mark.parametrize("redundant_bounds", [False, True])
@pytest.mark.parametrize("local_fallback", [False, True])
def test_clipped_scalar_minima_have_nonnegative_scaled_bound_duals(redundant_bounds, local_fallback):
    Q = sparse.diags([2., 4., 8.], format="csc")
    q = np.array([6., -16., -4.])
    G = sparse.csc_matrix([[-2., 0., 0.], [-1., 0., 0.], [0., 2., 0.],
                           [0., 1., 0.], [0., 0., -1.], [0., 0., 1.]])
    h = np.array([2., 2., 4., 3., 0., 1.])
    options = PresolveOptions(redundant_bounds=redundant_bounds, local_fallback=local_fallback,
                              separable_quadratics=True)
    with presolve_qp(Q, q, G=G, h=h, options=options) as reduced:
        assert reduced.n == 0
        x, _, z = reduced.recover([])
        assert_array_equal(x, [-1., 2., .5])
        assert_array_equal(z, [2., 0., 4., 0., 0., 0.])
        assert_allclose(Q @ x + q + G.T @ z, 0., atol=0.)
        assert np.max(G @ x - h) <= 0.
        assert_allclose(z * (G @ x - h), 0., atol=0.)


def test_bound_dual_updates_an_earlier_equality_pivot_in_reverse_order():
    Q = sparse.diags([1., 3., 1., 1.], format="csc")
    A = sparse.csc_matrix([[1., 0., 0., 0.], [0., 0., 1., 1.]])
    G = sparse.csc_matrix([[1., -1., 0., 0.], [0., 1., 0., 0.]])
    with presolve_qp(Q, np.zeros(4), A, [2., 4.], G, [-1., 10.], options=SEPARABLE) as reduced:
        assert_array_equal(reduced.variable_map, [2, 3])
        assert reduced.stats["separable_variables"] == 1
        x, y, z = reduced.recover([2., 2.], [-2.])
        assert_array_equal(x, [2., 3., 2., 2.])
        assert_array_equal(y, [-11., -2.])
        assert_array_equal(z, [9., 0.])
        assert_allclose(Q @ x + A.T @ y + G.T @ z, 0., atol=0.)


def test_ambiguous_bound_activity_retains_the_variable_and_independent_work():
    with presolve_qp(sparse.diags([3., 2.]), [1., -2.], G=[[-3., 0.]], h=[1.], options=SEPARABLE) as reduced:
        assert_array_equal(reduced.variable_map, [0])
        assert_array_equal(reduced.inequality_map, [0])
        assert_array_equal(reduced.G.toarray(), [[-3.]])
        assert_array_equal(reduced.h, [1.])
        assert reduced.stats["separable_variables"] == 1
        assert reduced.stats["separable_ambiguous_variables"] == 1
        x, _, z = reduced.recover([-1./3.], z=[0.])
        assert_allclose(sparse.diags([3., 2.]) @ x + [1., -2.] + np.array([-3. * z[0], 0.]), 0., atol=1e-15)


def test_scalar_overflow_does_not_abandon_an_independent_safe_minimum():
    with presolve_qp(sparse.diags([1e-300, 1.]), [1e300, 1.], options=SEPARABLE) as reduced:
        assert_array_equal(reduced.variable_map, [0])
        assert reduced.stats["separable_variables"] == 1
        assert reduced.stats["separable_ambiguous_variables"] == 1
        assert reduced.objective_offset == -.5


def test_scalar_pass_does_not_discard_deferred_contradictory_bounds():
    options = PresolveOptions(local_fallback=True, separable_quadratics=True)
    with presolve_qp(sparse.eye(2), [0., 1.], G=[[1., 0.], [-1., 0.]],
                    h=[1., -1.-1e-14], options=options) as reduced:
        assert_array_equal(reduced.variable_map, [0])
        assert reduced.stats["separable_variables"] == 1
        assert reduced.stats["deferred_variables"] == 1
        assert_array_equal(reduced.h, [1., -1.-1e-14])


def test_initial_version_keeps_quadratic_coupling_even_to_fixed_variables():
    Q = sparse.csc_matrix([[2., .25], [.25, 1.]])
    with presolve_qp(Q, [0., 0.], [[1., 0.]], [2.], options=SEPARABLE) as reduced:
        assert_array_equal(reduced.variable_map, [1])
        assert reduced.stats["separable_variables"] == 0
    with presolve_qp(sparse.diags([0., 2.]), [0., -2.], options=SEPARABLE) as reduced:
        assert_array_equal(reduced.variable_map, [0])
        assert reduced.stats["separable_variables"] == 1


def test_scalar_pass_is_explicit_and_rejects_non_boolean_options():
    assert PresolveOptions().separable_quadratics is False
    with pytest.raises(TypeError):
        PresolveOptions(separable_quadratics=1)
    with presolve_qp(sparse.eye(2), [1., -1.]) as reduced:
        assert reduced.status == "unchanged"
        assert reduced.stats["separable_variables"] == 0


@pytest.mark.parametrize("seed", range(12))
def test_random_scaled_bounds_and_equality_cascade_recover_original_kkt(seed):
    rng = np.random.default_rng(seed)
    n = 13
    diagonal = np.exp2(rng.integers(-4, 5, n))
    unconstrained = rng.integers(-16, 17, n) / 4.
    lower = rng.integers(-8, 5, n) / 4.
    upper = lower + rng.integers(1, 13, n) / 4.
    x_expected = np.clip(unconstrained, lower, upper)
    x_expected[0] = 2.
    Q = sparse.diags(diagonal, format="csc")
    q = -diagonal * unconstrained
    A = sparse.csc_matrix(([1.], ([0], [0])), shape=(1, n))
    G = np.zeros((2 * (n - 1), n))
    h = np.zeros(len(G))
    for j in range(1, n):
        for side in range(2):
            row = 2 * (j - 1) + side
            scale = np.exp2(rng.integers(-4, 5))
            coefficient = scale if side else -scale
            G[row, j] = coefficient
            G[row, 0] = rng.integers(-4, 5) / 4.
            h[row] = coefficient * (upper[j] if side else lower[j]) + 2. * G[row, 0]
    order = rng.permutation(len(G))
    G, h = sparse.csc_matrix(G[order]), h[order]
    with presolve_qp(Q, q, A, [2.], G, h, options=SEPARABLE) as reduced:
        assert reduced.n == 0
        assert reduced.stats["separable_variables"] == n - 1
        x, y, z = reduced.recover([])
        assert_array_equal(x, x_expected)
        assert_allclose(A @ x, [2.], atol=0.)
        assert np.max(G @ x - h) <= 0.
        assert np.min(z) >= 0.
        assert_allclose(Q @ x + q + A.T @ y + G.T @ z, 0., atol=2e-13, rtol=0.)
        assert_allclose(z * (G @ x - h), 0., atol=0.)
        assert_allclose(reduced.objective_offset, .5 * x @ (Q @ x) + q @ x, atol=2e-13, rtol=0.)
