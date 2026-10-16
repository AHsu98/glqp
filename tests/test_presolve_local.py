"""Local deferral retains uncertain rows while preserving independent work."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import sparse

from glqp.presolve import PresolveInfeasible, PresolveOptions, presolve_qp


LOCAL = PresolveOptions(local_fallback=True)


def test_local_deferral_does_not_drop_equality_pivot_or_ambiguous_row():
    A = sparse.csc_matrix([[3., 0.], [6., 0.], [0., 1.]])
    b = np.array([1., 2., 4.])
    with presolve_qp(sparse.eye(2), [-1./3., 0.], A, b) as old:
        assert old.status == "inconclusive"
        assert old.n == 2
    with presolve_qp(sparse.eye(2), [-1./3., 0.], A, b, options=LOCAL) as reduced:
        assert reduced.status == "reduced"
        assert_array_equal(reduced.variable_map, [0])
        assert_array_equal(reduced.equality_map, [0, 1])
        assert_array_equal(reduced.A.toarray(), [[3.], [6.]])
        assert_array_equal(reduced.b, [1., 2.])
        assert reduced.stats["deferred_variables"] == 1
        assert reduced.stats["retained_ambiguous_rows"] == 1
        x, y, z = reduced.recover([1./3.], [0., 0.])
        assert_array_equal(x, [1./3., 4.])
        assert_allclose(A @ x, b, atol=1e-15)
        assert_allclose(x + [-1./3., 0.] + A.T @ y, 0., atol=1e-15)
        assert z.size == 0


def test_local_deferral_keeps_both_fixed_bound_pivots_and_the_blocking_inequality():
    G = sparse.csc_matrix([[3., 0.], [-3., 0.], [6., 0.]])
    A = sparse.csc_matrix([[0., 1.]])
    with presolve_qp(sparse.eye(2), [-1./3., 0.], A, [4.], G, [1., -1., 2.], options=LOCAL) as reduced:
        assert reduced.status == "reduced"
        assert reduced.stats["fixed_variables"] == 1
        assert reduced.stats["removed_inequalities"] == 0
        assert_array_equal(reduced.inequality_map, [0, 1, 2])
        assert_array_equal(reduced.G.toarray(), [[3.], [-3.], [6.]])
        x, y, z = reduced.recover([1./3.], z=[0., 0., 0.])
        assert_allclose(G @ x, [1., -1., 2.], atol=1e-15)
        assert_allclose(x + [-1./3., 0.] + A.T @ y + G.T @ z, 0., atol=1e-15)


def test_near_conflicting_bounds_are_preserved_while_other_variables_are_fixed():
    G = sparse.csc_matrix([[1., 0.], [-1., 0.]])
    h = np.array([1., -1. - 1e-14])
    with presolve_qp(sparse.eye(2), [0., 0.], [[0., 1.]], [3.], G, h, options=LOCAL) as reduced:
        assert_array_equal(reduced.variable_map, [0])
        assert_array_equal(reduced.G.toarray(), G[:, :1].toarray())
        assert_array_equal(reduced.h, h)
        assert reduced.stats["deferred_variables"] == 1
        assert reduced.stats["retained_ambiguous_rows"] == 2
    with pytest.raises(PresolveInfeasible):
        presolve_qp(sparse.eye(2), [0., 0.], [[0., 1.]], [3.], G, h,
                    options=PresolveOptions(local_fallback=True, infeasibility_guard=0.))


def test_chain_retains_late_ambiguous_variable_and_recovers_earlier_pivot_dual():
    A = sparse.csc_matrix([[1., 0., 0.], [1., 3., 0.], [0., 6., 0.], [0., 0., 1.]])
    b = np.array([2., 3., 2., 4.])
    Q = sparse.csc_matrix([[4., .5, 1.], [.5, 3., 0.], [1., 0., 2.]])
    expected_x = np.array([2., 1./3., 4.])
    expected_y = np.array([1., 2., -1., 3.])
    q = -(Q @ expected_x + A.T @ expected_y)
    with presolve_qp(Q, q, A, b, options=LOCAL) as reduced:
        assert_array_equal(reduced.variable_map, [1])
        assert_array_equal(reduced.equality_map, [1, 2])
        x, y, _ = reduced.recover([1./3.], expected_y[reduced.equality_map])
        assert_allclose(x, expected_x, rtol=0, atol=0)
        assert_allclose(A @ x, b, atol=1e-15)
        assert_allclose(Q @ x + q + A.T @ y, 0, atol=3e-15)


def test_local_fallback_is_opt_in_and_does_not_hide_proven_infeasibility():
    assert PresolveOptions().local_fallback is False
    with pytest.raises(TypeError):
        PresolveOptions(local_fallback=1)
    with pytest.raises(PresolveInfeasible):
        presolve_qp(sparse.eye(2), [0., 0.], [[1., 0.], [1., 0.], [0., 1.]],
                    [1., 2., 4.], options=LOCAL)
    with presolve_qp(sparse.eye(1), [0.], [[3.], [6.]], [1., 2.], options=LOCAL) as reduced:
        assert reduced.status == "inconclusive"
        assert reduced.n == 1
        assert reduced.stats["deferred_variables"] == 1


@pytest.mark.parametrize("seed", range(12))
def test_permuted_scaled_ambiguous_chains_preserve_original_kkt(seed):
    rng = np.random.default_rng(seed)
    optimum = np.array([2., 1./3., 4., 1./3., .5, -.25])
    R = rng.normal(size=(6, 6))
    Q = sparse.csc_matrix(R.T @ R + np.eye(6))
    A = np.array([[1., 0., 0., 0., 0., 0.],
                  [1., 3., 0., 0., 0., 0.],
                  [0., 6., 0., 0., 0., 0.],
                  [0., 0., 1., 0., 0., 0.],
                  [0., 0., 0., 0., 1., 2.]])
    G = np.zeros((5, 6))
    G[:3, 3] = [3., -3., 6.]
    G[3, 4], G[4, 5] = 1., -1.
    b, h = A @ optimum, G @ optimum + [0., 0., 0., 1., 1.]
    y = rng.normal(size=len(b))
    # A positive lower-bound multiplier exercises dual scattering on the
    # retained ambiguous pair. This lower bound is never redundant.
    z = np.array([0., 2., 0., 0., 0.])
    q = -(Q @ optimum + A.T @ y + G.T @ z)
    eq_order, ineq_order = rng.permutation(len(b)), rng.permutation(len(h))
    eq_scale = rng.choice([-1., 1.], len(b)) * np.exp2(rng.integers(-5, 6, len(b)))
    ineq_scale = np.exp2(rng.integers(-5, 6, len(h)))
    A, b, y = A[eq_order] * eq_scale[:, None], b[eq_order] * eq_scale, y[eq_order] / eq_scale
    G, h, z = G[ineq_order] * ineq_scale[:, None], h[ineq_order] * ineq_scale, z[ineq_order] / ineq_scale
    A, G = sparse.csc_matrix(A), sparse.csc_matrix(G)

    with presolve_qp(Q, q, A, b, G, h, options=LOCAL) as reduced:
        assert reduced.status == "reduced"
        assert reduced.stats["fixed_variables"] >= 1
        assert reduced.stats["deferred_variables"] >= 1
        assert reduced.stats["retained_ambiguous_rows"] >= 1
        xr = optimum[reduced.variable_map]
        x, yr, zr = reduced.recover(xr, y[reduced.equality_map], z[reduced.inequality_map])
        assert_allclose(x, optimum, rtol=0, atol=0)
        assert_allclose(A @ x, b, rtol=0, atol=1e-13)
        assert np.max(G @ x - h) <= 1e-13
        assert np.min(zr) >= 0.
        assert_allclose(Q @ x + q + A.T @ yr + G.T @ zr, 0., atol=3e-13, rtol=0)
        assert_allclose(zr * (G @ x - h), 0., atol=1e-13, rtol=0)
        original_value = .5 * x @ (Q @ x) + q @ x
        reduced_value = .5 * xr @ (reduced.Q @ xr) + reduced.q @ xr + reduced.objective_offset
        assert_allclose(original_value, reduced_value, atol=1e-12, rtol=0)
