"""Independent adversarial checks of reduction and original dual recovery."""

import numpy as np
import pytest
from scipy import sparse

from glqp.presolve import PresolveOptions, presolve_qp


@pytest.mark.parametrize("seed", range(8))
def test_row_orders_and_dyadic_scalings_preserve_a_mixed_known_kkt_point(seed):
    rng = np.random.default_rng(seed)
    n = 9
    R = rng.normal(size=(n, n))
    Q = sparse.csc_matrix(R.T@R+np.eye(n))
    optimum = np.array([.5, -.25, 1., -2., .75, -.5, 1.5, -.75, .25])
    # Equality pivots expose a chain; bounds fix a later variable and expose
    # another equality pivot. The final equality stays in the reduced QP.
    A = np.array([[2., 0., 0., 0., 0., 0., 0., 0., 0.],
                  [1., -4., 0., 0., 0., 0., 0., 0., 0.],
                  [0., 1., 2., 0., 0., 0., 0., 0., 0.],
                  [0., 0., 0., 1., -2., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0., 1., 2., -1., 0.]])
    G = np.zeros((8, n))
    G[0, 3], G[1, 3], G[2, 3] = 2., -2., 1.
    G[3, 5], G[4, 5] = 1., 2.
    G[5, [0, 1, 2]], G[6, 8], G[7, 8] = [1., -1., 2.], 1., -1.
    slack = np.array([0., 0., 1., 0., 0., 2., 2., 3.])
    b, h = A@optimum, G@optimum+slack
    y = rng.normal(size=len(b))
    z = np.array([1., 0., 0., 1.7, 0., 0., 0., 0.])
    q = -(Q@optimum+A.T@y+G.T@z)

    eq_order, ineq_order = rng.permutation(len(b)), rng.permutation(len(h))
    eq_scale = rng.choice([-1., 1.], len(b))*np.exp2(rng.integers(-5, 6, len(b)))
    ineq_scale = np.exp2(rng.integers(-5, 6, len(h)))
    A, b, y = A[eq_order]*eq_scale[:, None], b[eq_order]*eq_scale, y[eq_order]/eq_scale
    G, h, z = G[ineq_order]*ineq_scale[:, None], h[ineq_order]*ineq_scale, z[ineq_order]/ineq_scale
    A, G = sparse.csc_matrix(A), sparse.csc_matrix(G)
    with presolve_qp(Q, q, A, b, G, h) as reduced:
        assert reduced.status == "reduced"
        assert reduced.stats["fixed_variables"] == 5
        # Same-variable tied bounds can choose a different retained dual row.
        # Build a valid reduced multiplier directly from its original scaling.
        zr = np.zeros(reduced.G.shape[0])
        chosen = next(j for j, row in enumerate(reduced.inequality_map)
                      if G[row, 5] > 0 and h[row] == G[row, 5]*optimum[5])
        zr[chosen] = 1.7/G[reduced.inequality_map[chosen], 5]
        xr = optimum[reduced.variable_map]
        recovered = reduced.recover(xr, y[reduced.equality_map], zr)
        x, yr, z = recovered
        np.testing.assert_allclose(x, optimum, rtol=0, atol=0)
        np.testing.assert_allclose(A@x, b, rtol=0, atol=1e-12)
        assert np.max(G@x-h) <= 1e-12
        assert np.min(z) >= -1e-12
        np.testing.assert_allclose(Q@x+q+A.T@yr+G.T@z, 0., atol=2e-12, rtol=0)
        np.testing.assert_allclose(z*(G@x-h), 0., atol=1e-12, rtol=0)
        original_value = .5*x@(Q@x)+q@x
        reduced_value = .5*xr@(reduced.Q@xr)+reduced.q@xr+reduced.objective_offset
        np.testing.assert_allclose(original_value, reduced_value, atol=1e-12, rtol=0)


def test_elimination_overflow_keeps_the_original_problem():
    Q = sparse.csc_matrix([[1.]])
    A = sparse.csc_matrix([[1e-300]])
    with presolve_qp(Q, [0.], A, [1e300]) as reduced:
        assert reduced.status == "inconclusive"
        assert reduced.n == 1
        np.testing.assert_array_equal(reduced.A.toarray(), A.toarray())
        np.testing.assert_array_equal(reduced.b, [1e300])


def test_empty_rows_are_removed_with_all_optional_passes_disabled():
    Q = sparse.eye(2, format="csc")
    A, G = sparse.csc_matrix((2, 2)), sparse.csc_matrix((2, 2))
    with presolve_qp(Q, [0., 0.], A, [0., 0.], G, [0., 2.],
                    options=PresolveOptions(False, False, False)) as reduced:
        assert reduced.status == "reduced"
        assert reduced.n == 2
        assert reduced.A.shape == reduced.G.shape == (0, 2)
        x, y, z = reduced.recover([0., 0.])
        np.testing.assert_array_equal(x, [0., 0.])
        np.testing.assert_array_equal(y, [0., 0.])
        np.testing.assert_array_equal(z, [0., 0.])


@pytest.mark.parametrize("name", ["Q", "q", "A", "b", "G", "h"])
def test_real_qp_contract_rejects_complex_coefficients_before_casting(name):
    inputs = dict(Q=sparse.csc_matrix([[1.]]), q=np.array([0.]),
                  A=sparse.csc_matrix([[1.]]), b=np.array([1.]),
                  G=sparse.csc_matrix([[1.]]), h=np.array([2.]))
    value = inputs[name].astype(complex)
    if sparse.issparse(value):
        value.data[0] += 2j
    else:
        value[0] += 2j
    inputs[name] = value
    with pytest.raises((TypeError, ValueError), match="real|complex"):
        presolve_qp(**inputs)


@pytest.mark.parametrize("name", ["x", "y", "z"])
def test_recovery_rejects_complex_solution_vectors(name):
    with presolve_qp(sparse.eye(2), [0., 0.], [[1., 1.]], [0.],
                    [[1., -1.]], [1.]) as reduced:
        values = dict(x=np.zeros(2), y=np.zeros(1), z=np.zeros(1))
        values[name] = values[name].astype(complex)
        values[name][0] = 1j
        with pytest.raises((TypeError, ValueError), match="real|complex"):
            reduced.recover(**values)
