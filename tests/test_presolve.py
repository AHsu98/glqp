"""Native structural reductions checked in original primal/dual coordinates."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import sparse

from glqp.presolve import PresolveInfeasible, PresolveOptions, prepare_native, presolve_qp


def objective(Q, q, x):
    return .5 * x @ (Q @ x) + q @ x


def check_kkt(Q, q, A, b, G, h, solution, atol=1e-11):
    x, y, z = solution
    assert_allclose(A @ x, b, atol=atol)
    assert np.max(G @ x - h, initial=0) <= atol
    assert np.min(z, initial=0) >= -atol
    assert_allclose(Q @ x + q + A.T @ y + G.T @ z, 0, atol=atol)
    assert_allclose(z * (G @ x - h), 0, atol=atol)


def test_fixed_bounds_objective_and_duals():
    Q = sparse.csc_matrix([[4., 1., 2.], [1., 3., .5], [2., .5, 5.]])
    q = np.array([-2., 4., -3.])
    A, b = sparse.csc_matrix((0, 3)), np.empty(0)
    G = sparse.csc_matrix([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
                           [0., -1., 0.], [1., 0., 0.]])
    h = np.array([2., -2., -1., 1., 3.])
    with presolve_qp(Q, q, A, b, G, h, objective_constant=7.) as reduced:
        assert reduced.status == "reduced"
        assert_array_equal(reduced.variable_map, [2])
        assert reduced.G.shape == (0, 1)
        x = -reduced.q / reduced.Q.diagonal()
        solution = reduced.recover(x)
        check_kkt(Q, q, A, b, G, h, solution)
        assert_allclose(objective(Q, q, solution[0]) + 7.,
                        objective(reduced.Q, reduced.q, x) + reduced.objective_offset)


def test_equality_chain_reverse_dual_substitution_and_zero_dimension():
    Q = sparse.diags([2., 3., 4., 5.], format="csc")
    q = np.array([10., -4., 3., -1.])
    A = sparse.csc_matrix([[1., 0., 0., 0.], [2., 1., 0., 0.],
                           [0., 3., 1., 0.], [1., 0., 2., 1.]])
    b = np.array([1., 4., 5., 2.])
    G, h = sparse.csc_matrix((0, 4)), np.empty(0)
    with presolve_qp(Q, q, A, b) as reduced:
        assert reduced.n == 0
        assert reduced.stats["fixed_variables"] == 4
        solution = reduced.recover([])
        check_kkt(Q, q, A, b, G, h, solution)
        assert_allclose(reduced.objective_offset, objective(Q, q, solution[0]))


def test_redundant_singleton_bounds_keep_original_dual_positions():
    Q, q = sparse.eye(2, format="csc"), np.array([-4., 0.])
    A, b = sparse.csc_matrix((0, 2)), np.empty(0)
    G = sparse.csc_matrix([[1., 0.], [2., 0.], [-1., 0.], [1., 0.], [0., 0.]])
    h = np.array([2., 2., 0., 1., 3.])
    with presolve_qp(Q, q, A, b, G, h) as reduced:
        assert reduced.G.shape[0] == 3  # Different-scale tied bounds retained conservatively.
        z = np.zeros(reduced.G.shape[0])
        z[list(reduced.inequality_map).index(1)] = 1.5
        check_kkt(Q, q, A, b, G, h, reduced.recover([1., 0.], z=z))


@pytest.mark.parametrize("A,b,G,h", [
    ([[0.]], [1.], [], []),
    ([], [], [[0.]], [-1.]),
    ([[1.], [1.]], [0., 1.], [], []),
    ([], [], [[1.], [-1.]], [0., -1.]),
])
def test_certified_infeasibility(A, b, G, h):
    with pytest.raises(PresolveInfeasible):
        presolve_qp(sparse.eye(1), [0.], sparse.csc_matrix(A).reshape((len(b), 1)), b,
                    sparse.csc_matrix(G).reshape((len(h), 1)), h)


def test_near_equal_bounds_are_never_fixed_by_tolerance():
    with presolve_qp(sparse.eye(1), [0.], G=sparse.csc_matrix([[1.], [-1.]]),
                    h=[1. + 1e-12, -1.]) as result:
        assert result.status == "unchanged"
        assert result.n == 1


def test_near_inconsistent_constraints_fall_back_without_claiming_feasibility():
    for A, b, G, h in (([[1.], [1.]], [1., 1. + 1e-14], None, None),
                        (None, None, [[1.], [-1.]], [1., -1. - 1e-14])):
        with presolve_qp(sparse.eye(1), [0.], A, b, G, h) as result:
            assert result.status == "inconclusive"
            assert result.n == 1
        with pytest.raises(PresolveInfeasible):
            presolve_qp(sparse.eye(1), [0.], A, b, G, h,
                        options=PresolveOptions(infeasibility_guard=0.))


def test_ambiguous_roundoff_falls_back_to_original_problem():
    A = sparse.csc_matrix([[3.], [6.]])
    with presolve_qp(sparse.eye(1), [0.], A, [1., 2.]) as result:
        assert result.status == "inconclusive"
        assert result.n == 1
        assert_allclose(result.A.toarray(), A.toarray())
        assert_allclose(result.recover([1. / 3.], [0., 0.])[0], [1. / 3.])


def test_passes_can_be_disabled_independently():
    Q = sparse.eye(2)
    A = sparse.csc_matrix([[1., 0.]])
    G = sparse.csc_matrix([[0., 1.], [0., -1.], [0., 1.]])
    b, h = [2.], [1., -1., 2.]
    with presolve_qp(Q, [0., 0.], A, b, G, h,
                    options=PresolveOptions(False, False, False)) as result:
        assert result.status == "unchanged"
    with presolve_qp(Q, [0., 0.], A, b, G, h,
                    options=PresolveOptions(False, True, False)) as result:
        assert_array_equal(result.variable_map, [0])
        assert result.A.shape == (1, 1)


def test_invalid_input_and_closed_recovery():
    with pytest.raises(ValueError, match="symmetric"):
        presolve_qp(sparse.csc_matrix([[1., 1.], [0., 1.]]), [0., 0.])
    with pytest.raises(ValueError, match="finite"):
        presolve_qp(sparse.eye(1), [np.nan])
    with pytest.raises(TypeError):
        PresolveOptions(fixed_bounds=1)
    result = presolve_qp(sparse.eye(1), [0.])
    result.close()
    with pytest.raises(RuntimeError, match="closed"):
        result.recover([0.])


def test_random_fixed_qps_preserve_original_kkt_and_objective():
    rng = np.random.default_rng(20261016)
    for _ in range(20):
        n, count = 20, 8
        R = rng.standard_normal((n, n))
        dense = R.T @ R + np.eye(n)
        Q = sparse.csc_matrix(dense)
        q = rng.standard_normal(n)
        targets = rng.integers(-5, 6, count).astype(float)
        A = sparse.eye(n, format="csc")[:count]
        G, h = sparse.csc_matrix((0, n)), np.empty(0)
        with presolve_qp(Q, q, A, targets) as result:
            x = np.linalg.solve(result.Q.toarray(), -result.q)
            solution = result.recover(x)
            check_kkt(Q, q, A, targets, G, h, solution)
            assert_allclose(objective(Q, q, solution[0]),
                            objective(result.Q, result.q, x) + result.objective_offset)


def test_native_build_provenance_has_source_and_binary_digests():
    info = prepare_native()
    assert len(info["source_sha256"]) == len(info["library_sha256"]) == 64
    assert "-ffp-contract=off" in info["flags"]
    assert "-ffast-math" not in info["flags"]
