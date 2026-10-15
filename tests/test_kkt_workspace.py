"""Check reusable assembly against independent dense equations and old assembly."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import block_array, csc_array, diags_array, eye_array

from glqp import GLQP, SolverSettings
from glqp.kkt import KKTWorkspace
from glqp.obj import LogisticNLL
import glqp.solver as solver_module


def reference_assembly(H, C, E, y, s, rx, rp, rc, req, rho):
    """Original sparse block construction retained only as a regression oracle."""
    w = np.sqrt(y + rho)
    Cw = C.multiply(w[:, None])
    matrix = block_array([
        [H + rho*eye_array(H.shape[0]), Cw.T, E.T],
        [Cw, -diags_array(s + rho*y + rho**2), None],
        [E, None, -rho*eye_array(E.shape[0])],
    ], format="csc")
    return matrix, np.hstack([-rx, (1/w)*rc - w*rp, -req]), w


@pytest.mark.parametrize("n,k,p", [(3, 2, 1), (3, 0, 1), (3, 2, 0), (1, 0, 0)])
def test_repeated_assembly_matches_equations_without_accumulating_shifts(n, k, p):
    rng = np.random.default_rng(13)
    H = csc_array(rng.normal(size=(n, n)))
    H = csc_array(H.T @ H)
    # Missing Hessian diagonals must still have reusable regularization slots.
    H[0, 0] = 0.
    H.eliminate_zeros()
    C = csc_array(rng.normal(size=(k, n)))
    E = csc_array(rng.normal(size=(p, n)))
    workspace = KKTWorkspace(H, C, E)
    assert workspace.matrix.indices.dtype == np.int32
    buffers = (workspace.matrix, workspace.matrix.data, workspace.matrix.indices,
               workspace.matrix.indptr, workspace.rhs, workspace.w)
    original = [matrix.toarray().copy() for matrix in (H, C, E)]
    for rho in [.2, 0., 1e-7, 0.]:
        y, s = rng.uniform(.1, 2., size=(2, k))
        rx, rp, rc, req = [rng.normal(size=size) for size in (n, k, k, p)]
        matrix, rhs, w = workspace.assemble(H, y, s, rx, rp, rc, req, rho)
        expected, expected_rhs, _ = reference_assembly(H, C, E, y, s, rx, rp, rc, req, rho)
        assert_array_equal(matrix.toarray(), expected.toarray())
        assert_array_equal(rhs, expected_rhs)
        # Independent block equations also check signs, offsets, and scaling.
        dense = np.block([
            [H.toarray() + rho*np.eye(n), C.toarray().T*w, E.toarray().T],
            [w[:, None]*C.toarray(), -np.diag(s + rho*y + rho**2), np.zeros((k, p))],
            [E.toarray(), np.zeros((p, k)), -rho*np.eye(p)],
        ])
        assert_array_equal(matrix.toarray(), dense)
        current = (matrix, matrix.data, matrix.indices, matrix.indptr, rhs, w)
        assert all(old is new for old, new in zip(buffers, current))
        for supplied, before in zip((H, C, E), original):
            assert_array_equal(supplied.toarray(), before)


def test_hessian_pattern_changes_and_zero_crossings_do_not_leave_stale_entries():
    C, E = csc_array([[1., -1., 0.]]), csc_array([[0., 1., 1.]])
    matrices = [
        [[3., 1., 0.], [1., 3., 0.], [0., 0., 3.]],
        [[3., 0., 1.], [0., 3., 0.], [1., 0., 3.]],  # Same nnz, different pattern.
        np.diag([3., 3., 3.]),
        [[3., 1., 0.], [1., 3., 0.], [0., 0., 3.]],
    ]
    workspace = KKTWorkspace(csc_array(matrices[0]), C, E)
    args = (np.array([.4]), np.array([.8]), np.ones(3), np.ones(1), np.ones(1), np.ones(1), .02)
    for values in matrices:
        H = csc_array(values)
        actual, rhs, _ = workspace.assemble(H, *args)
        expected, expected_rhs, _ = reference_assembly(H, C, E, *args)
        assert_array_equal(actual.toarray(), expected.toarray())
        assert_array_equal(rhs, expected_rhs)
    # In-place numerical changes with retained explicit zeros require updates too.
    H.data[:] = 0.
    actual, _, _ = workspace.assemble(H, *args)
    expected, _, _ = reference_assembly(H, C, E, *args)
    assert_array_equal(actual.toarray(), expected.toarray())


def noncanonical(values):
    canonical = csc_array(values)
    data, indices, indptr = [], [], [0]
    for col in range(canonical.shape[1]):
        for j in reversed(range(canonical.indptr[col], canonical.indptr[col+1])):
            data.extend([canonical.data[j]/2, canonical.data[j]/2])
            indices.extend([canonical.indices[j], canonical.indices[j]])
        indptr.append(len(data))
    return csc_array((data, indices, indptr), shape=canonical.shape)


def test_unsorted_duplicate_input_entries_are_combined_without_mutation():
    H = noncanonical([[4., 1.], [1., 3.]])
    C = noncanonical([[1., 2.], [-1., 3.]])
    E = noncanonical([[2., 1.]])
    originals = [(a.data.copy(), a.indices.copy(), a.indptr.copy()) for a in (H, C, E)]
    workspace = KKTWorkspace(H, C, E)
    args = (np.array([.5, 2.]), np.ones(2), np.ones(2), np.ones(2), np.ones(2), np.ones(1), .1)
    matrix, rhs, _ = workspace.assemble(H, *args)
    expected, expected_rhs, _ = reference_assembly(H, C, E, *args)
    assert_allclose(matrix.toarray(), expected.toarray(), atol=1e-15)
    assert_array_equal(rhs, expected_rhs)
    for supplied, before in zip((H, C, E), originals):
        for actual, original in zip((supplied.data, supplied.indices, supplied.indptr), before):
            assert_array_equal(actual, original)


@pytest.mark.parametrize("rho", [0., .03])
def test_newton_direction_matches_independent_dense_system(rho):
    problem = GLQP(Q=csc_array([[3., .2], [.2, 2.]]),
                   C=csc_array([[1., 2.], [-1., 0.]]), c=np.ones(2),
                   E=csc_array([[1., -1.]]), e=np.zeros(1))
    x, y, s, _, _ = problem.initialize(settings=SolverSettings(tol=1e-10))
    rx, rp, rc, req = np.array([.2, -.7]), np.array([.3, .1]), np.array([.5, -.4]), np.array([.2])
    H = problem.get_H(problem.A @ x)
    matrix, rhs, w = reference_assembly(H, problem.C, problem.E, y, s, rx, rp, rc, req, rho)
    reference = np.linalg.solve(matrix.toarray(), rhs)
    dx, ds, dy, dnu, *_ = problem.solve_KKT(x, y, s, H, rx, rp, rc, req, 0., prox_reg=rho)
    direction = np.r_[dx, dy/w, dnu]
    assert_allclose(direction, reference, atol=1e-10, rtol=1e-10)
    assert_allclose(matrix @ direction, rhs, atol=1e-10, rtol=1e-10)
    assert_allclose(ds, -rp - problem.C @ dx + rho*dy, atol=1e-15)


def test_new_solve_refreshes_changed_problem_matrices():
    problem = GLQP(Q=csc_array(np.eye(2)), b=np.ones(2),
                   C=csc_array([[1., 0.]]), c=np.array([.2]),
                   E=csc_array([[0., 1.]]), e=np.array([.3]))
    problem.solve(verbose=False)
    old = problem._kkt_workspace
    problem.Q = csc_array([[3., .2], [.2, 2.]])
    problem.C = csc_array([[0., 1.]])
    problem.E = csc_array([[1., 1.]])
    fresh = GLQP(Q=problem.Q, b=problem.b, C=problem.C, c=problem.c, E=problem.E, e=problem.e)
    x, result = problem.solve(verbose=False)
    reference, reference_result = fresh.solve(verbose=False)
    assert result.termination_tag == reference_result.termination_tag == "optimal"
    assert_array_equal(x, reference)
    assert problem._kkt_workspace is not old


@pytest.mark.parametrize("with_glm", [False, True])
def test_complete_iteration_history_matches_original_assembly(monkeypatch, with_glm):
    kwargs = dict(Q=csc_array(np.eye(2)), b=np.array([-1., .8]),
                  C=csc_array([[1., 0.], [0., -1.]]), c=np.array([.4, .8]))
    if with_glm:
        # At zero the off-diagonal GLM Hessian cancels, then appears as x moves.
        kwargs.update(A=csc_array([[1., 1.], [1., -1.]]), f=LogisticNLL([0., 1.], [1., 1.]))
    x, result = GLQP(**kwargs).solve(verbose=False)

    class OriginalAssembly:
        def __init__(self, H, C, E):
            self.C, self.E = C, E

        def assemble(self, H, *args):
            return reference_assembly(H, self.C, self.E, *args)

    monkeypatch.setattr(solver_module, "KKTWorkspace", OriginalAssembly)
    reference, old_result = GLQP(**kwargs).solve(verbose=False)
    assert result.termination_tag == old_result.termination_tag == "optimal"
    assert_array_equal(x, reference)
    for name in ("y", "s", "nu"):
        assert_array_equal(getattr(result, name), getattr(old_result, name))
    assert_array_equal(result.history.drop(columns="time"), old_result.history.drop(columns="time"))
