"""Reusable factor-input storage must match ordinary sparse addition exactly."""
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from scipy.sparse import csc_array, eye_array

from glqp.linear import ShiftedSystemWorkspace, factor_and_solve
from glqp import GLQP, SolverSettings
from glqp.obj import LogisticNLL
import glqp.linear as linear


def assert_same_storage(actual, expected):
    expected = csc_array(expected)
    expected.sum_duplicates()
    expected.sort_indices()
    assert actual.shape == expected.shape
    assert_array_equal(actual.indptr, expected.indptr)
    assert_array_equal(actual.indices, expected.indices)
    assert_array_equal(actual.data, expected.data)


def test_repeated_shifts_reuse_storage_and_do_not_accumulate_or_mutate_inputs():
    G = csc_array([[3., .5], [.5, -2.]])
    shift = csc_array(np.diag([1., -1.]))
    original = [(item.data.copy(), item.indices.copy(), item.indptr.copy()) for item in (G, shift)]
    workspace = ShiftedSystemWorkspace()
    first = workspace.assemble(G, shift, 1e-8)
    buffers = first.data, first.indices, first.indptr
    for tau in (1e-5, 0., 2e-8, .1):
        actual = workspace.assemble(G, shift, tau)
        assert actual is first
        assert all(old is new for old,new in zip(buffers, (actual.data, actual.indices, actual.indptr)))
        assert_same_storage(actual, G+tau*shift)
    for supplied, before in zip((G, shift), original):
        for values, frozen in zip((supplied.data, supplied.indices, supplied.indptr), before):
            assert_array_equal(values, frozen)


def test_values_patterns_and_exact_zero_crossings_follow_sparse_addition():
    G = csc_array([[1., 0., .5], [0., -1., 0.], [.5, 0., 0.]])
    shift = csc_array([[1., .2, 0.], [.2, 1., 0.], [0., 0., -1.]])
    workspace = ShiftedSystemWorkspace()
    for tau in (1., 0., .1, 0., 1.):
        assert_same_storage(workspace.assemble(G, shift, tau), G+tau*shift)
    G.data *= 2
    shift.data *= .5
    assert_same_storage(workspace.assemble(G, shift, .1), G+.1*shift)
    # Change only indices in existing storage, while retaining canonical order.
    shift.indices[1] = 2
    assert_same_storage(workspace.assemble(G, shift, .2), G+.2*shift)
    replacement = csc_array([[2., .1, 0.], [.1, 3., 0.], [0., 0., -2.]])
    assert_same_storage(workspace.assemble(replacement, shift, .3), replacement+.3*shift)


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_noncanonical_and_other_dtypes_preserve_generic_arithmetic(dtype):
    G = csc_array((np.array([1e15, 1., -1e15, 3.], dtype=dtype),
        [0, 0, 0, 1], [0, 3, 4]), shape=(2,2))
    shift = csc_array(eye_array(2, dtype=dtype))
    workspace = ShiftedSystemWorkspace()
    before = G.data.copy()
    assert not G.has_canonical_format
    assert_same_storage(workspace.assemble(G, shift, .1), G+.1*shift)
    assert_array_equal(G.data, before)


@pytest.mark.parametrize("tau", [0., 2e-8, 1e-3])
def test_reused_shifted_matrix_preserves_solutions_and_all_accuracy_fields(tau):
    G = csc_array([[2., .1, .5], [.1, 3., -.2], [.5, -.2, -1.]])
    shift = csc_array(np.diag([1., 1., -1.]))
    workspace = ShiftedSystemWorkspace()
    factors = [None, None]
    for scale in (1., 1.5, .5):
        rhs = np.array([1., -2., 3.])*scale
        options = dict(target_atol=1e-14, residual_blocks={"primal":slice(0,2), "dual":slice(2,None)})
        original = factor_and_solve(G, rhs, shift, tau, factors[0], **options)
        reused = factor_and_solve(G, rhs, shift, tau, factors[1], workspace=workspace, **options)
        factors = original.factorization, reused.factorization
        assert_array_equal(original.solution, reused.solution)
        assert original.info == reused.info
        # Refinement continues to certify the unshifted G, never the buffer.
        assert reused.info.absolute_error == np.max(np.abs(rhs-G@reused.solution))


def test_retry_reads_the_larger_shift_and_original_values(monkeypatch):
    solver = linear.qdldl.Solver
    received = []

    def fail_once(matrix, **kwargs):
        received.append(matrix.toarray().copy())
        if len(received) == 1:
            raise RuntimeError("injected numerical failure")
        return solver(matrix, **kwargs)

    monkeypatch.setattr(linear.qdldl, "Solver", fail_once)
    G = csc_array(np.diag([2., -3.]))
    shift = csc_array(np.diag([1., -1.]))
    result = factor_and_solve(G, np.array([4., 3.]), shift, 2e-8, None,
        workspace=ShiftedSystemWorkspace())
    assert result.info.attempts == 2
    assert_array_equal(received[0], (G+2e-8*shift).toarray())
    assert_array_equal(received[1], (G+2e-7*shift).toarray())


@pytest.mark.parametrize("nonlinear", [False, True])
def test_whole_solve_matches_sparse_addition_after_model_changes(monkeypatch, nonlinear):
    design = csc_array([[1., 0., .2], [0., 1., -.3], [1., .5, 0.],
                        [-1., 0., 1.], [.2, -.5, 1.], [0., 1., .4]])
    likelihood = dict(A=design, f=LogisticNLL(np.array([1., 0., 1., 0., 1., 0.]),
                                            np.full(6, 1/6))) if nonlinear else {}
    problem = GLQP(Q=csc_array(np.diag([.3, .4, .5])), b=np.array([.2, -.1, .3]),
                   C=csc_array(np.vstack([np.eye(3), -np.eye(3)])), c=np.ones(6),
                   E=csc_array([[1., 1., 0.]]), e=np.array([.2]), **likelihood)
    settings = SolverSettings(tol=1e-9, max_stagnation=100)
    previous_workspace = problem._shift_workspace
    for changed in (False, True):
        if changed:
            problem.Q = csc_array([[.4, .1, 0.], [.1, .5, 0.], [0., 0., .6]])
            if nonlinear:
                problem.A.data *= .8
        _, reused = problem.solve(verbose=False, settings=settings)
        assert problem._shift_workspace is not previous_workspace
        previous_workspace = problem._shift_workspace
        with monkeypatch.context() as patch:
            patch.setattr(ShiftedSystemWorkspace, "assemble",
                          lambda self, G, shift, tau: csc_array(G + tau*shift))
            _, original = problem.solve(verbose=False, settings=settings)
        assert reused.termination_tag == original.termination_tag == "optimal"
        assert reused.history.drop(columns="time").equals(original.history.drop(columns="time"))
        assert reused.last_linear_solve == original.last_linear_solve
        for name in ("x", "y", "s", "nu"):
            assert_array_equal(getattr(reused, name), getattr(original, name))
