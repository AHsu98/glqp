"""Independent dense solves and residuals exercise each backend lifecycle."""
import importlib.util

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import csc_array, diags_array, eye_array

from glqp.backends import BACKENDS, FactorizationError, backend_type, backend_options


@pytest.fixture(params=tuple(BACKENDS))
def factory(request):
    if request.param == 'feral' and importlib.util.find_spec('feral') is None:
        pytest.skip('optional feral-solver is not installed')
    return BACKENDS[request.param]


def saddle_matrix(seed=17):
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(4, 7))
    diagonal = rng.uniform(.5, 3., size=7)
    return csc_array(np.block([[np.diag(diagonal), design.T],
                              [design, -.02 * np.eye(4)]]))


def test_indefinite_factor_update_and_rhs_ownership(factory):
    matrix = saddle_matrix()
    rhs = np.arange(matrix.shape[0], dtype=float) - 3.
    preserved = rhs.copy()
    factor = factory(matrix)
    try:
        for scale in (1., 1.25, .3, 4.):
            updated = matrix * scale
            assert factor.matches(updated)
            factor.update(updated)
            solution = factor.solve(rhs)
            assert_allclose(updated @ solution, rhs, rtol=2e-12, atol=1e-12)
            assert_allclose(solution, np.linalg.solve(updated.toarray(), rhs), rtol=1e-11, atol=1e-11)
            assert_allclose(rhs, preserved, rtol=0., atol=0.)
        info = factor.diagnostics()
        assert info['backend'] == factor.name
        assert info['factor_nnz'] > 0
        if factor.name == 'feral':
            assert info['symbolic_calls'] == 1
            assert info['inner_refinement'] is False
        if factor.name == 'pardiso':
            assert info['inner_refinements'] == 0
            assert info['positive_eigenvalues'] == 7
            assert info['negative_eigenvalues'] == 4
    finally:
        factor.close()
        factor.close()


def test_pattern_changes_are_detected(factory):
    original = csc_array(diags_array([1., -2., 3.]))
    changed = original.toarray()
    changed[0, 1] = changed[1, 0] = .1
    factor = factory(original)
    try:
        assert factor.matches(original)
        assert not factor.matches(csc_array(changed))
        assert not factor.matches(csc_array(eye_array(4)))
    finally:
        factor.close()


def test_backend_rejects_malformed_native_inputs(factory):
    with pytest.raises(ValueError, match='square float64 CSC'):
        factory(csc_array([[1., 2.]]))
    bad = csc_array([[1., 0.], [0., np.nan]])
    with pytest.raises(ValueError, match='finite'):
        factory(bad)
    factor = factory(saddle_matrix())
    try:
        with pytest.raises(ValueError, match='RHS'):
            factor.solve(np.ones(12))
        with pytest.raises(ValueError, match='RHS'):
            factor.solve(np.ones((11, 1)))
    finally:
        factor.close()


def test_requested_system_residual_and_numeric_reuse(factory):
    from glqp.linear import factor_and_solve
    matrix = saddle_matrix()
    rhs = np.arange(matrix.shape[0], dtype=float) - 3.
    shift = csc_array(diags_array([1.] * 7 + [-1.] * 4))
    first = factor_and_solve(matrix, rhs, shift, 1e-5, None,
                            target_atol=1e-13, backend=factory.name)
    try:
        residual = rhs - matrix @ first.solution
        assert_allclose(first.info.absolute_error, np.max(np.abs(residual)), rtol=0., atol=0.)
        assert first.info.status == 'accurate'
        assert first.info.backend == factory.name
        second = factor_and_solve(matrix, 1.2 * rhs, shift, 1e-5, first.factorization,
                                 target_atol=1e-13, backend=factory.name)
        assert second.factorization is first.factorization
        assert_allclose(matrix @ second.solution, 1.2 * rhs, rtol=1e-6, atol=1e-9)
    finally:
        first.factorization.close()


def test_switching_backend_releases_native_context():
    from glqp.linear import factor_and_solve
    matrix = saddle_matrix()
    rhs = np.ones(matrix.shape[0])
    shift = csc_array(eye_array(matrix.shape[0]))
    first = factor_and_solve(matrix, rhs, shift, 0., None, backend='pardiso')
    second = factor_and_solve(matrix, rhs, shift, 0., first.factorization, backend='superlu')
    try:
        assert second.factorization.name == 'superlu'
        with pytest.raises(RuntimeError, match='closed'):
            first.factorization.solve(rhs)
    finally:
        second.factorization.close()


def test_pardiso_disables_forced_inner_refinement():
    # An extreme diagonal range triggers perturbation on current MKL builds.
    # Even when a build chooses no perturbation, assert the explicit controls.
    matrix = csc_array(np.diag([1e-20, -1., 2.]))
    factor = BACKENDS['pardiso'](matrix)
    try:
        assert factor.iparm[7] == 0
        assert factor.iparm[20] == 3
        factor.solve(np.ones(3))
        assert factor.diagnostics()['inner_refinements'] == 0
    finally:
        factor.close()
    with pytest.raises(RuntimeError, match='closed'):
        factor.solve(np.ones(3))


def test_pardiso_constructor_failure_releases_analyzed_native_storage(monkeypatch):
    from glqp.backends import PardisoFactorization
    original = PardisoFactorization._call
    phases = []

    def injected(self, phase, rhs=None):
        phases.append(phase)
        if phase == 22:
            raise FactorizationError('injected failed numeric phase')
        return original(self, phase, rhs)

    monkeypatch.setattr(PardisoFactorization, '_call', injected)
    with pytest.raises(FactorizationError, match='injected'):
        PardisoFactorization(saddle_matrix())
    assert phases == [11, 22, -1]


def test_pardiso_cached_pointers_match_array_checked_binding_after_updates():
    import ctypes
    import gc
    from sparse_dot_mkl._mkl_interface._cfunctions import MKL

    argument_types = MKL._pardiso.argtypes
    matrix = saddle_matrix()
    factor = BACKENDS['pardiso'](matrix)
    rhs = np.arange(2 * matrix.shape[0], dtype=float)[::-2]
    rhs.flags.writeable = False
    preserved = rhs.copy()
    previous = None
    try:
        for scale in (1., 1e-7, 1e7, .5):
            factor.update(matrix * scale)
            # Drop temporary update arrays before using cached native pointers.
            gc.collect()
            solution = factor.solve(rhs)
            checked_rhs = np.ascontiguousarray(rhs)
            expected = np.empty_like(checked_rhs)
            integer = MKL.MKL_INT
            arg = lambda value: ctypes.byref(integer(value))
            error = integer(0)
            MKL._pardiso(factor._pt, arg(1), arg(1), arg(-2), arg(33), arg(matrix.shape[0]),
                factor._values, factor._indptr, factor._indices, factor._perm,
                arg(1), factor.iparm, arg(0), checked_rhs, expected, ctypes.byref(error))
            assert error.value == 0
            np.testing.assert_array_equal(solution, expected)
            np.testing.assert_array_equal(rhs, preserved)
            if previous is not None:
                assert not np.shares_memory(solution, previous)
            previous = solution
        # The instance binding must not disable validation for other MKL users.
        assert MKL._pardiso.argtypes is argument_types
    finally:
        factor.close()


@pytest.mark.parametrize('name', ['not-a-backend', None, []])
def test_unknown_backend_is_not_a_silent_fallback(name):
    with pytest.raises(ValueError, match='Unknown linear backend'):
        backend_type(name)


@pytest.mark.parametrize('name', ['qdldl', 'superlu', 'feral'])
def test_singular_matrix_reports_numerical_failure(name):
    if name == 'feral' and importlib.util.find_spec('feral') is None:
        pytest.skip('optional feral-solver is not installed')
    with pytest.raises(FactorizationError):
        BACKENDS[name](csc_array(np.diag([1., 0., -1.])))


def test_feral_parallel_budget_must_be_explicit(monkeypatch):
    pytest.importorskip('feral')
    monkeypatch.setenv('BENCH_THREADS', '4')
    monkeypatch.setenv('RAYON_NUM_THREADS', '8')
    with pytest.raises(ValueError, match='RAYON_NUM_THREADS'):
        BACKENDS['feral'](saddle_matrix())


@pytest.mark.parametrize('name,options', [
    ('qdldl', {'refinement': 2}), ('superlu', {'ordering': 'unknown'}),
    ('feral', {'scaling': 'unknown'}), ('feral', {'ordering': 'unknown'}),
    ('pardiso', {'matching': False}), ('pardiso', {'pivot_exponent': True}),
    ('pardiso', {'pivot_exponent': 21}), ('pardiso', {'scaling': 1}),
    ('pardiso', ['ordering']),
])
def test_invalid_backend_options_are_rejected_before_factorization(name, options):
    from glqp.linear import factor_and_solve
    with pytest.raises(ValueError):
        backend_options(name, options)
    with pytest.raises(ValueError):
        factor_and_solve(csc_array([[1.]]), np.zeros(1), csc_array([[1.]]), 0., None,
                         backend=name, backend_options=options)


def test_factor_reuse_includes_canonical_options():
    from glqp.linear import factor_and_solve
    matrix = saddle_matrix()
    shift = csc_array(eye_array(matrix.shape[0]))
    rhs = np.ones(matrix.shape[0])
    first = factor_and_solve(matrix, rhs, shift, 0., None, backend='pardiso')
    same = factor_and_solve(matrix, rhs, shift, 0., first.factorization,
                           backend='pardiso', backend_options={'ordering': 'metis'})
    assert same.factorization is first.factorization
    changed = factor_and_solve(matrix, rhs, shift, 0., same.factorization,
                              backend='pardiso', backend_options={'ordering': 'amd'})
    try:
        assert changed.factorization is not same.factorization
        assert changed.factorization.options['ordering'] == 'amd'
        with pytest.raises(RuntimeError, match='closed'):
            same.factorization.solve(rhs)
    finally:
        changed.factorization.close()


def test_feral_auto_ordering_still_uses_verified_float64_solve():
    pytest.importorskip('feral')
    from glqp import GLQP, SolverSettings
    model = GLQP(Q=csc_array([[2., .5], [.5, 1.]]), b=np.array([1., 2.]))
    x, result = model.solve(verbose=False, settings=SolverSettings(
        linear_solver_backend='feral', linear_solver_options={'ordering': 'auto'}))
    assert result.termination_tag == 'optimal'
    assert_allclose(np.array([[2., .5], [.5, 1.]]) @ x, [1., 2.], atol=1e-6)


def test_qdldl_cached_upper_triangle_is_bitwise_identical_to_native_full_input():
    import qdldl
    from glqp.backends import QDLDLFactorization
    matrix = saddle_matrix()
    cached = QDLDLFactorization(matrix)
    reference = qdldl.Solver(matrix)
    rhs = np.random.default_rng(43).normal(size=matrix.shape[0])
    upper = cached._upper
    upper_data = upper.data
    for scale in (1., 1.1, 1e-7, 1e7, .5):
        updated = matrix * scale
        cached.update(updated)
        reference.update(updated)
        left_l, left_d, left_p = cached.factor.factors()
        right_l, right_d, right_p = reference.factors()
        for left, right in ((left_l.data, right_l.data), (left_l.indices, right_l.indices),
                            (left_l.indptr, right_l.indptr), (left_d, right_d), (left_p, right_p),
                            (cached.solve(rhs), reference.solve(rhs))):
            np.testing.assert_array_equal(left, right)
        assert cached._upper is upper
        assert cached._upper.data is upper_data


def test_qdldl_updates_do_not_reextract_sparse_triangles(monkeypatch):
    import scipy.sparse
    from glqp.backends import QDLDLFactorization
    matrix = saddle_matrix()
    cached = QDLDLFactorization(matrix)

    def unexpected(*args, **kwargs):
        raise AssertionError('Sparse triangular extraction should not run during updates')

    monkeypatch.setattr(scipy.sparse, 'triu', unexpected)
    cached.update(matrix * 1.1)
    assert_allclose(matrix @ cached.solve(np.ones(matrix.shape[0])),
                    np.ones(matrix.shape[0]) / 1.1, atol=1e-12)
