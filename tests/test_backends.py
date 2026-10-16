"""Independent dense solves and residuals exercise each backend lifecycle."""
import importlib.util

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import csc_array, diags_array, eye_array

from glqp.backends import BACKENDS, FactorizationError, backend_type


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
