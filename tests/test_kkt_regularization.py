"""Independent block-equation checks for separate proximal shifts."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import csc_array

from glqp.kkt import KKTWorkspace


def make_system(n, k, p):
    rng = np.random.default_rng(817)
    h = rng.normal(size=(n, n))
    h = h.T @ h + np.eye(n)
    c = rng.normal(size=(k, n))
    e = rng.normal(size=(p, n))
    y, s = rng.uniform(.2, 2., size=(2, k))
    residuals = tuple(rng.normal(size=size) for size in (n, k, k, p))
    return tuple(csc_array(a) for a in (h, c, e)), y, s, residuals


def dense_four_blocks(h, c, e, y, s, rho_x, rho_y, rho_s):
    """The original equations, before eliminating ds or scaling dy."""
    n, k, p = len(h), len(c), len(e)
    return np.block([
        [h + rho_x*np.eye(n), np.zeros((n, k)), c.T, e.T],
        [c, np.eye(k), -rho_y*np.eye(k), np.zeros((k, p))],
        [np.zeros((k, n)), np.diag(y + rho_s), np.diag(s), np.zeros((k, p))],
        [e, np.zeros((p, k)), np.zeros((p, k)), -rho_y*np.eye(p)],
    ])


@pytest.mark.parametrize("shape", [(4, 3, 2), (4, 0, 2), (4, 3, 0), (2, 0, 0)])
@pytest.mark.parametrize("shifts", [(.13, .07, .23), (.2, .03, 0.),
                                   (0., .2, 0.), (.2, 0., .03), (0., 0., 0.)])
def test_elimination_matches_independent_four_block_system(shape, shifts):
    n, k, p = shape
    rho_x, rho_y, rho_s = shifts
    (h, c, e), y, s, residuals = make_system(n, k, p)
    workspace = KKTWorkspace(h, c, e)
    matrix, rhs, w = workspace.assemble(
        h, y, s, *residuals, rho_x,
        dual_reg=rho_y, complementarity_reg=rho_s)
    scaled = np.linalg.solve(matrix.toarray(), rhs)
    dx, dy, dnu = scaled[:n], w*scaled[n:n+k], scaled[n+k:]
    ds = -residuals[1] - c @ dx + rho_y*dy
    direction = np.r_[dx, ds, dy, dnu]
    blocks = dense_four_blocks(h.toarray(), c.toarray(), e.toarray(),
                               y, s, *shifts)
    expected = np.linalg.solve(blocks, -np.concatenate(residuals))
    assert_allclose(direction, expected, rtol=5e-13, atol=5e-13)
    assert_allclose(blocks @ direction, -np.concatenate(residuals),
                    rtol=5e-13, atol=5e-13)


def test_consistent_blocks_are_the_jacobian_of_fixed_center_residuals():
    """Check the physical residual map, independently of reduced KKT storage."""
    (h, c, e), y, s, _ = make_system(4, 3, 2)
    h, c, e = (a.toarray() for a in (h, c, e))
    rng = np.random.default_rng(183)
    x, nu = rng.normal(size=4), rng.normal(size=2)
    center_x, center_y, center_nu = (rng.normal(size=size) for size in (4, 3, 2))
    linear, bounds, equalities = (rng.normal(size=size) for size in (4, 3, 2))
    rho_x, rho_y, mu = .13, .07, .11

    def residual(z):
        x, s, y, nu = z[:4], z[4:7], z[7:10], z[10:]
        return np.r_[
            h @ x + linear + c.T @ y + e.T @ nu + rho_x*(x - center_x),
            c @ x + s - bounds - rho_y*(y - center_y),
            s*y - mu,
            e @ x - equalities - rho_y*(nu - center_nu),
        ]

    point = np.r_[x, s, y, nu]
    # Complex-step derivatives avoid subtractive cancellation and do not reuse
    # any of the workspace's elimination/scaling equations.
    jacobian = np.column_stack([
        np.imag(residual(point + 1e-30j*column))/1e-30
        for column in np.eye(len(point))
    ])
    expected = dense_four_blocks(h, c, e, y, s, rho_x, rho_y, 0.)
    assert_allclose(jacobian, expected, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("keyword", ["dual_reg", "complementarity_reg"])
def test_one_omitted_shift_inherits_primal_shift(keyword):
    (h, c, e), y, s, residuals = make_system(4, 3, 2)
    workspace = KKTWorkspace(h, c, e)
    matrix, rhs, w = workspace.assemble(h, y, s, *residuals, .13, **{keyword: .07})
    actual = (matrix.toarray().copy(), rhs.copy(), w.copy())
    explicit = dict(dual_reg=.13, complementarity_reg=.13)
    explicit[keyword] = .07
    matrix, rhs, w = workspace.assemble(h, y, s, *residuals, .13, **explicit)
    for a, b in zip(actual, (matrix.toarray(), rhs, w)):
        assert_array_equal(a, b)


def test_two_rhs_solves_preserve_matrix_and_reuse_current_scaling():
    (h, c, e), y, s, residuals = make_system(4, 3, 2)
    workspace = KKTWorkspace(h, c, e)
    matrix, rhs, w = workspace.assemble(
        h, y, s, *residuals, .13, dual_reg=.07, complementarity_reg=0.)
    original_data, original_weights = matrix.data.copy(), w.copy()
    other_residuals = tuple(-2*r for r in residuals)
    new_rhs = workspace.assemble_rhs(*other_residuals)
    assert new_rhs is rhs
    assert_array_equal(matrix.data, original_data)
    assert_array_equal(w, original_weights)
    scaled = np.linalg.solve(matrix.toarray(), new_rhs)
    dx, dy, dnu = scaled[:4], w*scaled[4:7], scaled[7:]
    ds = -other_residuals[1] - c @ dx + .07*dy
    blocks = dense_four_blocks(h.toarray(), c.toarray(), e.toarray(),
                               y, s, .13, .07, 0.)
    assert_allclose(blocks @ np.r_[dx, ds, dy, dnu],
                    -np.concatenate(other_residuals), rtol=1e-13, atol=1e-13)


def test_switching_back_to_legacy_assembly_clears_all_independent_shifts():
    (h, c, e), y, s, residuals = make_system(4, 3, 2)
    workspace = KKTWorkspace(h, c, e)
    initial = workspace.assemble(h, y, s, *residuals, .013)
    before = (initial[0].data.copy(), initial[1].copy(), initial[2].copy())
    workspace.assemble(h, y, s, *residuals, .13,
                       dual_reg=.07, complementarity_reg=0.)
    matrix, rhs, w = workspace.assemble(h, y, s, *residuals, .013)
    assert_array_equal(matrix.data, before[0])
    assert_array_equal(rhs, before[1])
    assert_array_equal(w, before[2])


@pytest.mark.parametrize("keyword", ["prox_reg", "dual_reg", "complementarity_reg"])
@pytest.mark.parametrize("invalid", [-1., np.inf, np.nan, [0., 1.], 1j, "invalid"])
def test_explicit_regularization_rejects_invalid_values_before_mutation(keyword, invalid):
    (h, c, e), y, s, residuals = make_system(4, 3, 2)
    workspace = KKTWorkspace(h, c, e)
    matrix, rhs, w = workspace.assemble(h, y, s, *residuals, .013)
    before = (matrix.data.copy(), rhs.copy(), w.copy())
    options = dict(prox_reg=.13, dual_reg=.07, complementarity_reg=0.)
    options[keyword] = invalid
    with pytest.raises(ValueError, match=f"{keyword} must be a finite nonnegative scalar"):
        workspace.assemble(h, y, s, *residuals, **options)
    for actual, expected in zip((matrix.data, rhs, w), before):
        assert_array_equal(actual, expected)
