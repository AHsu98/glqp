"""Residual screening checked against independent fresh floating-point blocks."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse._base import _spbase

from glqp.qp_ray import QPResidualNorms, QPResidualRay
from glqp.util import SparseTransposeCache


def fresh(model, values, step):
    x = values['x']+step*values['dx']
    y = values['y']+step*values['dy']
    s = values['s']+step*values['ds']
    nu = values['nu']+step*values['dnu']
    gradient = values['Qx']+step*values['Qdx']
    return (gradient+model.C.T@y+model.E.T@nu-model.b,
            model.C@x+s-model.c, y*s, model.E@x-model.e)


def make_ray(seed=123, scale=1., k=7, p=3):
    rng = np.random.default_rng(seed)
    n = 9
    model = SimpleNamespace(C=sparse.csc_array(scale*rng.normal(size=(k, n))),
        E=sparse.csc_array(scale*rng.normal(size=(p, n))),
        b=scale*rng.normal(size=n), c=scale*rng.normal(size=k), e=scale*rng.normal(size=p),
        _transpose_views=SparseTransposeCache())
    values = dict(x=rng.normal(size=n), y=rng.uniform(.1, 2., size=k),
        s=rng.uniform(.1, 2., size=k), nu=rng.normal(size=p),
        dx=rng.normal(size=n), dy=rng.normal(size=k), ds=rng.normal(size=k),
        dnu=rng.normal(size=p), Qx=scale*rng.normal(size=n), Qdx=scale*rng.normal(size=n))
    # Deliberately arbitrary/inexact directions with independent clipping.
    values['dy'][::2] = 0.
    values['ds'][1::2] = 0.
    rx, rp, _, req = fresh(model, values, 0.)
    norms = QPResidualNorms(model.C, model.E)
    ray = QPResidualRay(model, norms, rx=rx, rp=rp, req=req, **values)
    return model, values, ray, norms


def maximum(blocks):
    return max(float(np.max(np.abs(block), initial=0.)) for block in blocks)


@pytest.mark.parametrize('scale', [1e-9, 1., 1e12])
@pytest.mark.parametrize('step', [1e-6, .17, .9])
def test_actual_clipped_inexact_polynomial_and_error_envelope(scale, step):
    model, values, ray, _ = make_ray(scale=scale)
    trial_y, trial_s = values['y']+step*values['dy'], values['s']+step*values['ds']
    predicted = ray.residuals(step, trial_y, trial_s)
    actual = fresh(model, values, step)
    errors = ray._errors(step)
    for cache_block, fresh_block, error in zip(predicted, actual, errors):
        assert np.max(np.abs(cache_block-fresh_block), initial=0.) <= error
    # An exact-Newton shortcut would be incorrect for these directions.
    assert not np.allclose(predicted[0], (1-step)*ray.rx, rtol=1e-12, atol=scale*1e-14)
    np.testing.assert_array_equal(predicted[2], trial_y*trial_s)
    for limit in (0., maximum(actual)*.5, maximum(actual), maximum(actual)*2):
        report = ray.screen(step, trial_y, trial_s, residual_limit=limit, tolerance=1e-15)
        if report.can_reject:
            assert maximum(actual) >= max(limit, 1e-15)


def test_constructor_and_complementarity_rejection_do_no_products_or_scans():
    model, values, _, _ = make_ray()
    rx, rp, _, req = fresh(model, values, 0.)
    norms = QPResidualNorms(model.C, model.E)
    with patch.object(_spbase, '__matmul__', side_effect=AssertionError('unexpected product')):
        ray = QPResidualRay(model, norms, rx=rx, rp=rp, req=req, **values)
        report = ray.screen(.5, np.full_like(values['y'], 2.), np.full_like(values['s'], 3.),
                            residual_limit=5., tolerance=1e-7, allow_affine=False)
    assert report.can_reject and report.reason == 'complementarity'
    assert report.lower_bound == 6. and report.error_bound == 0.
    assert ray.slopes is None and ray.vector_norms is None and norms.bounds is None


@pytest.mark.parametrize('limit,tolerance,reject', [
    (6., 1e-7, True), (1e-7, 6., True),
    (np.nextafter(6., np.inf), 1e-7, False),
    (1e-7, np.nextafter(6., np.inf), False)])
def test_complementarity_screen_respects_both_strict_thresholds(limit, tolerance, reject):
    _, values, ray, _ = make_ray()
    report = ray.screen(.5, np.full_like(values['y'], 2.), np.full_like(values['s'], 3.),
        residual_limit=limit, tolerance=tolerance, allow_affine=False)
    assert report.can_reject is reject
    assert ray.slopes is None


def test_affine_bound_can_exclude_acceptance_without_complementarity_dominance():
    model = SimpleNamespace(C=sparse.csc_array((1, 1)), E=sparse.csc_array((0, 1)),
        b=np.zeros(1), c=np.array([-10.]), e=np.empty(0), _transpose_views=SparseTransposeCache())
    values = dict(x=np.zeros(1), y=np.array([1e-10]), s=np.array([1e-10]), nu=np.empty(0),
        dx=np.zeros(1), dy=np.zeros(1), ds=np.zeros(1), dnu=np.empty(0),
        Qx=np.zeros(1), Qdx=np.zeros(1))
    rx, rp, _, req = fresh(model, values, 0.)
    ray = QPResidualRay(model, QPResidualNorms(model.C, model.E), rx=rx, rp=rp, req=req, **values)
    report = ray.screen(.2, values['y'], values['s'], residual_limit=1., tolerance=1e-7)
    assert report.can_reject and report.reason == 'residual_gates_excluded'
    assert 1. < report.lower_bound <= maximum(fresh(model, values, .2))


def test_upward_product_retains_positive_underflow_bound():
    from glqp.qp_ray import _mul
    smallest = np.nextafter(0., 1.)
    assert _mul(smallest, .5) == smallest
    assert _mul(smallest, 0.) == 0.


def test_fallback_and_deferred_screen_remain_lazy():
    _, values, ray, norms = make_ray()
    with patch.object(_spbase, '__matmul__', side_effect=AssertionError('unexpected product')):
        fallback = ray.screen(.2, values['y'], values['s'], residual_limit=0., tolerance=0.,
                              fallback_eligible=True)
        deferred = ray.screen(.2, values['y'], values['s'], residual_limit=1e10, tolerance=1e-7,
                              allow_affine=False)
    assert not fallback.can_reject and fallback.reason == 'fallback_requires_fresh'
    assert not deferred.can_reject and deferred.reason == 'affine_screen_deferred'
    assert ray.slopes is None and norms.bounds is None


def test_four_direction_products_are_reused_with_shared_lazy_norms():
    model, values, ray, norms = make_ray()
    calls = []
    original = _spbase.__matmul__

    def observed(matrix, vector):
        calls.append(matrix.shape)
        return original(matrix, vector)

    with patch.object(_spbase, '__matmul__', observed):
        for step in (.1, .2, .4):
            ray.screen(step, values['y']+step*values['dy'], values['s']+step*values['ds'],
                       residual_limit=1e100, tolerance=1e-7)
    assert len(calls) == 4
    assert norms.bounds is not None
    original_bounds = norms.bounds
    assert norms.get() is original_bounds


def test_large_cancellation_requests_fresh_instead_of_false_rejection():
    model = SimpleNamespace(C=sparse.csc_array([[-1e16, 0.]]),
        E=sparse.csc_array((0, 2)), b=np.zeros(2), c=np.zeros(1), e=np.empty(0),
        _transpose_views=SparseTransposeCache())
    values = dict(x=np.zeros(2), y=np.ones(1), s=np.array([1e-10]), nu=np.empty(0),
        dx=np.zeros(2), dy=np.zeros(1), ds=np.zeros(1), dnu=np.empty(0),
        Qx=np.array([1e16, 0.]), Qdx=np.array([1., 0.]))
    rx, rp, _, req = fresh(model, values, 0.)
    ray = QPResidualRay(model, QPResidualNorms(model.C, model.E), rx=rx, rp=rp, req=req, **values)
    report = ray.screen(.5, values['y'], values['s'], residual_limit=.1, tolerance=1e-7)
    assert report.predicted_norm == .5
    assert maximum(fresh(model, values, .5)) < .1
    assert report.error_bound > .5
    assert not report.can_reject
    assert 0. <= report.lower_bound <= maximum(fresh(model, values, .5))


def test_stationarity_uncertainty_does_not_hide_primal_rejection():
    model = SimpleNamespace(C=sparse.csc_array([[1e12]]), E=sparse.csc_array((0, 1)),
        b=np.zeros(1), c=np.array([-1e-4]), e=np.empty(0),
        _transpose_views=SparseTransposeCache())
    values = dict(x=np.zeros(1), y=np.ones(1), s=np.array([1e-10]), nu=np.empty(0),
        dx=np.zeros(1), dy=np.zeros(1), ds=np.zeros(1), dnu=np.empty(0),
        Qx=np.array([-1e12]), Qdx=np.zeros(1))
    rx, rp, _, req = fresh(model, values, 0.)
    ray = QPResidualRay(model, QPResidualNorms(model.C, model.E), rx=rx, rp=rp, req=req, **values)
    report = ray.screen(.5, values['y'], values['s'], residual_limit=5e-5, tolerance=1e-7)
    assert report.error_bound > report.predicted_norm  # the global envelope is inconclusive
    assert report.can_reject and report.lower_bound >= 5e-5
    assert report.lower_bound <= maximum(fresh(model, values, .5))


@pytest.mark.parametrize('k,p', [(0, 0), (0, 3), (7, 0)])
def test_empty_constraint_blocks_have_valid_error_envelopes(k, p):
    model, values, ray, _ = make_ray(k=k, p=p)
    step = .2
    trial_y, trial_s = values['y']+step*values['dy'], values['s']+step*values['ds']
    report = ray.screen(step, trial_y, trial_s, residual_limit=1e100, tolerance=1e-7)
    assert report.reason == 'possible_acceptance_requires_fresh'
    actual = fresh(model, values, step)
    assert abs(report.predicted_norm-maximum(actual)) <= report.error_bound


def test_nonfinite_direction_cannot_reject_via_affine_cache():
    _, values, ray, _ = make_ray()
    values['Qdx'][0] = np.inf
    report = ray.screen(.2, values['y'], values['s'], residual_limit=1e100, tolerance=1e-7)
    assert not report.can_reject and report.reason == 'nonfinite_prediction'


def test_all_borrowed_inputs_remain_unchanged():
    _, values, ray, _ = make_ray()
    before = {name: value.copy() for name, value in values.items()}
    trial_y, trial_s = values['y']+.2*values['dy'], values['s']+.2*values['ds']
    ray.screen(.2, trial_y, trial_s, residual_limit=1e100, tolerance=1e-7)
    for name in values:
        np.testing.assert_array_equal(values[name], before[name])


def test_envelope_is_reused_for_contraction_and_rebuilt_for_growth(monkeypatch):
    model, values, ray, _ = make_ray()
    calls = []
    original = ray._errors
    def counted(step):
        calls.append(step)
        return original(step)
    monkeypatch.setattr(ray, '_errors', counted)
    for step in (.8, .4, .01, 1., .3):
        trial_y, trial_s = values['y']+step*values['dy'], values['s']+step*values['ds']
        report = ray.screen(step, trial_y, trial_s, residual_limit=1e100, tolerance=1e-7)
        actual = fresh(model, values, step)
        predicted = ray.residuals(step, trial_y, trial_s)
        assert all(np.max(np.abs(a-b), initial=0.) <= error
                   for a,b,error in zip(actual, predicted, report.block_error_bounds))
    assert calls == [.8, 1.]


@pytest.mark.parametrize('step,limit,tol', [(np.nan, 1., 1e-7), (0., 1., 1e-7),
    (.2, np.inf, 1e-7), (.2, 1., -1.)])
def test_invalid_controls_require_fresh_without_warming_cache(step, limit, tol):
    _, values, ray, norms = make_ray()
    report = ray.screen(step, values['y'], values['s'], residual_limit=limit, tolerance=tol)
    assert not report.can_reject and report.reason == 'invalid_controls'
    assert ray.slopes is None and norms.bounds is None
