"""Conservative rejection screening for an actual fixed QP direction.

This cache never accepts a point. A possible residual acceptance, an enabled
fallback, or an uncertain/nonfinite prediction requires the caller's unchanged
fresh products and predicates. Original residuals need not satisfy an exact
Newton equation: proximal regularization, inexact solves and clipping are all
allowed. Matrices and borrowed vectors must remain unchanged during the ray.

Bounds assume float64 arithmetic with gradual underflow and no flushing of
subnormal inputs. They use the standard sparse-dot/addition model, inflated
absolute row sums, and a normal-underflow allowance per arithmetic operation.
They bound differences from the caller's fresh constraint products combined
with Qx+t*Qdx, not from a newly evaluated Q@(x+t*dx). They are conservative
roundoff envelopes, not interval-arithmetic certificates for arbitrary BLAS,
hardware failures, changing matrices, or differently grouped residuals.
"""
from dataclasses import dataclass
import math

import numpy as np


# eps is twice the round-to-nearest unit roundoff: using k*eps covers a
# length-k sparse dot with up to 2k separate multiply/add operations.
_EPS = float(np.finfo(float).eps)
_TINY = float(np.finfo(float).tiny)


def _norm(vector):
    return float(np.max(np.abs(vector), initial=0.))


def _up(value):
    return math.nextafter(value, math.inf) if value > 0 else float(value)


def _sum(*values):
    result = 0.
    for value in values:
        result = _up(result + value)
    return result


def _mul(first, second):
    product = first*second
    if product == 0 and first > 0 and second > 0:
        return math.nextafter(0., math.inf)
    return _up(product)


def _roundoff(operations, magnitude):
    scaled = operations*_EPS
    if not math.isfinite(magnitude) or scaled >= .5:
        return np.inf
    gamma = _up(scaled/(1-scaled))
    return _sum(_mul(gamma, magnitude), _mul(operations, _TINY))


@dataclass(frozen=True)
class _MatrixBound:
    norm: float
    row_nnz: int

    def error(self, vector_norm):
        return _roundoff(self.row_nnz, _mul(self.norm, vector_norm))

    def size(self, vector_norm):
        return _sum(_mul(self.norm, vector_norm), self.error(vector_norm))


def _matrix_bound(matrix):
    rows = matrix.tocsr()
    count = int(np.max(np.diff(rows.indptr), initial=0))
    if count == 0:
        return _MatrixBound(0., 0)
    sums = np.asarray(abs(rows).sum(axis=1)).reshape(-1)
    observed = float(np.max(sums, initial=0.))
    scaled = count*_EPS
    # Inflate the nonnegative row-sum computation before using it as a bound.
    bound = np.inf if scaled >= .5 else _up(_sum(observed, _mul(count, _TINY))/(1-scaled))
    return _MatrixBound(bound, count)


class QPResidualNorms:
    """Lazy matrix bounds, created anew for each solve and shared by its rays."""
    def __init__(self, C, E):
        self.C, self.E = C, E
        self.bounds = None

    def get(self):
        if self.bounds is None:
            self.bounds = dict(C=_matrix_bound(self.C), CT=_matrix_bound(self.C.T),
                               E=_matrix_bound(self.E), ET=_matrix_bound(self.E.T))
        return self.bounds


@dataclass(frozen=True)
class QPResidualScreen:
    predicted_norm: float | None
    error_bound: float | None
    lower_bound: float | None
    can_reject: bool
    reason: str
    block_error_bounds: tuple | None = None


class QPResidualRay:
    """A lazy original-residual polynomial from the supplied clipped direction.

The base rx/rp/req must be the current original residuals from the solver's
normal grouping, computed with the supplied Qx and the same x/y/s/nu. Qdx is
already available from the objective ray. Constructor calls perform no matrix
products or scans. The first residual prediction uses exactly C@dx, E@dx,
C.T@dy and E.T@dnu; subsequent predictions use vector arithmetic only.
"""
    def __init__(self, model, norms, *, rx, rp, req, x, y, s, nu,
                 dx, dy, ds, dnu, Qx, Qdx):
        self.model, self.norms = model, norms
        self.rx, self.rp, self.req = rx, rp, req
        self.x, self.y, self.s, self.nu = x, y, s, nu
        self.dx, self.dy, self.ds, self.dnu = dx, dy, ds, dnu
        self.Qx, self.Qdx = Qx, Qdx
        self.slopes = None
        self.direction_products = None
        self.vector_norms = None
        self.envelope_step = 0.
        self.envelope = None

    def _initialize(self):
        if self.slopes is None:
            model = self.model
            CT = model._transpose_views.get('C', model.C)
            ET = model._transpose_views.get('E', model.E)
            CTdy, ETdnu = CT@self.dy, ET@self.dnu
            Cdx, Edx = model.C@self.dx, model.E@self.dx
            self.direction_products = CTdy, ETdnu, Cdx, Edx
            self.slopes = self.Qdx+CTdy+ETdnu, Cdx+self.ds, Edx
            names = ('x', 'y', 's', 'nu', 'dx', 'dy', 'ds', 'dnu', 'Qx', 'Qdx', 'rx', 'rp', 'req')
            self.vector_norms = {name: _norm(getattr(self, name)) for name in names}
            self.vector_norms.update(b=_norm(model.b), c=_norm(model.c), e=_norm(model.e))
            self.slope_norms = tuple(_norm(direction) for direction in self.slopes)

    def residuals(self, step, trial_y, trial_s):
        """Predict blocks; supplied dual/slack arrays must be the actual trial."""
        self._initialize()
        dx, dp, de = self.slopes
        return (self.rx+step*dx, self.rp+step*dp,
                trial_y*trial_s, self.req+step*de)

    def _errors(self, step):
        """Bound fresh-minus-cached blocks, including base cancellation error."""
        t = abs(float(step))
        v = self.vector_norms
        C, CT, E, ET = (self.norms.get()[key] for key in ('C', 'CT', 'E', 'ET'))
        exact_size = {name: _sum(v[name], _mul(t, v['d'+name])) for name in ('x', 'y', 's', 'nu')}
        formation_error = {name: _roundoff(2, value) for name, value in exact_size.items()}
        trial_size = {name: _sum(exact_size[name], formation_error[name]) for name in exact_size}

        def fresh_dot(bound, name):
            error = _sum(bound.error(trial_size[name]), _mul(bound.norm, formation_error[name]))
            size = _sum(_mul(bound.norm, exact_size[name]), error)
            return size, error

        cty, cty_error = fresh_dot(CT, 'y')
        etnu, etnu_error = fresh_dot(ET, 'nu')
        cx, cx_error = fresh_dot(C, 'x')
        ex, ex_error = fresh_dot(E, 'x')
        gradient = _sum(v['Qx'], _mul(t, v['Qdx']))
        gradient_error = _roundoff(2, gradient)
        fresh_x = _sum(gradient_error, cty_error, etnu_error,
            _roundoff(3, _sum(gradient, gradient_error, cty, etnu, v['b'])))
        fresh_p = _sum(cx_error, formation_error['s'],
            _roundoff(2, _sum(cx, trial_size['s'], v['c'])))
        fresh_e = _sum(ex_error, _roundoff(1, _sum(ex, v['e'])))

        base_x = _sum(CT.error(v['y']), ET.error(v['nu']),
            _roundoff(3, _sum(v['Qx'], CT.size(v['y']), ET.size(v['nu']), v['b'])))
        base_p = _sum(C.error(v['x']), _roundoff(2, _sum(C.size(v['x']), v['s'], v['c'])))
        base_e = _sum(E.error(v['x']), _roundoff(1, _sum(E.size(v['x']), v['e'])))
        slope_x = _sum(CT.error(v['dy']), ET.error(v['dnu']),
            _roundoff(2, _sum(v['Qdx'], CT.size(v['dy']), ET.size(v['dnu']))))
        slope_p = _sum(C.error(v['dx']), _roundoff(1, _sum(C.size(v['dx']), v['ds'])))
        slope_e = E.error(v['dx'])
        errors = []
        for base, slope, fresh, residual, direction in zip(
                (base_x, base_p, base_e), (slope_x, slope_p, slope_e),
                (fresh_x, fresh_p, fresh_e), ('rx', 'rp', 'req'), self.slope_norms):
            combine = _roundoff(2, _sum(v[residual], _mul(t, direction)))
            errors.append(_sum(base, _mul(t, slope), combine, fresh))
        # Complementarity is the exact same elementwise operation on the same
        # rounded trial arrays as the fresh check, so no difference bound needed.
        return errors[0], errors[1], 0., errors[2]

    def screen(self, step, trial_y, trial_s, *, residual_limit, tolerance,
               fallback_eligible=False, allow_affine=True):
        """Reject only when the roundoff envelope excludes both residual gates.

The caller must perform fresh verification for every report with can_reject
False. It must also run its usual finite/interior and Armijo gates first. A
fallback-eligible step always requests fresh verification without warming the
cache. This method never changes a direction, target, or acceptance threshold.
"""
        if fallback_eligible:
            return QPResidualScreen(None, None, None, False, 'fallback_requires_fresh')
        if (not np.isfinite(step) or step <= 0 or not np.isfinite(residual_limit)
                or not np.isfinite(tolerance) or tolerance < 0):
            return QPResidualScreen(None, None, None, False, 'invalid_controls')
        with np.errstate(over='ignore', invalid='ignore'):
            complementarity = _norm(trial_y*trial_s)
        if np.isfinite(complementarity) and complementarity >= max(residual_limit, tolerance):
            return QPResidualScreen(complementarity, 0., complementarity, True, 'complementarity')
        if not allow_affine:
            return QPResidualScreen(None, None, None, False, 'affine_screen_deferred')
        with np.errstate(over='ignore', invalid='ignore', under='ignore'):
            blocks = self.residuals(step, trial_y, trial_s)
            block_norms = tuple(_norm(block) for block in blocks)
            predicted = max(block_norms)
            # Every bound term is nonnegative and nondecreasing in |step|.
            # Reuse the first envelope as backtracking contracts the ray; grow
            # it only for a caller that later requests a larger step.
            if self.envelope is None or step > self.envelope_step:
                self.envelope = self._errors(step)
                self.envelope_step = step
            errors = self.envelope
            error = max(errors)
        if (not all(np.all(np.isfinite(block)) for block in blocks)
                or not all(np.isfinite(value) for value in errors)):
            return QPResidualScreen(predicted, error, None, False, 'nonfinite_prediction', errors)
        # Each block has its own error envelope. A large stationarity error
        # must not hide a decisive primal/equality residual lower bound.
        lower = max(0., max(math.nextafter(norm-bound, -math.inf)
                            for norm, bound in zip(block_norms, errors)))
        reject = lower >= max(residual_limit, tolerance)
        return QPResidualScreen(predicted, error, lower, reject,
            'residual_gates_excluded' if reject else 'possible_acceptance_requires_fresh', errors)
