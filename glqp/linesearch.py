"""Cached scalar search for a fixed GLM direction and barrier target.

This module receives no matrices: the caller supplies A@dx and the quadratic
ray coefficients once. It does not change the direction, regularization,
barrier target, or acceptance inequality. Derivatives choose trial steps only;
every accepted point passes the original finite, interior, strict Armijo test.
The centered proximal residual merit and non-descent restoration use other
searches. Inputs and loss parameters must remain unchanged during a search.
"""
from dataclasses import dataclass

import numpy as np

from .obj import (GaussianNLL, LogisticNLL, PoissonNLL, _weighted_logistic_tail,
                  _gaussian_value, _gaussian_gradient, _poisson_value,
                  _poisson_gradient, _poisson_curvature)
from .search_control import TrialDecision, search_trials
from .step_proposal import safeguarded_step


@dataclass(frozen=True)
class ScalarTrial:
    step: float
    merit: float
    predictor: np.ndarray
    slack: np.ndarray


@dataclass(frozen=True)
class ScalarSearchResult:
    trial: ScalarTrial | None
    status: str
    counters: dict
    history: tuple | None


def _weighted_square_sum(weights, values):
    """Avoid squaring a large direction before multiplying small curvature."""
    if np.all(weights >= 0):
        scaled = np.sqrt(weights)*values
        return float(np.dot(scaled, scaled))
    return float(np.dot(weights*values, values))


class CachedGLMRay:
    """The legacy merit on x+t*dx, s+t*ds, with mu held fixed.

    bx=b@x and bdx=b@dx use the solver's objective sign ``-b@x``.
    ``xQdx`` assumes the symmetric quadratic model used by the solver. Arrays
    are borrowed, never mutated. Exact built-in loss classes reuse their scalar
    intermediates across value/derivative requests; subclasses and custom losses
    retain their public callbacks. No derivative is needed for an accepted full
    step. Accepted derivatives are deliberately not a solver reuse contract.

    Replacing methods on a built-in class itself is unsupported: use a subclass
    or custom loss to change its mathematics. Observational class wrappers are
    harmless only if they preserve values; their callback counts omit fused ray
    evaluations. Use the ray's evaluation counters when instrumenting searches.
    """

    def __init__(self, loss, z, dz, s, ds, mu, xQx, xQdx, dxQdx, bx, bdx):
        self.loss = loss
        self.z, self.dz = np.asarray(z, dtype=float), np.asarray(dz, dtype=float)
        self.s, self.ds = np.asarray(s, dtype=float), np.asarray(ds, dtype=float)
        if (self.z.ndim != 1 or self.z.shape != self.dz.shape
                or self.s.ndim != 1 or self.s.shape != self.ds.shape):
            raise ValueError('Scalar ray vectors must have matching one-dimensional shapes')
        if not np.isfinite(mu) or mu < 0:
            raise ValueError('The fixed barrier target must be finite and nonnegative')
        self.mu = float(mu)
        self.xQx, self.xQdx, self.dxQdx = float(xQx), float(xQdx), float(dxQdx)
        self.bx, self.bdx = float(bx), float(bdx)
        # Instance callback overrides may be instrumentation or custom behavior;
        # neither should be bypassed by an exact-type fused kernel.
        overridden = any(name in getattr(loss, '__dict__', {})
                         for name in ('f', 'd1f', 'd2f', '__call__'))
        self._kind = None if overridden else type(loss)
        self._builtin = self._kind in (GaussianNLL, LogisticNLL, PoissonNLL)
        if self._builtin:
            self._m = loss.m
            self._y = np.asarray(loss.y, dtype=float)
            self._w = np.broadcast_to(loss.w, self._y.shape)
        self._active = None
        self._active_ready = False
        self._model_allowed = None
        self.value_evaluations = 0
        self.derivative_evaluations = 0
        self._last_trial = self._last_aux = self._last_derivatives = None

    @property
    def model_allowed(self):
        """Defer model checks until a full step has actually been rejected."""
        if self._model_allowed is None:
            self._model_allowed = bool(np.isfinite(self.dxQdx) and self.dxQdx >= 0
                and (not self._builtin
                     or np.all(np.isfinite(self._w) & (self._w >= 0))))
        return self._model_allowed

    def _poisson_observations(self, predictor):
        # Match the public all-nonzero fast path, and construct an active mask
        # only for this loss. Parameters are fixed for the duration of the ray.
        if not self._active_ready:
            active = self._w != 0
            self._active = None if np.all(active) else active
            self._active_ready = True
        z = predictor[:self._m]
        if self._active is None:
            return z, self._y, self._w
        return z[self._active], self._y[self._active], self._w[self._active]

    def evaluate(self, step):
        """Evaluate the absolute merit, preserving existing loss arithmetic."""
        self.value_evaluations += 1
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            predictor = self.z + step*self.dz
            slack = self.s + step*self.ds
            aux = None
            if self._kind is PoissonNLL:
                observed, poisson_y, poisson_w = self._poisson_observations(predictor)
            else:
                observed = predictor[:self._m] if self._builtin else predictor
            # Built-in padding and exact-zero-weight Poisson observations are
            # intentionally unused. Custom callbacks receive the whole vector.
            if (not np.all(np.isfinite(observed))
                    or not np.all(np.isfinite(slack)) or np.any(slack <= 0)):
                merit = np.inf
            else:
                if self._kind is GaussianNLL:
                    error = predictor[:self._m]-self._y
                    value = _gaussian_value(observed, self._y, self._w, error)
                    aux = error
                elif self._kind is LogisticNLL:
                    z = predictor[:self._m]
                    tail, weighted_tail = _weighted_logistic_tail(z, self._w)
                    softplus = self._w*np.log1p(tail)
                    subnormal = tail < np.finfo(float).tiny
                    softplus[subnormal] = weighted_tail[subnormal]
                    linear = np.where(z >= 0, (1-self._y)*z, -self._y*z)
                    value = np.sum(softplus+self._w*linear)
                    aux = tail, weighted_tail
                elif self._kind is PoissonNLL:
                    # Skip zero weights before exp, matching PoissonNLL exactly.
                    rate = np.exp(observed)
                    value = _poisson_value(observed, poisson_y, poisson_w, rate)
                    aux = rate
                else:
                    value = self.loss(predictor)
                quadratic = .5*(self.xQx+2*step*self.xQdx+step*step*self.dxQdx)
                barrier = -self.mu*np.sum(np.log(slack)) if self.mu else 0.
                merit = float(value+quadratic-(self.bx+step*self.bdx)+barrier)
        trial = ScalarTrial(float(step), float(merit), predictor, slack)
        self._last_trial, self._last_aux = trial, aux
        self._last_derivatives = None
        return trial

    def derivatives(self, trial):
        """Return the first and second ray derivatives at a finite point.

        The point must be the latest evaluation so cached loss data cannot be
        confused with another trial. Nonfinite objective values never trigger
        custom derivative callbacks. A nonfinite derivative merely disables its
        proposal; it cannot invalidate a previously accepted finite trial.
        Numerical/domain exceptions inside custom derivative callbacks have the
        same fallback. Conversion, shape, and programming errors remain visible.
        """
        if trial is not self._last_trial:
            raise ValueError('Derivatives require the latest scalar trial')
        if not np.isfinite(trial.merit):
            return np.nan, np.nan
        if self._last_derivatives is not None:
            return self._last_derivatives
        self.derivative_evaluations += 1
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            if self._kind is GaussianNLL:
                dz = self.dz[:self._m]
                gradient = _gaussian_gradient(trial.predictor[:self._m], self._y,
                                               self._w, self._last_aux)
                curvature = self._w
            elif self._kind is LogisticNLL:
                z, dz = trial.predictor[:self._m], self.dz[:self._m]
                tail, weighted_tail = self._last_aux
                probability = weighted_tail/(1+tail)
                gradient = np.where(z >= 0, self._w*(1-self._y)-probability,
                                    probability-self._w*self._y)
                curvature = weighted_tail/(1+tail)**2
            elif self._kind is PoissonNLL:
                dz = self.dz[:self._m]
                observed, y, w = self._poisson_observations(trial.predictor)
                if self._active is not None:
                    dz = dz[self._active]
                rate = self._last_aux
                gradient = _poisson_gradient(observed, y, w, rate)
                curvature = _poisson_curvature(observed, w, rate)
            else:
                dz = self.dz
                # A finite boundary value need not have finite derivatives,
                # e.g. -sqrt(z) at z=0. These derivatives only propose a step;
                # an undefined proposal must not prevent ordinary contraction.
                # Catch only callback invocations so malformed return values
                # and errors in our own arithmetic cannot be silently hidden.
                try:
                    gradient_value = self.loss.d1f(trial.predictor)
                    curvature_value = self.loss.d2f(trial.predictor)
                except (ArithmeticError, ValueError):
                    self._last_derivatives = np.nan, np.nan
                    return self._last_derivatives
                gradient = np.asarray(gradient_value)
                curvature = np.asarray(curvature_value)
                if gradient.shape != dz.shape or curvature.shape != dz.shape:
                    raise ValueError('Custom loss derivatives must match the predictor shape')
            barrier_slope = barrier_curvature = 0.
            if self.mu:
                ratio = self.ds/trial.slack
                barrier_slope = -np.sum(self.mu*ratio)
                scaled_barrier = np.sqrt(self.mu)*ratio
                barrier_curvature = np.dot(scaled_barrier, scaled_barrier)
            slope = (np.dot(gradient, dz)+self.xQdx+trial.step*self.dxQdx-self.bdx
                     +barrier_slope)
            second = (_weighted_square_sum(curvature, dz)+self.dxQdx
                      +barrier_curvature)
        self._last_derivatives = float(slope), float(second)
        return self._last_derivatives

    def poisson_proposal(self, slope0, curvature0):
        """A log-growth step proposal, not an acceptance or curvature bound.

        With L=max positive predictor change, constant-base Poisson curvature
        grows at most exponentially at rate L. Integrating that model suggests
        log1p(-slope0*L/curvature0)/L. Including quadratic/barrier curvature in
        curvature0 is useful scaling but does NOT prove the same growth bound
        for the barrier. Every resulting point must still pass Armijo.
        """
        if (self._kind is not PoissonNLL or not self.model_allowed
                or not np.isfinite(curvature0) or curvature0 <= 0):
            return None
        growth = float(np.max(self.dz[:self._m][self._w > 0], initial=0.))
        if not np.isfinite(growth) or growth <= 0:
            return None
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            # The log form avoids overflow in (-slope0)*growth/curvature0.
            argument = np.log(-slope0)+np.log(growth)-np.log(curvature0)
            step = float(np.logaddexp(0., argument)/growth)
        return step if np.isfinite(step) and step > 0 else None


def search_scalar_ray(ray, *, tmax, merit0, slope0, armijo_param, additive_eps,
                      max_trials, trial_is_valid=None, record_history=False):
    """Try full step, then safeguarded scalar proposals for a descent ray.

    ``trial_is_valid(t)`` lets the caller check finite x/y and dual positivity.
    Slack positivity is checked here too. A rejected upper step is strictly
    reduced. Ordinary Newton/quadratic proposals stay within [0.1*t,0.5*t];
    a single Poisson growth proposal may jump farther after the full-step trial.
    Nonfinite/negative curvature and known negative weights or quadratic
    curvature use halving. Custom losses need not be convex for Armijo safety.
    ``value_evaluations`` includes the optional base-model evaluation at zero;
    ``trial_count`` counts positive-step acceptance attempts only.
    """
    if (not np.isfinite(armijo_param) or not 0 < armijo_param < 1
            or not np.isfinite(additive_eps) or additive_eps < 0
            or not isinstance(max_trials, (int, np.integer)) or max_trials < 0):
        raise ValueError('Invalid scalar line-search controls')
    counters = dict(trial_count=0, value_evaluations=0, derivative_evaluations=0,
                    nonfinite_count=0, invalid_trial_count=0, armijo_failure_count=0,
                    poisson_model_steps=0, newton_steps=0, interpolation_steps=0,
                    contraction_steps=0)
    history = [] if record_history else None
    value_start, derivative_start = ray.value_evaluations, ray.derivative_evaluations

    def finish(trial, status):
        counters['value_evaluations'] = ray.value_evaluations-value_start
        counters['derivative_evaluations'] = ray.derivative_evaluations-derivative_start
        return ScalarSearchResult(trial, status, counters,
                                  None if history is None else tuple(history))

    if not np.isfinite(merit0):
        return finish(None, 'invalid_base_merit')
    if not np.isfinite(slope0) or slope0 >= 0:
        return finish(None, 'not_descent')
    if not np.isfinite(tmax) or tmax <= 0:
        return finish(None, 'invalid_step')

    proposal_kind = 'full_step'
    poisson_model_tried = False

    def attempt(step, index):
        if not np.isfinite(step) or step <= 0:
            return TrialDecision(reason='step_underflow', stop=True)
        counters['trial_count'] += 1
        point = None
        valid = trial_is_valid is None or trial_is_valid(step)
        if not valid:
            reason = 'invalid_trial'
            counters['invalid_trial_count'] += 1
        else:
            point = ray.evaluate(step)
            if not np.isfinite(point.merit):
                reason = 'nonfinite_merit'
                counters['nonfinite_count'] += 1
            else:
                with np.errstate(over='ignore', invalid='ignore'):
                    bound = merit0+armijo_param*step*slope0+additive_eps
                if np.isfinite(bound) and point.merit < bound:
                    if history is not None:
                        history.append(dict(step=step, merit=point.merit,
                                            proposal=proposal_kind, reason='armijo'))
                    return TrialDecision(point, 'armijo', accepted=True)
                reason = 'armijo'
                counters['armijo_failure_count'] += 1
        if history is not None:
            history.append(dict(step=step,
                                merit=None if point is None or not np.isfinite(point.merit)
                                else point.merit, proposal=proposal_kind, reason=reason))
        return TrialDecision(point, reason)

    def propose(step, decision):
        nonlocal proposal_kind, poisson_model_tried
        point = decision.point
        proposed = None
        proposal_kind = 'contraction'
        if (not poisson_model_tried and ray._kind is PoissonNLL and ray.model_allowed):
            poisson_model_tried = True
            base = ray.evaluate(0.)
            _, curvature = ray.derivatives(base)
            candidate = ray.poisson_proposal(slope0, curvature)
            if candidate is not None and candidate < .5*step:
                proposed, proposal_kind = candidate, 'poisson_model'
                counters['poisson_model_steps'] += 1
            # Evaluating the base replaced the cached rejected point.
            point = None
        if (proposed is None and ray.model_allowed and point is not None
                and np.isfinite(point.merit)):
            derivative, curvature = ray.derivatives(point)
            proposed, proposal_kind = safeguarded_step(
                step, point.merit, merit0, slope0, derivative, curvature)
            counters[proposal_kind+'_steps'] += 1
        if proposed is None:
            proposed = .5*step
            counters['contraction_steps'] += 1
        return proposed

    result = search_trials(float(tmax), max_trials, attempt, propose)
    return finish(result.point, result.reason)
