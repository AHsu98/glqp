"""Separable GLM losses evaluated on the first ``len(y)`` predictors.

Predictor vectors may include trailing coordinates unused by the likelihood.
Gradients and Hessian diagonals have the input length, with zeros in those
coordinates. Likelihood constants independent of the predictors are omitted.
"""

import numpy as np

from . import weighted_loss


def _pad_derivative(values, size):
    """Return a full predictor derivative without copying an exact-size result."""
    if len(values) == size:
        return values
    result = np.zeros(size)
    result[:len(values)] = values
    return result


def _gaussian_value(z, y, w, error=None):
    """Keep ordinary arithmetic; rescue squares formed outside float64 range."""
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        error = z-y if error is None else error
        squared = error**2
        value = .5*np.dot(w, squared)
        if (not np.isfinite(value)
                or np.any((squared < weighted_loss.TINY) & (error != 0) & (np.abs(w) > 1))):
            if np.all(np.isfinite(z)) and np.all(np.isfinite(y)) and np.all(np.isfinite(w)):
                value = weighted_loss.gaussian_value(z, y, np.broadcast_to(w, z.shape))
        return value


def _gaussian_gradient(z, y, w, error=None):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        error = z-y if error is None else error
        gradient = w*error
        if not np.all(np.isfinite(error)):
            rescue = ~np.isfinite(error) & np.isfinite(z) & np.isfinite(y) & np.isfinite(w)
            weights = np.broadcast_to(w, z.shape)[rescue]
            gradient[rescue] = weighted_loss.gaussian_gradient(z[rescue], y[rescue], weights)
        return gradient


def _poisson_value(z, y, w, rate):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        response = y*z
        value = np.dot(w, rate-response)
        underflow = ((rate < weighted_loss.TINY)
                     | ((np.abs(response) < weighted_loss.TINY) & (y != 0) & (z != 0)))
        if not np.isfinite(value) or np.any(underflow & (np.abs(w) > 1)):
            value = weighted_loss.poisson_value(z, y, w, rate)
        return value


def _poisson_gradient(z, y, w, rate):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        gradient = w*(rate-y)
        rescue = (~np.isfinite(gradient) | ~np.isfinite(rate)
                  | ((rate < weighted_loss.TINY) & (np.abs(w) > 1)))
        if np.any(rescue):
            weights = np.broadcast_to(w, z.shape)
            gradient[rescue] = weighted_loss.poisson_gradient(
                z[rescue], y[rescue], weights[rescue], rate[rescue])
        return gradient


def _poisson_curvature(z, w, rate):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        curvature = w*rate
        rescue = ~np.isfinite(rate) | ((rate < weighted_loss.TINY) & (np.abs(w) > 1))
        if np.any(rescue):
            weights = np.broadcast_to(w, z.shape)
            curvature[rescue] = weighted_loss.poisson_curvature(z[rescue], weights[rescue], rate[rescue])
        return curvature


def _poisson_rate(z):
    # An overflowing unweighted rate may still have a finite weighted value.
    with np.errstate(over='ignore', under='ignore'):
        return np.exp(z)


def _weighted_logistic_tail(z, weights):
    """Return exp(-abs(z)) and its weighted value without premature underflow.

    Ordinary predictors use direct multiplication. A positive large weight can
    rescue a subnormal unweighted tail; only that rare case uses logarithms.
    """
    magnitude = np.abs(z)
    tail = np.exp(-magnitude)
    weighted = weights * tail
    rescued = ((tail < np.finfo(float).tiny) & (weights > 0)
               & np.isfinite(weights) & np.isfinite(magnitude))
    if np.any(rescued):
        selected_weights = np.broadcast_to(weights, tail.shape)[rescued]
        weighted[rescued] = np.exp(np.log(selected_weights) - magnitude[rescued])
    return tail, weighted


class LogisticNLL:
    def __init__(self, y, w):
        """
        y: array-like of binary responses (0 or 1)
        w: array-like of corresponding weights
        """
        self.y = np.array(y, dtype=float)
        self.w = np.array(w, dtype=float)
        self.m = len(self.y)

    def f(self, z):
        """
        Computes the negative weighted logistic log-likelihood:
            sum_i w_i [ log(1 + exp(z_i)) - y_i * z_i ]
        """
        z = np.asarray(z[:self.m], dtype=float)
        tail, weighted_tail = _weighted_logistic_tail(z, self.w)
        # Separate the small softplus tail before the label-dependent linear
        # term. logaddexp(0, z) - z loses that tail for large positive z.
        weighted_softplus = self.w * np.log1p(tail)
        subnormal = tail < np.finfo(float).tiny
        weighted_softplus[subnormal] = weighted_tail[subnormal]
        linear = np.where(z >= 0, (1-self.y)*z, -self.y*z)
        return np.sum(weighted_softplus + self.w*linear)

    def d1f(self, z):
        """
        Computes the first derivative (gradient) w.r.t. z:
            w_i [ sigma(z_i) - y_i ]
        """
        tot_z = len(z)
        z = np.asarray(z[:self.m], dtype=float)
        tail, weighted_tail = _weighted_logistic_tail(z, self.w)
        weighted_probability = weighted_tail / (1+tail)
        # These forms also preserve the existing algebra for nonbinary labels.
        grad = np.where(z >= 0, self.w*(1-self.y)-weighted_probability,
                        weighted_probability-self.w*self.y)
        return _pad_derivative(grad, tot_z)

    def d2f(self, z):
        """
        Computes the second derivative (Hessian diagonal) w.r.t. z:
            w_i * sigma(z_i) * (1 - sigma(z_i))
        Evaluate through the small sigmoid tail to avoid 1 - expit(z)
        rounding to zero for moderately large positive predictors.
        """
        tot_z = len(z)
        z = np.asarray(z[:self.m], dtype=float)
        tail, weighted_tail = _weighted_logistic_tail(z, self.w)
        hess_diag = weighted_tail / (1+tail)**2
        return _pad_derivative(hess_diag, tot_z)

    def __call__(self, z):
        return self.f(z)
    
class DummyGLM:
    def __init__(self):
        """
        Empty glm term for default in glqp in QP mode
        """

    def f(self, z):
        return 0.

    def d1f(self, z):
        return np.zeros(1)

    def d2f(self, z):
        return np.zeros(1)

    def __call__(self, z):
        return self.f(z)
    
class GaussianNLL:
    # GLQP may reuse this loss's Hessian inside a solve. Its public d2f and
    # GLQP.get_H still evaluate current weights and design on every call.
    hessian_is_constant = True

    def __init__(self, y, w):
        """
        y: array-like of real-valued responses
        w: array-like of corresponding weights
        """
        self.y = np.array(y, dtype=float)
        self.w = np.array(w, dtype=float)
        self.m = len(self.y)

    def f(self, z):
        """
        Computes the weighted least squares objective:
            0.5 * sum_i w_i * (z_i - y_i)**2
        """
        z = np.asarray(z[:self.m], dtype=float)
        return _gaussian_value(z, self.y, self.w)

    def d1f(self, z):
        """
        Computes the first derivative (gradient) w.r.t. z:
        """
        tot_z = len(z)
        z = np.asarray(z[:self.m], dtype=float)
        grad = _gaussian_gradient(z, self.y, self.w)
        return _pad_derivative(grad, tot_z)

    def d2f(self, z):
        """
        Computes the second derivative (Hessian diagonal) w.r.t. z:
        """
        tot_z = len(z)
        # Return an owned array even on the no-padding fast path.
        return self.w.copy() if tot_z == self.m else _pad_derivative(self.w, tot_z)

    def __call__(self, z):
        return self.f(z)

class PoissonNLL:
    def __init__(self, y, w):
        """
        y: array-like of nonnegative count responses
        w: array-like of corresponding weights
        """
        self.y = np.array(y, dtype=float)
        self.w = np.array(w, dtype=float)
        self.m = len(self.y)

    def _observations(self, z):
        """Zero-weight observations contribute exactly zero, even at large z."""
        z = np.asarray(z[:self.m], dtype=float)
        active = self.w != 0
        return z, None if np.all(active) else active

    def f(self, z):
        """
        Computes the negative weighted Poisson log-likelihood:
            sum_i w_i * (exp(z_i) - y_i * z_i)
        Omits the constant sum_i w_i * log(y_i!).
        """
        z, active = self._observations(z)
        if active is not None:
            z, y, w = z[active], self.y[active], self.w[active]
        else:
            y, w = self.y, self.w
        return _poisson_value(z, y, w, _poisson_rate(z))

    def d1f(self, z):
        """
        Computes the first derivative (gradient) w.r.t. z:
        """
        tot_z = len(z)
        z, active = self._observations(z)
        if active is not None:
            grad = np.zeros(self.m)
            grad[active] = _poisson_gradient(z[active], self.y[active], self.w[active],
                                            _poisson_rate(z[active]))
            return _pad_derivative(grad, tot_z)
        grad = _poisson_gradient(z, self.y, self.w, _poisson_rate(z))
        return _pad_derivative(grad, tot_z)

    def d2f(self, z):
        """
        Computes the second derivative (Hessian diagonal) w.r.t. z:
        """
        tot_z = len(z)
        z, active = self._observations(z)
        if active is not None:
            hessian = np.zeros(self.m)
            hessian[active] = _poisson_curvature(z[active], self.w[active],
                                               _poisson_rate(z[active]))
            return _pad_derivative(hessian, tot_z)
        return _pad_derivative(_poisson_curvature(z, self.w, _poisson_rate(z)), tot_z)

    def __call__(self, z):
        return self.f(z)
