"""Separable GLM losses evaluated on the first ``len(y)`` predictors.

Predictor vectors may include trailing coordinates unused by the likelihood.
Gradients and Hessian diagonals have the input length, with zeros in those
coordinates. Likelihood constants independent of the predictors are omitted.
"""

import numpy as np


def _pad_derivative(values, size):
    """Return a full predictor derivative without copying an exact-size result."""
    if len(values) == size:
        return values
    result = np.zeros(size)
    result[:len(values)] = values
    return result


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
        return 0.5 * np.dot(self.w,(self.y - z) ** 2)

    def d1f(self, z):
        """
        Computes the first derivative (gradient) w.r.t. z:
        """
        tot_z = len(z)
        z = np.asarray(z[:self.m], dtype=float)
        grad = self.w*(z-self.y)
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
            return np.dot(self.w[active], np.exp(z[active])-self.y[active]*z[active])
        u = np.exp(z)
        return np.dot(self.w,u - self.y*z)

    def d1f(self, z):
        """
        Computes the first derivative (gradient) w.r.t. z:
        """
        tot_z = len(z)
        z, active = self._observations(z)
        if active is not None:
            grad = np.zeros(self.m)
            grad[active] = self.w[active]*(np.exp(z[active])-self.y[active])
            return _pad_derivative(grad, tot_z)
        u = np.exp(z)
        grad = self.w*(u - self.y)
        return _pad_derivative(grad, tot_z)

    def d2f(self, z):
        """
        Computes the second derivative (Hessian diagonal) w.r.t. z:
        """
        tot_z = len(z)
        z, active = self._observations(z)
        if active is not None:
            hessian = np.zeros(self.m)
            hessian[active] = self.w[active]*np.exp(z[active])
            return _pad_derivative(hessian, tot_z)
        u = np.exp(z)
        return _pad_derivative(self.w*u, tot_z)

    def __call__(self, z):
        return self.f(z)
