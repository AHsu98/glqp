import numpy as np
from scipy.special import expit
from numpy import logaddexp

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
        z = np.array(z[:self.m], dtype=float)
        return np.sum(self.w * (logaddexp(0, z[:self.m]) - self.y * z[:self.m]))

    def d1f(self, z):
        """
        Computes the first derivative (gradient) w.r.t. z:
            w_i [ sigma(z_i) - y_i ]
        """
        tot_z = len(z)
        z = np.array(z[:self.m], dtype=float)
        sig_z = expit(z)
        grad = self.w * (sig_z - self.y)
        return np.hstack([grad,np.zeros(tot_z-self.m)])

    def d2f(self, z):
        """
        Computes the second derivative (Hessian diagonal) w.r.t. z:
            w_i * sigma(z_i) * (1 - sigma(z_i))
        Using expit(z) for numerical stability.
        """
        tot_z = len(z)
        z = np.array(z[:self.m], dtype=float)
        sig_z = expit(z)
        hess_diag = self.w * sig_z * (1.0 - sig_z)
        return np.hstack([hess_diag,np.zeros(tot_z - self.m)])

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
        Computes the negative weighted least squares objective:
        """
        z = np.array(z[:self.m], dtype=float)
        return 0.5 * np.dot(self.w,(self.y - z) ** 2)

    def d1f(self, z):
        """
        Computes the first derivative (gradient) w.r.t. z:
        """
        tot_z = len(z)
        z = np.array(z[:self.m], dtype=float)
        grad = self.w*(z-self.y)
        return np.hstack([grad,np.zeros(tot_z-self.m)])#zero pad

    def d2f(self, z):
        """
        Computes the second derivative (Hessian diagonal) w.r.t. z:
        """
        tot_z = len(z)
        return np.hstack([self.w*np.ones(tot_z),np.zeros(tot_z - self.m)])

    def __call__(self, z):
        return self.f(z)

class PoissonNLL:
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
        Computes the negative weighted least squares objective:
        """
        z = np.array(z[:self.m], dtype=float)
        u = np.exp(z)
        return 0.5 * np.dot(self.w,u - self.y*z)

    def d1f(self, z):
        """
        Computes the first derivative (gradient) w.r.t. z:
        """
        tot_z = len(z)
        z = np.array(z[:self.m], dtype=float)
        u = np.exp(z)
        grad = self.w*(u - self.y)
        return np.hstack([grad,np.zeros(tot_z-self.m)])#zero pad

    def d2f(self, z):
        """
        Computes the second derivative (Hessian diagonal) w.r.t. z:
        """
        tot_z = len(z)
        u = np.exp(z)
        return np.hstack([self.w*u,np.zeros(tot_z - self.m)])

    def __call__(self, z):
        return self.f(z)
