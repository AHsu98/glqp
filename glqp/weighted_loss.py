"""Exceptional-range arithmetic for weighted Gaussian and Poisson losses.

The ordinary kernels live in ``obj``. These helpers keep a binary exponent
separate from each float64 mantissa until weighting and subtraction are done.
They do not clip predictors, change weights, or depend on extended precision.
"""
from fractions import Fraction

import numpy as np


TINY = np.finfo(float).tiny


def product(*values):
    """Represent a product as (mantissa, exponent), without range loss."""
    mantissa, exponent = np.frexp(values[0])
    for value in values[1:]:
        factor, power = np.frexp(value)
        mantissa, exponent = mantissa*factor, exponent+power
    return mantissa, exponent


def multiply(parts, *values):
    mantissa, exponent = product(*values)
    return parts[0]*mantissa, parts[1]+exponent


def subtract(left, right):
    # A zero operand must not force a tiny nonzero operand to exponent zero.
    lm, le = left
    rm, re = right
    exponent = np.maximum(np.where(lm != 0, le, re),
                          np.where(rm != 0, re, le))
    mantissa = np.ldexp(lm, le-exponent)-np.ldexp(rm, re-exponent)
    normalized, adjustment = np.frexp(mantissa)
    return normalized, exponent+adjustment


def rounded(parts):
    return np.ldexp(*parts)


def total(parts):
    """Sum weighted terms, retaining cancellation across overflowed terms.

    Almost all rescued observations round individually to finite numbers. Only
    a nonfinite mixed-sign sum needs exact summation of binary mantissas; this
    is a rare fallback, not an arbitrary-precision likelihood evaluation.
    """
    mantissa, exponent = parts
    values = rounded(parts)
    result = np.sum(values)
    if np.isfinite(result) or not np.all(np.isfinite(mantissa)):
        return float(result)
    if np.all(mantissa >= 0):
        return np.inf
    if np.all(mantissa <= 0):
        return -np.inf
    exact = sum((Fraction(float(m))*Fraction(2)**int(e)
                 for m, e in zip(mantissa, exponent) if m), Fraction())
    try:
        return float(exact)
    except OverflowError:
        return np.inf if exact > 0 else -np.inf


def exponential(z, rate):
    """Represent exp(z), rescuing only nonfinite or subnormal raw rates.

    exp(z/4)**4 avoids subtracting large logarithms and covers every exponent
    that a finite nonzero float64 weight can rescue. Beyond that range the
    weighted rate really is zero or infinite. Safe raw exponentials retain
    their original rounded values.
    """
    mantissa, exponent = np.frexp(rate)
    rescue = np.isfinite(z) & ((rate < TINY) | ~np.isfinite(rate))
    if np.any(rescue):
        quarter, power = np.frexp(np.exp(.25*z[rescue]))
        mantissa[rescue], exponent[rescue] = quarter**4, 4*power
    return mantissa, exponent


def gaussian_value(z, y, w):
    active = w != 0
    z, y, w = z[active], y[active], w[active]
    residual = subtract(product(z), product(y))
    squared = residual[0]**2, 2*residual[1]
    return total(multiply(squared, .5, w))


def gaussian_gradient(z, y, w):
    return rounded(multiply(subtract(product(z), product(y)), w))


def poisson_value(z, y, w, rate):
    difference = subtract(exponential(z, rate), product(y, z))
    return total(multiply(difference, w))


def poisson_gradient(z, y, w, rate):
    difference = subtract(exponential(z, rate), product(y))
    return rounded(multiply(difference, w))


def poisson_curvature(z, w, rate):
    return rounded(multiply(exponential(z, rate), w))
