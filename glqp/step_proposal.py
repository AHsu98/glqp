"""Safeguarded proposals for a rejected scalar merit trial.

The model selects a smaller step; the search's acceptance policy still decides
whether that step is usable. No objective, derivative, or matrix is evaluated
here. All inputs describe one fixed direction and barrier target.
"""

import numpy as np


def safeguarded_step(step, merit, merit0, slope0, slope, curvature, *,
                     require_positive_step=False):
    """Return ``(next_step, kind)`` using Newton, interpolation, or halving.

    A finite first derivative and positive finite curvature are required for
    either model proposal. A Newton proposal inside [0.1*step, 0.5*step] wins;
    otherwise fit the rejected merit using the base value and slope, and clip
    its minimizer to that interval. An unusable model simply halves the step.

    The caller checks model eligibility and supplies a finite negative base
    slope. ``require_positive_step`` preserves the QP proposal's underflow
    handling: if a model proposal rounds to zero, use halving instead. The GLM
    policy permits that zero proposal and lets its trial loop report underflow.
    Invalid input steps return zero for either policy.
    """
    if not np.isfinite(step) or step <= 0:
        return 0., 'contraction'
    step = float(step)
    if not np.isfinite(slope) or not np.isfinite(curvature) or curvature <= 0:
        return .5*step, 'contraction'

    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        candidate = step-np.divide(slope, curvature)
        if ((not require_positive_step or candidate > 0)
                and .1*step <= candidate <= .5*step):
            return float(candidate), 'newton'

        denominator = (merit-merit0)/step-slope0
        if (not require_positive_step
                or (np.isfinite(denominator) and denominator > 0)):
            fraction = np.divide(-.5*slope0, denominator)
            if np.isfinite(fraction) and fraction > 0:
                candidate = step*float(np.clip(fraction, .1, .5))
                if not require_positive_step or candidate > 0:
                    return float(candidate), 'interpolation'
    return .5*step, 'contraction'
