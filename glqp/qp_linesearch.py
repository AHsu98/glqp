"""Scalar step proposals after all existing QP acceptance tests reject.

This helper chooses a smaller trial only. It neither accepts a point nor changes
the solver's Armijo, original-KKT restoration, or configured fallback predicates.
The caller supplies the actual post-clipping direction and coefficients from Q,
not from the physically regularized Newton matrix. No matrices are stored here.
"""
import numpy as np

from .step_proposal import safeguarded_step


class QPScalarProposer:
    """Quadratic-plus-barrier ray model with a fixed direction and target.

    Arrays are borrowed and must remain unchanged during this search. ``trial_s``
    supplied to :meth:`propose` is the caller's actual ``s + step*ds``. The base
    slope includes the barrier term; xQdx/bdx are the separate quadratic/linear
    coefficients. Creating the proposer does not evaluate ray derivatives.

    Ordinary proposals lie in [0.1*step, 0.5*step]. Invalid models use halving;
    invalid/nonpositive input steps return zero so the caller can terminate.
    The caller must still run every existing acceptance test at the new point.
    """

    def __init__(self, s, ds, mu, xQdx, dxQdx, bdx, merit0, slope0):
        self.s = np.asarray(s, dtype=float)
        self.ds = np.asarray(ds, dtype=float)
        if self.s.ndim != 1 or self.s.shape != self.ds.shape:
            raise ValueError('QP ray slack vectors must have matching one-dimensional shapes')
        if not np.isfinite(mu) or mu < 0:
            raise ValueError('The fixed barrier target must be finite and nonnegative')
        self.mu = float(mu)
        self.xQdx, self.dxQdx, self.bdx = float(xQdx), float(dxQdx), float(bdx)
        self.merit0, self.slope0 = float(merit0), float(slope0)
        self.counters = dict(derivative_evaluations=0, newton_steps=0,
                             interpolation_steps=0, contraction_steps=0)

    def _result(self, step, kind):
        self.counters[kind+'_steps'] += 1
        return float(step), kind

    def propose(self, step, trial_s, merit):
        """Return ``(next_step, kind)`` after the caller rejects this trial.

        For phi(t)=constant+t*(xQdx-bdx)+.5*t^2*dxQdx
        -mu*sum(log(s+t*ds)), use its exact scalar first/second derivatives.
        A valid Newton proposal is preferred; otherwise fit the rejected value
        to a quadratic using the base value and slope. This is sufficient-
        decrease step selection, not an exact ray minimizer or an acceptance.
        """
        if not np.isfinite(step) or step <= 0:
            return self._result(0., 'contraction')
        step = float(step)
        trial_s = np.asarray(trial_s, dtype=float)
        if trial_s.shape != self.s.shape:
            raise ValueError('QP trial slack must match the base slack shape')
        if (not np.isfinite(merit) or not np.isfinite(self.merit0)
                or not np.isfinite(self.slope0) or self.slope0 >= 0
                or not np.isfinite(self.xQdx) or not np.isfinite(self.bdx)
                or not np.isfinite(self.dxQdx) or self.dxQdx < 0
                or not np.all(np.isfinite(trial_s)) or np.any(trial_s <= 0)):
            return self._result(.5*step, 'contraction')

        self.counters['derivative_evaluations'] += 1
        with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
            barrier_slope = barrier_curvature = 0.
            if self.mu:
                ratio = self.ds/trial_s
                barrier_slope = -np.sum(self.mu*ratio)
                # Weight before squaring: ratio**2 can overflow even when the
                # weighted barrier curvature is representable.
                scaled = np.sqrt(self.mu)*ratio
                barrier_curvature = np.dot(scaled, scaled)
            slope = self.xQdx+step*self.dxQdx-self.bdx+barrier_slope
            curvature = self.dxQdx+barrier_curvature
        next_step, kind = safeguarded_step(
            step, merit, self.merit0, self.slope0, slope, curvature,
            require_positive_step=True)
        return self._result(next_step, kind)
