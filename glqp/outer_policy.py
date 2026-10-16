"""Opt-in outer-policy experiments, separate from the accepted legacy path."""
from dataclasses import dataclass
import numbers

import numpy as np


def validate_repair_and_fallback_settings(settings):
    if (not isinstance(settings.slack_repair, str)
            or settings.slack_repair not in ('legacy', 'off', 'guarded')):
        raise ValueError("slack_repair must be 'legacy', 'off', or 'guarded'")
    if (not isinstance(settings.newton_fallback_policy, str)
            or settings.newton_fallback_policy not in ('finite', 'progress', 'off')):
        raise ValueError("newton_fallback_policy must be 'finite', 'progress', or 'off'")
    limit = settings.repair_growth_limit
    if (isinstance(limit, (bool, np.bool_)) or not isinstance(limit, numbers.Real)
            or not np.isfinite(limit) or limit < 1):
        raise ValueError('repair_growth_limit must be finite and at least 1')


def _inf(value):
    return float(np.max(np.abs(value))) if len(value) else 0.


@dataclass
class SlackRepairProposal:
    """Candidate storage is separate; the caller applies it only if accepted."""
    s: np.ndarray
    rp: np.ndarray
    rc: np.ndarray
    kkt_residual: float
    accepted: bool
    metrics: dict


def guarded_slack_repair(C, c, x, y, s, rx, rp, req, old_kkt, mu_bar,
                         indices, threshold, growth_limit, tol):
    """Bound primal and unperturbed KKT growth before applying slack repair."""
    candidate_s = s.copy()
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        candidate_s[indices] = threshold*mu_bar/y[indices]
        candidate_rp = C@x + candidate_s - c
        candidate_rc = y*candidate_s
        primal_limit = max(growth_limit*_inf(rp), tol)
        kkt_limit = max(growth_limit*old_kkt, tol)
    candidate_primal = _inf(candidate_rp)
    candidate_kkt = max(_inf(rx), candidate_primal, _inf(candidate_rc), _inf(req))
    finite_positive = bool(
        np.all(np.isfinite(candidate_s)) and np.all(candidate_s > 0)
        and np.all(np.isfinite(candidate_rp)) and np.all(np.isfinite(candidate_rc))
        and np.all(np.isfinite(rx)) and np.all(np.isfinite(req)))
    reasons = []
    if not finite_positive:
        reasons.append('nonfinite_or_nonpositive_candidate')
    if not candidate_primal <= primal_limit:
        reasons.append('primal_growth')
    if not candidate_kkt <= kkt_limit:
        reasons.append('kkt_growth')
    accepted = not reasons
    metrics = dict(proposed=True, proposed_count=int(np.count_nonzero(indices)),
                   applied=accepted, rejected=not accepted,
                   rejection_reason=reasons[0] if reasons else None,
                   rejection_reasons=tuple(reasons), candidate_finite_positive=finite_positive,
                   candidate_primal_inf=candidate_primal, candidate_kkt_inf=candidate_kkt,
                   primal_limit=float(primal_limit), kkt_limit=float(kkt_limit))
    return SlackRepairProposal(candidate_s, candidate_rp, candidate_rc,
                               candidate_kkt, accepted, metrics)
