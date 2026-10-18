"""Trial scheduling shared by objective-plus-barrier acceptance policies.

The controller knows neither the merit nor the acceptance inequalities. An
attempt owns its validity checks and observations; a proposer sees only rejected
points and is never called after the last available attempt. Centered proximal
search uses a different residual merit and retains its own controller.
"""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrialDecision:
    point: Any = None
    reason: str = ''
    accepted: bool = False
    stop: bool = False


def search_trials(step, max_trials, attempt, propose):
    """Try the boundary step first, then propose only after rejection.

    ``attempt(step, index)`` returns a :class:`TrialDecision`. Its point is
    passed to ``propose(step, decision)`` only if another trial is available.
    Only an accepted decision can carry a point out of the search. This keeps
    products belonging to rejected trials out of accepted-iterate reuse.
    """
    for index in range(max_trials):
        decision = attempt(step, index)
        if decision.accepted:
            return decision
        if decision.stop:
            return TrialDecision(reason=decision.reason, stop=True)
        if index + 1 < max_trials:
            step = propose(step, decision)
    return TrialDecision(reason='trial_limit', stop=True)
