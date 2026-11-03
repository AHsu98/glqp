"""Policy-independent scheduling and ownership contracts for trial searches."""
import pytest

from glqp.search_control import TrialDecision, search_trials


def test_controller_proposes_only_between_rejected_attempts():
    events = []
    accepted_point = object()

    def attempt(step, index):
        events.append(('attempt', step, index))
        return TrialDecision(accepted_point, 'accepted', accepted=index == 2)

    def propose(step, decision):
        events.append(('proposal', step))
        return .5*step

    result = search_trials(1., 5, attempt, propose)
    assert result.accepted and result.point is accepted_point
    assert events == [('attempt', 1., 0), ('proposal', 1.),
                      ('attempt', .5, 1), ('proposal', .5), ('attempt', .25, 2)]


@pytest.mark.parametrize('stop', [False, True])
def test_rejected_payload_never_escapes_exhausted_or_stopped_search(stop):
    rejected_products = object()

    def attempt(step, index):
        return TrialDecision(rejected_products, 'invalid_step', stop=stop)

    def unexpected(*args):
        pytest.fail('No proposal is needed after the last attempt or a stop')

    result = search_trials(1., 1, attempt, unexpected)
    assert not result.accepted and result.stop and result.point is None
    assert result.reason == ('invalid_step' if stop else 'trial_limit')


def test_zero_budget_performs_no_attempt_or_proposal():
    def unexpected(*args):
        pytest.fail('An exhausted budget cannot evaluate a trial or model')

    result = search_trials(1., 0, unexpected, unexpected)
    assert result.reason == 'trial_limit' and result.point is None
