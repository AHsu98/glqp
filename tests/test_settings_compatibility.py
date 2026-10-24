"""Experimental keyword-only settings preserve existing positional arguments."""
from dataclasses import asdict
import inspect

import pytest

from glqp import SolverSettings


# Captured from the measured candidate 32ffecb38abdea32b5931cfb3a7262ec2390d530.
# Removing only its new policy field recovers the positional argument order
# from the preceding accepted version. Keep the full reference so a signature
# repair cannot silently change any benchmarked numerical default.
MEASURED_DEFAULTS = {
    'max_precenter': 100,
    'max_iter': 200,
    'tol': 1e-8,
    'boundary_frac': .99,
    'gamma': .5,
    'min_mu': 1e-12,
    'min_mu_policy': 'constraint_count',
    'tau_reg': 2e-8,
    'max_linesearch_steps': 50,
    'glm_linesearch': 'scalar',
    'qp_linesearch': 'backtracking',
    'qp_residual_screen': 'off',
    'max_iterative_refinement': 10,
    'max_time': 600.,
    'max_stagnation': 20,
    'armijo_additive_eps': 1e-8,
    'armijo_param': .005,
    'let_newton_cook': 0.,
    'prox_reg': 2e-8,
    'proximal_policy': 'legacy',
    'prox_dual_reg': None,
    'proximal_center_reduction': .95,
    'proximal_inner_tolerance': .1,
    'proximal_corrector_limit': .5,
    'proximal_reg_update': 'residual',
    'linear_solver_backend': 'qdldl',
    'linear_solver_options': None,
    'barrier_update_guard': 'legacy',
    'barrier_reduction_limit': .1,
    'barrier_kkt_floor': .01,
    'initial_mu_policy': 'fixed',
    'initial_mu_scale': 1.,
    'slack_repair': 'legacy',
    'repair_growth_limit': 1.1,
    'newton_fallback_policy': 'finite',
    'step_strategy': 'newton',
    'iteration_diagnostics': False,
}
OLD_POSITIONAL_NAMES = [name for name in MEASURED_DEFAULTS if name != 'min_mu_policy']
NEW_KEYWORD_DEFAULTS = dict(prox_residual_factor=1e-3,
                            linear_atol_factor=.05, linear_rtol=1e-6)


def test_floor_policy_is_keyword_only_and_old_positional_order_is_preserved():
    parameters = inspect.signature(SolverSettings).parameters
    positional = [name for name, parameter in parameters.items()
                  if parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional == OLD_POSITIONAL_NAMES
    for name in ('min_mu_policy', *NEW_KEYWORD_DEFAULTS):
        assert parameters[name].kind == inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize('policy', ['fixed', 'constraint_count'])
def test_existing_full_positional_call_preserves_values_after_minimum(policy):
    expected = dict(MEASURED_DEFAULTS, **NEW_KEYWORD_DEFAULTS, min_mu_policy=policy,
                    tau_reg=7e-9, max_linesearch_steps=13, max_time=123.,
                    barrier_kkt_floor=.03, iteration_diagnostics=True)
    settings = SolverSettings(*(expected[name] for name in OLD_POSITIONAL_NAMES),
                              min_mu_policy=policy)
    assert asdict(settings) == expected


def test_signature_fix_keeps_all_measured_candidate_defaults():
    actual = asdict(SolverSettings())
    assert actual == dict(MEASURED_DEFAULTS, **NEW_KEYWORD_DEFAULTS)
    assert all(type(actual[name]) is type(value) for name, value in MEASURED_DEFAULTS.items())
