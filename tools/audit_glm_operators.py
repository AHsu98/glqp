#!/usr/bin/env python3
"""Audit severe GLM operator diagnostics and fallback, with no timing claims.

Runs each severe case with diagnostics off/on and checks exact solution and
common non-time history parity. Saves full diagnostics, final arrays and
independent original-model validation. Separate bounded logistic controls vary
only the explicitly enabled Newton fallback policy. Sources must stay frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import sys
import warnings

from diagnose_scalar_search import source_hashes, write_json


def make_model(problem):
    import glqp
    from glqp.obj import GaussianNLL, LogisticNLL, PoissonNLL
    from scipy import sparse
    loss = {'gaussian': GaussianNLL, 'logistic': LogisticNLL,
            'poisson': PoissonNLL}[problem.loss]
    return glqp.GLQP(f=loss(problem.response, problem.weights), A=problem.X,
        Q=sparse.diags(problem.ridge, format='csc'), b=-problem.q,
        C=problem.C if len(problem.c) else None, c=problem.c if len(problem.c) else None,
        E=problem.E if len(problem.e) else None, e=problem.e if len(problem.e) else None)


def result_arrays(result):
    return {key: getattr(result, key) for key in ('x', 'y', 's', 'nu')}


def common_history_equal(first, second):
    columns = [name for name in first.columns if 'time' not in name.lower()
               and name in second.columns]
    return first[columns].equals(second[columns])


def summarize_diagnostics(diagnostics):
    points = [diagnostics['initial']] + [row['iterate'] for row in diagnostics['iterations']
                                        if row.get('iterate') is not None]
    predictors = [point['glm_predictor'] for point in points]
    hessians = [row['glm_hessian'] for row in diagnostics['iterations'] if 'glm_hessian' in row]
    return dict(final_predictor=predictors[-1],
        maximum_predictor_drift_inf=max(row['carried_minus_fresh_inf'] for row in predictors),
        maximum_stationarity_drift_inf=max(row['stationarity_drift_inf'] for row in predictors),
        maximum_hessian_action_difference_inf=max(
            (row['assembled_minus_original_inf'] for row in hessians), default=0.),
        maximum_relative_hessian_action_difference=max(
            (row['relative_to_original_inf'] for row in hessians), default=0.),
        hessian_actions=hessians)


def run_case(problem, args):
    import numpy as np
    import glqp
    from glqp_bench.adapters import glqp_settings
    from glqp_bench.glm import glm_hash, validate_glm
    output = args.output / problem.name
    output.mkdir()
    results = []
    for enabled in (False, True):
        settings = glqp_settings(glqp, tol=args.tol, time_limit=600., settings_profile='glm-v2')
        settings.iteration_diagnostics = enabled
        with warnings.catch_warnings(record=True) as caught:
            _, result = make_model(problem).solve(verbose=False, settings=settings)
        mode = 'observed' if enabled else 'plain'
        np.savez_compressed(output / f'{mode}-solution.npz', **result_arrays(result))
        result.history.to_csv(output / f'{mode}-history.csv', index=False)
        record = dict(termination=result.termination_tag, iterations=len(result.history),
            settings=asdict(settings), warnings=[str(item.message) for item in caught],
            validation=validate_glm(problem, result.x, result.nu[:len(problem.e)], result.y[:len(problem.c)]))
        write_json(output / f'{mode}-result.json', record)
        if enabled:
            write_json(output / 'diagnostics.json.gz', result.diagnostics)
        results.append(result)
    plain, observed = results
    arrays_equal = all(np.array_equal(result_arrays(plain)[key], result_arrays(observed)[key])
                       for key in result_arrays(plain))
    history_equal = common_history_equal(plain.history, observed.history)
    parity = arrays_equal and history_equal and plain.termination_tag == observed.termination_tag
    record = dict(problem_id=problem.name, data_sha256=glm_hash(problem),
        termination=observed.termination_tag, iterations=len(observed.history),
        validation=validate_glm(problem, observed.x, observed.nu[:len(problem.e)], observed.y[:len(problem.c)]),
        exact_diagnostic_parity=parity, exact_arrays=arrays_equal, exact_common_history=history_equal,
        **summarize_diagnostics(observed.diagnostics))
    write_json(output / 'summary.json', record)
    if not parity:
        raise RuntimeError(f'Diagnostics changed trajectory for {problem.name}')
    return record


def gram_controls():
    import numpy as np
    from scipy import sparse
    from glqp import GLQP
    from glqp.glm_diagnostics import glm_hessian_diagnostics
    from glqp.obj import GaussianNLL
    records = []
    for scale, ridge in ((1., 0.), (1e8, 1e-7)):
        A = sparse.csc_array([[scale, scale], [1e-8*scale, 0.]])
        model = GLQP(A=A, Q=sparse.eye_array(2, format='csc')*ridge,
                     f=GaussianNLL(np.zeros(2), np.ones(2)))
        z, dx = np.zeros(2), np.array([1., -1.])
        records.append(dict(design=A.toarray(), ridge=ridge, direction=dx,
            diagnostic=glm_hessian_diagnostics(model, model.get_H(z), z, dx)))
    return records


def fallback_controls(output):
    import numpy as np
    import glqp
    from scipy import sparse
    from scipy.optimize import brentq
    from scipy.special import expit
    from glqp.obj import LogisticNLL
    from glqp_bench.adapters import glqp_settings
    reference = brentq(lambda x: expit(x) - .1 + 1e-4*x, -3., 0.)
    records = []
    for policy in ('off', 'finite', 'progress'):
        model = glqp.GLQP(f=LogisticNLL([0., 1.], [.9, .1]),
            A=sparse.csc_array([[1.], [1.]]), Q=sparse.csc_array([[1e-4]]))
        settings = glqp_settings(glqp, tol=1e-7, time_limit=600., settings_profile='glm-v2')
        settings.let_newton_cook = .5
        settings.newton_fallback_policy = policy
        settings.max_iter, settings.max_stagnation = 100, 20
        settings.iteration_diagnostics = True
        with warnings.catch_warnings(record=True) as caught:
            _, result = model.solve(x0=np.array([5.]), verbose=False, settings=settings)
        np.savez_compressed(output / f'fallback-{policy}-solution.npz', **result_arrays(result))
        write_json(output / f'fallback-{policy}-diagnostics.json.gz', result.diagnostics)
        result.history.to_csv(output / f'fallback-{policy}-history.csv', index=False)
        searches = [row['line_search'] for row in result.diagnostics['iterations']
                    if row.get('line_search') is not None]
        records.append(dict(policy=policy, settings=asdict(settings), termination=result.termination_tag,
            iterations=len(result.history), x=float(result.x[0]), reference_x=reference,
            original_gradient=float(expit(result.x[0]) - .1 + 1e-4*result.x[0]),
            fallback_acceptances=sum(row['acceptance_reason'] == 'newton_fallback' for row in searches),
            algorithms=sorted(set(row.get('algorithm', 'unspecified') for row in searches)),
            warnings=[str(item.message) for item in caught]))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--benchmark-repo', type=Path, default=Path(__file__).resolve().parents[2] / 'glqp-bench')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=6)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0):
        parser.error('Requested CPU is outside affinity')
    os.sched_setaffinity(0, {args.cpu})
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['MKL_DYNAMIC'] = 'FALSE'
    args.source, args.benchmark_repo, args.output = [path.resolve() for path in
                                                  (args.source, args.benchmark_repo, args.output)]
    sys.path.insert(0, str(args.source))
    sys.path.insert(0, str(args.benchmark_repo / 'src'))
    import glqp
    from glqp_bench.glm import build_glm
    from glqp_bench.glm_stress import stress_specs
    if not Path(glqp.__file__).resolve().is_relative_to(args.source):
        raise RuntimeError(f'Wrong source imported: {glqp.__file__}')
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes(args.source)
    cases = args.cases or [item['problem_id'] for item in stress_specs('hard-v1')]
    records = []
    for name in cases:
        row = run_case(build_glm(name), args)
        records.append(row)
        print(f"{name}: {row['termination']}, {row['iterations']} iterations, exact parity={row['exact_diagnostic_parity']}", flush=True)
    fallback = fallback_controls(args.output)
    unchanged = hashes == source_hashes(args.source)
    write_json(args.output / 'summary.json', dict(
        purpose='Untimed numerical diagnostics; no comparative performance claims',
        source=str(args.source), source_sha256=hashes, source_unchanged=unchanged,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        affinity=sorted(os.sched_getaffinity(0)), python=sys.version,
        results=records, gram_controls=gram_controls(), fallback_controls=fallback))
    if not unchanged:
        raise RuntimeError('Source changed during audit; results retained but not a frozen-source comparison')


if __name__ == '__main__':
    main()
