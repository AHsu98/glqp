#!/usr/bin/env python3
"""Observe QP search work with plain/diagnostic byte parity, never timing claims.

Uses the canonical MM loader and benchmark-v1 (tol1e-7, cap40, fallback .5).
Every variant runs on fresh plain and observed models. Sparse vector products
are counted separately for residual trials, ordinary solver work and diagnostic
observers. Full search/proposal diagnostics and non-time histories are retained.
Only the requested search/cache flags differ across variants. Source mutation
aborts an attempt, which must be retained and retried in a new output directory.

Activate the numerical environment before running. This tool shares small
serialization/parity helpers with diagnose_scalar_search.py in this directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import warnings

from diagnose_scalar_search import (THREAD_VARIABLES, WorkObserver, compare_results,
                                    finite_json, history_record, source_hashes, write_json)


DEFAULT_CASES = ('QPCBOEI2', 'YAO', 'LISWET1', 'QAFIRO', 'PRIMAL4', 'QSHELL')
VARIANTS = {'backtracking': ('backtracking', 'off'), 'scalar': ('scalar', 'off'),
    'backtracking-complementarity': ('backtracking', 'complementarity'),
    'backtracking-bounded': ('backtracking', 'bounded'),
    'scalar-complementarity': ('scalar', 'complementarity'),
    'scalar-bounded': ('scalar', 'bounded')}


class QPWorkObserver(WorkObserver):
    def __init__(self, model, source):
        super().__init__(model, source)
        self.residual_products = []
        self.first_direction = None
        self.solver_filename = str(source / 'glqp/solver.py')
        self.diagnostics_filename = str(source / 'glqp/diagnostics.py')

    def attribution(self):
        if self.phase != 'solver':
            return self.phase
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_filename == self.diagnostics_filename:
                return 'diagnostic_observer'
            frame = frame.f_back
        return 'solver'

    def products_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(model, x, y, nu):
            frame = sys._getframe(1)
            while frame is not None and frame.f_code.co_name != 'solve':
                frame = frame.f_back
            local = {} if frame is None else frame.f_locals
            is_trial = local.get('trial_x') is x and local.get('x') is not x
            previous = self.phase
            if is_trial:
                self.phase = 'qp_trial_residual'
                self.residual_products.append(dict(iteration=local.get('iteration_number'),
                    step=local.get('t'), fallback_eligible=local.get('newton_fallback')))
            try:
                return original(model, x, y, nu)
            finally:
                self.phase = previous
        return wrapped

    def profile(self, frame, event, value):
        # Profiling sees every Python/C event: reject almost all of them before
        # constructing paths or inspecting frame locals.
        if frame.f_code.co_name != 'merit_line' or event not in ('call', 'return'):
            return
        if frame.f_code.co_filename != self.solver_filename:
            return
        super().profile(frame, event, value)
        if (self.first_direction is None and event == 'return'
                and frame.f_code.co_filename == self.solver_filename):
            local = frame.f_back.f_locals
            self.first_direction = {key: local[key].copy()
                for key in ('x', 's', 'y', 'nu', 'dx', 'ds', 'dy', 'dnu')}

    def install(self, stack):
        from scipy.sparse._base import _spbase
        from glqp.util import SparseTransposeCache
        stack.enter_context(patch.object(_spbase, '__matmul__', self.matmul_wrapper(_spbase.__matmul__)))
        stack.enter_context(patch.object(SparseTransposeCache, 'get',
            self.transpose_wrapper(SparseTransposeCache.get)))
        stack.enter_context(patch.object(type(self.model), '_constraint_products',
            self.products_wrapper(type(self.model)._constraint_products)))
        previous = sys.getprofile()
        stack.callback(sys.setprofile, previous)
        sys.setprofile(self.profile)


def summarize_searches(diagnostics, observer):
    searches = [item['line_search'] for item in diagnostics['iterations']
                if item.get('line_search') is not None]
    acceptance = Counter(item['acceptance_reason'] for item in searches)
    rejected = Counter()
    proposals = Counter()
    for search in searches:
        rejected.update(search['rejection_reasons'])
        proposals.update(search.get('qp_scalar_counters') or {})
    directions = [item['direction'] for item in diagnostics['iterations']
                  if item.get('direction') is not None]
    return dict(counts=dict(sorted(observer.counts.items())), search_count=len(searches),
        trial_count=sum(item['trial_count'] for item in searches),
        residual_check_count=sum(item['residual_check_count'] for item in searches),
        residual_product_evaluations=len(observer.residual_products),
        residual_product_evaluations_with_fallback_eligible=sum(
            item['fallback_eligible'] is True for item in observer.residual_products),
        armijo_failure_count=sum(item['armijo_failure_count'] for item in searches),
        acceptance_reasons={str(key): value for key, value in acceptance.items()},
        rejection_reasons=dict(rejected), proposal_counts=dict(proposals),
        scalar_requested_searches=sum(bool(item.get('qp_scalar_requested')) for item in searches),
        scalar_eligible_searches=sum(bool(item.get('qp_scalar_eligible')) for item in searches),
        fallback_eligible_count=sum(item.get('fallback_eligible_count', 0) for item in searches),
        proposals_crossing_fallback=sum(bool(proposal['crosses_fallback']) for item in searches
            for proposal in (item.get('qp_scalar_proposals') or [])),
        direction_edits=sum(item.get('edited_ds_count', 0)+item.get('edited_dy_count', 0)
                            for item in directions),
        first_search=None if not searches else searches[0],
        note='Trial-product phase is recognized from actual trial-array identity before acceptance. '
             'Diagnostic observer products are separate; other_sparse includes linear refinement. '
             'See full diagnostics for proposal records and all residual/cache decisions.')


def solve_case(problem, variant, observed, args, output):
    import numpy as np
    import glqp
    from glqp_bench.adapters import glqp_settings
    from glqp_bench.qp import validate_solution

    settings = glqp_settings(glqp, tol=1e-7, time_limit=600., settings_profile='benchmark-v1')
    mode, screen = VARIANTS[variant]
    if not hasattr(settings, 'qp_linesearch'):
        raise ValueError('Solver does not expose qp_linesearch yet')
    if screen != 'off' and not hasattr(settings, 'qp_residual_screen'):
        raise ValueError('Solver does not expose qp_residual_screen yet')
    settings.qp_linesearch = mode
    if hasattr(settings, 'qp_residual_screen'):
        settings.qp_residual_screen = screen
    settings.iteration_diagnostics = observed
    model = glqp.GLQP(Q=problem.P, b=-problem.q, C=problem.G, c=problem.h,
                      E=problem.A, e=problem.b)
    observer = QPWorkObserver(model, args.source) if observed else None
    with warnings.catch_warnings(record=True) as caught, ExitStack() as stack:
        warnings.simplefilter('always')
        if observer is not None:
            observer.install(stack)
        _, result = model.solve(verbose=False, settings=settings)
    arrays = {key: np.asarray(getattr(result, key)) for key in ('x', 'y', 's', 'nu')}
    np.savez_compressed(output / 'solution.npz', **arrays)
    record = dict(settings=asdict(settings), native=result.termination_tag,
        iterations=len(result.history), exception=None if result.exception is None else repr(result.exception),
        primal=result.primal, dual_res=result.dual_res, cons_viol=result.cons_viol, feasible=result.feasible,
        validation=validate_solution(problem, result.x, result.nu[:problem.p], result.y[:problem.m]),
        arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype),
            sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key, value in arrays.items()},
        initialization_linear_solve=result.initialization_linear_solve,
        last_linear_solve=result.last_linear_solve,
        warnings=dict(Counter(str(item.message) for item in caught)))
    write_json(output / 'result.json', record)
    write_json(output / 'history.json.gz', history_record(result.history))
    if observer is not None:
        record['work'] = summarize_searches(result.diagnostics, observer)
        write_json(output / 'work.json', record['work'])
        write_json(output / 'diagnostics.json.gz', result.diagnostics)
        write_json(output / 'trials.json.gz', dict(
            merit_trials=observer.backtracking_trials,
            residual_product_evaluations=observer.residual_products,
            note='Merit traces omit pre-evaluation invalid trials, which remain in native diagnostics.'))
        if observer.first_direction is not None:
            np.savez_compressed(output / 'first-direction.npz', **observer.first_direction)
    return result, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=6)
    parser.add_argument('--cases', nargs='+', default=DEFAULT_CASES)
    parser.add_argument('--variants', nargs='+', choices=tuple(VARIANTS), default=('backtracking', 'scalar'))
    args = parser.parse_args()
    for name in ('source', 'benchmark_repo', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {args.cpu})
    for variable in THREAD_VARIABLES:
        os.environ[variable] = '1'
    os.environ['MKL_DYNAMIC'] = 'FALSE'
    sys.path[:0] = [str(args.source), str(args.benchmark_repo / 'src')]
    baseline = source_hashes(args.source)
    import numpy as np
    import scipy
    import glqp
    from threadpoolctl import threadpool_info
    from glqp_bench.qp import load_problem
    if Path(glqp.__file__).resolve().parent != args.source / 'glqp':
        raise RuntimeError(f'Wrong solver import: {glqp.__file__}')
    inputs = [args.benchmark_repo / 'src/glqp_bench/qp.py',
              args.benchmark_repo / 'src/glqp_bench/adapters.py']
    inputs += [args.benchmark_repo / 'mm_problems/MAT_Files' / (name+'.mat') for name in args.cases]
    auxiliary = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    summary = dict(purpose='Untimed QP search work and plain/observed correctness parity',
        source=str(args.source), source_sha256=baseline, source_unchanged=True,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).with_name('diagnose_scalar_search.py').read_bytes()).hexdigest(),
        auxiliary_sha256=auxiliary,
        environment=dict(cpu=args.cpu, affinity=sorted(os.sched_getaffinity(0)), native_threads=1,
            python=sys.version, numpy=np.__version__, scipy=scipy.__version__, threadpools=threadpool_info()),
        settings_contract='benchmark-v1, tol1e-7, maxiter10000, stagnation5000, cap40, let_newton_cook=.5',
        cases=[])
    write_json(args.output / 'summary.json', summary)

    def check_source():
        same = source_hashes(args.source) == baseline
        same = same and all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
                            for name, digest in auxiliary.items())
        if not same:
            summary['source_unchanged'] = False
            write_json(args.output / 'summary.json', summary)
            raise RuntimeError('Sources or inputs changed; retain this attempt and use a fresh output directory')

    for name in args.cases:
        path = args.benchmark_repo / 'mm_problems/MAT_Files' / (name+'.mat')
        problem = load_problem(path)
        case_output = args.output / name
        case_output.mkdir()
        case = dict(problem_id=name, data_sha256=auxiliary[str(path)], variants={})
        for variant in args.variants:
            results, records = {}, {}
            for label in ('plain', 'observed'):
                destination = case_output / variant / label
                destination.mkdir(parents=True)
                check_source()
                results[label], records[label] = solve_case(problem, variant, label == 'observed', args, destination)
                check_source()
            parity = compare_results(results['plain'], results['observed'])
            # A timeout is not useful parity evidence even if both endpoints
            # happen to match: observation overhead consumes the native budget.
            if 'time' in records['plain']['native'] or 'time' in records['observed']['native']:
                parity.update(conclusive=False, passed=False)
            write_json(case_output / variant / 'parity.json', parity)
            case['variants'][variant] = dict(native=records['plain']['native'],
                iterations=records['plain']['iterations'], validation=records['plain']['validation'],
                parity=parity, work=records['observed']['work'])
            print(json.dumps(finite_json(dict(problem=name, variant=variant,
                native=records['plain']['native'], iterations=records['plain']['iterations'],
                accepted=records['plain']['validation']['accepted'], parity=parity['passed'],
                trials=records['observed']['work']['trial_count'],
                residual_checks=records['observed']['work']['residual_check_count']))), flush=True)
        summary['cases'].append(case)
        write_json(args.output / 'summary.json', summary)
    check_source()
    summary['parity_passed'] = all(variant['parity']['passed'] for case in summary['cases']
                                 for variant in case['variants'].values())
    summary['artifact_sha256'] = {str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.rglob('*')) if path.is_file() and path != args.output / 'summary.json'}
    write_json(args.output / 'summary.json', summary)
    return 0 if summary['parity_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
