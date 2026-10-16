"""Paired same-process work diagnostic, not a cold benchmark campaign.

Requires the sibling glqp-bench package for MM loading and independent checks.
Run with one CPU and numerical thread when collecting exploratory profiles;
use glqp-bench's parallel-8 campaigns for revision acceptance.
"""
import argparse
import cProfile
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pstats
import sys
import time

import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy import sparse
from scipy.special import expit
from glqp_bench.qp import load_problem, validate_solution


def load_package(name, source):
    path = Path(source) / "glqp" / "__init__.py"
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_glm(params, glm, result, tolerance=1e-6):
    """Check the original bound-constrained GLM equations independently."""
    loss, response, weights = glm
    z = params['A'] @ result.x
    mean = {'GaussianNLL':lambda: z, 'LogisticNLL':lambda: expit(z),
            'PoissonNLL':lambda: np.exp(z)}[loss]()
    gradient = params['A'].T @ (weights*(mean-response)) + params['Q'] @ result.x
    slack = params['c'] - params['C'] @ result.x
    errors = dict(
        stationarity=float(np.max(np.abs(gradient + params['C'].T @ result.y))),
        inequality=float(max(0., -np.min(slack))),
        complementarity=float(np.max(np.abs(result.y*slack))),
        dual_sign=float(max(0., -np.min(result.y))))
    finite = np.isfinite(result.x).all() and np.isfinite(result.y).all()
    accepted = finite and all(np.isfinite(v) and v <= tolerance for v in errors.values())
    return dict(accepted=bool(accepted), tolerance=tolerance, **errors)


def source_hashes(source):
    root = Path(source)
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root/'glqp').glob('*.py'))}


parser = argparse.ArgumentParser()
parser.add_argument('--reference', required=True)
parser.add_argument('--candidate', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--benchmark-repo', type=Path,
                    default=Path(__file__).resolve().parents[2]/'glqp-bench')
parser.add_argument('--cases', nargs='+', help='Subset of the fixed QP/GLM panel')
parser.add_argument('--profile-case', default='QPCBOEI2')
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
modules = {label: load_package('measure_'+label, source) for label, source in
           [('reference', args.reference), ('candidate', args.candidate)]}
settings = dict(tol=1e-7, max_iterative_refinement=10, max_stagnation=5000,
                max_iter=10000, max_linesearch_steps=40, let_newton_cook=.5, max_time=60)
cases = {}
for name in ['HS21', 'QAFIRO', 'CVXQP1_S', 'QPCBOEI2', 'CONT-101', 'Q25FV47']:
    p = load_problem(args.benchmark_repo/'mm_problems/MAT_Files'/f'{name}.mat')
    cases[name] = (dict(Q=p.P, b=-p.q, E=p.A if p.p else None,
                        e=p.b if p.p else None, C=p.G if p.m else None,
                        c=p.h if p.m else None), p, None)
rng = np.random.default_rng(17)
A = sparse.csc_array(rng.normal(size=(1000, 40)) / 10)
Q = .3*sparse.eye_array(40, format='csc')
C = sparse.vstack([sparse.eye_array(40), -sparse.eye_array(40)], format='csc')
for loss in ['GaussianNLL', 'LogisticNLL', 'PoissonNLL']:
    responses = {'GaussianNLL':rng.normal(size=1000),
                 'LogisticNLL':rng.integers(0,2,size=1000),
                 'PoissonNLL':rng.poisson(1,size=1000)}[loss]
    cases[loss] = (dict(A=A, Q=Q, C=C, c=np.ones(80)), None,
                   (loss, responses, rng.uniform(.5,1.5,size=1000)))

if args.cases:
    unknown = set(args.cases) - cases.keys()
    if unknown:
        parser.error(f'Unknown cases: {sorted(unknown)}')
    cases = {name:cases[name] for name in args.cases}
if args.profile_case not in cases:
    parser.error('--profile-case must be one of the selected cases')

summaries = []
for index, (name, (kwargs, qp, glm)) in enumerate(cases.items()):
    results = {}
    timings = {label:[] for label in modules}
    for repeat in range(args.repeats):
        order = list(modules) if (repeat+index)%2==0 else list(reversed(modules))
        for label in order:
            mod = modules[label]
            params = kwargs.copy()
            if glm:
                loss, y, w = glm
                params['f'] = getattr(sys.modules[mod.__name__+'.obj'], loss)(y,w)
            model = mod.GLQP(**params)
            start = time.perf_counter()
            x, result = model.solve(settings=mod.SolverSettings(**settings), verbose=False)
            timings[label].append(time.perf_counter()-start)
            results[label] = (x,result)
    bx, br = results['reference']
    cx, cr = results['candidate']
    columns = [c for c in br.history if c != 'time']
    same_shape = len(br.history) == len(cr.history)
    numeric_columns = [c for c in columns if is_numeric_dtype(br.history[c].dtype)]
    same_history = same_shape and all(c in cr.history and np.array_equal(br.history[c],cr.history[c]) for c in columns)
    differences = {c:float(np.max(np.abs(br.history[c].to_numpy()-cr.history[c].to_numpy())))
                   for c in numeric_columns} if same_shape else {}
    item = dict(problem=name, timings=timings,
                median_ratio=float(np.median(timings['candidate'])/np.median(timings['reference'])),
                statuses=[br.termination_tag, cr.termination_tag],
                iterations=[len(br.history),len(cr.history)], identical_history=same_history,
                max_x_difference=float(np.max(np.abs(bx-cx))), max_history_difference=differences)
    if qp:
        item['validation']={label:validate_solution(qp, r.x, r.nu[:qp.p], r.y[:qp.m])
                            for label, (_,r) in results.items()}
    else:
        item['validation']={label:validate_glm(params, glm, r)
                            for label,(_,r) in results.items()}
    for label,(_,r) in results.items():
        r.history.to_csv(out/f'{name}-{label}.csv',index=False)
    summaries.append(item)
    print(json.dumps({k:item[k] for k in ['problem','median_ratio','statuses','iterations','identical_history','max_x_difference']}),flush=True)

profiles = {}
for label, mod in modules.items():
    kwargs, _, glm = cases[args.profile_case]
    params = kwargs.copy()
    if glm:
        loss, y, w = glm
        params['f'] = getattr(sys.modules[mod.__name__+'.obj'], loss)(y,w)
    profiler = cProfile.Profile()
    profiler.enable()
    mod.GLQP(**params).solve(settings=mod.SolverSettings(**settings), verbose=False)
    profiler.disable()
    profiler.dump_stats(str(out/f'{label}.prof'))
    stats = pstats.Stats(profiler)
    profiles[label] = [{'file':key[0], 'line':key[1], 'function':key[2],
        'calls':value[1], 'self_seconds':value[2], 'cumulative_seconds':value[3]}
        for key,value in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:60]]
summary = dict(kind='exploratory same-process paired timing; not a cold campaign',
               repeats=args.repeats,
               profile_case=args.profile_case,
               affinity=sorted(os.sched_getaffinity(0)), sources={label:dict(path=str(source),
               solver_sha256=hashlib.sha256((Path(source)/'glqp/solver.py').read_bytes()).hexdigest(),
               file_sha256=source_hashes(source))
               for label,source in [('reference',args.reference),('candidate',args.candidate)]},
               settings=settings, cases=summaries, profiles=profiles)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
