"""Inspect PRIMAL4 solve work; this diagnostic does not measure performance."""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    sys.path[:0] = [str(campaign/'source/glqp'), str(campaign/'source/benchmark/src')]
    import numpy as np
    import glqp
    from glqp.presolve import PresolveOptions, presolve_qp
    from glqp_bench.qp import load_problem, validate_solution
    module = importlib.import_module('glqp.solver')
    problem = load_problem(campaign/'source/benchmark/mm_problems/MAT_Files/PRIMAL4.mat')
    original = module.factor_and_solve
    panels = {}
    for variant, scalar in [('glqp_base', False), ('glqp_candidate', True)]:
        record = json.loads((campaign/f'source/benchmark/runs/mm/records/{variant}/PRIMAL4__r000.json').read_text())
        settings = module.SolverSettings(**record['diagnostics']['solver_settings'])
        trace = []

        def observe(matrix, *arguments, **keywords):
            result = original(matrix, *arguments, **keywords)
            trace.append(dict(shape=list(matrix.shape), matrix_nnz=int(matrix.nnz),
                factor=result.factorization.diagnostics(), solve=asdict(result.info)))
            return result

        with presolve_qp(problem.P, problem.q, problem.A, problem.b, problem.G, problem.h,
            objective_constant=problem.objective_constant,
            options=PresolveOptions(local_fallback=True, separable_quadratics=scalar)) as reduced:
            model = glqp.GLQP(Q=reduced.Q, b=-reduced.q,
                C=reduced.G if len(reduced.h) else None, c=reduced.h if len(reduced.h) else None,
                E=reduced.A if len(reduced.b) else None, e=reduced.b if len(reduced.b) else None)
            module.factor_and_solve = observe
            try:
                x, result = model.solve(verbose=False, settings=settings)
            finally:
                module.factor_and_solve = original
            x, y, z = reduced.recover(x, np.asarray(result.nu)[:len(reduced.b)], np.asarray(result.y)[:len(reduced.h)])
            validation = validate_solution(problem, x, y, z, tol=1e-6)
        if not validation['accepted']:
            raise ValueError(f'Diagnostic failed original KKT validation: {variant}')
        panels[variant] = dict(iterations=len(result.history), validation=validation, trace=trace,
            factor_nonzero_counts=sorted({row['factor']['factor_nnz'] for row in trace}),
            total_factor_attempts=sum(row['solve']['attempts'] for row in trace),
            total_refinements=sum(row['solve']['refinements'] for row in trace))
    evidence = dict(kind='Untimed work-count diagnosis; no wall speedup claims', problem='PRIMAL4',
        campaign=str(campaign), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        solver_commit=json.loads((campaign/'manifest.json').read_text())['glqp']['commit'], panels=panels)
    with args.output.open('x') as stream:
        json.dump(evidence, stream, indent=2)
        stream.write('\n')
    print(json.dumps({name:{key:value for key,value in panel.items() if key not in ('trace','validation')}
                     for name,panel in panels.items()}, indent=2))


if __name__ == '__main__':
    main()
