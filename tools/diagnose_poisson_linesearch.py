"""Record first-direction Poisson trials without changing the search algorithm."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path,
        default=Path(__file__).resolve().parents[2] / "glqp-bench")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limits", type=int, nargs="+", default=[50, 100])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    package = importlib.util.spec_from_file_location("poisson_probe", source / "glqp/__init__.py",
        submodule_search_locations=[str(source / "glqp")])
    backend = importlib.util.module_from_spec(package)
    sys.modules["poisson_probe"] = backend
    package.loader.exec_module(backend)
    objective = sys.modules["poisson_probe.obj"].PoissonNLL
    sys.path.insert(0, str(args.benchmark_repo.resolve() / "src"))
    from glqp_bench.glm import build_glm, glm_hash, validate_glm
    from glqp_bench.adapters import glqp_settings
    records = []
    for constraint in ("free", "box", "affine"):
        problem = build_glm(f"glm_bike_{constraint}_v1")
        for limit in args.limits:
            settings = glqp_settings(backend, tol=1e-7, time_limit=60, settings_profile="glm-v1")
            settings.max_linesearch_steps = limit
            model = backend.GLQP(f=objective(problem.response, problem.weights), A=problem.X,
                Q=sparse.diags(problem.ridge, format="csc"), b=-problem.q,
                E=problem.E if len(problem.e) else None, e=problem.e if len(problem.e) else None,
                C=problem.C if len(problem.c) else None, c=problem.c if len(problem.c) else None)
            trials, directions = [], []

            def trace(frame, event, value):
                if event != "return" or frame.f_code.co_filename != str(source / "glqp/solver.py"):
                    return
                if frame.f_code.co_name == "solve_KKT" and not directions:
                    directions.append(value[0].copy())
                if frame.f_code.co_name != "merit_line" or frame.f_back.f_locals["iteration_number"] != 0:
                    return
                local, outer = frame.f_locals, frame.f_back.f_locals
                bound = outer["merit0"] + settings.armijo_param*local["t"]*outer["merit_slope"] + settings.armijo_additive_eps
                trials.append(dict(loop_index=outer["linesearch_step"]+1, t=float(local["t"]),
                    tmax=float(outer["tmax"]), predictor_min=float(np.min(local["trial_z"])),
                    predictor_max=float(np.max(local["trial_z"])), merit=float(value) if np.isfinite(value) else None,
                    armijo_bound=float(bound), armijo_satisfied=bool(np.isfinite(value) and value < bound)))

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sys.setprofile(trace)
                try:
                    x, result = model.solve(verbose=False, settings=settings)
                finally:
                    sys.setprofile(None)
            validation = validate_glm(problem, x, result.nu[:len(problem.e)], result.y[:len(problem.c)])
            record = dict(problem_id=problem.name, canonical_data_sha256=glm_hash(problem),
                settings=vars(settings), status=result.termination_tag, iterations=len(result.history),
                response_mean=float(np.mean(problem.response)), weight_sum=float(np.sum(problem.weights)),
                intercept_ridge=float(problem.ridge[-1]), initial_intercept_direction=float(directions[0][-1]),
                minimum_trial_fraction=0.9**(limit-1), first_direction_trials=trials,
                first_finite_trial=next((item["loop_index"] for item in trials if item["merit"] is not None), None),
                first_armijo_trial=next((item["loop_index"] for item in trials if item["armijo_satisfied"]), None),
                validation=validation, warnings=[str(item.message) for item in caught])
            records.append(record)
            name = f"{problem.name}-cap{limit}"
            result.history.to_csv(args.output / f"{name}.csv", index=False)
            np.savez_compressed(args.output / f"{name}.npz", x=x, y=result.nu, z=result.y,
                first_direction=directions[0])
            print(json.dumps({key:record[key] for key in ("problem_id", "status", "iterations",
                "initial_intercept_direction", "first_finite_trial", "first_armijo_trial")}), flush=True)
    sources = {str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((source / "glqp").glob("*.py"))}
    summary = dict(kind="Instrumented functional diagnostic; no timing claims", source=str(source),
        source_sha256=sources, benchmark_glm_sha256=hashlib.sha256(
            (args.benchmark_repo / "src/glqp_bench/glm.py").read_bytes()).hexdigest(), results=records)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
