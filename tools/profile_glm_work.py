"""Paired GLM work profiles on canonical glqp-bench data; not cold timings."""
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
from scipy import sparse


def load_package(name, source):
    path = Path(source).resolve() / "glqp/__init__.py"
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def hashes(source):
    return {str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((source / "glqp").glob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path,
        default=Path(__file__).resolve().parents[2] / "glqp-bench")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-case")
    parser.add_argument("--settings-profile", default="glm-v2")
    args = parser.parse_args()
    if args.repeats < 1 or len(set(args.cases)) != len(args.cases):
        parser.error("Use a positive repeat count and unique cases")
    if args.profile_case and args.profile_case not in args.cases:
        parser.error("The profile case must be selected")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.benchmark_repo.resolve() / "src"))
    from glqp_bench.glm import build_glm, glm_hash, glm_payload, validate_glm
    from glqp_bench.adapters import glqp_settings
    sources = dict(reference=args.reference.resolve(), candidate=args.candidate.resolve())
    initial_hashes = {label:hashes(source) for label,source in sources.items()}
    modules = {label:load_package("profile_glm_"+label, source) for label,source in sources.items()}
    bench_sources = {name:hashlib.sha256((args.benchmark_repo / "src/glqp_bench" / name).read_bytes()).hexdigest()
        for name in ("glm.py", "adapters.py")}
    results = []
    for index, name in enumerate(args.cases):
        problem = build_glm(name)  # Generation and validation are never timed.
        settings = {label:glqp_settings(module, tol=1e-7, time_limit=60,
            settings_profile=args.settings_profile) for label,module in modules.items()}

        def solve(label):
            module = modules[label]
            objective = getattr(sys.modules[module.__name__+".obj"],
                {"gaussian":"GaussianNLL", "logistic":"LogisticNLL", "poisson":"PoissonNLL"}[problem.loss])
            model = module.GLQP(f=objective(problem.response, problem.weights), A=problem.X,
                Q=sparse.diags(problem.ridge, format="csc"), b=-problem.q,
                E=problem.E if len(problem.e) else None, e=problem.e if len(problem.e) else None,
                C=problem.C if len(problem.c) else None, c=problem.c if len(problem.c) else None)
            start = time.perf_counter()
            _, result = model.solve(verbose=False, settings=settings[label])
            return result, time.perf_counter()-start

        records, timings = {}, {label:[] for label in modules}
        for repeat in range(args.repeats):
            order = list(modules) if (repeat+index)%2 == 0 else list(reversed(modules))
            for label in order:
                result, elapsed = solve(label)
                timings[label].append(elapsed)
                records[label] = result
        reference, candidate = records["reference"], records["candidate"]
        equal_history = (reference.history.drop(columns="time").equals(candidate.history.drop(columns="time")))
        record = dict(problem_id=name, canonical_data_sha256=glm_hash(problem),
            dimensions=dict(observations=problem.X.shape[0], variables=problem.X.shape[1], design_nnz=problem.X.nnz),
            settings={label:vars(setting) for label,setting in settings.items()}, timings=timings,
            median_ratio=float(np.median(timings["candidate"])/np.median(timings["reference"])),
            statuses={label:result.termination_tag for label,result in records.items()},
            iterations={label:len(result.history) for label,result in records.items()},
            identical_non_time_history=equal_history,
            identical_solution_arrays={key:bool(np.array_equal(getattr(reference,key),getattr(candidate,key)))
                for key in ("x", "y", "s", "nu")},
            validation={label:validate_glm(problem, result.x, result.nu[:len(problem.e)], result.y[:len(problem.c)])
                for label,result in records.items()})
        for label,result in records.items():
            result.history.to_csv(args.output / f"{name}-{label}.csv", index=False)
            np.savez_compressed(args.output / f"{name}-{label}.npz", **glm_payload(problem),
                x=result.x, y=result.nu[:len(problem.e)], z=result.y[:len(problem.c)], s=result.s)
        if name == args.profile_case:
            record["profiles"] = {}
            for label in modules:
                profiler = cProfile.Profile()
                profiler.enable()
                solve(label)
                profiler.disable()
                profiler.dump_stats(str(args.output / f"{label}.prof"))
                stats = pstats.Stats(profiler)
                record["profiles"][label] = [dict(file=key[0], line=key[1], function=key[2], calls=value[1],
                    self_seconds=value[2], cumulative_seconds=value[3])
                    for key,value in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:80]]
        results.append(record)
        print(json.dumps({key:record[key] for key in ("problem_id", "median_ratio", "statuses", "iterations",
            "identical_non_time_history", "identical_solution_arrays")}), flush=True)
    summary = dict(kind="Exploratory paired same-process solves; not a cold benchmark campaign",
        source_hashes=initial_hashes, source_paths={label:str(source) for label,source in sources.items()},
        source_hashes_unchanged=initial_hashes=={label:hashes(source) for label,source in sources.items()},
        benchmark_sources=bench_sources, benchmark_repo=str(args.benchmark_repo.resolve()),
        affinity=sorted(os.sched_getaffinity(0)), thread_environment={name:os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        settings_profile=args.settings_profile, repeats=args.repeats, results=results)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
