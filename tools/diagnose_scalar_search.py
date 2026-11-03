#!/usr/bin/env python3
"""Compare GLM search work at equal settings, without performance timing claims.

Each mode runs plain and then with observers on fresh models. The observer
counts sparse vector products and public scalar-loss calls, preserves rejected
merit trials, and checks byte parity with the plain result. Built-in scalar-ray
kernels bypass public loss callbacks: their value/derivative counters are
reported separately. Diagnostic observer products are also separated from
solver products. No elapsed duration is used as comparative evidence.

Output directories must be new. Changed source files abort the run, retaining
the partial attempt. Run again in a different directory after source freezes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import functools
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import warnings


DEFAULT_CASES = (
    "glm_stress_hard_poisson_range_free_v1",
    "glm_stress_hard_poisson_range_box_v1",
    "glm_stress_hard_poisson_range_affine_v1",
    "glm_bike_box_v1",
    "glm_logistic_small_affine_s123_v1",
    "glm_gaussian_small_affine_s123_v1",
)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def finite_json(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    payload = json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n"
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as stream:
            stream.write(payload)
    else:
        path.write_text(payload)


def source_hashes(source):
    result = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted((source / "glqp").rglob("*.py"))}
    if not result:
        raise ValueError(f"No solver Python sources in {source}")
    return result


def history_record(history):
    columns = [name for name in history.columns if "time" not in name.lower()]
    return dict(columns=columns, rows=history[columns].to_dict(orient="records"))


class WorkObserver:
    """Tool-local, read-only instrumentation; no solver arithmetic changes."""
    def __init__(self, model, source):
        self.model, self.source = model, source
        self.counts = Counter()
        self.matrices = {id(getattr(model, key)): (getattr(model, key), key)
                         for key in ("A", "Q", "C", "E")}
        self.phase = "solver"
        self.scalar_searches, self.backtracking_trials = [], []
        self.first_ray = None

    def attribution(self):
        if self.phase != "solver":
            return self.phase
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_filename == str(self.source / "glqp/diagnostics.py"):
                return "diagnostic_observer"
            frame = frame.f_back
        return "solver"

    def loss_wrapper(self, name, original):
        @functools.wraps(original)
        def wrapped(loss, *args, **kwargs):
            self.counts[f"{self.attribution()}.loss.{name}"] += 1
            return original(loss, *args, **kwargs)
        return wrapped

    def matmul_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(matrix, other):
            import numpy as np
            if np.ndim(other) == 1:
                registered = self.matrices.get(id(matrix))
                name = registered[1] if registered is not None else "other_sparse"
                self.counts[f"{self.attribution()}.matvec.{name}"] += 1
            return original(matrix, other)
        return wrapped

    def transpose_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(cache, name, matrix):
            result = original(cache, name, matrix)
            registered = self.matrices.get(id(matrix))
            if registered is not None:
                self.matrices[id(result)] = result, registered[1] + ".T"
            return result
        return wrapped

    def scalar_wrapper(self, original):
        @functools.wraps(original)
        def wrapped(ray, **kwargs):
            if self.first_ray is None:
                self.first_ray = {key: getattr(ray, key).copy() for key in ("z", "dz", "s", "ds")}
            previous = self.phase
            self.phase = "scalar_search"
            try:
                result = original(ray, **kwargs)
            finally:
                self.phase = previous
            self.scalar_searches.append(dict(
                index=len(self.scalar_searches), status=result.status,
                controls={key: kwargs[key] for key in ("tmax", "merit0", "slope0",
                          "armijo_param", "additive_eps", "max_trials")},
                counters=result.counters, trials=result.history,
                accepted_step=None if result.trial is None else result.trial.step))
            return result
        return wrapped

    def profile(self, frame, event, value):
        if (frame.f_code.co_filename != str(self.source / "glqp/solver.py")
                or frame.f_code.co_name != "merit_line"):
            return
        if event == "call":
            self.phase = "backtracking_merit"
        elif event == "return":
            import numpy as np
            self.phase = "solver"
            local, outer = frame.f_locals, frame.f_back.f_locals
            step, settings = local["t"], outer["settings"]
            bound = (outer["merit0"] + settings.armijo_param*step*outer["merit_slope"]
                     + settings.armijo_additive_eps)
            self.backtracking_trials.append(dict(
                iteration=outer["iteration_number"], trial=outer["linesearch_step"]+1,
                step=step, merit=value, armijo_bound=bound,
                armijo=bool(np.isfinite(value) and value < bound),
                predictor_min=float(np.min(local["trial_z"])),
                predictor_max=float(np.max(local["trial_z"]))))
            if self.first_ray is None:
                self.first_ray = {key: outer[key].copy() for key in ("z", "dz", "s", "ds")}

    def install(self, stack):
        from scipy.sparse._base import _spbase
        import glqp.solver as solver_module
        from glqp.util import SparseTransposeCache
        objective = type(self.model.f)
        for name in ("f", "d1f", "d2f"):
            stack.enter_context(patch.object(objective, name,
                self.loss_wrapper(name, getattr(objective, name))))
        stack.enter_context(patch.object(_spbase, "__matmul__", self.matmul_wrapper(_spbase.__matmul__)))
        stack.enter_context(patch.object(SparseTransposeCache, "get",
            self.transpose_wrapper(SparseTransposeCache.get)))
        stack.enter_context(patch.object(solver_module, "search_scalar_ray",
            self.scalar_wrapper(solver_module.search_scalar_ray)))
        previous = sys.getprofile()
        stack.callback(sys.setprofile, previous)
        sys.setprofile(self.profile)


def solve_case(problem, mode, instrumented, args, output):
    import numpy as np
    from scipy import sparse
    import glqp
    from glqp.obj import GaussianNLL, LogisticNLL, PoissonNLL
    from glqp_bench.adapters import glqp_settings
    from glqp_bench.glm import validate_glm

    settings = glqp_settings(glqp, tol=1e-7, time_limit=600., settings_profile="glm-v2")
    settings.glm_linesearch = mode
    settings.iteration_diagnostics = instrumented
    objective = {"gaussian": GaussianNLL, "logistic": LogisticNLL, "poisson": PoissonNLL}[problem.loss]
    model = glqp.GLQP(f=objective(problem.response, problem.weights), A=problem.X,
        Q=sparse.diags(problem.ridge, format="csc"), b=-problem.q,
        C=problem.C if len(problem.c) else None, c=problem.c if len(problem.c) else None,
        E=problem.E if len(problem.e) else None, e=problem.e if len(problem.e) else None)
    observer = WorkObserver(model, args.source) if instrumented else None
    with warnings.catch_warnings(record=True) as caught, ExitStack() as stack:
        warnings.simplefilter("always")
        if observer is not None:
            observer.install(stack)
        _, result = model.solve(verbose=False, settings=settings)
    arrays = {key: np.asarray(getattr(result, key)) for key in ("x", "y", "s", "nu")}
    np.savez_compressed(output / "solution.npz", **arrays)
    record = dict(settings=asdict(settings), native=result.termination_tag,
        iterations=len(result.history), exception=None if result.exception is None else repr(result.exception),
        primal=result.primal, dual_res=result.dual_res, cons_viol=result.cons_viol, feasible=result.feasible,
        validation=validate_glm(problem, result.x, result.nu[:len(problem.e)], result.y[:len(problem.c)]),
        arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype),
                         sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key, value in arrays.items()},
        initialization_linear_solve=result.initialization_linear_solve,
        last_linear_solve=result.last_linear_solve,
        warnings=dict(Counter(str(item.message) for item in caught)))
    write_json(output / "result.json", record)
    write_json(output / "history.json.gz", history_record(result.history))
    if observer is not None:
        searches = [item["line_search"] for item in result.diagnostics["iterations"] if "line_search" in item]
        scalar_counts = Counter()
        for search in observer.scalar_searches:
            scalar_counts.update(search["counters"])
        record["work"] = dict(counts=dict(sorted(observer.counts.items())),
            scalar_totals=dict(scalar_counts), search_count=len(searches),
            trial_count=sum(item["trial_count"] for item in searches),
            backtracking_merit_evaluations=len(observer.backtracking_trials),
            first_search=None if not searches else searches[0],
            note="Public loss callbacks exclude fused scalar-ray kernels; scalar value counts include base evaluations. "
                 "Diagnostic observer matvecs are excluded from solver attribution; other_sparse includes linear refinement.")
        write_json(output / "work.json", record["work"])
        write_json(output / "diagnostics.json.gz", result.diagnostics)
        write_json(output / "trials.json.gz", dict(scalar_searches=observer.scalar_searches,
            backtracking_merit_trials=observer.backtracking_trials,
            note="Backtracking list records evaluated merits; pre-evaluation invalid trials are counted in diagnostics."))
        if observer.first_ray is not None:
            np.savez_compressed(output / "first-ray.npz", **observer.first_ray)
    return result, record


def compare_results(plain, observed):
    import numpy as np
    arrays = {key: bool(np.asarray(getattr(plain, key)).dtype == np.asarray(getattr(observed, key)).dtype
              and np.asarray(getattr(plain, key)).shape == np.asarray(getattr(observed, key)).shape
              and np.asarray(getattr(plain, key)).tobytes() == np.asarray(getattr(observed, key)).tobytes())
              for key in ("x", "y", "s", "nu")}
    columns = [name for name in plain.history.columns if "time" not in name.lower()]
    history = all(name in observed.history for name in columns) and plain.history[columns].equals(observed.history[columns])
    fields = {key: getattr(plain, key) == getattr(observed, key)
              for key in ("termination_tag", "primal", "dual_res", "cons_viol", "feasible")}
    conclusive = plain.exception is None and observed.exception is None
    return dict(passed=conclusive and all(arrays.values()) and history and all(fields.values()),
        conclusive=conclusive, array_bytes=arrays, common_non_time_history=history,
        compared_history_columns=columns, summaries=fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=6)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    args = parser.parse_args()
    for name in ("source", "benchmark_repo", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {args.cpu})
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    sys.path[:0] = [str(args.source), str(args.benchmark_repo / "src")]
    baseline = source_hashes(args.source)
    import numpy as np
    import scipy
    import glqp
    from threadpoolctl import threadpool_info
    from glqp_bench.glm import build_glm, glm_hash, glm_payload
    if Path(glqp.__file__).resolve().parent != args.source / "glqp":
        raise RuntimeError(f"Wrong solver import: {glqp.__file__}")
    summary = dict(purpose="Untimed equal-settings scalar search work and correctness diagnostic",
        source=str(args.source), source_sha256=baseline, source_unchanged=True,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment=dict(cpu=args.cpu, affinity=sorted(os.sched_getaffinity(0)),
            native_threads=1, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
            threadpools=threadpool_info()),
        settings_contract="glm-v2, tol=1e-7, max_iter=1000, max_linesearch_steps=100; only search mode and observers differ",
        cases=[])
    write_json(args.output / "summary.json", summary)
    for name in args.cases:
        problem = build_glm(name)
        case_output = args.output / name
        case_output.mkdir()
        np.savez_compressed(case_output / "problem.npz", **glm_payload(problem))
        case = dict(problem_id=name, loss=problem.loss, data_sha256=glm_hash(problem), modes={})
        for mode in ("backtracking", "scalar"):
            results, records = {}, {}
            for label in ("plain", "observed"):
                destination = case_output / mode / label
                destination.mkdir(parents=True)
                if source_hashes(args.source) != baseline:
                    summary["source_unchanged"] = False
                    write_json(args.output / "summary.json", summary)
                    raise RuntimeError("Solver sources changed; partial attempt retained. Use a new output directory.")
                results[label], records[label] = solve_case(problem, mode, label == "observed", args, destination)
                if source_hashes(args.source) != baseline:
                    summary["source_unchanged"] = False
                    write_json(args.output / "summary.json", summary)
                    raise RuntimeError("Solver sources changed during solve; partial attempt retained.")
            parity = compare_results(results["plain"], results["observed"])
            write_json(case_output / mode / "parity.json", parity)
            case["modes"][mode] = dict(native=records["plain"]["native"],
                iterations=records["plain"]["iterations"], validation=records["plain"]["validation"],
                parity=parity, work=records["observed"]["work"])
            print(json.dumps(finite_json(dict(problem=name, mode=mode,
                native=records["plain"]["native"], iterations=records["plain"]["iterations"],
                accepted=records["plain"]["validation"]["accepted"],
                parity=parity["passed"], trials=records["observed"]["work"]["trial_count"]))), flush=True)
        summary["cases"].append(case)
        write_json(args.output / "summary.json", summary)
    summary["parity_passed"] = all(mode["parity"]["passed"] for case in summary["cases"]
                                   for mode in case["modes"].values())
    summary["source_unchanged"] = source_hashes(args.source) == baseline
    summary["artifact_sha256"] = {str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.output.rglob("*")) if path.is_file() and path != args.output / "summary.json"}
    write_json(args.output / "summary.json", summary)
    return 0 if summary["parity_passed"] and summary["source_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
