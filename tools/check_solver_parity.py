#!/usr/bin/env python3
"""Check exact solver parity between two source trees, without timing claims.

Each tree is imported in its own fresh subprocess. Arrays are compared byte for
byte; all history columns except ``time``, native termination, effective
settings, final summaries, and linear-solve reports are compared exactly.
Optional full diagnostics are compared too. No approximate tolerance is used.

Example (activate the GLQP environment first)::

    python tools/check_solver_parity.py --reference /tmp/frozen-old \
        --candidate /tmp/frozen-new --output /tmp/parity --cpu 4 \
        --suite all --diagnostics both --benchmark-root ../glqp-bench

Freeze both source trees before running. The output records their hashes and
checks that neither changed during the run. MM inputs use the benchmark's
canonical MAT loader; its source and selected input files are also hashed.
Failed optimization is a valid parity observation. A time-limit termination,
uncaught exception, or exception-bearing result is inconclusive and makes the
check fail.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fnmatch
import gzip
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


POLICIES = {
    "accepted_newton": {},
    "legacy_pc": {"step_strategy": "predictor_corrector"},
    "no_slack": {"proximal_policy": "no_slack"},
    "centered_newton_residual": {"proximal_policy": "centered"},
    "centered_newton_progress": {
        "proximal_policy": "centered", "proximal_reg_update": "progress"},
    "centered_pc_progress": {
        "proximal_policy": "centered", "proximal_reg_update": "progress",
        "step_strategy": "predictor_corrector"},
}
SMALL_PROBLEMS = (
    "qp_unconstrained", "qp_equality", "qp_box_and_equality",
    "qp_inconsistent_slacks", "glm_gaussian", "glm_logistic", "glm_poisson",
    "glm_poisson_no_inequalities",
)
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_manifest(source):
    manifest = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((source / "glqp").rglob("*.py"))}
    if "glqp/__init__.py" not in manifest:
        raise ValueError(f"Empty or invalid GLQP package: {source}")
    return manifest


def numerical_versions():
    versions = {}
    for name in ("numpy", "scipy", "pandas", "qdldl", "sparse-dot-mkl", "mkl"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def make_jobs(args):
    jobs = []
    if args.suite in ("small", "all"):
        diagnostics = {"off": [False], "on": [True], "both": [False, True]}[args.diagnostics]
        for problem in SMALL_PROBLEMS:
            # These cases explicitly distinguish no settings argument from
            # SolverSettings(). In QP mode their let_newton_cook defaults differ.
            for mode in ("implicit", "explicit_defaults"):
                jobs.append(dict(name=f"{problem}/{mode}", problem=problem,
                                 settings_mode=mode, settings={}, diagnostics=False))
            for policy, overrides in POLICIES.items():
                for enabled in diagnostics:
                    settings = dict(max_iter=300, max_time=600.,
                                    iteration_diagnostics=enabled, **overrides)
                    jobs.append(dict(name=f"{problem}/{policy}/diagnostics_{'on' if enabled else 'off'}",
                                     problem=problem, settings_mode="custom",
                                     settings=settings, diagnostics=enabled))
    if args.suite in ("mm", "all"):
        for problem in ("QPCBOEI2", "YAO", "QSHELL", "LISWET1"):
            policies = ("centered_newton_progress",) if problem == "LISWET1" else (
                "accepted_newton", "centered_newton_progress")
            for policy in policies:
                settings = dict(tol=1e-7, max_iter=10000, max_stagnation=5000,
                                max_linesearch_steps=40, let_newton_cook=.5,
                                max_time=600., iteration_diagnostics=args.mm_diagnostics,
                                **POLICIES[policy])
                jobs.append(dict(name=f"mm_{problem}/{policy}", problem=problem,
                                 mm=True, settings_mode="custom", settings=settings,
                                 diagnostics=args.mm_diagnostics))
    if args.case:
        jobs = [job for job in jobs if any(fnmatch.fnmatchcase(job["name"], pattern)
                                          for pattern in args.case)]
    if not jobs:
        raise ValueError("No cases matched the selected suite and --case patterns")
    return jobs


def small_problem(name):
    """Construct fresh deterministic inputs for each solve."""
    import numpy as np
    from scipy.sparse import csc_array, eye_array
    from glqp.obj import GaussianNLL, LogisticNLL, PoissonNLL

    rng = np.random.default_rng(145)
    raw = rng.normal(size=(12, 12))
    Q = raw.T @ raw + .2 * np.eye(12)
    C = np.vstack([np.eye(12), -np.eye(12)])
    E = rng.normal(size=(2, 12))
    equality_b, mixed_b = rng.normal(size=12), 3 * rng.normal(size=12)
    if name == "qp_unconstrained":
        return dict(Q=eye_array(3), b=np.array([1., -2., .3])), {}
    if name == "qp_equality":
        return dict(Q=csc_array(Q), b=equality_b, E=csc_array(E), e=np.array([.2, -.1])), {}
    if name == "qp_box_and_equality":
        return dict(Q=csc_array(Q), b=mixed_b, C=csc_array(C), c=np.full(24, .5),
                    E=csc_array(E), e=np.array([.2, -.1])), {}
    if name == "qp_inconsistent_slacks":
        return (dict(Q=eye_array(2), b=np.array([1., -.5]),
                     C=csc_array([[1., 0.], [0., -1.]]), c=np.array([.2, .4])),
                dict(x0=np.array([2., -3.]), s0=np.array([4., .01]), y0=np.array([.2, 3.])))
    objectives = {
        "glm_gaussian": (GaussianNLL, [.2, -.3, .6]),
        "glm_logistic": (LogisticNLL, [0., 1., 1.]),
        "glm_poisson": (PoissonNLL, [0., 3., 2.]),
        "glm_poisson_no_inequalities": (PoissonNLL, [0., 3., 2.]),
    }
    objective, response = objectives[name]
    problem = dict(f=objective(response, [.7, 1.3, .2]),
                   A=csc_array([[1., -.2], [.4, .8], [1.5, .3]]),
                   Q=csc_array(np.diag([.2, .4])), b=np.array([.1, -.2]),
                   C=csc_array([[1., 0.], [-1., 0.]]), c=np.array([.2, .4]),
                   E=csc_array([[1., 1.]]), e=np.array([.7]))
    if name == "glm_poisson_no_inequalities":
        for key in ("b", "C", "c"):
            del problem[key]
        problem["Q"] = eye_array(2)
        return problem, {}
    return problem, dict(x0=np.array([3., -2.3]))


def encode(value):
    """Retain exact scalar floating-point values, including signed zero/NaN."""
    import numpy as np

    if isinstance(value, (float, np.floating)):
        return {"float_hex": float(value).hex()}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [encode(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def collect_result(result):
    import numpy as np

    arrays = {}
    for name in ("x", "y", "s", "nu"):
        value = np.asarray(getattr(result, name))
        raw = value.tobytes(order="C")
        arrays[name] = dict(dtype=value.dtype.str, shape=list(value.shape),
                            bytes_hex=raw.hex(), sha256=hashlib.sha256(raw).hexdigest())
    history = result.history.drop(columns=["time"], errors="ignore")
    exception = result.exception
    return dict(
        termination=dict(tag=result.termination_tag, iterations=len(history),
                         exception=None if exception is None else dict(
                             type=type(exception).__name__, message=str(exception))),
        arrays=arrays,
        history=dict(columns=list(history.columns), dtypes=[str(v) for v in history.dtypes],
                     index=encode(history.index.tolist()), values=encode(history.to_dict(orient="list"))),
        final=encode({key: getattr(result, key) for key in
                      ("primal", "dual_res", "cons_viol", "feasible")}),
        settings=encode(asdict(result.settings)),
        initialization_linear_solve=encode(result.initialization_linear_solve),
        last_linear_solve=encode(result.last_linear_solve),
        diagnostics=encode(result.diagnostics),
    )


def worker(spec_path):
    """All numerical imports happen after affinity and source selection."""
    spec = json.loads(spec_path.read_text())
    source, output = Path(spec["source"]), Path(spec["output"])
    os.sched_setaffinity(0, {spec["cpu"]})
    sys.path.insert(0, str(source))
    import glqp
    from glqp import GLQP, SolverSettings

    if not Path(glqp.__file__).resolve().is_relative_to(source):
        raise RuntimeError(f"Imported the wrong GLQP: {glqp.__file__}")
    if any(job.get("mm") for job in spec["jobs"]):
        sys.path.insert(0, str(Path(spec["benchmark_root"]) / "src"))
        from glqp_bench.qp import load_problem
    write_json(output / "environment.json", dict(
        python=sys.version, executable=sys.executable, imported_glqp=str(glqp.__file__),
        affinity=sorted(os.sched_getaffinity(0)),
        numerical_packages=numerical_versions(),
        native_threads={key: os.environ.get(key) for key in THREAD_VARIABLES}))
    for index, job in enumerate(spec["jobs"]):
        try:
            if job.get("mm"):
                qp = load_problem(Path(spec["mm_directory"]) / (job["problem"] + ".mat"))
                problem = dict(Q=qp.P, b=-qp.q, C=qp.G, c=qp.h, E=qp.A, e=qp.b)
                starts = {}
            else:
                problem, starts = small_problem(job["problem"])
            options = {}
            if job["settings_mode"] != "implicit":
                options["settings"] = SolverSettings(**job["settings"])
            _, result = GLQP(**problem).solve(verbose=False, **starts, **options)
            outcome = collect_result(result)
        except Exception as error:
            outcome = dict(worker_error=dict(type=type(error).__name__, message=str(error),
                                             traceback=traceback.format_exc()))
        with (output / f"{index:04d}.json.gz").open("wb") as stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
                compressed.write(json_bytes(outcome))
        print(json.dumps(dict(case=job["name"], termination=outcome.get("termination"),
                              worker_error=outcome.get("worker_error"))), flush=True)


def difference_paths(first, second, path="$", limit=12):
    """Bound the human-readable mismatch report; full outcomes remain on disk."""
    if first == second:
        return []
    if type(first) is not type(second):
        return [path]
    differences = []
    if isinstance(first, dict):
        for key in dict.fromkeys([*first, *second]):
            child = f"{path}.{key}"
            differences.extend([child] if key not in first or key not in second else
                               difference_paths(first[key], second[key], child, limit))
            if len(differences) >= limit:
                break
    elif isinstance(first, list):
        if len(first) != len(second):
            differences.append(path + ".length")
        for index, (left, right) in enumerate(zip(first, second)):
            differences.extend(difference_paths(left, right, f"{path}[{index}]", limit))
            if len(differences) >= limit:
                break
    else:
        differences.append(path)
    return differences[:limit]


def compare_results(output, jobs):
    comparisons = []
    for index, job in enumerate(jobs):
        paths = {role: output / role / f"{index:04d}.json.gz" for role in ("reference", "candidate")}
        values = {role: json.loads(gzip.decompress(path.read_bytes())) for role, path in paths.items()}
        first, second = values["reference"], values["candidate"]
        complete = not any("worker_error" in value or value.get("termination", {}).get("exception") is not None
                           for value in values.values())
        budget_hit = any("time" in str(value.get("termination", {}).get("tag", "")).lower()
                         for value in values.values())
        comparisons.append(dict(
            case=job["name"], equal=first == second, conclusive=complete and not budget_hit,
            components={key: first.get(key) == second.get(key) for key in dict.fromkeys([*first, *second])},
            differences=difference_paths(first, second),
            reference_termination=first.get("termination"), candidate_termination=second.get("termination"),
            artifact_sha256={role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in paths.items()},
            component_sha256={role: {key: digest(item) for key, item in value.items()}
                              for role, value in values.items()},
        ))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--suite", choices=("small", "mm", "all"), default="small")
    parser.add_argument("--diagnostics", choices=("off", "on", "both"), default="both",
                        help="Custom small cases; public defaults always retain diagnostics off")
    parser.add_argument("--mm-diagnostics", action="store_true", help="Include full MM diagnostics (large)")
    parser.add_argument("--benchmark-root", type=Path, default=Path(__file__).resolve().parents[2] / "glqp-bench")
    parser.add_argument("--mm-directory", type=Path)
    parser.add_argument("--case", action="append", help="Select case names by shell-style pattern; repeatable")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    jobs = make_jobs(args)
    if args.list_cases:
        print("\n".join(job["name"] for job in jobs))
        return 0
    if args.cpu not in os.sched_getaffinity(0):
        parser.error(f"CPU {args.cpu} is outside the available affinity")
    sources = {role: getattr(args, role).resolve() for role in ("reference", "candidate")}
    for source in sources.values():
        if not (source / "glqp" / "__init__.py").is_file():
            parser.error(f"Expected a source tree containing glqp/__init__.py: {source}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"Output directory must be empty: {output}")
    benchmark = args.benchmark_root.resolve()
    mm_directory = (args.mm_directory or benchmark / "mm_problems" / "MAT_Files").resolve()
    auxiliary = {}
    if any(job.get("mm") for job in jobs):
        paths = [benchmark / "src" / "glqp_bench" / "qp.py"]
        paths += [mm_directory / (name + ".mat") for name in sorted(
            {job["problem"] for job in jobs if job.get("mm")})]
        auxiliary = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    manifests = {role: source_manifest(source) for role, source in sources.items()}
    write_json(output / "inputs.json", dict(jobs=jobs, source_paths={key: str(value) for key, value in sources.items()},
                                           source_sha256=manifests, auxiliary_sha256=auxiliary))
    environment = dict(os.environ, **{key: "1" for key in THREAD_VARIABLES},
                       MKL_DYNAMIC="FALSE", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    for role, source in sources.items():
        destination = output / role
        destination.mkdir()
        spec = dict(source=str(source), output=str(destination), cpu=args.cpu, jobs=jobs,
                    benchmark_root=str(benchmark), mm_directory=str(mm_directory))
        spec_path = output / f"{role}-worker.json"
        write_json(spec_path, spec)
        print(f"Checking {role}: {len(jobs)} untimed cases on CPU {args.cpu}", flush=True)
        with (destination / "worker.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path)],
                           cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    comparisons = compare_results(output, jobs)
    unchanged = manifests == {role: source_manifest(source) for role, source in sources.items()}
    auxiliary_unchanged = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value
                              for path, value in auxiliary.items())
    write_json(output / "comparisons.json", comparisons)
    summary = dict(purpose="Exact untimed correctness parity; no performance claims", cpu=args.cpu,
                   native_threads=1, cases=len(jobs), equal=sum(row["equal"] for row in comparisons),
                   conclusive=sum(row["conclusive"] for row in comparisons),
                   source_files_unchanged=unchanged, auxiliary_files_unchanged=auxiliary_unchanged,
                   tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   source_sha256=manifests, comparisons_sha256=digest(comparisons),
                   all_equal=unchanged and auxiliary_unchanged and all(
                       row["equal"] and row["conclusive"] for row in comparisons),
                   failures=[row for row in comparisons if not row["equal"] or not row["conclusive"]])
    write_json(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("all_equal", "cases", "equal", "conclusive")}, indent=2))
    return 0 if summary["all_equal"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]))
    else:
        raise SystemExit(main())
