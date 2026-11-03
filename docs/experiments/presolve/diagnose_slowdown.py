"""Frozen, cold-worker pass ablations for a presolve convergence regression."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("full", "keep_bounds", "keep_equalities"))
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    reference = workspace / "glqp-bench/runs/research2-20261016-presolve-full"
    frozen = output / "source/benchmark"
    source = output / "source/glqp"
    stages = {
        "full": {"problems": ["DUAL2", "DUAL4", "QFFFFF80"],
                 "options": {"glqp_candidate": {"presolve": {"enabled": True}}}},
        "keep_bounds": {"problems": ["QFFFFF80"],
                        "options": {"glqp_base": {"presolve": {"enabled": True}},
                                    "glqp_candidate": {"presolve": {"enabled": True, "redundant_bounds": False}}}},
        "keep_equalities": {"problems": ["QFFFFF80"],
                            "options": {"glqp_candidate": {"presolve": {"enabled": True, "singleton_equalities": False}}}},
    }
    if args.stage:
        sys.path.insert(0, str(frozen / "src"))
        from glqp_bench.execution import resolve_execution
        from glqp_bench.runner import run_suite
        stage = stages[args.stage]
        run_suite(frozen / "runs" / args.stage, ["glqp_base", "glqp_candidate"],
                  stage["problems"], repeats=3, time_limit=60, tol=1e-7, validation_tol=1e-6,
                  glqp_sources={key: source for key in ("glqp_base", "glqp_candidate")},
                  settings_profile="benchmark-v1", execution=resolve_execution(cpu=8, root=frozen),
                  solver_options=stage["options"])
        return

    sys.path.insert(0, str(reference / "source/benchmark/src"))
    from glqp_bench.campaign import BENCHMARK_PATHS, snapshot_git, verify_snapshot
    output.mkdir(parents=True, exist_ok=False)
    copied_driver = output / "driver.py"
    copied_driver.write_bytes(Path(__file__).read_bytes())
    reference_manifest = json.loads((reference / "manifest.json").read_text())
    benchmark = snapshot_git(workspace / "glqp-bench", reference_manifest["benchmark"]["commit"],
                             frozen, BENCHMARK_PATHS)
    solver = snapshot_git(workspace / "glqp", reference_manifest["baseline"]["commit"], source)
    evidence = {"kind": "Focused presolve pass ablation; cold workers, diagnostic CPU 8",
                "reference_campaign": str(reference), "reference_manifest_sha256": digest(reference / "manifest.json"),
                "driver_sha256": digest(copied_driver), "benchmark": benchmark, "solver": solver,
                "cpu": 8, "threads": 1, "repeats": 3, "stages": stages, "results": {}}
    (output / "diagnostic.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for name in stages:
        subprocess.run([sys.executable, str(copied_driver), "--workspace", str(workspace),
                        "--output", str(output), "--stage", name], check=True, cwd=frozen)
        rows = {}
        for path in sorted((frozen / "runs" / name / "records").glob("*/*.json")):
            record = json.loads(path.read_text())
            key = record["problem_id"] + "/" + record["solver"]
            rows.setdefault(key, []).append({"record": str(path.relative_to(output)), "record_sha256": digest(path),
                "status": record["status"], "wall_seconds": record["timings"]["wall_seconds"],
                "iterations": record["diagnostics"].get("iterations"), "validation": record["validation"],
                "presolve": record["diagnostics"].get("presolve")})
        evidence["results"][name] = {key: {"observations": values,
            "median_wall_seconds": statistics.median(item["wall_seconds"] for item in values),
            "iterations": [item["iterations"] for item in values]} for key, values in rows.items()}
        (output / "diagnostic.json").write_text(json.dumps(evidence, indent=2) + "\n")
    verify_snapshot(frozen, benchmark)
    verify_snapshot(source, solver)
    evidence["source_integrity"] = "verified"
    (output / "diagnostic.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for name, rows in evidence["results"].items():
        print(name, {key: {field: item[field] for field in ("median_wall_seconds", "iterations")}
                     for key, item in rows.items()})


if __name__ == "__main__":
    main()
