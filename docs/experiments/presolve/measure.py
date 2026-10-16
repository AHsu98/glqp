"""Measure presolve alone on bundled inputs; deliberately does not time solves.

Run from the solver checkout with the sibling benchmark package installed:
  taskset -c 8 python docs/experiments/presolve/measure.py --output result.json
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

SOLVER_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOLVER_ROOT))
from glqp.presolve import PresolveOptions, prepare_native, presolve_qp
from glqp_bench.qp import load_problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=SOLVER_ROOT.parent / "glqp-bench/mm_problems/MAT_Files")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--local-fallback", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    options = PresolveOptions(local_fallback=args.local_fallback)
    evidence = {"kind": "presolve-only diagnostic, not solver performance", "native": prepare_native(),
                "options": asdict(options), "affinity": sorted(os.sched_getaffinity(0)),
                "repeats": args.repeats, "problems": []}
    rng = np.random.default_rng(20261016)
    for path in sorted(args.inputs.glob("*.mat")):
        qp = load_problem(path)
        samples = []
        for _ in range(args.repeats):
            with presolve_qp(qp.P, qp.q, qp.A, qp.b, qp.G, qp.h, options=options) as reduced:
                samples.append(dict(reduced.stats))
                x = rng.standard_normal(reduced.n)
                original_x, _, _ = reduced.recover(x)
                original_objective = .5 * original_x @ (qp.P @ original_x) + qp.q @ original_x
                reduced_objective = .5 * x @ (reduced.Q @ x) + reduced.q @ x + reduced.objective_offset
                if not np.isclose(original_objective, reduced_objective, rtol=2e-12, atol=1e-8):
                    raise AssertionError(f"Objective transformation mismatch: {qp.name}")
                np.testing.assert_allclose((qp.A @ original_x - qp.b)[reduced.equality_map],
                                           reduced.A @ x - reduced.b, rtol=2e-12, atol=1e-8)
                np.testing.assert_allclose((qp.G @ original_x - qp.h)[reduced.inequality_map],
                                           reduced.G @ x - reduced.h, rtol=2e-12, atol=1e-8)
                item = {"name": qp.name, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "original": [qp.n, qp.p, qp.m, qp.P.nnz + qp.A.nnz + qp.G.nnz],
                        "reduced": [reduced.n, reduced.A.shape[0], reduced.G.shape[0],
                                    reduced.Q.nnz + reduced.A.nnz + reduced.G.nnz],
                        "status": reduced.status, "message": reduced.message}
        item["samples"] = samples
        item["median_seconds"] = {name: statistics.median(sample[name] for sample in samples)
                                  for name in ("pack_seconds", "native_seconds", "total_seconds")}
        evidence["problems"].append(item)
    evidence["summary"] = {
        "problems": len(evidence["problems"]),
        "reduced": sum(p["status"] == "reduced" for p in evidence["problems"]),
        "inconclusive": sum(p["status"] == "inconclusive" for p in evidence["problems"]),
        "fixed_variables": sum(p["original"][0] - p["reduced"][0] for p in evidence["problems"]),
        "removed_equalities": sum(p["original"][1] - p["reduced"][1] for p in evidence["problems"]),
        "removed_inequalities": sum(p["original"][2] - p["reduced"][2] for p in evidence["problems"]),
        "median_problem_seconds": statistics.median(p["median_seconds"]["total_seconds"] for p in evidence["problems"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence["summary"], indent=2))


if __name__ == "__main__":
    main()
