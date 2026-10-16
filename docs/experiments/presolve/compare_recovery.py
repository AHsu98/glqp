"""Focused alternating original-coordinate recovery timing diagnostic."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from glqp import presolve as candidate
from glqp_bench.qp import QP, load_problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=101)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("presolve_reference", args.reference / "glqp/presolve.py")
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    versions = {"reference": reference, "candidate": candidate}
    evidence = {"kind": "recovery-only diagnostic, not solve speedup", "reference_commit": args.reference_commit,
                "builds": {name: module.prepare_native() for name, module in versions.items()},
                "affinity": sorted(os.sched_getaffinity(0)), "repeats": args.repeats, "problems": []}
    inputs = ROOT.parent / "glqp-bench/mm_problems/MAT_Files"
    problems = [load_problem(inputs / f"{name}.mat") for name in ("QPCBOEI2", "QISRAEL", "QSHELL")]
    n = 100000
    problems.append(QP(sparse.diags([-np.ones(n-1), 4*np.ones(n), -np.ones(n-1)], [-1, 0, 1]),
                       np.zeros(n), sparse.csc_matrix((0, n)), np.empty(0),
                       sparse.csc_matrix(([1., 1.], ([0, 1], [0, 0])), shape=(2, n)),
                       np.array([1., 2.]), "synthetic-tridiagonal-100k"))
    rng = np.random.default_rng(64)
    for qp in problems:
        result = {name: module.presolve_qp(qp.P, qp.q, qp.A, qp.b, qp.G, qp.h)
                  for name, module in versions.items()}
        new = result["candidate"]
        x, y, z = (rng.standard_normal(size) for size in (new.n, new.A.shape[0], new.G.shape[0]))
        samples = {name: [] for name in versions}
        for repeat in range(args.repeats + 1):
            outputs = {}
            for name in (list(versions) if repeat % 2 else list(reversed(versions))):
                start = time.perf_counter()
                outputs[name] = result[name].recover(x, y, z)
                elapsed = time.perf_counter() - start
                if repeat:
                    samples[name].append(elapsed)
            for old, current in zip(outputs["reference"], outputs["candidate"]):
                np.testing.assert_array_equal(old, current)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        evidence["problems"].append({"name": qp.name, "fixed_variables": new.stats["fixed_variables"],
                                     "samples": samples, "medians": medians,
                                     "candidate_reference_ratio": medians["candidate"] / medians["reference"],
                                     "identical_recovery": True})
        for value in result.values():
            value.close()
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps([{key: row[key] for key in ("name", "candidate_reference_ratio", "identical_recovery")}
                      for row in evidence["problems"]], indent=2))


if __name__ == "__main__":
    main()
