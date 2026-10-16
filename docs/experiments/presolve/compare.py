"""Interleave old/new native presolvers and require identical outputs.

The reference directory must contain an exported glqp/presolve.py and its
glqp/_native sources. This tool never imports a reference solver algorithm.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from glqp import presolve as candidate
from glqp_bench.qp import load_problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("presolve_reference", args.reference / "glqp/presolve.py")
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    versions = {"reference": reference, "candidate": candidate}
    evidence = {"kind": "interleaved presolve-only diagnostic, not solve speedup",
                "reference_commit": args.reference_commit,
                "builds": {name: module.prepare_native() for name, module in versions.items()},
                "affinity": sorted(os.sched_getaffinity(0)), "repeats": args.repeats,
                "problems": []}
    rng = np.random.default_rng(200814)
    for index, path in enumerate(sorted((ROOT.parent / "glqp-bench/mm_problems/MAT_Files").glob("*.mat"))):
        qp = load_problem(path)
        samples = {name: [] for name in versions}
        for repeat in range(args.repeats):
            order = list(versions) if (index + repeat) % 2 == 0 else list(reversed(versions))
            result = {}
            for name in order:
                result[name] = versions[name].presolve_qp(qp.P, qp.q, qp.A, qp.b, qp.G, qp.h)
                samples[name].append({key: result[name].stats[key] for key in ("native_seconds", "total_seconds")})
            old, new = result["reference"], result["candidate"]
            assert old.status == new.status
            for name in ("Q", "A", "G"):
                before, after = getattr(old, name), getattr(new, name)
                assert before.shape == after.shape
                for buffer in ("indptr", "indices", "data"):
                    np.testing.assert_array_equal(getattr(before, buffer), getattr(after, buffer))
            for name in ("q", "b", "h", "variable_map", "equality_map", "inequality_map", "objective_offset"):
                np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
            x, y, z = (rng.standard_normal(size) for size in (new.n, new.A.shape[0], new.G.shape[0]))
            for before, after in zip(old.recover(x, y, z), new.recover(x, y, z)):
                np.testing.assert_array_equal(before, after)
            old.close()
            new.close()
        medians = {name: {key: statistics.median(row[key] for row in rows)
                          for key in ("native_seconds", "total_seconds")}
                   for name, rows in samples.items()}
        evidence["problems"].append({"name": qp.name, "samples": samples, "medians": medians,
                                      "output_and_recovery_identical": True})
    evidence["summary"] = {"problems": len(evidence["problems"]), "identical_outputs": True}
    for key in ("native_seconds", "total_seconds"):
        ratios = [row["medians"]["candidate"][key] / row["medians"]["reference"][key]
                  for row in evidence["problems"]]
        evidence["summary"][key + "_geomean_ratio"] = float(np.exp(np.mean(np.log(ratios))))
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence["summary"], indent=2))


if __name__ == "__main__":
    main()
