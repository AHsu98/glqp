"""Read-only structural survey; row hashing here is diagnostic Python code."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from glqp.presolve import prepare_native, presolve_qp
from glqp_bench.qp import load_problem


def survey(A, b, G, h):
    equality = {}
    inequality = {}
    counts = {"duplicate_equalities": 0, "same_lhs_different_eq_rhs": 0,
              "dominated_duplicate_inequalities": 0, "inequalities_implied_by_equalities": 0}
    for matrix, rhs, is_eq in ((A, b, True), (G, h, False)):
        csr = matrix.tocsr()
        for row in np.flatnonzero(np.diff(csr.indptr) > 1):
            sl = slice(csr.indptr[row], csr.indptr[row + 1])
            key = csr.indices[sl].astype(np.int64, copy=False).tobytes(), csr.data[sl].tobytes()
            if is_eq:
                if key in equality:
                    counts["duplicate_equalities" if equality[key] == rhs[row]
                           else "same_lhs_different_eq_rhs"] += 1
                else:
                    equality[key] = rhs[row]
            elif key in equality and rhs[row] >= equality[key]:
                counts["inequalities_implied_by_equalities"] += 1
            elif key in inequality:
                counts["dominated_duplicate_inequalities"] += 1
                inequality[key] = min(inequality[key], rhs[row])
            else:
                inequality[key] = rhs[row]
    return counts


def main():
    output = Path(sys.argv[1])
    evidence = {"kind": "structural opportunity survey, not timing", "affinity": sorted(os.sched_getaffinity(0)),
                "native": prepare_native(), "problems": []}
    for path in sorted((ROOT.parent / "glqp-bench/mm_problems/MAT_Files").glob("*.mat")):
        problem = load_problem(path)
        with presolve_qp(problem.P, problem.q, problem.A, problem.b, problem.G, problem.h) as reduced:
            row = {"name": problem.name, "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "status": reduced.status, "message": reduced.message,
                   "original": [problem.n, problem.p, problem.m],
                   "reduced": [reduced.n, reduced.A.shape[0], reduced.G.shape[0]],
                   "original_opportunities": survey(problem.A, problem.b, problem.G, problem.h),
                   "reduced_opportunities": survey(reduced.A, reduced.b, reduced.G, reduced.h)}
            evidence["problems"].append(row)
    summary = {}
    for which in ("original_opportunities", "reduced_opportunities"):
        summary[which] = {key: sum(row[which][key] for row in evidence["problems"])
                          for key in evidence["problems"][0][which]}
    evidence["summary"] = summary
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    for row in evidence["problems"]:
        if any(row["reduced_opportunities"].values()) or row["status"] == "inconclusive":
            print(row["name"], row["status"], row["reduced_opportunities"])


if __name__ == "__main__":
    main()
