"""Small algebra checks and retained-history extraction for the research memo.

This does not run an optimization problem or report performance. By default it
checks the current KKT equations. --include-local-history also reads the ignored
diagnostic artifacts retained in the shared workspace on 2026-10-16.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.sparse import csc_array


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO))
from glqp.kkt import KKTWorkspace  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(WORKSPACE))


def maxnorm(value):
    return float(np.max(np.abs(value), initial=0.0))


def algebra_check():
    H = csc_array([[3.0, 0.25], [0.25, 2.0]])
    C = csc_array([[1.0, -0.4], [0.3, 1.2]])
    E = csc_array([[0.5, 1.0]])
    s, y = np.array([0.2, 1.3]), np.array([0.4, 0.8])
    rd, rp, re = np.array([0.7, -0.2]), np.array([0.1, -0.3]), np.array([0.15])
    rc = s * y - 0.07
    workspace = KKTWorkspace(H, C, E)
    rows = []
    for rho in (0.0, 2e-8, 1e-2):
        G, rhs, w = workspace.assemble(H, y, s, rd, rp, rc, re, rho)
        reduced = np.linalg.solve(G.toarray(), rhs)
        dx, dy, dnu = reduced[:2], w * reduced[2:4], reduced[4:]
        ds = -rp - C @ dx + rho * dy
        defects = {
            "stationarity": H @ dx + C.T @ dy + E.T @ dnu + rd,
            "inequality": C @ dx + ds + rp,
            "equality": E @ dx + re,
            "complementarity": y * ds + s * dy + rc,
        }
        predicted = {
            "stationarity": -rho * dx,
            "inequality": rho * dy,
            "equality": rho * dnu,
            "complementarity": -rho * ds,
        }
        errors = {key: maxnorm(defects[key] - predicted[key]) for key in defects}
        assert max(errors.values()) < 2e-14, errors
        rows.append({
            "rho": rho,
            "requested_system_residual_inf": maxnorm(G @ reduced - rhs),
            "original_newton_defect_inf": {key: maxnorm(value) for key, value in defects.items()},
            "derived_defect_identity_error_inf": errors,
        })

    # Conditional repair example, not a claim that a solver trajectory reaches it.
    s, y = np.array([1e-6, 1.0]), np.array([1e-9, 1.0])
    mu_bar = float(s @ y / 2)
    target = 0.1 * mu_bar
    repaired_s = target / y[0]
    repaired_y = target / s[0]
    return {
        "dense_checks": rows,
        "conditional_repair_example": {
            "assumptions": "C=I, x=0, c=s; the tiny-step trigger is assumed, not simulated",
            "s": s.tolist(), "y": y.tolist(), "mu_bar": mu_bar,
            "target_product": target,
            "slack_only_new_s0": repaired_s,
            "slack_only_primal_residual_increase": repaired_s - s[0],
            "dual_only_new_y0": repaired_y,
            "dual_only_stationarity_residual_change": repaired_y - y[0],
        },
    }


def local_history():
    csv_path = WORKSPACE / ".conda/iteration-work-round5-shift/QPCBOEI2-candidate.csv"
    summary_path = csv_path.with_name("summary.json")
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    original = json.loads(summary_path.read_text())
    current_hashes = {name: sha256(REPO / name) for name in
                      ("glqp/solver.py", "glqp/kkt.py", "glqp/linear.py")}
    assert all(current_hashes[name] == original["sources"]["candidate"]["file_sha256"][name]
               for name in current_hashes), "History belongs to different numerical source"
    qpc = {
        "csv": relative(csv_path), "csv_sha256": sha256(csv_path),
        "source_manifest": relative(summary_path), "source_manifest_sha256": sha256(summary_path),
        "source_file_sha256": current_hashes,
        "settings": original["settings"], "original_affinity": original["affinity"],
        "iterations": len(rows),
        "linear_status_accurate": sum(row["lin_status"] == "accurate" for row in rows),
        "mu_equals_0_01_kkt": sum(math.isclose(float(row["mu"]), 0.01 * float(row["KKT_res"]),
                                              rel_tol=1e-12, abs_tol=0.0) for row in rows),
        "step_equals_0_9_power_7": sum(math.isclose(float(row["step"]), 0.9**7,
                                                   rel_tol=0.0, abs_tol=1e-15) for row in rows),
        "selected_rows": [{key: row[key] for key in
                            ("iter", "KKT_res", "dual_res", "cons_viol", "comp_res", "mu", "step")}
                          for row in (rows[i] for i in (0, 100, 1000, 2000, 3000, len(rows)-1))],
        "limitation": "Acceptance branch, repair size, and original Newton defect were not logged",
    }

    root = WORKSPACE / "glqp-bench/runs/research2-20261016-implementation-full/source/benchmark/runs"
    paths = sorted(root.glob("*/records/glqp_candidate/*.json"))
    assert len(paths) == 178, len(paths)
    failures = []
    accepted = 0
    for path in paths:
        record = json.loads(path.read_text())
        validation = record.get("validation") or {}
        if validation.get("accepted"):
            accepted += 1
            continue
        failures.append({
            "problem_id": record["problem_id"], "status": record["status"],
            "solver_status": record["solver_status"],
            "iterations": record["diagnostics"].get("iterations"),
            "failure_reasons": validation.get("failure_reasons"),
            "residuals": {key: value for key, value in validation.items()
                          if key.endswith("_scaled") or key == "tolerance"},
            "record": relative(path), "sha256": sha256(path),
        })
    return {"qpcboei2": qpc,
            "full_qp_candidate": {"total": len(paths), "accepted": accepted, "failures": failures}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-local-history", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "purpose": "Algebra and retained-history checks only; no performance measurements",
        "captured_date": "2026-10-16",
        "repo_head_at_capture": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "file_sha256": {name: sha256(REPO / name) for name in
                        ("glqp/solver.py", "glqp/kkt.py", "glqp/linear.py", "glqp/util.py")},
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "numpy_version": np.__version__,
        "algebra": algebra_check(),
    }
    if args.include_local_history:
        report["retained_history"] = local_history()
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
