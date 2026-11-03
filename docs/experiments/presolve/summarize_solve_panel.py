"""Summarize completed matched QP campaigns; never launch a solver.

Run only after the machine-wide timing lock is released. Wall comparisons use
problems accepted on every repetition of both variants, and report coverage
separately so timeouts or inaccurate solutions cannot disappear into a mean.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


VARIANTS = ("glqp_current", "glqp_base", "glqp_candidate")
PAIRS = (("glqp_current", "glqp_base"), ("glqp_current", "glqp_candidate"),
         ("glqp_base", "glqp_candidate"))
RESIDUALS = ("primal_residual_scaled", "stationarity_residual_scaled",
             "complementarity_residual_scaled", "dual_sign_violation_scaled")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(value) for value in values)) if values else None


def summarize_variant(records, expected_repeats, time_limit):
    repeats = sorted(record["repeat"] for _, record in records)
    if repeats != list(range(expected_repeats)):
        raise ValueError(f"Missing or duplicated repetitions: {repeats}")
    accepted = [record for _, record in records
                if record["status"] == "solved" and record["validation"]["accepted"]
                and record["timings"]["wall_seconds"] <= time_limit]
    result = {
        "records": [{"path": str(path), "sha256": digest(path)} for path, _ in records],
        "status_counts": dict(Counter(record["status"] for _, record in records)),
        "accurate_over_budget": sum(record["status"] == "solved" and record["validation"]["accepted"]
                                    and record["timings"]["wall_seconds"] > time_limit for _, record in records),
        "accepted_repeats": len(accepted), "all_accepted": len(accepted) == expected_repeats,
        "wall_seconds": [record["timings"]["wall_seconds"] for _, record in records],
        "iterations": [record["diagnostics"].get("iterations") for _, record in records],
        "solver_statuses": [record.get("solver_status") for _, record in records],
        "failure_reasons": [record["validation"].get("failure_reasons", []) for _, record in records],
    }
    if result["all_accepted"]:
        for name in ("wall_seconds", "setup_seconds", "solve_seconds"):
            result["median_" + name] = statistics.median(record["timings"][name] for _, record in records)
        result["max_residuals"] = {name: max(record["validation"][name] for record in accepted)
                                   for name in RESIDUALS}
    presolves = [record["diagnostics"].get("presolve") for _, record in records]
    if all(presolves):
        structural = [{"status": item["status"], "original_dimensions": item["original_dimensions"],
                       "reduced_dimensions": item["reduced_dimensions"],
                       "stats": {name: value for name, value in item["stats"].items()
                                 if not name.endswith("_seconds")}}
                      for item in presolves]
        if any(item != structural[0] for item in structural[1:]):
            raise ValueError("Presolve structure changed between repetitions")
        result["presolve"] = structural[0]
        result["presolve"].update({"median_" + name: statistics.median(item[name] for item in presolves)
                                  for name in ("presolve_seconds", "postsolve_seconds")})
        result["presolve"].update({"median_" + name: statistics.median(item["stats"][name] for item in presolves)
                                  for name in ("pack_seconds", "native_seconds", "total_seconds")})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--report", type=Path, help="approved benchmark report with artifact/isolation audits")
    mode.add_argument("--provisional", action="store_true", help="inspect raw outcomes before isolation approval; do not publish timings")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Only completed campaigns may be summarized")
    report = args.report.resolve() if args.report else None
    if report:
        approved = json.loads((report / "summary.json").read_text())
        if Path(approved["run"]).resolve() != run or approved["isolation"]["status"] not in ("clean", "clean_selected_waves"):
            raise ValueError("Require the matching approved isolation report")
        artifact_audit = json.loads((report / "artifact-audit.json").read_text())
        if not artifact_audit or not all(audit["passed"] for audit in artifact_audit.values()):
            raise ValueError("Original-coordinate artifact audit did not pass")
    else:
        subrun_manifest = json.loads((run / manifest["subrun"] / "manifest.json").read_text())
        approved = {"problem_count": len(subrun_manifest["problem_ids"]), "isolation": {"status": "unreviewed"}}
    selection_path = report / "wave-selection.json" if report else None
    if selection_path and selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        if not selection["complete"]:
            raise ValueError("Wave selection is incomplete")
        selected_sources = [(Path(wave["campaign_root"]), set(wave["problem_ids"]))
                            for wave in selection["selected_waves"]]
    else:
        selected_sources = [(run, None)]
    grouped = defaultdict(lambda: defaultdict(list))
    source_cache = {}
    for source, selected_names in selected_sources:
        if source not in source_cache:
            source_manifest = json.loads((source / "manifest.json").read_text())
            if source_manifest["status"] != "completed" or source_manifest["glqp"]["commit"] != manifest["glqp"]["commit"]:
                raise ValueError("Unfinished or mismatched numerical source in selected campaign")
            records_root = source / source_manifest["subrun"] / "records"
            source_cache[source] = [(path, json.loads(path.read_text())) for variant in VARIANTS
                                    for path in sorted((records_root / variant).glob("*.json"))]
        for path, record in source_cache[source]:
            if selected_names is None or record["problem_id"] in selected_names:
                grouped[record["problem_id"]][record["solver"]].append((path, record))
    if not grouped:
        raise ValueError("No GLQP records found")
    if manifest.get("problem_ids") and set(grouped) != set(manifest["problem_ids"]):
        raise ValueError("Problem records do not match the manifest selection")
    problems = []
    for name, variants in sorted(grouped.items()):
        if set(variants) != set(VARIANTS):
            raise ValueError(f"Incomplete variant set for {name}")
        problems.append({"name": name, "variants": {
            variant: summarize_variant(variants[variant], manifest["repeats"], manifest["time_limit"]) for variant in VARIANTS}})
    if len(problems) != approved["problem_count"]:
        raise ValueError("Approved report and detailed records have different problem counts")
    summary = {"problems": len(problems), "repeats": manifest["repeats"], "variants": {}, "pairs": {}}
    for variant in VARIANTS:
        accepted = [row["variants"][variant] for row in problems if row["variants"][variant]["all_accepted"]]
        counts = Counter()
        for row in problems:
            counts.update(row["variants"][variant]["status_counts"])
        summary["variants"][variant] = {
            "status_counts": dict(counts), "all_repetitions_accepted_problems": len(accepted),
            "accurate_over_budget": sum(row["variants"][variant]["accurate_over_budget"] for row in problems),
            "max_residuals": {name: max((row["max_residuals"][name] for row in accepted), default=None)
                              for name in RESIDUALS}}
        reported = approved.get("solver_summary", {}).get(variant)
        if reported and reported["accepted_problems"] != len(accepted):
            raise ValueError(f"Acceptance count disagrees with approved report: {variant}")
    for before, after in PAIRS:
        common = [row for row in problems if all(row["variants"][v]["all_accepted"] for v in (before, after))]
        ratios = {row["name"]: row["variants"][after]["median_wall_seconds"] /
                  row["variants"][before]["median_wall_seconds"] for row in common}
        summary["pairs"][before + "__" + after] = {
            "common_accepted_problems": len(common), "wall_geomean_ratio": geometric_mean(list(ratios.values())),
            "wall_median_ratio": statistics.median(ratios.values()) if ratios else None,
            "summed_common_wall_ratio": sum(row["variants"][after]["median_wall_seconds"] for row in common) /
                                        sum(row["variants"][before]["median_wall_seconds"] for row in common) if common else None,
            "newly_accepted": [row["name"] for row in problems if row["variants"][after]["all_accepted"]
                               and not row["variants"][before]["all_accepted"]],
            "lost_accepted": [row["name"] for row in problems if row["variants"][before]["all_accepted"]
                              and not row["variants"][after]["all_accepted"]],
            "ratios": ratios,
            "below_0_95": [name for name, ratio in ratios.items() if ratio < .95],
            "above_1_05": [name for name, ratio in ratios.items() if ratio > 1.05],
        }
    summary["scalar_groups"] = {}
    scalar_common = [row for row in problems if all(row["variants"][v]["all_accepted"]
                                                    for v in ("glqp_base", "glqp_candidate"))]
    for name, changed in (("scalar_changed", True), ("no_scalar_reduction", False)):
        rows = [row for row in scalar_common if bool(row["variants"]["glqp_candidate"]["presolve"]
                ["stats"]["separable_variables"]) == changed]
        group = {"problems": len(rows), "names": [row["name"] for row in rows],
                 "wall_geomean_ratio": geometric_mean([row["variants"]["glqp_candidate"]["median_wall_seconds"] /
                                                       row["variants"]["glqp_base"]["median_wall_seconds"] for row in rows])}
        for variant in ("glqp_base", "glqp_candidate"):
            items = [row["variants"][variant] for row in rows]
            if items:
                group[variant] = {"median_" + timing: statistics.median(item["presolve"]["median_" + timing] for item in items)
                                  for timing in ("presolve_seconds", "postsolve_seconds", "native_seconds")}
                group[variant]["median_presolve_postsolve_fraction"] = statistics.median(
                    (item["presolve"]["median_presolve_seconds"] + item["presolve"]["median_postsolve_seconds"]) /
                    item["median_wall_seconds"] for item in items)
        summary["scalar_groups"][name] = group
    evidence = {"kind": "provisional raw presolve analysis; timings unapproved" if args.provisional else
                        "matched complete-solve presolve analysis", "run": str(run),
                "manifest_sha256": digest(manifest_path), "script_sha256": digest(Path(__file__)),
                "approved_report": str(report) if report else None,
                "approved_summary_sha256": digest(report / "summary.json") if report else None,
                "artifact_audit_sha256": digest(report / "artifact-audit.json") if report else None,
                "isolation": approved["isolation"],
                "wave_selection_sha256": digest(selection_path) if selection_path and selection_path.is_file() else None,
                "solver_commit": manifest["glqp"]["commit"], "benchmark_commit": manifest["benchmark"]["commit"],
                "execution": manifest["execution"], "solver_options": manifest["solver_options"],
                "solver_tolerance": manifest["solver_tolerance"], "validation_tolerance": manifest["validation_tolerance"],
                "time_limit": manifest["time_limit"], "summary": summary, "problems": problems}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({**summary,
                      "pairs": {name: {key: value for key, value in pair.items()
                                       if key not in ("ratios", "below_0_95", "above_1_05")}
                                for name, pair in summary["pairs"].items()},
                      "scalar_groups": {name: {key: value for key, value in group.items() if key != "names"}
                                        for name, group in summary["scalar_groups"].items()}}, indent=2))


if __name__ == "__main__":
    main()
