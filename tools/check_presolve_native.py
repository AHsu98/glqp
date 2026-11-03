"""Compile and run standalone native presolver checks with ASan and UBSan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON validation evidence path")
    args = parser.parse_args()
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    sources = ["glqp/_native/presolve.cpp", "glqp/_native/presolve.h", "tests/native_presolve_sanitizer.cpp"]
    flags = ["-std=c++17", "-O1", "-g", "-ffp-contract=off", "-Wall", "-Wextra", "-Wpedantic",
             "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-fno-sanitize-recover=all"]
    evidence = {
        "kind": "standalone C ABI sanitizer validation",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "compiler": compiler,
        "compiler_version": subprocess.check_output([*compiler, "--version"], text=True),
        "flags": flags,
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {"ASAN_OPTIONS": "detect_leaks=1:halt_on_error=1",
                        "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"},
    }
    with tempfile.TemporaryDirectory(prefix="glqp-presolve-sanitizer-") as build:
        binary = Path(build) / "native-presolve-test"
        command = [*compiler, *flags, "-I", str(ROOT / "glqp/_native"),
                   str(ROOT / sources[0]), str(ROOT / sources[2]), "-o", str(binary)]
        started = time.perf_counter()
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=120)
        evidence["compile"] = {"returncode": compiled.returncode, "stdout": compiled.stdout,
                               "stderr": compiled.stderr, "seconds": time.perf_counter() - started}
        exit_code = compiled.returncode
        if compiled.returncode == 0:
            evidence["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
            started = time.perf_counter()
            checked = subprocess.run([str(binary)], env={**os.environ, **evidence["environment"]},
                                     capture_output=True, text=True, timeout=30)
            evidence["run"] = {"returncode": checked.returncode, "stdout": checked.stdout,
                               "stderr": checked.stderr, "seconds": time.perf_counter() - started}
            exit_code = checked.returncode
        evidence["passed"] = exit_code == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(evidence.get("run", evidence["compile"])["stdout"], end="")
    print(evidence.get("run", evidence["compile"])["stderr"], end="")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
