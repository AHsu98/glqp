"""The standalone file interface exposes the same opt-in structural passes."""
import os
from pathlib import Path
import shlex
import subprocess

import pytest


@pytest.fixture(scope="module")
def presolve_cli(tmp_path_factory):
    native = Path(__file__).resolve().parents[1] / "glqp/_native"
    binary = tmp_path_factory.mktemp("presolve-cli") / "glqp-presolve"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    subprocess.run([*compiler, "-std=c++17", "-O0", "-ffp-contract=off",
                    str(native / "presolve.cpp"), str(native / "presolve_cli.cpp"),
                    "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


def run_cli(binary, args=(), data=""):
    return subprocess.run([str(binary), *args], input=data, capture_output=True, text=True)


def test_scalar_minimization_is_explicit_and_updates_objective_constant(presolve_cli):
    data = "GQP1\n1 0 0 5\n-2\n1\n0 0 2\n0\n0\n"
    default = run_cli(presolve_cli, data=data)
    scalar = run_cli(presolve_cli, ["--separable-quadratics", "-"], data)
    assert default.returncode == scalar.returncode == 0
    assert default.stdout.splitlines()[:2] == ["GQP1", "1 0 0 5"]
    assert scalar.stdout.splitlines()[:2] == ["GQP1", "0 0 0 4"]


def test_equality_pass_can_be_disabled_for_the_standalone_client(presolve_cli, tmp_path):
    data = "GQP1\n1 1 0 0\n0\n2\n1\n0 0 2\n1\n0 0 1\n0\n"
    path = tmp_path / "fixed.gqp"
    path.write_text(data)
    fixed = run_cli(presolve_cli, [str(path)])
    retained = run_cli(presolve_cli, ["--no-singleton-equalities", str(path)])
    assert fixed.returncode == retained.returncode == 0
    assert fixed.stdout.splitlines()[:2] == ["GQP1", "0 0 0 4"]
    assert retained.stdout.splitlines()[:2] == ["GQP1", "1 1 0 0"]


def test_local_fallback_is_an_explicit_cli_option(presolve_cli):
    data = "GQP1\n2 2 0 0\n0 0\n1 1.00000000000001\n2\n0 0 2\n1 1 2\n2\n0 0 1\n0 1 1\n0\n"
    default = run_cli(presolve_cli, ["--separable-quadratics"], data)
    local = run_cli(presolve_cli, ["--separable-quadratics", "--local-fallback"], data)
    assert default.returncode == 3
    assert local.returncode == 0
    assert local.stdout.splitlines()[:2] == ["GQP1", "1 2 0 0"]


def test_fixed_and_redundant_bounds_have_independent_cli_switches(presolve_cli):
    fixed_pair = "GQP1\n1 0 2 0\n0\n-1 1\n1\n0 0 2\n0\n2\n0 0 -1\n0 1 1\n"
    fixed = run_cli(presolve_cli, data=fixed_pair)
    retained = run_cli(presolve_cli, ["--no-fixed-bounds"], fixed_pair)
    assert fixed.returncode == retained.returncode == 0
    assert fixed.stdout.splitlines()[:2] == ["GQP1", "0 0 0 1"]
    assert retained.stdout.splitlines()[:2] == ["GQP1", "1 0 2 0"]
    duplicate = "GQP1\n1 0 2 0\n0\n1 1\n1\n0 0 2\n0\n2\n0 0 1\n0 1 1\n"
    pruned = run_cli(presolve_cli, data=duplicate)
    retained = run_cli(presolve_cli, ["--no-redundant-bounds"], duplicate)
    assert pruned.returncode == retained.returncode == 0
    assert pruned.stdout.splitlines()[:2] == ["GQP1", "1 0 1 0"]
    assert retained.stdout.splitlines()[:2] == ["GQP1", "1 0 2 0"]


@pytest.mark.parametrize("args", [
    ["--unknown"], ["--infeasibility-guard"], ["--infeasibility-guard", "1e-12junk"],
    ["--infeasibility-guard", "-1"], ["--infeasibility-guard", "nan"], ["-", "-"],
])
def test_invalid_cli_options_fail_cleanly(presolve_cli, args):
    valid = "GQP1\n0 0 0 0\n0\n0\n0\n"
    result = run_cli(presolve_cli, args, valid)
    assert result.returncode == 4
    assert result.stderr.strip()
    assert not result.stdout
