#!/usr/bin/env python3
"""Instrument a few GLM solves and inspect their actual Newton systems.

This is a numerical/work diagnostic, never a comparative timing benchmark.
Callback states are copied; matrix actions and dense references are evaluated
after the solve. Instrumentation and cProfile overhead remain in the solve.
Component times are inclusive and overlap: do not add them together.

The requested KKT matrix includes physical proximal coefficients. The separate
numerical shift only defines the factor/preconditioner; refinement targets the
requested matrix. A float64 weighted-design QR control avoids forming a Gram
matrix, but is not an exact/high-precision solve. Extended precision references
hold the solver's computed D fixed, distinguishing assembly error from error
in the scalar curvature formula. No conditioning inference uses entry ranges.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import cProfile
from dataclasses import asdict
import functools
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys
import time
from unittest.mock import patch


def inf_norm(value):
    import numpy as np
    return float(np.max(np.abs(value), initial=0.))


def finite_json(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")


def source_hashes(source):
    return {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((source / "glqp").rglob("*.py"))}


def matrix_stats(matrix):
    import numpy as np
    values = np.abs(matrix.data)
    positive = values[values > 0]
    return dict(shape=list(matrix.shape), stored_nnz=int(matrix.nnz),
                nonzero_values=int(np.count_nonzero(matrix.data)),
                density=float(matrix.nnz / max(1, matrix.shape[0] * matrix.shape[1])),
                array_bytes=sum(value.nbytes for value in (matrix.data, matrix.indices, matrix.indptr)),
                nonfinite_values=int(np.count_nonzero(~np.isfinite(matrix.data))),
                min_nonzero_abs=float(positive.min()) if len(positive) else None,
                max_abs=inf_norm(matrix.data))


def curvature_report(loss, z, weights, curvature):
    """Inspect tails separately from the accuracy of assembling A.T D A."""
    import numpy as np
    D = np.asarray(curvature)
    positive = D[(D > 0) & np.isfinite(D)]
    record = dict(size=len(D), positive_count=len(positive), zero_count=int(np.count_nonzero(D == 0)),
                  negative_count=int(np.count_nonzero(D < 0)),
                  nonfinite_count=int(np.count_nonzero(~np.isfinite(D))),
                  minimum_positive=float(positive.min()) if len(positive) else None,
                  maximum=inf_norm(D), predictor_min=float(np.min(z)), predictor_max=float(np.max(z)))
    if loss == "logistic":
        tail = np.exp(-np.abs(z[:len(weights)]))
        stable = weights * tail / (1 + tail)**2
        observed = D[:len(weights)]
        lost = (observed == 0) & (stable > 0)
        record["logistic_tail_control"] = dict(
            formula="w*exp(-abs(z))/(1+exp(-abs(z)))**2",
            absolute_difference_inf=inf_norm(observed - stable),
            positive_control_but_zero_solver_count=int(np.count_nonzero(lost)),
            lost_curvature_sum=float(np.sum(stable[lost])),
            max_relative_error_where_control_positive=inf_norm(
                (observed[stable > 0] - stable[stable > 0]) / stable[stable > 0]))
    return record


def action_comparison(A, Q, D, H, vectors, *, extended=False):
    """Compare identical vectors against assembled and factored actions."""
    import numpy as np
    if extended:
        dense_A, dense_Q = A.toarray().astype(np.longdouble), Q.toarray().astype(np.longdouble)
        extended_D = np.asarray(D, dtype=np.longdouble)
    records = []
    for name, vector in vectors.items():
        assembled = H @ vector
        factored = Q @ vector + A.T @ (D * (A @ vector))
        row = dict(vector=name, norm_inf=inf_norm(vector),
                   assembled_minus_factored_inf=inf_norm(assembled - factored),
                   relative_to_factored_inf=inf_norm(assembled - factored) / max(inf_norm(factored), np.finfo(float).tiny),
                   assembled_rayleigh=float(vector @ assembled), factored_rayleigh=float(vector @ factored))
        if extended:
            v = np.asarray(vector, dtype=np.longdouble)
            reference = dense_Q @ v + dense_A.T @ (extended_D * (dense_A @ v))
            scale = max(inf_norm(reference), np.finfo(float).tiny)
            row.update(extended_action_inf=inf_norm(reference),
                       extended_rayleigh=float(v @ reference),
                       assembled_minus_extended_inf=inf_norm(assembled.astype(np.longdouble) - reference),
                       factored_minus_extended_inf=inf_norm(factored.astype(np.longdouble) - reference),
                       assembled_relative_to_extended_inf=inf_norm(assembled.astype(np.longdouble) - reference) / scale,
                       factored_relative_to_extended_inf=inf_norm(factored.astype(np.longdouble) - reference) / scale)
        records.append(row)
    return records


def extended_hessian(A, Q, D):
    import numpy as np
    design = A.toarray().astype(np.longdouble)
    return Q.toarray().astype(np.longdouble) + design.T @ (np.asarray(D, dtype=np.longdouble)[:, None] * design)


def weighted_design_reference(A, Q, D, H):
    """Bounded SVD reference for diagonal nonnegative Q and curvature D."""
    import numpy as np
    from scipy import linalg, sparse
    diagonal = Q.diagonal()
    if (Q - sparse.diags(diagonal)).nnz or np.any(diagonal < 0) or np.any(D < 0):
        return None, dict(skipped="Weighted-design reference requires diagonal nonnegative Q and D")
    B = np.vstack([np.sqrt(D)[:, None] * A.toarray(), np.diag(np.sqrt(diagonal))])
    _, singular, right = linalg.svd(B, full_matrices=False, lapack_driver="gesvd")
    reference = extended_hessian(A, Q, D)
    difference = H.toarray().astype(np.longdouble) - reference
    condition = singular[0] / singular[-1] if singular[-1] else np.inf
    return right[-1], dict(
        method="Float64 SVD of [sqrt(D) A; sqrt(Q)], plus longdouble Gram of original float64 A,Q,D",
        extended_dtype=str(reference.dtype), extended_epsilon=float(np.finfo(np.longdouble).eps),
        float64_epsilon=float(np.finfo(float).eps), singular_values=singular.tolist(),
        weighted_design_condition_2=float(condition), implied_hessian_condition_2=float(condition**2),
        condition_note="SVD estimate, not a certified condition bound; no entry-range inference",
        gram_error_max_abs=inf_norm(difference),
        gram_error_relative_max_abs=inf_norm(difference) / max(inf_norm(reference), np.finfo(float).tiny))


def system_residuals(G, shift, rhs, solution, tau):
    """Keep requested-system accuracy distinct from factor-input accuracy."""
    requested = G @ solution - rhs
    shifted = (G + tau * shift) @ solution - rhs
    return dict(requested_residual_inf=inf_norm(requested),
                shifted_residual_inf=inf_norm(shifted),
                numerical_shift_action_inf=inf_norm(tau * (shift @ solution)),
                requested_rhs_scaled_inf=inf_norm(requested) / max(1., inf_norm(rhs)))


def qr_newton_control(A, Q, D, G, rhs, rho):
    """Solve the positive Schur reduction via a stacked-design QR factor.

    G = [[H+rho I, V.T], [V, -diag(p)]], p>0. Elimination gives
    B.T B dx = rhs_x + V.T(rhs_dual/p), with rows of B formed directly
    from sqrt(D)A, sqrt(Q+rho I), and V/sqrt(p). This is a diagnostic
    control and changes neither the production matrix nor the production solve.
    """
    import numpy as np
    from scipy import linalg, sparse
    n = A.shape[1]
    diagonal = Q.diagonal()
    lower = G[n:, n:]
    pivots = -lower.diagonal()
    if ((Q - sparse.diags(diagonal)).nnz or np.any(diagonal + rho <= 0)
            or np.any(D < 0) or (lower - sparse.diags(lower.diagonal())).nnz
            or np.any(pivots <= 0)):
        return None, dict(skipped="QR reduction requires diagonal positive primal ridge and negative diagonal dual block")
    V = G[n:, :n].toarray()
    B = np.vstack([np.sqrt(D)[:, None] * A.toarray(), np.diag(np.sqrt(diagonal + rho)),
                   V / np.sqrt(pivots)[:, None]])
    reduced_rhs = rhs[:n] + V.T @ (rhs[n:] / pivots)
    _, R, permutation = linalg.qr(B, mode="economic", pivoting=True)
    intermediate = linalg.solve_triangular(R.T, reduced_rhs[permutation], lower=True)
    permuted_dx = linalg.solve_triangular(R, intermediate)
    dx = np.empty(n)
    dx[permutation] = permuted_dx
    solution = np.concatenate([dx, (V @ dx - rhs[n:]) / pivots])
    singular = linalg.svdvals(B)
    reference_G = G.toarray().astype(np.longdouble)
    reference_G[:n, :n] = extended_hessian(A, Q, D) + np.longdouble(rho) * np.eye(n, dtype=np.longdouble)
    return solution, dict(
        method="Float64 pivoted QR of stacked weighted design; two triangular solves; no Gram assembly",
        reduced_design_shape=list(B.shape), reduced_design_condition_2=float(singular[0] / singular[-1]),
        requested_residual_inf=inf_norm(G @ solution - rhs),
        extended_operator_residual_inf=inf_norm(reference_G @ solution.astype(np.longdouble) - rhs),
        extended_reference_note="Frozen float64 curvature and scaled dual blocks, extended objective Hessian plus rho I")


class WorkMeter:
    """Inclusive component times and counts from the diagnostic solve only."""
    def __init__(self):
        self.records = {}

    def wrap(self, name, function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                record = self.records.setdefault(name, dict(calls=0, diagnostic_inclusive_seconds=0.))
                record["calls"] += 1
                record["diagnostic_inclusive_seconds"] += time.perf_counter() - start
        return measured


class StateCapture:
    def __init__(self, selected):
        self.selected = set(selected)
        self.states, self.newton = {}, {}
        self.latest = None
        self.active = None
        self.last_gradient_predictor = None
        self.last_scalar_gradient = None

    def capture_gradient(self, original):
        @functools.wraps(original)
        def wrapped(z):
            result = original(z)
            self.last_gradient_predictor = z.copy()
            self.last_scalar_gradient = result.copy()
            return result
        return wrapped

    def callback(self, snapshot):
        state = {key: getattr(snapshot, key).copy() for key in ("x", "y", "s", "nu")}
        # The driver evaluates the accepted gradient or reuses an accepted
        # residual trial's gradient before invoking this callback. Subsequent
        # rejected trials cannot alter these owned callback copies.
        state["carried_z"] = self.last_gradient_predictor.copy()
        state["carried_scalar_gradient"] = self.last_scalar_gradient.copy()
        state.update({key: getattr(snapshot, key) for key in
                      ("iteration", "stage", "objective", "barrier_target", "kkt_residual",
                       "relative_complementarity", "native_converged")})
        self.latest = state
        if snapshot.iteration in self.selected:
            self.states[snapshot.iteration] = state

    def capture_kkt(self, original):
        @functools.wraps(original)
        def wrapped(x, y, s, H, rx, rp, rc, req, mu, **kwargs):
            iteration = self.latest["iteration"]
            self.active = iteration if iteration in self.selected else None
            if self.active is not None:
                self.newton[iteration] = dict(H=H.copy(), x=x.copy(), y=y.copy(), s=s.copy(),
                    rx=rx.copy(), rp=rp.copy(), rc=rc.copy(), req=req.copy(), target=float(mu),
                    rho=float(kwargs["prox_reg"]),
                    delta=float(kwargs.get("dual_reg", kwargs["prox_reg"])),
                    slack_shift=float(kwargs.get("complementarity_reg", kwargs["prox_reg"])))
            try:
                return original(x, y, s, H, rx, rp, rc, req, mu, **kwargs)
            finally:
                self.active = None
        return wrapped

    def capture_linear(self, original):
        @functools.wraps(original)
        def wrapped(G, rhs, *args, **kwargs):
            result = original(G, rhs, *args, **kwargs)
            if self.active is not None:
                factor = result.factorization
                self.newton[self.active].update(G=G.copy(), rhs=rhs.copy(),
                    shift=kwargs["reg_shift"].copy(), solution=result.solution.copy(), info=asdict(result.info),
                    factor={} if factor is None else factor.diagnostics())
            return result
        return wrapped


def profile_rows(profiler):
    stats = pstats.Stats(profiler)
    return [dict(file=key[0], line=key[1], function=key[2], primitive_calls=value[0], calls=value[1],
                 diagnostic_self_seconds=value[2], diagnostic_cumulative_seconds=value[3])
            for key, value in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:100]]


def analyze_case(problem, model, capture, output, args):
    import numpy as np
    from scipy import sparse
    for name in ("A", "Q", "C", "E"):
        sparse.save_npz(output / f"{name}.npz", getattr(model, name))
    if capture.latest is not None:
        capture.states[capture.latest["iteration"]] = capture.latest
    reports = []
    for iteration, state in sorted(capture.states.items()):
        fresh_z = model.A @ state["x"]
        z = state["carried_z"]
        D = model.f.d2f(z)
        H = model.get_H(z)
        carried_gradient = model.f.d1f(z)
        fresh_gradient = model.f.d1f(fresh_z)
        shared_stationarity = model.Q @ state["x"] - model.b + model.C.T @ state["y"] + model.E.T @ state["nu"]
        predictor = dict(
            observation="Most recent d1f input copied at initial/accepted callback; analysis evaluations occur after solve",
            carried_minus_fresh_inf=inf_norm(z-fresh_z),
            relative_drift_inf=inf_norm(z-fresh_z) / max(1., inf_norm(fresh_z)),
            scalar_gradient_drift_inf=inf_norm(carried_gradient-fresh_gradient),
            stationarity_drift_inf=inf_norm(model.A.T @ (carried_gradient-fresh_gradient)),
            carried_stationarity_inf=inf_norm(shared_stationarity + model.A.T @ carried_gradient),
            fresh_stationarity_inf=inf_norm(shared_stationarity + model.A.T @ fresh_gradient),
            carried_minus_fresh_loss=float(model.f(z)-model.f(fresh_z)),
            recomputed_gradient_matches_observed=bool(np.array_equal(carried_gradient, state["carried_scalar_gradient"])))
        rng = np.random.default_rng(715)
        vectors = {f"random_{index}": rng.normal(size=model.n) for index in range(3)}
        for vector in vectors.values():
            vector /= np.linalg.norm(vector)
        bounded = (model.n <= args.dense_max_features
                   and (model.A.shape[0] + model.n) * model.n <= args.dense_max_elements)
        use_reference = args.dense_reference and iteration == args.reference_state and bounded
        dense = None
        if use_reference:
            weak_vector, dense = weighted_design_reference(model.A, model.Q, D, H)
            if weak_vector is not None:
                vectors["weakest_weighted_design_svd_mode"] = weak_vector
        elif args.dense_reference and iteration == args.reference_state:
            dense = dict(skipped="Dense dimension/element budget exceeded")
        prefix = output / f"state-{iteration:05d}"
        np.savez_compressed(str(prefix) + ".npz", **{key: state[key] for key in ("x", "y", "s", "nu")},
                            carried_z=z, fresh_z=fresh_z, D=D,
                            **{f"probe_{name}": value for name, value in vectors.items()})
        sparse.save_npz(str(prefix) + "-H.npz", H)
        row = dict(state={key: value for key, value in state.items()
                          if key not in ("x", "y", "s", "nu", "carried_z", "carried_scalar_gradient")},
                   predictor_drift=predictor,
                   curvature=curvature_report(problem.loss, z, problem.weights, D),
                   H=matrix_stats(H), symmetry_defect_inf=inf_norm((H-H.T).data),
                   action_comparison=action_comparison(model.A, model.Q, D, H, vectors, extended=use_reference),
                   dense_reference=dense)
        actual = capture.newton.get(iteration)
        if actual is not None and "G" in actual:
            G, shift, rhs, solution = (actual[key] for key in ("G", "shift", "rhs", "solution"))
            np.savez_compressed(str(prefix) + "-newton.npz", **{key: actual[key] for key in
                                ("rhs", "solution", "rx", "rp", "rc", "req")})
            sparse.save_npz(str(prefix) + "-requested-G.npz", G)
            sparse.save_npz(str(prefix) + "-factor-shift.npz", shift)
            info = actual["info"]
            linear = dict(requested_G=matrix_stats(G),
                          factor_input=matrix_stats(G + info["tau_reg"] * shift),
                          factor=actual["factor"], linear_info=info,
                          outer_regularization={key: actual[key] for key in ("rho", "delta", "slack_shift")},
                          barrier_target=actual["target"],
                          **system_residuals(G, shift, rhs, solution, info["tau_reg"]))
            linear["captured_minus_recomputed_hessian_inf"] = inf_norm((actual["H"]-H).data)
            if use_reference and G.shape[0]**2 <= args.dense_max_elements:
                control, control_info = qr_newton_control(model.A, model.Q, D, G, rhs, actual["rho"])
                linear["qr_control"] = control_info
                if control is not None:
                    reference_G = G.toarray().astype(np.longdouble)
                    reference_G[:model.n, :model.n] = extended_hessian(model.A, model.Q, D) + (
                        np.longdouble(actual["rho"]) * np.eye(model.n, dtype=np.longdouble))
                    linear["production_extended_operator_residual_inf"] = inf_norm(
                        reference_G @ solution.astype(np.longdouble) - rhs)
                    control_info["relative_direction_difference_inf"] = inf_norm(solution-control) / max(1., inf_norm(control))
                    np.savez_compressed(str(prefix) + "-qr-control.npz", solution=control)
            row["actual_next_newton_system"] = linear
        reports.append(row)
    return reports


def run_case(problem, backend, args):
    import numpy as np
    from scipy import sparse
    from glqp_bench.adapters import glqp_settings
    from glqp_bench.glm import glm_hash, glm_payload, validate_glm
    from glqp.backends import backend_type
    from glqp.kkt import KKTWorkspace
    from glqp.linear import ShiftedSystemWorkspace
    import glqp.solver as solver_module
    import glqp.linear as linear_module
    from glqp.obj import GaussianNLL, LogisticNLL, PoissonNLL

    output = args.output / problem.name
    output.mkdir()
    settings = glqp_settings(backend, tol=1e-7, time_limit=600., settings_profile=args.settings_profile)
    for key, value in json.loads(args.settings_json).items():
        if not hasattr(settings, key):
            raise ValueError(f"Unknown solver setting: {key}")
        setattr(settings, key, value)
    if settings.step_strategy != "newton":
        raise ValueError("This focused probe captures actual Newton calls; step_strategy must be newton")
    settings.iteration_diagnostics = False
    objective = {"gaussian": GaussianNLL, "logistic": LogisticNLL, "poisson": PoissonNLL}[problem.loss]
    model = backend.GLQP(f=objective(problem.response, problem.weights), A=problem.X,
        Q=sparse.diags(problem.ridge, format="csc"), b=-problem.q,
        C=problem.C if len(problem.c) else None, c=problem.c if len(problem.c) else None,
        E=problem.E if len(problem.e) else None, e=problem.e if len(problem.e) else None)
    capture, meter = StateCapture(args.states), WorkMeter()
    profiler = cProfile.Profile()
    factor_class = backend_type(settings.linear_solver_backend)
    with ExitStack() as stack:
        stack.enter_context(patch.object(model.f, "d1f", capture.capture_gradient(model.f.d1f)))
        stack.enter_context(patch.object(model, "get_H", meter.wrap("objective_hessian", model.get_H)))
        stack.enter_context(patch.object(model, "solve_KKT", capture.capture_kkt(model.solve_KKT)))
        stack.enter_context(patch.object(solver_module, "factor_and_solve",
            capture.capture_linear(meter.wrap("verified_linear_solve", solver_module.factor_and_solve))))
        for owner, method, label in ((KKTWorkspace, "assemble", "kkt_assembly"),
                (KKTWorkspace, "_build", "kkt_pattern_build"),
                (ShiftedSystemWorkspace, "assemble", "factor_shift_assembly"),
                (factor_class, "__init__", "factor_create"), (factor_class, "update", "factor_update"),
                (factor_class, "solve", "factor_inverse_application"),
                (linear_module, "_gmres_refine", "gmres_refinement")):
            stack.enter_context(patch.object(owner, method, meter.wrap(label, getattr(owner, method))))
        profiler.enable()
        try:
            _, result = model.solve(verbose=False, settings=settings, iteration_callback=capture.callback)
        finally:
            profiler.disable()
    profiler.dump_stats(str(output / "solve.prof"))
    result.history.to_csv(output / "history.csv", index=False)
    np.savez_compressed(output / "problem-and-solution.npz", **glm_payload(problem),
                        x=result.x, y=result.y, s=result.s, nu=result.nu)
    report = dict(problem_id=problem.name, data_sha256=glm_hash(problem), settings=asdict(settings),
        termination=result.termination_tag, iterations=len(result.history),
        exception=None if result.exception is None else str(result.exception),
        validation=validate_glm(problem, result.x, result.nu[:len(problem.e)], result.y[:len(problem.c)]),
        matrices={name: matrix_stats(getattr(model, name)) for name in ("A", "Q", "C", "E")},
        dummy_inequality=model.dummy_ineq,
        work_attribution=dict(note="Instrumented inclusive diagnostic times overlap; not comparative performance",
                              diagnostic_profile_total_seconds=pstats.Stats(profiler).total_tt,
                              components=meter.records, cprofile_top=profile_rows(profiler)),
        states=analyze_case(problem, model, capture, output, args))
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--benchmark-repo", type=Path, default=Path(__file__).resolve().parents[2] / "glqp-bench")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--cpu", type=int, default=6)
    parser.add_argument("--states", type=int, nargs="+", default=[0, 1, 5, 20])
    parser.add_argument("--settings-profile", default="glm-v2")
    parser.add_argument("--settings-json", default="{}", help="JSON setting overrides; Newton only")
    parser.add_argument("--dense-reference", action="store_true")
    parser.add_argument("--reference-state", type=int, default=0)
    parser.add_argument("--dense-max-features", type=int, default=256)
    parser.add_argument("--dense-max-elements", type=int, default=2_000_000)
    args = parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("Requested CPU is outside the process affinity")
    if not 1 <= args.dense_max_features <= 256 or args.dense_max_elements <= 0:
        parser.error("Dense reference requires 1..256 features and a positive element budget")
    if any(index < 0 for index in args.states) or len(set(args.cases)) != len(args.cases):
        parser.error("Use nonnegative state indices and unique cases")
    if args.dense_reference and args.reference_state not in args.states:
        parser.error("--reference-state must be included in --states")
    os.sched_setaffinity(0, {args.cpu})
    thread_names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")
    for name in thread_names:
        os.environ[name] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    args.source, args.benchmark_repo, args.output = (path.resolve() for path in
                                                   (args.source, args.benchmark_repo, args.output))
    sys.path.insert(0, str(args.source))
    sys.path.insert(0, str(args.benchmark_repo / "src"))
    import glqp
    from glqp_bench.glm import build_glm
    if not Path(glqp.__file__).resolve().is_relative_to(args.source):
        raise RuntimeError(f"Wrong GLQP imported: {glqp.__file__}")
    args.output.mkdir(parents=True, exist_ok=False)
    initial_hashes = source_hashes(args.source)
    reports = []
    for name in args.cases:
        report = run_case(build_glm(name), glqp, args)
        reports.append(dict(problem_id=name, termination=report["termination"], iterations=report["iterations"],
                            accepted=report["validation"]["accepted"], report=f"{name}/report.json"))
        print(json.dumps(reports[-1]), flush=True)
    write_json(args.output / "summary.json", dict(
        purpose="Instrumented numerical and work diagnostic; no comparative timing claims",
        source=str(args.source), source_sha256=initial_hashes,
        source_unchanged=initial_hashes == source_hashes(args.source),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        benchmark_sources={str(path.relative_to(args.benchmark_repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted((args.benchmark_repo / "src/glqp_bench").glob("glm*.py"))},
        python=sys.version, affinity=sorted(os.sched_getaffinity(0)),
        numerical_threads={name: os.environ.get(name) for name in thread_names}, results=reports))


if __name__ == "__main__":
    main()
