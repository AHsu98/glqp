"""Optional native structural QP presolve with original-coordinate recovery.

The independent C++ core implements ``min .5*x.T@Q@x + q.T@x`` subject to
``A@x == b`` and ``G@x <= h``. For GLQP, pass ``q=-problem.b``, equalities
``A=problem.E, b=problem.e``, and inequalities ``G=problem.C, h=problem.c``.
Native compilation is explicit through :func:`prepare_native` or lazy on the
first call. Benchmark runners should prepare before their measured section.
"""
from __future__ import annotations

import ctypes as ct
from dataclasses import asdict, dataclass
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import tempfile
import time

import numpy as np
from scipy import sparse

_DOUBLE = ct.POINTER(ct.c_double)
_INDEX = ct.POINTER(ct.c_int64)


class _Matrix(ct.Structure):
    _fields_ = [("rows", ct.c_int64), ("cols", ct.c_int64), ("nnz", ct.c_int64),
                ("indptr", _INDEX), ("indices", _INDEX), ("data", _DOUBLE)]


class _Options(ct.Structure):
    _fields_ = [("singleton_equalities", ct.c_int), ("fixed_bounds", ct.c_int),
                ("redundant_bounds", ct.c_int), ("infeasibility_guard", ct.c_double)]


@dataclass(frozen=True)
class PresolveOptions:
    """Independent structural passes; no numeric elimination tolerance."""
    singleton_equalities: bool = True
    fixed_bounds: bool = True
    redundant_bounds: bool = True
    infeasibility_guard: float = 1e-12

    def __post_init__(self):
        if any(type(getattr(self, name)) is not bool for name in
               ("singleton_equalities", "fixed_bounds", "redundant_bounds")):
            raise TypeError("Presolve pass options must be bool")
        if not np.isfinite(self.infeasibility_guard) or self.infeasibility_guard < 0:
            raise ValueError("infeasibility_guard must be finite and nonnegative")


class PresolveInfeasible(ValueError):
    """The native presolver certified a contradiction in the constraints."""


def _configure_library(path):
    library = ct.CDLL(str(path))
    library.gqp_presolve.argtypes = [_Matrix, _DOUBLE, _Matrix, _DOUBLE,
                                     _Matrix, _DOUBLE, _Options]
    library.gqp_presolve.restype = ct.c_void_p
    library.gqp_free.argtypes = [ct.c_void_p]
    library.gqp_free.restype = None
    for name, output, extra in (
        ("gqp_status_code", ct.c_int, []), ("gqp_message", ct.c_char_p, []),
        ("gqp_objective_offset", ct.c_double, []),
        ("gqp_get_matrix", _Matrix, [ct.c_int]),
        ("gqp_get_vector", _DOUBLE, [ct.c_int]),
        ("gqp_get_map", _INDEX, [ct.c_int]),
        ("gqp_get_stat", ct.c_int64, [ct.c_int]),
    ):
        function = getattr(library, name)
        function.argtypes, function.restype = [ct.c_void_p, *extra], output
    library.gqp_recover.argtypes = [ct.c_void_p, *([_DOUBLE] * 6)]
    library.gqp_recover.restype = ct.c_int
    return library


@functools.lru_cache(maxsize=1)
def _native():
    """Build once per compiler/source key, serialized across cold workers."""
    import fcntl

    source = Path(__file__).resolve().parent / "_native"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler:
        raise RuntimeError("CXX must name a C++17 compiler")
    version = subprocess.run([*compiler, "--version"], check=True, capture_output=True,
                             text=True, timeout=20).stdout
    flags = ["-O3", "-DNDEBUG", "-std=c++17", "-shared", "-fPIC", "-ffp-contract=off"]
    contents = {name: (source / name).read_bytes() for name in ("presolve.cpp", "presolve.h")}
    source_hash = hashlib.sha256(b"".join(name.encode() + b"\0" + data
                                          for name, data in contents.items())).hexdigest()
    identity = {"source_sha256": source_hash, "compiler": compiler,
                "compiler_version": version, "flags": flags,
                "machine": platform.machine(), "system": platform.system()}
    build_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get("GLQP_NATIVE_CACHE",
                 str(Path(tempfile.gettempdir()) / f"glqp-native-{os.getuid()}")))
    directory = cache / build_key
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "libglqp_presolve.so"
    metadata = directory / "build.json"
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file() or not metadata.is_file():
            started = time.perf_counter()
            temporary = directory / f"library-{os.getpid()}.so"
            try:
                subprocess.run([*compiler, *flags, str(source / "presolve.cpp"),
                                "-o", str(temporary)], check=True, capture_output=True,
                               text=True, timeout=120)
                identity["compile_seconds"] = time.perf_counter() - started
                identity["library_sha256"] = hashlib.sha256(temporary.read_bytes()).hexdigest()
                temporary.replace(library)
                metadata.write_text(json.dumps(identity, sort_keys=True, indent=2) + "\n")
            finally:
                temporary.unlink(missing_ok=True)
        provenance = json.loads(metadata.read_text())
        if provenance["library_sha256"] != hashlib.sha256(library.read_bytes()).hexdigest():
            raise RuntimeError(f"Native presolver cache checksum mismatch: {library}")
    provenance.update(build_key=build_key, library=str(library))
    return _configure_library(library), provenance


def prepare_native() -> dict:
    """Prepare the native library and return source/build provenance."""
    return dict(_native()[1])


def _vector(value, size, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape ({size},)")
    return result


def _matrix(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = sparse.csc_matrix(shape, dtype=np.float64) if value is None else sparse.csc_matrix(value, dtype=np.float64, copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    result.indptr = np.asarray(result.indptr, dtype=np.int64)
    result.indices = np.asarray(result.indices, dtype=np.int64)
    return result


def _view(matrix):
    return _Matrix(*matrix.shape, matrix.nnz, matrix.indptr.ctypes.data_as(_INDEX),
                   matrix.indices.ctypes.data_as(_INDEX), matrix.data.ctypes.data_as(_DOUBLE))


def _pointer(vector):
    return vector.ctypes.data_as(_DOUBLE)


def _copy(pointer, length, dtype):
    return np.ctypeslib.as_array(pointer, shape=(length,)).copy() if length else np.empty(0, dtype=dtype)


def _copy_matrix(view):
    return sparse.csc_matrix((_copy(view.data, view.nnz, np.float64),
                              _copy(view.indices, view.nnz, np.int64),
                              _copy(view.indptr, view.cols + 1, np.int64)),
                             shape=(view.rows, view.cols))


class PresolvedQP:
    """Reduced data plus an owning native reconstruction tape.

    ``status`` is ``reduced``, ``unchanged``, or ``inconclusive``. The last
    returns the original QP, so uncertain arithmetic cannot silently alter it.
    A certified infeasibility raises :class:`PresolveInfeasible` instead.
    ``recover`` returns ``(x, equality_dual, inequality_dual)`` in original
    coordinates. Inequality duals use the nonnegative convention.
    """
    def close(self):
        handle = getattr(self, "_handle", None)
        if handle:
            self._library.gqp_free(handle)
            self._handle = None
        self._closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def n(self):
        return self.q.size

    def recover(self, x, y=None, z=None):
        """Reverse fixed-variable substitutions, including their duals."""
        x = _vector(x, self.n, "x")
        y = _vector(np.zeros(self.A.shape[0]) if y is None else y, self.A.shape[0], "y")
        z = _vector(np.zeros(self.G.shape[0]) if z is None else z, self.G.shape[0], "z")
        if self._closed:
            raise RuntimeError("PresolvedQP is closed")
        if self.status in ("unchanged", "inconclusive"):
            return x, y, z
        original = tuple(np.empty(count, dtype=np.float64) for count in self.original_dimensions)
        status = self._library.gqp_recover(self._handle, _pointer(x), _pointer(y), _pointer(z),
                                           *map(_pointer, original))
        if status:
            raise ArithmeticError(f"Native solution reconstruction failed ({status})")
        return original


def presolve_qp(Q, q, A=None, b=None, G=None, h=None, *,
                options: PresolveOptions | None = None, objective_constant=0.) -> PresolvedQP:
    """Presolve a finite convex QP with a full exactly symmetric Hessian.

    Convexity is the caller's responsibility. Nonconvexity is not diagnosed.
    No arrays are modified. All native work, packing, and reconstruction are
    linear in sparse input/output size, including symmetry validation.
    """
    library, provenance = _native()
    started = time.perf_counter()
    options = options or PresolveOptions()
    if not isinstance(options, PresolveOptions):
        raise TypeError("options must be PresolveOptions")
    q = np.asarray(q)
    if q.ndim != 1:
        raise ValueError("q must be one-dimensional")
    n = q.size
    p = 0 if A is None else np.shape(A)[0]
    m = 0 if G is None else np.shape(G)[0]
    matrices = tuple(_matrix(value, shape, name) for value, shape, name in
                     ((Q, (n, n), "Q"), (A, (p, n), "A"), (G, (m, n), "G")))
    vectors = (_vector(q, n, "q"), _vector(np.zeros(p) if b is None else b, p, "b"),
               _vector(np.zeros(m) if h is None else h, m, "h"))
    if not np.isfinite(objective_constant):
        raise ValueError("objective_constant must be finite")
    packed_seconds = time.perf_counter() - started
    native_started = time.perf_counter()
    handle = library.gqp_presolve(_view(matrices[0]), _pointer(vectors[0]),
                                  _view(matrices[1]), _pointer(vectors[1]),
                                  _view(matrices[2]), _pointer(vectors[2]),
                                  _Options(**asdict(options)))
    native_seconds = time.perf_counter() - native_started
    output = PresolvedQP()
    output._library, output._handle = library, handle
    output._closed = False
    status = library.gqp_status_code(handle)
    output.message = library.gqp_message(handle).decode()
    if status == 2:
        raise PresolveInfeasible(output.message)
    if status == 4:
        raise ValueError(output.message)
    if status not in (0, 1, 3):
        raise RuntimeError(output.message)
    output.status = {0: "unchanged", 1: "reduced", 3: "inconclusive"}[status]
    output.original_dimensions = (n, p, m)
    output.options, output.build = options, dict(provenance)
    if status in (0, 3):
        output.Q, output.A, output.G = matrices
        output.q, output.b, output.h = vectors
        output.variable_map, output.equality_map, output.inequality_map = (
            np.arange(size, dtype=np.int64) for size in (n, p, m))
        output.objective_offset = float(objective_constant)
    else:
        output.Q, output.A, output.G = (_copy_matrix(library.gqp_get_matrix(handle, i)) for i in range(3))
        sizes = (output.Q.shape[1], output.A.shape[0], output.G.shape[0])
        output.q, output.b, output.h = (_copy(library.gqp_get_vector(handle, i), count, np.float64)
                                        for i, count in enumerate(sizes))
        output.variable_map, output.equality_map, output.inequality_map = (
            _copy(library.gqp_get_map(handle, i), count, np.int64) for i, count in enumerate(sizes))
        output.objective_offset = float(objective_constant) + library.gqp_objective_offset(handle)
    names = ("fixed_variables", "removed_equalities", "removed_inequalities", "singleton_rows",
             "propagation_visits", "input_nonzeros")
    output.stats = {name: int(library.gqp_get_stat(handle, i)) for i, name in enumerate(names)}
    if status in (0, 3):
        # Identity reconstruction needs no native copy of the problem data.
        library.gqp_free(handle)
        output._handle = None
    output.stats.update(pack_seconds=packed_seconds, native_seconds=native_seconds,
                        total_seconds=time.perf_counter() - started)
    return output
