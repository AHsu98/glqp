"""Private checked QDLDL binding with explicit, reproducible native preparation.

The installed ``qdldl`` module is never changed. Cold benchmark workers can load
an already prepared library via GLQP_CHECKED_QDLDL_LIBRARY and its matching
GLQP_CHECKED_QDLDL_MANIFEST, before entering their measured solve section.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile

_MODULE = '_glqp_qdldl_checked'
_PYBIND_VERSION = '2.13.6'
_FLAGS = ['-O3', '-fno-fast-math', '-ffp-contract=off', '-fPIC', '-fvisibility=hidden']


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _sources():
    root = Path(__file__).resolve().parent/'_native/checked_qdldl'
    files = {str(path.relative_to(root)): _digest(path)
             for path in sorted(root.rglob('*')) if path.is_file()}
    files['loader:checked_qdldl.py'] = _digest(Path(__file__))
    return root, files, _key(files)


def _python_abi():
    return dict(version=sys.version, cache_tag=sys.implementation.cache_tag,
                soabi=sysconfig.get_config_var('SOABI'),
                extension_suffix=sysconfig.get_config_var('EXT_SUFFIX'),
                machine=platform.machine(), system=platform.system(),
                pointer_bits=struct.calcsize('P')*8)


def _compiler(variable, default):
    command = shutil.which(os.environ.get(variable, default))
    if command is None:
        raise RuntimeError(f'{variable} must name an available compiler executable')
    version = subprocess.run([command,'--version'], check=True, capture_output=True,
                             text=True, timeout=20).stdout
    return dict(command=command, version=version)


def _build_identity(source_hash, files):
    if sys.platform != 'linux' or struct.calcsize('P') != 8:
        raise RuntimeError('The checked QDLDL source builder currently supports 64-bit Linux')
    try:
        import pybind11
    except ImportError as error:
        raise ImportError(f'Building checked QDLDL requires pybind11=={_PYBIND_VERSION}; '
                          'install the native-build extra or supply a frozen library/manifest') from error
    if pybind11.__version__ != _PYBIND_VERSION:
        raise RuntimeError(f'Checked QDLDL builds require pybind11=={_PYBIND_VERSION}')
    include = Path(pybind11.get_include())
    headers = {str(path.relative_to(include)): _digest(path)
               for path in sorted(include.rglob('*.h'))}
    cmake = shutil.which('cmake')
    if cmake is None:
        raise RuntimeError('Building checked QDLDL requires CMake')
    cmake_version = subprocess.run([cmake,'--version'], check=True, capture_output=True,
                                   text=True, timeout=20).stdout
    identity = dict(source_sha256=source_hash, source_files=files,
                    compiler=dict(c=_compiler('CC','cc'), cxx=_compiler('CXX','c++')),
                    compiler_flags=_FLAGS, cxx_standard=17, python_abi=_python_abi(),
                    pybind11_version=pybind11.__version__, pybind11_headers_sha256=_key(headers),
                    pybind11_headers=headers, cmake_version=cmake_version,
                    qdldl_version='0.1.9.post1', float_type='float64', index_type='int64')
    return identity, include, cmake


def _validate(library, manifest, source_hash):
    metadata = json.loads(manifest.read_text())
    identity = metadata['identity']
    if metadata['module_name'] != _MODULE or metadata['build_key'] != _key(identity):
        raise RuntimeError('Invalid checked QDLDL build identity')
    if identity['source_sha256'] != source_hash or identity['python_abi'] != _python_abi():
        raise RuntimeError('Checked QDLDL library does not match the current source/Python ABI')
    if metadata['library_sha256'] != _digest(library):
        raise RuntimeError('Checked QDLDL library checksum mismatch')
    return dict(metadata, library=str(library.resolve()), manifest=str(manifest.resolve()),
                manifest_sha256=_digest(manifest))


def _prepare():
    source, files, source_hash = _sources()
    frozen_library = os.environ.get('GLQP_CHECKED_QDLDL_LIBRARY')
    frozen_manifest = os.environ.get('GLQP_CHECKED_QDLDL_MANIFEST')
    if bool(frozen_library) != bool(frozen_manifest):
        raise ValueError('Set both GLQP_CHECKED_QDLDL_LIBRARY and GLQP_CHECKED_QDLDL_MANIFEST')
    if frozen_library:
        library, manifest = Path(frozen_library).resolve(), Path(frozen_manifest).resolve()
        return library, _validate(library, manifest, source_hash)

    import fcntl
    identity, pybind_include, cmake = _build_identity(source_hash, files)
    key = _key(identity)
    cache = Path(os.environ.get('GLQP_CHECKED_QDLDL_CACHE',
                 str(Path(tempfile.gettempdir())/f'glqp-checked-qdldl-{os.getuid()}')))
    directory = cache/key
    directory.mkdir(parents=True,exist_ok=True)
    library = directory/(_MODULE+sysconfig.get_config_var('EXT_SUFFIX'))
    manifest = directory/'build.json'
    with (directory/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not library.is_file() or not manifest.is_file():
            # Build in a separate directory, then atomically publish a complete
            # library and manifest. No download, pip call or fallback is hidden.
            with tempfile.TemporaryDirectory(prefix='build-',dir=directory) as temporary:
                build = Path(temporary)
                configure = [cmake,'-S',str(source),'-B',str(build),
                    '-DCMAKE_BUILD_TYPE=', '-DCMAKE_C_FLAGS=', '-DCMAKE_CXX_FLAGS=',
                    '-DCMAKE_MODULE_LINKER_FLAGS=', '-DCMAKE_SHARED_LINKER_FLAGS=',
                    '-DCMAKE_C_COMPILER='+identity['compiler']['c']['command'],
                    '-DCMAKE_CXX_COMPILER='+identity['compiler']['cxx']['command'],
                    '-DPYTHON_INCLUDE_DIR='+sysconfig.get_path('include'),
                    '-DPYBIND11_INCLUDE_DIR='+str(pybind_include),
                    '-DPYTHON_EXTENSION_SUFFIX='+sysconfig.get_config_var('EXT_SUFFIX')]
                subprocess.run(configure,check=True,capture_output=True,text=True,timeout=120)
                subprocess.run([cmake,'--build',str(build),'--parallel','1'],
                               check=True,capture_output=True,text=True,timeout=240)
                compiled = build/library.name
                metadata = dict(module_name=_MODULE,build_key=key,identity=identity,
                                library_sha256=_digest(compiled))
                compiled.replace(library)
                temporary_manifest = directory/f'build-{os.getpid()}.json'
                temporary_manifest.write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
                temporary_manifest.replace(manifest)
        return library,_validate(library,manifest,source_hash)


@functools.lru_cache(maxsize=1)
def _native():
    library, provenance = _prepare()
    spec = importlib.util.spec_from_file_location(_MODULE,library)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load checked QDLDL extension: {library}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module,provenance


def prepare_checked_qdldl() -> dict:
    """Build or verify/load a frozen binding and return its full provenance.

    Set library/manifest overrides before the first call in a cold process.
    Preparation errors are configuration/build failures, never numerical retry
    events. The manifest remains valid when copied alongside its library.
    """
    return dict(_native()[1])


def Solver(*args, **kwargs):
    """Create a checked native solver using the upstream-compatible API."""
    return _native()[0].Solver(*args,**kwargs)
