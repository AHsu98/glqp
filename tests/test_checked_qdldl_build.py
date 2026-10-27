"""Frozen private libraries verify source/ABI/checksums without build tools."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import glqp.checked_qdldl as checked


def frozen_copy(tmp_path):
    provenance = checked.prepare_checked_qdldl()
    library, manifest = tmp_path/'private.so', tmp_path/'build.json'
    shutil.copy2(provenance['library'],library)
    shutil.copy2(provenance['manifest'],manifest)
    return library,manifest,provenance


def test_numeric_c_sources_and_licenses_preserve_pinned_upstream_bytes():
    root = Path(checked.__file__).parent/'_native/checked_qdldl'
    metadata = json.loads((root/'upstream.json').read_text())
    changed = []
    for name,sha in metadata['original_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:
            changed.append(name)
    assert sorted(changed)==sorted(metadata['modified_files'])
    assert all(name.startswith('cpp/')for name in changed)
    assert metadata['numeric_c_sources_unchanged']


def test_frozen_copy_loads_in_cold_process_without_build_headers_or_compiler(tmp_path):
    library,manifest,provenance = frozen_copy(tmp_path)
    env=dict(os.environ,GLQP_CHECKED_QDLDL_LIBRARY=str(library),
             GLQP_CHECKED_QDLDL_MANIFEST=str(manifest),CXX='/missing/compiler')
    code='''import json,sys
sys.modules['pybind11']=None
from glqp.checked_qdldl import prepare_checked_qdldl
print(json.dumps(prepare_checked_qdldl()))
'''
    result=subprocess.run([sys.executable,'-c',code],env=env,check=True,
                          text=True,capture_output=True,timeout=30)
    copied=json.loads(result.stdout)
    assert copied['library']==str(library)
    assert copied['manifest']==str(manifest)
    assert copied['build_key']==provenance['build_key']
    assert copied['library_sha256']==provenance['library_sha256']
    assert copied['manifest_sha256']==provenance['manifest_sha256']


@pytest.mark.parametrize('change',['binary','source','abi'])
def test_frozen_artifact_mismatch_fails_before_import(tmp_path,monkeypatch,change):
    library,manifest,_=frozen_copy(tmp_path)
    monkeypatch.setenv('GLQP_CHECKED_QDLDL_LIBRARY',str(library))
    monkeypatch.setenv('GLQP_CHECKED_QDLDL_MANIFEST',str(manifest))
    if change=='binary':
        with library.open('ab')as stream:stream.write(b'changed')
    else:
        metadata=json.loads(manifest.read_text())
        key='source_sha256' if change=='source' else 'python_abi'
        metadata['identity'][key]='mismatch'
        metadata['build_key']=checked._key(metadata['identity'])
        manifest.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError,match='checksum|source/Python ABI'):
        checked._prepare()


def test_frozen_paths_must_be_supplied_together(tmp_path,monkeypatch):
    monkeypatch.setenv('GLQP_CHECKED_QDLDL_LIBRARY',str(tmp_path/'private.so'))
    monkeypatch.delenv('GLQP_CHECKED_QDLDL_MANIFEST',raising=False)
    with pytest.raises(ValueError,match='both'):
        checked._prepare()
