import pytest
from pathlib import Path
from eegprep.cli.core import _make_manifest_relative, read_manifest

def test_manifest_relative_paths(tmp_path):
    # Test internal paths (should become relative)
    base_dir = tmp_path / "project"
    manifest_path = base_dir / "derivatives" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    
    input_file = base_dir / "rawdata" / "sub-01" / "eeg.set"
    input_file.parent.mkdir(parents=True)
    
    sidecar_file = base_dir / "derivatives" / "eeg.fdt"
    
    # External file
    external_file = tmp_path / "other_drive" / "ext.set"
    
    manifest = {
        "schema_version": "eegprep.manifest.v2",
        "input_files": [{"path": str(input_file)}, {"path": str(external_file)}],
        "output_files": [{"path": str(sidecar_file)}]
    }
    
    rel_manifest = _make_manifest_relative(manifest, manifest_path)
    
    # Internal paths should be relative (POSIX format)
    assert rel_manifest["input_files"][0]["path"] == "../rawdata/sub-01/eeg.set"
    assert rel_manifest["output_files"][0]["path"] == "eeg.fdt"
    
    # External path becomes relative but points outside
    assert rel_manifest["input_files"][1]["path"] == "../../other_drive/ext.set"
    
    # Write it to disk and test read_manifest
    import json
    manifest_path.write_text(json.dumps(rel_manifest))
    
    read_back = read_manifest(manifest_path)
    # They should all be expanded to absolute paths
    assert read_back["input_files"][0]["path"] == str(input_file.resolve())
    assert read_back["output_files"][0]["path"] == str(sidecar_file.resolve())
    assert read_back["input_files"][1]["path"] == str(external_file.resolve())

def test_windows_paths(tmp_path, monkeypatch):
    import os
    
    base_dir = tmp_path / "win_project"
    manifest_path = base_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    
    # Mock os.path.relpath to simulate Windows ValueError for different drives
    original_relpath = os.path.relpath
    def mock_relpath(path, start):
        if str(path).startswith("D:"):
            raise ValueError("path is on mount 'D:', start on mount 'C:'")
        return original_relpath(path, start)
    
    monkeypatch.setattr(os.path, "relpath", mock_relpath)
    
    # External Windows drive path
    external_file = "D:\\data\\ext.set"
    
    manifest = {
        "schema_version": "eegprep.manifest.v2",
        "input_files": [{"path": external_file}],
        "output_files": []
    }
    
    rel_manifest = _make_manifest_relative(manifest, manifest_path)
    
    # Because of ValueError, it should remain absolute
    assert rel_manifest["input_files"][0]["path"] == external_file
