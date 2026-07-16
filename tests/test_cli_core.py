import sys
from pathlib import Path
from eegprep.cli.core import _get_active_packages, software_info, build_manifest

def test_get_active_packages_is_fixed_and_deterministic():
    """Contract test: Ensure packages are resolved from a fixed list without depending on sys.modules."""
    packages = _get_active_packages()
    
    # We should have a list of dicts with 'name' and 'version'
    assert isinstance(packages, list)
    for pkg in packages:
        assert "name" in pkg
        assert "version" in pkg
        assert "source" not in pkg  # Provenance source is no longer extracted here

    # Modifying sys.modules should not change the result
    sys.modules["fake_module_test"] = object()
    packages_after = _get_active_packages()
    assert packages == packages_after
    
def test_software_info_contract():
    """Contract test: Validate the structure of software_info output."""
    info = software_info()
    assert "eegprep_version" in info
    assert "python_version" in info
    assert "platform" in info
    assert "packages" in info

def test_build_manifest_contract():
    """Contract test: Validate that build_manifest successfully builds the structure."""
    manifest = build_manifest(
        command="test_command",
        input_files=[{"path": "fake_input.set", "sha256": "dummy_input"}],
        output_files=[{"path": "fake_output.set", "sha256": "dummy"}],
        parameters={"some_param": 42},
        started_at="2024-01-01T00:00:00Z"
    )
    
    assert manifest["schema_version"] == "eegprep.manifest.v1"
    assert manifest["command"] == "test_command"
    assert len(manifest["input_files"]) == 1
    assert "packages" in manifest["software"]
    assert "started_at" in manifest["runtime"]
    assert "finished_at" in manifest["runtime"]
