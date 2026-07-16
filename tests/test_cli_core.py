from eegprep.cli.core import build_manifest


def test_build_manifest_copies_warnings():
    # Test that the warnings list is copied and not mutated in place
    input_warnings = ["test_warning_1"]
    manifest = build_manifest(
        command="test",
        input_files=[],
        output_files=[],
        parameters={},
        started_at="2026-07-15T00:00:00Z",
        warnings=input_warnings,
    )
    assert "test_warning_1" in manifest["warnings"]
    assert manifest["warnings"] is not input_warnings

    # Mutating the manifest's warning list shouldn't affect the input
    manifest["warnings"].append("new_warning")
    assert "new_warning" not in input_warnings


def test_build_manifest_software_serialization():
    # Test that the software info is serialized correctly and no top-level math_backend_info exists
    manifest = build_manifest(
        command="test",
        input_files=[],
        output_files=[],
        parameters={},
        started_at="2026-07-15T00:00:00Z",
    )
    assert "software" in manifest
    assert "eegprep_version" in manifest["software"]
    assert "python_version" in manifest["software"]
    assert "math_backend_info" in manifest["software"]
    assert "math_backend_info" not in manifest  # Should not be at the top level
