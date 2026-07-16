from __future__ import annotations

import json

from eegprep.cli import core


def test_build_manifest_copies_warnings_and_serializes_software(monkeypatch):
    software = {
        "eegprep_version": "test",
        "math_backend_info": {"loaded_libraries": [{"internal_api": "openblas"}]},
    }
    monkeypatch.setattr(core, "software_info", lambda: software)
    input_warnings = ["test warning"]

    manifest = core.build_manifest(
        command="test",
        input_files=[],
        output_files=[],
        parameters={},
        started_at="2026-07-15T00:00:00Z",
        warnings=input_warnings,
    )

    assert manifest["software"] == software
    assert "math_backend_info" not in manifest
    assert manifest["warnings"] == input_warnings
    assert manifest["warnings"] is not input_warnings
    json.dumps(core.json_safe(manifest))

    manifest["warnings"].append("new warning")
    assert input_warnings == ["test warning"]


def test_software_info_reports_logical_cpu_count(monkeypatch):
    backend_info = {"loaded_libraries": []}
    monkeypatch.setattr("eegprep.utils.math_backend.get_math_backend_info", lambda: backend_info)
    monkeypatch.setattr(core.os, "cpu_count", lambda: 12)

    info = core.software_info()

    assert info["logical_cpu_count"] == 12
    assert info["math_backend_info"] is backend_info
