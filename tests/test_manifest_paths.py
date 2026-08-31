import json
import os
from pathlib import Path

import pytest

from eegprep.cli.core import (
    MANIFEST_SCHEMA_VERSION,
    build_manifest,
    read_manifest,
    write_manifest,
    write_manifest_file,
)


def _build_manifest(input_path: Path, output_paths: list[Path | str]) -> dict:
    return build_manifest(
        command="test",
        input_files=[input_path],
        output_files=[{"path": str(path), "type": "artifact"} for path in output_paths],
        parameters={},
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
    )


def test_manifest_v2_round_trip_uses_manifest_relative_posix_paths(tmp_path):
    project = tmp_path / "project"
    input_path = project / "raw" / "sub-01.set"
    output_path = project / "derivatives" / "sub-01-clean.set"
    sidecar_path = project / "derivatives" / "sub-01-clean.fdt"
    manifest_path = project / "derivatives" / "manifests" / "sub-01.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"input")

    manifest = _build_manifest(input_path, [output_path, sidecar_path])
    original_paths = [item["path"] for item in manifest["input_files"] + manifest["output_files"]]

    entry = write_manifest_file(manifest_path, manifest)

    assert entry["path"] == str(manifest_path)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert stored["input_files"][0]["path"] == "../../raw/sub-01.set"
    assert [item["path"] for item in stored["output_files"]] == [
        "../sub-01-clean.set",
        "../sub-01-clean.fdt",
    ]
    assert [item["path"] for item in manifest["input_files"] + manifest["output_files"]] == original_paths

    loaded = read_manifest(manifest_path)
    assert [item["path"] for item in loaded["input_files"] + loaded["output_files"]] == original_paths


def test_build_manifest_normalizes_runtime_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_path = Path("raw/input.set")
    output_path = Path("derivatives/output.set")
    input_path.parent.mkdir()
    input_path.write_bytes(b"input")

    manifest = build_manifest(
        command="test",
        input_files=[input_path, {"path": str(input_path), "sha256": "recorded"}],
        output_files=[{"path": str(output_path), "type": "artifact"}],
        parameters={},
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
    )

    assert manifest["input_files"][0]["path"] == str((tmp_path / input_path).resolve())
    assert manifest["input_files"][1]["path"] == str((tmp_path / input_path).resolve())
    assert manifest["output_files"][0]["path"] == str((tmp_path / output_path).resolve())


def test_write_manifest_preserves_relative_return_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "input.set"
    input_path.write_bytes(b"input")
    manifest = _build_manifest(input_path, [tmp_path / "output.set"])
    manifest_path = Path("records/manifest.json")

    result = write_manifest(manifest_path, manifest)
    file_manifest_path = Path("records/manifest-file.json")
    entry = write_manifest_file(file_manifest_path, manifest)

    assert result == manifest_path
    assert (tmp_path / manifest_path).is_file()
    assert entry["path"] == str(file_manifest_path)
    assert (tmp_path / file_manifest_path).is_file()


@pytest.mark.parametrize("schema_version", ["eegprep.manifest.v1", "eegprep.manifest.v3"])
def test_read_manifest_does_not_reinterpret_other_schema_paths(tmp_path, schema_version):
    manifest_path = tmp_path / "manifest.json"
    payload = {
        "schema_version": schema_version,
        "input_files": [{"path": "../legacy/input.set"}],
        "output_files": [{"path": "output.set"}],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert read_manifest(manifest_path) == payload


def test_write_manifest_does_not_rewrite_v1_paths(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    absolute_path = str((tmp_path / "input.set").resolve())
    payload = {
        "schema_version": "eegprep.manifest.v1",
        "input_files": [{"path": absolute_path}],
        "output_files": [],
    }

    write_manifest(manifest_path, payload)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == payload


def test_read_manifest_preserves_foreign_absolute_paths(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    foreign_path = "/volume/input.set" if os.name == "nt" else r"C:\data\input.set"
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "input_files": [{"path": foreign_path}],
        "output_files": [],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert read_manifest(manifest_path)["input_files"][0]["path"] == foreign_path


@pytest.mark.skipif(os.name != "nt", reason="Windows drive semantics")
def test_manifest_keeps_paths_on_another_windows_drive_absolute(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_drive = manifest_path.drive.casefold()
    other_drive = next(
        f"{letter}:" for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ" if f"{letter}:".casefold() != manifest_drive
    )
    foreign_path = rf"{other_drive}\data\input.set"
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "input_files": [{"path": foreign_path}],
        "output_files": [],
    }

    write_manifest(manifest_path, payload)

    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert stored["input_files"][0]["path"] == foreign_path
