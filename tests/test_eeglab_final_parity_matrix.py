from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.eeglab_final_parity_matrix import (
    VALID_STATUSES,
    discover_final_eeglab_paths,
    load_matrix,
    validate_matrix_file,
    validate_matrix_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "docs/parity/eeglab_final_parity_matrix.json"


def test_committed_final_parity_matrix_validates() -> None:
    _require_eeglab_reference()

    report = validate_matrix_file(MATRIX_PATH, REPO_ROOT)

    assert report.ok, [error.as_text() for error in report.errors]
    assert report.row_count == 31
    assert report.expected_eeglab_count == len(discover_final_eeglab_paths(REPO_ROOT)) == 180


def test_final_matrix_uses_complete_status_taxonomy() -> None:
    payload = load_matrix(MATRIX_PATH)
    statuses = {row["status"] for row in payload["rows"]}

    assert tuple(payload["metadata"]["status_taxonomy"]) == VALID_STATUSES
    assert statuses <= set(VALID_STATUSES)
    assert "optional_dependency" in statuses


def test_final_closeout_has_no_unimplemented_or_documentation_gap_rows() -> None:
    payload = load_matrix(MATRIX_PATH)
    remaining = [(row["row_id"], row["status"]) for row in payload["rows"] if row["status"] in {"port", "docs_gap"}]

    assert remaining == []


def test_final_partial_rows_are_explicit_backend_boundaries() -> None:
    payload = load_matrix(MATRIX_PATH)
    partial_rows = [row for row in payload["rows"] if row["status"] == "partial"]

    assert partial_rows
    for row in partial_rows:
        explanation = f"{row['rationale']} {row['test_notes']}".lower()
        assert "limitation" in explanation or "boundary" in explanation


def test_final_matrix_discovers_final_epic_paths_without_third_party_bloat() -> None:
    _require_eeglab_reference()

    paths = discover_final_eeglab_paths(REPO_ROOT)

    assert "plugins/clean_rawdata/asr_process_r.m" in paths
    assert "plugins/ICLabel/viewprops/pop_viewprops.m" in paths
    assert "functions/@memmapdata/memmapdata.m" in paths
    assert "tutorial_scripts/eeglab_history.m" in paths
    assert "plugins/ICLabel/matconvnet/matlab/vl_setupnn.m" not in paths
    assert "plugins/clean_rawdata/manopt/examples/nonlinear_eigenspace.m" not in paths


def test_final_matrix_uses_snapshot_when_nested_plugin_checkout_is_empty(tmp_path: Path) -> None:
    eeglab_root = tmp_path / "src/eegprep/eeglab"
    docs_root = tmp_path / "docs/parity"
    (eeglab_root / "functions/@eegobj").mkdir(parents=True)
    (eeglab_root / "plugins/clean_rawdata").mkdir(parents=True)
    docs_root.mkdir(parents=True)
    (eeglab_root / "functions/@eegobj/display.m").write_text("% display\n", encoding="utf-8")
    (docs_root / "eeglab_final_reference_paths.txt").write_text(
        "functions/@eegobj/display.m\nplugins/clean_rawdata/asr_process_r.m\n",
        encoding="utf-8",
    )

    paths = discover_final_eeglab_paths(tmp_path)

    assert paths == {"functions/@eegobj/display.m", "plugins/clean_rawdata/asr_process_r.m"}


def test_final_validator_fails_when_expected_path_is_unclassified() -> None:
    _require_eeglab_reference()

    payload = load_matrix(MATRIX_PATH)
    changed = copy.deepcopy(payload)
    removed = changed["rows"][0]["source_paths"].pop(0)

    report = validate_matrix_payload(changed, REPO_ROOT)

    assert not report.ok
    assert f"missing classification for {removed}" in _messages(report)


def test_final_validator_fails_when_expected_path_is_duplicated() -> None:
    _require_eeglab_reference()

    payload = load_matrix(MATRIX_PATH)
    changed = copy.deepcopy(payload)
    duplicated = changed["rows"][0]["source_paths"][0]
    changed["rows"][1]["source_paths"].append(duplicated)

    report = validate_matrix_payload(changed, REPO_ROOT)

    assert not report.ok
    assert f"duplicates expected EEGLAB source_path {duplicated!r}" in _messages(report)


def test_final_validator_requires_optional_dependency_contract() -> None:
    _require_eeglab_reference()

    payload = load_matrix(MATRIX_PATH)
    changed = copy.deepcopy(payload)
    row = next(row for row in changed["rows"] if row["status"] == "optional_dependency")
    row.pop("optional_dependency")

    report = validate_matrix_payload(changed, REPO_ROOT)

    assert not report.ok
    assert "optional_dependency: is required for optional_dependency rows" in _messages(report)


def test_final_validator_rejects_inconsistent_phase_ownership() -> None:
    _require_eeglab_reference()

    payload = load_matrix(MATRIX_PATH)
    changed = copy.deepcopy(payload)
    row = next(row for row in changed["rows"] if row["responsible_phase"] != "none")
    status = row["status"]
    row["responsible_phase"] = "none"
    row["phase_issue"] = None

    report = validate_matrix_payload(changed, REPO_ROOT)

    assert not report.ok
    assert f"responsible_phase: is required for {status!r}" in _messages(report)


def test_final_validator_rejects_phase_on_skip_rows() -> None:
    _require_eeglab_reference()

    payload = load_matrix(MATRIX_PATH)
    changed = copy.deepcopy(payload)
    row = next(row for row in changed["rows"] if row["status"] == "stale_skip")
    row["responsible_phase"] = "phase_7"
    row["phase_issue"] = changed["metadata"]["phase_issues"]["phase_7"]

    report = validate_matrix_payload(changed, REPO_ROOT)

    assert not report.ok
    assert "responsible_phase: must be none for skip rows" in _messages(report)
    assert "phase_issue: must be null for skip rows" in _messages(report)


def test_final_validator_requires_docs_architecture_sections() -> None:
    _require_eeglab_reference()

    payload = load_matrix(MATRIX_PATH)
    changed = copy.deepcopy(payload)
    changed["metadata"]["docs_architecture"]["sections"] = [
        section for section in changed["metadata"]["docs_architecture"]["sections"] if section["id"] != "concepts"
    ]

    report = validate_matrix_payload(changed, REPO_ROOT)

    assert not report.ok
    assert "must define section 'concepts'" in _messages(report)


def test_final_cli_json_report_is_machine_readable(capsys) -> None:
    _require_eeglab_reference()

    from tools.eeglab_final_parity_matrix import main

    exit_code = main([str(MATRIX_PATH), "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["ok"] is True
    assert report["expected_eeglab_count"] == 180


def _require_eeglab_reference() -> None:
    if discover_final_eeglab_paths(REPO_ROOT):
        return
    pytest.skip("EEGLAB reference tree is not initialized under src/eegprep/eeglab")


def _messages(report) -> str:
    return "\n".join(error.as_text() for error in report.errors)
