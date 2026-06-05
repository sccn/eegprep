from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.eeglab_parity_matrix import (
    VALID_STATUSES,
    discover_in_scope_eeglab_paths,
    load_matrix,
    validate_matrix_file,
    validate_matrix_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "docs/parity/eeglab_core_parity_matrix.json"


def test_committed_eeglab_parity_matrix_validates() -> None:
    _require_eeglab_reference()

    report = validate_matrix_file(MATRIX_PATH, REPO_ROOT)

    assert report.ok, [error.as_text() for error in report.errors]
    assert report.row_count == report.expected_eeglab_count == len(discover_in_scope_eeglab_paths(REPO_ROOT))


def test_parity_matrix_uses_complete_status_taxonomy() -> None:
    payload = load_matrix(MATRIX_PATH)
    statuses = {row["status"] for row in payload["rows"]}

    assert statuses == set(VALID_STATUSES)


def test_validator_fails_when_in_scope_eeglab_function_is_unclassified() -> None:
    _require_eeglab_reference()

    payload = load_matrix(MATRIX_PATH)
    changed = copy.deepcopy(payload)
    removed = changed["rows"].pop(0)

    report = validate_matrix_payload(changed, REPO_ROOT)

    assert not report.ok
    assert f"missing classification for {removed['eeglab_path']}" in _messages(report)


def test_validator_fails_when_matrix_classifies_out_of_scope_eeglab_path() -> None:
    _require_eeglab_reference()

    payload = load_matrix(MATRIX_PATH)
    changed = copy.deepcopy(payload)
    changed["rows"][0]["eeglab_path"] = "functions/popfunc/not_a_real_function.m"
    changed["rows"][0]["eeglab_name"] = "not_a_real_function"

    report = validate_matrix_payload(changed, REPO_ROOT)

    assert not report.ok
    assert "classifies out-of-scope EEGLAB path functions/popfunc/not_a_real_function.m" in _messages(report)


def test_validator_reports_missing_eeglab_reference_tree_clearly(tmp_path: Path) -> None:
    payload = load_matrix(MATRIX_PATH)

    report = validate_matrix_payload(payload, tmp_path)

    messages = _messages(report)
    assert not report.ok
    assert "EEGLAB reference tree is missing or empty" in messages
    assert "classifies out-of-scope EEGLAB path" not in messages


def test_validator_rejects_incomplete_stale_skip_policy() -> None:
    _require_eeglab_reference()

    payload = load_matrix(MATRIX_PATH)
    changed = copy.deepcopy(payload)
    stale_row = next(row for row in changed["rows"] if row["status"] == "stale_skip")
    stale_row["stale_policy"]["likely_user_alias"] = True

    report = validate_matrix_payload(changed, REPO_ROOT)

    assert not report.ok
    assert "stale_policy.likely_user_alias: must be false" in _messages(report)


def test_cli_json_report_is_machine_readable(capsys) -> None:
    _require_eeglab_reference()

    from tools.eeglab_parity_matrix import main

    exit_code = main([str(MATRIX_PATH), "--json"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 0
    assert report["ok"] is True
    assert report["row_count"] == report["expected_eeglab_count"]


def test_development_docs_explain_matrix_updates_and_runtime_boundary() -> None:
    docs = (REPO_ROOT / "docs/source/development.rst").read_text(encoding="utf-8")

    assert "docs/parity/eeglab_core_parity_matrix.json" in docs
    assert "uv run --no-sync python -m tools.eeglab_parity_matrix" in docs
    assert "package code under ``src/eegprep`` must not read, import" in docs
    assert "Use ``stale_skip`` only when every stale-policy field" in docs


def _require_eeglab_reference() -> None:
    if discover_in_scope_eeglab_paths(REPO_ROOT):
        return
    pytest.skip("EEGLAB reference tree is not initialized under src/eegprep/eeglab")


def _messages(report) -> str:
    return "\n".join(error.as_text() for error in report.errors)
