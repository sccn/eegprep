from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools.eeglab_parity_matrix import (
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


def test_explicit_plugin_rows_are_in_scope_when_reference_functions_exist(tmp_path: Path) -> None:
    function_root = tmp_path / "src/eegprep/eeglab/functions/popfunc"
    function_root.mkdir(parents=True)
    (function_root / "pop_dummy.m").write_text("function pop_dummy\nend\n", encoding="utf-8")

    paths = discover_in_scope_eeglab_paths(tmp_path)

    assert "functions/popfunc/pop_dummy.m" in paths
    assert "plugins/clean_rawdata/clean_asr.m" in paths


def _require_eeglab_reference() -> None:
    if discover_in_scope_eeglab_paths(REPO_ROOT):
        return
    pytest.skip("EEGLAB reference tree is not initialized under src/eegprep/eeglab")


def _messages(report) -> str:
    return "\n".join(error.as_text() for error in report.errors)
