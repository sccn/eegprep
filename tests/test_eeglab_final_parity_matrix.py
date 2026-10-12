from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools.eeglab_final_parity_matrix import (
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


def _require_eeglab_reference() -> None:
    if discover_final_eeglab_paths(REPO_ROOT):
        return
    pytest.skip("EEGLAB reference tree is not initialized under src/eegprep/eeglab")


def _messages(report) -> str:
    return "\n".join(error.as_text() for error in report.errors)
