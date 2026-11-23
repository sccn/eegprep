"""Selection rules for ICLabel int8 class-coverage candidates.

The frozen ds002680 evaluation set has no Heart, Line Noise, or Channel Noise
components and one Muscle component. These tests pin the finder to a synthetic
teacher matrix so it keeps only those four classes and refuses the frozen subjects.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from eegprep.plugins.ICLabel.pop_icflag import ICLABEL_CLASSES
from tools.iclabel.find_class_candidates import (
    file_sha256,
    find_class_candidates,
    verify_annex_key,
)


def _probabilities() -> np.ndarray:
    """One teacher row per class, then a second Muscle row. Column order is ICLABEL_CLASSES."""
    rows = []
    for index in range(len(ICLABEL_CLASSES)):
        row = np.full(len(ICLABEL_CLASSES), 0.01)
        row[index] = 0.94
        rows.append(row)
    muscle = np.full(len(ICLABEL_CLASSES), 0.02)
    muscle[1] = 0.88
    rows.append(muscle)
    return np.asarray(rows, dtype=np.float64)


def _provenance(**overrides) -> dict:
    record = {
        "dataset_id": "ds007955",
        "version_or_doi": "10.18112/openneuro.ds007955.v1.0.0",
        "path": "sub-01/eeg/sub-01_task-wordassociation_eeg.set",
        "annex_key": "SHA256E-s3215952--5ee94efd108d534def2b7d5e93b25ac15ff1f8258b12b2434173ea0a22d8d126.set",
        "sha256": "5ee94efd108d534def2b7d5e93b25ac15ff1f8258b12b2434173ea0a22d8d126",
        "ica_method": "picard",
        "ica_seed": 404,
        "subject": "sub-01",
    }
    record.update(overrides)
    return record


def test_synthetic_probabilities_keep_only_uncovered_classes():
    records = find_class_candidates(_probabilities(), _provenance())

    assert [(record["component_index"], record["teacher_class"]) for record in records] == [
        (1, "Muscle"),
        (3, "Heart"),
        (4, "Line Noise"),
        (5, "Channel Noise"),
        (7, "Muscle"),
    ]
    assert records[1]["teacher_probability"] == pytest.approx(0.94)
    assert records[4]["teacher_probability"] == pytest.approx(0.88)
    for record in records:
        assert record["dataset_id"] == "ds007955"
        assert record["version_or_doi"] == "10.18112/openneuro.ds007955.v1.0.0"
        assert record["path"].endswith(".set")
        assert record["annex_key"].startswith("SHA256E-")
        assert record["sha256"] == _provenance()["sha256"]
        assert record["ica_method"] == "picard"
        assert record["ica_seed"] == 404
        assert record["subject"] == "sub-01"


def test_empty_component_matrix_returns_no_candidates():
    assert find_class_candidates(np.zeros((0, 7)), _provenance()) == []


@pytest.mark.parametrize("subject", ["sub-002", "sub-015", "002", "15"])
@pytest.mark.parametrize("dataset_id", ["ds002680", "on002680"])
def test_frozen_evaluation_subjects_are_refused(dataset_id, subject):
    with pytest.raises(ValueError, match="find_class_candidates") as caught:
        find_class_candidates(_probabilities(), _provenance(dataset_id=dataset_id, subject=subject))

    message = str(caught.value)
    assert "sub-002" in message and "sub-015" in message
    assert "different subject or dataset" in message


@pytest.mark.parametrize(
    ("dataset_id", "subject"),
    [
        ("ds002680", "sub-001"),
        ("ds002680", "sub-016"),
        ("ds007955", "sub-002"),
        ("on007955", "sub-015"),
    ],
)
def test_recordings_outside_the_frozen_subjects_are_kept(dataset_id, subject):
    records = find_class_candidates(_probabilities(), _provenance(dataset_id=dataset_id, subject=subject))

    assert records
    assert {record["subject"] for record in records} == {subject}
    assert {record["dataset_id"] for record in records} == {dataset_id}


def test_missing_provenance_names_the_field_and_the_fix():
    provenance = _provenance()
    del provenance["sha256"]

    with pytest.raises(ValueError, match="find_class_candidates") as caught:
        find_class_candidates(_probabilities(), provenance)

    message = str(caught.value)
    assert "sha256" in message
    assert "Add sha256" in message


def test_probability_matrix_must_have_one_column_per_class():
    with pytest.raises(ValueError, match="find_class_candidates") as caught:
        find_class_candidates(np.zeros((3, 6)), _provenance())

    assert "(components, 7)" in str(caught.value)


def test_nonfinite_probabilities_are_refused():
    probabilities = _probabilities()
    probabilities[3, 3] = np.nan

    with pytest.raises(ValueError, match="find_class_candidates") as caught:
        find_class_candidates(probabilities, _provenance())

    assert "finite" in str(caught.value)


def test_annex_key_must_match_the_recording_bytes(tmp_path: Path):
    payload = b"tiny-recording"
    path = tmp_path / "recording.set"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    key = f"SHA256E-s{len(payload)}--{digest}.set"

    assert file_sha256(path) == digest
    verify_annex_key(path, key)

    with pytest.raises(ValueError, match="find_class_candidates") as caught:
        verify_annex_key(path, "SHA256E-s" + str(len(payload)) + "--" + "0" * 64 + ".set")

    message = str(caught.value)
    assert "annex key" in message
    assert digest in message
