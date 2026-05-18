"""Create a minimal EEGPrep STUDY from loaded datasets."""

from __future__ import annotations

from typing import Any


def pop_study(
    STUDY: dict[str, Any] | None = None,
    ALLEEG: list[dict[str, Any]] | None = None,
    *,
    name: str = "EEGPrep study",
    design: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Create a STUDY structure from loaded EEG datasets."""
    datasets = [] if ALLEEG is None else list(ALLEEG)
    study = {} if STUDY is None else dict(STUDY)
    study.update(
        {
            "name": name or study.get("name") or "EEGPrep study",
            "datasetinfo": [_dataset_info(index, eeg) for index, eeg in enumerate(datasets, start=1)],
            "design": [] if design is None else [{"name": design}],
            "etc": study.get("etc", {}),
        }
    )
    return study, datasets, "STUDY = pop_study([], ALLEEG, 'gui', 'on');"


def _dataset_info(index: int, eeg: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "setname": eeg.get("setname", ""),
        "filename": eeg.get("filename", ""),
        "filepath": eeg.get("filepath", ""),
        "subject": eeg.get("subject", ""),
        "condition": eeg.get("condition", ""),
        "session": eeg.get("session", []),
        "group": eeg.get("group", ""),
    }
