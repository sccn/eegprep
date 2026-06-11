"""Find dataset-level variables available for STUDY designs."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.studyfunc._study_utils import RESERVED_VARIABLE_FIELDS, _empty_value, ensure_study


def std_findgroupvars(STUDY: dict[str, Any]) -> list[str]:
    """Return dataset variables that are constant within each subject."""
    study = ensure_study(STUDY)
    datasetinfo = [info for info in study.get("datasetinfo") or [] if isinstance(info, dict)]
    subjects = _unique([str(info.get("subject") or "") for info in datasetinfo])
    names = _candidate_fields(datasetinfo)
    valid: list[str] = []
    for name in names:
        values_by_subject = {subject: [] for subject in subjects}
        for info in datasetinfo:
            value = info.get(name)
            if _empty_value(value):
                continue
            values_by_subject[str(info.get("subject") or "")].append(value)
        if values_by_subject and all(len(_unique_values(values)) == 1 for values in values_by_subject.values()):
            valid.append(name)
    preferred = [field for field in ("condition", "group", "session", "run") if field in valid]
    return preferred + sorted(name for name in valid if name not in preferred)


def _candidate_fields(datasetinfo: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for info in datasetinfo:
        if not isinstance(info, dict):
            continue
        for key in info:
            if key in RESERVED_VARIABLE_FIELDS or key == "subject":
                continue
            if key not in names:
                names.append(key)
    return names


def _unique(values: list[Any]) -> list[Any]:
    output: list[Any] = []
    for value in values:
        if not any(_same_value(value, existing) for existing in output):
            output.append(value)
    return output


def _unique_values(values: list[Any]) -> list[Any]:
    return _unique([value for value in values if not _empty_value(value)])


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            list(left) == list(right) if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) else False
        )
    return left == right


__all__ = ["std_findgroupvars"]
