"""Assign EEGLAB-style levels to STUDY design variables."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.studyfunc._study_utils import _empty_value, ensure_study


def std_addvarlevel(STUDY: dict[str, Any], designind: int | list[int] | None = None) -> dict[str, Any]:
    """Return ``STUDY`` with ``one``/``two`` design-variable levels.

    EEGLAB marks variables from per-trial metadata as first-level (``"one"``)
    and stable dataset/subject metadata as second-level (``"two"``). Dataset
    metadata that changes within one subject is first-level because it cannot
    be modeled as an inter-subject factor.
    """
    study = ensure_study(STUDY)
    study = deepcopy(study)
    designs = [design for design in study.get("design") or [] if isinstance(design, dict)]
    if not designs:
        study["design"] = []
        return study

    selected = _design_indices(designs, designind)
    datasetinfo = [info for info in study.get("datasetinfo") or [] if isinstance(info, dict)]
    subjects = [str(info.get("subject") or "") for info in datasetinfo]
    for index in selected:
        design = designs[index - 1]
        variables = []
        for variable in design.get("variable") or []:
            if not isinstance(variable, dict):
                continue
            updated = deepcopy(variable)
            if "levelindex" not in updated:
                updated["levelindex"] = _level_indices(updated)
            updated["level"] = _variable_level(updated.get("label"), datasetinfo, subjects)
            variables.append(updated)
        design["variable"] = variables
        designs[index - 1] = design
    study["design"] = designs
    return study


def _design_indices(designs: list[dict[str, Any]], designind: int | list[int] | None) -> list[int]:
    if designind is None:
        return list(range(1, len(designs) + 1))
    if isinstance(designind, (list, tuple)):
        indices = [int(item) for item in designind]
    else:
        indices = [int(designind)]
    for index in indices:
        if index < 1 or index > len(designs):
            raise ValueError(f"designind must be 1-based and within 1..{len(designs)}")
    return indices


def _variable_level(label: Any, datasetinfo: list[dict[str, Any]], subjects: list[str]) -> str:
    label = str(label or "")
    if not label or not any(label in info for info in datasetinfo):
        return "one"
    for subject in sorted(set(subjects)):
        values = [info.get(label) for info, current in zip(datasetinfo, subjects) if current == subject]
        values = [_hashable_value(value) for value in values if not _empty_value(value)]
        if len(set(values)) > 1:
            return "one"
    return "two"


def _level_indices(variable: dict[str, Any]) -> list[int]:
    values = variable.get("value")
    if isinstance(values, (list, tuple)):
        return list(range(1, len(values) + 1))
    return []


def _hashable_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return repr(tuple(_hashable_value(item) for item in value))
    return repr(value)


__all__ = ["std_addvarlevel"]
