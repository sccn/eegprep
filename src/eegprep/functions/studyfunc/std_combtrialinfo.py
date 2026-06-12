"""Combine STUDY dataset-level metadata into trialinfo rows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.studyfunc._study_utils import trialinfo_rows


DATASETINFO_TRIAL_EXCLUDE = {"filepath", "filename", "comps", "trialinfo"}


def std_combtrialinfo(datasetinfo: Any, inds: Any, trials: Any = None) -> list[dict[str, Any]]:
    """Return trial rows enriched with selected ``datasetinfo`` fields."""
    infos = _datasetinfo_rows(datasetinfo)
    selected = _selected_indices(infos, inds)
    trial_counts = _trial_counts(infos, trials)
    rows: list[dict[str, Any]] = []
    for index in selected:
        info = infos[index - 1]
        base_rows = trialinfo_rows(info.get("trialinfo"))
        if not base_rows:
            base_rows = [{} for _trial in range(trial_counts[index - 1])]
        for base in base_rows:
            row = deepcopy(base)
            for key, value in info.items():
                if key not in DATASETINFO_TRIAL_EXCLUDE:
                    row[key] = deepcopy(value)
            rows.append(row)
    return rows


def _datasetinfo_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _selected_indices(infos: list[dict[str, Any]], inds: Any) -> list[int]:
    if isinstance(inds, str):
        selected = [index for index, info in enumerate(infos, start=1) if str(info.get("subject") or "") == inds]
        if not selected:
            raise ValueError(f"Subject name {inds!r} is invalid")
        return selected
    if isinstance(inds, np.ndarray):
        inds = inds.tolist()
    if not isinstance(inds, (list, tuple)):
        inds = [inds]
    selected = [int(index) for index in inds]
    invalid = [index for index in selected if index < 1 or index > len(infos)]
    if invalid:
        raise ValueError(f"datasetinfo indices out of range: {invalid}")
    return selected


def _trial_counts(infos: list[dict[str, Any]], trials: Any) -> list[int]:
    if trials is None:
        return [max(1, len(trialinfo_rows(info.get("trialinfo")))) for info in infos]
    if isinstance(trials, np.ndarray):
        trials = trials.tolist()
    if not isinstance(trials, (list, tuple)):
        trials = [trials for _info in infos]
    counts = [int(value) for value in trials]
    if len(counts) < len(infos):
        counts.extend([1] * (len(infos) - len(counts)))
    return counts[: len(infos)]


__all__ = ["std_combtrialinfo"]
