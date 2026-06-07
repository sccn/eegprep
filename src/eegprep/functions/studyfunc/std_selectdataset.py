"""Select STUDY datasets and trials by independent-variable values."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.studyfunc._study_utils import _empty_value, as_alleeg_list, ensure_study, sync_datasetinfo
from eegprep.functions.studyfunc.std_indvarmatch import std_indvarmatch


def std_selectdataset(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    indvar: str | None = "",
    indvarvals: Any = None,
    verboseFlag: str = "verbose",
) -> tuple[list[int], list[list[int]]]:
    """Return selected 1-based dataset indices and trial indices.

    Dataset-level variables are matched against ``STUDY.datasetinfo``. If the
    variable is not a dataset field, trial-level values are matched against
    ``STUDY.datasetinfo[*].trialinfo`` and dataset indices with at least one
    matching trial are returned.
    """
    _ = verboseFlag
    datasets = as_alleeg_list(ALLEEG)
    study = sync_datasetinfo(ensure_study(STUDY), datasets)
    infos = [info for info in study.get("datasetinfo") or [] if isinstance(info, dict)]
    trialselect = _all_trials(infos, datasets)
    if _empty_value(indvar):
        return list(range(1, len(infos) + 1)), trialselect

    label = str(indvar)
    values = _selection_values(indvarvals)
    if _dataset_field_present(infos, label):
        all_values = [info.get(label) for info in infos]
        selected: list[int] = []
        for value in values:
            selected.extend(std_indvarmatch(value, all_values))
        return _unique(selected), trialselect

    selected_trials: list[list[int]] = [[] for _info in infos]
    found_field = False
    for index, info in enumerate(infos):
        rows = _trial_rows(info.get("trialinfo"))
        if not rows:
            continue
        if any(label in row for row in rows):
            found_field = True
        row_indices: list[int] = []
        for value in values:
            row_indices.extend(
                row_index
                for row_index, row in enumerate(rows, start=1)
                if label in row and std_indvarmatch(value, [row.get(label)])
            )
        selected_trials[index] = _unique(row_indices)
    if not found_field:
        raise ValueError(f"Unknown STUDY independent variable: {label}")
    selected = [index for index, trials in enumerate(selected_trials, start=1) if trials]
    return selected, selected_trials


def _selection_values(value: Any) -> list[Any]:
    if isinstance(value, str) and " - " in value:
        return [part for part in value.split(" - ") if part]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _dataset_field_present(infos: list[dict[str, Any]], label: str) -> bool:
    return any(label in info and not _empty_value(info.get(label)) for info in infos)


def _all_trials(infos: list[dict[str, Any]], datasets: list[dict[str, Any]]) -> list[list[int]]:
    selections: list[list[int]] = []
    for index, info in enumerate(infos):
        rows = _trial_rows(info.get("trialinfo"))
        if rows:
            selections.append(list(range(1, len(rows) + 1)))
            continue
        trials = 1
        if index < len(datasets):
            trials = int(datasets[index].get("trials", 1) or 1)
        selections.append(list(range(1, trials + 1)))
    return selections


def _trial_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _unique(values: list[int]) -> list[int]:
    output: list[int] = []
    for value in values:
        value = int(value)
        if value not in output:
            output.append(value)
    return output


__all__ = ["std_selectdataset"]
