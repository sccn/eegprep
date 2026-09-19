"""Arrange dataset/component measure caches into STUDY design cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from eegprep.functions.popfunc.plot_utils import numeric_vector
from eegprep.functions.studyfunc._cluster_utils import cluster_list, sets_array
from eegprep.functions.studyfunc.std_readdata import component_dataset_axis, component_measure_axis


@dataclass(frozen=True)
class GroupedMeasure:
    """A condition-by-group measure grid with case metadata."""

    cells: list[list[np.ndarray]]
    conditions: list[str]
    groups: list[str]
    cases: list[list[list[str]]]

    def output(self) -> list[np.ndarray] | list[list[np.ndarray]]:
        """Use a flat condition list for one-factor designs."""
        if len(self.groups) == 1:
            return [row[0] for row in self.cells]
        return self.cells


def group_channel_measures(
    study: dict[str, Any],
    data: list[np.ndarray],
    datatype: str,
    *,
    design: int,
    subject: Any = None,
) -> GroupedMeasure:
    """Combine selected channel caches and split their dataset axis by design."""
    arrays = [np.asarray(values) for values in data]
    if not arrays:
        raise ValueError("No channel measure data were selected")
    expected = arrays[0].shape
    if any(array.shape != expected for array in arrays):
        raise ValueError("selected channel caches must have matching shapes")
    raw = np.stack(arrays, axis=1)
    if len(arrays) == 1:
        raw = raw[:, 0, ...]
    dataset_ids = np.arange(1, raw.shape[0] + 1, dtype=int)
    return _group_cases(study, raw, dataset_ids, datatype, design=design, subject=subject, component_cases=False)


def group_component_measures(
    study: dict[str, Any],
    data: np.ndarray,
    datatype: str,
    *,
    cluster_index: int,
    components: Any,
    design: int,
    subject: Any = None,
) -> GroupedMeasure:
    """Split parent or child component measures by STUDY design cells."""
    clusters = cluster_list(study)
    source = clusters[0]
    values = np.asarray(data)
    if cluster_index == 1:
        dataset_axis = component_dataset_axis(source, values.shape[0])
        component_axis = component_measure_axis(source, values.shape[1])
        selection = _component_positions(components, component_axis)
        rows = []
        dataset_ids = []
        case_labels = []
        for dataset_position, dataset_id in enumerate(dataset_axis.tolist()):
            for component_position in selection.tolist():
                row = values[dataset_position, component_position, ...]
                if np.isnan(row).all():
                    continue
                component_id = int(component_axis[component_position])
                rows.append(row)
                dataset_ids.append(int(dataset_id))
                case_labels.append(_component_label(study, int(dataset_id), component_id))
    else:
        cluster = clusters[cluster_index - 1]
        sets = sets_array(cluster.get("sets")).astype(int)[0]
        comps = np.asarray(cluster.get("comps") or [], dtype=int).ravel()
        positions = _cluster_member_positions(components, comps.size)
        rows = [values[position] for position in positions]
        dataset_ids = [int(sets[position]) for position in positions]
        case_labels = [_component_label(study, int(sets[position]), int(comps[position])) for position in positions]
    if not rows:
        raise ValueError("Selected STUDY cluster contains no cached components")
    return _group_cases(
        study,
        np.asarray(rows),
        np.asarray(dataset_ids, dtype=int),
        datatype,
        design=design,
        subject=subject,
        component_cases=True,
        case_labels=case_labels,
    )


def _group_cases(
    study: dict[str, Any],
    raw: np.ndarray,
    dataset_ids: np.ndarray,
    datatype: str,
    *,
    design: int,
    subject: Any,
    component_cases: bool,
    case_labels: list[str] | None = None,
) -> GroupedMeasure:
    design_info = _design(study, design)
    variables = [item for item in design_info.get("variable") or [] if isinstance(item, dict)]
    if len(variables) > 2:
        raise NotImplementedError("STUDY measure plots support at most two design variables")
    conditions = _levels(variables[0]) if variables else [("All conditions", None)]
    groups = _levels(variables[1]) if len(variables) > 1 else [("All groups", None)]
    datasetinfo = study.get("datasetinfo") or []
    selected_subjects = _subject_values(subject)
    allowed_cases = {str(value) for value in (design_info.get("cases") or {}).get("value", [])}

    cells: list[list[np.ndarray]] = []
    labels: list[list[list[str]]] = []
    for condition_label, condition_value in conditions:
        row = []
        row_labels = []
        for group_label, group_value in groups:
            positions = []
            labels_for_cell = []
            for position, dataset_id in enumerate(dataset_ids.tolist()):
                info = datasetinfo[dataset_id - 1]
                subject_name = str(info.get("subject") or f"S{dataset_id}")
                if selected_subjects and subject_name not in selected_subjects:
                    continue
                if allowed_cases and subject_name not in allowed_cases:
                    continue
                if variables and not _matches(info.get(variables[0].get("label")), condition_value):
                    continue
                if len(variables) > 1 and not _matches(info.get(variables[1].get("label")), group_value):
                    continue
                positions.append(position)
                labels_for_cell.append(case_labels[position] if case_labels else subject_name)
            selected = raw[np.asarray(positions, dtype=int), ...] if positions else raw[:0, ...]
            if not component_cases:
                selected, labels_for_cell = _average_repeated_subjects(selected, labels_for_cell)
            row.append(_case_last(selected, datatype))
            row_labels.append(labels_for_cell)
        cells.append(row)
        labels.append(row_labels)
    return GroupedMeasure(
        cells,
        [label for label, _value in conditions],
        [label for label, _value in groups],
        labels,
    )


def _design(study: dict[str, Any], design: int) -> dict[str, Any]:
    designs = study.get("design") or []
    if not designs:
        return {"variable": [], "cases": {"value": []}}
    if design < 1 or design > len(designs):
        raise ValueError(f"design must be 1-based and within 1..{len(designs)}")
    return designs[design - 1]


def _levels(variable: dict[str, Any]) -> list[tuple[str, Any]]:
    values = list(variable.get("value") or [])
    if not values:
        return [(str(variable.get("label") or "All"), None)]
    return [(_level_label(value), value) for value in values]


def _level_label(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " + ".join(str(item) for item in value)
    return str(value)


def _matches(actual: Any, level: Any) -> bool:
    if level is None:
        return True
    if isinstance(level, (list, tuple)):
        return any(_matches(actual, item) for item in level)
    return _equal(actual, level)


def _equal(left: Any, right: Any) -> bool:
    try:
        return bool(float(left) == float(right))
    except (TypeError, ValueError):
        return str(left) == str(right)


def _average_repeated_subjects(values: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    ordered = list(dict.fromkeys(labels))
    if len(ordered) == len(labels):
        return values, labels
    averaged = [np.nanmean(values[np.asarray([label == subject for label in labels])], axis=0) for subject in ordered]
    return np.asarray(averaged), ordered


def _case_last(values: np.ndarray, datatype: str) -> np.ndarray:
    if datatype in {"erp", "spec"}:
        if values.ndim not in {2, 3}:
            raise ValueError("line-measure caches must have case, sample, and optional channel axes")
        order = (values.ndim - 1,) if values.ndim == 2 else (2, 1)
        return np.transpose(values, (*order, 0))
    if values.ndim not in {3, 4}:
        raise ValueError("time-frequency caches must have case, frequency, time, and optional channel axes")
    order = (1, 2) if values.ndim == 3 else (2, 3, 1)
    return np.transpose(values, (*order, 0))


def _component_positions(components: Any, axis: np.ndarray) -> np.ndarray:
    if components is None or (isinstance(components, str) and components.lower() == "all"):
        return np.arange(axis.size, dtype=int)
    requested = numeric_vector(components, dtype=int)
    positions = []
    for component in requested.tolist():
        found = np.where(axis == int(component))[0]
        if not found.size:
            raise ValueError(f"component {component} is not present in the parent cluster cache")
        positions.append(int(found[0]))
    return np.asarray(positions, dtype=int)


def _cluster_member_positions(components: Any, count: int) -> np.ndarray:
    if components is None or (isinstance(components, str) and components.lower() == "all"):
        return np.arange(count, dtype=int)
    requested = numeric_vector(components, dtype=int)
    if np.any(requested < 1) or np.any(requested > count):
        raise ValueError(f"comps must be 1-based cluster member positions within 1..{count}")
    return requested - 1


def _component_label(study: dict[str, Any], dataset_id: int, component_id: int) -> str:
    info = (study.get("datasetinfo") or [{}])[dataset_id - 1]
    return f"{info.get('subject') or f'S{dataset_id}'}/IC{component_id}"


def _subject_values(subject: Any) -> set[str]:
    if subject is None or subject == "":
        return set()
    if isinstance(subject, str):
        return {subject}
    if isinstance(subject, np.ndarray):
        subject = subject.tolist()
    if isinstance(subject, (list, tuple, set)):
        return {str(item) for item in subject}
    return {str(subject)}


__all__ = ["GroupedMeasure", "group_channel_measures", "group_component_measures"]
