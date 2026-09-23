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
    caches: list[dict[str, Any]] | None = None,
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
    trial_values, trialinfo = _channel_trial_cases(caches or [], datatype, raw.shape[0])
    return _group_cases(
        study,
        raw,
        dataset_ids,
        datatype,
        design=design,
        subject=subject,
        component_cases=False,
        trial_values=trial_values,
        trialinfo=trialinfo,
    )


def group_component_measures(
    study: dict[str, Any],
    data: np.ndarray,
    datatype: str,
    *,
    cluster_index: int,
    components: Any,
    design: int,
    subject: Any = None,
    source_cache: dict[str, Any] | None = None,
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
        trial_values = []
        trialinfo = []
        for dataset_position, dataset_id in enumerate(dataset_axis.tolist()):
            for component_position in selection.tolist():
                row = values[dataset_position, component_position, ...]
                if np.isnan(row).all():
                    continue
                component_id = int(component_axis[component_position])
                rows.append(row)
                dataset_ids.append(int(dataset_id))
                case_labels.append(_component_label(study, int(dataset_id), component_id))
                trial_value, trial_rows = _component_trial_case(
                    source_cache or source, datatype, dataset_position, component_position
                )
                trial_values.append(trial_value)
                trialinfo.append(trial_rows)
    else:
        cluster = clusters[cluster_index - 1]
        sets = sets_array(cluster.get("sets")).astype(int)[0]
        comps = np.asarray(cluster.get("comps") or [], dtype=int).ravel()
        positions = _cluster_member_positions(components, comps.size)
        rows = [values[position] for position in positions]
        dataset_ids = [int(sets[position]) for position in positions]
        case_labels = [_component_label(study, int(sets[position]), int(comps[position])) for position in positions]
        trial_values = []
        trialinfo = []
        source_cache = source_cache or source
        dataset_axis = component_dataset_axis(
            source_cache, np.asarray(source_cache.get(_data_field(datatype))).shape[0]
        )
        component_axis = component_measure_axis(
            source_cache, np.asarray(source_cache.get(_data_field(datatype))).shape[1]
        )
        for position in positions.tolist():
            dataset_position = _axis_position(dataset_axis, int(sets[position]), "dataset")
            component_position = _axis_position(component_axis, int(comps[position]), "component")
            trial_value, trial_rows = _component_trial_case(
                source_cache, datatype, dataset_position, component_position
            )
            trial_values.append(trial_value)
            trialinfo.append(trial_rows)
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
        trial_values=trial_values,
        trialinfo=trialinfo,
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
    trial_values: list[np.ndarray | None] | None = None,
    trialinfo: list[list[dict[str, Any]]] | None = None,
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
            selected_values = []
            labels_for_cell = []
            for position, dataset_id in enumerate(dataset_ids.tolist()):
                info = datasetinfo[dataset_id - 1]
                subject_name = str(info.get("subject") or f"S{dataset_id}")
                if selected_subjects and subject_name not in selected_subjects:
                    continue
                if allowed_cases and subject_name not in allowed_cases:
                    continue
                levels = [condition_value, group_value]
                trial_value = trial_values[position] if trial_values else None
                trial_rows = trialinfo[position] if trialinfo else []
                if not trial_rows:
                    trial_rows = [row for row in info.get("trialinfo") or [] if isinstance(row, dict)]
                matches, trial_mask = _case_selection(info, variables, levels, trial_rows, trial_value)
                if not matches:
                    continue
                value = raw[position]
                if trial_mask is not None:
                    value = _aggregate_trials(trial_value[..., trial_mask], datatype)
                selected_values.append(value)
                labels_for_cell.append(case_labels[position] if case_labels else subject_name)
            selected = np.asarray(selected_values) if selected_values else raw[:0, ...]
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


def _case_selection(
    info: dict[str, Any],
    variables: list[dict[str, Any]],
    levels: list[Any],
    trialinfo: list[dict[str, Any]],
    trial_values: np.ndarray | None,
) -> tuple[bool, np.ndarray | None]:
    trial_mask = None
    for variable, level in zip(variables, levels):
        label = str(variable.get("label") or "")
        if label in info and _has_dataset_value(info[label]):
            if not _matches(info[label], level):
                return False, None
            continue
        if not any(label in row for row in trialinfo):
            return False, None
        if trial_values is None:
            raise ValueError(f"design variable {label!r} is trial-level; rerun std_precomp with savetrials='on'")
        if len(trialinfo) != trial_values.shape[-1]:
            raise ValueError("single-trial measure cache and trialinfo lengths do not match")
        current = np.asarray([_matches(row.get(label), level) for row in trialinfo], dtype=bool)
        trial_mask = current if trial_mask is None else trial_mask & current
    if trial_mask is not None and not np.any(trial_mask):
        return False, None
    return True, trial_mask


def _has_dataset_value(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and value == "")


def _aggregate_trials(values: np.ndarray, datatype: str) -> np.ndarray:
    if datatype in {"spec", "ersp"}:
        power = np.nanmean(values, axis=-1)
        return 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))
    if datatype == "itc":
        return np.abs(np.nanmean(np.exp(1j * values), axis=-1))
    return np.nanmean(values, axis=-1)


def _channel_trial_cases(
    caches: list[dict[str, Any]], datatype: str, dataset_count: int
) -> tuple[list[np.ndarray | None], list[list[dict[str, Any]]]]:
    field = _trial_field(datatype)
    info_field = _trialinfo_field(datatype)
    values: list[np.ndarray | None] = []
    all_trialinfo: list[list[dict[str, Any]]] = []
    for dataset_position in range(dataset_count):
        channel_trials = []
        rows: list[dict[str, Any]] = []
        for cache in caches:
            stored = cache.get(field)
            if not isinstance(stored, list) or dataset_position >= len(stored):
                channel_trials = []
                break
            channel_trials.append(np.asarray(stored[dataset_position], dtype=float))
            cached_rows = cache.get(info_field)
            if isinstance(cached_rows, list) and dataset_position < len(cached_rows):
                rows = [row for row in cached_rows[dataset_position] if isinstance(row, dict)]
        if channel_trials:
            shape = channel_trials[0].shape
            if any(item.shape != shape for item in channel_trials):
                raise ValueError("selected channel single-trial caches must have matching shapes")
            values.append(channel_trials[0] if len(channel_trials) == 1 else np.stack(channel_trials, axis=0))
        else:
            values.append(None)
        all_trialinfo.append(rows)
    return values, all_trialinfo


def _component_trial_case(
    source: dict[str, Any], datatype: str, dataset_position: int, component_position: int
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    stored = source.get(_trial_field(datatype))
    value = None
    if isinstance(stored, list) and dataset_position < len(stored):
        dataset = stored[dataset_position]
        if isinstance(dataset, list) and component_position < len(dataset) and dataset[component_position] is not None:
            value = np.asarray(dataset[component_position], dtype=float)
    stored_rows = source.get(_trialinfo_field(datatype))
    rows = []
    if isinstance(stored_rows, list) and dataset_position < len(stored_rows):
        rows = [row for row in stored_rows[dataset_position] if isinstance(row, dict)]
    return value, rows


def _trial_field(datatype: str) -> str:
    return f"{datatype}datatrials"


def _trialinfo_field(datatype: str) -> str:
    return f"{datatype}trialinfo"


def _data_field(datatype: str) -> str:
    return f"{datatype}data"


def _axis_position(axis: np.ndarray, value: int, label: str) -> int:
    found = np.where(axis == value)[0]
    if not found.size:
        raise ValueError(f"{label} {value} is not present in the parent cluster cache")
    return int(found[0])


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
