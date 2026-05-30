"""Build EEGLAB-style STUDY component preclustering arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._plot_utils import component_maps, python_literal
from eegprep.functions.popfunc._pop_utils import parse_numeric_sequence
from eegprep.functions.studyfunc._cluster_utils import (
    checked_study_and_datasets,
    cluster_command,
    dataset_for_study_set,
    ensure_parent_cluster,
    rows_for_cluster,
)


DEFAULT_PRECLUST = ({"measure": "scalp", "npca": 3, "norm": 1, "weight": 1, "abso": 1},)
COMPONENT_MEASURE_FIELDS = {
    "erp": ("erpdata", "erptimes", None),
    "spec": ("specdata", None, "specfreqs"),
    "ersp": ("erspdata", "ersptimes", "erspfreqs"),
    "itc": ("itcdata", "itctimes", "itcfreqs"),
}


def std_preclust(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    cluster_ind: int | None = 1,
    *preproc: Any,
    return_com: bool = False,
) -> Any:
    """Build the PCA-reduced matrix used by ``pop_clust``.

    Component ERP, spectrum, ERSP, and ITC inputs are read from the Phase 5b
    cached component measure fields on ``STUDY.cluster[0]``. Scalp maps and
    dipoles can be read directly from loaded ICA and DIPFIT fields.
    """
    study, datasets = checked_study_and_datasets(STUDY, ALLEEG)
    cluster_index = int(cluster_ind or 1)
    study = ensure_parent_cluster(study, datasets)
    sets, comps = rows_for_cluster(study, datasets, cluster_index)
    specs = normalize_preclust_specs(preproc)
    clustdata = np.empty((comps.size, 0), dtype=float)
    stored_specs: list[dict[str, Any]] = []
    finaldim: int | None = None

    for spec in specs:
        measure = str(spec["measure"]).lower()
        if measure == "finaldim":
            finaldim = int(spec.get("npca") or spec.get("dims") or 0) or None
            stored_specs.append(spec)
            continue
        raw = _measure_matrix(study, datasets, sets, comps, measure, spec)
        npca = _bounded_npca(spec.get("npca"), raw)
        reduced = _reduce_measure(raw, measure, npca)
        if int(spec.get("norm", 1)):
            reduced = _normalize_first_pc(reduced)
        weight = float(spec.get("weight", 1) or 1)
        clustdata = np.concatenate([clustdata, reduced * weight], axis=1)
        stored_specs.append(spec)

    if clustdata.size == 0:
        raise ValueError("std_preclust requires at least one component measure")
    if finaldim is not None and finaldim > 0 and clustdata.shape[1] > finaldim:
        clustdata = _pca_scores(clustdata, finaldim)

    study.setdefault("etc", {}).setdefault("eegprep", {})
    preclust = {
        "preclustdata": clustdata.tolist(),
        "preclustparams": stored_specs,
        "preclustcomps": _preclust_components(sets, comps),
        "clustlevel": cluster_index,
    }
    study["etc"]["preclust"] = preclust
    study["etc"]["eegprep"]["preclust_contract"] = {
        "component_measure_root": "STUDY.cluster[0]",
        "component_measure_fields": {measure: fields[0] for measure, fields in COMPONENT_MEASURE_FIELDS.items()},
        "rows": "STUDY.etc.preclust.preclustcomps",
    }
    study["cluster"][cluster_index - 1]["preclust"] = {
        "preclustparams": stored_specs,
        "preclustdata": clustdata.tolist(),
    }
    study["saved"] = "no"
    command = cluster_command(
        "std_preclust",
        ("STUDY", "ALLEEG"),
        "STUDY",
        "ALLEEG",
        python_literal(cluster_index),
        python_literal(stored_specs),
    )
    return (study, datasets, command) if return_com else (study, datasets)


def normalize_preclust_specs(preproc: tuple[Any, ...] | list[Any] | None) -> list[dict[str, Any]]:
    """Normalize EEGLAB cell-like measure specs into dictionaries."""
    items = list(preproc or [])
    if len(items) == 1 and isinstance(items[0], (list, tuple)) and not _looks_like_one_spec(items[0]):
        items = list(items[0])
    if not items:
        items = list(DEFAULT_PRECLUST)
    return [_normalize_one_spec(item) for item in items]


def _normalize_one_spec(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"measure": item.lower(), "npca": 5, "norm": 1, "weight": 1}
    if isinstance(item, dict):
        spec = {str(key).lower(): value for key, value in item.items()}
        spec["measure"] = str(spec.pop("command", spec.get("measure", ""))).lower()
    elif isinstance(item, (list, tuple)) and item:
        spec = {"measure": str(item[0]).lower()}
        if len(item[1:]) % 2:
            raise ValueError("preclust measure options must be key/value pairs")
        for index in range(1, len(item), 2):
            spec[str(item[index]).lower()] = item[index + 1]
    else:
        raise ValueError("preclust measures must be strings, dicts, or key/value sequences")
    if not spec.get("measure"):
        raise ValueError("preclust measure spec requires a measure name")
    if spec["measure"] != "finaldim":
        spec.setdefault("npca", 5)
        spec.setdefault("norm", 1)
        spec.setdefault("weight", 1)
    return spec


def _looks_like_one_spec(value: Any) -> bool:
    return bool(value) and isinstance(value[0], str)


def _measure_matrix(
    study: dict[str, Any],
    datasets: list[dict[str, Any]],
    sets: np.ndarray,
    comps: np.ndarray,
    measure: str,
    spec: dict[str, Any],
) -> np.ndarray:
    rows = []
    for row, comp in enumerate(comps):
        study_set = int(sets[0, row])
        eeg = dataset_for_study_set(study, datasets, study_set)
        if measure == "scalp":
            maps = component_maps(eeg)
            if comp < 1 or comp > maps.shape[1]:
                raise ValueError(f"Dataset {study_set} does not contain ICA component {comp}")
            values = maps[:, comp - 1]
            if int(spec.get("abso", 1)):
                values = np.abs(values)
        elif measure == "dipoles":
            values = _dipole_position(eeg, int(comp), study_set)
        elif measure == "moments":
            values = _dipole_moment(eeg, int(comp), study_set)
        else:
            values = _component_measure_values(study, study_set, int(comp), measure, spec)
        rows.append(np.asarray(values, dtype=float).ravel())
    width = {row.size for row in rows}
    if len(width) != 1:
        raise ValueError(f"Precomputed component measure '{measure}' has inconsistent feature lengths")
    return np.vstack(rows)


def _component_measure_values(
    study: dict[str, Any], study_set: int, component: int, measure: str, spec: dict[str, Any]
) -> np.ndarray:
    fields = COMPONENT_MEASURE_FIELDS.get(measure)
    if fields is None:
        raise ValueError(f"Unsupported preclustering measure '{measure}'")
    parent = (study.get("cluster") or [{}])[0]
    data_field, time_field, freq_field = fields
    if data_field not in parent:
        raise ValueError(
            f"Precomputed component measure '{measure}' is missing; "
            f"run pop_precomp(STUDY, ALLEEG, 'components', {measure}='on') before std_preclust"
        )
    data = np.asarray(parent[data_field], dtype=float)
    if data.ndim < 3:
        raise ValueError(f"Component measure '{measure}' requires dataset x component data")
    dataset_axis = _measure_axis(parent, "datasets", data.shape[0], fallback_start=1)
    component_axis = _measure_axis(parent, "components", data.shape[1], fallback_start=1)
    dataset_index = _axis_position(dataset_axis, study_set, f"dataset {study_set}", measure)
    component_index = _axis_position(component_axis, component, f"component {component}", measure)
    values = np.asarray(data[dataset_index, component_index], dtype=float)
    source = {"times": parent.get(time_field, []) if time_field is not None else []}
    if freq_field is not None:
        source["freqs"] = parent.get(freq_field, [])
    return _slice_measure_axes(values, source, measure, spec)


def _measure_axis(parent: dict[str, Any], key: str, count: int, *, fallback_start: int) -> np.ndarray:
    raw_measureinfo = parent.get("measureinfo")
    measureinfo: dict[str, Any] = raw_measureinfo if isinstance(raw_measureinfo, dict) else {}
    values = _as_int_vector(measureinfo.get(key))
    if values.size == count:
        return values
    if key == "datasets":
        values = _as_int_vector(parent.get("sets"))
    else:
        values = _as_int_vector(parent.get("comps"))
    unique_values = np.asarray(_unique_preserving_order(values.tolist()), dtype=int)
    if unique_values.size == count:
        return unique_values
    return np.arange(fallback_start, fallback_start + count, dtype=int)


def _as_int_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=int)
    array = np.asarray(value, dtype=int)
    if array.size == 0:
        return np.asarray([], dtype=int)
    return array.ravel()


def _axis_position(axis: np.ndarray, value: int, label: str, measure: str) -> int:
    matches = np.where(axis == value)[0]
    if matches.size == 0:
        raise ValueError(f"Component measure '{measure}' is missing {label}")
    return int(matches[0])


def _unique_preserving_order(values: list[int]) -> list[int]:
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _slice_measure_axes(values: np.ndarray, source: dict[str, Any], measure: str, spec: dict[str, Any]) -> np.ndarray:
    if measure == "erp":
        return _slice_one_axis(values, source.get("times"), spec.get("timewindow"))
    if measure == "spec":
        return _slice_one_axis(values, source.get("freqs"), spec.get("freqrange"))
    if measure in {"ersp", "itc"} and values.ndim >= 2:
        freqs = np.asarray(source.get("freqs") or [], dtype=float).ravel()
        times = np.asarray(source.get("times") or [], dtype=float).ravel()
        out = values
        freqrange = _bounds(spec.get("freqrange"))
        timewindow = _bounds(spec.get("timewindow"))
        if freqrange.size and freqs.size == out.shape[0]:
            out = out[(freqs >= freqrange[0]) & (freqs <= freqrange[1]), :]
        if timewindow.size and times.size == out.shape[1]:
            out = out[:, (times >= timewindow[0]) & (times <= timewindow[1])]
        return out.ravel()
    return values.ravel()


def _slice_one_axis(values: np.ndarray, axis_values: Any, bounds: Any) -> np.ndarray:
    parsed = _bounds(bounds)
    axis = np.asarray(axis_values or [], dtype=float).ravel()
    flat = values.ravel()
    if parsed.size == 0 or axis.size != flat.size:
        return flat
    mask = (axis >= parsed[0]) & (axis <= parsed[1])
    if not np.any(mask):
        raise ValueError("preclust measure range does not contain any samples")
    return flat[mask]


def _bounds(value: Any) -> np.ndarray:
    parsed = np.asarray(parse_numeric_sequence(value, dtype=float), dtype=float)
    if parsed.size == 0:
        return parsed
    if parsed.size != 2:
        raise ValueError("preclust ranges must contain [min max]")
    return parsed


def _dipole_position(eeg: dict[str, Any], component: int, study_set: int) -> np.ndarray:
    model = _dipole_model(eeg, component, study_set)
    pos = np.asarray(model.get("posxyz", []), dtype=float)
    if pos.ndim == 1 and pos.size == 3:
        return pos
    if pos.ndim == 2 and pos.shape[1] == 3:
        return pos[_leftmost_dipole(pos)]
    raise ValueError(f"Some dipole information is missing for component {component} of dataset {study_set}")


def _dipole_moment(eeg: dict[str, Any], component: int, study_set: int) -> np.ndarray:
    model = _dipole_model(eeg, component, study_set)
    pos = np.asarray(model.get("posxyz", []), dtype=float)
    mom = np.asarray(model.get("momxyz", []), dtype=float)
    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    if mom.ndim == 1:
        mom = mom.reshape(1, -1)
    if pos.shape[1:] != (3,) or mom.shape[1:] != (3,):
        raise ValueError(f"Some dipole moment information is missing for component {component} of dataset {study_set}")
    index = _leftmost_dipole(pos)
    values = mom[index]
    return -values if float(np.sum(pos[index] * values)) < 0 else values


def _dipole_model(eeg: dict[str, Any], component: int, study_set: int) -> dict[str, Any]:
    models = eeg.get("dipfit", {}).get("model", [])
    if isinstance(models, dict):
        models = [models]
    if component < 1 or component > len(models) or not isinstance(models[component - 1], dict):
        raise ValueError(f"Some dipole information is missing for component {component} of dataset {study_set}")
    return models[component - 1]


def _leftmost_dipole(pos: np.ndarray) -> int:
    nonzero = np.where(np.any(pos != 0, axis=1))[0]
    if nonzero.size == 0:
        return 0
    return int(nonzero[np.argmax(pos[nonzero, 1])])


def _bounded_npca(value: Any, data: np.ndarray) -> int:
    npca = 5 if value is None else int(value)
    return max(1, min(npca, data.shape[0], data.shape[1]))


def _reduce_measure(data: np.ndarray, measure: str, npca: int) -> np.ndarray:
    if measure == "dipoles":
        radius = np.sqrt(np.sum(data[:, :3] ** 2, axis=1))
        scale = float(np.std(radius)) or 1.0
        return data[:, :3] / scale
    scores = _pca_scores(data, npca)
    return np.abs(scores) if measure == "erp" else scores


def _pca_scores(data: np.ndarray, npca: int) -> np.ndarray:
    centered = np.nan_to_num(data - np.nanmean(data, axis=0, keepdims=True))
    u_mat, singular, _vh = np.linalg.svd(centered, full_matrices=False)
    return u_mat[:, :npca] * singular[:npca]


def _normalize_first_pc(data: np.ndarray) -> np.ndarray:
    scale = float(np.std(data[:, 0])) if data.size else 0.0
    if scale == 0:
        return data
    return data / scale


def _preclust_components(sets: np.ndarray, comps: np.ndarray) -> list[dict[str, int]]:
    return [{"set": int(sets[0, index]), "comp": int(comp)} for index, comp in enumerate(comps)]


__all__ = ["DEFAULT_PRECLUST", "normalize_preclust_specs", "std_preclust"]
