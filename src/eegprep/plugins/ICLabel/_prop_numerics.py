"""Data assembly and numerical helpers for ICLabel property dashboards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from eegprep.functions.popfunc._plot_utils import (
    channel_labels,
    component_activations,
    component_channel_indices,
    component_map_data,
    eeg_epoch_data,
    eeg_times_ms,
    numeric_vector,
    parse_plot_options_text,
)
from eegprep.functions.popfunc._rejection import component_rejection_flags, one_based_indices
from eegprep.functions.sigprocfunc.spectopo import compute_spectra
from eegprep.plugins.dipfit._utils import normalize_model_list


DEFAULT_ICLABEL_CLASSES = ("Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other")


@dataclass(frozen=True)
class ClassifierData:
    """Normalized component-classifier output from ``EEG.etc.ic_classification``."""

    name: str
    classes: tuple[str, ...]
    probabilities: np.ndarray


@dataclass(frozen=True)
class DipfitData:
    """Normalized localized DIPFIT model for one ICA component."""

    positions: np.ndarray
    moments: np.ndarray | None
    rv_percent: float | None
    dmr: float | None
    coordformat: str


@dataclass(frozen=True)
class ExtendedPropertyData:
    """Data assembled for one extended channel/component property dashboard."""

    typecomp: int
    index: int
    label: str
    figure_title: str
    topography_title: str
    topography_values: Any
    topography_chanlocs: list[dict[str, Any]]
    activity: np.ndarray
    times_ms: np.ndarray
    activity_title: str
    image_data: np.ndarray
    image_extent: tuple[float, float, float, float]
    image_title: str
    spectrum_freqs: np.ndarray
    spectrum_power: np.ndarray
    spectrum_title: str
    classifier: ClassifierData | None
    class_probabilities: np.ndarray | None
    pvaf: float | None
    dipfit: DipfitData | None
    rejected: bool | None


def classifier_names(EEG: dict[str, Any]) -> list[str]:
    """Return classifier field names available under ``EEG.etc.ic_classification``."""
    etc = EEG.get("etc") or {}
    if not isinstance(etc, dict):
        return []
    classifications = etc.get("ic_classification") or {}
    if not isinstance(classifications, dict):
        return []
    return [str(name) for name in classifications if str(name)]


def classifier_default_index(classifiers: list[str]) -> int:
    """Return EEGLAB popup index for the default component classifier."""
    for index, name in enumerate(classifiers, start=1):
        if name.lower() == "iclabel":
            return index
    return 1


def classifier_name_from_gui(EEG: dict[str, Any], value: Any) -> str:
    """Resolve a GUI popup value to a classifier field name."""
    classifiers = classifier_names(EEG)
    if not classifiers:
        return ""
    if isinstance(value, str):
        for classifier in classifiers:
            if classifier.lower() == value.lower():
                return classifier
        return classifiers[classifier_default_index(classifiers) - 1]
    try:
        index = int(value) - 1
    except (TypeError, ValueError):
        index = classifier_default_index(classifiers) - 1
    if 0 <= index < len(classifiers):
        return classifiers[index]
    return classifiers[classifier_default_index(classifiers) - 1]


def resolve_classifier_data(
    EEG: dict[str, Any],
    classifier_name: str = "",
    *,
    component_total: int | None = None,
    require: bool = False,
) -> ClassifierData | None:
    """Return normalized classifier data or ``None`` when no classifier is available."""
    classifiers = classifier_names(EEG)
    if not classifiers:
        if require:
            raise ValueError("No component classifier data found in EEG.etc.ic_classification")
        return None
    resolved_name = _resolve_classifier_name(classifiers, classifier_name)
    record = (EEG.get("etc") or {})["ic_classification"][resolved_name]
    if not isinstance(record, dict):
        raise ValueError(f"Classifier {resolved_name!r} must be stored as a dictionary")
    probabilities = np.asarray(record.get("classifications", []), dtype=float)
    if probabilities.ndim != 2 or probabilities.size == 0:
        raise ValueError(f"Classifier {resolved_name!r} is missing a 2-D classifications matrix")
    if component_total is not None and probabilities.shape[0] != int(component_total):
        raise ValueError(
            f"Classifier {resolved_name!r} has {probabilities.shape[0]} rows for {component_total} ICA components"
        )
    classes = _classifier_classes(record, resolved_name, probabilities.shape[1])
    return ClassifierData(resolved_name, classes, probabilities)


def resolve_dipfit_data(EEG: dict[str, Any], component_index: int) -> DipfitData | None:
    """Return normalized DIPFIT model data for a 1-based component index."""
    models = normalize_model_list(EEG)
    index = int(component_index)
    if index < 1:
        raise ValueError("component index must be 1-based")
    if index > len(models):
        return None
    model = models[index - 1]
    raw_positions = np.asarray(model.get("posxyz", []), dtype=float)
    if raw_positions.size == 0:
        return None
    positions = _dipfit_matrix(raw_positions, "posxyz", index)
    if not np.all(np.isfinite(positions)):
        raise ValueError(f"DIPFIT model for component {index} contains non-finite posxyz values")
    moments = _dipfit_moments(model.get("momxyz", []), positions.shape[0], index)
    rv = _finite_float(model.get("rv"))
    coordformat = ""
    dipfit = EEG.get("dipfit")
    if isinstance(dipfit, dict):
        coordformat = str(dipfit.get("coordformat") or "")
    return DipfitData(
        positions=positions,
        moments=moments,
        rv_percent=None if rv is None else rv * 100.0,
        dmr=_dipole_moment_ratio(moments),
        coordformat=coordformat,
    )


def selected_property_indices(
    EEG: dict[str, Any],
    typecomp: int | bool,
    values: Any,
    *,
    default_all: bool = True,
) -> list[int]:
    """Normalize EEGLAB-facing channel/component selections to 1-based indices."""
    limit = int(EEG.get("nbchan", 0) or 0) if int(bool(typecomp)) else component_count(EEG)
    return one_based_indices(values, limit=limit, default_all=default_all)


def component_count(EEG: dict[str, Any]) -> int:
    """Return the number of available ICA components."""
    icaact = EEG.get("icaact")
    if icaact is not None and np.asarray(icaact).size:
        values = np.asarray(icaact)
        if values.ndim >= 2:
            return int(values.shape[0])
    weights = np.asarray(EEG.get("icaweights", []))
    if weights.ndim == 2 and weights.size:
        return int(weights.shape[0])
    winv = np.asarray(EEG.get("icawinv", []))
    if winv.ndim == 2 and winv.size:
        return int(winv.shape[1])
    return 0


def has_component_classifier(EEG: dict[str, Any], classifier_name: str = "") -> bool:
    """Return whether usable classifier data are available for the current ICA."""
    try:
        return (
            resolve_classifier_data(EEG, classifier_name, component_total=component_count(EEG), require=False)
            is not None
        )
    except ValueError as exc:
        if "missing a 2-D classifications matrix" in str(exc):
            return False
        raise


def component_rejection_status(
    EEG: dict[str, Any],
    component_index: int,
    *,
    component_total: int | None = None,
) -> bool:
    """Return the current ``EEG.reject.gcompreject`` status for one component."""
    total = component_count(EEG) if component_total is None else int(component_total)
    index = int(component_index)
    if index < 1 or index > total:
        raise ValueError("component index is outside available ICA components")
    return bool(component_rejection_flags(EEG, total, create=False)[index - 1])


def build_extended_property_data(
    EEG: dict[str, Any],
    typecomp: int | bool,
    index: int,
    *,
    spec_opt: Any = None,
    erp_opt: Any = None,
    classifier_name: str = "",
) -> ExtendedPropertyData:
    """Assemble the dashboard data for one EEGLAB-facing channel/component index."""
    del erp_opt
    typecomp = int(bool(typecomp))
    data = eeg_epoch_data(EEG)
    times_ms = eeg_times_ms(EEG)
    if typecomp:
        return _channel_dashboard_data(EEG, data, times_ms, int(index), spec_opt)
    return _component_dashboard_data(EEG, data, times_ms, int(index), spec_opt, classifier_name)


def _channel_dashboard_data(
    EEG: dict[str, Any],
    data: np.ndarray,
    times_ms: np.ndarray,
    index: int,
    spec_opt: Any,
) -> ExtendedPropertyData:
    labels = channel_labels(EEG)
    if index < 1 or index > data.shape[0]:
        raise ValueError("channel index is outside available channels")
    label = labels[index - 1] if index - 1 < len(labels) else str(index)
    activity = np.array(data[index - 1 : index], dtype=float, copy=True)
    spectrum_freqs, spectrum_power = _spectrum(activity, EEG, spec_opt)
    image_data, image_extent, image_title = _activity_image(activity, times_ms, f"Epoched Channel {label} Activity")
    return ExtendedPropertyData(
        typecomp=1,
        index=index,
        label=label,
        figure_title=f"Channel {label} - pop_prop_extended()",
        topography_title=f"Channel {label}",
        topography_values=index,
        topography_chanlocs=list(EEG.get("chanlocs", []) or []),
        activity=activity,
        times_ms=times_ms,
        activity_title="Channel Time Series",
        image_data=image_data,
        image_extent=image_extent,
        image_title=image_title,
        spectrum_freqs=spectrum_freqs,
        spectrum_power=spectrum_power,
        spectrum_title="Channel Activity Power Spectrum",
        classifier=None,
        class_probabilities=None,
        pvaf=None,
        dipfit=None,
        rejected=None,
    )


def _component_dashboard_data(
    EEG: dict[str, Any],
    data: np.ndarray,
    times_ms: np.ndarray,
    index: int,
    spec_opt: Any,
    classifier_name: str,
) -> ExtendedPropertyData:
    activity_all = component_activations(EEG)
    maps, map_chanlocs = component_map_data(EEG)
    if index < 1 or index > activity_all.shape[0]:
        raise ValueError("component index is outside available ICA components")
    activity = np.array(activity_all[index - 1 : index], dtype=float, copy=True)
    classifier = resolve_classifier_data(EEG, classifier_name, component_total=activity_all.shape[0], require=False)
    probabilities = (
        None if classifier is None else np.array(classifier.probabilities[index - 1], dtype=float, copy=True)
    )
    spectrum_freqs, spectrum_power = _spectrum(activity, EEG, spec_opt)
    image_data, image_extent, image_title = _activity_image(activity, times_ms, f"Epoched IC{index} Activity")
    return ExtendedPropertyData(
        typecomp=0,
        index=index,
        label=f"IC{index}",
        figure_title=f"IC{index} - pop_prop_extended()",
        topography_title=f"IC{index}",
        topography_values=maps[:, index - 1],
        topography_chanlocs=map_chanlocs,
        activity=activity,
        times_ms=times_ms,
        activity_title=f"Scrolling IC{index} Activity",
        image_data=image_data,
        image_extent=image_extent,
        image_title=image_title,
        spectrum_freqs=spectrum_freqs,
        spectrum_power=spectrum_power,
        spectrum_title=f"IC{index} Activity Power Spectrum",
        classifier=classifier,
        class_probabilities=probabilities,
        pvaf=_component_pvaf(EEG, data, maps, activity, index),
        dipfit=resolve_dipfit_data(EEG, index),
        rejected=component_rejection_status(EEG, index, component_total=activity_all.shape[0]),
    )


def _spectrum(activity: np.ndarray, EEG: dict[str, Any], spec_opt: Any) -> tuple[np.ndarray, np.ndarray]:
    options = parse_plot_options_text(spec_opt)
    flat = np.asarray(activity, dtype=float).reshape(1, -1)
    spectra, freqs, _std = compute_spectra(
        flat,
        int(EEG.get("pnts", flat.shape[1]) or flat.shape[1]),
        float(EEG.get("srate", 1.0) or 1.0),
        winsize=_first_int(options.get("winsize")),
        overlap=_first_int(options.get("overlap")) or 0,
        nfft=_first_int(options.get("nfft")),
    )
    return freqs, spectra[0]


def _activity_image(
    activity: np.ndarray,
    times_ms: np.ndarray,
    epoched_title: str,
) -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    trace = np.asarray(activity[0], dtype=float)
    trace = trace - float(np.nanmean(trace))
    if trace.ndim == 1:
        trace = trace[:, np.newaxis]
    if trace.shape[1] > 1:
        image = trace.T
        extent = (float(times_ms[0]), float(times_ms[-1]), 1.0, float(trace.shape[1]))
        return image, extent, epoched_title
    flat = trace[:, 0]
    line_count = min(200, max(1, int(np.floor(np.sqrt(flat.size)))))
    frame_count = max(1, flat.size // line_count)
    image = flat[: line_count * frame_count].reshape(line_count, frame_count)
    extent = (0.0, float(frame_count - 1), 1.0, float(line_count))
    return image, extent, "Continuous Data"


def _dipfit_matrix(values: np.ndarray, field_name: str, component_index: int) -> np.ndarray:
    matrix = values
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise ValueError(
            f"DIPFIT model for component {component_index} must contain {field_name} rows with 3 coordinates"
        )
    return np.array(matrix[:, :3], dtype=float, copy=True)


def _dipfit_moments(values: Any, position_count: int, component_index: int) -> np.ndarray | None:
    raw_moments = np.asarray(values, dtype=float)
    if raw_moments.size == 0:
        return None
    moments = _dipfit_matrix(raw_moments, "momxyz", component_index)
    if moments.shape[0] != position_count:
        raise ValueError(f"DIPFIT model for component {component_index} must have matching posxyz and momxyz rows")
    if not np.all(np.isfinite(moments)):
        raise ValueError(f"DIPFIT model for component {component_index} contains non-finite momxyz values")
    return moments


def _dipole_moment_ratio(moments: np.ndarray | None) -> float | None:
    if moments is None or moments.shape[0] != 2:
        return None
    norms = np.linalg.norm(moments, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        return None
    ratio = float(norms[0] / norms[1])
    return ratio if ratio >= 1.0 else 1.0 / ratio


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(np.asarray(value).reshape(()))
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _component_pvaf(
    EEG: dict[str, Any],
    data: np.ndarray,
    maps: np.ndarray,
    activity: np.ndarray,
    index: int,
) -> float | None:
    if maps.shape[0] == 0:
        return None
    icachansind = component_channel_indices(EEG, data.shape[0])
    if maps.shape[0] != icachansind.size:
        return None
    flat_data = data[icachansind, :, :].reshape(icachansind.size, -1)
    component_trace = activity.reshape(1, -1)
    projection = maps[:, index - 1 : index] @ component_trace
    datavar = float(np.nanmean(np.nanvar(flat_data, axis=1)))
    if not np.isfinite(datavar) or datavar <= 0:
        return None
    projvar = float(np.nanmean(np.nanvar(flat_data - projection, axis=1)))
    if not np.isfinite(projvar):
        return None
    return 100.0 * (1.0 - projvar / datavar)


def _resolve_classifier_name(classifiers: list[str], classifier_name: str) -> str:
    if classifier_name:
        for name in classifiers:
            if name.lower() == str(classifier_name).lower():
                return name
        raise ValueError(f"Classifier {classifier_name!r} was not found in EEG.etc.ic_classification")
    return classifiers[classifier_default_index(classifiers) - 1]


def _classifier_classes(record: dict[str, Any], classifier_name: str, class_count: int) -> tuple[str, ...]:
    raw_classes = record.get("classes", [])
    classes = [str(item) for item in np.asarray(raw_classes, dtype=object).ravel().tolist() if str(item)]
    if not classes:
        if classifier_name.lower() == "iclabel" and class_count == len(DEFAULT_ICLABEL_CLASSES):
            return DEFAULT_ICLABEL_CLASSES
        return tuple(f"Class {index}" for index in range(1, class_count + 1))
    if len(classes) != class_count:
        raise ValueError(
            f"Classifier {classifier_name!r} has {class_count} probability columns but {len(classes)} class names"
        )
    return tuple(classes)


def _first_int(value: Any) -> int | None:
    vector = numeric_vector(value)
    if vector.size == 0:
        return None
    return int(vector[0])


__all__ = [
    "DEFAULT_ICLABEL_CLASSES",
    "ClassifierData",
    "DipfitData",
    "ExtendedPropertyData",
    "build_extended_property_data",
    "classifier_default_index",
    "classifier_name_from_gui",
    "classifier_names",
    "component_count",
    "component_rejection_status",
    "has_component_classifier",
    "resolve_classifier_data",
    "resolve_dipfit_data",
    "selected_property_indices",
]
