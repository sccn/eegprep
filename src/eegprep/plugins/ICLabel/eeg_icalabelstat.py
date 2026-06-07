"""EEGLAB ICLabel component summary statistics."""

from __future__ import annotations

import sys
from typing import Any, TextIO

import numpy as np


DEFAULT_ICLABEL_CLASSES = ("Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other")


def eeg_icalabelstat(
    EEG: dict[str, Any],
    threshold: float | list[float] | tuple[float, ...] | np.ndarray = 0.9,
    *,
    verbose: bool = True,
    stream: TextIO | None = None,
) -> dict[str, Any]:
    """Return and optionally print ICLabel class statistics.

    This mirrors EEGLAB's ``plugins/ICLabel/eeg_icalabelstat.m`` summary,
    including class order and the historical ``IClabel`` console label, while
    returning structured Python statistics for programmatic use.

    Args:
        EEG: EEGPrep/EEGLAB-style dataset with ICLabel classifications under
            ``EEG["etc"]["ic_classification"]["ICLabel"]``.
        threshold: Scalar probability threshold applied to every class, or one
            threshold per ICLabel class.
        verbose: Print EEGLAB-style summary lines when true.
        stream: Destination for printed summary lines. Defaults to stdout.

    Returns:
        Dictionary containing class names, thresholds, per-class counts,
        1-based component indices above threshold, mean probabilities, dominant
        class counts, and current rejected/kept tallies.
    """
    iclabel_state = _iclabel_state(EEG)
    classifications = _classification_matrix(iclabel_state)
    classes = _class_names(iclabel_state, classifications.shape[1])
    thresholds = _threshold_vector(threshold, len(classes))
    flags = _rejection_flags(EEG, classifications.shape[0])

    above_threshold = classifications > thresholds[np.newaxis, :]
    counts = np.sum(above_threshold, axis=0).astype(int)
    component_indices = [
        (np.flatnonzero(above_threshold[:, class_index]) + 1).astype(int).tolist()
        for class_index in range(len(classes))
    ]
    rejected_counts = np.sum(above_threshold & flags[:, np.newaxis], axis=0).astype(int)
    kept_counts = np.sum(above_threshold & ~flags[:, np.newaxis], axis=0).astype(int)
    dominant_class_indices = np.argmax(classifications, axis=1)

    stats = {
        "classes": classes,
        "threshold": thresholds,
        "component_count": int(classifications.shape[0]),
        "counts": counts,
        "component_indices": component_indices,
        "mean_probability": np.mean(classifications, axis=0),
        "dominant_counts": np.bincount(dominant_class_indices, minlength=len(classes)).astype(int),
        "dominant_class_indices": (dominant_class_indices + 1).astype(int),
        "rejected_component_count": int(np.sum(flags)),
        "kept_component_count": int(classifications.shape[0] - np.sum(flags)),
        "rejected_counts": rejected_counts,
        "kept_counts": kept_counts,
    }
    if verbose:
        _print_summary(stats, stream or sys.stdout)
    return stats


def _iclabel_state(EEG: dict[str, Any]) -> dict[str, Any]:
    try:
        state = EEG["etc"]["ic_classification"]["ICLabel"]
    except KeyError as exc:
        raise ValueError("No ICLabel classifications found. Run pop_iclabel first.") from exc
    if not isinstance(state, dict):
        raise ValueError("No ICLabel classifications found. Run pop_iclabel first.")
    return state


def _classification_matrix(iclabel_state: dict[str, Any]) -> np.ndarray:
    try:
        classifications = np.asarray(iclabel_state["classifications"], dtype=float)
    except KeyError as exc:
        raise ValueError("No ICLabel classifications found. Run pop_iclabel first.") from exc
    if classifications.ndim != 2 or classifications.size == 0:
        raise ValueError("ICLabel classifications must be a non-empty 2D array.")
    if not np.isfinite(classifications).all():
        raise ValueError("ICLabel classifications must contain finite probabilities.")
    return classifications


def _class_names(iclabel_state: dict[str, Any], class_count: int) -> list[str]:
    classes = iclabel_state.get("classes")
    if classes is None or np.asarray(classes, dtype=object).size == 0:
        if class_count == len(DEFAULT_ICLABEL_CLASSES):
            return list(DEFAULT_ICLABEL_CLASSES)
        raise ValueError("ICLabel classes are missing and classifications do not have the 7 standard columns.")
    names = [_string_name(value) for value in np.asarray(classes, dtype=object).ravel().tolist()]
    if len(names) != class_count:
        raise ValueError(f"ICLabel class list has {len(names)} labels for {class_count} probability columns.")
    return names


def _string_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _threshold_vector(threshold: float | list[float] | tuple[float, ...] | np.ndarray, class_count: int) -> np.ndarray:
    values = np.asarray(threshold, dtype=float)
    if values.ndim == 0 or values.size == 1:
        thresholds = np.repeat(float(values.reshape(-1)[0]), class_count)
    else:
        thresholds = values.reshape(-1)
        if thresholds.size != class_count:
            raise ValueError(f"threshold must be scalar or contain {class_count} values.")
    if not np.isfinite(thresholds).all() or (thresholds < 0).any() or (thresholds > 1).any():
        raise ValueError("ICLabel thresholds must be finite probabilities between 0 and 1.")
    return thresholds.astype(float, copy=True)


def _rejection_flags(EEG: dict[str, Any], component_count: int) -> np.ndarray:
    reject = EEG.get("reject") or {}
    flags = reject.get("gcompreject") if isinstance(reject, dict) else None
    if flags is None:
        return np.zeros(component_count, dtype=bool)
    normalized = np.asarray(flags).reshape(-1)
    if normalized.size != component_count:
        raise ValueError("EEG.reject.gcompreject must match the number of ICLabel components.")
    return normalized.astype(bool)


def _print_summary(stats: dict[str, Any], stream: TextIO) -> None:
    classes = stats["classes"]
    counts = stats["counts"]
    thresholds = stats["threshold"]
    component_count = stats["component_count"]
    for class_name, count, class_threshold in zip(classes, counts, thresholds):
        label = f'IClabel class "{class_name}"'
        percent = int(np.round(float(class_threshold) * 100))
        print(f"{label:>30}: {int(count)}/{component_count} components at {percent}% threshold", file=stream)


__all__ = ["DEFAULT_ICLABEL_CLASSES", "eeg_icalabelstat"]
