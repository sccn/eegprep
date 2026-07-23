"""Shared browser-backed activity views for property plots."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc.plot_utils import channel_labels, component_activations
from eegprep.functions.sigprocfunc.eegplot import eegplot


def property_activity_browser(
    EEG: dict[str, Any],
    typecomp: int | bool,
    index: int,
    *,
    scroll_event: int | bool = 1,
    show: bool = False,
) -> Any:
    """Build or open the scrolling activity browser for one property item."""
    if int(bool(typecomp)):
        data, eloc_file, title = _channel_activity(EEG, int(index))
    else:
        data, eloc_file, title = _component_activity(EEG, int(index))
    return eegplot(
        data,
        srate=float(EEG.get("srate", 256) or 256),
        limits=[float(EEG.get("xmin", 0.0) or 0.0) * 1000.0, float(EEG.get("xmax", 0.0) or 0.0) * 1000.0],
        events=EEG.get("event", []) if int(bool(scroll_event)) else [],
        eloc_file=eloc_file,
        title=title,
        dispchans=1,
        spacing=_activity_spacing(data),
        show=show,
    )


def _channel_activity(EEG: dict[str, Any], index: int) -> tuple[np.ndarray, Any, str]:
    data = np.asarray(EEG.get("data"), dtype=float)
    if index < 1 or index > data.shape[0]:
        raise ValueError("channel index is outside available channels")
    chanlocs = EEG.get("chanlocs", [])
    if isinstance(chanlocs, np.ndarray):
        chanlocs = chanlocs.tolist()
    eloc_file = [chanlocs[index - 1]] if isinstance(chanlocs, list) and index - 1 < len(chanlocs) else [index]
    labels = channel_labels(EEG)
    label = labels[index - 1] if index - 1 < len(labels) else str(index)
    return np.array(data[index - 1 : index], copy=True), eloc_file, f"Channel {label} activity -- eegplot()"


def _component_activity(EEG: dict[str, Any], index: int) -> tuple[np.ndarray, Any, str]:
    acts = component_activations(EEG)
    if index < 1 or index > acts.shape[0]:
        raise ValueError("component index is outside available ICA components")
    return np.array(acts[index - 1 : index], copy=True), [index], f"Scrolling IC{index} Activity -- eegplot()"


def _activity_spacing(data: np.ndarray) -> float:
    flat = np.asarray(data, dtype=float).reshape(1, -1)
    sample = flat[:, : min(1000, flat.shape[1])]
    spacing = float(np.nanstd(sample) * 3.0)
    if not np.isfinite(spacing) or spacing <= 0:
        return 1.0
    if spacing > 10:
        return float(round(spacing))
    return spacing


__all__ = ["property_activity_browser"]
