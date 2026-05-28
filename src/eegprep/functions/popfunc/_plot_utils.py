"""Shared helpers for EEGLAB-style plotting pop functions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list


def as_eeg_list(value: Any) -> list[dict[str, Any]]:
    """Return one or more EEG dictionaries as a list."""
    if isinstance(value, list):
        if not all(isinstance(item, dict) for item in value):
            raise ValueError("Expected EEG dataset dictionaries")
        return value
    if not isinstance(value, dict):
        raise ValueError("Expected an EEG dataset")
    return [value]


def eeg_data_array(EEG: dict[str, Any]) -> np.ndarray:
    """Return channel-major EEG data as a float array."""
    data = np.asarray(EEG.get("data"), dtype=float)
    if data.ndim not in {2, 3}:
        raise ValueError("EEG.data must be a 2-D or 3-D channel-major array")
    return data


def eeg_epoch_data(EEG: dict[str, Any]) -> np.ndarray:
    """Return EEG data as ``channels x points x trials``."""
    data = eeg_data_array(EEG)
    if data.ndim == 2:
        return data[:, :, np.newaxis]
    return data


def eeg_times_ms(EEG: dict[str, Any]) -> np.ndarray:
    """Return one epoch worth of time values in milliseconds."""
    pnts = int(EEG.get("pnts", 0) or 0)
    if pnts <= 0:
        raise ValueError("EEG.pnts must be positive")
    times = np.asarray(EEG.get("times", []), dtype=float).ravel()
    if times.size == pnts:
        xmax = float(EEG.get("xmax", 0) or 0)
        if times.size > 1 and np.nanmax(np.abs(times)) <= max(abs(xmax), 1.0) * 2:
            return times * 1000.0
        return times
    xmin = float(EEG.get("xmin", 0) or 0)
    xmax = float(EEG.get("xmax", xmin) or xmin)
    return np.linspace(xmin * 1000.0, xmax * 1000.0, pnts)


def data_time_slice(EEG: dict[str, Any], timerange: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Return epoch data and time values restricted to ``timerange`` in ms."""
    data = eeg_epoch_data(EEG)
    times = eeg_times_ms(EEG)
    if timerange is None or _is_empty_sequence(timerange):
        return data, times
    bounds = numeric_vector(timerange)
    if bounds.size != 2:
        raise ValueError("timerange must contain [min max] in milliseconds")
    mask = (times >= bounds[0]) & (times <= bounds[1])
    if not np.any(mask):
        raise ValueError("timerange does not contain any samples")
    return data[:, mask, :], times[mask]


def channel_labels(EEG: dict[str, Any]) -> list[str]:
    """Return channel labels with EEGLAB-style numeric fallbacks."""
    labels = []
    for index, chanloc in enumerate(chanlocs_as_list(EEG.get("chanlocs", [])), start=1):
        label = str(chanloc.get("labels") or "").strip()
        labels.append(label or str(index))
    nbchan = int(EEG.get("nbchan", len(labels)) or len(labels))
    while len(labels) < nbchan:
        labels.append(str(len(labels) + 1))
    return labels


def selected_indices(values: Any, maximum: int, *, default_all: bool = True) -> np.ndarray:
    """Normalize EEGLAB-facing 1-based indices to Python 0-based indices."""
    if values is None or _is_empty_sequence(values):
        if default_all:
            return np.arange(maximum, dtype=int)
        raise ValueError("At least one index is required")
    numeric = numeric_vector(values, dtype=float)
    if numeric.size == 0 and default_all:
        return np.arange(maximum, dtype=int)
    indices = numeric.astype(int)
    if np.any(indices != numeric):
        raise ValueError("Indices must be integers")
    if np.any(indices < 1) or np.any(indices > maximum):
        raise ValueError(f"Indices must be 1-based and within 1..{maximum}")
    return indices - 1


def component_activations(EEG: dict[str, Any]) -> np.ndarray:
    """Return ICA activations as ``components x points x trials``."""
    icaact = EEG.get("icaact")
    if icaact is not None and np.asarray(icaact).size:
        data = np.asarray(icaact, dtype=float)
        if data.ndim == 2:
            return data[:, :, np.newaxis]
        if data.ndim == 3:
            return data
    weights = np.asarray(EEG.get("icaweights", []), dtype=float)
    sphere = np.asarray(EEG.get("icasphere", []), dtype=float)
    if weights.size == 0 or sphere.size == 0:
        raise ValueError("no ICA activations or weights for this dataset")
    data = eeg_epoch_data(EEG)
    flat = data.reshape(data.shape[0], -1)
    acts = (weights @ sphere) @ flat
    return acts.reshape(weights.shape[0], data.shape[1], data.shape[2])


def component_maps(EEG: dict[str, Any]) -> np.ndarray:
    """Return ICA inverse maps as ``channels x components``."""
    icawinv = np.asarray(EEG.get("icawinv", []), dtype=float)
    if icawinv.size == 0:
        raise ValueError("no ICA maps for this dataset")
    if icawinv.ndim != 2:
        raise ValueError("EEG.icawinv must be 2-D")
    return icawinv


def numeric_vector(value: Any, *, dtype: Any = float) -> np.ndarray:
    """Parse EEGLAB-style numeric vectors from strings, lists, or arrays."""
    if value is None:
        return np.asarray([], dtype=dtype)
    if isinstance(value, np.ndarray):
        return value.astype(dtype).ravel()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.asarray([value], dtype=dtype)
    if isinstance(value, str):
        text = value.strip().strip("[]")
        if not text:
            return np.asarray([], dtype=dtype)
        values = []
        for token in text.replace(",", " ").split():
            if ":" in token:
                values.extend(colon_sequence(token))
            else:
                values.append(float(token))
        return np.asarray(values, dtype=dtype)
    if isinstance(value, Iterable):
        return np.asarray(list(value), dtype=dtype).ravel()
    return np.asarray([value], dtype=dtype)


def colon_sequence(token: str) -> list[float]:
    """Parse MATLAB ``start:stop`` or ``start:step:stop`` tokens."""
    pieces = token.split(":")
    if len(pieces) not in {2, 3}:
        raise ValueError(f"Invalid colon range: {token}")
    start = float(pieces[0])
    if len(pieces) == 2:
        stop = float(pieces[1])
        step = 1.0 if stop >= start else -1.0
    else:
        step = float(pieces[1])
        stop = float(pieces[2])
    if step == 0 or not np.all(np.isfinite([start, step, stop])):
        raise ValueError(f"Invalid colon range: {token}")
    if (stop - start) * step < 0:
        return []
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    values = [float(start + index * step) for index in range(max(count, 0))]
    if values and np.isclose(values[-1], stop, rtol=0.0, atol=max(abs(step), 1.0) * 1e-12):
        values[-1] = float(stop)
    return values


def python_literal(value: Any) -> str:
    """Return a pasteable Python literal for EEGPrep console history."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return "float('nan')"
        if np.isposinf(value):
            return "float('inf')"
        if np.isneginf(value):
            return "float('-inf')"
        if value.is_integer():
            return str(int(value))
    if isinstance(value, list):
        return "[" + ", ".join(python_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "(" + ", ".join(python_literal(item) for item in value) + ("," if len(value) == 1 else "") + ")"
    return repr(value)


def history_command(function_name: str, *args: Any, eeg_name: str = "EEG", **kwargs: Any) -> str:
    """Build a valid Python console command for a plotting pop function."""
    pieces = [eeg_name]
    pieces.extend(python_literal(arg) for arg in args)
    pieces.extend(f"{key}={python_literal(value)}" for key, value in kwargs.items() if value is not None)
    return f"{function_name}({', '.join(pieces)})"


def _is_empty_sequence(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().strip("[]") == ""
    if isinstance(value, np.ndarray):
        return value.size == 0
    return isinstance(value, (list, tuple)) and len(value) == 0
