"""Convert BioSig event tables to EEGPrep event dictionaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def biosig2eeglabevent(
    EVENT: dict[str, Any],
    interval: Any = None,
    importEDFplus: bool = False,
) -> list[dict[str, Any]]:
    """Convert a BioSig ``EVENT`` mapping into EEG event records.

    BioSig ``POS`` values and returned EEG ``latency`` values are one-based
    sample positions. When ``interval=[first, last]`` is supplied, both bounds
    are one-based and inclusive, and retained latencies are rebased so
    ``first`` becomes sample 1.
    """
    if not isinstance(EVENT, dict):
        raise ValueError("biosig2eeglabevent: EVENT must be a mapping")
    fields = {name: _field_values(EVENT[name]) for name in ("TYP", "POS", "DUR", "CHN") if name in EVENT}
    events = _initial_events(EVENT.get("Teeg"))
    lengths = [len(values) for values in fields.values()]
    count = len(events) if events else (lengths[0] if lengths else 0)
    if any(length != count for length in lengths):
        raise ValueError("biosig2eeglabevent: TYP, POS, DUR, CHN, and Teeg lengths must agree")
    if not events:
        events = [{} for _ in range(count)]

    selected = list(range(count))
    bounds = _interval(interval)
    if bounds is not None:
        if "POS" not in fields:
            raise ValueError("biosig2eeglabevent: interval selection requires EVENT.POS")
        selected = [index for index, value in enumerate(fields["POS"]) if bounds[0] <= float(value) <= bounds[1]]

    include_duration = "DUR" in fields and _contains_nonzero(fields["DUR"])
    include_channel = "CHN" in fields and _contains_nonzero(fields["CHN"])
    result: list[dict[str, Any]] = []
    for index in selected:
        event = deepcopy(events[index])
        if "TYP" in fields:
            event["type"] = _edf_type(fields["TYP"][index], EVENT, importEDFplus)
        if "POS" in fields:
            position = float(fields["POS"][index])
            event["latency"] = position if bounds is None else position - bounds[0] + 1
        if include_duration:
            duration = fields["DUR"][index]
            if bounds is not None:
                duration = min(float(duration), bounds[1] - float(fields["POS"][index]))
            event["duration"] = duration
        if include_channel:
            event["chanindex"] = fields["CHN"][index]
        result.append(event)
    return result


def _field_values(value: Any) -> list[Any]:
    if isinstance(value, bytes):
        return [bytes([item]) for item in value]
    if isinstance(value, str):
        return list(value)
    return np.asarray(value).reshape(-1).tolist()


def _initial_events(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [deepcopy(value)]
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("biosig2eeglabevent: Teeg must contain event dictionaries")
    return deepcopy(value)


def _interval(value: Any) -> tuple[float, float] | None:
    if value is None or np.asarray(value).size == 0:
        return None
    bounds = np.asarray(value, dtype=float).reshape(-1)
    if bounds.size != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise ValueError("biosig2eeglabevent: interval must be [first, last] with inclusive one-based bounds")
    return float(bounds[0]), float(bounds[1])


def _contains_nonzero(values: list[Any]) -> bool:
    for value in values:
        if isinstance(value, (str, bytes)):
            if value:
                return True
        elif float(value) != 0:
            return True
    return False


def _edf_type(value: Any, event: dict[str, Any], enabled: bool) -> Any:
    if not enabled or not isinstance(value, (int, float, np.integer, np.floating)) or value <= 255:
        return value
    indices = _field_values(event.get("CodeIndex", []))
    descriptions = _field_values(event.get("CodeDesc", []))
    for index, description in zip(indices, descriptions):
        if index == value:
            return description
    return value


__all__ = ["biosig2eeglabevent"]
