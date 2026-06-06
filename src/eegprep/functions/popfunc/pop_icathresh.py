"""Legacy ICA threshold compatibility wrapper."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import format_history_value, parse_numeric_sequence


def pop_icathresh(
    EEG: dict[str, Any] | None = None,
    threshval: Any | None = None,
    rejmethod: str = "current",
    rejvalue: Any = 25,
    interact: int | bool = 1,
    *,
    return_com: bool = False,
):
    """Set ICA rejection thresholds and update ``reject.gcompreject``."""
    if EEG is None:
        return (None, "") if return_com else (None, [])
    out = copy.deepcopy(EEG)
    reject = out.setdefault("reject", {})
    stats = out.get("stats") or {}
    _apply_threshold_values(reject, threshval)
    if str(rejmethod).lower() == "percent":
        _thresholds_from_percent(reject, stats, rejvalue)
    elif str(rejmethod).lower() not in {"current", "dataset"}:
        raise ValueError("rejmethod must be 'current', 'percent', or 'dataset'")
    method = str(rejmethod).lower()
    if method == "dataset":
        flags = _dataset_flags(rejvalue)
        if flags.size:
            reject["gcompreject"] = flags
    else:
        reject["gcompreject"] = _component_flags(reject, stats)
    command = _history_command(threshval, rejmethod, rejvalue, interact)
    return (out, command) if return_com else (out, np.asarray(reject["gcompreject"], dtype=bool))


def _apply_threshold_values(reject: dict[str, Any], threshval: Any | None) -> None:
    values = parse_numeric_sequence(threshval, dtype=float)
    if values:
        if len(values) != 3:
            raise ValueError("threshval must contain entropy, activity-kurtosis, and map-kurtosis thresholds")
        reject["threshentropy"] = values[0]
        reject["threshkurtact"] = values[1]
        reject["threshkurtdist"] = values[2]
    reject.setdefault("threshentropy", np.inf)
    reject.setdefault("threshkurtact", np.inf)
    reject.setdefault("threshkurtdist", np.inf)


def _thresholds_from_percent(reject: dict[str, Any], stats: dict[str, Any], rejvalue: Any) -> None:
    percents = parse_numeric_sequence(rejvalue, dtype=float)
    if not percents:
        percents = [25.0]
    while len(percents) < 3:
        percents.append(percents[-1])
    for field, threshold_field, percent in (
        ("compenta", "threshentropy", percents[0]),
        ("compkurta", "threshkurtact", percents[1]),
        ("compkurtdist", "threshkurtdist", percents[2]),
    ):
        values = np.sort(np.asarray(stats.get(field, []), dtype=float).ravel())
        if values.size:
            index = int(np.clip(round((100.0 - percent) * values.size / 100.0), 0, values.size - 1))
            reject[threshold_field] = float(values[index])


def _component_flags(reject: dict[str, Any], stats: dict[str, Any]) -> np.ndarray:
    entropy = np.asarray(stats.get("compenta", []), dtype=float).ravel()
    kurtact = np.asarray(stats.get("compkurta", []), dtype=float).ravel()
    kurtdist = np.asarray(stats.get("compkurtdist", []), dtype=float).ravel()
    count = max(entropy.size, kurtact.size, kurtdist.size, np.asarray(reject.get("gcompreject", [])).size)
    if count == 0:
        return np.asarray([], dtype=int)
    entropy = _pad(entropy, count)
    kurtact = _pad(kurtact, count)
    kurtdist = _pad(kurtdist, count)
    flags = ((entropy > float(reject["threshentropy"])) & (kurtact > float(reject["threshkurtact"]))) | (
        kurtdist > float(reject["threshkurtdist"])
    )
    return flags.astype(int)


def _dataset_flags(rejvalue: Any) -> np.ndarray:
    if isinstance(rejvalue, dict):
        return np.asarray((rejvalue.get("reject") or {}).get("gcompreject", []), dtype=int).ravel()
    return np.asarray(parse_numeric_sequence(rejvalue, dtype=int), dtype=int).ravel()


def _pad(values: np.ndarray, count: int) -> np.ndarray:
    if values.size == count:
        return values
    out = np.zeros(count, dtype=float)
    out[: values.size] = values
    return out


def _history_command(threshval: Any, rejmethod: str, rejvalue: Any, interact: int | bool) -> str:
    history_value = _dataset_flags(rejvalue).tolist() if str(rejmethod).lower() == "dataset" else rejvalue
    args = [
        format_history_value(parse_numeric_sequence(threshval, dtype=float), cell_for_sequence=None),
        format_history_value(str(rejmethod)),
        format_history_value(history_value, cell_for_sequence=None),
        str(int(bool(interact))),
    ]
    return "EEG = pop_icathresh(EEG, " + ", ".join(args) + ");"


__all__ = ["pop_icathresh"]
