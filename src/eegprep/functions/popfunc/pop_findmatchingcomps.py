"""Find ICA components whose scalp maps match provided artifact maps."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args


def pop_findmatchingcomps(
    EEG: dict[str, Any] | None = None,
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
):
    """Find current-dataset components matching another set of scalp maps."""
    if EEG is None:
        return (None, "") if return_com else (None, [], [])
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    out = copy.deepcopy(EEG)
    match_maps, matchinput_output = _match_maps_from_options(options)
    candidate_maps = np.asarray(out.get("icawinv", []), dtype=float)
    if candidate_maps.ndim != 2 or candidate_maps.size == 0:
        raise ValueError("EEG.icawinv component maps are required")
    match_maps = _orient_match_maps(match_maps, candidate_maps.shape[0])
    correlations = _absolute_column_correlations(candidate_maps, match_maps)
    best = np.argmax(correlations, axis=0)
    best_values = correlations[best, np.arange(correlations.shape[1])]
    selected = best_values > float(options.get("corrthresh", 0.92))
    matchic = (best[selected] + 1).astype(int).tolist()
    matchinput = (np.flatnonzero(selected) + 1).astype(int).tolist() if matchinput_output else []
    if int(options.get("rejflag", 0)):
        reject = out.setdefault("reject", {})
        flags = np.asarray(reject.get("gcompreject", []), dtype=int).ravel()
        if flags.size != candidate_maps.shape[1]:
            flags = np.zeros(candidate_maps.shape[1], dtype=int)
        flags[[index - 1 for index in matchic]] = 1
        reject["gcompreject"] = flags
    if selected.sum() < selected.size and str(options.get("nomatcherror", "off")).lower() == "on":
        raise ValueError("Not all tagged components were matched")
    command = _history_command(options, match_maps)
    return (out, command) if return_com else (out, matchic, matchinput)


def _match_maps_from_options(options: dict[str, Any]) -> tuple[np.ndarray, bool]:
    if options.get("dataset") is not None and options.get("matchcomps") is not None:
        raise ValueError("dataset and matchcomps cannot both be provided")
    if options.get("matchcomps") is not None:
        return np.asarray(options["matchcomps"], dtype=float), True
    dataset = options.get("dataset")
    if dataset is None:
        raise ValueError("Either dataset or matchcomps must be provided")
    if isinstance(dataset, int):
        alleeg = options.get("alleeg")
        if not isinstance(alleeg, list) or dataset < 1 or dataset > len(alleeg):
            raise ValueError("Integer dataset requires an ALLEEG list and a valid 1-based index")
        dataset = alleeg[dataset - 1]
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an EEG dictionary, or a 1-based index with ALLEEG")
    maps = np.asarray(dataset.get("icawinv", []), dtype=float)
    if maps.ndim != 2 or maps.size == 0:
        raise ValueError("Matching dataset does not contain component maps")
    flags = np.asarray((dataset.get("reject") or {}).get("gcompreject", []), dtype=bool).ravel()
    if flags.size != maps.shape[1] or not flags.any():
        raise ValueError("Matching dataset does not contain components marked for rejection")
    return maps[:, flags], False


def _orient_match_maps(match_maps: np.ndarray, channel_count: int) -> np.ndarray:
    if match_maps.ndim == 1:
        match_maps = match_maps[:, np.newaxis]
    if match_maps.ndim != 2:
        raise ValueError("matchcomps must be a 1-D or 2-D array")
    if match_maps.shape[0] == channel_count:
        return match_maps
    if match_maps.shape[1] == channel_count:
        return match_maps.T
    raise ValueError("The tagged components do not have the same number of channels")


def _absolute_column_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=0, keepdims=True)
    right_centered = right - right.mean(axis=0, keepdims=True)
    left_norm = np.linalg.norm(left_centered, axis=0)
    right_norm = np.linalg.norm(right_centered, axis=0)
    left_norm[left_norm == 0] = 1.0
    right_norm[right_norm == 0] = 1.0
    return np.abs((left_centered / left_norm).T @ (right_centered / right_norm))


def _history_command(options: dict[str, Any], match_maps: np.ndarray) -> str:
    pieces: list[str] = []
    history_options = dict(options)
    if "dataset" in history_options:
        history_options.pop("dataset", None)
        history_options.pop("alleeg", None)
        history_options["matchcomps"] = match_maps
    for key in ("dataset", "matchcomps", "corrthresh", "rejflag", "nomatcherror"):
        if key in history_options:
            value = history_options[key]
            pieces.extend([format_history_value(key), format_history_value(value, cell_for_sequence=None)])
    return "EEG = pop_findmatchingcomps(EEG, " + ", ".join(pieces) + ");"


__all__ = ["pop_findmatchingcomps"]
