"""Interpolate missing channels across STUDY datasets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.plot_utils import numeric_vector
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.eeg_interp import eeg_interp
from eegprep.functions.studyfunc._study_utils import as_alleeg_list, build_python_call, ensure_study, merged_chanlocs
from eegprep.functions.studyfunc.std_checkset import std_checkset


def std_interp(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    chans: Any = None,
    method: str = "spherical",
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Interpolate selected missing channels into every STUDY dataset."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if "channels" in options and chans is None:
        chans = options.pop("channels")
    method = str(options.pop("method", method) or "spherical")
    if options:
        raise ValueError(f"Unknown std_interp option(s): {', '.join(sorted(options))}")
    study, datasets = std_checkset(ensure_study(STUDY), as_alleeg_list(ALLEEG))
    if not datasets:
        raise ValueError("std_interp requires ALLEEG datasets")
    all_locs = merged_chanlocs(datasets)
    requested_locs = _target_locs(all_locs, chans)
    requested_labels = [str(loc.get("labels") or "") for loc in requested_locs]
    if not requested_locs:
        raise ValueError("std_interp requires channel locations to interpolate")
    updated = []
    changed = []
    for index, eeg in enumerate(datasets, start=1):
        current_locs = chanlocs_as_list(eeg.get("chanlocs"))
        before = [str(loc.get("labels") or "") for loc in current_locs]
        target_locs = requested_locs if chans is None else _union_locs(all_locs, current_locs, requested_locs)
        after_labels = [str(loc.get("labels") or "") for loc in target_locs]
        if before == after_labels:
            updated.append(eeg)
            continue
        interpolated = eeg_interp(eeg, deepcopy(target_locs), method=method)
        interpolated["saved"] = "no"
        updated.append(interpolated)
        changed.append(index)
    study.setdefault("etc", {}).setdefault("eegprep", {})["std_interp"] = {
        "channels": requested_labels,
        "method": method,
        "changed_datasets": changed,
    }
    study["saved"] = "no" if changed else study.get("saved", "no")
    command = build_python_call(
        ("STUDY", "ALLEEG"),
        "std_interp",
        "STUDY",
        "ALLEEG",
        chans=chans,
        method=method,
    )
    return (study, updated, command) if return_com else (study, updated)


def _target_locs(locs: list[dict[str, Any]], chans: Any) -> list[dict[str, Any]]:
    if chans is None or (isinstance(chans, str) and chans == ""):
        return deepcopy(locs)
    if isinstance(chans, str):
        requested = [chans]
    elif isinstance(chans, (list, tuple)) and all(isinstance(item, str) for item in chans):
        requested = list(chans)
    else:
        indices = numeric_vector(chans, dtype=int)
        if indices.size == 0:
            return deepcopy(locs)
        if np.any(indices < 1) or np.any(indices > len(locs)):
            raise ValueError(f"chans must be 1-based and within 1..{len(locs)}")
        return [deepcopy(locs[int(index) - 1]) for index in indices]
    lookup = {str(loc.get("labels") or "").lower(): loc for loc in locs}
    missing = [label for label in requested if label.lower() not in lookup]
    if missing:
        raise ValueError(f"Channel named '{missing[0]}' not found in any dataset")
    selected = []
    selected_labels = {label.lower() for label in requested}
    for loc in locs:
        label = str(loc.get("labels") or "").lower()
        if label in selected_labels:
            selected.append(deepcopy(loc))
    return selected


def _union_locs(
    all_locs: list[dict[str, Any]], current_locs: list[dict[str, Any]], requested_locs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    labels = {str(loc.get("labels") or "").lower() for loc in current_locs}
    labels.update(str(loc.get("labels") or "").lower() for loc in requested_locs)
    return [deepcopy(loc) for loc in all_locs if str(loc.get("labels") or "").lower() in labels]


__all__ = ["std_interp"]
