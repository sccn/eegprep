"""EEGLAB-style wrapper for recentering channel coordinates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_numeric_sequence
from eegprep.functions.sigprocfunc.chancenter import chancenter
from eegprep.functions.sigprocfunc.convertlocs import convertlocs


def pop_chancenter(
    chans: Any,
    center: Any = None,
    omitchans: Any = None,
    *,
    gui: bool = False,
    renderer: Any | None = None,
    return_com: bool = False,
) -> Any:
    """Recenter channel locations using a known or optimized sphere center."""
    del renderer
    if gui:
        return (chans, "") if return_com else chans
    is_eeg = isinstance(chans, dict) and "chanlocs" in chans
    output = deepcopy(chans)
    locs = chanlocs_as_list(output.get("chanlocs") if is_eeg else output)
    selected = _selected_indices(locs, omitchans)
    omitted = {index: deepcopy(loc) for index, loc in enumerate(locs) if index not in selected}
    if selected:
        x = [locs[index]["X"] for index in selected]
        y = [locs[index]["Y"] for index in selected]
        z = [locs[index]["Z"] for index in selected]
        new_x, new_y, new_z, _newcenter, _optim = chancenter(x, y, z, center)
        for offset, index in enumerate(selected):
            locs[index]["X"] = float(new_x[offset])
            locs[index]["Y"] = float(new_y[offset])
            locs[index]["Z"] = float(new_z[offset])
        locs = convertlocs(locs, "cart2all")
        for index, loc in omitted.items():
            locs[index] = loc
    command = _history_command(center, omitchans)
    if is_eeg:
        output["chanlocs"] = locs
        output["saved"] = "no"
        output = eeg_checkset(output)
        return (output, command) if return_com else output
    return (locs, command) if return_com else locs


def _selected_indices(locs: list[dict[str, Any]], omitchans: Any) -> list[int]:
    omitted = (
        {int(index) - 1 for index in parse_numeric_sequence(omitchans, dtype=int)}
        if omitchans not in (None, "", [])
        else set()
    )
    selected = []
    for index, loc in enumerate(locs):
        if index in omitted:
            continue
        if all(_has_number(loc, field) for field in ("X", "Y", "Z")):
            selected.append(index)
    return selected


def _has_number(loc: dict[str, Any], field: str) -> bool:
    try:
        return np.isfinite(float(loc[field]))
    except (KeyError, TypeError, ValueError):
        return False


def _history_command(center: Any, omitchans: Any) -> str:
    center_value = [] if center is None else center
    omit_value = [] if omitchans is None else omitchans
    return (
        "EEG = pop_chancenter(EEG, "
        f"{format_history_value(center_value, empty_sequence='[]')}, "
        f"{format_history_value(omit_value, empty_sequence='[]')});"
    )


__all__ = ["pop_chancenter"]
