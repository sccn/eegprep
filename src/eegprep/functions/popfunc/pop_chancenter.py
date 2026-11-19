"""EEGLAB-style wrapper for recentering channel coordinates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.miscfunc.matlab_inputs import is_matlab_empty
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
    if selected:
        x = [locs[index]["X"] for index in selected]
        y = [locs[index]["Y"] for index in selected]
        z = [locs[index]["Z"] for index in selected]
        new_x, new_y, new_z, newcenter, _optim = chancenter(x, y, z, center)
        for offset, index in enumerate(selected):
            locs[index]["X"] = float(np.asarray(new_x).ravel()[offset])
            locs[index]["Y"] = float(np.asarray(new_y).ravel()[offset])
            locs[index]["Z"] = float(np.asarray(new_z).ravel()[offset])
        locs = convertlocs(locs, "cart2all")
    command = _history_command(center, omitchans)
    fitted = newcenter if selected else np.empty((0, 0))
    if is_eeg:
        output["chanlocs"] = locs
        output["saved"] = "no"
        output = eeg_checkset(output)
        return (output, command) if return_com else output
    return (locs, command) if return_com else _ChanCenterResult(locs, fitted, command)


class _ChanCenterResult(list):
    """Channel locations that also expose MATLAB's center and command outputs."""

    matlab_outputs = ("chans", "center", "command")

    def __init__(self, chans: Any, center: Any, command: str):
        locations = list(chans) if not isinstance(chans, dict) else []
        super().__init__(locations)
        self.chans = chans
        self.center = np.asarray(center)
        self.command = command


def _selected_indices(locs: list[dict[str, Any]], omitchans: Any) -> list[int]:
    omitted = (
        set()
        if is_matlab_empty(omitchans)
        else {int(index) - 1 for index in parse_numeric_sequence(omitchans, dtype=int)}
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
