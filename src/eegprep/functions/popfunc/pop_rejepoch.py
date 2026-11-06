"""Reject pre-marked epochs from an EEGLAB-style dataset."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._rejection import bool_marks
from eegprep.functions.adminfunc.eeg_checkset import _matlab_num2str
from eegprep.functions.adminfunc.vararg2str import _matlab_contarray
from eegprep.functions.popfunc.pop_select import pop_select


def pop_rejepoch(
    EEG: dict[str, Any],
    tmprej: Any = None,
    confirm: int | bool = 1,
    *,
    return_com: bool = False,
):
    """Remove rejected epochs.

    ``tmprej`` may be either a boolean vector of length ``EEG.trials`` or a
    list of EEGLAB-facing 1-based epoch indices. ``confirm`` is accepted for
    EEGLAB signature parity. ``confirm=-1`` keeps an empty result
    (``pop_rejepoch.m:96-97``, ``erroronempty`` off). Any other confirm value
    raises when every epoch is marked. This function does not open a dialog;
    the Reject marked epochs menu asks before it calls ``pop_rejepoch``.
    """
    if EEG is None:
        return (None, "") if return_com else None
    trials = int(EEG.get("trials", 1) or 1)
    if trials <= 1:
        raise ValueError("pop_rejepoch requires epoched data")
    if tmprej is None:
        tmprej = (EEG.get("reject") or {}).get("rejglobal", [])
    marks = bool_marks(tmprej, trials)
    indices = np.flatnonzero(marks) + 1
    if indices.size == 0:
        command = _history_command([])
        return (EEG, command) if return_com else EEG
    allow_empty = int(confirm) == -1
    if indices.size == trials and not allow_empty:
        raise ValueError(
            "pop_rejepoch: rejecting every epoch would leave the dataset empty. "
            "Clear at least one epoch mark, then choose Reject marked epochs again."
        )
    output = pop_select(
        EEG,
        "notrial",
        indices.tolist(),
        gui=False,
        erroronempty="off" if allow_empty else "on",
    )
    command = _history_command(indices.tolist())
    return (output, command) if return_com else output


def _history_command(indices: Any) -> str:
    # pop_rejepoch.m:101
    values = [float(item) for item in list(indices)]
    if not values:
        body = "[]"
    elif len(values) == 1:
        body = _matlab_num2str(values[0])
    else:
        body = "[" + _matlab_contarray(values) + "] "
    return f"EEG = pop_rejepoch( EEG, {body},0);"
