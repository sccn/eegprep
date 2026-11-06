"""Reject pre-marked epochs from an EEGLAB-style dataset."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.adminfunc.vararg2str import matlab_array2str
from eegprep.functions.popfunc._rejection import bool_marks
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
    list of EEGLAB-facing 1-based epoch indices. The history command is always
    ``pop_rejepoch( EEG, indices ,0)`` (``pop_rejepoch.m:101``). ``confirm=-1``
    keeps an empty result (``pop_rejepoch.m:96-97``, ``erroronempty`` off).
    Any other confirm value raises when every epoch is marked. This function
    does not open a dialog. The scroll window asks before removing epochs, and
    the Reject marked epochs menu superposes marks before it calls this function.
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
    return f"EEG = pop_rejepoch( EEG, {matlab_array2str(indices)},0);"
