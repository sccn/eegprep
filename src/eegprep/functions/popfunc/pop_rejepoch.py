"""Reject pre-marked epochs from an EEGLAB-style dataset."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import format_history_value
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
    list of EEGLAB-facing 1-based epoch indices. ``confirm`` is accepted for
    EEGLAB signature parity; browser-style confirmation is not opened in
    EEGPrep.
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
        command = _history_command([], confirm)
        return (EEG, command) if return_com else EEG
    output = pop_select(EEG, "notrial", indices.tolist(), gui=False)
    command = _history_command(indices.tolist(), confirm)
    return (output, command) if return_com else output


def _history_command(indices: Any, confirm: int | bool) -> str:
    return f"EEG = pop_rejepoch(EEG, {format_history_value(indices, cell_for_sequence=None)}, {int(bool(confirm))});"
