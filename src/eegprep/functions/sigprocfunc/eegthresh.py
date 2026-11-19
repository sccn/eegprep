"""EEGLAB-compatible amplitude threshold rejection helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._rejection import eegthresh_marks, expand_values, parse_numeric_sequence


def eegthresh(
    signal: Any,
    pnts: int,
    electrodes: Any,
    negthresh: Any,
    posthresh: Any,
    timerange: Any,
    starttime: Any,
    endtime: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reject epochs with out-of-bounds values in a time window."""
    data = np.asarray(signal, dtype=float)
    frames = int(pnts)
    if data.ndim == 2:
        if data.shape[1] % frames:
            raise ValueError("pnts must divide the signal frame count")
        data = data.reshape(data.shape[0], frames, data.shape[1] // frames, order="F")
    elif data.ndim != 3:
        raise ValueError("signal must be 2-D or 3-D")
    elecs = parse_numeric_sequence(electrodes, dtype=int)
    if not elecs:
        elecs = list(range(1, data.shape[0] + 1))
    time_values = parse_numeric_sequence(timerange, dtype=float)
    if len(time_values) != 2:
        raise ValueError("timerange must contain [min max]")
    starts = expand_values(starttime, len(elecs), time_values[0])
    ends = expand_values(endtime, len(elecs), time_values[1])
    marks, row_marks = eegthresh_marks(data, elecs, negthresh, posthresh, tuple(time_values), starts, ends)
    rejected_index = np.flatnonzero(marks)
    accepted_index = np.flatnonzero(~marks)
    rejected = rejected_index + 1
    accepted = accepted_index + 1
    selected_rows = np.asarray([index - 1 for index in elecs], dtype=int)
    if accepted_index.size == 0:
        newsignal = np.empty((0, 0))
    else:
        newsignal = data[np.ix_(selected_rows, np.arange(data.shape[1]), accepted_index)]
        if np.asarray(signal).ndim == 2:
            newsignal = newsignal.reshape(len(selected_rows), -1, order="F")
        elif accepted_index.size == 1:
            newsignal = newsignal[:, :, 0]
    elec = (
        row_marks[selected_rows[:, np.newaxis], rejected_index[np.newaxis, :]]
        if rejected_index.size
        else np.zeros((len(elecs), 0), dtype=bool)
    )
    return accepted, rejected, newsignal, elec


__all__ = ["eegthresh"]
