"""Boundary-aware FIR filtering helper."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.plugins.firfilt.fir_filterdcpadded import fir_filterdcpadded
from eegprep.plugins.firfilt.findboundaries import findboundaries


def firfiltsplit(
    EEG: dict[str, Any],
    b: Any,
    causal: bool = False,
    usefftfilt: bool = False,
    chaninds: Any = None,
) -> dict[str, Any]:
    """Filter continuous chunks split at boundaries or each epoch independently."""
    output = deepcopy(EEG)
    data = np.asarray(output["data"])
    if data.ndim not in {2, 3}:
        raise ValueError("FIR filtering supports continuous or epoched EEG data")
    nbchan = int(output.get("nbchan", data.shape[0]))
    pnts = int(output.get("pnts", data.shape[1]))
    trials = int(output.get("trials", data.shape[2] if data.ndim == 3 else 1))
    channels = _chaninds(chaninds, nbchan)
    flattened = data.transpose(0, 2, 1).reshape(nbchan, pnts * trials) if data.ndim == 3 else data.copy()
    filtered = flattened.astype(float, copy=True)
    boundaries = _epoched_boundaries(pnts, trials) if trials > 1 else _continuous_boundaries(output, pnts)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop <= start:
            continue
        segment = filtered[np.ix_(channels, np.arange(start, stop))]
        filtered[np.ix_(channels, np.arange(start, stop))] = fir_filterdcpadded(
            b,
            1,
            segment.T,
            causal=causal,
            usefftfilt=usefftfilt,
        ).T
    output["data"] = filtered.reshape(nbchan, trials, pnts).transpose(0, 2, 1) if data.ndim == 3 else filtered
    output["icaact"] = np.array([])
    output["saved"] = "no"
    return output


def _chaninds(chaninds: Any, nbchan: int) -> list[int]:
    if chaninds is None:
        return list(range(nbchan))
    values = np.asarray(chaninds).ravel().tolist()
    if not values:
        return list(range(nbchan))
    indices = [int(value) - 1 for value in values]
    if any(index < 0 or index >= nbchan for index in indices):
        raise ValueError("Channel indices must be 1-based and within EEG.nbchan")
    return list(dict.fromkeys(indices))


def _continuous_boundaries(EEG: dict[str, Any], pnts: int) -> np.ndarray:
    starts = [int(value) - 1 for value in findboundaries(EEG.get("event"))]
    starts = [start for start in starts if 0 <= start <= pnts]
    starts.append(pnts)
    return np.asarray(sorted(set(starts)), dtype=int)


def _epoched_boundaries(pnts: int, trials: int) -> np.ndarray:
    return np.arange(0, pnts * trials + 1, pnts, dtype=int)
