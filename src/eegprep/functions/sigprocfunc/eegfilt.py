"""Legacy EEGLAB FIR filtering for channel-major arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.plugins.firfilt._filtering import apply_eegfilt_legacy, design_eegfilt_legacy


def eegfilt(
    data: Any,
    srate: float,
    locutoff: float,
    hicutoff: float,
    epochframes: int | None = None,
    filtorder: int | None = None,
    revfilt: bool = False,
    firtype: str = "firls",
    causal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter an array using the legacy EEGLAB ``eegfilt`` design.

    Data are shaped ``(channels, frames * epochs)``. Each epoch is filtered
    independently when ``epochframes`` is supplied, preventing information
    from leaking across trial boundaries.

    Returns:
        ``(filtered_data, filter_coefficients)``.
    """
    array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("data must have shape (channels, frames * epochs)")
    if array.shape[0] > 1 and array.shape[1] == 1:
        raise ValueError("input data should be a row vector")
    total_frames = array.shape[1]
    samples_per_epoch = total_frames if epochframes in (None, 0) else int(epochframes)
    if samples_per_epoch <= 0 or total_frames % samples_per_epoch:
        raise ValueError("epochframes must be positive and divide the data length")

    coefficients, order = design_eegfilt_legacy(
        float(srate),
        locutoff=float(locutoff),
        hicutoff=float(hicutoff),
        filtorder=filtorder,
        revfilt=bool(revfilt),
        firtype=firtype,
    )
    trial_count = total_frames // samples_per_epoch
    shaped = array.reshape(array.shape[0], trial_count, samples_per_epoch).transpose(0, 2, 1)
    dataset = {
        "data": shaped,
        "nbchan": array.shape[0],
        "pnts": samples_per_epoch,
        "trials": trial_count,
        "event": [],
    }
    filtered = apply_eegfilt_legacy(dataset, coefficients, causal=bool(causal), filtorder=order)["data"]
    flattened = np.asarray(filtered).transpose(0, 2, 1).reshape(array.shape[0], total_frames)
    return flattened, coefficients


__all__ = ["eegfilt"]
