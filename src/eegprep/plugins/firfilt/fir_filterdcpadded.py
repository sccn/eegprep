"""DC-padded FIR filtering helper."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import fftconvolve, lfilter


def fir_filterdcpadded(
    b: Any,
    a: Any,
    data: Any,
    causal: bool = False,
    usefftfilt: bool = False,
) -> np.ndarray:
    """Pad data with endpoint constants and filter along the first axis."""
    coeffs = np.asarray(b, dtype=float).ravel()
    denom = np.asarray(a, dtype=float).ravel()
    if coeffs.size == 0 or denom.size != 1 or denom[0] != 1:
        raise ValueError(
            "Not a FIR filter. onepass-zerophase and onepass-minphase filtering is available for FIR filters only."
        )
    if coeffs.size % 2 != 1:
        raise ValueError("Filter order is not even.")
    x = np.asarray(data)
    if x.ndim == 1:
        x = x[:, np.newaxis]
        squeeze = True
    elif x.ndim == 2:
        squeeze = False
    else:
        raise ValueError("data must be 1D or 2D with time along the first axis.")
    group_delay = (coeffs.size - 1) // 2
    if group_delay == 0:
        return np.asarray(data).copy()
    if not causal:
        is_sym = np.allclose(coeffs[:group_delay], coeffs[:group_delay:-1])
        is_antisym = np.allclose(coeffs[:group_delay], -coeffs[:group_delay:-1]) and np.isclose(coeffs[group_delay], 0)
        if not (is_sym or is_antisym):
            raise ValueError(
                "Filter is not anti-/symmetric. For onepass-zerophase filtering the filter must be anti-/symmetric."
            )
    was_single = x.dtype == np.float32
    x = x.astype(float, copy=False)
    start_width = 2 * group_delay if causal else group_delay
    start_pad = np.repeat(x[:1, :], start_width, axis=0)
    end_pad = np.empty((0, x.shape[1]), dtype=float) if causal else np.repeat(x[-1:, :], group_delay, axis=0)
    padded = np.concatenate([start_pad, x, end_pad], axis=0)
    if usefftfilt:
        filtered = fftconvolve(padded, coeffs[:, np.newaxis], mode="full", axes=0)[: padded.shape[0], :]
    else:
        filtered = lfilter(coeffs, [1.0], padded, axis=0)
    filtered = filtered[2 * group_delay :, :]
    if was_single:
        filtered = filtered.astype(np.float32)
    if squeeze:
        return filtered[:, 0]
    return filtered
