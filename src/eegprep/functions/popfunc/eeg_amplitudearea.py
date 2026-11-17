"""Mean ERP amplitude over a time window."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline


def eeg_amplitudearea(
    EEG: dict[str, Any],
    channels: Any,
    resrate: float,
    wstart: float,
    wend: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed mean ERP amplitude within a millisecond window.

    Channel numbers are 1-based. ``resrate`` is samples per millisecond: the
    window is resampled in steps of ``1/resrate`` along ``EEG.times``.
    EEGLAB's help text describes ``resrate`` as hertz.
    """
    if wstart > wend:
        raise ValueError("wstart must not be greater than wend")
    if wstart == wend:
        raise ValueError("the integration window must have nonzero duration")
    if resrate <= 0:
        raise ValueError("resrate must be positive")
    data = np.asarray(EEG["data"], dtype=float)
    if data.ndim != 3:
        raise ValueError("EEG.data must be channel x sample x epoch")
    times = np.asarray(EEG.get("times", []), dtype=float)
    if times.size != data.shape[1]:
        times = np.linspace(float(EEG["xmin"]) * 1000, float(EEG["xmax"]) * 1000, data.shape[1])
    if wstart < times[0] or wend > times[-1]:
        raise ValueError("integration window must lie within EEG.times")
    channel_array = np.asarray(channels, dtype=int).ravel()
    if np.any(channel_array < 1) or np.any(channel_array > data.shape[0]):
        raise ValueError("channels must be 1-based and within EEG.nbchan")

    start_index = int(np.argmin(np.abs(times - wstart)))
    end_index = int(np.argmin(np.abs(times - wend)))
    if times[start_index] > wstart:
        start_index -= 1
    if times[end_index] < wend:
        end_index += 1
    start_index = max(start_index, 0)
    end_index = min(end_index, len(times) - 1)
    support_times = times[start_index : end_index + 1]
    if support_times.size < 2:
        raise ValueError("the integration window needs at least two supporting samples")

    step = 1.0 / float(resrate)
    resampled_times = [float(wstart)]
    while resampled_times[-1] < wend:
        resampled_times.append(resampled_times[-1] + step)
    grid = np.asarray(resampled_times)
    erp = np.mean(data, axis=2)
    amplitudes = np.empty(len(channel_array), dtype=float)
    for output_index, channel in enumerate(channel_array):
        channel_position = int(channel) - 1
        samples = CubicSpline(support_times, erp[channel_position, start_index : end_index + 1])(grid)
        area = 0.0
        for index in range(len(grid) - 1):
            width = min(grid[index + 1], wend) - grid[index]
            if width <= 0:
                break
            fraction = width / (grid[index + 1] - grid[index])
            endpoint = samples[index] + fraction * (samples[index + 1] - samples[index])
            area += width * (samples[index] + endpoint) / 2
        amplitudes[output_index] = area / (wend - wstart)
    return channel_array, amplitudes


__all__ = ["eeg_amplitudearea"]
