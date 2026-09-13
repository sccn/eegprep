"""Vector interpolation with optional moving averaging."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve

from ._validation import real_array
from .misc import round_mat


def vectdata(
    data: Any,
    times: Any,
    *,
    timesout: Any,
    method: Literal["linear", "cubic", "nearest", "v4"] = "linear",
    average: float | None = None,
    avgtype: Literal["const", "gauss"] = "const",
    border: Literal["on", "off"] = "off",
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate along the final data axis and optionally smooth first.

    ``average`` is a window width in the units of ``times``. MATLAB's legacy
    ``v4`` biharmonic ``griddata`` method has no well-defined one-dimensional
    equivalent and is rejected explicitly.
    """
    values = np.asarray(data)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("data must be numeric")
    input_times = real_array(times, "times").reshape(-1)
    output_times = real_array(timesout, "timesout").reshape(-1)
    was_vector = values.ndim == 1
    if was_vector:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != input_times.size:
        raise ValueError("data must be a vector or 2-D array whose final axis matches times")
    if input_times.size < 2 or np.any(np.diff(input_times) <= 0):
        raise ValueError("times must be strictly increasing and contain at least two points")
    if method == "v4":
        raise NotImplementedError("MATLAB's legacy griddata v4 method has no 1-D SciPy equivalent")
    if method not in {"linear", "cubic", "nearest"}:
        raise ValueError("method must be linear, cubic, nearest, or v4")
    if border not in {"on", "off"}:
        raise ValueError("border must be on or off")
    if avgtype not in {"const", "gauss"}:
        raise ValueError("avgtype must be const or gauss")

    working_times = input_times
    working_values = values
    if average is not None:
        if average <= 0:
            raise ValueError("average must be positive")
        spacings = np.diff(working_times)
        if not np.allclose(spacings, spacings[0], rtol=0, atol=1e-8):
            point_count = int(np.ceil((working_times[-1] - working_times[0]) / np.mean(spacings))) + 1
            uniform_times = np.linspace(working_times[0], working_times[-1], point_count)
            working_values = _interpolate(working_values, working_times, uniform_times, method)
            working_times = uniform_times
        window_points = max(1, int(round_mat(average / np.diff(working_times).mean())))
        kernel = _smoothing_kernel(window_points, avgtype)
        working_values = _smooth(working_values, kernel, correct_border=border == "on")
    result = _interpolate(working_values, working_times, output_times, method)
    return (result[0] if was_vector else result), output_times


def _interpolate(values: np.ndarray, times: np.ndarray, output: np.ndarray, method: str) -> np.ndarray:
    interpolator = interp1d(times, values, kind=method, axis=1, bounds_error=True, assume_sorted=True)
    return np.asarray(interpolator(output))


def _smoothing_kernel(points: int, avgtype: str) -> np.ndarray:
    if avgtype == "const":
        return np.ones(points, dtype=float) / points
    if avgtype == "gauss":
        locations = np.arange(points, dtype=float) - (points - 1) / 2
        width = max(0.15 * points, np.finfo(float).eps)
        kernel = np.exp(-0.5 * (locations / width) ** 2)
        return kernel / kernel.sum()
    raise ValueError("avgtype must be const or gauss")


def _smooth(values: np.ndarray, kernel: np.ndarray, *, correct_border: bool) -> np.ndarray:
    kernel_2d = kernel.reshape(1, -1)
    smoothed = convolve(values, kernel_2d, mode="same")
    if not correct_border:
        return smoothed
    normalization = convolve(np.ones_like(values), kernel_2d, mode="same")
    return smoothed / normalization


__all__ = ["vectdata"]
