"""EEGLAB-compatible conversion of epoch latency (ms) to nearest epoch frame."""

from __future__ import annotations

import math
from typing import Any


def eeg_ms2f(EEG: dict[str, Any], ms: float) -> int:
    """Convert an epoch latency in milliseconds to the nearest epoch frame.

    Ports EEGLAB's eeg_ms2f.m. The returned frame number is **1-based** to match
    EEGLAB; convert to a 0-based index at the point of use.

    Args:
        EEG: EEG structure with ``xmin``/``xmax`` (epoch limits, seconds) and
            ``pnts`` (samples per epoch).
        ms: Epoch latency in milliseconds.

    Returns:
        The nearest 1-based epoch frame number.

    Raises:
        ValueError: If the latency falls outside ``[xmin, xmax]``.
    """
    seconds = ms / 1000.0
    xmin = EEG["xmin"]
    xmax = EEG["xmax"]
    if seconds < xmin or seconds > xmax:
        raise ValueError("time out of range")
    frac = (EEG["pnts"] - 1) * (seconds - xmin) / (xmax - xmin)
    return 1 + _round_half_away_from_zero(frac)


def _round_half_away_from_zero(value: float) -> int:
    """Round to the nearest integer with ties going away from zero (MATLAB ``round``)."""
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


__all__ = ["eeg_ms2f"]
