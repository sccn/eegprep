"""Inter-trial coherence from single-trial spectral estimates."""

from __future__ import annotations

from typing import Any

import numpy as np


def newtimefitc(tfdecomp: Any, itctype: str = "phasecoher") -> np.ndarray:
    """Compute EEGLAB-style inter-trial coherence.

    Args:
        tfdecomp: Complex spectral estimates with trials on the last axis.
        itctype: ``"phasecoher"``, ``"phasecoher2"``, or ``"coher"``.

    Returns:
        Complex coherence values with the trial axis removed.
    """
    values = np.asarray(tfdecomp, dtype=complex)
    if values.ndim < 3:
        raise ValueError("tfdecomp must contain frequency, time, and trial dimensions")
    trial_axis = values.ndim - 1
    mode = str(itctype).lower()
    magnitude = np.maximum(np.abs(values), np.finfo(float).tiny)
    if mode == "coher":
        denominator = np.sqrt(np.sum(values * np.conjugate(values), axis=trial_axis) * values.shape[trial_axis])
        return np.sum(values, axis=trial_axis) / np.maximum(denominator, np.finfo(float).tiny)
    if mode == "phasecoher2":
        return np.sum(values, axis=trial_axis) / np.maximum(np.sum(magnitude, axis=trial_axis), np.finfo(float).tiny)
    if mode == "phasecoher":
        return np.mean(values / magnitude, axis=trial_axis)
    raise ValueError("itctype must be 'phasecoher', 'phasecoher2', or 'coher'")


__all__ = ["newtimefitc"]
