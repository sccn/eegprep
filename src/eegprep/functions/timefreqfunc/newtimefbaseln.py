"""Average baseline correction for EEGLAB-style ``newtimef`` power."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.timefreqfunc.newtimeftrialbaseln import baseline_indices


def newtimefbaseln(
    power: Any,
    timesout: Any,
    *,
    baseline: Any = 0,
    powbase: Any = None,
    basenorm: str = "off",
    commonbase: str = "off",
    singletrials: str = "on",
    trialbase: str = "off",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply EEGLAB average baseline correction to absolute power.

    Args:
        power: Power array, usually ``freqs x times x trials``.
        timesout: Time values in milliseconds.
        baseline: Scalar threshold, ``[min, max]`` interval, or NaN to disable.
        powbase: Optional externally supplied baseline spectrum.
        basenorm: ``"on"`` for z-like baseline normalization.
        commonbase: Average baseline spectra across a list of power arrays.
        singletrials: Whether trials remain present on the last axis.
        trialbase: Trial baseline mode from ``newtimef``.

    Returns:
        ``(corrected_power, baseline_indices, baseline_values)``.
    """
    powers = [np.asarray(power, dtype=float).copy()]
    corrected, baseln, bases = _newtimefbaseln_list(
        powers,
        timesout,
        baseline=baseline,
        powbase=powbase,
        basenorm=basenorm,
        commonbase=commonbase,
        singletrials=singletrials,
        trialbase=trialbase,
    )
    return corrected[0], baseln, bases[0]


def _newtimefbaseln_list(
    powers: list[np.ndarray],
    timesout: Any,
    *,
    baseline: Any,
    powbase: Any,
    basenorm: str,
    commonbase: str,
    singletrials: str,
    trialbase: str,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    baseln = baseline_indices(timesout, baseline)
    baselines: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for values in powers:
        mean, std = _baseline_mean_std(values, baseln, powbase, singletrials=singletrials, trialbase=trialbase)
        baselines.append(mean)
        stds.append(std)

    if str(commonbase).lower() == "on" and baselines:
        common_mean = sum(baselines) / len(baselines)
        common_std = sum(stds) / len(stds)
        baselines = [common_mean for _ in baselines]
        stds = [common_std for _ in stds]

    baseline_values = np.asarray(baseline, dtype=float).reshape(-1)
    disabled = baseline_values.size and np.isnan(baseline_values[0])
    normalize = str(basenorm).lower() == "on"
    corrected: list[np.ndarray] = []
    for values, mean, std in zip(powers, baselines, stds):
        if not disabled and str(trialbase).lower() != "on":
            if normalize:
                values = (values - mean) / np.maximum(std, np.finfo(float).tiny)
            else:
                values = values / np.maximum(mean, np.finfo(float).tiny)
        corrected.append(values)
    return corrected, baseln, [_plot_baseline_value(base) for base in baselines]


def _baseline_mean_std(
    values: np.ndarray,
    baseln: np.ndarray,
    powbase: Any,
    *,
    singletrials: str,
    trialbase: str,
) -> tuple[np.ndarray, np.ndarray]:
    supplied = _powbase_array(powbase)
    if supplied is not None:
        mean = supplied
        if mean.ndim == 1:
            mean = mean[:, np.newaxis]
        return _broadcast_baseline(mean, values), np.ones_like(_broadcast_baseline(mean, values))

    time_axis = values.ndim - 2
    if str(singletrials).lower() == "on" and str(trialbase).lower() == "off":
        baseline_source = np.nanmean(values, axis=-1)
    else:
        baseline_source = values
    selected = np.take(baseline_source, baseln, axis=time_axis if baseline_source.ndim == values.ndim else -1)
    mean = np.nanmean(selected, axis=time_axis if baseline_source.ndim == values.ndim else -1, keepdims=True)
    ddof = 1 if baseln.size > 1 else 0
    std = np.nanstd(selected, axis=time_axis if baseline_source.ndim == values.ndim else -1, ddof=ddof, keepdims=True)
    return _broadcast_baseline(mean, values), _broadcast_baseline(std, values)


def _broadcast_baseline(baseline: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.asarray(baseline, dtype=float)
    while result.ndim < values.ndim:
        result = np.expand_dims(result, axis=-1)
    return result


def _plot_baseline_value(baseline: np.ndarray) -> np.ndarray:
    value = np.asarray(baseline, dtype=float)
    if value.ndim > 2:
        value = np.nanmean(value, axis=-1)
    return np.squeeze(value)


def _powbase_array(powbase: Any) -> np.ndarray | None:
    if powbase is None:
        return None
    values = np.asarray(powbase, dtype=float)
    if values.size == 0 or np.isnan(values.reshape(-1)[0]):
        return None
    return values


__all__ = ["newtimefbaseln"]
