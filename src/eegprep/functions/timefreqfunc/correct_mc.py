"""Estimate independent time-frequency comparisons for correction."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from eegprep.functions.timefreqfunc.newtimef import newtimef


def correct_mc(
    EEG: dict[str, Any],
    cycles: Any = (3, 0.5),
    freqrange: Any = (2, 50),
    timesout: Any = (5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40),
) -> tuple[int, np.ndarray]:
    """Estimate an EEGLAB-style multiple-comparison correction count.

    EEGLAB fits a Ramberg-Schmeiser distribution to neighboring-bin
    correlations. EEGPrep uses Pearson p-values for the same neighbor
    correlations so the helper remains standalone and deterministic.
    """
    data = np.asarray(EEG.get("data"), dtype=float)
    if data.ndim == 3:
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    if data.ndim != 2:
        raise ValueError("EEG.data must be channel x points or channel x points x trials")
    pnts = int(EEG.get("pnts", data.shape[1]) or data.shape[1])
    srate = float(EEG.get("srate", 1) or 1)
    xmin = float(EEG.get("xmin", 0) or 0)
    xmax = float(EEG.get("xmax", (pnts - 1) / srate) or (pnts - 1) / srate)
    time_counts = np.asarray(timesout, dtype=int).ravel()
    freq_values = np.asarray(freqrange, dtype=float).ravel()
    if freq_values.size != 2:
        raise ValueError("freqrange must contain [min max]")
    nfreqs = int(np.ceil(np.log2(freq_values[1])))
    pvalues = np.full((nfreqs, time_counts.size), np.nan, dtype=float)
    for time_index, count in enumerate(time_counts):
        channel_ersp = []
        for channel in range(data.shape[0]):
            result = newtimef(
                data[channel, :pnts],
                pnts,
                [xmin * 1000.0, xmax * 1000.0],
                srate,
                cycles,
                timesout=int(count),
                freqscale="log",
                nfreqs=nfreqs,
                freqs=freq_values,
                plot="off",
            )
            channel_ersp.append(result.ersp)
        stacked = np.stack(channel_ersp, axis=0)
        for freq_index in range(stacked.shape[1]):
            correlations = []
            for channel in range(stacked.shape[0]):
                if stacked.shape[2] < 2:
                    continue
                corr = np.corrcoef(stacked[channel, freq_index, :-1], stacked[channel, freq_index, 1:])[0, 1]
                if np.isfinite(corr):
                    correlations.append(corr)
            if correlations:
                _statistic, pvalue = stats.ttest_1samp(correlations, 0.0, nan_policy="omit")
                pvalues[freq_index, time_index] = pvalue
    threshold = 0.05 / max(1, pvalues.size)
    ncorrect = 0
    for row in pvalues:
        significant = np.nonzero(row < threshold)[0]
        if significant.size:
            ncorrect += int(time_counts[significant[0]])
    return ncorrect, pvalues


__all__ = ["correct_mc"]
