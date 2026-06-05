"""EEGLAB-style ``newcrossf`` numerical core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from eegprep.functions.timefreqfunc.bootstat import exact_p_values
from eegprep.functions.timefreqfunc.newtimef import compute_time_frequency
from eegprep.functions.timefreqfunc.newtimefitc import newtimefitc
from eegprep.functions.timefreqfunc.newtimeftrialbaseln import baseline_indices


@dataclass(frozen=True)
class CrossFrequencyResult:
    """Computed coherence/cross-spectrum arrays."""

    coherence: np.ndarray
    phase: np.ndarray
    times: np.ndarray
    freqs: np.ndarray
    allcoher: np.ndarray
    alltf_x: np.ndarray
    alltf_y: np.ndarray
    figure: Any | None = None
    rboot: np.ndarray | None = None
    pvalues: np.ndarray | None = None
    significant: np.ndarray | None = None
    lagmap: np.ndarray | None = None


def newcrossf(
    x: Any,
    y: Any,
    frames: int,
    tlimits: Any,
    srate: float,
    cycles: Any = 0,
    **kwargs: Any,
) -> CrossFrequencyResult:
    """Compute an EEGLAB-like event-related coherence decomposition."""
    options = dict(kwargs)
    mode = str(options.pop("type", "phasecoher")).lower()
    alpha = _alpha_value(options.pop("alpha", None))
    rboot = _boot_array(options.pop("rboot", None))
    boottype = str(options.pop("boottype", "shuffle")).lower()
    baseboot = options.pop("baseboot", 1)
    subitc = str(options.get("subitc", "off"))
    shuffle_count = int(_first_numeric(options.pop("shuffle", 0), 0))
    amplag = _numeric_vector(options.pop("amplag", 0), dtype=int)
    if amplag.size == 0:
        amplag = np.asarray([0], dtype=int)
    naccu = int(_first_numeric(options.pop("naccu", 200), 200))
    rng = options.pop("rng", None)
    plot = options.pop("plot", "on")
    plotamp = options.pop("plotamp", options.pop("plotersp", "on"))
    plotphase = options.pop("plotphase", "on")
    plottype = str(options.pop("plottype", "image")).lower()
    title = str(options.pop("title", "Cross-coherence"))
    options.pop("condboot", None)
    compute_options = _compute_time_frequency_options(options)

    freqs, times, tf_x = compute_time_frequency(x, frames, tlimits, srate, cycles, **compute_options)
    freqs_y, times_y, tf_y = compute_time_frequency(y, frames, tlimits, srate, cycles, **compute_options)
    if tf_x.shape != tf_y.shape or not np.array_equal(freqs, freqs_y) or not np.array_equal(times, times_y):
        raise ValueError("newcrossf inputs must have matching frames, trials, times, and frequencies")
    if shuffle_count:
        tf_y = _shuffle_trials(tf_y, shuffle_count, rng)
    if subitc.lower() == "on":
        tf_x = _remove_itc(tf_x)
        tf_y = _remove_itc(tf_y)

    if tf_x.shape[2] == 1 and mode != "crossspec":
        mode = "crossspec"
    coherence_complex, allcoher, lagmap = _coherence(tf_x, tf_y, mode=mode, amplag=amplag, alpha=alpha)
    coherence = np.abs(coherence_complex)
    phase = np.angle(coherence_complex)

    pvalues = None
    significant = None
    if alpha is not None:
        if rboot is None:
            boot_indices = _bootstrap_indices(times, baseboot)
            rboot, surrogates = _bootstrap_coherence(
                tf_x,
                tf_y,
                mode,
                alpha=alpha,
                naccu=naccu,
                boottype=boottype,
                base_indices=boot_indices,
                rng=rng,
            )
            pvalues = exact_p_values(coherence, surrogates)
            significant = pvalues <= alpha
        else:
            significant = coherence >= _threshold_vector(rboot, coherence.shape)

    figure = None
    if _is_on(plot):
        figure = _plot_cross_frequency(
            coherence,
            phase,
            times,
            freqs,
            title=title,
            plotamp=_is_on(plotamp),
            plotphase=_is_on(plotphase),
            plottype=plottype,
            significant=significant,
        )
    return CrossFrequencyResult(
        coherence, phase, times, freqs, allcoher, tf_x, tf_y, figure, rboot, pvalues, significant, lagmap
    )


def _compute_time_frequency_options(options: dict[str, Any]) -> dict[str, Any]:
    compute_keys = {
        "winsize",
        "freqs",
        "freqrange",
        "nfreqs",
        "freqscale",
        "timesout",
        "padratio",
        "overlap",
    }
    supported_keys = compute_keys | {"subitc"}
    unknown = sorted(set(options) - supported_keys)
    if unknown:
        raise TypeError(f"Unsupported newcrossf option(s): {', '.join(unknown)}")
    if "freqrange" in options and "freqs" not in options:
        options["freqs"] = options.pop("freqrange")
    return {key: options[key] for key in compute_keys if key in options}


def _coherence(
    tf_x: np.ndarray,
    tf_y: np.ndarray,
    *,
    mode: str,
    amplag: np.ndarray,
    alpha: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    cross = tf_x * np.conjugate(tf_y)
    if mode == "coher":
        denominator = np.sqrt(np.sum(np.abs(tf_x) ** 2, axis=2) * np.sum(np.abs(tf_y) ** 2, axis=2))
        coherence_complex = np.sum(cross, axis=2) / np.maximum(denominator, np.finfo(float).tiny)
        return coherence_complex, cross, None
    if mode == "crossspec":
        return np.mean(cross, axis=2), cross, None
    if mode == "phasecoher2":
        coherence_complex = np.sum(cross, axis=2) / np.maximum(np.sum(np.abs(cross), axis=2), np.finfo(float).tiny)
        return coherence_complex, cross, None
    if mode == "phasecoher":
        unit = cross / np.maximum(np.abs(cross), np.finfo(float).tiny)
        return np.mean(unit, axis=2), cross, None
    if mode == "amp":
        return _amplitude_correlation(tf_x, tf_y, amplag=amplag, alpha=alpha), cross, None
    raise NotImplementedError("newcrossf supports type='phasecoher', 'phasecoher2', 'coher', 'crossspec', or 'amp'")


def _amplitude_correlation(
    tf_x: np.ndarray,
    tf_y: np.ndarray,
    *,
    amplag: np.ndarray,
    alpha: float | None,
) -> np.ndarray:
    amp_x = np.abs(tf_x)
    amp_y = np.abs(tf_y)
    results = np.zeros((tf_x.shape[0], tf_x.shape[1], amplag.size), dtype=complex)
    pvalues = np.ones(results.shape, dtype=float)
    for lag_index, lag in enumerate(amplag):
        for freq_index in range(tf_x.shape[0]):
            start = max(0, -int(lag))
            stop = min(tf_x.shape[1] - int(lag), tf_x.shape[1])
            for time_index in range(start, stop):
                first = amp_x[freq_index, time_index, :]
                second = amp_y[freq_index, time_index + int(lag), :]
                if np.nanstd(first) == 0 or np.nanstd(second) == 0:
                    continue
                corr, pvalue = stats.pearsonr(first, second)
                results[freq_index, time_index, lag_index] = corr
                pvalues[freq_index, time_index, lag_index] = pvalue
    if alpha is not None:
        results[pvalues > alpha] = 0
    if amplag.size == 1:
        return results[:, :, 0]
    max_indices = np.argmax(np.abs(results), axis=2)
    rows, cols = np.indices(results.shape[:2])
    lagmap = amplag[max_indices]
    max_result = results[rows, cols, max_indices]
    max_lag = max(1, int(np.nanmax(np.abs(amplag))))
    return max_result * np.exp(1j * lagmap / max_lag)


def _bootstrap_coherence(
    tf_x: np.ndarray,
    tf_y: np.ndarray,
    mode: str,
    *,
    alpha: float,
    naccu: int,
    boottype: str,
    base_indices: np.ndarray,
    rng: Any,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(rng)
    surrogates = np.empty((int(naccu), tf_x.shape[0], tf_x.shape[1]), dtype=float)
    source_x = tf_x[:, base_indices, :] if base_indices.size else tf_x
    source_y = tf_y[:, base_indices, :] if base_indices.size else tf_y
    threshold_source = np.empty((int(naccu), tf_x.shape[0], max(1, source_x.shape[1])), dtype=float)
    for index in range(int(naccu)):
        sample_x, sample_y = _resample_pair(tf_x, tf_y, generator, boottype=boottype)
        coher, _allcoher, _lagmap = _coherence(sample_x, sample_y, mode=mode, amplag=np.asarray([0]), alpha=None)
        surrogates[index] = np.abs(coher)
        boot_x, boot_y = _resample_pair(source_x, source_y, generator, boottype=boottype)
        boot_coher, _allcoher, _lagmap = _coherence(boot_x, boot_y, mode=mode, amplag=np.asarray([0]), alpha=None)
        threshold_source[index] = np.abs(boot_coher)
    thresholds = _upper_thresholds_by_frequency(threshold_source, alpha=alpha)
    return thresholds, surrogates


def _resample_pair(
    tf_x: np.ndarray, tf_y: np.ndarray, generator: np.random.Generator, *, boottype: str
) -> tuple[np.ndarray, np.ndarray]:
    mode = str(boottype).lower()
    if mode in {"shuffle", "shufftrials"}:
        indices = generator.permutation(tf_y.shape[2])
        return tf_x, tf_y[:, :, indices]
    if mode in {"rand", "randall"}:
        phases = np.exp(1j * generator.uniform(0.0, 2.0 * np.pi, size=tf_x.shape))
        return tf_x * phases, tf_y
    raise ValueError("boottype must be 'shuffle', 'shufftrials', 'rand', or 'randall'")


def _upper_thresholds_by_frequency(values: np.ndarray, *, alpha: float) -> np.ndarray:
    reshaped = values.transpose(1, 0, 2).reshape(values.shape[1], -1)
    sorted_values = np.sort(reshaped, axis=1)
    tail_count = max(1, int(round(sorted_values.shape[1] * alpha)))
    return np.nanmean(sorted_values[:, -tail_count:], axis=1)


def _shuffle_trials(tf_y: np.ndarray, count: int, rng: Any) -> np.ndarray:
    generator = np.random.default_rng(rng)
    shuffled = np.asarray(tf_y).copy()
    for _ in range(int(count)):
        shuffled = shuffled[:, :, generator.permutation(shuffled.shape[2])]
    return shuffled


def _remove_itc(tfdata: np.ndarray) -> np.ndarray:
    itc = newtimefitc(tfdata, "phasecoher")
    magnitude = np.maximum(np.abs(tfdata), np.finfo(float).tiny)
    return (tfdata - magnitude * itc[:, :, np.newaxis]) / magnitude


def _bootstrap_indices(times: np.ndarray, baseboot: Any) -> np.ndarray:
    values = _numeric_vector(baseboot)
    if values.size == 0:
        return np.nonzero(times <= 0)[0]
    if values.size == 1:
        if values[0] == 0:
            return np.asarray([], dtype=int)
        indices = np.nonzero(times <= values[0])[0]
        return indices if indices.size else np.arange(times.size, dtype=int)
    return baseline_indices(times, values)


def _threshold_vector(thresholds: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(thresholds, dtype=float).squeeze()
    if values.ndim == 0:
        return np.full(target_shape, float(values))
    if values.ndim == 1:
        return values[:, np.newaxis]
    return values


def _plot_cross_frequency(
    coherence: np.ndarray,
    phase: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    title: str,
    plotamp: bool,
    plotphase: bool,
    plottype: str,
    significant: np.ndarray | None,
):
    panels = int(plotamp) + int(plotphase)
    if panels == 0:
        return None
    fig, axes = plt.subplots(panels, 1, figsize=(7.5, 5.0), squeeze=False)
    row = 0
    if plotamp:
        _plot_panel(
            axes[row, 0],
            fig,
            coherence,
            times,
            freqs,
            title=title,
            label="Coherence",
            plottype=plottype,
            significant=significant,
            vmin=0,
            vmax=max(1.0, float(np.nanmax(coherence))),
        )
        row += 1
    if plotphase:
        _plot_panel(
            axes[row, 0],
            fig,
            phase,
            times,
            freqs,
            title="" if plotamp else title,
            label="Phase (rad)",
            plottype=plottype,
            significant=significant,
            vmin=-np.pi,
            vmax=np.pi,
        )
    axes[panels - 1, 0].set_xlabel("Time (ms)")
    fig.tight_layout()
    return fig


def _plot_panel(
    axis,
    figure,
    values: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    title: str,
    label: str,
    plottype: str,
    significant: np.ndarray | None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if plottype == "curve":
        for freq_index, freq in enumerate(freqs):
            line_values = values[freq_index]
            if significant is not None:
                line_values = np.where(significant[freq_index], line_values, np.nan)
            axis.plot(times, line_values, label=f"{freq:g} Hz")
        if freqs.size <= 12:
            axis.legend(loc="best", fontsize="small")
    elif plottype == "image":
        image_values = values if significant is None else np.where(significant, values, np.nan)
        image = axis.imshow(
            image_values,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        figure.colorbar(image, ax=axis, label=label)
    else:
        raise ValueError("plottype must be 'image' or 'curve'")
    if title:
        axis.set_title(title)
    axis.set_ylabel("Frequency (Hz)")


def _alpha_value(alpha: Any) -> float | None:
    values = _numeric_vector(alpha)
    if values.size == 0 or values[0] == 0 or np.isnan(values[0]):
        return None
    if values[0] <= 0 or values[0] > 0.5:
        raise ValueError("alpha must be in the interval (0, 0.5]")
    return float(values[0])


def _boot_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    values = np.asarray(value, dtype=float)
    if values.size == 0 or np.isnan(values.reshape(-1)[0]):
        return None
    return values.squeeze()


def _numeric_vector(value: Any, *, dtype: Any = float) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    if isinstance(value, np.ndarray):
        return value.astype(dtype).ravel()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.asarray([value], dtype=dtype)
    if isinstance(value, str):
        text = value.strip().strip("[]")
        if not text:
            return np.asarray([], dtype=dtype)
        return np.asarray([float(token) for token in text.replace(",", " ").split()], dtype=dtype)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=dtype).ravel()
    return np.asarray([value], dtype=dtype)


def _first_numeric(value: Any, default: float) -> float:
    values = _numeric_vector(value)
    return float(values[0]) if values.size else float(default)


def _is_on(value: Any) -> bool:
    return str(value).lower() not in {"0", "false", "off", "no", "none"}


__all__ = ["CrossFrequencyResult", "newcrossf"]
