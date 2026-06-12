"""EEGLAB-style ``newtimef`` numerical core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._pop_utils import is_on as _is_on
from eegprep.functions.popfunc._pop_utils import parse_numeric_sequence
from eegprep.functions.statistics.fdr import fdr
from eegprep.functions.timefreqfunc.bootstat import bootstrap_threshold, exact_p_values
from eegprep.functions.timefreqfunc.newtimefbaseln import newtimefbaseln
from eegprep.functions.timefreqfunc.newtimefitc import newtimefitc
from eegprep.functions.timefreqfunc.newtimeftrialbaseln import baseline_indices, newtimeftrialbaseln
from eegprep.functions.timefreqfunc.timefreq import timefreq


@dataclass(frozen=True)
class TimeFrequencyResult:
    """Computed event-related spectral perturbation and ITC arrays."""

    ersp: np.ndarray
    itc: np.ndarray
    powbase: np.ndarray
    times: np.ndarray
    freqs: np.ndarray
    tfdata: np.ndarray
    figure: Any | None = None
    erspboot: np.ndarray | None = None
    itcboot: np.ndarray | None = None
    ersp_pvalues: np.ndarray | None = None
    itc_pvalues: np.ndarray | None = None
    ersp_significant: np.ndarray | None = None
    itc_significant: np.ndarray | None = None
    timewarp_markers: np.ndarray | None = None


def newtimef(
    data: Any,
    frames: int,
    tlimits: Any,
    srate: float,
    cycles: Any = 0,
    *,
    winsize: Any = None,
    freqs: Any = None,
    freqrange: Any = None,
    nfreqs: Any = None,
    freqscale: Any = None,
    timesout: Any = None,
    padratio: Any = None,
    overlap: Any = None,
    baseline: Any = 0,
    scale: str = "log",
    basenorm: str = "off",
    trialbase: str = "off",
    powbase: Any = None,
    alpha: Any = None,
    pboot: Any = None,
    rboot: Any = None,
    erspboot: Any = None,
    itcboot: Any = None,
    naccu: int = 200,
    boottype: str = "shuffle",
    baseboot: Any = 1,
    mcorrect: str = "none",
    itctype: str = "phasecoher",
    type: str | None = None,
    subitc: str = "off",
    plottype: str = "image",
    plot: Any = "on",
    plotersp: Any = "on",
    plotitc: Any = "on",
    plotphase: Any = "off",
    title: str = "Time-frequency",
    rng: Any = None,
    detrend: str = "off",
    causal: str = "off",
    wletmethod: str = "dftfilt3",
    timewarp: Any = None,
    timewarpms: Any = None,
    timewarpidx: Any = None,
    vert: Any = None,
    verbose: str = "off",
) -> TimeFrequencyResult:
    """Compute an EEGLAB-like ERSP/ITC time-frequency decomposition."""
    if overlap is not None:
        raise NotImplementedError("newtimef does not implement the 'overlap' option")
    if str(plotphase).strip().lower() not in {"off", "0", "false", "no", "none"}:
        raise NotImplementedError("newtimef does not implement the 'plotphase' option")
    if freqs is None and freqrange is not None:
        freqs = freqrange
    if type is not None:
        itctype = type
    scale_mode = str(scale).lower()
    normalize_baseline = str(basenorm).lower()
    if normalize_baseline == "on":
        scale_mode = "abs"
    if scale_mode not in {"log", "abs"}:
        raise ValueError("scale must be 'log' or 'abs'")

    timestretch, timewarp_markers = _timewarp_options(timewarp, timewarpms, timewarpidx, frames, tlimits, srate)
    vertical_markers = timewarp_markers
    if vertical_markers is None:
        vert_values = _numeric_vector(vert)
        vertical_markers = vert_values if vert_values.size else None
        if vertical_markers is not None:
            _validate_vertical_markers(vertical_markers, tlimits)

    decomp = _compute_decomposition(
        data,
        frames,
        tlimits,
        srate,
        cycles,
        winsize=winsize,
        freqs=freqs,
        nfreqs=nfreqs,
        freqscale=freqscale,
        timesout=timesout,
        padratio=padratio,
        itctype=itctype,
        subitc=subitc,
        detrend=detrend,
        causal=causal,
        wletmethod=wletmethod,
        timestretch=timestretch,
        verbose=verbose,
    )
    tfdata = _normalize_fft_tfdata(decomp.tfdata, decomp.cycles, decomp.winsize)
    power = np.abs(tfdata) ** 2
    corrected_power = newtimeftrialbaseln(
        power,
        decomp.times,
        baseline=baseline,
        basenorm=normalize_baseline,
        trialbase=trialbase,
    )
    powbase_values = powbase
    if powbase_values is not None and scale_mode == "log":
        powbase_array = np.asarray(powbase_values, dtype=float)
        if powbase_array.size and not np.isnan(powbase_array.reshape(-1)[0]):
            powbase_values = 10.0 ** (powbase_array / 10.0)
    corrected_power, baseln, powbase_array = newtimefbaseln(
        corrected_power,
        decomp.times,
        baseline=baseline,
        powbase=powbase_values,
        basenorm=normalize_baseline,
        trialbase=trialbase,
        singletrials="on",
    )
    ersp_power = np.nanmean(corrected_power, axis=2)
    ersp = _power_to_output(ersp_power, scale_mode)
    itc = newtimefitc(tfdata, itctype)

    alpha_value = _alpha_value(alpha)
    supplied_erspboot = _first_not_none(erspboot, pboot)
    supplied_itcboot = _first_not_none(itcboot, rboot)
    ersp_boot = _boot_array(supplied_erspboot)
    itc_boot = _boot_array(supplied_itcboot)
    ersp_pvalues = None
    itc_pvalues = None
    ersp_significant = None
    itc_significant = None
    if alpha_value is not None:
        boot_indices = _bootstrap_indices(decomp.times, baseline, baseboot, baseln)
        if ersp_boot is None:
            ersp_boot, ersp_surrogates = _bootstrap_power(
                corrected_power,
                scale_mode,
                alpha=alpha_value,
                naccu=naccu,
                boottype=boottype,
                base_indices=boot_indices,
                rng=rng,
            )
            ersp_pvalues = exact_p_values(ersp, ersp_surrogates)
            ersp_significant = _significance_mask(ersp_pvalues, alpha_value, mcorrect)
        else:
            ersp_significant = _threshold_mask(ersp, ersp_boot)
        if itc_boot is None:
            itc_boot, itc_surrogates = _bootstrap_itc(
                tfdata,
                itctype,
                alpha=alpha_value,
                naccu=naccu,
                boottype=boottype,
                base_indices=boot_indices,
                rng=rng,
            )
            itc_pvalues = exact_p_values(np.abs(itc), itc_surrogates)
            itc_significant = _significance_mask(itc_pvalues, alpha_value, mcorrect)
        else:
            itc_significant = np.abs(itc) >= _threshold_vector(itc_boot, itc.shape)

    figure = None
    if _is_on(plot):
        figure = _plot_time_frequency(
            ersp,
            np.abs(itc),
            decomp.times,
            decomp.freqs,
            title=str(title),
            plotersp=_is_on(plotersp),
            plotitc=_is_on(plotitc),
            plottype=str(plottype).lower(),
            ersp_significant=ersp_significant,
            itc_significant=itc_significant,
            vertical_markers=vertical_markers,
        )
    return TimeFrequencyResult(
        ersp,
        itc,
        np.asarray(powbase_array).squeeze(),
        decomp.times,
        decomp.freqs,
        tfdata,
        figure,
        ersp_boot,
        itc_boot,
        ersp_pvalues,
        itc_pvalues,
        ersp_significant,
        itc_significant,
        vertical_markers,
    )


def compute_time_frequency(
    data: Any,
    frames: int,
    tlimits: Any,
    srate: float,
    cycles: Any = 0,
    *,
    winsize: Any = None,
    freqs: Any = None,
    nfreqs: Any = None,
    freqscale: Any = None,
    timesout: Any = None,
    padratio: Any = None,
    overlap: Any = None,
    timewarp: Any = None,
    timewarpms: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(freqs, times_ms, tfdata)`` for one signal."""
    if overlap is not None:
        raise NotImplementedError("compute_time_frequency does not implement the 'overlap' option")
    timestretch, _markers = _timewarp_options(timewarp, timewarpms, None, frames, tlimits, srate)
    decomp = _compute_decomposition(
        data,
        frames,
        tlimits,
        srate,
        cycles,
        winsize=winsize,
        freqs=freqs,
        nfreqs=nfreqs,
        freqscale=freqscale,
        timesout=timesout,
        padratio=padratio,
        timestretch=timestretch,
    )
    return decomp.freqs, decomp.times, decomp.tfdata


def _compute_decomposition(
    data: Any,
    frames: int,
    tlimits: Any,
    srate: float,
    cycles: Any,
    *,
    winsize: Any = None,
    freqs: Any = None,
    nfreqs: Any = None,
    freqscale: Any = None,
    timesout: Any = None,
    padratio: Any = None,
    itctype: str = "phasecoher",
    subitc: str = "off",
    detrend: str = "off",
    causal: str = "off",
    wletmethod: str = "dftfilt3",
    timestretch: Any = None,
    verbose: str = "off",
):
    explicit_times, ntimesout = _split_timesout(timesout)
    return timefreq(
        data,
        srate,
        frames=int(frames),
        cycles=cycles,
        winsize=None if winsize is None else int(_first_numeric(winsize, 0)),
        tlimits=tlimits,
        freqs=freqs,
        nfreqs=nfreqs,
        freqscale=str(freqscale or "linear"),
        timesout=explicit_times,
        ntimesout=ntimesout,
        padratio=int(_first_numeric(padratio, 2)),
        itctype=itctype,
        subitc=subitc,
        detrend=detrend,
        causal=causal,
        timestretch=timestretch,
        wletmethod=wletmethod,
        verbose=verbose,
    )


def _timewarp_options(
    timewarp: Any,
    timewarpms: Any,
    timewarpidx: Any,
    frames: int,
    tlimits: Any,
    srate: float,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, np.ndarray | None]:
    if _is_empty_value(timewarp):
        return None, None
    markers_ms = np.asarray(timewarp, dtype=float)
    if markers_ms.ndim == 1:
        markers_ms = markers_ms.reshape(1, -1)
    if markers_ms.ndim != 2:
        raise ValueError("timewarp must be a (trials, epoch_events) matrix in milliseconds")
    if markers_ms.shape[1] == 0:
        return None, None
    limits = _tlimits_vector(tlimits)
    frame_count = int(frames)
    marker_frames = np.rint((markers_ms - limits[0]) / 1000.0 * float(srate)).astype(int) + 1
    _validate_timewarp_frames(marker_frames, frame_count, "Time warping events")

    reference_ms = _numeric_vector(timewarpms)
    reference_frames = np.asarray([], dtype=float)
    if reference_ms.size:
        if reference_ms.size != markers_ms.shape[1]:
            raise ValueError("timewarpms must have one latency per timewarp event column")
        reference_frames = np.rint((reference_ms - limits[0]) / 1000.0 * float(srate)).astype(int) + 1
        _validate_timewarp_frames(reference_frames, frame_count, "Time warping reference latencies")
    reference_for_markers = reference_frames if reference_frames.size else np.median(marker_frames, axis=0)
    plot_indices = _timewarp_plot_indices(timewarpidx, markers_ms.shape[1])
    markers = ((reference_for_markers[plot_indices] - 1.0) / float(srate) + limits[0] / 1000.0) * 1000.0
    _validate_vertical_markers(markers, limits)
    return (marker_frames.astype(float), reference_frames.astype(float)), markers


def _validate_timewarp_frames(values: np.ndarray, frames: int, label: str) -> None:
    if np.max(values) > frames - 2 or np.min(values) < 3:
        raise ValueError(f"{label} must be inside the epochs")


def _timewarp_plot_indices(timewarpidx: Any, event_count: int) -> np.ndarray:
    values = _numeric_vector(timewarpidx, dtype=int)
    if values.size == 0:
        return np.arange(event_count, dtype=int)
    if np.any(values < 1) or np.any(values > event_count):
        raise ValueError(f"timewarpidx values must be 1-based and within 1..{event_count}")
    return values.astype(int) - 1


def _tlimits_vector(tlimits: Any) -> np.ndarray:
    limits = _numeric_vector(tlimits)
    if limits.size != 2:
        raise ValueError("tlimits must contain [min max] in milliseconds")
    return limits.astype(float)


def _validate_vertical_markers(markers: np.ndarray, tlimits: Any) -> None:
    limits = _tlimits_vector(tlimits)
    if np.min(markers) < limits[0] or np.max(markers) > limits[1]:
        raise ValueError("vertical line ('vert') latency outside of epoch boundaries")


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip() in {"[]", "{}"}
    return np.asarray(value).size == 0


def _split_timesout(timesout: Any) -> tuple[Any, Any]:
    values = _numeric_vector(timesout)
    if values.size == 0:
        return None, 200
    if values.size == 1:
        return None, int(values[0])
    return values, None


def _normalize_fft_tfdata(tfdata: np.ndarray, cycles: np.ndarray, winsize: int) -> np.ndarray:
    if cycles[0] == 0:
        return 2.0 / 0.375 * tfdata / float(winsize)
    return tfdata


def _power_to_output(power: np.ndarray, scale: str) -> np.ndarray:
    if scale == "log":
        return 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))
    return power


def _alpha_value(alpha: Any) -> float | None:
    values = _numeric_vector(alpha)
    if values.size == 0 or values[0] == 0 or np.isnan(values[0]):
        return None
    if values[0] <= 0 or values[0] > 0.5:
        raise ValueError("alpha must be in the interval (0, 0.5]")
    return float(values[0])


def _first_not_none(first: Any, second: Any) -> Any:
    return first if first is not None else second


def _boot_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    values = np.asarray(value, dtype=float)
    if values.size == 0 or np.isnan(values.reshape(-1)[0]):
        return None
    return values.squeeze()


def _bootstrap_indices(times: np.ndarray, baseline: Any, baseboot: Any, baseln: np.ndarray) -> np.ndarray:
    values = _numeric_vector(baseboot)
    if values.size == 0:
        return baseln
    if values.size == 1:
        if values[0] == 0:
            return np.asarray([], dtype=int)
        baseline_values = _numeric_vector(baseline)
        if baseline_values.size and not np.isnan(baseline_values[0]):
            return baseln
        indices = np.nonzero(times <= 0)[0]
        return indices if indices.size else np.arange(times.size, dtype=int)
    return baseline_indices(times, values)


def _bootstrap_power(
    power: np.ndarray,
    scale: str,
    *,
    alpha: float,
    naccu: int,
    boottype: str,
    base_indices: np.ndarray,
    rng: Any,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(rng)
    surrogates = np.empty((int(naccu), power.shape[0], power.shape[1]), dtype=float)
    boot_source = power[:, base_indices, :] if base_indices.size else power
    threshold_source = np.empty((int(naccu), power.shape[0], max(1, boot_source.shape[1])), dtype=float)
    for index in range(int(naccu)):
        sample = _resample_trials(power, generator, boottype)
        surrogates[index] = _power_to_output(np.nanmean(sample, axis=2), scale)
        threshold_sample = _resample_trials(boot_source, generator, boottype)
        threshold_source[index] = _power_to_output(np.nanmean(threshold_sample, axis=2), scale)
    thresholds = _thresholds_by_frequency(threshold_source, alpha=alpha, both=True)
    return thresholds, surrogates


def _bootstrap_itc(
    tfdata: np.ndarray,
    itctype: str,
    *,
    alpha: float,
    naccu: int,
    boottype: str,
    base_indices: np.ndarray,
    rng: Any,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(rng)
    surrogates = np.empty((int(naccu), tfdata.shape[0], tfdata.shape[1]), dtype=float)
    boot_source = tfdata[:, base_indices, :] if base_indices.size else tfdata
    threshold_source = np.empty((int(naccu), tfdata.shape[0], max(1, boot_source.shape[1])), dtype=float)
    for index in range(int(naccu)):
        sample = _resample_trials(tfdata, generator, boottype, complex_phase=True)
        surrogates[index] = np.abs(newtimefitc(sample, itctype))
        threshold_sample = _resample_trials(boot_source, generator, boottype, complex_phase=True)
        threshold_source[index] = np.abs(newtimefitc(threshold_sample, itctype))
    thresholds = _thresholds_by_frequency(threshold_source, alpha=alpha, both=False)
    return thresholds, surrogates


def _resample_trials(
    values: np.ndarray,
    generator: np.random.Generator,
    boottype: str,
    *,
    complex_phase: bool = False,
) -> np.ndarray:
    mode = str(boottype).lower()
    sample = np.asarray(values).copy()
    if mode in {"shuffle", "shufftrials"}:
        trial_indices = generator.integers(0, sample.shape[2], size=sample.shape[2])
        return sample[:, :, trial_indices]
    if mode in {"rand", "randall"}:
        if complex_phase or np.iscomplexobj(sample):
            return sample * np.exp(1j * generator.uniform(0.0, 2.0 * np.pi, size=sample.shape))
        signs = generator.choice(np.asarray([-1.0, 1.0]), size=sample.shape)
        return sample * signs
    raise ValueError("boottype must be 'shuffle', 'shufftrials', 'rand', or 'randall'")


def _thresholds_by_frequency(values: np.ndarray, *, alpha: float, both: bool) -> np.ndarray:
    nfreq = values.shape[1]
    pooled = values.transpose(0, 2, 1).reshape(-1, nfreq)
    bootside = "both" if both else "upper"
    thresholds = np.asarray(bootstrap_threshold(pooled, alpha=alpha, bootside=bootside))
    return thresholds.reshape(nfreq, 2) if both else thresholds.reshape(nfreq)


def _significance_mask(pvalues: np.ndarray, alpha: float, correction: str) -> np.ndarray:
    mode = str(correction).lower()
    if mode == "fdr":
        threshold = float(fdr(pvalues, alpha).threshold)
        if threshold == 0:
            return np.zeros_like(pvalues, dtype=bool)
        return pvalues <= threshold
    return pvalues <= alpha


def _threshold_mask(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    threshold_values = np.asarray(thresholds)
    if threshold_values.ndim == 1:
        threshold_values = np.stack([-threshold_values, threshold_values], axis=1)
    if threshold_values.shape[-1] != 2:
        raise ValueError("ERSP bootstrap thresholds must contain lower and upper limits")
    lower = threshold_values[:, 0][:, np.newaxis]
    upper = threshold_values[:, 1][:, np.newaxis]
    return (values <= lower) | (values >= upper)


def _threshold_vector(thresholds: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(thresholds, dtype=float).squeeze()
    if values.ndim == 0:
        return np.full(target_shape, float(values))
    if values.ndim == 1:
        return values[:, np.newaxis]
    return values


def _plot_time_frequency(
    ersp: np.ndarray,
    itc: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    title: str,
    plotersp: bool,
    plotitc: bool,
    plottype: str,
    ersp_significant: np.ndarray | None,
    itc_significant: np.ndarray | None,
    vertical_markers: np.ndarray | None,
):
    panels = int(plotersp) + int(plotitc)
    if panels == 0:
        return None
    fig, axes = plt.subplots(panels, 1, figsize=(7.5, 5.0), squeeze=False)
    row = 0
    if plotersp:
        _plot_panel(
            axes[row, 0],
            fig,
            ersp,
            times,
            freqs,
            title=title,
            label="ERSP",
            plottype=plottype,
            significant=ersp_significant,
            vertical_markers=vertical_markers,
        )
        row += 1
    if plotitc:
        _plot_panel(
            axes[row, 0],
            fig,
            itc,
            times,
            freqs,
            title="" if plotersp else title,
            label="ITC",
            plottype=plottype,
            significant=itc_significant,
            vmin=0.0,
            vmax=max(1.0, float(np.nanmax(itc))),
            vertical_markers=vertical_markers,
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
    vertical_markers: np.ndarray | None,
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
    if vertical_markers is not None:
        for marker in np.asarray(vertical_markers, dtype=float).ravel():
            axis.axvline(float(marker), color="m", linewidth=1.0)
    axis.set_ylabel("Frequency (Hz)")


def _numeric_vector(value: Any, *, dtype: Any = float) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    if isinstance(value, str) and value.strip() == "":
        return np.asarray([], dtype=dtype)
    return np.asarray(parse_numeric_sequence(value, dtype=dtype), dtype=dtype).ravel()


def _first_numeric(value: Any, default: float) -> float:
    values = _numeric_vector(value)
    return float(values[0]) if values.size else float(default)


__all__ = ["TimeFrequencyResult", "compute_time_frequency", "newtimef"]
