"""EEGLAB-style ``newtimef`` numerical core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.miscfunc.value_parsing import is_empty_value as _is_empty_value
from eegprep.functions.miscfunc.value_parsing import is_on as _is_on
from eegprep.functions.miscfunc.value_parsing import parse_numeric_sequence
from eegprep.functions.statistics.fdr import fdr
from eegprep.functions.timefreqfunc._bootstrap import (
    bootstrap_indices as shared_bootstrap_indices,
    resample_trials,
    threshold_vector as _threshold_vector,
    thresholds_by_frequency,
)
from eegprep.functions.timefreqfunc.newtimefbaseln import newtimefbaseln
from eegprep.functions.timefreqfunc.newtimefitc import newtimefitc
from eegprep.functions.timefreqfunc.newtimefpowerunit import newtimefpowerunit
from eegprep.functions.timefreqfunc.newtimeftrialbaseln import newtimeftrialbaseln
from eegprep.functions.timefreqfunc.timefreq import as_channel_epoch_data, timefreq


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
    plotphase: Any = "on",
    plotphasesign: Any = "on",
    plotphaseonly: Any = "off",
    pcontour: Any = "off",
    erspmax: Any = None,
    itcmax: Any = None,
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
    if not _is_on(plotphase):
        plotphasesign = plotphase  # EEGLAB: plotphase='off' turns off the ITC phase-sign (newtimef.m line 603)
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
            ersp_boot, ersp_null = _bootstrap_power(
                corrected_power,
                scale_mode,
                alpha=alpha_value,
                naccu=naccu,
                boottype=boottype,
                base_indices=boot_indices,
                rng=rng,
            )
            ersp_pvalues = _baseline_pvalues(ersp, ersp_null)
            ersp_significant = _significance_mask(ersp_pvalues, alpha_value, mcorrect)
        else:
            ersp_significant = _threshold_mask(ersp, ersp_boot)
        if itc_boot is None:
            itc_boot, itc_null = _bootstrap_itc(
                tfdata,
                itctype,
                alpha=alpha_value,
                naccu=naccu,
                boottype=boottype,
                base_indices=boot_indices,
                rng=rng,
            )
            itc_pvalues = _baseline_pvalues(np.abs(itc), itc_null)
            itc_significant = _significance_mask(itc_pvalues, alpha_value, mcorrect)
        else:
            itc_significant = np.abs(itc) >= _threshold_vector(itc_boot, itc.shape)

    figure = None
    if _is_on(plot):
        unit = newtimefpowerunit({"scale": scale_mode, "baseline": baseline, "basenorm": normalize_baseline})
        ersp_baseval = 1.0 if scale_mode == "abs" and normalize_baseline == "off" else 0.0
        limits = _numeric_vector(tlimits)
        epoch_data = as_channel_epoch_data(data, frames=int(frames))[0]
        erp_full = np.nanmean(epoch_data, axis=1)
        full_times = np.linspace(float(limits[0]), float(limits[-1]), int(frames))
        erp = erp_full[[int(np.argmin(np.abs(full_times - center))) for center in decomp.times]]
        spectrum = np.asarray(powbase_array, dtype=float).reshape(-1)
        baseline_spectrum = (
            _power_to_output(spectrum, scale_mode)
            if spectrum.size == decomp.freqs.size and np.isfinite(spectrum).any()
            else np.zeros(decomp.freqs.size)
        )
        figure = _plot_time_frequency(
            ersp,
            itc,
            decomp.times,
            decomp.freqs,
            title=str(title),
            plotersp=_is_on(plotersp),
            plotitc=_is_on(plotitc),
            plottype=str(plottype).lower(),
            ersp_significant=ersp_significant,
            itc_significant=itc_significant,
            vertical_markers=vertical_markers,
            erspmax=erspmax,
            itcmax=itcmax,
            unit=unit,
            ersp_baseval=ersp_baseval,
            powbase=baseline_spectrum,
            erp=erp,
            ersp_boot=ersp_boot,
            itc_boot=itc_boot,
            plotphasesign=_is_on(plotphasesign),
            plotphaseonly=_is_on(plotphaseonly),
            pcontour=_is_on(pcontour),
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
    return shared_bootstrap_indices(times, baseline=baseline, baseboot=baseboot, baseln=baseln)


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
    # EEGLAB draws the surrogate distribution from the baseline (bootstat with
    # 'basevect'); the same baseline null feeds both the threshold and the
    # per-frequency p-values (newtimef.m lines 1282-1286, 1369).
    generator = np.random.default_rng(rng)
    boot_source = power[:, base_indices, :] if base_indices.size else power
    baseline_null = np.empty((int(naccu), power.shape[0], max(1, boot_source.shape[1])), dtype=float)
    for index in range(int(naccu)):
        sample = resample_trials(boot_source, generator, boottype)
        baseline_null[index] = _power_to_output(np.nanmean(sample, axis=2), scale)
    thresholds = _thresholds_by_frequency(baseline_null, alpha=alpha, both=True)
    return thresholds, baseline_null


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
    # As for power, the ITC null comes from the baseline single-trial estimates
    # (newtimef.m lines 1336-1347, 1382); it feeds the threshold and p-values.
    generator = np.random.default_rng(rng)
    boot_source = tfdata[:, base_indices, :] if base_indices.size else tfdata
    baseline_null = np.empty((int(naccu), tfdata.shape[0], max(1, boot_source.shape[1])), dtype=float)
    for index in range(int(naccu)):
        sample = resample_trials(boot_source, generator, boottype, complex_phase=True)
        baseline_null[index] = np.abs(newtimefitc(sample, itctype))
    thresholds = _thresholds_by_frequency(baseline_null, alpha=alpha, both=False)
    return thresholds, baseline_null


def _thresholds_by_frequency(values: np.ndarray, *, alpha: float, both: bool) -> np.ndarray:
    return thresholds_by_frequency(values, alpha=alpha, bootside="both" if both else "upper")


def _baseline_pvalues(observed: np.ndarray, baseline_null: np.ndarray) -> np.ndarray:
    """Two-sided p-values of each cell against the per-frequency baseline null.

    Mirrors EEGLAB ``compute_pvals``: one null distribution per frequency (drawn
    from the baseline and shared across all output times), against which every
    time-frequency value is ranked by its distance from the surrogate mean.
    """
    observed_values = np.asarray(observed, dtype=float)
    null = np.moveaxis(np.asarray(baseline_null, dtype=float), 1, 0).reshape(observed_values.shape[0], -1)
    center = np.nanmean(null, axis=1, keepdims=True)
    null_distance = np.sort(np.abs(null - center), axis=1)
    distance = np.abs(observed_values - center)
    sample_count = null_distance.shape[1]
    pvalues = np.empty_like(observed_values)
    for freq_index in range(observed_values.shape[0]):
        below = np.searchsorted(null_distance[freq_index], distance[freq_index], side="left")
        pvalues[freq_index] = 1.0 - below / sample_count
    return pvalues


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


_BASE_POS = (0.13, 0.11, 0.775, 0.815)  # EEGLAB's default axes rectangle; panels are placed relative to it
_IMAGE_COLORMAP = "jet"
_MARGIN_SIZE = 0.1  # thickness of the marginal panels below and left of each image (EEGLAB plottimef)


def _axes_rect(nx: float, ny: float, nw: float, nh: float) -> list[float]:
    """Map an EEGLAB ``plottimef`` normalized position into figure coordinates."""
    x0, y0, w0, h0 = _BASE_POS
    return [x0 + nx * w0, y0 + ny * h0, nw * w0, nh * h0]


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
    erspmax: Any = None,
    itcmax: Any = None,
    unit: str = "dB",
    ersp_baseval: float = 0.0,
    powbase: Any = None,
    erp: Any = None,
    ersp_boot: Any = None,
    itc_boot: Any = None,
    plotphasesign: bool = True,
    plotphaseonly: bool = False,
    pcontour: bool = False,
):
    panels = int(plotersp) + int(plotitc)
    if panels == 0:
        return None
    if plottype == "curve":
        return _plot_curve_figure(
            ersp,
            np.abs(np.asarray(itc)),
            times,
            freqs,
            title=title,
            plotersp=plotersp,
            plotitc=plotitc,
            ersp_significant=ersp_significant,
            itc_significant=itc_significant,
            vertical_markers=vertical_markers,
        )
    if plottype != "image":
        raise ValueError("plottype must be 'image' or 'curve'")

    fig = plt.figure(figsize=(7.2, 6.2))
    if plotersp and plotitc:
        ersp_ordinate, itc_ordinate, height = 0.67, 0.1, 0.33
    else:
        ersp_ordinate = itc_ordinate = 0.1
        height = 0.9
    if plotersp:
        vmin, vmax = _ersp_color_axis(ersp, erspmax, ersp_baseval)
        _draw_image_panel(
            fig,
            ersp,
            times,
            freqs,
            ordinate=ersp_ordinate,
            height=height,
            vmin=vmin,
            vmax=vmax,
            significant=ersp_significant,
            baseval=ersp_baseval,
            colorbar_title=f"ERSP({unit})",
            vertical_markers=vertical_markers,
            pcontour=pcontour,
        )
        extremes = np.stack([np.nanmin(ersp, axis=0), np.nanmax(ersp, axis=0)])
        _draw_time_marginal(
            fig,
            ordinate=ersp_ordinate - _MARGIN_SIZE,
            times=times,
            series=extremes,
            ylabel=unit,
            vertical_markers=vertical_markers,
        )
        _draw_freq_marginal(
            fig,
            ordinate=ersp_ordinate,
            height=height,
            freqs=freqs,
            curve=powbase,
            overlays=_spectrum_overlays(powbase, ersp_boot, freqs.size),
            value_label=unit,
        )
    if plotitc:
        itc_magnitude = np.abs(np.asarray(itc))
        phase_only = plotphaseonly or (itc_magnitude.size > 0 and abs(float(itc_magnitude.flat[0]) - 1.0) < 1e-4)
        if phase_only:
            itc_display = np.angle(np.asarray(itc)) / np.pi * 180.0  # phase in degrees
            itc_vmin, itc_vmax, itc_title, itc_clip = -180.0, 180.0, "ITC phase", False
        else:
            # phase-sign colors the coherence magnitude by the sign of its imaginary part
            itc_display = np.sign(np.imag(np.asarray(itc))) * itc_magnitude if plotphasesign else itc_magnitude
            itc_vmin, itc_vmax = _itc_color_axis(itc_magnitude, itcmax)
            itc_title, itc_clip = "ITC", True  # magnitude and phase-sign share a [0, max] colorbar
        _draw_image_panel(
            fig,
            itc_display,
            times,
            freqs,
            ordinate=itc_ordinate,
            height=height,
            vmin=itc_vmin,
            vmax=itc_vmax,
            significant=itc_significant,
            baseval=0.0,
            colorbar_title=itc_title,
            vertical_markers=vertical_markers,
            colorbar_positive_only=itc_clip,
            pcontour=pcontour,
        )
        if erp is not None:
            _draw_time_marginal(
                fig,
                ordinate=itc_ordinate - _MARGIN_SIZE,
                times=times,
                series=np.asarray(erp, dtype=float).reshape(1, -1),
                ylabel="µV",
                vertical_markers=vertical_markers,
                zero_line=True,
            )
        itc_overlays = [np.asarray(itc_boot, dtype=float).reshape(-1)] if itc_boot is not None else []
        _draw_freq_marginal(
            fig,
            ordinate=itc_ordinate,
            height=height,
            freqs=freqs,
            curve=np.nanmean(itc_magnitude, axis=1),
            overlays=itc_overlays,
            value_label="ERP",  # EEGLAB labels the marginal-ITC value axis 'ERP'
        )
    if title:
        x0, y0, _w0, h0 = _BASE_POS
        fig.text(x0 - 0.039, y0 + 1.01 * h0, str(title), ha="left", va="bottom", fontsize=10, fontweight="bold")
    return fig


def _ersp_color_axis(ersp: np.ndarray, erspmax: Any, baseval: float) -> tuple[float, float]:
    """EEGLAB ERSP color limits: user ``erspmax`` or an auto symmetric scale."""
    vmin, vmax = _color_limits(erspmax)
    if vmax is not None:
        return vmin, vmax
    peak = float(np.nanmax(np.abs(ersp))) if np.size(ersp) else 0.0
    if baseval == 1.0:  # abs power as % of baseline: EEGLAB centers the scale on 1
        return (2.0 - peak, peak) if peak > 1.0 else (peak, 2.0 - peak)
    half = peak / 2.0 if peak else 1.0
    return -half, half


def _itc_color_axis(itc: np.ndarray, itcmax: Any) -> tuple[float, float]:
    """EEGLAB ITC color limits: symmetric about zero, capped at 1 when auto."""
    vmin, vmax = _color_limits(itcmax)
    if vmax is not None:
        return vmin, vmax
    peak = min(float(np.nanmax(np.abs(itc))), 1.0) if np.size(itc) else 1.0
    peak = peak or 1.0
    return -peak, peak


def _draw_image_panel(
    fig,
    values: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    ordinate: float,
    height: float,
    vmin: float,
    vmax: float,
    significant: np.ndarray | None,
    baseval: float,
    colorbar_title: str,
    vertical_markers: np.ndarray | None,
    colorbar_positive_only: bool = False,
    pcontour: bool = False,
):
    axis = fig.add_axes(_axes_rect(0.1, ordinate, 0.8, height))
    # Non-significant cells collapse to the baseline value (the symmetric jet scale
    # renders it green); with pcontour they stay visible and significance is drawn
    # as a contour outline instead (EEGLAB masks to baseval, not NaN).
    array = values if significant is None or pcontour else np.where(significant, values, baseval)
    image = axis.imshow(
        array,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        interpolation="nearest",
        cmap=_IMAGE_COLORMAP,
        vmin=vmin,
        vmax=vmax,
    )
    if significant is not None and pcontour:
        axis.contour(times, freqs, np.asarray(significant, dtype=float), levels=[0.5], colors="k", linewidths=0.25)
    axis.axvline(0.0, color="m", linestyle="--", linewidth=1.0)  # stimulus onset
    if vertical_markers is not None:
        for marker in np.asarray(vertical_markers, dtype=float).ravel():
            axis.axvline(float(marker), color="m", linewidth=1.0)
    axis.set_xlim(times[0], times[-1])  # keep the image span; the time-0 line is clipped if outside
    # EEGLAB strips the image axes; the marginal panels carry the time/frequency labels.
    axis.set_xticks([])
    axis.set_yticks([])
    colorbar_axis = fig.add_axes(_axes_rect(0.95, ordinate, 0.05, height))
    fig.colorbar(image, cax=colorbar_axis)
    if colorbar_positive_only:
        colorbar_axis.set_ylim(0.0, vmax)
    colorbar_axis.set_title(colorbar_title, fontsize=9)
    return axis


def _draw_time_marginal(
    fig,
    *,
    ordinate: float,
    times: np.ndarray,
    series: np.ndarray,
    ylabel: str,
    vertical_markers: np.ndarray | None,
    zero_line: bool = False,
):
    """Draw curves below an image sharing its time axis (ERSP min/max, or the ERP)."""
    axis = fig.add_axes(_axes_rect(0.1, ordinate, 0.8, _MARGIN_SIZE))
    for row in np.atleast_2d(np.asarray(series, dtype=float)):
        axis.plot(times, row, linewidth=1.0)
    if zero_line:
        axis.plot([times[0], times[-1]], [0.0, 0.0], color="k", linewidth=0.8)
    axis.axvline(0.0, color="m", linestyle="--", linewidth=1.0)
    if vertical_markers is not None:
        for marker in np.asarray(vertical_markers, dtype=float).ravel():
            axis.axvline(float(marker), color="m", linewidth=1.0)
    axis.set_xlim(times[0], times[-1])
    axis.set_xlabel("Time (ms)")
    axis.set_ylabel(ylabel)
    axis.yaxis.set_label_position("right")
    axis.yaxis.tick_right()
    return axis


def _draw_freq_marginal(
    fig,
    *,
    ordinate: float,
    height: float,
    freqs: np.ndarray,
    curve: Any,
    overlays: list[np.ndarray],
    value_label: str,
):
    """Draw a rotated marginal (value vs frequency) to the left of an image."""
    axis = fig.add_axes(_axes_rect(0.0, ordinate, _MARGIN_SIZE, height))
    values = _numeric_vector(curve)
    if values.size == freqs.size and np.isfinite(values).any():
        axis.plot(values, freqs, color="C0", linewidth=1.0)
    for overlay in overlays:
        overlay_values = np.asarray(overlay, dtype=float)
        if overlay_values.size == freqs.size:
            axis.plot(overlay_values, freqs, color="g", linewidth=1.0)
            axis.plot(overlay_values, freqs, color="k", linestyle=":", linewidth=1.0)
    if freqs[0] != freqs[-1]:
        axis.set_ylim(freqs[0], freqs[-1])
    axis.set_ylabel("Frequency (Hz)")
    axis.set_xlabel(value_label)
    return axis


def _spectrum_overlays(spectrum: Any, boot: Any, nfreq: int) -> list[np.ndarray]:
    """Baseline-spectrum significance envelope: ``mbase + [lower, upper]`` thresholds."""
    if spectrum is None or boot is None:
        return []
    values = _numeric_vector(spectrum)
    boot_values = np.asarray(boot, dtype=float)
    if values.size != nfreq or boot_values.ndim != 2 or boot_values.shape[0] != nfreq:
        return []
    return [values + boot_values[:, 0], values + boot_values[:, 1]]


def _plot_curve_figure(
    ersp: np.ndarray,
    itc: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    title: str,
    plotersp: bool,
    plotitc: bool,
    ersp_significant: np.ndarray | None,
    itc_significant: np.ndarray | None,
    vertical_markers: np.ndarray | None,
):
    panels = int(plotersp) + int(plotitc)
    fig, axes = plt.subplots(panels, 1, figsize=(7.5, 5.0), squeeze=False)
    row = 0
    if plotersp:
        _plot_curve_panel(
            axes[row, 0],
            ersp,
            times,
            freqs,
            title=title,
            significant=ersp_significant,
            vertical_markers=vertical_markers,
        )
        row += 1
    if plotitc:
        _plot_curve_panel(
            axes[row, 0],
            itc,
            times,
            freqs,
            title="" if plotersp else title,
            significant=itc_significant,
            vertical_markers=vertical_markers,
        )
    axes[panels - 1, 0].set_xlabel("Time (ms)")
    fig.tight_layout()
    return fig


def _plot_curve_panel(
    axis,
    values: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    title: str,
    significant: np.ndarray | None,
    vertical_markers: np.ndarray | None,
) -> None:
    for freq_index, freq in enumerate(freqs):
        line_values = values[freq_index]
        if significant is not None:
            line_values = np.where(significant[freq_index], line_values, np.nan)
        axis.plot(times, line_values, label=f"{freq:g} Hz")
    if freqs.size <= 12:
        axis.legend(loc="best", fontsize="small")
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


def _color_limits(value: Any) -> tuple[float | None, float | None]:
    """Return symmetric ``(-m, m)`` limits from a scalar ``m``, or a ``[min, max]`` pair as is.

    Empty or zero input returns ``(None, None)`` so the caller keeps auto limits.
    """
    values = _numeric_vector(value)
    if values.size >= 2:
        return float(values[0]), float(values[1])
    if values.size == 0 or values[0] == 0:
        return None, None
    magnitude = abs(float(values[0]))
    return -magnitude, magnitude


__all__ = ["TimeFrequencyResult", "compute_time_frequency", "newtimef"]
