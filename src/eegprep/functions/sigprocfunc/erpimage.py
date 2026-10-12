"""EEGLAB-style ERP image: sort, smooth and image single trials, with ERP and spectral traces.

The numbers come from :mod:`eegprep.functions.sigprocfunc.erpimage_engine`
(NumPy only); this module parses EEGLAB arguments, applies the optional
Butterworth ``'filt'`` (SciPy) and draws the figure with Matplotlib.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from eegprep.functions.sigprocfunc.erpimage_engine import (
    ErpimageComputation,
    ErpimageOptions,
    compute_erpimage,
    find_corresponding_point,
    order_of_magnitude,
    parse_erpimage_options,
    welch_psd,
)
from eegprep.functions.sigprocfunc.topoplot import plot_channel_location, topoplot

# Line widths, fonts and colors from erpimage.m and icadefs.m.
SORTWIDTH = 2.5
ZEROWIDTH = 3.0
VERTWIDTH = 2.5
HORZWIDTH = 2.1
ERPDATAWIDTH = 2.0
ERPZEROWIDTH = 2.0
LABELFONT = 10
TICKFONT = 10
DOTSTYLE = "k--"
BACKCOLOR = (0.93, 0.96, 1.0)
FILLCOLOR = (0.66, 0.76, 1.0)
WINFILLCOLOR = (0.88, 0.92, 1.0)
MATLAB_DEFAULT_COLOR = "#0072BD"
ERP_TRACE_COLORS = ((0.0, 0.0, 1.0), (0.25, 0.0, 0.75), (0.75, 0.0, 0.25), (1.0, 0.0, 0.0))
ERP_LEGENDS = {
    2: ("Lower 50%", "Higher 50%"),
    3: ("Lowest 33%", "Middle 33%", "Highest 33%"),
    4: ("Lowest 25%", "2nd Lowest 25%", "3rd Lowest 25%", "Highest 25%"),
}
SORTED_TRIAL_LABELS = {
    "phase": "Phase-sorted Trials",
    "amp": "Amplitude-sorted Trials",
    "value": "Voltage-sorted Trials",
}
BUTTERWORTH_ORDER = 3
LABEL_OFFSET_FRACTION = 0.03  # value labels left of the trace panels, as a fraction of the time span
TITLE_OFFSET_FRACTION = 0.08  # rotated ERSP/ITC labels further left


class ErpimageResult(NamedTuple):
    """EEGLAB ``erpimage`` outputs in their MATLAB order, then the figure.

    Unpacks like MATLAB's ``[outdata, outvar, ...] = erpimage(...)``. Arrays
    follow :class:`~eegprep.functions.sigprocfunc.erpimage_engine.ErpimageComputation`:
    ``sortidx`` holds 0-based indices into the input trials and absent outputs
    are ``None``. ``axhndls`` is (image, colorbar, ERP, amplitude, coherence)
    axes, with ``None`` for panels not drawn, followed by the scalp-map and
    spectrum insets when present.
    """

    outdata: np.ndarray
    outvar: Any
    outtrials: np.ndarray
    limits: np.ndarray
    axhndls: tuple[Any, ...]
    erp: np.ndarray | None
    amps: np.ndarray | None
    cohers: np.ndarray | None
    cohsig: float | None
    ampsig: np.ndarray | None
    outamps: np.ndarray | None
    phsangls: np.ndarray | None
    phsamp: np.ndarray | None
    sortidx: np.ndarray
    erpsig: np.ndarray | None
    figure: Any


def erpimage(
    data: Any,
    sortvar: Any = None,
    times: Any = None,
    title: str = "",
    avewidth: Any = 1,
    decimate: Any = 0,
    *options: Any,
    target: Any = None,
    title_fontsize: float | None = None,
    rng: Any = None,
    **kwargs: Any,
) -> ErpimageResult:
    """Image single trials sorted and smoothed across trials, as EEGLAB ``erpimage``.

    The positional arguments follow EEGLAB: ``data`` is (frames, trials) for
    one channel (or a (1, frames*trials) concatenation), ``sortvar`` holds one
    value per trial (trials with NaN are dropped), ``times`` is latencies in ms
    or ``[start_ms, frames, srate]``, ``avewidth`` is the moving-average width
    in trials and ``decimate`` the window advance. EEGLAB options follow either
    as ``'key', value`` tokens (``erpimage(data, rt, times, '', 5, 1, 'erp',
    'cbar')``) or as keywords (``erp=True, cbar=True``). Supported options:
    ``align``, ``timewarp``, ``renorm``, ``noplot``, ``NoShow``, ``nosort``,
    ``replace_ties``, ``valsort``, ``phasesort``, ``ampsort``, ``sortwin``,
    ``showwin``, ``plotamps``, ``limits``, ``sortvar_limits``, ``signif``,
    ``caxis``, ``erp`` (1 to 4 percentile ERPs), ``erpalpha``, ``erpstd``,
    ``erp_grid``, ``erp_vltg_ticks``, ``rmerp``, ``coher``, ``srate``,
    ``cycles``, ``cbar``, ``cbar_title``, ``topo``, ``spec``, ``specaxis``,
    ``horz``, ``vert``, ``auxvar``, ``sortvarpercent``, ``noxlabel``,
    ``yerplabel``, ``avg_type``, ``img_trialax_label``, ``img_trialax_ticks``,
    ``baseline``, ``baselinedb``, ``filt`` and ``plotmode``.

    Numbers match EEGLAB. Where EEGLAB's behavior contradicts its own help,
    EEGPrep follows the help: ``sortidx`` indexes the input trials, ``noplot``
    hides only the sortvar trace, and ``erpalpha`` draws unbiased random signs
    for its permutation test (EEGLAB's ``sign(randn-0.5)`` is +1 only 31% of
    the time, https://github.com/sccn/eeglab/issues/976), so ``erpalpha``
    thresholds differ slightly from MATLAB. Non-integer boxcar widths raise
    ``ValueError``, as MATLAB R2025b does.

    Args:
        target: Matplotlib Figure or SubFigure to draw into instead of a new
            figure window.
        title_fontsize: font size of the image title.
        rng: ``numpy.random.Generator`` or ``RandomState`` for the coherence
            bootstrap and the ``erpalpha`` permutations.

    Returns:
        :class:`ErpimageResult` with the 15 EEGLAB outputs and the figure
        (``None`` with ``NoShow``).
    """
    opts = parse_erpimage_options(options, **kwargs)
    if opts.title is not None:
        title = opts.title
    result = compute_erpimage(
        data,
        sortvar,
        times,
        avewidth,
        decimate,
        opts,
        rng=rng,
        plotted=not opts.noshow,
        trial_filter=butterworth_filter,
    )
    if opts.noshow:
        return _result(result, (None, None, None, None, None), None)
    figure, axes = _draw(result, opts, str(title or ""), target, title_fontsize)
    return _result(result, axes, figure)


def butterworth_filter(data: np.ndarray, filt: np.ndarray, srate: float) -> np.ndarray:
    """EEGLAB ``'filt'``: third-order Butterworth, zero phase (``filtfilt``), per trial.

    ``[0 hi]`` is a low-pass, ``[lo srate/2]`` a high-pass, two negative edges
    a band-stop and anything else a band-pass. Padding matches MATLAB's
    ``filtfilt`` (odd extension of three filter orders).
    """
    if filt[0] == 0:
        b, a = signal.butter(BUTTERWORTH_ORDER, filt[1] * 2 / srate, "low")
    elif filt[1] == srate / 2:
        b, a = signal.butter(BUTTERWORTH_ORDER, filt[0] * 2 / srate, "high")
    elif filt[0] < 0:
        b, a = signal.butter(BUTTERWORTH_ORDER, -filt * 2 / srate, "bandstop")
    else:
        b, a = signal.butter(BUTTERWORTH_ORDER, filt * 2 / srate, "bandpass")
    return signal.filtfilt(b, a, data, axis=0, padtype="odd", padlen=3 * (max(len(a), len(b)) - 1))


def _result(result: ErpimageComputation, axes: tuple[Any, ...], figure: Any) -> ErpimageResult:
    return ErpimageResult(
        result.outdata,
        result.outvar,
        result.outtrials,
        result.limits,
        axes,
        result.erp,
        result.amps,
        result.cohers,
        result.cohsig,
        result.ampsig,
        result.outamps,
        result.phsangls,
        result.phsamp,
        result.sortidx,
        result.erpsig,
        figure,
    )


def _outsort(result: ErpimageComputation) -> np.ndarray:
    return np.asarray(result.outvar[0] if isinstance(result.outvar, tuple) else result.outvar, dtype=float)


def _draw(
    result: ErpimageComputation, opts: ErpimageOptions, title: str, target: Any, title_fontsize: float | None
) -> tuple[Any, tuple[Any, ...]]:
    has_erp = result.erp is not None
    has_coher = result.amps is not None and result.cohers is not None
    insets = opts.topo is not None or opts.spec is not None
    height_ratios: list[float] = [1.0] if insets else []
    height_ratios.append(3.0)
    if has_erp:
        height_ratios.append(1.0)
    if has_coher:
        height_ratios.extend([1.0, 1.0])
    figure = target if target is not None else plt.figure(figsize=(7.5, 1.2 * sum(height_ratios) + 0.6))
    # A narrow colorbar column only when a colorbar is drawn; image and traces stay flush like EEGLAB.
    grid = figure.add_gridspec(
        nrows=len(height_ratios),
        ncols=2 if opts.cbar else 1,
        width_ratios=[20, 1] if opts.cbar else [1],
        height_ratios=height_ratios,
        hspace=0.15 if insets else 0.0,
        wspace=0.04,
    )
    row = 1 if insets else 0
    image_ax = figure.add_subplot(grid[row, 0])
    colorbar_ax = figure.add_subplot(grid[row, 1]) if opts.cbar else None
    image = _draw_image(image_ax, result, opts, title, title_fontsize, has_erp)
    if colorbar_ax is not None:
        _draw_colorbar(figure, image, colorbar_ax, opts.cbar_title)
    erp_ax = amp_ax = coh_ax = None
    if has_erp:
        row += 1
        erp_ax = figure.add_subplot(grid[row, 0], sharex=image_ax)
        _draw_erp(erp_ax, result, opts, has_coher)
        # EEGLAB blanks the image's x-tick labels so the trace below carries the time axis.
        image_ax.tick_params(labelbottom=False)
    if has_coher:
        amp_ax = figure.add_subplot(grid[row + 1, 0], sharex=image_ax)
        coh_ax = figure.add_subplot(grid[row + 2, 0], sharex=image_ax)
        _draw_amplitude(amp_ax, result)
        _draw_coherence(coh_ax, result, opts)
    axes: tuple[Any, ...] = (image_ax, colorbar_ax, erp_ax, amp_ax, coh_ax)
    if insets:
        axes += _draw_insets(figure, grid, result, opts)
    return figure, axes


def _pixel_extent(values: np.ndarray) -> tuple[float, float]:
    step = float(values[1] - values[0]) if values.size > 1 else 1.0
    return float(values[0]) - step / 2, float(values[-1]) + step / 2


def _draw_image(
    ax: Any, result: ErpimageComputation, opts: ErpimageOptions, title: str, title_fontsize: float | None, has_erp: bool
) -> Any:
    x0, x1 = _pixel_extent(result.times)
    y0, y1 = _pixel_extent(result.outtrials)
    # imagesc centers pixels on (times, outtrials); the axis then clips to the data range.
    image = ax.imshow(
        result.outdata.T,
        aspect="auto",
        origin="lower",
        extent=(x0, x1, y0, y1),
        cmap="turbo",
        interpolation="nearest",
        vmin=result.clim[0],
        vmax=result.clim[1],
    )
    trial_low, trial_high = float(np.min(result.outtrials)), float(np.max(result.outtrials))
    if result.trial_limits is not None:
        trial_low, trial_high = result.trial_limits
    if trial_high <= trial_low:
        trial_low, trial_high = trial_low - 0.5, trial_high + 0.5
    ax.set_xlim(*result.timelimits)
    ax.set_ylim(trial_low, trial_high)
    outsort = _outsort(result)
    aligned = not np.isnan(result.aligntime)
    _draw_vertical_on_image(ax, result, outsort, aligned)
    _draw_horizontal(ax, result, opts, outsort)
    zero = result.aligntime if aligned else 0.0
    if result.times[0] <= zero <= result.times[-1]:
        ax.plot([zero, zero], [trial_low, trial_high], color="k", linewidth=ZEROWIDTH)
    clipped = np.clip(outsort, result.timelimits[0], result.timelimits[1])
    if result.show_sortvar:
        if aligned:
            ax.plot([result.aligntime] * 2, [trial_low, trial_high], color="k", linewidth=SORTWIDTH)
            ax.plot(result.aligntime - clipped, result.outtrials, color="k", linewidth=ZEROWIDTH)
        else:
            ax.plot(clipped, result.outtrials, color="k", linewidth=SORTWIDTH)
    _draw_auxvar(ax, result, aligned)
    _set_trial_axis(ax, result, opts, outsort)
    if opts.plotmode == "topo":
        x_low, x_high = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.text(x_low + 0.1 * (x_high - x_low), y_high - 0.2 * (y_high - y_low), title)
        ax.axis("off")
    else:
        ax.set_title(title, fontsize=title_fontsize if title_fontsize is not None else LABELFONT)
    ax.set_facecolor(BACKCOLOR)
    ax.tick_params(labelsize=TICKFONT)
    if not has_erp and not opts.noxlabel:
        ax.set_xlabel("Time (ms)" if result.times_passed else "Frames", fontsize=LABELFONT)
    return image


def _line_style(style: str | None) -> str:
    return style or DOTSTYLE


def _draw_vertical_on_image(ax: Any, result: ErpimageComputation, outsort: np.ndarray, aligned: bool) -> None:
    if result.verttimes is None:
        return
    if result.verttimes.ndim == 2:
        for smoothed in result.vert_smoothed:
            if smoothed.size == result.outtrials.size:
                xvalues = result.aligntime + smoothed - outsort if aligned else smoothed
                ax.plot(xvalues, result.outtrials, DOTSTYLE, linewidth=VERTWIDTH)
        return
    top = float(np.max(result.outtrials))
    styles = (
        result.auxcolors if result.auxcolors is not None and len(result.auxcolors) == result.verttimes.size else None
    )
    for index, latency in enumerate(result.verttimes):
        if aligned:
            ax.plot(result.aligntime + latency - outsort, result.outtrials, DOTSTYLE, linewidth=VERTWIDTH)
        else:
            style = _line_style(styles[index] if styles is not None else None)
            ax.plot([latency, latency], [0, top], style, linewidth=VERTWIDTH)


def _draw_horizontal(ax: Any, result: ErpimageComputation, opts: ErpimageOptions, outsort: np.ndarray) -> None:
    if opts.horz is None:
        return
    sortvar_units = bool(opts.img_trialax_label) and str(opts.img_trialax_label).lower() != "trials"
    for epoch in np.asarray(opts.horz, dtype=float).ravel():
        position: float | None = float(epoch)
        if sortvar_units:
            inside = float(np.min(outsort)) < epoch < float(np.max(outsort))
            position = find_corresponding_point(float(epoch), outsort, result.outtrials) if inside else None
        if position is not None:
            ax.plot(result.timelimits, [position, position], "-", color=MATLAB_DEFAULT_COLOR, linewidth=HORZWIDTH)


def _draw_auxvar(ax: Any, result: ErpimageComputation, aligned: bool) -> None:
    width = ZEROWIDTH if aligned else SORTWIDTH
    if result.auxvar is not None:
        given = list(result.auxcolors) if result.auxcolors else ["k"]
        colors = [given[index % len(given)] for index in range(result.auxvar.shape[0])]
        for row, color in zip(result.auxvar, colors, strict=True):
            if color:
                ax.plot(row, result.outtrials, color, linewidth=width)
    for trace in result.percentiles:
        ax.plot(result.aligntime - trace if aligned else trace, result.outtrials, color="k", linewidth=SORTWIDTH)


def _set_trial_axis(ax: Any, result: ErpimageComputation, opts: ErpimageOptions, outsort: np.ndarray) -> None:
    label = opts.img_trialax_label
    if result.sort_mode in SORTED_TRIAL_LABELS:
        ax.set_ylabel(SORTED_TRIAL_LABELS[result.sort_mode], fontsize=LABELFONT)
        return
    ax.set_ylabel(label or "", fontsize=LABELFONT)
    if not label:
        return
    if label.lower() != "trials":
        if opts.sortvar_limits is not None:
            low, high = float(opts.sortvar_limits[0]), float(opts.sortvar_limits[1])
        else:
            low, high = float(np.min(outsort)), float(np.max(outsort))
        values = _tick_values(low, high, opts.img_trialax_ticks)
        positions = [find_corresponding_point(float(value), outsort, result.outtrials) for value in values]
        keep = [index for index, position in enumerate(positions) if position is not None]
        ax.set_yticks([positions[index] for index in keep], [f"{values[index]:g}" for index in keep])
        return
    if result.sort_mode == "nosort":
        return
    if opts.img_trialax_ticks is not None:
        ax.set_yticks(opts.img_trialax_ticks)
        return
    low, high = ax.get_ylim()
    ax.set_yticks(_tick_values(low, high, None))


def _tick_values(low: float, high: float, requested: np.ndarray | None) -> np.ndarray:
    """EEGLAB's order-of-magnitude trial-axis ticks within [low, high]."""
    if requested is not None:
        values = np.unique(np.asarray(requested, dtype=float))
    else:
        order = order_of_magnitude(high - low)
        bounds = np.array([low, high]) / order
        start, stop = np.sign(bounds) * np.floor(np.abs(bounds) + 0.5) * order
        values = np.arange(start, stop + order / 2, order)
    return values[(values >= low) & (values <= high)]


def _draw_colorbar(figure: Any, image: Any, ax: Any, cbar_title: str | None) -> None:
    colorbar = figure.colorbar(image, cax=ax)
    vmin, vmax = image.get_clim()
    if vmax > vmin:
        ticks, labels = _colorbar_ticks(vmin, vmax)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{value:g}" for value in labels])
    colorbar.ax.tick_params(labelsize=TICKFONT)
    if cbar_title:
        ax.set_title(_tex_label(cbar_title), fontsize=TICKFONT)


def _colorbar_ticks(vmin: float, vmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Five evenly spaced ticks with EEGLAB cbar's decade-based label rounding (cbar.m)."""
    ticks = np.linspace(vmin, vmax, 5)
    scale = max(abs(vmin), abs(vmax))
    dec = int(np.floor(np.log10(scale)))
    if dec < 1:
        labels = np.round(ticks * 10.0 ** (1 - dec)) * 10.0 ** (dec - 1)
    elif dec == 1:
        labels = np.round(ticks * 10.0 ** (2 - dec)) * 10.0 ** (dec - 2)
    else:
        labels = np.round(ticks)
    return ticks, labels


def _tex_label(text: str) -> str:
    """EEGLAB passes TeX labels such as ``'\\muV'``; render the Greek mu directly."""
    return text.replace("\\mu", "µ")


def _plot_trace(
    ax: Any,
    times: np.ndarray,
    trace: np.ndarray,
    limits: tuple[float, float, float, float],
    *,
    signif: np.ndarray | None = None,
    stdev: np.ndarray | None = None,
    windows: np.ndarray | None = None,
    grid: bool = False,
    ticks: np.ndarray | None = None,
) -> None:
    """EEGLAB ``plot1trace``: trace(s) with optional significance band, std and sort windows."""
    traces = np.atleast_2d(trace)
    valid = not np.any(np.isnan(limits))
    if windows is not None:
        for window in np.atleast_2d(windows):
            hann = _hanning(window.size)
            hann = hann / np.max(hann)
            if valid:
                lower, upper = hann * limits[2], hann * limits[3]
            else:
                lower, upper = hann * 2 * np.nanmin(traces), hann * 2 * np.nanmax(traces)
            ax.fill(np.r_[window, window[::-1]], np.r_[lower, upper], color=WINFILLCOLOR, edgecolor=WINFILLCOLOR)
    if signif is not None:
        band = np.real(np.atleast_2d(signif))
        ax.fill(np.r_[times, times[::-1]], np.r_[band[0], band[1][::-1]], facecolor=FILLCOLOR, edgecolor=FILLCOLOR)
    if stdev is not None:
        ax.plot(times, traces[0] + stdev, "r--", linewidth=1)
        ax.plot(times, traces[0] - stdev, "r--", linewidth=1)
    handles = [
        ax.plot(times, row, linewidth=ERPDATAWIDTH, color=ERP_TRACE_COLORS[index % len(ERP_TRACE_COLORS)], zorder=3)[0]
        for index, row in enumerate(traces)
    ]
    if traces.shape[0] in ERP_LEGENDS:
        ax.legend(handles, ERP_LEGENDS[traces.shape[0]], fontsize=TICKFONT - 2)
    if not valid:
        return
    if limits[1] > limits[0] and limits[3] > limits[2]:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(1.1 * limits[2], 1.1 * limits[3])
    ax.plot(limits[:2], [0, 0], color="k", linewidth=ERPZEROWIDTH)
    ax.plot([0, 0], [1.1 * limits[2], 1.1 * limits[3]], color="k", linewidth=ERPZEROWIDTH)
    ax.set_yticks(ticks if ticks is not None else _trace_ticks(limits[2], limits[3]))
    if grid:
        ax.yaxis.grid(True, linestyle="--")


def _trace_ticks(low: float, high: float) -> np.ndarray:
    """EEGLAB ``plot1trace`` y ticks: four steps over 80% of the range, through zero."""
    shrunk_low, shrunk_high = 0.8 * low, 0.8 * high
    step = (shrunk_high - shrunk_low) / 4
    if step > 1:
        step = float(np.floor(step + 0.5))
    else:
        order = order_of_magnitude(step)
        step = float(np.floor(step / order + 0.5) * order)
    if step <= 0:
        return np.array([0.0])
    if np.sign(shrunk_high) * np.sign(shrunk_low) == 1:
        return np.unique(np.arange(shrunk_low, shrunk_high + step / 2, step))
    upward = np.arange(0.0, high + step / 2, step)
    downward = -np.arange(0.0, -low + step / 2, step)
    return np.unique(np.r_[upward[upward <= high], downward[downward >= low]])


def _hanning(length: int) -> np.ndarray:
    """EEGLAB ``makehanning``: symmetric Hann window without zero end points."""
    half = np.arange(1, length // 2 + 1) if length % 2 == 0 else np.arange(1, (length + 1) // 2 + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * half / (length + 1)))
    return np.r_[window, window[::-1]] if length % 2 == 0 else np.r_[window, window[-2::-1]]


def _vertical_marks(result: ErpimageComputation) -> list[tuple[float, str]]:
    """Latencies EEGLAB marks on the trace panels (median of per-trial times, aligned when needed)."""
    if result.verttimes is None:
        return []
    if result.verttimes.ndim == 2:
        ordered = np.sort(result.verttimes, axis=0)
        latencies = ordered[int(np.ceil(ordered.shape[0] / 2)) - 1]
    else:
        latencies = result.verttimes.ravel()
    styles = (
        result.vert_colors if result.vert_colors is not None and len(result.vert_colors) == latencies.size else None
    )
    outsort = _outsort(result)
    marks = []
    for index, latency in enumerate(latencies):
        if not np.isnan(result.aligntime):
            latency = float(np.median(result.aligntime + latency - outsort))
        marks.append((float(latency), _line_style(styles[index] if styles is not None else None)))
    return marks


def _value_labels(ax: Any, result: ErpimageComputation, values: tuple[float, ...], label: str | None) -> None:
    """Value labels (and a rotated panel label) left of a trace panel, as EEGLAB writes them."""
    span = result.timelimits[1] - result.timelimits[0]
    offset = result.timelimits[0] - LABEL_OFFSET_FRACTION * span
    for value in values:
        ax.text(offset, value, f"{value:.3g}", ha="right", fontsize=TICKFONT)
    if label:
        middle = (max(values) + min(values)) / 2
        ax.text(
            offset - TITLE_OFFSET_FRACTION * span,
            middle,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=LABELFONT,
        )


def _draw_erp(ax: Any, result: ErpimageComputation, opts: ErpimageOptions, has_coher: bool) -> None:
    low, high = result.erp_limits
    limits = (float(result.timelimits[0]), float(result.timelimits[1]), low, high)
    windows = result.times[result.winlocs] if opts.showwin and result.winlocs is not None else None
    _plot_trace(
        ax,
        result.times,
        result.erp,
        limits,
        signif=result.erpsig if not opts.erpstd else None,
        stdev=result.erp_std if opts.erpstd else None,
        windows=windows,
        grid=opts.erp_grid,
        ticks=opts.erp_vltg_ticks,
    )
    if not np.isnan(result.aligntime):
        for latency in (result.aligntime, 0.0):
            ax.plot([latency, latency], [1.1 * low, 1.1 * high], color="k", linewidth=ZEROWIDTH)
    for latency, style in _vertical_marks(result):
        ax.plot([latency, latency], [low, high], style, linewidth=VERTWIDTH)
    ax.set_ylabel(_tex_label(opts.yerplabel), fontsize=LABELFONT)
    ax.set_facecolor(BACKCOLOR)
    ax.tick_params(labelsize=TICKFONT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if has_coher:
        # With amplitude and coherence panels EEGLAB hides the ERP axes and writes its range.
        bottom, top = ax.get_ylim()
        ax.axis("off")
        _value_labels(ax, result, (top, bottom), None)
        return
    if not opts.noxlabel:
        ax.set_xlabel("Time (ms)" if result.times_passed else "Frames", fontsize=LABELFONT)


def _draw_amplitude(ax: Any, result: ErpimageComputation) -> None:
    minamp, maxamp = result.amp_limits
    limits = (float(result.timelimits[0]), float(result.timelimits[1]), minamp, maxamp)
    signif = None
    if result.cohsig_flag and result.ampsig is not None:
        half = float(np.real(result.ampsig[0] - np.mean(result.ampsig)))
        signif = np.vstack([np.full(result.times.size, half), np.full(result.times.size, -half)])
    _plot_trace(ax, result.times, result.amps, limits, signif=signif)
    if not np.isnan(result.aligntime):
        ax.plot([result.aligntime] * 2, [1.1 * minamp, 1.1 * maxamp], color="k")
    for latency, style in _vertical_marks(result):
        ax.plot([latency, latency], [minamp, maxamp], style, linewidth=VERTWIDTH)
    ax.set_xlim(*result.timelimits)
    ax.set_ylim(minamp, maxamp)
    ax.axis("off")
    _value_labels(ax, result, (maxamp, 0.0, minamp), "ERSP")
    span = result.timelimits[1] - result.timelimits[0]
    ax.text(
        result.timelimits[0] + span / 13,
        minamp + 11 / 13 * (maxamp - minamp),
        f"{result.baseamp:.4g} dB",
        fontsize=TICKFONT,
    )


def _draw_coherence(ax: Any, result: ErpimageComputation, opts: ErpimageOptions) -> None:
    mincoh, maxcoh = result.coh_limits
    limits = (float(result.timelimits[0]), float(result.timelimits[1]), mincoh, maxcoh)
    signif = None
    if result.cohsig_flag and result.cohsig is not None:
        signif = np.vstack([np.full(result.times.size, result.cohsig), np.zeros(result.times.size)])
    _plot_trace(ax, result.times, result.cohers, limits, signif=signif)
    if not np.isnan(result.aligntime):
        ax.plot([result.aligntime] * 2, [1.1 * mincoh, 1.1 * maxcoh], color="k")
    for latency, style in _vertical_marks(result):
        ax.plot([latency, latency], [mincoh, maxcoh], style, linewidth=VERTWIDTH)
    ax.set_yticks([])
    ax.spines["left"].set_color(BACKCOLOR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_facecolor(BACKCOLOR)
    _value_labels(ax, result, (maxcoh, 0.0), "ITC")
    if not opts.noxlabel:
        ax.set_xlabel("Time (ms)" if result.times_passed else "Frames", fontsize=LABELFONT)
    low, high = ax.get_ylim()
    span = result.timelimits[1] - result.timelimits[0]
    ax.text(
        result.timelimits[0] + 10 / 13 * span,
        low + 8 / 13 * (high - low),
        f"{result.coherfreq:.4g} Hz",
        fontsize=TICKFONT,
    )


def _draw_insets(figure: Any, grid: Any, result: ErpimageComputation, opts: ErpimageOptions) -> tuple[Any, ...]:
    """Scalp map at upper left and data spectrum at upper right, as EEGLAB places them."""
    insets: tuple[Any, ...] = ()
    cell = grid[0, 0].get_position(figure)
    width, height = figure.figure.get_size_inches()
    if opts.topo is not None:
        # Keep the scalp map square in display coordinates.
        ax = figure.add_axes((cell.x0 + 0.10 * cell.width, cell.y0, cell.height * height / width, cell.height))
        values = np.asarray(opts.topo[0], dtype=float).ravel()
        chanlocs = list(opts.topo[1])
        if values.size == 1:
            plot_channel_location(ax, chanlocs, int(values[0]))
        else:
            topoplot(values, chanlocs, electrodes="off", axes=ax)
        insets += (ax,)
    if opts.spec is not None:
        ax = figure.add_axes((cell.x0 + 0.82 * cell.width, cell.y0, 0.15 * cell.width, cell.height))
        psd, freqs = welch_psd(result.urdata.reshape(-1, order="F"), result.times.size, result.srate)
        with np.errstate(divide="ignore"):
            decibels = 10 * np.log10(psd)
        ax.plot(freqs, decibels)
        band = (freqs >= opts.spec[0]) & (freqs <= opts.spec[1])
        ax.set_xscale(opts.specaxis)
        ax.set_xlim(opts.spec[0], opts.spec[1])
        if np.any(band):
            ax.set_ylim(np.min(decibels[band]) - 1, np.max(decibels[band]) + 1)
        ax.set_ylabel("dB", fontsize=LABELFONT)
        ax.tick_params(labelsize=LABELFONT)
        ax.set_facecolor(BACKCOLOR)
        if not np.isnan(result.coherfreq):
            ax.plot([result.coherfreq] * 2, ax.get_ylim(), "r")
        insets += (ax,)
    return insets


__all__ = ["ErpimageResult", "butterworth_filter", "erpimage"]
