"""Draw a window of signal.

Needs the ``plot`` extra. matplotlib is 9.8 MB of the Pyodide download, which is why it
is not in the base install: it arrives when a plot is actually asked for.

**Nothing here touches ``matplotlib.pyplot``.** pyplot picks a backend at import, and in a
browser that means a backend wanting the DOM; the object-oriented ``Figure`` API needs
none. It is also the shape the browser case actually wants, which is a PNG to hand to the
page rather than a window to show. :func:`to_png` does that step. A test asserts pyplot
stays unimported, because importing it anywhere in this module would be invisible
natively and fatal in the browser.

**The unit comes from the window, and is omitted when the window has none.** A
:class:`~eegprep_lean.window.Window` read through :func:`~eegprep_lean.window.read_window`
carries the unit its channels declare, and the scale bar names it. When the window spans
channels that do not agree on one, or holds stored counts rather than converted values,
there is nothing to name and the bar says so rather than guessing. An earlier version of
this module asserted units were unknowable; that was wrong, and it came from reading the
level-0 array, which carries the conversion but not the unit it produces. They are on the
channel group, and :mod:`eegprep_lean.channels` reads them.

**Traces are demeaned for display by default.** Level-0 channels carry per-channel DC
offsets in the thousands, and they differ between channels by more than the signal spans,
so without it every trace is a flat line at its own level and the plot shows nothing.
That is display only: it never touches the window. ``demean=False`` draws the values as
they are. Note this is demeaning, not detrending: no linear trend is removed.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .window import Window

#: Percentiles used for each channel's amplitude when choosing a default trace spacing.
#: Not min and max: one sample of amplifier saturation would otherwise set the scale for
#: the whole plot.
SPACING_PERCENTILES = (0.5, 99.5)

#: How much room to leave between traces, as a multiple of the amplitude the spacing is
#: derived from. 1.0 would have neighboring traces touch at their extremes.
SPACING_HEADROOM = 1.2

#: Used when the median channel is flat, where a data-derived spacing would be zero and
#: stack every trace on one line. The median, not every channel: a group where half the
#: channels are flat still lands here, which is the right answer, because the spacing rule
#: below follows the median by design.
FALLBACK_SPACING = 1.0


def default_spacing(data: np.ndarray) -> float:
    """Vertical distance between traces, from the data itself.

    The **median** channel amplitude, not the maximum. A single high-amplitude channel is
    ordinary in raw EEG, and scaling the plot to it would flatten every other channel to
    an unreadable line. Scaling to the median leaves that one channel overlapping its
    neighbors and the rest legible, which is the better failure of the two. Pass
    ``spacing=`` to override.
    """
    low, high = np.percentile(np.asarray(data, dtype=np.float64), SPACING_PERCENTILES, axis=1)
    amplitude = float(np.median(high - low))
    return amplitude * SPACING_HEADROOM if amplitude > 0 else FALLBACK_SPACING


def _resolve_labels(labels: Sequence[str] | None, window: Window) -> list[str]:
    """Row labels: the caller's, else the window's own, else the channel indices.

    The window's labels come from the channel group, which is where the recording's own
    names live. Falling back to indices only when the group did not supply them keeps a
    plot from showing ``0, 1, 2`` for a recording that calls them ``E1, E2, E3``.
    """
    channels = window.channels
    if labels is None:
        return list(window.labels) if window.labels else [str(channel) for channel in channels]
    labels = list(labels)
    if len(labels) != len(channels):
        raise ValueError(
            f"got {len(labels)} labels for {len(channels)} channels in the window. "
            "Labels are positional, so a mismatch would silently rename every trace "
            f"below the first missing one (window channels: {list(channels)})"
        )
    return labels


def plot_window(
    window: Window,
    *,
    ax: Axes | None = None,
    labels: Sequence[str] | None = None,
    spacing: float | None = None,
    demean: bool = True,
    scalebar: bool = True,
    **line_kwargs: Any,
) -> Axes:
    """Draw a window as stacked traces, first channel at the top.

    Returns the :class:`~matplotlib.axes.Axes` so a caller can keep styling it. When
    ``ax`` is None a new :class:`~matplotlib.figure.Figure` is made without pyplot; reach
    the figure as ``ax.figure`` and :func:`to_png` it.

    Trace lines carry ``gid="trace-<channel>"`` and the scale bar ``gid="scalebar"``, so
    they can be found again on an Axes the caller also drew on.
    """
    data = np.asarray(window.data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"expected a channels-by-samples window, got shape {data.shape}")
    row_labels = _resolve_labels(labels, window)

    if demean:
        data = data - data.mean(axis=1, keepdims=True)
    step = float(spacing) if spacing is not None else default_spacing(data)

    if ax is None:
        ax = Figure(figsize=(10, max(2.0, 0.28 * len(row_labels) + 1.0))).add_subplot(1, 1, 1)

    times = window.times_s
    # Channel 0 on top, counting downward, as every EEG viewer draws it.
    baselines = [-index * step for index in range(len(row_labels))]
    line_kwargs.setdefault("linewidth", 0.6)
    for row, (channel, baseline) in enumerate(zip(window.channels, baselines, strict=True)):
        ax.plot(times, data[row] + baseline, gid=f"trace-{channel}", **line_kwargs)

    ax.set_yticks(baselines)
    ax.set_yticklabels(row_labels)
    ax.set_ylim(baselines[-1] - 1.6 * step, baselines[0] + 1.0 * step)
    ax.set_xlim(float(times[0]), float(times[-1]))
    # Absolute in the recording, not in the window: a window starting at sample 2500 of a
    # 250 Hz recording begins at 10 s, and labeling it 0 would misplace every annotation
    # a caller draws on top.
    ax.set_xlabel("time (s)")
    ax.set_ylabel("channel")
    # Says when level 0 is not the rate the recording was acquired at, because a reader
    # who assumes it is has silently lost half the bandwidth they think they have.
    title = f"{window.group_name}, {window.rate:g} Hz"
    if window.was_resampled:
        title += f" (resampled from {window.original_rate:g} Hz)"
    ax.set_title(title)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if scalebar:
        _draw_scalebar(ax, times, baselines[-1] - 1.3 * step, step, _amplitude_word(window))
    return ax


def _amplitude_word(window: Window) -> str:
    """What the scale bar's number is measured in.

    ``counts`` for stored integers, the channels' own unit when they agree on one, and
    ``units`` when they do not. Never a modality default: magnetoencephalography is a
    Tesla-based unit, not a voltage, so a per-modality guess is wrong by a factor nobody
    notices on a plot.
    """
    if not window.physical:
        return "counts"
    return window.unit or "units"


def _draw_scalebar(ax: Axes, times: np.ndarray, bottom: float, step: float, word: str) -> None:
    """A bar one trace-spacing tall, labeled with its size.

    This is how amplitude is stated on a stacked plot, where the axis carries channel
    names rather than a scale: the reader learns how big a deflection is without the plot
    needing a second axis for it. The word changes with what was read, because stored
    counts and converted values are not the same quantity.
    """
    span = float(times[-1]) - float(times[0])
    x = float(times[-1]) - 0.02 * span if span > 0 else float(times[-1])
    ax.plot([x, x], [bottom, bottom + step], color="black", linewidth=1.5, gid="scalebar")
    ax.text(
        x - 0.01 * span,
        bottom + step / 2,
        f"{step:.3g} {word}",
        ha="right",
        va="center",
        fontsize="small",
        gid="scalebar-label",
    )


def to_png(figure: Figure, *, dpi: int = 100) -> bytes:
    """Render a figure to PNG bytes.

    The browser path: Pyodide has no window to show a plot in, so a figure reaches the
    page as bytes. Takes a Figure rather than an Axes because that is what renders;
    ``ax.figure`` is the one :func:`plot_window` made.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    return buffer.getvalue()
