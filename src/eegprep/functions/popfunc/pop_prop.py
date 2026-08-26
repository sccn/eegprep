"""EEGLAB-style channel/component properties plot."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._property_browser import property_activity_browser
from eegprep.functions.popfunc.plot_utils import (
    channel_labels,
    component_activations,
    component_map_data,
    eeg_epoch_data,
    eeg_times_ms,
    history_command,
    numeric_vector,
    parse_plot_options_text,
    show_figures,
)
from eegprep.functions.sigprocfunc.erpimage import erpimage
from eegprep.functions.sigprocfunc.spectopo import compute_spectra
from eegprep.functions.sigprocfunc.topoplot import plot_channel_location, topoplot

# EEGLAB pop_prop() spectrum-axis label and single-channel marker size.
SPECTRUM_YLABEL = r"Power 10*log$_{10}$($\mu$V$^2$/Hz)"
CHANNEL_MARKER_SIZE = 80


def pop_prop(
    EEG: dict[str, Any] | None = None,
    typecomp: int = 1,
    chanorcomp: Any = None,
    winhandle: Any = None,
    spec_opt: Any = None,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: str | bool = "on",
    scroll_event: int | bool = 1,
    show_activity: bool = False,
    return_com: bool = False,
):
    """Plot properties of one channel or independent component.

    Pass ``plot='off'`` to build and return the figure without opening a window.
    """
    if EEG is None:
        return (None, "") if return_com else None
    typecomp = int(typecomp)
    if gui is None:
        gui = chanorcomp is None
    if gui:
        result = _run_gui(EEG, typecomp=typecomp, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        chanorcomp = result["chanorcomp"]
        spec_opt = result["spec_opt"]
    indices = numeric_vector(chanorcomp if chanorcomp is not None else 1, dtype=int)
    figures = [_plot_one_property(EEG, typecomp, int(index), spec_opt) for index in indices]
    for figure, index in zip(figures, indices):
        figure.eegprep_activity_view = property_activity_browser(
            EEG,
            typecomp,
            int(index),
            scroll_event=scroll_event,
            show=show_activity,
        )
    command = history_command("pop_prop", typecomp, indices.astype(int).tolist(), winhandle, spec_opt)
    show_figures(figures, plot=plot)
    return (
        (figures[0] if len(figures) == 1 else figures, command)
        if return_com
        else figures[0]
        if len(figures) == 1
        else figures
    )


def pop_prop_dialog_spec(EEG: dict[str, Any], *, typecomp: int = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_prop``."""
    label = "Channel" if int(typecomp) else "Component"
    labels = channel_labels(EEG)
    selector_enabled = bool(int(typecomp)) and bool(labels)
    return DialogSpec(
        title=f"{label} properties - pop_prop()",
        controls=(
            ControlSpec("text", f"{label} index(ices) to plot:"),
            ControlSpec("edit", tag="chanorcomp", value="1"),
            ControlSpec(
                "pushbutton",
                "...",
                tag="chanorcomp_button",
                enabled=selector_enabled,
                callback=CallbackSpec(
                    "select_channels",
                    params={
                        "button": "chanorcomp_button",
                        "target": "chanorcomp",
                        "channels": labels,
                        "selectionmode": "single",
                        "return_indices": True,
                    },
                    matlab_callback=("pop_chansel({tmpchanlocs.labels}, 'withindex', 'on', 'selectionmode', 'single')"),
                ),
            ),
            ControlSpec("text", "Spectral options (see spectopo() help):"),
            ControlSpec("edit", tag="spec_opt", value="'freqrange', [2, 50]"),
            ControlSpec("spacer"),
        ),
        geometry=((2, 1, 0.5), (2, 1, 0.5)),
        function_name="pop_prop",
        eeglab_source="functions/popfunc/pop_prop.m",
        help_text="pophelp('pop_prop')",
        size=(615, 199),
    )


def _run_gui(EEG: dict[str, Any], *, typecomp: int, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_prop_dialog_spec(EEG, typecomp=typecomp), renderer=renderer)
    if result is None:
        return None
    return {
        "chanorcomp": numeric_vector(result.get("chanorcomp", 1), dtype=int).tolist(),
        "spec_opt": str(result.get("spec_opt", "") or ""),
    }


def _plot_one_property(EEG: dict[str, Any], typecomp: int, index: int, spec_opt: Any):
    """Build the EEGLAB three-panel figure: scalp map, ERP image, power spectrum."""
    data = eeg_epoch_data(EEG)
    times = eeg_times_ms(EEG)
    trials = int(EEG.get("trials", 1) or 1)

    fig = plt.figure(figsize=(5, 5), layout="constrained")  # EEGLAB uses a 500x500 square
    top, bottom = fig.subfigures(2, 1, height_ratios=(3, 2))
    topo_sf, erp_sf = top.subfigures(1, 2, width_ratios=(2, 3))
    topo_ax = topo_sf.add_subplot(1, 1, 1)
    spec_ax = bottom.add_subplot(1, 1, 1)

    if typecomp:
        if index < 1 or index > data.shape[0]:
            raise ValueError("channel index is outside available channels")
        trace = data[index - 1]
        basename = f"Channel {index}"
        plot_channel_location(topo_ax, chanlocs_as_list(EEG.get("chanlocs", [])), index, markersize=CHANNEL_MARKER_SIZE)
        spectrum_input = data[index - 1 : index]
        mapnorm = None
    else:
        acts = component_activations(EEG)
        maps, map_chanlocs = component_map_data(EEG)
        if index < 1 or index > acts.shape[0]:
            raise ValueError("component index is outside available ICA components")
        trace = acts[index - 1]
        basename = f"IC{index}"
        topoplot(maps[:, index - 1], map_chanlocs, axes=topo_ax, electrodes="off")
        spectrum_input = acts[index - 1 : index]
        mapnorm = np.asarray(EEG["icawinv"], dtype=float)[:, index - 1]
    topo_ax.set_title(basename, fontsize=14)

    _draw_erp_image(erp_sf, trace, times, trials, basename, float(EEG.get("srate", 1) or 1))
    _draw_spectrum(spec_ax, EEG, spectrum_input, spec_opt, mapnorm)
    fig.suptitle(f"pop_prop() - {basename} properties", fontweight="bold")
    return fig


def _draw_erp_image(
    container: Any, trace: np.ndarray, times: np.ndarray, trials: int, basename: str, srate: float
) -> None:
    """Draw the ERP-image panel, offset-subtracted like EEGLAB nan_mean."""
    trace_2d = trace if trace.ndim == 2 else trace[:, np.newaxis]
    if trials > 1:
        offset = float(np.nanmean(trace_2d))
        smooth = 1 if trials < 6 else 3
        erpimage(
            trace_2d - offset,
            times=times,
            title=f"{basename} activity (global offset {offset:.3f})",
            smooth=smooth,
            caxis=2.0 / 3.0,
            cbar=True,
            plot_erp=True,
            target=container,
        )
    else:
        _draw_continuous_erp_image(container, trace_2d.reshape(-1), srate)


def _draw_continuous_erp_image(container: Any, samples: np.ndarray, srate: float) -> None:
    """Reshape continuous data into ~1s lines for an ERP image (EEGLAB pop_prop)."""
    offset = float(np.nanmean(samples))
    lines = 200.0
    while samples.size < lines * srate:
        lines *= 0.9
    lines = int(round(lines))
    if lines <= 2:
        ax = container.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.text(0.1, 0.3, "No ERP image plotted\nfor small continuous data")
        return
    frames = samples.size // lines
    # EEGLAB reshapes column-major into (frames, lines); the degenerate eegtimes it
    # builds is replaced here with a plain sample-index axis.
    image = samples[: frames * lines].reshape(frames, lines, order="F") - offset
    smooth = 1 if lines < 10 else 3
    erpimage(
        image, title="Continuous data", smooth=smooth, caxis=2.0 / 3.0, cbar=True, plot_erp=False, target=container
    )


def _draw_spectrum(ax: Any, EEG: dict[str, Any], spectrum_input: np.ndarray, spec_opt: Any, mapnorm: Any) -> None:
    """Draw the activity power spectrum from raw per-epoch data (spectopo)."""
    spec_options = parse_plot_options_text(spec_opt)
    spectra, freqs, _std = compute_spectra(
        spectrum_input,
        int(EEG.get("pnts", spectrum_input.shape[1]) or spectrum_input.shape[1]),
        float(EEG.get("srate", 1) or 1),
        winsize=_first_int(spec_options.get("winsize")),
        overlap=_first_int(spec_options.get("overlap")) or 0,
        nfft=_first_int(spec_options.get("nfft")),
        mapnorm=mapnorm,
    )
    ax.plot(freqs, spectra[0], color="black")
    freqrange = numeric_vector(spec_options.get("freqrange", []))
    if freqrange.size == 2:
        ax.set_xlim(float(freqrange[0]), float(freqrange[1]))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(SPECTRUM_YLABEL)
    ax.set_title("Activity power spectrum")
    ax.grid(True, alpha=0.25)


def _first_int(value: Any) -> int | None:
    vector = numeric_vector(value)
    if vector.size == 0:
        return None
    return int(vector[0])


__all__ = ["pop_prop", "pop_prop_dialog_spec"]
