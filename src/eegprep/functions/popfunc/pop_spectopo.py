"""EEGLAB-style wrapper for channel/component spectra and maps."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc.plot_utils import (
    component_activations,
    component_map_data,
    data_time_slice,
    history_command,
    numeric_vector,
    parse_plot_options_text,
    selected_indices,
    show_figures,
)
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.sigprocfunc.spectopo import plot_component_spectra, spectopo


def pop_spectopo(
    EEG: dict[str, Any] | None = None,
    dataflag: int = 1,
    timerange: Any = None,
    process: str = "EEG",
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: str = "on",
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot channel or component spectra and scalp maps.

    Pass ``plot='off'`` to build and return the figure without opening a window.
    """
    if EEG is None:
        return (None, "") if return_com else None
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    dataflag = int(dataflag)
    if gui is None:
        gui = timerange is None and not options
    if gui:
        result = _run_gui(EEG, dataflag=dataflag, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        timerange = result["timerange"]
        process = result["process"]
        options.update(result["options"])

    # "plot" gates the window here; keep it out of the core options so it can't
    # also suppress figure drawing.
    plot = str(options.pop("plot", plot))

    data, times = data_time_slice(EEG, timerange)
    history_options = {key: value for key, value in options.items() if key not in {"plotchan", "icamode"}}
    srate = float(EEG.get("srate", 1) or 1)
    freqs = numeric_vector(options.pop("freqs", options.pop("freq", []))).tolist()
    freqrange = numeric_vector(options.pop("freqrange", [])).tolist()
    percent = float(options.pop("percent", 100))

    if dataflag:
        nicamaps = np.asarray([], dtype=int)
        icamaps = np.asarray([], dtype=int)
        spectral_options, topoplot_options = _split_spectopo_options(options)
        plot_data = _channel_spectral_data(data, process)
        spectra, frequency_values, speccomp, contrib, specstd, figure = spectopo(
            plot_data,
            int(plot_data.shape[1]),
            srate,
            percent=percent,
            freqs=freqs,
            freqrange=freqrange,
            chanlocs=EEG.get("chanlocs", []),
            topoplot_options=topoplot_options,
            title="",  # EEGLAB spectopo adds no default suptitle
            **spectral_options,
        )
    else:
        _raise_for_unsupported_component_options(options)
        comp_data, component_numbers = _component_spectral_data(EEG, timerange, options.get("icacomps"))
        icawinv, chanlocs = component_map_data(EEG)
        nicamaps = numeric_vector(options.pop("nicamaps", []), dtype=int)
        icamaps = numeric_vector(options.pop("icamaps", []), dtype=int)
        spectral_options, topoplot_options = _split_spectopo_options(options)
        # EEGLAB rescales each component's whole-scalp spectrum by its projection
        # strength: add 10*log10(sqrt(mean(icawinv_col^4))) (spectopo 'normal' icamode).
        comp_spectra, frequency_values, speccomp, contrib, specstd, _ = spectopo(
            comp_data, int(comp_data.shape[1]), srate, percent=percent, plot="off", **spectral_options
        )
        projection = np.asarray(icawinv, dtype=float)[:, component_numbers - 1]
        comp_spectra = comp_spectra + (10.0 * np.log10(np.sqrt(np.nanmean(projection**4, axis=0))))[:, None]
        # Channel-data spectra drive the bold black RMS curve and the composite freq map.
        channel_spectra, *_ = spectopo(
            _channel_spectral_data(data, process),
            int(data.shape[1]),
            srate,
            percent=percent,
            plot="off",
            **spectral_options,
        )
        figure = plot_component_spectra(
            comp_spectra,
            frequency_values,
            component_numbers=component_numbers,
            icawinv=np.asarray(icawinv, dtype=float),
            chanlocs=chanlocs,
            channel_spectra=channel_spectra,
            channel_chanlocs=EEG.get("chanlocs", []),
            freq=freqs,
            freqrange=freqrange,
            nicamaps=int(nicamaps[0]) if nicamaps.size else 5,
            icamaps=icamaps.tolist() if icamaps.size else None,
            topoplot_options=topoplot_options,
        )
        spectra = comp_spectra

    result = {
        "spectra": spectra,
        "freqs": frequency_values,
        "speccomp": speccomp,
        "contrib": contrib,
        "specstd": specstd,
        "figure": figure,
        "times": times,
    }
    command = history_command(
        "pop_spectopo",
        eeg_name="EEG",
        dataflag=dataflag,
        timerange=timerange,
        process=None if process == "EEG" else process,
        **history_options,
    )
    show_figures(figure, plot=plot)
    return (result, command) if return_com else result


def pop_spectopo_dialog_spec(EEG: dict[str, Any], *, dataflag: int = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_spectopo``."""
    continuous = int(EEG.get("trials", 1) or 1) == 1
    controls = [
        ControlSpec("text", "Epoch time range to analyze [min_ms max_ms]:"),
        ControlSpec(
            "edit", tag="timerange", value=f"{float(EEG.get('xmin', 0)) * 1000:g} {float(EEG.get('xmax', 0)) * 1000:g}"
        ),
    ]
    if dataflag:
        controls.extend(
            [
                ControlSpec("text", "Percent data to sample (1 to 100):"),
                ControlSpec("edit", tag="percent", value="100"),
                ControlSpec("text", "Frequencies to plot as scalp maps (Hz):"),
                ControlSpec("edit", tag="freqs", value="6 10 22"),
            ]
        )
        if not continuous:
            controls.extend(
                [
                    ControlSpec("text", "Apply to EEG|ERP|BOTH:"),
                    ControlSpec("edit", tag="process", value="EEG"),
                ]
            )
    else:
        n_components = np.asarray(EEG.get("icaweights", [])).shape[0]
        controls.extend(
            [
                ControlSpec("text", "Frequency (Hz) to analyze:"),
                ControlSpec("edit", tag="freqs", value="10"),
                ControlSpec("text", "Electrode number to analyze ([]=elec with max power; 0=whole scalp):"),
                ControlSpec("edit", tag="plotchan", value="0"),
                ControlSpec("text", "Percent data to sample (1 to 100):"),
                ControlSpec("edit", tag="percent", value="20"),
                ControlSpec("text", "Components to include in the analysis:"),
                ControlSpec("edit", tag="icacomps", value=f"1:{n_components}" if n_components else ""),
                ControlSpec("text", "Number of largest-contributing components to map:"),
                ControlSpec("edit", tag="nicamaps", value="5"),
                ControlSpec("text", "     Else, map only these component numbers:"),
                ControlSpec("edit", tag="icamaps", value=""),
                ControlSpec("text", "[Checked] Compute comp spectra; [Unchecked] (data-comp) spectra:"),
                ControlSpec("checkbox", tag="icamode", value=True),
            ]
        )
    controls.extend(
        [
            ControlSpec("text", "Plotting frequency range [lo_Hz hi_Hz]:"),
            ControlSpec("edit", tag="freqrange", value="2 25"),
            ControlSpec("text", "Spectral and scalp map options (see topoplot):"),
            ControlSpec("edit", tag="options", value="electrodes='off'"),
        ]
    )
    return DialogSpec(
        title=("Channel" if dataflag else "Component") + " spectra and maps -- pop_spectopo()",
        controls=tuple(controls),
        geometry=tuple((2, 1) for _ in range(len(controls) // 2)),
        function_name="pop_spectopo",
        eeglab_source="functions/popfunc/pop_spectopo.m",
        help_text="pophelp('pop_spectopo')",
        size=(646, 299) if dataflag else (937, 476),
    )


def _run_gui(EEG: dict[str, Any], *, dataflag: int, renderer: Any | None = None) -> dict[str, Any] | None:
    spec = pop_spectopo_dialog_spec(EEG, dataflag=dataflag)
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return None
    options = parse_plot_options_text(result.get("options", ""))
    gui_options = {
        **options,
        "percent": float(result.get("percent", 100) or 100),
        "freqs": numeric_vector(result.get("freqs", [])).tolist(),
        "freqrange": numeric_vector(result.get("freqrange", [])).tolist(),
    }
    if not dataflag:
        gui_options.update(
            {
                "plotchan": numeric_vector(result.get("plotchan", [])).tolist(),
                "icacomps": numeric_vector(result.get("icacomps", []), dtype=int).tolist(),
                "nicamaps": numeric_vector(result.get("nicamaps", []), dtype=int).tolist(),
                "icamaps": numeric_vector(result.get("icamaps", []), dtype=int).tolist(),
                "icamode": bool(result.get("icamode", True)),
            }
        )
    return {
        "timerange": numeric_vector(result.get("timerange", [])).tolist(),
        "process": str(result.get("process", "EEG") or "EEG"),
        "options": gui_options,
    }


def _channel_spectral_data(data: np.ndarray, process: str) -> np.ndarray:
    if data.ndim == 2:
        return data
    mode = process.upper()
    if mode == "ERP":
        return np.nanmean(data, axis=2)
    # Keep 3-D so ``spectopo`` runs per-epoch pwelch and averages linear power,
    # as EEGLAB does. Flattening here with numpy's C-order would interleave trials
    # of Fortran-contiguous EEG data.
    return data


def _component_spectral_data(EEG: dict[str, Any], timerange: Any, components: Any) -> tuple[np.ndarray, np.ndarray]:
    acts = component_activations(EEG)
    times = np.asarray([])
    if timerange is not None:
        _data, times = data_time_slice(EEG, timerange)
    if times.size:
        full_times = np.linspace(float(EEG.get("xmin", 0)) * 1000.0, float(EEG.get("xmax", 0)) * 1000.0, acts.shape[1])
        mask = (full_times >= times[0]) & (full_times <= times[-1])
        acts = acts[:, mask, :]
    indices = selected_indices(components, acts.shape[0])
    return acts[indices, :, :], indices + 1


def _raise_for_unsupported_component_options(options: dict[str, Any]) -> None:
    """Fail loudly when component controls EEGPrep does not implement are set to non-default values.

    EEGLAB's component dialog offers ``plotchan`` (``0`` = whole scalp) and ``icamode``
    (checked = component spectra). EEGPrep only computes whole-scalp component spectra, so a
    non-default ``plotchan`` (a specific electrode or ``[]`` = max-power electrode) or an unchecked
    ``icamode`` ((data-comp) spectra) is rejected instead of being silently ignored.
    """
    if "plotchan" in options:
        plotchan = numeric_vector(options["plotchan"])
        if plotchan.size != 1 or int(plotchan[0]) != 0:
            raise ValueError(
                "pop_spectopo only supports whole-scalp component spectra (plotchan=0); "
                "per-electrode or max-power projection is not available in EEGPrep"
            )
    if "icamode" in options and not bool(options["icamode"]):
        raise ValueError(
            "pop_spectopo only supports component spectra (icamode on); (data-comp) spectra is not available in EEGPrep"
        )


def _split_spectopo_options(options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    spectral_keys = {"plot", "winsize", "overlap", "nfft"}
    spectral = {key: options[key] for key in spectral_keys if key in options}
    if "winsize" in spectral:
        spectral["winsize"] = int(numeric_vector(spectral["winsize"])[0])
    if "overlap" in spectral:
        spectral["overlap"] = int(numeric_vector(spectral["overlap"])[0])
    if "nfft" in spectral:
        spectral["nfft"] = int(numeric_vector(spectral["nfft"])[0])
    topoplot = {key: value for key, value in options.items() if key not in spectral_keys}
    for unused in {"icacomps", "icamaps", "nicamaps", "plotchan", "icamode"}:
        topoplot.pop(unused, None)
    return spectral, topoplot


__all__ = ["pop_spectopo", "pop_spectopo_dialog_spec"]
