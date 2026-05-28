"""EEGLAB-style wrapper for channel/component spectra and maps."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import (
    component_activations,
    data_time_slice,
    numeric_vector,
    python_literal,
)
from eegprep.functions.popfunc._pop_utils import parse_key_value_args, parse_text_tokens
from eegprep.functions.sigprocfunc.spectopo import spectopo


def pop_spectopo(
    EEG: dict[str, Any] | None = None,
    dataflag: int = 1,
    timerange: Any = None,
    process: str = "EEG",
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot channel or component spectra and scalp maps."""
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

    data, times = data_time_slice(EEG, timerange)
    if dataflag:
        plot_data = _channel_spectral_data(data, process)
        title = "Channel spectra and maps"
    else:
        plot_data = _component_spectral_data(EEG, timerange)
        title = "Component spectra and maps"
    freqs = numeric_vector(options.pop("freqs", options.pop("freq", []))).tolist()
    freqrange = numeric_vector(options.pop("freqrange", [])).tolist()
    percent = float(options.pop("percent", 100))
    spectra, frequency_values, speccomp, contrib, specstd, figure = spectopo(
        plot_data,
        int(plot_data.shape[1]),
        float(EEG.get("srate", 1) or 1),
        percent=percent,
        freqs=freqs,
        freqrange=freqrange,
        chanlocs=EEG.get("chanlocs", []) if dataflag else None,
        title=title,
        **options,
    )
    result = {
        "spectra": spectra,
        "freqs": frequency_values,
        "speccomp": speccomp,
        "contrib": contrib,
        "specstd": specstd,
        "figure": figure,
        "times": times,
    }
    command = _history_command(dataflag, timerange, process, freqs, freqrange, percent, options)
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
            ControlSpec("edit", tag="options", value="'electrodes', 'off'"),
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
    options = _parse_options_text(result.get("options", ""))
    gui_options = {
        **options,
        "percent": float(result.get("percent", 100) or 100),
        "freqs": numeric_vector(result.get("freqs", [])).tolist(),
        "freqrange": numeric_vector(result.get("freqrange", [])).tolist(),
    }
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
    return data.reshape(data.shape[0], -1)


def _component_spectral_data(EEG: dict[str, Any], timerange: Any) -> np.ndarray:
    acts = component_activations(EEG)
    times = np.asarray([])
    if timerange is not None:
        _data, times = data_time_slice(EEG, timerange)
    if times.size:
        full_times = np.linspace(float(EEG.get("xmin", 0)) * 1000.0, float(EEG.get("xmax", 0)) * 1000.0, acts.shape[1])
        mask = (full_times >= times[0]) & (full_times <= times[-1])
        acts = acts[:, mask, :]
    return acts.reshape(acts.shape[0], -1)


def _parse_options_text(text: Any) -> dict[str, Any]:
    tokens = parse_text_tokens(text)
    if not tokens:
        return {}
    if len(tokens) % 2:
        raise ValueError("Spectral options must be key/value pairs")
    return {str(tokens[index]).lower(): tokens[index + 1] for index in range(0, len(tokens), 2)}


def _history_command(
    dataflag: int,
    timerange: Any,
    process: str,
    freqs: list[float],
    freqrange: list[float],
    percent: float,
    options: dict[str, Any],
) -> str:
    kwargs: dict[str, Any] = {
        "dataflag": int(dataflag),
        "timerange": timerange,
        "process": process,
        "freqs": freqs,
        "freqrange": freqrange,
        "percent": percent,
    }
    kwargs.update(options)
    pieces = ["EEG", *(f"{key}={python_literal(value)}" for key, value in kwargs.items())]
    return f"pop_spectopo({', '.join(pieces)})"


__all__ = ["pop_spectopo", "pop_spectopo_dialog_spec"]
