"""EEGLAB-style wrapper for channel/component time-frequency plots."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.plot_utils import (
    channel_labels,
    component_activations,
    component_map_data,
    data_time_slice,
    history_command,
    numeric_vector,
    parse_plot_options_text,
    show_figures,
)
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.timefreqfunc.newtimef import newtimef


def pop_newtimef(
    EEG: dict[str, Any] | None = None,
    typeproc: int = 1,
    num: Any = None,
    tlimits: Any = None,
    cycles: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: str | bool = "on",
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot a channel or component ERSP/ITC decomposition.

    Pass ``plot='off'`` to build and return the figure without opening a window.
    """
    if EEG is None:
        return (None, "") if return_com else None
    typeproc = int(typeproc)
    if gui is None:
        gui = num is None
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui:
        result = _run_gui(EEG, typeproc=typeproc, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        num = result["num"]
        tlimits = result["tlimits"]
        cycles = result["cycles"]
        options.update(result["options"])
    if num is None:
        num = 1
    if tlimits is None:
        tlimits = [float(EEG.get("xmin", 0)) * 1000.0, float(EEG.get("xmax", 0)) * 1000.0]
    if cycles is None:
        cycles = [3, 0.8]
    data, times = _selected_signal(EEG, typeproc, num, tlimits)
    plot_options = {"title": "", **_topo_options(EEG, typeproc, num), **options}
    result = newtimef(
        data, data.shape[0], [times[0], times[-1]], float(EEG.get("srate", 1) or 1), cycles, **plot_options
    )
    command = history_command("pop_newtimef", typeproc, _first_index(num), tlimits, cycles, **options)
    show_figures(result.figure, plot=plot)
    return (result, command) if return_com else result


def pop_newtimef_dialog_spec(EEG: dict[str, Any], *, typeproc: int = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_newtimef``."""
    is_channel = bool(int(typeproc))
    labels = channel_labels(EEG)
    controls = [
        ControlSpec("text", "Channel number" if is_channel else "Component number", font_weight="bold"),
        ControlSpec("edit", tag="num", value="1"),
        ControlSpec(
            "pushbutton",
            "...",
            tag="num_button",
            enabled=is_channel and bool(labels),
            callback=CallbackSpec(
                "select_channels",
                params={
                    "button": "num_button",
                    "target": "num",
                    "channels": labels,
                    "selectionmode": "single",
                    "return_indices": True,
                },
                matlab_callback="pop_chansel({tmpchanlocs.labels}, 'withindex', 'on', 'selectionmode', 'single')",
            ),
        ),
        ControlSpec("spacer"),
        ControlSpec("text", "Sub epoch time limits [min max] (msec)", font_weight="bold"),
        ControlSpec(
            "edit",
            tag="tlimits",
            value=f"{float(EEG.get('xmin', 0)) * 1000:g} {float(EEG.get('xmax', 0)) * 1000:g}",
        ),
        ControlSpec("popupmenu", "|".join(_NTIMES_CHOICES), tag="ntimesout", value=4),
        ControlSpec("spacer"),
        ControlSpec("text", "Frequency limits [min max] (Hz) or sequence", font_weight="bold"),
        ControlSpec("edit", tag="freqs", value=""),
        ControlSpec("popupmenu", "|".join(_NFREQS_CHOICES), tag="nfreqs", value=1),
        ControlSpec("checkbox", "Log spaced", tag="freqscale", value=False),
        ControlSpec("text", "Baseline limits [min max] (msec) (0->pre-stim.)", font_weight="bold"),
        ControlSpec("edit", tag="baseline", value="0"),
        ControlSpec("popupmenu", "|".join(_BASELINE_CHOICES), tag="basenorm", value=1),
        ControlSpec("checkbox", "No baseline", tag="nobase", value=False),
        ControlSpec("text", "Wavelet cycles [min max/fact] or sequence", font_weight="bold"),
        ControlSpec("edit", tag="cycles", value="3 0.8"),
        ControlSpec("checkbox", "Use FFT", tag="fft", value=False),
        ControlSpec(
            "pushbutton",
            "tf cycle calc",
            tag="calcpush",
            enabled=True,
            callback=CallbackSpec(
                "tf_cycle_calc",
                params={"button": "calcpush", "freqs": "freqs", "cycles": "cycles", "fft": "fft"},
                matlab_callback="{@comcalc, EEG.srate, EEG.xmin}",
            ),
        ),
        ControlSpec("text", "ERSP color limits [max] (min=-max)", font_weight="bold"),
        ControlSpec("edit", tag="erspmax", value=""),
        ControlSpec("checkbox", "see log power (set)", tag="scale", value=True),
        ControlSpec("spacer"),
        ControlSpec("text", "ITC color limits [max]", font_weight="bold"),
        ControlSpec("edit", tag="itcmax", value=""),
        ControlSpec("checkbox", "plot ITC phase (set)", tag="plotphase", value=False),
        ControlSpec("spacer"),
        ControlSpec("text", "Bootstrap significance level (Ex: 0.01 -> 1%)", font_weight="bold"),
        ControlSpec("edit", tag="alpha", value=""),
        ControlSpec("checkbox", "FDR correct (set)", tag="fdr", value=False),
        ControlSpec("spacer"),
        ControlSpec("text", "Optional newtimef() arguments (see Help)", font_weight="bold"),
        ControlSpec("edit", tag="options", value=""),
        ControlSpec("spacer"),
        ControlSpec("checkbox", "Plot Event Related Spectral Power", tag="plotersp", value=True),
        ControlSpec("checkbox", "Plot Inter Trial Coherence", tag="plotitc", value=True),
        ControlSpec(
            "checkbox",
            "Plot curve at each frequency",
            tag="plotcurve",
            value=False,
        ),
    ]
    return DialogSpec(
        title=(
            "Plot channel time frequency -- pop_newtimef()"
            if is_channel
            else "Plot component time frequency -- pop_newtimef()"
        ),
        controls=tuple(controls),
        geometry=((1, 0.3, 0.25, 0.75),) + ((1, 0.3, 0.6, 0.4),) * 7 + ((0.975, 1.27), (1,), (1.2, 1, 1.2)),
        function_name="pop_newtimef",
        eeglab_source="functions/popfunc/pop_newtimef.m",
        help_text="pophelp('pop_newtimef')",
        size=(1059, 511),
        row_spacing=4,
        extra_stylesheet="""
            QDialog#pop_newtimef QLabel,
            QDialog#pop_newtimef QCheckBox,
            QDialog#pop_newtimef QLineEdit,
            QDialog#pop_newtimef QPushButton,
            QDialog#pop_newtimef QComboBox {
                font-size: 11px;
            }
            QDialog#pop_newtimef QLineEdit,
            QDialog#pop_newtimef QPushButton,
            QDialog#pop_newtimef QComboBox {
                min-height: 15px;
                max-height: 15px;
            }
            QDialog#pop_newtimef QCheckBox::indicator {
                width: 11px;
                height: 11px;
            }
        """,
    )


def _run_gui(EEG: dict[str, Any], *, typeproc: int, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_newtimef_dialog_spec(EEG, typeproc=typeproc), renderer=renderer)
    if result is None:
        return None
    options = parse_plot_options_text(result.get("options", ""))
    freqs = numeric_vector(result.get("freqs", [])).tolist()
    if freqs:
        options["freqs"] = freqs
    baseline = numeric_vector(result.get("baseline", [])).tolist()
    if bool(result.get("nobase", False)):
        options["baseline"] = [float("nan")]
    elif baseline:
        options["baseline"] = baseline
    alpha = numeric_vector(result.get("alpha", []))
    if alpha.size:
        options["alpha"] = float(alpha[0])
    erspmax = numeric_vector(result.get("erspmax", []))
    if erspmax.size:
        options["erspmax"] = erspmax.tolist() if erspmax.size > 1 else float(erspmax[0])
    itcmax = numeric_vector(result.get("itcmax", []))
    if itcmax.size:
        options["itcmax"] = itcmax.tolist() if itcmax.size > 1 else float(itcmax[0])
    if bool(result.get("fdr", False)):
        options["mcorrect"] = "fdr"
    if bool(result.get("freqscale", False)):
        options["freqscale"] = "log"
    if not bool(result.get("scale", True)):
        options["scale"] = "abs"
    if not bool(result.get("plotersp", True)):
        options["plotersp"] = "off"
    if not bool(result.get("plotitc", True)):
        options["plotitc"] = "off"
    if not bool(result.get("plotphase", False)):
        options["plotphase"] = "off"
    if bool(result.get("plotcurve", False)):
        options["plottype"] = "curve"
    _add_popup_options(
        options,
        int(result.get("ntimesout", 4) or 4),
        int(result.get("nfreqs", 1) or 1),
        int(result.get("basenorm", 1) or 1),
        freqs,
    )
    if freqs == [] and "winsize" not in options and float(EEG.get("xmin", 0) or 0) < 0:
        options["winsize"] = max(2, int(round(-float(EEG.get("xmin", 0)) * float(EEG.get("srate", 1) or 1))))
    cycles = [0] if bool(result.get("fft", False)) else numeric_vector(result.get("cycles", "3 0.8")).tolist()
    return {
        "num": _first_index(result.get("num", 1)),
        "tlimits": numeric_vector(result.get("tlimits", [])).tolist(),
        "cycles": cycles,
        "options": options,
    }


def _selected_signal(EEG: dict[str, Any], typeproc: int, num: Any, tlimits: Any) -> tuple[np.ndarray, np.ndarray]:
    data, times = data_time_slice(EEG, tlimits)
    index = int(_first_index(num)) - 1
    if typeproc == 1:
        if index < 0 or index >= data.shape[0]:
            raise ValueError(f"channel number must be 1-based and within 1..{data.shape[0]}")
        return data[index, :, :], times
    acts = component_activations(EEG)
    full_times = np.linspace(float(EEG.get("xmin", 0)) * 1000.0, float(EEG.get("xmax", 0)) * 1000.0, acts.shape[1])
    bounds = numeric_vector(tlimits)
    if bounds.size == 2:
        mask = (full_times >= bounds[0]) & (full_times <= bounds[1])
        acts = acts[:, mask, :]
        full_times = full_times[mask]
    if index < 0 or index >= acts.shape[0]:
        raise ValueError(f"component number must be 1-based and within 1..{acts.shape[0]}")
    return acts[index, :, :], full_times


def _topo_options(EEG: dict[str, Any], typeproc: int, num: Any) -> dict[str, Any]:
    """Scalp-map inset options (topovec/elocs/caption) when channel locations exist.

    Mirrors EEGLAB ``pop_newtimef``: a channel passes its 1-based index (marked on a
    blank head) and label; a component passes its ``icawinv`` map column and ``IC n``.
    """
    chanlocs = chanlocs_as_list(EEG.get("chanlocs", []))
    if not chanlocs or not _has_theta(chanlocs[0]):
        return {}
    index = int(_first_index(num)) - 1
    if typeproc == 1:
        if not (0 <= index < len(chanlocs)) or not _has_theta(chanlocs[index]):
            return {}
        labels = channel_labels(EEG)
        caption = labels[index] if index < len(labels) else f"Channel {index + 1}"
        return {"topovec": index + 1, "elocs": chanlocs, "caption": caption}
    if EEG.get("icawinv") is None:
        return {}
    maps, map_chanlocs = component_map_data(EEG)
    if not (0 <= index < maps.shape[1]):
        return {}
    return {"topovec": maps[:, index], "elocs": map_chanlocs, "caption": f"IC {index + 1}"}


def _has_theta(chanloc: dict[str, Any]) -> bool:
    theta = chanloc.get("theta")
    if theta is None:
        return False
    try:
        return bool(np.isfinite(float(theta)))
    except (TypeError, ValueError):
        return False


def _add_popup_options(options: dict[str, Any], ntimesout: int, nfreqs: int, basenorm: int, freqs: list[float]) -> None:
    ntimes_values = {1: 50, 2: 100, 3: 150, 4: 200, 5: 300, 6: 400}
    if ntimesout in ntimes_values:
        options["timesout"] = ntimes_values[ntimesout]
    padratio_values = {1: 1, 2: 2, 3: 4}
    if nfreqs in padratio_values:
        options["padratio"] = padratio_values[nfreqs]
    elif nfreqs == 4 and freqs:
        options["nfreqs"] = len(freqs)
    if basenorm in {2, 4}:
        options["basenorm"] = "on"
    if basenorm >= 3:
        options["trialbase"] = "full"


def _first_index(value: Any) -> int:
    values = numeric_vector(value, dtype=int)
    if values.size != 1:
        raise ValueError("pop_newtimef requires exactly one channel/component number")
    return int(values[0])


_NTIMES_CHOICES = (
    "Use 50 time points",
    "Use 100 time points",
    "Use 150 time points",
    "Use 200 time points",
    "Use 300 time points",
    "Use 400 time points",
)
_NFREQS_CHOICES = (
    "Use limits, padding 1",
    "Use limits, padding 2",
    "Use limits, padding 4",
    "Use actual freqs.",
)
_BASELINE_CHOICES = (
    "Use divisive baseline (DIV)",
    "Use standard deviation (STD)",
    "Use single trial DIV baseline",
    "Use single trial STD baseline",
)


__all__ = ["pop_newtimef", "pop_newtimef_dialog_spec"]
