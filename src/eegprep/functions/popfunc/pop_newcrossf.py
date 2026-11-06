"""EEGLAB-style wrapper for channel/component cross-coherence plots."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc.plot_utils import (
    component_activations,
    eeg_epoch_data,
    history_command,
    numeric_vector,
    parse_plot_options_text,
    show_figures,
)
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.timefreqfunc.newcrossf import newcrossf


def pop_newcrossf(
    EEG: dict[str, Any] | None = None,
    typeproc: int = 1,
    num1: Any = None,
    num2: Any = None,
    tlimits: Any = None,
    cycles: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: str | bool = "on",
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot event-related channel/component cross-coherence.

    Pass ``plot='off'`` to build and return the figure without opening a window.
    """
    if EEG is None:
        return (None, "") if return_com else None
    typeproc = int(typeproc)
    if gui is None:
        gui = num1 is None or num2 is None
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui:
        result = _run_gui(EEG, typeproc=typeproc, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        num1 = result["num1"]
        num2 = result["num2"]
        tlimits = result["tlimits"]
        cycles = result["cycles"]
        options.update(result["options"])
    if num1 is None:
        num1 = 1
    if num2 is None:
        num2 = 2
    if tlimits is None:
        tlimits = [float(EEG.get("xmin", 0)) * 1000.0, float(EEG.get("xmax", 0)) * 1000.0]
    if cycles is None:
        cycles = [3, 0.5]
    signal1, signal2, tlimits_ms = _selected_signals(EEG, typeproc, num1, num2, tlimits)
    result = newcrossf(
        signal1, signal2, signal1.shape[0], tlimits_ms, float(EEG.get("srate", 1) or 1), cycles, **options
    )
    command = history_command(
        "pop_newcrossf", typeproc, _first_index(num1), _first_index(num2), tlimits, cycles, **options
    )
    show_figures(result.figure, plot=plot)
    return (result, command) if return_com else result


def pop_newcrossf_dialog_spec(EEG: dict[str, Any], *, typeproc: int = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_newcrossf``."""
    is_channel = bool(int(typeproc))
    unit = "channel" if is_channel else "component"
    controls = [
        ControlSpec("text", f"First {unit} number", font_weight="bold"),
        ControlSpec("edit", tag="num1", value="1"),
        ControlSpec("spacer"),
        ControlSpec("text", f"Second {unit} number", font_weight="bold"),
        ControlSpec("edit", tag="num2", value="2"),
        ControlSpec("spacer"),
        ControlSpec("text", "Epoch time range [min max] (msec)", font_weight="bold"),
        ControlSpec(
            "edit",
            tag="tlimits",
            value=f"{_dialog_msec(EEG.get('xmin', 0))} {_dialog_msec(EEG.get('xmax', 0))}",
        ),
        ControlSpec("spacer"),
        ControlSpec("text", "Wavelet cycles (0->FFT, see Help)", font_weight="bold"),
        ControlSpec("edit", tag="cycles", value="3 0.5"),
        ControlSpec("spacer"),
        ControlSpec("text", "[set]->log. scale for frequencies (match STUDY)", font_weight="bold"),
        ControlSpec("checkbox", tag="logscale", value=False),
        ControlSpec("spacer"),
        ControlSpec("text", "[set]->Linear coher / [unset]->Phase coher", font_weight="bold"),
        ControlSpec("checkbox", tag="coher", value=False),
        ControlSpec("spacer"),
        ControlSpec("text", "Bootstrap significance level (Ex: 0.01 -> 1%)", font_weight="bold"),
        ControlSpec("edit", tag="alpha", value=""),
        ControlSpec("spacer"),
        ControlSpec("text", "Optional newcrossf() arguments (see Help)", font_weight="bold"),
        ControlSpec("edit", tag="options", value="'padratio', 1"),
        ControlSpec("spacer"),
        ControlSpec("checkbox", "Plot coherence amplitude", tag="plotamp", value=True),
        ControlSpec("checkbox", "Plot coherence phase", tag="plotphase", value=True),
    ]
    return DialogSpec(
        title=(
            "Plot channel cross-coherence -- pop_newcrossf()"
            if is_channel
            else "Plot component cross-coherence -- pop_newcrossf()"
        ),
        controls=tuple(controls),
        geometry=((1, 0.5, 0.5),) * 4 + ((0.92, 0.15, 0.73),) * 2 + ((1, 0.5, 0.5), (1, 0.8, 0.2), (1,), (1, 1)),
        function_name="pop_newcrossf",
        eeglab_source="functions/popfunc/pop_newcrossf.m",
        help_text="pophelp('pop_newcrossf')",
        size=(908, 476),
        row_spacing=4,
        extra_stylesheet="""
            QDialog#pop_newcrossf QLabel,
            QDialog#pop_newcrossf QCheckBox,
            QDialog#pop_newcrossf QLineEdit,
            QDialog#pop_newcrossf QPushButton {
                font-size: 11px;
            }
            QDialog#pop_newcrossf QLineEdit,
            QDialog#pop_newcrossf QPushButton {
                min-height: 15px;
                max-height: 15px;
            }
            QDialog#pop_newcrossf QCheckBox::indicator {
                width: 11px;
                height: 11px;
            }
        """,
    )


def _run_gui(EEG: dict[str, Any], *, typeproc: int, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_newcrossf_dialog_spec(EEG, typeproc=typeproc), renderer=renderer)
    if result is None:
        return None
    options = parse_plot_options_text(result.get("options", ""))
    options["type"] = "coher" if bool(result.get("coher", False)) else "phasecoher"
    if bool(result.get("logscale", False)):
        options["freqscale"] = "log"
    alpha = numeric_vector(result.get("alpha", []))
    if alpha.size:
        options["alpha"] = float(alpha[0])
    if not bool(result.get("plotamp", True)):
        options["plotamp"] = "off"
    if not bool(result.get("plotphase", True)):
        options["plotphase"] = "off"
    return {
        "num1": _first_index(result.get("num1", 1)),
        "num2": _first_index(result.get("num2", 2)),
        "tlimits": numeric_vector(result.get("tlimits", [])).tolist(),
        "cycles": numeric_vector(result.get("cycles", "3 0.5")).tolist(),
        "options": options,
    }


def _dialog_msec(seconds: Any) -> str:
    """pop_newcrossf.m:87 uses int2str, which truncates toward zero."""
    return str(int(np.trunc(float(seconds) * 1000.0)))


def _requested_tlimits(EEG: dict[str, Any], tlimits: Any) -> np.ndarray:
    bounds = numeric_vector(tlimits)
    if bounds.size == 2:
        return np.asarray([float(bounds[0]), float(bounds[1])], dtype=float)
    return np.asarray(
        [float(EEG.get("xmin", 0) or 0) * 1000.0, float(EEG.get("xmax", 0) or 0) * 1000.0],
        dtype=float,
    )


def _pointrange(EEG: dict[str, Any], tlimits: Any) -> slice:
    # pop_newcrossf.m:189 does not add 1 before rounding the last sample.
    pnts = int(EEG.get("pnts") or np.asarray(EEG["data"]).shape[1])
    bounds = numeric_vector(tlimits)
    if bounds.size != 2:
        return slice(0, pnts)
    xmin = float(EEG.get("xmin", 0.0) or 0.0)
    srate = float(EEG.get("srate", 1.0) or 1.0)
    first = int(np.round(max((float(bounds[0]) / 1000.0 - xmin) * srate, 1.0)))
    last = int(np.round(min((float(bounds[1]) / 1000.0 - xmin) * srate, float(pnts))))
    if last < first:
        raise ValueError(
            f"pop_newcrossf: time limits [{float(bounds[0]):g} {float(bounds[1]):g}] ms contain no samples "
            f"at {srate:g} Hz (xmin {xmin:g} s, {pnts} points). "
            "Widen the limits so they cover at least one sample."
        )
    return slice(first - 1, last)


def _selected_signals(
    EEG: dict[str, Any], typeproc: int, num1: Any, num2: Any, tlimits: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = _pointrange(EEG, tlimits)
    requested = _requested_tlimits(EEG, tlimits)
    first = int(_first_index(num1)) - 1
    second = int(_first_index(num2)) - 1
    if typeproc == 1:
        data = eeg_epoch_data(EEG)
        nchan = data.shape[0]
        if first < 0 or second < 0 or first >= nchan or second >= nchan:
            raise ValueError(
                f"pop_newcrossf: channel numbers must be from 1 to {nchan}. Channel 1 is the first channel."
            )
        return data[first, window, :], data[second, window, :], requested
    acts = component_activations(EEG)
    ncomp = acts.shape[0]
    if first < 0 or second < 0 or first >= ncomp or second >= ncomp:
        raise ValueError(
            f"pop_newcrossf: component numbers must be from 1 to {ncomp}. Component 1 is the first component."
        )
    return acts[first, window, :], acts[second, window, :], requested


def _first_index(value: Any) -> int:
    values = numeric_vector(value, dtype=int)
    if values.size != 1:
        raise ValueError("pop_newcrossf requires exactly one channel/component number")
    return int(values[0])


__all__ = ["pop_newcrossf", "pop_newcrossf_dialog_spec"]
