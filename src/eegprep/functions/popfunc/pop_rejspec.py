"""Reject epochs by spectral thresholding."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._eegplot_rejection import run_epoched_mark_rejection
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.popfunc._rejection import (
    parse_numeric_sequence,
    spectrum_marks,
)


def pop_rejspec(
    EEG: dict[str, Any] | list[dict[str, Any]],
    icacomp: int | bool = 1,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    command_callback: Any | None = None,
    show: bool = True,
    **kwargs: Any,
):
    """Mark or reject epochs by spectral power thresholds."""
    if EEG is None:
        return (None, "") if return_com else (None, [])
    options = _options_from_args(args, kwargs)
    if gui is None:
        gui = not options
    if isinstance(EEG, list):
        if gui:
            gui_options = _run_gui(EEG[0], int(bool(icacomp)), renderer=renderer)
            if gui_options is None:
                return (EEG, "") if return_com else (EEG, [])
            options.update(gui_options)
        outputs = [_apply_one(dataset, icacomp, options, display=False)[0] for dataset in EEG]
        command = _history_command(icacomp, options)
        return (outputs, command) if return_com else (outputs, [])
    if gui:
        gui_options = _run_gui(EEG, int(bool(icacomp)), renderer=renderer)
        if gui_options is None:
            return (EEG, "") if return_com else (EEG, [])
        options.update(gui_options)
    out, rejected, command = _apply_one(
        EEG,
        icacomp,
        options,
        display=bool(gui or _int_option(options.get("eegplotplotallrej", 0)) == 2),
        command_callback=command_callback,
        show=show,
    )
    return (out, command) if return_com else (out, rejected)


def pop_rejspec_dialog_spec(EEG: dict[str, Any], icacomp: int | bool = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_rejspec``."""
    is_data = int(bool(icacomp))
    rows = int(EEG.get("nbchan", 0) or 0) if is_data else int(np.asarray(EEG.get("icaweights", [])).shape[0])
    label = "Electrode indices (Ex: 2 4 6:8 10)" if is_data else "Component indices (Ex: 2 4 6:8 10)"
    return DialogSpec(
        title="Reject by data spectra -- pop_rejspec()",
        function_name="pop_rejspec",
        eeglab_source="functions/popfunc/pop_rejspec.m",
        help_text="pophelp('pop_rejspec')",
        size=(620, 356),
        geometry=(
            (1, 0.1, 0.75),
            (1, 0.26, 0.9),
            (1, 0.1, 0.75),
            (1, 0.1, 0.75),
            (1, 0.1, 0.75),
            (1, 0.1, 0.75),
            (1,),
            (1, 0.22, 0.85),
            (1, 0.22, 0.85),
        ),
        controls=(
            ControlSpec("text", label),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="elecrange", value=f"1:{rows}"),
            ControlSpec("text", "Spectrum computation method"),
            ControlSpec("spacer"),
            ControlSpec("popupmenu", "FFT|Multitaper", tag="method", value=2),
            ControlSpec("text", "Minimum power rejection threshold(s) (dB)"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="lowlim", value="-30"),
            ControlSpec("text", "Maximum power rejection threshold(s) (dB)"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="highlim", value="30"),
            ControlSpec("text", "Low frequency limit(s) (Hz)"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="lowfreq", value="15"),
            ControlSpec("text", "High frequency limit(s) (Hz)"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="highfreq", value="30"),
            ControlSpec("spacer"),
            ControlSpec("text", "Display previous rejection marks"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", tag="superpose", value=False),
            ControlSpec("text", "Reject marked trial(s)"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", tag="reject", value=False),
        ),
    )


def _options_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if args and not isinstance(args[0], str):
        values = list(args)
        options: dict[str, Any] = {}
        if len(values) > 0:
            options["elecrange"] = values[0]
        if len(values) > 2:
            options["threshold"] = [[values[1], values[2]]]
        if len(values) > 4:
            options["freqlimits"] = [[values[3], values[4]]]
        if len(values) > 5:
            options["eegplotplotallrej"] = values[5]
        if len(values) > 6:
            options["eegplotreject"] = values[6]
        return {**options, **kwargs}
    return parse_key_value_args(args, kwargs, lowercase_kwargs=True)


def _run_gui(EEG: dict[str, Any], icacomp: int, renderer: Any | None) -> dict[str, Any] | None:
    result = inputgui(pop_rejspec_dialog_spec(EEG, icacomp), renderer=renderer)
    if result is None:
        return None
    methods = ("fft", "multitaper")
    method_value = result.get("method", 1)
    method = methods[int(method_value) - 1] if isinstance(method_value, int) else str(method_value).lower()
    return {
        "elecrange": result.get("elecrange", ""),
        "method": method,
        "threshold": list(
            zip(
                parse_numeric_sequence(result.get("lowlim", "-30"), dtype=float),
                parse_numeric_sequence(result.get("highlim", "30"), dtype=float),
            )
        ),
        "freqlimits": list(
            zip(
                parse_numeric_sequence(result.get("lowfreq", "15"), dtype=float),
                parse_numeric_sequence(result.get("highfreq", "30"), dtype=float),
            )
        ),
        "eegplotplotallrej": int(bool(result.get("superpose", False))),
        "eegplotreject": int(bool(result.get("reject", False))),
    }


def _apply_one(
    EEG: dict[str, Any],
    icacomp: int | bool,
    options: dict[str, Any],
    *,
    display: bool = False,
    command_callback: Any | None = None,
    show: bool = True,
) -> tuple[dict[str, Any], list[int], str]:
    def _marks(out: dict[str, Any], data: np.ndarray, elecrange: list[int]):
        threshold = options.get("threshold", [-30, 30])
        freqlimits = options.get("freqlimits", [15, 30])
        method = str(options.get("method", "multitaper")).lower()
        marks, marks_e, spectra = spectrum_marks(
            data, elecrange, float(out.get("srate", 1.0)), threshold, freqlimits, method
        )
        if int(bool(icacomp)):
            out["specdata"] = spectra
        else:
            out["specicaact"] = spectra
        normalized_options = dict(options)
        normalized_options["elecrange"] = elecrange
        normalized_options["method"] = method
        normalized_options.setdefault("threshold", threshold)
        normalized_options.setdefault("freqlimits", freqlimits)
        normalized_options.setdefault("eegplotplotallrej", 0)
        normalized_options.setdefault("eegplotreject", 0)
        return marks, marks_e, normalized_options

    def _command(_elecrange: list[int], normalized_options: dict[str, Any]) -> str:
        return _history_command(icacomp, normalized_options)

    normalized_reject = int(bool(_int_option(options.get("eegplotreject", 0))))
    out, rejected, command, _normalized_options = run_epoched_mark_rejection(
        EEG,
        icacomp,
        options.get("elecrange"),
        _int_option(options.get("eegplotplotallrej", 0)),
        normalized_reject,
        marks_fn=_marks,
        kind="rejfreq",
        error_message="pop_rejspec requires epoched data",
        command_fn=_command,
        display=display,
        command_callback=command_callback,
        show=show,
    )
    return out, rejected, command


def _history_command(icacomp: int | bool, options: dict[str, Any]) -> str:
    values: list[Any] = [
        int(bool(icacomp)),
        "elecrange",
        options.get("elecrange", []),
        "method",
        options.get("method", "multitaper"),
        "threshold",
        options.get("threshold", [-30, 30]),
        "freqlimits",
        options.get("freqlimits", [15, 30]),
        "eegplotplotallrej",
        _int_option(options.get("eegplotplotallrej", 0)),
        "eegplotreject",
        int(bool(_int_option(options.get("eegplotreject", 0)))),
    ]
    return (
        "EEG = pop_rejspec(EEG, "
        + ", ".join(format_history_value(item, cell_for_sequence=None) for item in values)
        + ");"
    )


def _int_option(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"on", "yes", "true"}:
            return 1
        if lowered in {"off", "no", "false", ""}:
            return 0
    return int(value)
