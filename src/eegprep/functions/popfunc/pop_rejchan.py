"""Reject bad channels using EEGLAB-style summary measures."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc._rejection import copy_eeg, jointprob, one_based_indices, parse_numeric_sequence, rejkurt
from eegprep.functions.popfunc.pop_select import pop_select


def pop_rejchan(
    EEG: dict[str, Any],
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Reject channels by probability, kurtosis, spectrum, or standard deviation."""
    if EEG is None:
        return (None, "") if return_com else (None, [], [])
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = not options
    if gui:
        gui_options = _run_gui(EEG, renderer=renderer)
        if gui_options is None:
            return (EEG, "") if return_com else (EEG, [], [])
        options.update(gui_options)
    out, indices, measure, command = _apply_one(EEG, options)
    return (out, command) if return_com else (out, indices, measure)


def pop_rejchan_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_rejchan``."""
    return DialogSpec(
        title="Reject channel -- pop_rejchan()",
        function_name="pop_rejchan",
        eeglab_source="functions/popfunc/pop_rejchan.m",
        help_text="pophelp('pop_rejchan')",
        size=(520, 282),
        geometry=((2, 1.3), (2, 1.3), (2, 0.4, 0.9), (2, 1.3), (2, 1.3)),
        controls=(
            ControlSpec("text", "Electrode (number(s); Ex: 2 4 5)"),
            ControlSpec("edit", tag="elec", value=f"1:{int(EEG.get('nbchan', 0) or 0)}"),
            ControlSpec("text", "Measure to use"),
            ControlSpec(
                "popupmenu",
                "Probability|Kurtosis|Spectrum",
                tag="measure",
                value=2,
                callback=CallbackSpec(
                    "toggle_index_enabled",
                    {"source": "measure", "targets": ["freq_label", "freqrange"], "enabled_index": 3},
                ),
            ),
            ControlSpec("text", "Normalize measure (check=on)"),
            ControlSpec(
                "checkbox",
                tag="norm",
                value=True,
                callback=CallbackSpec(
                    "toggle_checked_text",
                    {
                        "source": "norm",
                        "target": "normlab",
                        "checked": "Z-score threshold [max] or [min max]",
                        "unchecked": "Absolute threshold [max] or [min max]",
                    },
                ),
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "Z-score threshold [max] or [min max]", tag="normlab"),
            ControlSpec("edit", tag="threshold", value="5"),
            ControlSpec("text", "Spectrum freq. range", tag="freq_label", enabled=False),
            ControlSpec(
                "edit",
                tag="freqrange",
                value=f"1 {float(EEG.get('srate', 1)) / 2:g}",
                enabled=False,
            ),
        ),
    )


def _run_gui(EEG: dict[str, Any], renderer: Any | None) -> dict[str, Any] | None:
    result = inputgui(pop_rejchan_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    measures = ("prob", "kurt", "spec")
    value = result.get("measure", 2)
    measure = measures[int(value) - 1] if isinstance(value, int) else str(value).lower()
    options = {
        "elec": result.get("elec", ""),
        "measure": measure,
        "norm": "on" if result.get("norm", False) else "off",
        "threshold": result.get("threshold", "5"),
    }
    if measure == "spec":
        options["freqrange"] = result.get("freqrange", "")
    return options


def _apply_one(EEG: dict[str, Any], options: dict[str, Any]) -> tuple[dict[str, Any], list[int], np.ndarray, str]:
    out = copy_eeg(EEG)
    data = np.asarray(out.get("data"), dtype=float)
    if data.ndim == 3:
        flat = data.reshape(data.shape[0], -1, order="F")
    elif data.ndim == 2:
        flat = data
    else:
        raise ValueError("EEG data must be 2-D or 3-D")
    elec = one_based_indices(options.get("elec"), limit=int(out.get("nbchan", flat.shape[0])), default_all=True)
    selected = [item - 1 for item in elec]
    threshold = parse_numeric_sequence(options.get("threshold", 400), dtype=float)
    measure_name = str(options.get("measure", "kurt")).lower()
    norm = str(options.get("norm", "off")).lower() == "on"
    measure, rejected_mask = _channel_measure(flat[selected], measure_name, threshold, norm, out, options)
    rejected = [elec[index] for index, flag in enumerate(rejected_mask) if flag]
    if rejected and str(options.get("indexonly", "off")).lower() != "on":
        out = pop_select(out, "nochannel", [index - 1 for index in rejected], gui=False)
    command = _history_command(options)
    return out, rejected, measure, command


def _channel_measure(
    flat: np.ndarray,
    measure_name: str,
    threshold: list[float],
    norm: bool,
    EEG: dict[str, Any],
    options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if measure_name == "prob":
        scores, rejected = jointprob(flat, threshold, options.get("precomp", []), 2 if norm else 0)
        return scores.ravel(), rejected.ravel()
    if measure_name == "kurt":
        scores, rejected = rejkurt(flat, threshold, options.get("precomp", []), 2 if norm else 0)
        return scores.ravel(), rejected.ravel()
    if measure_name == "std":
        measure = flat.std(axis=1)
        return measure, _threshold_measure(measure, threshold)
    if measure_name == "spec":
        spectra = np.abs(np.fft.rfft(flat - flat.mean(axis=1, keepdims=True), axis=1)) ** 2
        freqs = np.fft.rfftfreq(flat.shape[1], d=1 / float(EEG.get("srate", 1)))
        freqrange = parse_numeric_sequence(options.get("freqrange", [1, float(EEG.get("srate", 1)) / 2]), dtype=float)
        if len(freqrange) < 2:
            freqrange = [1, float(EEG.get("srate", 1)) / 2]
        mask = (freqs >= freqrange[0]) & (freqs <= freqrange[-1])
        power = 10 * np.log10(np.maximum(spectra[:, mask], np.finfo(float).tiny))
        if norm:
            mean = power.mean(axis=0, keepdims=True)
            std = power.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            power = (power - mean) / std
        measure = power.max(axis=1)
        return measure, _threshold_measure(measure, threshold)
    raise ValueError(
        "pop_rejchan: measure must be 'prob', 'kurt', 'spec', or 'std'. "
        "The dialog offers Probability, Kurtosis, and Spectrum; pass measure='std' to reject by standard deviation."
    )


def _threshold_measure(measure: np.ndarray, threshold: list[float]) -> np.ndarray:
    if not threshold:
        threshold = [400.0]
    if len(threshold) > 1:
        return (measure < threshold[0]) | (measure > threshold[-1])
    return measure > threshold[0]


def _history_command(options: dict[str, Any]) -> str:
    # pop_rejchan.m:246
    parts = []
    for key in ("elec", "threshold", "norm", "measure", "freqrange", "indexonly"):
        if key in options:
            parts.append(f"'{key}',{_matlab_history_value(options[key])}")
    return "EEG = pop_rejchan(EEG, " + ",".join(parts) + ");"


def _matlab_history_value(value: Any) -> str:
    if isinstance(value, str) and not _looks_numeric(value):
        return f"'{value}'"
    values = parse_numeric_sequence(value, dtype=float)
    if not values:
        return "[]"
    if len(values) == 1:
        return _matlab_number(values[0])
    return "[" + _matlab_contarray(values) + "] "


def _looks_numeric(value: str) -> bool:
    token = value.strip().strip("[]")
    if not token:
        return True
    return all(character.isdigit() or character in ".:-+eE,; " for character in token)


def _matlab_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _matlab_contarray(values: list[float]) -> str:
    # vararg2str.m:196-242
    array = [float(item) for item in values]
    if len(array) == 2:
        return f"{_matlab_number(array[0])} {_matlab_number(array[1])}"
    if any(not item.is_integer() for item in array):
        return " ".join(_matlab_number(item) for item in array)
    rendered = _matlab_number(array[0])
    skip = 0
    indent = array[1] - array[0]
    for index in range(1, len(array)):
        if array[index] != array[index - 1] + indent or indent == 0:
            if skip <= 1:
                if skip == 0:
                    rendered = f"{rendered} {_matlab_number(array[index])}"
                else:
                    rendered = f"{rendered} {_matlab_number(array[index - 1])} {_matlab_number(array[index])}"
            elif indent == 1:
                rendered = f"{rendered}:{_matlab_number(array[index - 1])} {_matlab_number(array[index])}"
            else:
                rendered = (
                    f"{rendered}:{_matlab_number(indent)}:{_matlab_number(array[index - 1])} "
                    f"{_matlab_number(array[index])}"
                )
            skip = 0
            indent = array[index] - array[index - 1]
        else:
            skip += 1
    if array[-1] == array[-2] + indent and skip != 0:
        if indent == 1:
            rendered = f"{rendered}:{_matlab_number(array[-1])}"
        elif indent == 0:
            rendered = f"{rendered} {_matlab_number(array[-1])}"
        else:
            rendered = f"{rendered}:{_matlab_number(indent)}:{_matlab_number(array[-1])}"
    return rendered
