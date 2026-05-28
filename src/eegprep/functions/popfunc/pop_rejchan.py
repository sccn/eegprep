"""Reject bad channels using EEGLAB-style summary measures."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
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
            ControlSpec("popupmenu", "Probability|Kurtosis|Spectrum|Standard deviation", tag="measure", value=2),
            ControlSpec("text", "Normalize measure (check=on)"),
            ControlSpec("checkbox", tag="norm", value=True),
            ControlSpec("spacer"),
            ControlSpec("text", "Z-score threshold [max] or [min max]"),
            ControlSpec("edit", tag="threshold", value="5"),
            ControlSpec("text", "Spectrum freq. range"),
            ControlSpec("edit", tag="freqrange", value=f"1 {float(EEG.get('srate', 1)) / 2:g}"),
        ),
    )


def _run_gui(EEG: dict[str, Any], renderer: Any | None) -> dict[str, Any] | None:
    result = inputgui(pop_rejchan_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    measures = ("prob", "kurt", "spec", "std")
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
    threshold = parse_numeric_sequence(options.get("threshold", 5), dtype=float)
    measure_name = str(options.get("measure", "kurt")).lower()
    norm = str(options.get("norm", "off")).lower() == "on"
    measure = _channel_measure(flat[selected], measure_name, threshold, norm, out, options)
    rejected_mask = _threshold_measure(measure, threshold)
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
) -> np.ndarray:
    if measure_name == "prob":
        scores, _rej = jointprob(flat, threshold, options.get("precomp", []), 2 if norm else 0)
        return scores.ravel()
    if measure_name == "kurt":
        scores, _rej = rejkurt(flat, threshold, options.get("precomp", []), 2 if norm else 0)
        return scores.ravel()
    if measure_name == "std":
        return flat.std(axis=1)
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
        return power.max(axis=1)
    raise ValueError("measure must be one of 'prob', 'kurt', 'spec', or 'std'")


def _threshold_measure(measure: np.ndarray, threshold: list[float]) -> np.ndarray:
    if not threshold:
        threshold = [400.0]
    if len(threshold) > 1:
        return (measure < threshold[0]) | (measure > threshold[-1])
    return measure > threshold[0]


def _history_command(options: dict[str, Any]) -> str:
    values: list[Any] = []
    for key in ("elec", "threshold", "measure", "norm", "freqrange", "indexonly"):
        if key in options:
            values.extend([key, options[key]])
    return (
        "EEG = pop_rejchan(EEG, "
        + ", ".join(format_history_value(item, cell_for_sequence=None) for item in values)
        + ");"
    )
