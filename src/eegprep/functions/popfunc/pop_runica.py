"""EEGLAB-style pop wrapper for ICA decomposition."""

from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc.eeg_decodechan import eeg_decodechan
from eegprep.functions.popfunc.eeg_runica import eeg_runica


_ALGORITHMS = (
    ("runica", "Extended Infomax (runica.m; default)", "'extended', 1"),
    ("runica", "Robust Extended Infomax (runica.m; slow)", "'extended', 1, 'lrate', 1e-5, 'maxsteps', 2000"),
)


def pop_runica(
    EEG,
    *args,
    icatype: str = "runica",
    options: Any = None,
    reorder: str | bool = "on",
    chanind: Any = None,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs,
):
    """Run ICA decomposition with EEGLAB ``pop_runica`` calling semantics."""
    parsed = _parse_key_value_args(args, kwargs)
    icatype = parsed.pop("icatype", icatype)
    options = parsed.pop("options", options)
    reorder = parsed.pop("reorder", reorder)
    chanind = parsed.pop("chanind", chanind)
    parsed.pop("dataset", None)
    parsed.pop("concatenate", None)
    parsed.pop("concatcond", None)

    if gui is None:
        gui = options is None and not parsed and chanind is None
    if gui:
        gui_result = _run_gui(EEG[0] if isinstance(EEG, list) else EEG, renderer=renderer)
        if gui_result is None:
            return (EEG, "") if return_com else EEG
        icatype = gui_result["icatype"]
        options = gui_result["options"]
        reorder = gui_result["reorder"]
        chanind = gui_result["chanind"]

    runica_options = _normalise_runica_options(options, parsed)
    if str(icatype).lower() != "runica":
        raise NotImplementedError("EEGPrep pop_runica currently supports icatype='runica'")

    if isinstance(EEG, list):
        output = [
            pop_runica(
                item,
                icatype=icatype,
                options=runica_options,
                reorder=reorder,
                chanind=chanind,
                gui=False,
            )
            for item in EEG
        ]
        command = _history_command(icatype, runica_options, reorder, chanind)
        return (output, command) if return_com else output

    output = _runica_on_dataset(EEG, runica_options, reorder=reorder, chanind=chanind)
    command = _history_command(icatype, runica_options, reorder, chanind)
    return (output, command) if return_com else output


def pop_runica_dialog_spec(EEG) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_runica``."""
    chanlocs = list(EEG.get("chanlocs", []) or [])
    labels = tuple(str(chan.get("labels", "")) for chan in chanlocs if isinstance(chan, dict))
    types = tuple(
        value for value in dict.fromkeys(
            str(chan.get("type", "")) for chan in chanlocs if isinstance(chan, dict) and chan.get("type", "")
        )
    )
    algorithm_labels = "|".join(description for _name, description, _options in _ALGORITHMS)
    return DialogSpec(
        title="Run ICA decomposition -- pop_runica()",
        function_name="pop_runica",
        eeglab_source="functions/popfunc/pop_runica.m",
        geometry=((2, 1.5), (2, 1.5), (1,), (2, 1, 1, 1)),
        size=(824, 334),
        help_text="pophelp('pop_runica')",
        controls=(
            ControlSpec("text", "ICA algorithm to use (click to select)"),
            ControlSpec("listbox", algorithm_labels, tag="icatype", value=1),
            ControlSpec("text", "Commandline options (See help messages)"),
            ControlSpec("edit", tag="params", value=_ALGORITHMS[0][2]),
            ControlSpec("checkbox", "Reorder components by variance (if that's not already the case)", tag="reorder", value=True),
            ControlSpec("text", "Use only channel type(s) or indices"),
            ControlSpec("edit", tag="chantype", value=""),
            ControlSpec(
                "pushbutton",
                "... types",
                tag="type_button",
                enabled=bool(types),
                callback=CallbackSpec(
                    "select_channels",
                    params={"button": "type_button", "target": "chantype", "channels": types},
                    matlab_callback="pop_chansel({tmpchanlocs.type}, 'withindex', 'off')",
                ),
            ),
            ControlSpec(
                "pushbutton",
                "... channels",
                tag="chan_button",
                enabled=bool(labels),
                callback=CallbackSpec(
                    "select_channels",
                    params={"button": "chan_button", "target": "chantype", "channels": labels},
                    matlab_callback="pop_chansel({tmpchanlocs.labels}, 'withindex', 'on')",
                ),
            ),
        ),
    )


def _run_gui(EEG, renderer=None):
    spec = pop_runica_dialog_spec(EEG)
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return None
    algorithm_index = int(result.get("icatype", 1)) - 1
    algorithm_index = max(0, min(algorithm_index, len(_ALGORITHMS) - 1))
    chan_text = str(result.get("chantype", "") or "").strip()
    return {
        "icatype": _ALGORITHMS[algorithm_index][0],
        "options": _parse_option_text(str(result.get("params", "") or "")),
        "reorder": "on" if result.get("reorder") else "off",
        "chanind": _parse_channel_text(chan_text, one_based=True) if chan_text else None,
    }


def _runica_on_dataset(EEG, options, *, reorder, chanind):
    output = copy.deepcopy(EEG)
    chanind = _resolve_chanind(output, chanind)
    if chanind is None:
        return eeg_runica(output, sortcomps=reorder, **options)

    subset = copy.deepcopy(output)
    subset["data"] = np.asarray(subset["data"])[chanind, ...]
    subset["nbchan"] = len(chanind)
    if subset.get("chanlocs"):
        subset["chanlocs"] = [subset["chanlocs"][index] for index in chanind]
    subset = eeg_runica(subset, sortcomps=reorder, **options)
    output["icasphere"] = subset["icasphere"]
    output["icaweights"] = subset["icaweights"]
    output["icawinv"] = subset["icawinv"]
    output["icaact"] = subset["icaact"]
    output["icachansind"] = np.asarray(chanind)
    return output


def _resolve_chanind(EEG, chanind):
    if chanind is None:
        return None
    if isinstance(chanind, str):
        if chanind == "":
            return None
        chanind = _parse_channel_text(chanind)
    if isinstance(chanind, np.ndarray):
        chanind = chanind.tolist()
    values = list(chanind)
    if not values:
        return None
    if all(isinstance(value, str) and not _is_int_text(value) for value in values):
        indices, _labels = eeg_decodechan(EEG, values, "labels", True)
        if len(indices) != len(values):
            indices, _labels = eeg_decodechan(EEG, values, "type", True)
        return list(indices)
    return [int(value) for value in values]


def _parse_key_value_args(args, kwargs):
    if len(args) % 2:
        raise ValueError("Key/value arguments must be in pairs")
    options = {str(key).lower(): value for key, value in kwargs.items()}
    for index in range(0, len(args), 2):
        key = args[index]
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if not isinstance(key, str):
            raise ValueError("Keys must be strings")
        options[key.lower()] = args[index + 1]
    return options


def _normalise_runica_options(options, extra):
    parsed = dict(extra)
    if options is None:
        parsed.setdefault("extended", 1)
        return parsed
    if isinstance(options, dict):
        parsed.update(options)
        return parsed
    if isinstance(options, str):
        parsed.update(_parse_option_text(options))
        return parsed
    if isinstance(options, (list, tuple)):
        parsed.update(_pairs_to_dict(options))
        return parsed
    raise TypeError("options must be a string, dict, list, tuple, or None")


def _parse_option_text(text):
    text = text.strip()
    if not text:
        return {}
    tokens = re.findall(r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)", text)
    return _pairs_to_dict([_parse_scalar(next(part for part in token if part)) for token in tokens])


def _pairs_to_dict(values):
    if len(values) % 2:
        raise ValueError("ICA options must be key/value pairs")
    return {str(values[index]).lower(): values[index + 1] for index in range(0, len(values), 2)}


def _parse_channel_text(text, *, one_based=False):
    values = [_parse_scalar(value) for value in re.split(r"[\s,]+", text.strip().strip("[]")) if value]
    if one_based:
        values = [value - 1 if isinstance(value, int) else value for value in values]
    return values


def _parse_scalar(value):
    value = str(value).strip()
    if _is_int_text(value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _is_int_text(value):
    return bool(re.fullmatch(r"[+-]?\d+", str(value).strip()))


def _history_command(icatype, options, reorder, chanind):
    parts = ["'icatype'", _history_value(str(icatype).lower())]
    for key, value in options.items():
        parts.extend([_history_value(key), _history_value(value)])
    if str(reorder).lower() != "on":
        parts.extend(["'reorder'", _history_value(reorder)])
    if chanind is not None:
        parts.extend(["'chanind'", _history_value(chanind)])
    return f"EEG = pop_runica(EEG, {', '.join(parts)});"


def _history_value(value):
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + " ".join(_history_value(item) for item in value) + "]"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
