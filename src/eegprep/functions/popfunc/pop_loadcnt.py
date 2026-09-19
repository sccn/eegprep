"""Import Neuroscan CNT recordings into EEGPrep datasets."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.adminfunc.storage import MemmapData
from eegprep.functions.miscfunc.value_parsing import parse_key_value_args
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.sigprocfunc.loadcnt import loadcnt


def pop_loadcnt(
    filename: str | Path,
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Load a Neuroscan CNT recording into an EEG dictionary.

    ``loadcnt`` options can be supplied as Python keywords or EEGLAB-style
    key/value pairs. CNT events are exposed with 1-based EEG latencies;
    response-only events are retained when ``keystroke="on"``.
    """
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    keystroke = _toggle(options.pop("keystroke", "off"), "keystroke")
    path = Path(filename).expanduser()
    cnt = loadcnt(path, **options)
    data = cnt["data"]
    header = cnt["header"]
    electrodes = cnt["electloc"]
    eeg = _empty_cnt_eeg(path, data, header, electrodes, cnt["ldnsamples"])
    eeg["event"] = [_eeg_event(event, keystroke=keystroke) for event in cnt["event"]]
    eeg["event"] = [event for event in eeg["event"] if event is not None]
    eeg = eeg_checkset(eeg, "eventconsistency")
    events = [dict(event) for event in eeg["event"]]
    eeg["urevent"] = np.asarray(
        [{key: value for key, value in event.items() if key != "urevent"} for event in events],
        dtype=object,
    )
    for index, event in enumerate(events):
        event["urevent"] = index
    eeg["event"] = np.asarray(events, dtype=object)
    history_options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    command = _history_command(path, history_options)
    eeg["history"] = command
    return (eeg, command) if return_com else eeg


def _empty_cnt_eeg(
    path: Path,
    data: np.ndarray | MemmapData,
    header: dict[str, Any],
    electrodes: list[dict[str, Any]],
    samples: int,
) -> dict[str, Any]:
    rate = float(header["rate"])
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": "CNT file",
            "filename": path.name,
            "filepath": str(path.parent),
            "subject": str(header.get("patient") or ""),
            "comments": f"Original file: {path}",
            "nbchan": int(header["nchannels"]),
            "pnts": int(samples),
            "trials": 1,
            "srate": rate,
            "xmin": 0.0,
            "xmax": (int(samples) - 1) / rate,
            "times": np.arange(int(samples), dtype=float) / rate * 1000,
            "data": data,
            "chanlocs": np.asarray(
                [
                    {
                        "labels": electrode["lab"],
                        "type": "EEG",
                        "urchan": index,
                    }
                    for index, electrode in enumerate(electrodes)
                ],
                dtype=object,
            ),
            "saved": "no",
        }
    )
    if isinstance(data, MemmapData):
        eeg["datfile"] = data.path.name
    return eeg


def _eeg_event(event: dict[str, Any], *, keystroke: bool) -> dict[str, Any] | None:
    stimulus = int(event["stimtype"])
    accept_code = int(event["accept_ev1"])
    if accept_code in {11, 14}:
        event_type: Any = "boundary"
    elif stimulus:
        event_type = stimulus
    elif not keystroke:
        return None
    elif int(event["keypad_accept"]):
        event_type = f"keypad{int(event['keypad_accept'])}"
    else:
        event_type = f"keyboard{int(event['keyboard'])}"
    result: dict[str, Any] = {
        "type": event_type,
        "latency": float(event["offset"]),
        "duration": 0.0,
    }
    for key in ("code", "accuracy", "accept", "epochevent"):
        if key in event:
            result[key] = deepcopy(event[key])
    return result


def _history_command(path: Path, options: dict[str, Any]) -> str:
    pieces = [format_history_value(path)]
    for key, value in options.items():
        pieces.extend([format_history_value(key), format_history_value(value, cell_for_sequence=None)])
    return f"EEG = pop_loadcnt({', '.join(pieces)});"


def _toggle(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "on":
        return True
    if text == "off":
        return False
    raise ValueError(f"{name} must be 'on' or 'off'")


__all__ = ["pop_loadcnt"]
