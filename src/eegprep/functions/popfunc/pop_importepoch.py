"""Import epoch metadata into an EEGPrep EEG dataset."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc._file_io import read_table_records
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args, parse_text_tokens


def pop_importepoch(
    EEG: dict[str, Any],
    filename: str | None = None,
    fieldlist: list[str] | tuple[str, ...] | str | None = None,
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import per-epoch metadata from a text table."""
    if filename is None:
        filename = kwargs.pop("filename", None)
    if filename is None:
        raise ValueError("pop_importepoch requires an epoch info file")
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    fields = _tokens(fieldlist)
    records = read_table_records(filename, fields=fields, skipline=int(options.get("headerlines", 0) or 0))
    trials = int(EEG.get("trials", 1) or 1)
    if trials <= 1:
        raise ValueError("pop_importepoch requires an epoched dataset")
    if len(records) != trials:
        raise ValueError("The number of imported epoch rows must match EEG.trials")
    out = deepcopy(EEG)
    out["epoch"] = records
    out["saved"] = "no"
    if str(options.get("clearevents", "on")).lower() in {"on", "yes", "true", "1"}:
        out["event"] = []
        out["urevent"] = []
    with strict_mode(False):
        out = eeg_checkset(out)
    command = _history_command(filename, fields, options)
    out["history"] = command if not out.get("history") else f"{out['history'].rstrip()}\n{command}"
    return (out, command) if return_com else out


def _tokens(value: Any) -> list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return [str(item) for item in parse_text_tokens(value)]
    return [str(item) for item in value]


def _history_command(filename: str, fields: list[str] | None, options: dict[str, Any]) -> str:
    pieces = [format_history_value(filename), format_history_value(fields or [])]
    for key in ["latencyfields", "durationfields", "typefield", "timeunit", "headerlines", "clearevents"]:
        if key in options:
            pieces.extend([format_history_value(key), format_history_value(options[key])])
    return f"EEG = pop_importepoch(EEG, {', '.join(pieces)});"
