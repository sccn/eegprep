"""Import channel data into an EEGPrep EEG structure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._file_io import eeg_from_data, infer_dataformat, load_data_array
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args


def pop_importdata(*args: Any, return_com: bool = False, **kwargs: Any) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import data from an array or supported file into an EEG dataset."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if "data" not in options:
        raise ValueError("pop_importdata requires a 'data' file or array")
    data_source = options["data"]
    nbchan = _optional_int(options.get("nbchan"))
    dataformat = infer_dataformat(data_source if isinstance(data_source, (str, Path)) else None, options.get("dataformat"))
    data = load_data_array(
        data_source,
        dataformat=dataformat,
        nbchan=nbchan,
        variable=options.get("variable"),
    )
    filename = str(Path(data_source).name) if isinstance(data_source, (str, Path)) else ""
    filepath = str(Path(data_source).parent) if isinstance(data_source, (str, Path)) else ""
    eeg = eeg_from_data(
        data,
        srate=float(options.get("srate", 1.0) or 1.0),
        setname=str(options.get("setname") or Path(filename).stem or "Imported data"),
        nbchan=nbchan,
        pnts=_optional_int(options.get("pnts")),
        xmin=float(options.get("xmin", 0.0) or 0.0),
        subject=str(options.get("subject") or ""),
        condition=str(options.get("condition") or ""),
        group=str(options.get("group") or ""),
        session=options.get("session"),
        comments=str(options.get("comments") or ""),
        ref=options.get("ref", "common"),
        filename=filename,
        filepath=filepath,
    )
    command = _history_command(options, dataformat)
    eeg["history"] = command
    return (eeg, command) if return_com else eeg


def _history_command(options: dict[str, Any], dataformat: str) -> str:
    values = dict(options)
    values["dataformat"] = dataformat
    pieces = []
    for key in [
        "data",
        "dataformat",
        "setname",
        "srate",
        "pnts",
        "xmin",
        "nbchan",
        "subject",
        "condition",
        "session",
        "group",
        "ref",
        "comments",
    ]:
        if key in values and _history_value_present(values[key]):
            pieces.extend([format_history_value(key), format_history_value(values[key])])
    return f"EEG = pop_importdata({', '.join(pieces)});"


def _history_value_present(value: Any) -> bool:
    if value is None:
        return False
    return not (isinstance(value, str) and value == "")


def _optional_int(value: Any) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    return int(value)
