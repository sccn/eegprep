"""Save EEGPrep STUDY structures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._file_io import write_json
from eegprep.functions.popfunc._pop_utils import format_history_value


def pop_savestudy(
    STUDY: dict[str, Any],
    EEG: dict[str, Any] | list[dict[str, Any]] | None = None,
    filename: str | Path | None = None,
    *,
    savemode: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Save a STUDY structure as JSON in a ``.study`` file."""
    if filename is None:
        filename = STUDY.get("filename")
    if filename is None:
        raise ValueError("pop_savestudy requires a filename")
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    study = dict(STUDY)
    study["filename"] = path.name
    study["filepath"] = str(path.parent)
    write_json(path, study)
    pieces = [
        format_history_value("filename"),
        format_history_value(path.name),
        format_history_value("filepath"),
        format_history_value(path.parent),
    ]
    if savemode:
        pieces.extend([format_history_value("savemode"), format_history_value("resave")])
    return study, f"STUDY = pop_savestudy(STUDY, EEG, {', '.join(pieces)});"
