"""Import Presentation LOG events into an EEGPrep EEG dataset."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_importevent import pop_importevent


def pop_importpres(
    EEG: dict[str, Any],
    filename: str | None = None,
    *,
    return_com: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import a Presentation LOG file using EEGPrep's generic event importer."""
    if filename is None:
        filename = kwargs.pop("filename", None)
    if filename is None:
        raise ValueError("pop_importpres requires a Presentation LOG filename")
    out = pop_importevent(
        EEG,
        "event",
        filename,
        "fields",
        kwargs.pop("fields", ["type", "latency"]),
        "timeunit",
        kwargs.pop("timeunit", float("nan")),
        return_com=True,
        **kwargs,
    )
    eeg, _command = out
    command = f"EEG = pop_importpres(EEG, {format_history_value(filename)});"
    eeg["history"] = command if not EEG.get("history") else f"{EEG['history'].rstrip()}\n{command}"
    return (eeg, command) if return_com else eeg
