"""Import ERPLAB event-list text files into an EEGPrep EEG dataset."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_importevent import pop_importevent


def pop_importerplab(
    EEG: dict[str, Any],
    filename: str | None = None,
    *,
    return_com: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import an ERPLAB-style event text file through the generic event importer."""
    if filename is None:
        filename = kwargs.pop("filename", None)
    if filename is None:
        raise ValueError("pop_importerplab requires an ERPLAB event filename")
    out = pop_importevent(
        EEG,
        "event",
        filename,
        "fields",
        kwargs.pop("fields", ["latency", "type"]),
        "timeunit",
        kwargs.pop("timeunit", float("nan")),
        return_com=True,
        **kwargs,
    )
    eeg, _command = out
    command = f"EEG = pop_importerplab(EEG, {format_history_value(filename)});"
    eeg["history"] = command if not EEG.get("history") else f"{EEG['history'].rstrip()}\n{command}"
    return (eeg, command) if return_com else eeg
