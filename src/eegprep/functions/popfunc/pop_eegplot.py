"""EEGLAB-style wrapper for the scrolling EEG browser."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc._plot_utils import history_command
from eegprep.functions.sigprocfunc.eegplot import eegplot


def pop_eegplot(
    EEG: dict[str, Any] | None = None,
    icacomp: int = 1,
    superpose: int = 0,
    reject: int = 1,
    topcommand: Any = None,
    *args: Any,
    return_com: bool = False,
    show: bool = True,
    **kwargs: Any,
):
    """Inspect EEG channel or component activity in the scrolling browser.

    Phase 1 opens the browser and records the EEGLAB-style command. Mark update
    and rejection side effects are intentionally left to later EEGBrowser
    phases, so the input EEG dictionary is returned unchanged when
    ``return_com=True``.
    """
    del topcommand
    if EEG is None:
        return (None, "") if return_com else None
    if icacomp not in {0, 1}:
        raise ValueError("icacomp must be 1 for data channels or 0 for components")
    options = dict(kwargs)
    options.setdefault("srate", EEG.get("srate", 256))
    options.setdefault(
        "limits", [float(EEG.get("xmin", 0.0) or 0.0) * 1000.0, float(EEG.get("xmax", 0.0) or 0.0) * 1000.0]
    )
    options.setdefault("events", EEG.get("event", []))
    if icacomp == 1:
        options.setdefault("eloc_file", EEG.get("chanlocs", []))
        options.setdefault("title", f"Scroll channel activities -- eegplot() -- {EEG.get('setname', '')}".rstrip())
        window = eegplot(EEG, *args, show=show, **options)
    else:
        options.setdefault("component", True)
        options.setdefault("title", f"Scroll component activities -- eegplot() -- {EEG.get('setname', '')}".rstrip())
        window = eegplot(EEG, *args, show=show, **options)
    command = history_command("pop_eegplot", icacomp, superpose, reject)
    return (EEG, command) if return_com else window


__all__ = ["pop_eegplot"]
