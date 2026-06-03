"""Plot/browser-style example action."""

from __future__ import annotations

from typing import Any, Callable


def pop_demo_browser(
    EEG: dict[str, Any] | None,
    *,
    command_callback: Callable[[dict[str, Any], str], None] | None = None,
    return_com: bool = False,
):
    """Return a lightweight browser model and report history through a callback."""
    if EEG is None:
        return (None, "") if return_com else None
    command = "LASTCOM = pop_demo_browser(EEG);"
    browser = {
        "kind": "example-browser",
        "setname": str(EEG.get("setname") or ""),
        "nbchan": int(EEG.get("nbchan", 0) or 0),
        "pnts": int(EEG.get("pnts", 0) or 0),
    }
    if command_callback is not None:
        command_callback(EEG, command)
    return (browser, command) if return_com else browser
