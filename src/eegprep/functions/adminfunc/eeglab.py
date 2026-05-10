"""EEGPrep EEGLAB-style GUI entry point."""

from __future__ import annotations

import argparse
from typing import Any

from eegprep.functions.guifunc.main_window import build_main_window
from eegprep.functions.guifunc.session import EEGPrepSession


def eeglab(
    onearg: str | None = None,
    *,
    session: EEGPrepSession | None = None,
    show: bool = True,
    block: bool = False,
    all_menus: bool | None = None,
    include_plugins: bool = True,
) -> Any:
    """Start the EEGPrep EEGLAB-style main window.

    Args:
        onearg: EEGLAB-like command. ``"nogui"`` returns a session only,
            ``"full"`` enables legacy/advanced menu items, and ``"versions"``
            returns the package version string.
        session: Optional existing GUI session.
        show: Show the Qt window before returning.
        block: Enter the Qt event loop.
        all_menus: Override ``EEG_OPTIONS["option_allmenus"]``.
        include_plugins: Include plugin-contributed menus.

    Returns:
        ``EEGPrepMainWindow`` by default, or ``EEGPrepSession`` for
        ``onearg="nogui"``.
    """
    if onearg == "versions":
        from eegprep import __version__

        return __version__
    gui_session = session or EEGPrepSession()
    if onearg == "nogui":
        return gui_session
    if onearg == "full":
        all_menus = True
    window = build_main_window(
        gui_session,
        all_menus=all_menus,
        include_plugins=include_plugins,
    )
    if block:
        return window.exec()
    if show:
        window.show()
    return window


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``eegprep-gui``."""
    parser = argparse.ArgumentParser(description="Launch the EEGPrep EEGLAB-style GUI.")
    parser.add_argument("--nogui", action="store_true", help="Initialize session state without opening a window")
    parser.add_argument("--full", action="store_true", help="Show EEGLAB legacy/advanced menu items")
    parser.add_argument("--no-plugins", action="store_true", help="Hide plugin-contributed menu items")
    args = parser.parse_args(argv)
    if args.nogui:
        eeglab("nogui", show=False)
        return 0
    eeglab(
        "full" if args.full else None,
        block=True,
        include_plugins=not args.no_plugins,
    )
    return 0
