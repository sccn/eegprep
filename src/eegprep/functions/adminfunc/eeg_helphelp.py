"""Print how EEGPrep help is organized."""

from __future__ import annotations

import inspect


def eeg_helphelp() -> None:
    """How EEGPrep help is organized.

    Menu items call the same functions available from ``eegprep-console``.
    When a function opens a dialog, the window names the function it will
    run. Help for that menu item is the help for the called function.

    Passing only an EEG or STUDY object to a ``pop_`` function opens its
    dialog. Passing the arguments directly runs the same workflow from
    Python. Functions that support ``return_com=True`` return a history
    command. ``eegh`` and the shared session keep that history so it can
    be copied into a script.

    Help for a function is its docstring. ``pop_`` functions usually
    collect arguments and call another function that processes the data;
    read that function's docstring to see the computation. The Help menu
    and dialog Help buttons open the packaged Markdown help for the same
    function.
    """
    print(inspect.getdoc(eeg_helphelp) or "")
