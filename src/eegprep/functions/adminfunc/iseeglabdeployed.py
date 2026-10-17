"""Report whether EEGPrep is running from a frozen application bundle."""

from __future__ import annotations

import sys


def iseeglabdeployed() -> int:
    """Return 1 when EEGPrep is running as a frozen packaged application.

    EEGLAB returns MATLAB ``isdeployed``, which is true only inside an
    application built with MATLAB Compiler. EEGPrep uses ``sys.frozen``,
    which PyInstaller, cx_Freeze, and py2app set on a frozen executable.
    A normal interpreter, including ``eegprep-console`` from a virtual
    environment, returns 0.
    """
    return int(bool(getattr(sys, "frozen", False)))
