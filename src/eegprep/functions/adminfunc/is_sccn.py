"""Detect whether this machine's host name looks like an SCCN host."""

from __future__ import annotations

import os
import subprocess


def is_sccn() -> int:
    """Return 1 when this Unix host name contains ``ucsd.edu``, else 0.

    EEGLAB runs ``hostname`` on Unix and searches that output, excluding its
    final character, for ``ucsd.edu``. The final character is the newline
    from the command. Other platforms, and any failure to read the host
    name, return 0.
    """
    if os.name != "posix":
        return 0
    try:
        completed = subprocess.run(["hostname"], check=False, capture_output=True, text=True)
    except OSError:
        return 0
    hostname = completed.stdout
    return 1 if "ucsd.edu" in hostname[:-1] else 0
