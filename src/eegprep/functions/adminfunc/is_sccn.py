"""Detect whether this machine's host name looks like an SCCN host."""

from __future__ import annotations

import os
import subprocess

_HOSTNAME_TIMEOUT_SECONDS = 5


def is_sccn() -> int:
    """Return 1 when this Unix host name contains ``ucsd.edu``, else 0.

    EEGLAB runs ``hostname`` on Unix and searches that output for
    ``ucsd.edu``. The command's trailing newline is whitespace, so the
    output is stripped rather than shortened by one character. Other
    platforms, a timed-out command, and any failure to read the host name
    return 0.
    """
    if os.name != "posix":
        return 0
    try:
        completed = subprocess.run(
            ["hostname"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_HOSTNAME_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if completed.returncode != 0:
        return 0
    return 1 if "ucsd.edu" in completed.stdout.strip() else 0
