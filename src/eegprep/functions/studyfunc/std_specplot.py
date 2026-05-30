"""Plot cached STUDY spectrum measures."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._std_measureplot import std_measureplot


def std_specplot(STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None, *args: Any, **kwargs: Any):
    """Read and plot precomputed STUDY spectrum measures."""
    return std_measureplot(STUDY, ALLEEG, "spec", *args, **kwargs)


__all__ = ["std_specplot"]
