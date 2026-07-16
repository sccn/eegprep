"""Explicit limitation for STUDY-level dipole plotting."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._source_limitations import raise_source_study_limitation


def std_dipplot(*args: Any, **kwargs: Any) -> None:
    """Report the standalone boundary for STUDY-level source plotting."""
    raise_source_study_limitation("std_dipplot", *args, **kwargs)


__all__ = ["std_dipplot"]
