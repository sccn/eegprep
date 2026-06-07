"""Explicit limitation for STUDY dipole-cluster preparation."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._source_limitations import raise_source_study_limitation


def std_dipoleclusters(*args: Any, **kwargs: Any) -> None:
    """Report the standalone boundary for STUDY dipole-cluster workflows."""
    raise_source_study_limitation("std_dipoleclusters", *args, **kwargs)


__all__ = ["std_dipoleclusters"]
