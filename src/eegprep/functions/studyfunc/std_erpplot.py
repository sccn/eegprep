"""Plot cached STUDY ERP measures."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._std_measureplot import std_measureplot


def std_erpplot(STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None, *args: Any, **kwargs: Any):
    """Read and plot precomputed STUDY ERP measures."""
    return std_measureplot(STUDY, ALLEEG, "erp", *args, **kwargs)


__all__ = ["std_erpplot"]
