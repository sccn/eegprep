"""Read EEGPrep-owned LIMO-compatible model and result files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.functions.studyfunc._limo_io import load_limo_result


def std_readfilelimo(source: Any) -> Any:
    """Load one or more EEGPrep LIMO ``.npz`` outputs.

    Dictionaries already in memory are returned unchanged. MATLAB ``.mat``
    LIMO structures remain an explicit external-toolbox boundary because their
    layout varies across LIMO releases.
    """
    if isinstance(source, dict):
        return source
    if isinstance(source, (list, tuple)):
        return [std_readfilelimo(item) for item in source]
    if isinstance(source, (str, Path)):
        return load_limo_result(source)
    raise TypeError("source must be a result dictionary, path, or sequence of either")


__all__ = ["std_readfilelimo"]
