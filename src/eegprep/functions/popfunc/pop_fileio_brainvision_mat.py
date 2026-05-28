"""Import BrainVision Analyzer MATLAB files through the EEGPrep File-IO path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_fileio import pop_fileio


_BRAINVISION_ANALYZER_SUFFIXES = {".mat"}


def pop_fileio_brainvision_mat(
    filename: str | Path, *, return_com: bool = False, **kwargs: Any
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import a BrainVision Analyzer MATLAB file.

    EEGLAB exposes this menu through the optional ``bva-io`` plugin. EEGPrep
    does not depend on that plugin at runtime, so this wrapper delegates MATLAB
    files to the existing standalone File-IO/MAT importer.
    """
    path = Path(filename)
    if path.suffix.lower() not in _BRAINVISION_ANALYZER_SUFFIXES:
        raise ValueError("pop_fileio_brainvision_mat supports BrainVision Analyzer .mat files")
    eeg, _command = cast(tuple[dict[str, Any], str], pop_fileio(path, return_com=True, **kwargs))
    command = f"EEG = pop_fileio_brainvision_mat({format_history_value(path)});"
    history = str(eeg.get("history") or "").strip()
    eeg["history"] = f"{history}\n{command}".lstrip()
    return (eeg, command) if return_com else eeg


__all__ = ["pop_fileio_brainvision_mat"]
