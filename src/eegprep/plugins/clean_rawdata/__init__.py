"""clean_rawdata plugin ports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "asr_calibrate": ("eegprep.plugins.clean_rawdata.asr_calibrate", "asr_calibrate"),
    "asr_process": ("eegprep.plugins.clean_rawdata.asr_process", "asr_process"),
    "clean_artifacts": ("eegprep.plugins.clean_rawdata.clean_artifacts", "clean_artifacts"),
    "clean_asr": ("eegprep.plugins.clean_rawdata.clean_asr", "clean_asr"),
    "clean_channels": ("eegprep.plugins.clean_rawdata.clean_channels", "clean_channels"),
    "clean_channels_nolocs": ("eegprep.plugins.clean_rawdata.clean_channels_nolocs", "clean_channels_nolocs"),
    "clean_drifts": ("eegprep.plugins.clean_rawdata.clean_drifts", "clean_drifts"),
    "clean_flatlines": ("eegprep.plugins.clean_rawdata.clean_flatlines", "clean_flatlines"),
    "clean_windows": ("eegprep.plugins.clean_rawdata.clean_windows", "clean_windows"),
    "pop_clean_rawdata": ("eegprep.plugins.clean_rawdata.pop_clean_rawdata", "pop_clean_rawdata"),
    "vis_artifacts": ("eegprep.plugins.clean_rawdata.vis_artifacts", "vis_artifacts"),
    "vis_artifacts_diagnostics": (
        "eegprep.plugins.clean_rawdata.vis_artifacts",
        "vis_artifacts_diagnostics",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
