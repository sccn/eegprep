"""firfilt plugin ports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "design_fir": ("eegprep.plugins.firfilt.design", "design_fir"),
    "design_kaiser": ("eegprep.plugins.firfilt.design", "design_kaiser"),
    "findboundaries": ("eegprep.plugins.firfilt.findboundaries", "findboundaries"),
    "fir_filterdcpadded": ("eegprep.plugins.firfilt.fir_filterdcpadded", "fir_filterdcpadded"),
    "firfiltreport": ("eegprep.plugins.firfilt.firfiltreport", "firfiltreport"),
    "firfiltsplit": ("eegprep.plugins.firfilt.firfiltsplit", "firfiltsplit"),
    "firgauss": ("eegprep.plugins.firfilt.firgauss", "firgauss"),
    "firws": ("eegprep.plugins.firfilt.firws", "firws"),
    "firwsord": ("eegprep.plugins.firfilt.firwsord", "firwsord"),
    "invfirwsord": ("eegprep.plugins.firfilt.invfirwsord", "invfirwsord"),
    "invkaiserbeta": ("eegprep.plugins.firfilt.invkaiserbeta", "invkaiserbeta"),
    "kaiserbeta": ("eegprep.plugins.firfilt.kaiserbeta", "kaiserbeta"),
    "minphaserceps": ("eegprep.plugins.firfilt.minphaserceps", "minphaserceps"),
    "plotfresp": ("eegprep.plugins.firfilt.plotfresp", "plotfresp"),
    "pop_eegfiltnew": ("eegprep.plugins.firfilt.pop_eegfiltnew", "pop_eegfiltnew"),
    "pop_firma": ("eegprep.plugins.firfilt.pop_firma", "pop_firma"),
    "pop_firpm": ("eegprep.plugins.firfilt.pop_firpm", "pop_firpm"),
    "pop_firpmord": ("eegprep.plugins.firfilt.pop_firpmord", "pop_firpmord"),
    "pop_firws": ("eegprep.plugins.firfilt.pop_firws", "pop_firws"),
    "pop_firwsord": ("eegprep.plugins.firfilt.pop_firwsord", "pop_firwsord"),
    "pop_kaiserbeta": ("eegprep.plugins.firfilt.pop_kaiserbeta", "pop_kaiserbeta"),
    "pop_xfirws": ("eegprep.plugins.firfilt.pop_xfirws", "pop_xfirws"),
    "windows": ("eegprep.plugins.firfilt.windows", "windows"),
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
