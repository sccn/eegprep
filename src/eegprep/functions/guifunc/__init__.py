"""EEGLAB-like GUI helpers for EEGPrep pop functions."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "CallbackSpec": ("eegprep.functions.guifunc.spec", "CallbackSpec"),
    "ControlSpec": ("eegprep.functions.guifunc.spec", "ControlSpec"),
    "DialogSpec": ("eegprep.functions.guifunc.spec", "DialogSpec"),
    "EEGPrepMainWindow": ("eegprep.functions.guifunc.main_window", "EEGPrepMainWindow"),
    "build_main_window": ("eegprep.functions.guifunc.main_window", "build_main_window"),
    "controls_by_tag": ("eegprep.functions.guifunc.spec", "controls_by_tag"),
    "inputgui": ("eegprep.functions.guifunc.inputgui", "inputgui"),
    "listdlg2": ("eegprep.functions.guifunc.listdlg2", "listdlg2"),
    "pophelp": ("eegprep.functions.guifunc.pophelp", "pophelp"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
