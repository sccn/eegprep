"""EEGLAB-like GUI helpers for EEGPrep pop functions."""

from .inputgui import inputgui
from .listdlg2 import listdlg2
from .main_window import EEGPrepMainWindow, build_main_window
from .pophelp import pophelp
from .spec import CallbackSpec, ControlSpec, DialogSpec, controls_by_tag

__all__ = [
    "CallbackSpec",
    "ControlSpec",
    "DialogSpec",
    "EEGPrepMainWindow",
    "build_main_window",
    "controls_by_tag",
    "inputgui",
    "listdlg2",
    "pophelp",
]
