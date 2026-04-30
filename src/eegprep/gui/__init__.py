"""GUI foundation for EEGLAB-compatible EEGPREP dialogs."""

from .inputgui import inputgui
from .spec import CallbackSpec, ControlSpec, DialogSpec, controls_by_tag

__all__ = [
    "CallbackSpec",
    "ControlSpec",
    "DialogSpec",
    "controls_by_tag",
    "inputgui",
]
