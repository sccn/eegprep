"""EEGLAB-like GUI helpers for EEGPrep pop functions."""

from .inputgui import inputgui
from .spec import CallbackSpec, ControlSpec, DialogSpec, controls_by_tag

__all__ = ["CallbackSpec", "ControlSpec", "DialogSpec", "controls_by_tag", "inputgui"]
