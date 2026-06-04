"""Minimal EEGPrep extension template package."""

from ._version import __version__
from .pop_template_gain import pop_template_gain, pop_template_gain_dialog_spec
from .registration import register
from .sample_data import load_sample_eeg

__all__ = [
    "__version__",
    "load_sample_eeg",
    "pop_template_gain",
    "pop_template_gain_dialog_spec",
    "register",
]
