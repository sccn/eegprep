"""Report the installed EEGPrep version."""

from __future__ import annotations

import math
from dataclasses import dataclass

from eegprep import __version__


@dataclass(frozen=True)
class EEGVersion:
    """Installed EEGPrep version and its numeric form.

    ``matlab_outputs`` is the order used when a caller asks for both values.
    ``matlab_primary`` is the string returned for a single output.

    EEGLAB's ``eeg_getversion`` returns the text embedded in that file, which
    is ``'dev'`` in a source checkout, plus a release date. EEGPrep reports
    the installed package version and does not invent a release date. The
    number follows EEGLAB's collapse: drop one trailing letter, then drop
    every dot after the first, so ``11.3.2.4b`` becomes ``11.324``. A value
    that is not numeric is ``None`` where EEGLAB returns ``[]``.
    """

    version: str
    number: float | None
    matlab_primary: str
    matlab_outputs = ("version", "number")


def numeric_version(version: str) -> float | None:
    """Return ``version`` collapsed the way ``eeg_getversion.m`` does."""
    text = version
    if text and _not_numeric(text[-1]):
        text = text[:-1]
    dots = [index for index, character in enumerate(text) if character == "."]
    if len(dots) > 1:
        drop = set(dots[1:])
        text = "".join(character for index, character in enumerate(text) if index not in drop)
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def eeg_getversion() -> EEGVersion:
    """Return the installed EEGPrep version and its numeric form."""
    return EEGVersion(version=__version__, number=numeric_version(__version__), matlab_primary=__version__)


def _not_numeric(character: str) -> bool:
    try:
        float(character)
    except ValueError:
        return True
    return False


__all__ = ["EEGVersion", "eeg_getversion", "numeric_version"]
