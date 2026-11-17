"""Legacy channel-name lookup compatibility."""

from __future__ import annotations

from typing import Any

from eegprep.functions.popfunc.eeg_decodechan import eeg_decodechan


def eeg_chaninds(EEG: Any, channel_names: Any, errorifnotfound: bool = True) -> list[int]:
    """Return 0-based indices for channel labels.

    This is EEGPrep's compatibility alias for :func:`eeg_decodechan`. These
    label lookups stay 0-based. ``getchanlist`` and ``eeg_amplitudearea`` use
    1-based channel numbers.
    """
    indices, _ = eeg_decodechan(EEG, channel_names, ignoremissing=not errorifnotfound)
    return indices


__all__ = ["eeg_chaninds"]
