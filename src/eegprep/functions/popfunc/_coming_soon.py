"""Placeholder helpers for EEGLAB menu actions not yet ported.

TODO: replace each coming-soon action with the matching EEGLAB-compatible
``pop_*`` or ``eeg_*`` implementation as those ports land in EEGPrep.
"""

from __future__ import annotations


def coming_soon(function_name: str) -> None:
    """Raise the standard placeholder error for unimplemented menu actions."""
    raise NotImplementedError(f"{function_name} is coming soon in EEGPrep")
