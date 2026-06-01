"""Explicit placeholders for EEGLAB main-window actions not yet ported.

Each placeholder carries either an implementation phase or an exclusion reason
so later phase agents can distinguish planned work from intentionally excluded
external-plugin workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PlaceholderMetadata:
    """Phase ownership or exclusion metadata for a placeholder action."""

    phase: str | None = None
    excluded_reason: str | None = None


PLACEHOLDER_ACTION_METADATA: Mapping[str, PlaceholderMetadata] = {}
PLACEHOLDER_ACTIONS = frozenset(PLACEHOLDER_ACTION_METADATA)


def placeholder_metadata(action: str) -> PlaceholderMetadata | None:
    """Return placeholder metadata for ``action`` when registered."""
    return PLACEHOLDER_ACTION_METADATA.get(action) or PLACEHOLDER_ACTION_METADATA.get(action.partition(":")[0])


def placeholder_inventory() -> Mapping[str, PlaceholderMetadata]:
    """Return the machine-readable placeholder inventory."""
    return PLACEHOLDER_ACTION_METADATA


def is_placeholder_action(action: str) -> bool:
    """Return whether a menu action has an explicit coming-soon placeholder."""
    return placeholder_metadata(action) is not None


def placeholder_message(action: str) -> str:
    """Build the shared EEGLAB-style placeholder message for ``action``."""
    return (
        f"{action} is not yet available in EEGPrep.\n\n"
        "Track progress or request this workflow at https://github.com/sccn/eegprep/issues."
    )
