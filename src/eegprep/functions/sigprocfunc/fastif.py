"""Expression-level conditional compatibility helper."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def fastif(condition: object, if_true: T, if_false: T) -> T:
    """Return ``if_true`` when ``condition`` is truthy, otherwise ``if_false``."""
    return if_true if bool(condition) else if_false


__all__ = ["fastif"]
