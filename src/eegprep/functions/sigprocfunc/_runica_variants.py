"""Shared option handling for historical runica variants."""

from __future__ import annotations

from typing import Any

from eegprep.functions.sigprocfunc.runica import runica


def runica_variant(data: Any, defaults: dict[str, Any], kwargs: dict[str, Any]) -> tuple:
    """Run the canonical engine after applying a variant's distinct defaults."""
    options = {**defaults, **kwargs}
    return runica(data, **options)


__all__ = ["runica_variant"]
