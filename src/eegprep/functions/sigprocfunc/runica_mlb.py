"""Compatibility entry point for the historical ``runica_mlb`` fork."""

from __future__ import annotations

from typing import Any

from eegprep.functions.sigprocfunc._runica_variants import runica_variant


_MLB_DEFAULTS = {
    "extended": 1,
    "posact": "on",
    "bias": "on",
    "anneal": 0.98,
    "maxsteps": 500,
}


def runica_mlb(data: Any, **kwargs: Any) -> tuple:
    """Run the MLB infomax defaults through the maintained ``runica`` engine."""
    return runica_variant(data, _MLB_DEFAULTS, kwargs)


__all__ = ["runica_mlb"]
