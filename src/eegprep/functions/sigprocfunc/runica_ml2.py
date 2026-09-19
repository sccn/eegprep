"""Compatibility entry point for the historical ``runica_ml2`` fork."""

from __future__ import annotations

from typing import Any

from eegprep.functions.sigprocfunc._runica_variants import runica_variant


_ML2_DEFAULTS = {
    "extended": 1,
    "posact": "on",
    "bias": "off",
    "anneal": 0.95,
    "maxsteps": 500,
}


def runica_ml2(data: Any, **kwargs: Any) -> tuple:
    """Run the ML2 infomax defaults through the maintained ``runica`` engine.

    The historical fixed 10,000-sample training block is intentionally not
    retained; EEGPrep uses ``runica``'s data-dependent block heuristic unless
    the caller supplies ``block``.
    """
    return runica_variant(data, _ML2_DEFAULTS, kwargs)


__all__ = ["runica_ml2"]
