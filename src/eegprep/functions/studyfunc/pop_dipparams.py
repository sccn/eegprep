"""Configure STUDY dipole plotting parameters."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_params import update_study_params


DIP_DEFAULTS = {
    "axistight": "off",
    "projimg": "off",
    "projlines": "off",
    "density": "off",
    "centrline": "on",
}


def pop_dipparams(STUDY: dict[str, Any], *args: Any, return_com: bool = False, **kwargs: Any) -> Any:
    """Set defaults or named options in ``STUDY.etc.dipparams``."""
    return update_study_params(
        STUDY,
        "dipparams",
        DIP_DEFAULTS,
        "pop_dipparams",
        args,
        kwargs,
        return_com=return_com,
    )


__all__ = ["pop_dipparams"]
