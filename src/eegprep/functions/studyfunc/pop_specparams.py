"""Configure STUDY spectrum plotting parameters."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_params import update_study_params


SPEC_DEFAULTS = {
    "topofreq": [],
    "freqrange": [],
    "ylim": [],
    "subtractsubjectmean": "off",
    "plotgroups": "apart",
    "plotconditions": "apart",
    "averagechan": "off",
    "detachplots": "on",
}


def pop_specparams(STUDY: dict[str, Any], *args: Any, return_com: bool = False, **kwargs: Any) -> Any:
    """Set defaults or named options in ``STUDY.etc.specparams``."""
    return update_study_params(
        STUDY,
        "specparams",
        SPEC_DEFAULTS,
        "pop_specparams",
        args,
        kwargs,
        invalidate_on=("freqrange", "subtractsubjectmean"),
        invalidated_fields=("specdata", "specfreqs"),
        return_com=return_com,
    )


__all__ = ["pop_specparams"]
