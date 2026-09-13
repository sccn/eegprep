"""Configure STUDY ERP plotting parameters."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_params import update_study_params


ERP_DEFAULTS = {
    "topotime": [],
    "filter": [],
    "timerange": [],
    "ylim": [],
    "plotgroups": "apart",
    "plotconditions": "apart",
    "averagechan": "off",
    "detachplots": "on",
}


def pop_erpparams(STUDY: dict[str, Any], *args: Any, return_com: bool = False, **kwargs: Any) -> Any:
    """Set defaults or named options in ``STUDY.etc.erpparams``."""
    return update_study_params(
        STUDY,
        "erpparams",
        ERP_DEFAULTS,
        "pop_erpparams",
        args,
        kwargs,
        invalidate_on=("timerange",),
        invalidated_fields=("erpdata", "erptimes"),
        return_com=return_com,
    )


__all__ = ["pop_erpparams"]
