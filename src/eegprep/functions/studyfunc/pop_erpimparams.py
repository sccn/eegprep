"""Configure STUDY ERP-image plotting parameters."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_params import update_study_params


ERPIMAGE_DEFAULTS = {
    "erpimageopt": [],
    "sorttype": "",
    "sortwin": [],
    "sortfield": "latency",
    "rmcomps": [],
    "interp": [],
    "timerange": [],
    "topotime": [],
    "colorlimits": [],
    "concatenate": "off",
    "nlines": 20,
    "smoothing": 10,
    "averagemode": "ave",
    "averagechan": "off",
}
ERPIMAGE_FIELDS = ("erpimdata", "erpimtimes", "erpimtrials", "erpimevents")


def pop_erpimparams(STUDY: dict[str, Any], *args: Any, return_com: bool = False, **kwargs: Any) -> Any:
    """Set defaults or named options in ``STUDY.etc.erpimparams``."""
    return update_study_params(
        STUDY,
        "erpimparams",
        ERPIMAGE_DEFAULTS,
        "pop_erpimparams",
        args,
        kwargs,
        invalidate_on=("timerange",),
        invalidated_fields=ERPIMAGE_FIELDS,
        return_com=return_com,
    )


__all__ = ["pop_erpimparams"]
