"""Configure STUDY ERSP and ITC plotting parameters."""

from __future__ import annotations

from typing import Any

from eegprep.functions.studyfunc._study_params import update_study_params


ERSP_DEFAULTS = {
    "topotime": [],
    "topofreq": [],
    "timerange": [],
    "freqrange": [],
    "ersplim": [],
    "itclim": [],
    "maskdata": "off",
    "averagemode": "rms",
    "averagechan": "off",
    "subbaseline": "off",
}
ERSP_FIELDS = (
    "erspdata",
    "ersptimes",
    "erspfreqs",
    "erspbase",
    "erspdatatrials",
    "erspsubjinds",
    "ersptrialinfo",
    "itcdata",
    "itctimes",
    "itcfreqs",
    "itcdatatrials",
    "itcsubjinds",
    "itctrialinfo",
)


def pop_erspparams(STUDY: dict[str, Any], *args: Any, return_com: bool = False, **kwargs: Any) -> Any:
    """Set defaults or named options in ``STUDY.etc.erspparams``."""
    return update_study_params(
        STUDY,
        "erspparams",
        ERSP_DEFAULTS,
        "pop_erspparams",
        args,
        kwargs,
        invalidate_on=("timerange", "freqrange", "subbaseline"),
        invalidated_fields=ERSP_FIELDS,
        return_com=return_com,
    )


__all__ = ["pop_erspparams"]
