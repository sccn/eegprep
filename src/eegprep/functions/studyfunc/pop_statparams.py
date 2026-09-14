"""Configure statistics used by STUDY measure plots."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.plot_utils import python_literal
from eegprep.functions.studyfunc._study_utils import ensure_study


COMMON_DEFAULTS = {
    "effect": "main",
    "groupstats": "off",
    "condstats": "off",
    "singletrials": "off",
    "mode": "eeglab",
}
EEGLAB_DEFAULTS = {"naccu": [], "alpha": np.nan, "method": "param", "mcorrect": "none"}
FIELDTRIP_DEFAULTS = {
    "naccu": [],
    "alpha": np.nan,
    "method": "analytic",
    "mcorrect": "none",
    "clusterparam": "'clusterstatistic','maxsum'",
    "channelneighbor": [],
    "channelneighborparam": "'method','triangulation'",
}
COMMON_OPTIONS = {"effect", "groupstats", "condstats", "singletrials", "mode"}
EEGLAB_OPTIONS = {"naccu", "alpha", "method", "mcorrect"}
FIELDTRIP_OPTIONS = {
    "fieldtripnaccu",
    "fieldtripalpha",
    "fieldtripmethod",
    "fieldtripmcorrect",
    "fieldtripclusterparam",
    "fieldtripchannelneighbor",
    "fieldtripchannelneighborparam",
}


def pop_statparams(STUDY: dict[str, Any], *args: Any, return_com: bool = False, **kwargs: Any) -> Any:
    """Set STUDY statistics options, including EEGLAB and FieldTrip sub-options."""
    is_study = isinstance(STUDY.get("etc"), dict)
    if is_study:
        study = ensure_study(STUDY)
        params = study["etc"].get("statistics")
    else:
        study = deepcopy(STUDY)
        params = study
    params = _with_defaults(params)
    options = _options(args, kwargs)
    unknown = sorted(set(options) - COMMON_OPTIONS - EEGLAB_OPTIONS - FIELDTRIP_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown pop_statparams option(s): {', '.join(unknown)}")
    for key, value in options.items():
        key = {"statistics": "method", "threshold": "alpha"}.get(key, key)
        if key in COMMON_OPTIONS:
            params[key] = deepcopy(value)
        elif key in EEGLAB_OPTIONS:
            params["eeglab"][key] = deepcopy(value)
        else:
            nested_key = key.removeprefix("fieldtrip")
            params["fieldtrip"][nested_key] = deepcopy(value)
            if nested_key == "channelneighborparam":
                params["fieldtrip"]["channelneighbor"] = []
    if is_study:
        study["etc"]["statistics"] = params
        if options:
            study["saved"] = "no"
    else:
        study = params
    command = _history_command(options)
    return (study, command) if return_com else study


def _with_defaults(params: Any) -> dict[str, Any]:
    output = deepcopy(params) if isinstance(params, dict) else {}
    for key, value in COMMON_DEFAULTS.items():
        output.setdefault(key, deepcopy(value))
    for section, defaults in (("eeglab", EEGLAB_DEFAULTS), ("fieldtrip", FIELDTRIP_DEFAULTS)):
        nested = output.get(section)
        if not isinstance(nested, dict):
            nested = {}
        for key, value in defaults.items():
            nested.setdefault(key, deepcopy(value))
        output[section] = nested
    return output


def _options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) == 1 and str(args[0]).lower() == "default":
        if kwargs:
            raise ValueError("'default' cannot be combined with parameter options")
        return {}
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if "statistics" in options:
        options["method"] = options.pop("statistics")
    if "threshold" in options:
        options["alpha"] = options.pop("threshold")
    return options


def _history_command(options: dict[str, Any]) -> str:
    pieces = ["STUDY"]
    pieces.extend(f"{key}={python_literal(value)}" for key, value in options.items())
    return f"STUDY = pop_statparams({', '.join(pieces)})"


__all__ = ["pop_statparams"]
